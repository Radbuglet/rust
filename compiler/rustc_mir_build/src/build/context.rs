#![expect(unused)]

use rustc_data_structures::fx::{FxHashMap, FxIndexMap, IndexEntry};
use rustc_hir::def_id::DefId;
use rustc_hir::Mutability;
use rustc_index::{Idx as _, IndexVec};
use rustc_middle::mir::{self, BasicBlock, Local, Operand, Place, Rvalue, SourceInfo};
use rustc_middle::thir::{self, ContextBinder, visit::{self as thir_visit, Visitor as _}};
use rustc_middle::ty::{self, Ty, TypeFoldable as _};
use rustc_span::DUMMY_SP;
use rustc_target::abi::FieldIdx;

use thir_visit::Visitor as _;

use super::Builder;

use crate::context::ContextBindTracker;

// Inner index map allows for deterministic iteration.
type BinderMap = FxHashMap<ContextBinder, FxIndexMap<DefId, ContextBinderItemInfo>>;

#[derive(Default)]
pub(crate) struct ContextBinderMap(Option<BinderMap>);

impl ContextBinderMap {
    fn expect_init(&self) -> &BinderMap {
        self.0.as_ref().expect("`define_context_locals` never called")
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct ContextBinderItemInfo {
    first_local: Local,
    muta: Mutability,
}

impl ContextBinderItemInfo {
    /// The local storing the pointer to the context item, of type `*mut T`
    pub(crate) fn ptr_local(self) -> Local {
        self.first_local
    }

    /// The local storing a borrowed reference to the context item, of type `&<muta> T`
    pub(crate) fn ref_local(self) -> Local {
        self.first_local.plus(1)
    }

    pub(crate) fn muta(self) -> Mutability {
        self.muta
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    pub(crate) fn define_context_locals(
        &mut self,
        source_info: SourceInfo,
        root_expr: thir::ExprId,
    ) {
        assert!(self.ctx_bind_values.0.is_none(), "`define_context_locals` called twice");

        let root_expr = &self.thir[root_expr];
        let mut visitor = BinderUseVisitor {
            builder: self,
            map: BinderMap::default(),
            bind_tracker: ContextBindTracker::default(),
            source_info,
        };
        visitor.visit_expr(root_expr);
        self.ctx_bind_values.0 = Some(visitor.map);
    }

    pub(crate) fn init_and_borrow_context_binder_locals(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        binder: ContextBinder,
        lt_limiter: Place<'tcx>,
    ) {
        let Some(binders) = self.ctx_bind_values.expect_init().get(&binder) else {
            return;
        };

        // Acquire a raw pointer for each context item.
        for (&item, &item_info) in binders {
            self.cfg.push_assign(
                block,
                source_info,
                item_info.ptr_local().into(),
                Rvalue::ContextRef(item),
            );
        }

        // Borrow each context item.
        let deref_proj = self.tcx.mk_place_elems(&[mir::PlaceElem::Deref]);

        for (&item, &item_info) in binders {
            let kind = match item_info.muta() {
                Mutability::Not => mir::BorrowKind::Shared,
                Mutability::Mut => mir::BorrowKind::Mut {
                    kind: mir::MutBorrowKind::Default,
                },
            };

            self.cfg.push_assign(
                block,
                source_info,
                item_info.ref_local().into(),
                Rvalue::Ref(self.tcx.lifetimes.re_erased, kind, Place {
                    local: item_info.ptr_local(),
                    projection: deref_proj,
                }),
            );
        }

        // Limit their lifetimes to the scope of the bind.
        let equate_refs = [lt_limiter].into_iter()
            .chain(binders.values().map(|item_info| Place::from(item_info.ref_local())))
            .collect::<Vec<_>>();

        self.equate_ref_lifetimes(block, source_info, &equate_refs);
    }

    pub(crate) fn lookup_context_binder(
        &self,
        item: DefId,
        binder: ContextBinder,
    ) -> ContextBinderItemInfo {
        *self.ctx_bind_values.expect_init()
            .get(&binder)
            .and_then(|v| v.get(&item))
            .unwrap_or_else(|| panic!("unknown context item {item:?} under binder {binder:?}"))
    }

    pub(crate) fn equate_ref_lifetimes(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        refs: &[Place<'tcx>],
    ) {
        // Create `ascribe_temp` temporary and give it the type...
        // `(&'erased {refs[0].muta} {refs[0].pointee}, ...)`
        let ascribe_temp_ty = refs.iter().map(|&ref_| ref_.ty(&self.local_decls, self.tcx).ty);
        let ascribe_temp_ty = Ty::new_tup_from_iter(self.tcx, ascribe_temp_ty);
        let refs_tys = match ascribe_temp_ty.kind() {
            ty::Tuple(tys) => tys,
            _ => unreachable!(),
        };
        let ascribe_temp = self.local_decls.push(mir::LocalDecl::new(
            ascribe_temp_ty,
            source_info.span,
        ));

        // Pack the various reborrows into `ascribe_temp`
        let mut assign_fields = IndexVec::new();

        for &ref_ in refs {
            assign_fields.push(mir::Operand::Move(ref_));
        }

        self.cfg.push_assign(
            block,
            source_info,
            ascribe_temp.into(),
            Rvalue::Aggregate(
                Box::new(mir::AggregateKind::Tuple),
                assign_fields,
            ),
        );

        // Ascribe the appropriate type of `ascribe_temp`
        let mut var_count = 1;
        let ascribe_temp_var_ty = refs.iter()
            .zip(refs_tys.iter())
            .map(|(&ref_, ref_ty)| {
                let (pointee, muta) = match ref_ty.kind() {
                    &ty::Ref(_re, pointee, muta) => (pointee, muta),
                    _ => unreachable!("expected reference, got {ref_ty}"),
                };

                let re = ty::Region::new_bound(
                    self.tcx,
                    ty::DebruijnIndex::ZERO,
                    ty::BoundRegion {
                        var: ty::BoundVar::ZERO,
                        kind: ty::BoundRegionKind::BrAnon,
                    }
                );

                // N.B. This old folds *free* regions.
                let pointee = pointee.fold_with(&mut ty::fold::RegionFolder::new(
                    self.tcx,
                    &mut |_, debrujin| {
                        let re = ty::Region::new_bound(
                            self.tcx,
                            debrujin,
                            ty::BoundRegion {
                                var: ty::BoundVar::from_u32(var_count),
                                kind: ty::BoundRegionKind::BrAnon,
                            },
                        );
                        var_count += 1;
                        re
                    },
                ));

                Ty::new_ref(self.tcx, re, pointee, muta)
            });
        let ascribe_temp_var_ty = Ty::new_tup_from_iter(self.tcx, ascribe_temp_var_ty);

        let ascribe_variables = (0..var_count).map(|_| ty::CanonicalVarInfo {
            kind: ty::CanonicalVarKind::Region(ty::UniverseIndex::ROOT)
        });
        let ascribe_variables = self.tcx.mk_canonical_var_infos_from_iter(ascribe_variables);

        let ascribe_idx =
            self.canonical_user_type_annotations.push(ty::CanonicalUserTypeAnnotation {
                span: source_info.span,
                user_ty: Box::new(ty::CanonicalUserType {
                    value: ty::UserType::Ty(ascribe_temp_var_ty),
                    max_universe: ty::UniverseIndex::ROOT,
                    defining_opaque_types: ty::List::empty(),
                    variables: ascribe_variables,
                }),
                inferred_ty: ascribe_temp_ty,
            });

        self.cfg.push(block, mir::Statement {
            source_info,
            kind: mir::StatementKind::AscribeUserType(
                Box::new((ascribe_temp.into(), mir::UserTypeProjection {
                    base: ascribe_idx,
                    projs: Vec::new(),
                })),
                ty::Variance::Invariant,
            ),
        });

        // Move the borrows back into the locals.
        for (i, &ref_) in refs.iter().enumerate() {
            self.cfg.push_assign(
                block,
                source_info,
                ref_,
                Rvalue::Use(Operand::Move(Place {
                    local: ascribe_temp,
                    projection: self.tcx.mk_place_elems(&[mir::PlaceElem::Field(
                        FieldIdx::from_usize(i),
                        refs_tys[i],
                    )]),
                })),
            );
        }
    }

    pub(crate) fn new_lt_limiter(&mut self, block: BasicBlock, source_info: SourceInfo) -> Place<'tcx> {
        let lt_limiter = self.temp(self.tcx.types.unit, source_info.span);
        self.cfg.push_assign_unit(block, source_info, lt_limiter, self.tcx);

        let lt_limiter_ref = self.temp(
            Ty::new_mut_ref(self.tcx, self.tcx.lifetimes.re_erased, self.tcx.types.unit),
            source_info.span,
        );
        self.cfg.push_assign(
            block,
            source_info,
            lt_limiter_ref,
            Rvalue::Ref(
                self.tcx.lifetimes.re_erased,
                mir::BorrowKind::Mut {
                    kind: mir::MutBorrowKind::Default,
                },
                lt_limiter,
            ),
        );

        lt_limiter_ref
    }
}

struct BinderUseVisitor<'a, 'thir, 'tcx> {
    builder: &'a mut Builder<'thir, 'tcx>,
    bind_tracker: ContextBindTracker,
    map: BinderMap,
    source_info: SourceInfo,
}

impl<'a, 'thir, 'tcx> BinderUseVisitor<'a, 'thir, 'tcx> {
    fn introduce_use(&mut self, item: DefId, muta: Mutability, binder: ContextBinder) {
        let tcx = self.builder.tcx;

        let item_entry = self.map.entry(binder)
            .or_default()
            .entry(item);

        match item_entry {
            IndexEntry::Vacant(entry) => {
                // ptr_local
                let first_local = self.builder.local_decls
                    .push(mir::LocalDecl::with_source_info(
                        tcx.context_ptr_ty(item),
                        self.source_info,
                    ));

                // ref_local
                let _ = self.builder.local_decls
                    .push(mir::LocalDecl::with_source_info(
                        tcx.context_ref_ty(item, muta, tcx.lifetimes.re_erased),
                        self.source_info,
                    ));

                entry.insert(ContextBinderItemInfo {
                    first_local,
                    muta,
                });
            }
            IndexEntry::Occupied(mut entry) => {
                let entry = entry.get_mut();
                if muta.is_mut() && entry.muta.is_not() {
                    entry.muta = Mutability::Mut;

                    self.builder.local_decls[entry.ref_local()].ty =
                        tcx.context_ref_ty(
                            item,
                            Mutability::Mut,
                            tcx.lifetimes.re_erased,
                        );
                }
            }
        }
    }

    fn visit_pack_shape(&mut self, shape: &'thir thir::PackShape<'tcx>) {
        match shape {
            &thir::PackShape::ExtractEnv(muta, item, binder) => {
                self.introduce_use(item, muta, binder);
            }
            thir::PackShape::Tuple(fields) => {
                for field in fields {
                    self.visit_pack_shape(field);
                }
            }
            thir::PackShape::ExtractLocal(..) | thir::PackShape::Error(..) => {
                // (does not introduce context uses)
            }
        }
    }
}

impl<'a, 'thir, 'tcx> thir_visit::Visitor<'thir, 'tcx> for BinderUseVisitor<'a, 'thir, 'tcx> {
    fn visit_block(&mut self, block: &'thir thir::Block) {
        let scope = self.bind_tracker.push_scope();

        for &stmt_id in &*block.stmts {
            use thir::StmtKind::*;

            let stmt = &self.thir()[stmt_id];

            match &stmt.kind {
                Expr { .. } => {}
                Let { .. } => {}
                &BindContext { bundle, .. } => {
                    let reified = self.builder.tcx.reified_bundle(self.thir()[bundle].ty);

                    for &item in reified.fields.keys() {
                        self.bind_tracker.bind(item, stmt_id);
                    }
                }
            }

            thir_visit::walk_stmt(self, stmt);
        }

        if let Some(expr) = block.expr {
            self.visit_expr(&self.thir()[expr]);
        }

        self.bind_tracker.pop_scope(scope);
    }

    fn visit_expr(&mut self, expr: &'thir thir::Expr<'tcx>) {
        use thir::ExprKind::*;

        match &expr.kind {
            &ContextRef { item, muta, binder, .. } => {
                self.introduce_use(item, muta, binder);
            }
            Pack { shape, .. } => {
                self.visit_pack_shape(shape);
            }
            &Call { ty, .. } => {
                if let Some(callee) = ty::extract_static_callee_for_context(self.builder.tcx, ty) {
                    for (item, muta) in self.builder.tcx.components_borrowed(callee).iter() {
                        self.introduce_use(item, muta, self.bind_tracker.resolve(item));
                    }
                }
            }

            Scope { .. }
            | Box { .. }
            | If { .. }
            | Deref { .. }
            | Binary { .. }
            | LogicalOp { .. }
            | Unary { .. }
            | Cast { .. }
            | Use { .. }
            | NeverToAny { .. }
            | PointerCoercion { .. }
            | Loop { .. }
            | Let { .. }
            | Match { .. }
            | Block { .. }
            | Assign { .. }
            | AssignOp { .. }
            | Field { .. }
            | Index { .. }
            | VarRef { .. }
            | UpvarRef { .. }
            | Borrow { .. }
            | RawBorrow { .. }
            | Break { .. }
            | Continue { .. }
            | Return { .. }
            | Become { .. }
            | ConstBlock { .. }
            | Repeat { .. }
            | Array { .. }
            | Tuple { .. }
            | Adt { .. }
            | PlaceTypeAscription { .. }
            | ValueTypeAscription { .. }
            | Closure { .. }
            | Literal { .. }
            | NonHirLiteral { .. }
            | ZstLiteral { .. }
            | NamedConst { .. }
            | ConstParam { .. }
            | StaticRef { .. }
            | InlineAsm { .. }
            | OffsetOf { .. }
            | ThreadLocalRef { .. }
            | Yield { .. } => {
                // (no context users directly introduced by expression)
            }
        }

        thir_visit::walk_expr(self, expr);
    }

    fn thir(&self) -> &'thir thir::Thir<'tcx> {
        self.builder.thir
    }
}
