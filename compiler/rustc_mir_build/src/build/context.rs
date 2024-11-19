#![expect(unused)]  // TODO: Remove

use rustc_data_structures::fx::{FxHashMap, FxIndexMap, IndexEntry};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::DefId;
use rustc_hir::Mutability;
use rustc_index::{Idx as _, IndexVec};
use rustc_middle::bug;
use rustc_middle::mir::{self, BasicBlock, Local, Operand, Place, Rvalue, SourceInfo};
use rustc_middle::thir::{self, visit::{self as thir_visit, Visitor as _}};
use rustc_middle::ty::{self, ContextBinder, Ty, TyCtxt, TypeFoldable as _};
use rustc_span::DUMMY_SP;
use rustc_target::abi::FieldIdx;

use thir_visit::Visitor as _;
use ty::ContextSolveStage::MirBuilding;

use super::Builder;

use crate::errors;

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
        assert!(!self.ctx_const_restrictions);

        let root_expr = &self.thir[root_expr];
        let mut visitor = BinderUseVisitor {
            builder: self,
            map: BinderMap::default(),
            bind_tracker: ty::ContextBindTracker::default(),
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
        assert!(!self.ctx_const_restrictions);

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
        let relate_refs = [lt_limiter].into_iter()
            .chain(binders.values().map(|item_info| Place::from(item_info.ref_local())))
            .collect::<Vec<_>>();

        let relate_csts = (1..relate_refs.len())
            .map(|binder| (0, binder))
            .collect::<Vec<_>>();

        self.relate_lifetimes(block, source_info, &relate_refs, &relate_csts);
    }

    pub(crate) fn lookup_context_binder(
        &self,
        item: DefId,
        binder: ContextBinder,
    ) -> ContextBinderItemInfo {
        assert!(!self.ctx_const_restrictions);

        *self.ctx_bind_values.expect_init()
            .get(&binder)
            .and_then(|v| v.get(&item))
            .unwrap_or_else(|| panic!("unknown context item {item:?} under binder {binder:?}"))
    }

    pub(crate) fn relate_lifetimes(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        refs: &[Place<'tcx>],
        constraints: &[(usize, usize)],
    ) {
        assert!(!self.ctx_const_restrictions);

        // Determine the type of the constraint marker. This will have the form...
        // `[(&'erased &'erased (), ..., &'erased &'erased ()); 0]`

        // Type: &'erased &'erased ()
        let single_constraint_ty = Ty::new_imm_ref(
            self.tcx,
            self.tcx.lifetimes.re_erased,
            Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_erased, self.tcx.types.unit)
        );

        // Type: `(&'erased &'erased (), ..., &'erased &'erased ())`
        let constraints_inner_ty = (0..constraints.len()).map(|_| single_constraint_ty);
        let constraints_inner_ty = Ty::new_tup_from_iter(self.tcx, constraints_inner_ty);

        // Type: `[(&'erased &'erased (), ..., &'erased &'erased ()); 0]`
        let constraints_ty = Ty::new_array(self.tcx, constraints_inner_ty, 0);

        // Create `ascribe_temp` temporary and give it the type...
        // `(&'erased {refs[0].muta} {refs[0].pointee}, ..., constraints_ty)`
        let ascribe_temp_ty = refs.iter()
            .map(|&ref_| ref_.ty(&self.local_decls, self.tcx).ty)
            .chain([constraints_ty]);
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

        let empty_array = self.local_decls.push(mir::LocalDecl::new(
            constraints_ty,
            source_info.span,
        ));
        self.cfg.push_assign(
            block,
            source_info,
            empty_array.into(),
            Rvalue::Aggregate(
                Box::new(mir::AggregateKind::Array(constraints_inner_ty)),
                IndexVec::new(),
            ),
        );
        assign_fields.push(mir::Operand::Move(empty_array.into()));

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
        let mut var_count = 0;
        let mut ascribe_temp_var_tys = Vec::new();

        // `.take()` ensures that we avoid the constraint marker.
        for ref_ty in refs_tys.iter().take(refs.len()) {
            let (pointee, muta) = match ref_ty.kind() {
                &ty::Ref(_re, pointee, muta) => (pointee, muta),
                _ => unreachable!("expected reference, got {ref_ty}"),
            };

            let re = ty::Region::new_bound(
                self.tcx,
                ty::DebruijnIndex::ZERO,
                ty::BoundRegion {
                    var: ty::BoundVar::from_u32(var_count),
                    kind: ty::BoundRegionKind::BrAnon,
                }
            );
            var_count += 1;

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

            ascribe_temp_var_tys.push(Ty::new_ref(self.tcx, re, pointee, muta));
        }

        let constraints_ascribe_ty = constraints.iter()
            .map(|(sub, sup)| {
                let outer_region = |idx: usize| {
                    let ty::Ref(re, _ty, _muta) = ascribe_temp_var_tys[idx].kind() else {
                        unreachable!();
                    };
                    *re
                };

                let sub = outer_region(*sub);
                let sup = outer_region(*sup);

                // We want `sub` to outlive `sup`. Recall that `&'a &'b ()` is only W.F. if `'b: 'a`.
                // In other words, the annotation is only allowed to make choices where `'b` outlives
                // `'a`. Substituting variable names, we need to use the type `&'sup &'sub`.
                Ty::new_imm_ref(
                    self.tcx,
                    sup,
                    Ty::new_imm_ref(self.tcx, sub, self.tcx.types.unit),
                )
            });

        let constraints_ascribe_ty = Ty::new_tup_from_iter(self.tcx, constraints_ascribe_ty);
        let constraints_ascribe_ty = Ty::new_array(self.tcx, constraints_ascribe_ty, 0);
        ascribe_temp_var_tys.push(constraints_ascribe_ty);

        let ascribe_temp_var_ty = Ty::new_tup_from_iter(self.tcx, ascribe_temp_var_tys.into_iter());

        let ascribe_variables = (0..var_count).map(|_| ty::CanonicalVarInfo {
            kind: ty::CanonicalVarKind::Region(ty::UniverseIndex::ROOT),
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

    pub(crate) fn new_lt_limiter_func(&mut self, block: BasicBlock, source_info: SourceInfo) -> Place<'tcx> {
        assert!(!self.ctx_const_restrictions);

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

    pub(crate) fn new_lt_limiter_static(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
    ) -> Place<'tcx> {
        assert!(!self.ctx_const_restrictions);

        let lt_limiter_ty = Ty::new_mut_ref(
            self.tcx,
            self.tcx.lifetimes.re_erased,
            self.tcx.types.unit,
        );

        let lt_limiter_ref = self.temp(lt_limiter_ty, source_info.span);

        self.cfg.push_assign(
            block,
            source_info,
            lt_limiter_ref,
            Rvalue::Use(
                Operand::Constant(Box::new(mir::ConstOperand {
                    span: source_info.span,
                    user_ty: None,
                    const_: mir::Const::Val(
                        mir::ConstValue::Scalar(
                            mir::interpret::Scalar::Int(
                                // Use 1 as the dangling pointer's address.
                                ty::ScalarInt::try_from_target_usize(1u32, self.tcx).unwrap(),
                            ),
                        ),
                        lt_limiter_ty,
                    ),
                })),
            ),
        );

        lt_limiter_ref
    }
}

struct BinderUseVisitor<'a, 'thir, 'tcx> {
    builder: &'a mut Builder<'thir, 'tcx>,
    bind_tracker: ty::ContextBindTracker,
    map: BinderMap,
    source_info: SourceInfo,
}

impl<'a, 'thir, 'tcx> BinderUseVisitor<'a, 'thir, 'tcx> {
    fn introduce_use(&mut self, item: DefId, muta: Mutability) {
        let tcx = self.builder.tcx;
        let binder_info = self.bind_tracker.resolve_rich(item);
        let binder = ContextBinder::from_info(binder_info);

        let muta = muta.min(binder_info.map_or(Mutability::Mut, |info| info.muta));

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
}

impl<'a, 'thir, 'tcx> thir_visit::Visitor<'thir, 'tcx> for BinderUseVisitor<'a, 'thir, 'tcx> {
    fn visit_block(&mut self, block: &'thir thir::Block) {
        let scope = self.bind_tracker.push_scope();
        thir_visit::walk_block(self, block);
        self.bind_tracker.pop_scope(scope);
    }

    fn visit_stmt(&mut self, stmt: &'thir thir::Stmt<'tcx>) {
        thir_visit::walk_stmt(self, stmt);

        // We bind the context after this statement has been visited to ensure that it isn't
        // visible to expressions in the statement.
        self.bind_tracker.bind_from_stmt(self.builder.tcx, MirBuilding, self.builder.thir, stmt);
    }

    fn visit_expr(&mut self, expr: &'thir thir::Expr<'tcx>) {
        let uses = ty::context_used_by_expr(
            self.builder.tcx,
            MirBuilding,
            &mut self.builder.ctx_pack_shapes,
            self.builder.thir,
            expr,
        );

        for (item, muta) in uses.concrete {
            self.introduce_use(item, muta);
        }

        thir_visit::walk_expr(self, expr);
    }

    fn thir(&self) -> &'thir thir::Thir<'tcx> {
        self.builder.thir
    }
}

// === Validation === //

pub(crate) fn check_for_fatal_context_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    thir: &thir::Thir<'tcx>,
    pack_shapes: &mut ty::PackShapeStore<'tcx>,
    const_restrictions: bool,
    expr: thir::ExprId,
) -> Result<(), ErrorGuaranteed> {
    let mut visitor = ContextFatalUseValidator {
        tcx,
        thir,
        pack_shapes,
        const_restrictions,
        err: None,
    };

    visitor.visit_expr(&thir[expr]);

    if let Some(err) = visitor.err {
        Err(err)
    } else {
        Ok(())
    }
}

struct ContextFatalUseValidator<'a, 'thir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: &'thir thir::Thir<'tcx>,
    pack_shapes: &'a mut ty::PackShapeStore<'tcx>,
    const_restrictions: bool,
    err: Option<ErrorGuaranteed>,
}

impl<'a, 'thir, 'tcx> ContextFatalUseValidator<'a, 'thir, 'tcx> {
    fn fatal(&mut self, err: ErrorGuaranteed) {
        self.err = Some(err);
    }

    fn visit_shape(&mut self, shape: &ty::PackShape<'_>) {
        match shape {
            ty::PackShape::Error(err) => {
                self.fatal(*err);
            }
            ty::PackShape::MakeTuple(fields) => {
                for field in fields {
                    self.visit_shape(field);
                }
            }
            ty::PackShape::MakeInfer { inner_shape, .. } => {
                self.visit_shape(inner_shape);
            }

            ty::PackShape::ExtractEnv(..)
            | ty::PackShape::ExtractLocalMove(..)
            | ty::PackShape::ExtractLocalRef(..) => {
                // (no additional validation)
            }

            ty::PackShape::ExtractEnvInfer(..)
            | ty::PackShape::ExtractLocalInferPlaceholder(..) => {
                bug!("{shape:?} does not show up after `GraphSolving`");
            }
        };
    }
}

impl<'a, 'thir, 'tcx> thir_visit::Visitor<'thir, 'tcx> for ContextFatalUseValidator<'a, 'thir, 'tcx> {
    fn visit_stmt(&mut self, stmt: &'thir thir::Stmt<'tcx>) {
        match &stmt.kind {
            &thir::StmtKind::BindContext { span, .. } => {
                if self.const_restrictions {
                    let err = self.tcx.dcx().emit_err(errors::ConstBindContextStmtUse {
                        span,
                    });
                    self.fatal(err);
                }
            }
            thir::StmtKind::Expr { .. } | thir::StmtKind::Let { .. } => {
                // (no additional validation)
            }
        }

        thir_visit::walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'thir thir::Expr<'tcx>) {
        use thir::ExprKind;

        match &expr.kind {
            ExprKind::ContextRef { .. } => {
                if self.const_restrictions {
                    let err = self.tcx.dcx().emit_err(errors::ConstContextItemUse {
                        span: expr.span,
                    });
                    self.fatal(err);
                }
            }
            ExprKind::Pack { .. } => {
                if self.const_restrictions {
                    let err = self.tcx.dcx().emit_err(errors::ConstPackExprUse {
                        span: expr.span,
                    });
                    self.fatal(err);
                }

                let shape = self.pack_shapes.resolve(
                    self.tcx,
                    MirBuilding,
                    self.thir,
                    expr,
                );

                self.visit_shape(&shape);
            }

            ExprKind::Scope { .. }
            | ExprKind::Box { .. }
            | ExprKind::If { .. }
            | ExprKind::Call { .. }
            | ExprKind::Deref { .. }
            | ExprKind::Binary { .. }
            | ExprKind::LogicalOp { .. }
            | ExprKind::Unary { .. }
            | ExprKind::Cast { .. }
            | ExprKind::Use { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::PointerCoercion { .. }
            | ExprKind::Loop { .. }
            | ExprKind::Let { .. }
            | ExprKind::Match { .. }
            | ExprKind::Block { .. }
            | ExprKind::Assign { .. }
            | ExprKind::AssignOp { .. }
            | ExprKind::Field { .. }
            | ExprKind::Index { .. }
            | ExprKind::VarRef { .. }
            | ExprKind::UpvarRef { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::RawBorrow { .. }
            | ExprKind::Break { .. }
            | ExprKind::Continue { .. }
            | ExprKind::Return { .. }
            | ExprKind::Become { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::Array { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Adt { .. }
            | ExprKind::PlaceTypeAscription { .. }
            | ExprKind::ValueTypeAscription { .. }
            | ExprKind::Closure { .. }
            | ExprKind::Literal { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ZstLiteral { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::StaticRef { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::OffsetOf { .. }
            | ExprKind::ThreadLocalRef { .. }
            | ExprKind::Yield { .. } => {
                // (no additional validation)
            }
        }

        thir_visit::walk_expr(self, expr);
    }

    fn thir(&self) -> &'thir thir::Thir<'tcx> {
        self.thir
    }
}
