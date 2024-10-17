#![expect(unused)]

use rustc_data_structures::fx::{FxHashMap, FxIndexMap, IndexEntry};
use rustc_hir::def_id::DefId;
use rustc_hir::Mutability;
use rustc_index::Idx;
use rustc_middle::mir::{self, BasicBlock, Local, Place, Rvalue, SourceInfo};
use rustc_middle::thir::{self, ContextBinder, visit::{self as thir_visit, Visitor as _}};
use rustc_span::DUMMY_SP;

use thir_visit::Visitor as _;

use super::Builder;

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
    pub(crate) fn define_context_locals(&mut self, root_expr: thir::ExprId) {
        assert!(self.context_binders.0.is_none(), "`define_context_locals` called twice");

        let root_expr = &self.thir.exprs[root_expr];
        let mut visitor = BinderUseVisitor {
            builder: self,
            map: BinderMap::default(),
        };
        visitor.visit_expr(root_expr);
        self.context_binders.0 = Some(visitor.map);
    }

    pub(crate) fn init_and_borrow_context_binder_locals(
        &mut self,
        block: BasicBlock,
        binder: ContextBinder,
    ) {
        let Some(binders) = self.context_binders.expect_init().get(&binder) else {
            return;
        };

        // Initialize the pointer locals
        let source_info = SourceInfo {  // TODO: Update this!
            span: DUMMY_SP,
            scope: self.source_scope,
        };

        for (&item, &item_info) in binders {
            self.cfg.push_assign(
                block,
                source_info,
                item_info.ptr_local().into(),
                Rvalue::ContextRef(item),
            );
        }

        // Reborrow them
        self.reborrow_context_binder_locals(block, binder);

        // Limit their lifetimes to the scope of the bind
        // TODO: Do this!
    }

    pub(crate) fn reborrow_context_binder_locals(&mut self, block: BasicBlock, binder: ContextBinder) {
        let Some(binders) = self.context_binders.expect_init().get(&binder) else {
            return;
        };

        let source_info = SourceInfo {  // TODO: Update this!
            span: DUMMY_SP,
            scope: self.source_scope,
        };
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
    }

    pub(crate) fn lookup_context_binder(
        &self,
        item: DefId,
        binder: ContextBinder,
    ) -> ContextBinderItemInfo {
        *self.context_binders.expect_init()
            .get(&binder)
            .and_then(|v| v.get(&item))
            .unwrap_or_else(|| panic!("unknown context item {item:?} under binder {binder:?}"))
    }
}

struct BinderUseVisitor<'a, 'thir, 'tcx> {
    builder: &'a mut Builder<'thir, 'tcx>,
    map: BinderMap,
}

impl<'a, 'thir, 'tcx> thir_visit::Visitor<'thir, 'tcx> for BinderUseVisitor<'a, 'thir, 'tcx> {
    fn visit_expr(&mut self, expr: &'thir thir::Expr<'tcx>) {
        if let &thir::ExprKind::ContextRef { item, muta, binder, .. } = &expr.kind {
            let tcx = self.builder.tcx;

            let source_info = SourceInfo {  // TODO: Update this!
                span: DUMMY_SP,
                scope: self.builder.source_scope,
            };

            let item_entry = self.map.entry(binder)
                .or_default()
                .entry(item);

            match item_entry {
                IndexEntry::Vacant(entry) => {
                    // ptr_local
                    let first_local = self.builder.local_decls
                        .push(mir::LocalDecl::with_source_info(
                            tcx.context_ptr_ty(item),
                            source_info,
                        ));

                    // ref_local
                    let _ = self.builder.local_decls
                        .push(mir::LocalDecl::with_source_info(
                            tcx.context_ref_ty(item, muta, tcx.lifetimes.re_erased),
                            source_info,
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

        thir_visit::walk_expr(self, expr);
    }

    fn thir(&self) -> &'thir thir::Thir<'tcx> {
        self.builder.thir
    }
}
