//! By this point, we have full type information in our IR so there's no point deferring
//! this discussion: we need to determine the mutability of our context borrow!
//!
//! That if, if we write:
//!
//! ```rust,ignore
//! #[context]
//! static MY_CX: Vec<u32>;
//!
//! fn demo() use MY_CX {
//!     MY_CX.len();
//!     MY_CX.clear();
//! }
//! ```
//!
//! ...the first line should reflect the fact that `MY_CX` is being immutably borrowed
//! while the second line should reflect the fact that `MY_CX` is being mutably borrowed.
//!
//! If we treated context items as if they were `ExprKind::VarRef`s, we'd have to
//! duplicate the logic used to invalid moves out of references. Instead, it's much
//! nicer to desugar this as an expression producing some kind of reference to our
//! context item (à la `ExprKind::ThreadLocalRef`) and dereference that.
//!
//! This then leaves an important question: what type do we give the "dereferencee?"
//!
//! We have two options:
//!
//! 1. We could determine the required mutability for the reference here and now like
//!    we do when desugaring overloaded dereferences to their UFCS form.
//!
//! 2. We could produce a mutable raw pointer and create an exemption for immediate
//!    dereferences of the pointer during THIR safety checking.
//!
//! The former has the advantage of making it really easy to detect which objects'
//! lifetimes depend upon which borrows of which context items: just find the lifetimes
//! of the references produced by these operations. However, to do this type of early
//! detection, we'd have to modify the type checker or the THIR lowering logic.
//!
//! The latter makes THIR generation really easy. However, if we choose this strategy,
//! we now get ourselves into a new and exciting issue: how will MIR passes learn
//! about the contextual borrow? `expr_as_place` works expression by expression so we
//! wouldn't know about the way in which the pointer is being borrowed until later.
//! If `Context` path expressions were resolved as `ExprKind::ContextRef`s with type
//! `*mut T`, the generated MIR would basically have to look something like this:
//!
//! ```rust,ignore
//! #[context]
//! static MY_CTX: Vec<u32>;
//!
//! fn my_demo() {
//!     let borrow = unsafe { &mut MY_CTX };
//! }
//! ```
//!
//! ```rust,ignore
//! // WARNING: This output format is intended for human consumers only
//! // and is subject to change without notice. Knock yourself out.
//! fn my_demo() -> () {
//!     let mut _0: ();
//!     let mut _1: &mut std::vec::Vec<u32>;
//!     let mut _2: *mut std::vec::Vec<u32>;
//!     scope 1 {
//!         debug borrow => _1;
//!     }
//!
//!     bb0: {
//!         _2 = &/*ctx*/ mut MY_CTX;
//!         _1 = &mut (*_2);
//!         return;
//!     }
//! }
//! ```
//!
//! Detecting that `_1` is actually borrowing a context item would require us to know
//! that `_2`'s pointer actually came from an `Rvalue::ContextRef`, which is impossible
//! to figure out generally without special-casing this exact codegen pattern or
//! introducing a new kind of pointer. Yuck!
//!
//! Our only other option to avoid modifying the type checker is to treat `ExprKind::ContextRef`
//! as an actual place. This would work elegantly because we'd wouldn't have to erase
//! the place into a raw pointer during either THIR lowering or MIR lowering but, in
//! order to accomplish this, we'd need to make the MIR's `Place` type accept context
//! items as the place's base, which seems like an unnecessarily large refactor.
//!
//! Anyways, that's why we resolve the borrow kind of context places so early.

use std::mem;

use rustc_ast::Mutability;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::region;
use rustc_middle::thir::*;
use rustc_middle::ty::TyCtxt;
use rustc_span::DUMMY_SP;

use super::Cx;

impl<'tcx> Cx<'tcx> {
    pub(crate) fn create_dummy_context_expr(
        &mut self,
        expr: &'tcx hir::Expr<'tcx>,
        id: DefId,
    ) -> ExprKind<'tcx> {
        let kind = ExprKind::ContextRef(id, Mutability::Not);
        let ty = self.tcx.context_ptr_ty(id, Mutability::Not, self.tcx.lifetimes.re_erased);
        let temp_lifetime = self
            .rvalue_scopes
            .temporary_scope(self.region_scope_tree, expr.hir_id.local_id);

        ExprKind::Deref {
            arg: self.thir.exprs.push(Expr { ty, temp_lifetime, span: expr.span, kind }),
        }
    }

    #[allow(unused_variables)]
    pub(crate) fn adjust_context_mutabilities(&mut self, id: ExprId, rvalue_mut: Mutability) {
        use ExprKind::*;

        let mut expr = mem::replace(&mut self.thir.exprs[id], dummy_expr(self.tcx));

        match &mut expr.kind {
            &mut Scope { region_scope: _, lint_level: _, value } => {
                self.adjust_context_mutabilities(value, rvalue_mut);
            },
            &mut Box { value } => {
                self.adjust_context_mutabilities(value, Mutability::Not);
            },
            &mut If { if_then_scope: _, cond, then, else_opt } => {
                self.adjust_context_mutabilities(cond, Mutability::Not);
                self.adjust_context_mutabilities(then, Mutability::Not);
                if let Some(else_opt) = else_opt {
                    self.adjust_context_mutabilities(else_opt, Mutability::Not);
                }
            },
            Call { ty: _, fun, args, from_hir_call: _, fn_span: _ } => {
                self.adjust_context_mutabilities(*fun, Mutability::Not);
                for &mut arg in args {
                    self.adjust_context_mutabilities(arg, Mutability::Not);
                }
            },
            &mut Deref { arg } => {
                self.adjust_context_mutabilities(arg, rvalue_mut);
            },
            &mut Binary { op: _, lhs, rhs } => {
                self.adjust_context_mutabilities(lhs, Mutability::Not);
                self.adjust_context_mutabilities(rhs, Mutability::Not);
            },
            &mut LogicalOp { op: _, lhs, rhs } => {
                self.adjust_context_mutabilities(lhs, Mutability::Not);
                self.adjust_context_mutabilities(rhs, Mutability::Not);
            },
            &mut Unary { op: _, arg } => {
                self.adjust_context_mutabilities(arg, Mutability::Not);
            },
            &mut Cast { source } => {
                self.adjust_context_mutabilities(source, Mutability::Not);
            },
            &mut Use { source } => {
                self.adjust_context_mutabilities(source, Mutability::Not);
            },
            &mut NeverToAny { source } => {
                self.adjust_context_mutabilities(source, Mutability::Not);
            },
            &mut PointerCoercion { cast: _, source, is_from_as_cast: _ } => {
                self.adjust_context_mutabilities(source, Mutability::Not);
            },
            &mut Loop { body } => {
                self.adjust_context_mutabilities(body, Mutability::Not);
            },
            &mut Let { expr, pat: _ } => {
                self.adjust_context_mutabilities(expr, Mutability::Not);
            },
            Match { scrutinee, scrutinee_hir_id: _, arms, match_source: _ } => {
                self.adjust_context_mutabilities(*scrutinee, Mutability::Not);

                for &mut arm in arms {
                    self.adjust_context_mutabilities_arm(arm);
                }
            },
            &mut Block { block } => {
                self.adjust_context_mutabilities_block(block);
            },
            &mut Assign { lhs, rhs } => {
                self.adjust_context_mutabilities(lhs, Mutability::Mut);
                self.adjust_context_mutabilities(rhs, Mutability::Not);
            },
            &mut AssignOp { op: _, lhs, rhs } => {
                self.adjust_context_mutabilities(lhs, Mutability::Mut);
                self.adjust_context_mutabilities(rhs, Mutability::Not);
            },
            &mut Field { lhs, variant_index: _, name: _ } => {
                self.adjust_context_mutabilities(lhs, rvalue_mut);
            },
            &mut Index { lhs, index } => {
                self.adjust_context_mutabilities(lhs, rvalue_mut);
                self.adjust_context_mutabilities(index, Mutability::Not);
            },
            &mut VarRef { id: _ } => {
                // (terminal)
            },
            &mut UpvarRef { closure_def_id: _, var_hir_id: _ } => {
                // (terminal)
            },
            &mut Borrow { borrow_kind, arg } => {
                self.adjust_context_mutabilities(arg, borrow_kind.to_mutbl_lossy());
            },
            &mut RawBorrow { mutability, arg } => {
                self.adjust_context_mutabilities(arg, mutability);
            },
            &mut Break { label: _, value } => {
                if let Some(value) = value {
                    self.adjust_context_mutabilities(value, Mutability::Not);
                }
            },
            &mut Continue { label: _ } => {
                // (terminal)
            },
            &mut Return { value } => {
                if let Some(value) = value {
                    self.adjust_context_mutabilities(value, Mutability::Not);
                }
            },
            &mut Become { value } => {
                self.adjust_context_mutabilities(value, Mutability::Not);
            },
            &mut ConstBlock { did: _, args: _ } => {
                // (terminal)
            },
            &mut Repeat { value, count: _ } => {
                self.adjust_context_mutabilities(value, Mutability::Not);
            },
            Array { fields } => {
                for &mut field in fields {
                    self.adjust_context_mutabilities(field, Mutability::Not);
                }
            },
            Tuple { fields } => {
                for &mut field in fields {
                    self.adjust_context_mutabilities(field, Mutability::Not);
                }
            },
            Adt(adt) => {
                for field in &adt.fields {
                    self.adjust_context_mutabilities(field.expr, Mutability::Not);
                }
            },
            &mut PlaceTypeAscription { source, user_ty: _, user_ty_span: _ } => {
                self.adjust_context_mutabilities(source, Mutability::Not);
            },
            &mut ValueTypeAscription { source, user_ty: _, user_ty_span: _ } => {
                self.adjust_context_mutabilities(source, Mutability::Not);
            },
            Closure(expr) => {
                for &upvar in &expr.upvars {
                    self.adjust_context_mutabilities(upvar, Mutability::Not);
                }

                for &(fake_read, _cause, _id) in &expr.fake_reads {
                    self.adjust_context_mutabilities(fake_read, Mutability::Not);
                }
            },
            &mut Literal { lit: _, neg: _ } => {
                // (terminal)
            },
            &mut NonHirLiteral { lit: _, user_ty: _ } => {
                // (terminal)
            },
            &mut ZstLiteral { user_ty: _ } => {
                // (terminal)
            },
            &mut NamedConst { def_id: _, args: _, user_ty: _ } => {
                // (terminal)
            },
            &mut ConstParam { param: _, def_id: _ } => {
                // (terminal)
            },
            &mut StaticRef { alloc_id: _, ty: _, def_id: _ } => {
                // (terminal)
            },
            InlineAsm(expr) => {
                todo!()
            },
            &mut OffsetOf { container: _, fields: _ } => {
                // (terminal)
            },
            &mut ThreadLocalRef(_did) => {
                // (terminal)
            },
            ContextRef(id, mutbl) => {
                if rvalue_mut.is_mut() {
                    *mutbl = Mutability::Mut;
                    expr.ty = self.tcx.context_ptr_ty(
                        *id,
                        Mutability::Mut,
                        self.tcx.lifetimes.re_erased,
                    );
                }
            },
            &mut Yield { value } => {
                self.adjust_context_mutabilities(value, Mutability::Not);
            },
        }

        self.thir.exprs[id] = expr;
    }

    fn adjust_context_mutabilities_block(&mut self, id: BlockId) {
        let block = mem::replace(&mut self.thir.blocks[id], dummy_block());

        for &stmt in &block.stmts {
            self.adjust_context_mutabilities_stmt(stmt);
        }

        if let Some(expr) = block.expr {
            self.adjust_context_mutabilities(expr, Mutability::Not);
        }

        self.thir.blocks[id] = block;
    }

    fn adjust_context_mutabilities_stmt(&mut self, id: StmtId) {
        let stmt = mem::replace(&mut self.thir.stmts[id], dummy_stmt());

        match &stmt.kind {
            &StmtKind::Expr { scope: _, expr } => {
                self.adjust_context_mutabilities(expr, Mutability::Not);
            }
            StmtKind::Let {
                remainder_scope: _,
                init_scope: _,
                pattern: _,
                initializer,
                else_block,
                lint_level: _,
                span: _,
            } => {
                if let Some(initializer) = *initializer {
                    self.adjust_context_mutabilities(initializer, Mutability::Not);
                }

                if let Some(else_block) = *else_block {
                    self.adjust_context_mutabilities_block(else_block);
                }
            }
        }

        self.thir.stmts[id] = stmt;
    }

    fn adjust_context_mutabilities_arm(&mut self, id: ArmId) {
        let arm = &self.thir.arms[id];

        let guard = arm.guard;
        let body = arm.body;

        if let Some(guard) = guard {
            self.adjust_context_mutabilities(guard, Mutability::Not);
        }

        self.adjust_context_mutabilities(body, Mutability::Not);
    }
}

fn dummy_block() -> Block {
    Block {
        targeted_by_break: false,
        region_scope: dummy_scope(),
        span: DUMMY_SP,
        stmts: Box::new([]),
        expr: None,
        safety_mode: BlockSafety::Safe,
    }
}

fn dummy_expr<'tcx>(tcx: TyCtxt<'tcx>) -> Expr<'tcx> {
    Expr {
        kind: ExprKind::Yield { value: ExprId::MAX },
        ty: tcx.types.unit,
        temp_lifetime: None,
        span: DUMMY_SP,
    }
}

fn dummy_stmt<'tcx>() -> Stmt<'tcx> {
    Stmt {
        kind: StmtKind::Expr {
            scope: dummy_scope(),
            expr: ExprId::MAX,
        }
    }
}

fn dummy_scope() -> region::Scope {
    region::Scope {
        id: hir::ItemLocalId::MAX,
        data: region::ScopeData::Node,
    }
}
