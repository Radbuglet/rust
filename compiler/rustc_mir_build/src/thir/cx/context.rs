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
        let kind = ExprKind::ContextRef {
            item: id,
            muta: Mutability::Not,
        };
        let ty = self.tcx.context_ref_ty(id, Mutability::Not, self.tcx.lifetimes.re_erased);
        let temp_lifetime = self
            .rvalue_scopes
            .temporary_scope(self.region_scope_tree, expr.hir_id.local_id);

        ExprKind::Deref {
            arg: self.thir.exprs.push(Expr { ty, temp_lifetime, span: expr.span, kind }),
        }
    }

    pub(crate) fn adjust_context(&mut self, id: ExprId, rvalue_mut: Mutability) {
        use ExprKind::*;

        let mut expr = mem::replace(&mut self.thir.exprs[id], dummy_expr(self.tcx));

        match &mut expr.kind {
            &mut Scope { region_scope: _, lint_level: _, value } => {
                self.adjust_context(value, rvalue_mut);
            },
            &mut Box { value } => {
                self.adjust_context(value, Mutability::Not);
            },
            &mut If { if_then_scope: _, cond, then, else_opt } => {
                self.adjust_context(cond, Mutability::Not);
                self.adjust_context(then, Mutability::Not);
                if let Some(else_opt) = else_opt {
                    self.adjust_context(else_opt, Mutability::Not);
                }
            },
            Call { ty: _, fun, args, from_hir_call: _, fn_span: _ } => {
                self.adjust_context(*fun, Mutability::Not);
                for &mut arg in args {
                    self.adjust_context(arg, Mutability::Not);
                }
            },
            &mut Deref { arg } => {
                self.adjust_context(arg, rvalue_mut);
            },
            &mut Binary { op: _, lhs, rhs } => {
                self.adjust_context(lhs, Mutability::Not);
                self.adjust_context(rhs, Mutability::Not);
            },
            &mut LogicalOp { op: _, lhs, rhs } => {
                self.adjust_context(lhs, Mutability::Not);
                self.adjust_context(rhs, Mutability::Not);
            },
            &mut Unary { op: _, arg } => {
                self.adjust_context(arg, Mutability::Not);
            },
            &mut Cast { source } => {
                self.adjust_context(source, Mutability::Not);
            },
            &mut Use { source } => {
                self.adjust_context(source, Mutability::Not);
            },
            &mut NeverToAny { source } => {
                self.adjust_context(source, Mutability::Not);
            },
            &mut PointerCoercion { cast: _, source, is_from_as_cast: _ } => {
                self.adjust_context(source, Mutability::Not);
            },
            &mut Loop { body } => {
                self.adjust_context(body, Mutability::Not);
            },
            &mut Let { expr, pat: _ } => {
                self.adjust_context(expr, Mutability::Not);
            },
            Match { scrutinee, scrutinee_hir_id: _, arms, match_source: _ } => {
                self.adjust_context(*scrutinee, Mutability::Not);

                for &mut arm in arms {
                    self.adjust_context_arm(arm);
                }
            },
            &mut Block { block } => {
                self.adjust_context_block(block);
            },
            &mut Assign { lhs, rhs } => {
                self.adjust_context(lhs, Mutability::Mut);
                self.adjust_context(rhs, Mutability::Not);
            },
            &mut AssignOp { op: _, lhs, rhs } => {
                self.adjust_context(lhs, Mutability::Mut);
                self.adjust_context(rhs, Mutability::Not);
            },
            &mut Field { lhs, variant_index: _, name: _ } => {
                self.adjust_context(lhs, rvalue_mut);
            },
            &mut Index { lhs, index } => {
                self.adjust_context(lhs, rvalue_mut);
                self.adjust_context(index, Mutability::Not);
            },
            &mut VarRef { id: _ } => {
                // (terminal)
            },
            &mut UpvarRef { closure_def_id: _, var_hir_id: _ } => {
                // (terminal)
            },
            &mut Borrow { borrow_kind, arg } => {
                self.adjust_context(arg, borrow_kind.to_mutbl_lossy());
            },
            &mut RawBorrow { mutability, arg } => {
                self.adjust_context(arg, mutability);
            },
            &mut Break { label: _, value } => {
                if let Some(value) = value {
                    self.adjust_context(value, Mutability::Not);
                }
            },
            &mut Continue { label: _ } => {
                // (terminal)
            },
            &mut Return { value } => {
                if let Some(value) = value {
                    self.adjust_context(value, Mutability::Not);
                }
            },
            &mut Become { value } => {
                self.adjust_context(value, Mutability::Not);
            },
            &mut ConstBlock { did: _, args: _ } => {
                // (terminal)
            },
            &mut Repeat { value, count: _ } => {
                self.adjust_context(value, Mutability::Not);
            },
            Array { fields } => {
                for &mut field in fields {
                    self.adjust_context(field, Mutability::Not);
                }
            },
            Tuple { fields } => {
                for &mut field in fields {
                    self.adjust_context(field, Mutability::Not);
                }
            },
            Adt(adt) => {
                for field in &adt.fields {
                    self.adjust_context(field.expr, Mutability::Not);
                }
            },
            &mut PlaceTypeAscription { source, user_ty: _, user_ty_span: _ } => {
                self.adjust_context(source, Mutability::Not);
            },
            &mut ValueTypeAscription { source, user_ty: _, user_ty_span: _ } => {
                self.adjust_context(source, Mutability::Not);
            },
            Closure(expr) => {
                for &upvar in &expr.upvars {
                    self.adjust_context(upvar, Mutability::Not);
                }

                for &(fake_read, _cause, _id) in &expr.fake_reads {
                    self.adjust_context(fake_read, Mutability::Not);
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
            InlineAsm(_expr) => {
                // TODO
            },
            &mut OffsetOf { container: _, fields: _ } => {
                // (terminal)
            },
            &mut ThreadLocalRef(_did) => {
                // (terminal)
            },
            ContextRef { item, muta } => {
                if rvalue_mut.is_mut() {
                    *muta = Mutability::Mut;
                    expr.ty = self.tcx.context_ref_ty(
                        *item,
                        Mutability::Mut,
                        self.tcx.lifetimes.re_erased,
                    );
                }
            },
            Pack { exprs, shape: _ } => {
                for &mut expr in exprs {
                    self.adjust_context(expr, Mutability::Not);
                }
            }
            &mut Yield { value } => {
                self.adjust_context(value, Mutability::Not);
            },
        }

        self.thir.exprs[id] = expr;
    }

    fn adjust_context_block(&mut self, id: BlockId) {
        let block = mem::replace(&mut self.thir.blocks[id], dummy_block());

        for &stmt in &block.stmts {
            self.adjust_context_stmt(stmt);
        }

        if let Some(expr) = block.expr {
            self.adjust_context(expr, Mutability::Not);
        }

        self.thir.blocks[id] = block;
    }

    fn adjust_context_stmt(&mut self, id: StmtId) {
        let stmt = mem::replace(&mut self.thir.stmts[id], dummy_stmt());

        match &stmt.kind {
            &StmtKind::Expr { scope: _, expr } => {
                self.adjust_context(expr, Mutability::Not);
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
                    self.adjust_context(initializer, Mutability::Not);
                }

                if let Some(else_block) = *else_block {
                    self.adjust_context_block(else_block);
                }
            }
            StmtKind::BindContext {
                remainder_scope: _,
                init_scope: _,
                bundle,
                span: _,
                self_id: _,
            } => {
                self.adjust_context(*bundle, Mutability::Not);
            }
        }

        self.thir.stmts[id] = stmt;
    }

    fn adjust_context_arm(&mut self, id: ArmId) {
        let arm = &self.thir.arms[id];

        let guard = arm.guard;
        let body = arm.body;

        if let Some(guard) = guard {
            self.adjust_context(guard, Mutability::Not);
        }

        self.adjust_context(body, Mutability::Not);
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
