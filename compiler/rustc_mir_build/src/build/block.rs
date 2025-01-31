use rustc_middle::middle::region::Scope;
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::{span_bug, ty};
use rustc_span::Span;
use tracing::debug;

use crate::build::DropKind;
use crate::build::ForGuard::OutsideGuard;
use crate::build::matches::{DeclareLetBindings, EmitStorageLive, ScheduleDrops};
use crate::build::{BlockAnd, BlockAndExtension, BlockFrame, Builder};

use ty::ContextSolveStage::MirBuilding;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    pub(crate) fn ast_block(
        &mut self,
        destination: Place<'tcx>,
        block: BasicBlock,
        ast_block: BlockId,
        source_info: SourceInfo,
    ) -> BlockAnd<()> {
        let context_scope = self.ctx_bind_tracker.push_scope();

        let Block { region_scope, span, ref stmts, expr, targeted_by_break, safety_mode: _ } =
            self.thir[ast_block];

        let res = self.in_scope((region_scope, source_info), LintLevel::Inherited, move |this| {
            if targeted_by_break {
                this.in_breakable_scope(None, destination, span, |this| {
                    Some(this.ast_block_stmts(destination, block, span, stmts, expr, region_scope))
                })
            } else {
                this.ast_block_stmts(destination, block, span, stmts, expr, region_scope)
            }
        });

        self.ctx_bind_tracker.pop_scope(context_scope);

        res
    }

    fn ast_block_stmts(
        &mut self,
        destination: Place<'tcx>,
        mut block: BasicBlock,
        span: Span,
        stmts: &[StmtId],
        expr: Option<ExprId>,
        region_scope: Scope,
    ) -> BlockAnd<()> {
        let this = self;

        // This convoluted structure is to avoid using recursion as we walk down a list
        // of statements. Basically, the structure we get back is something like:
        //
        //    let x = <init> in {
        //       expr1;
        //       let y = <init> in {
        //           expr2;
        //           expr3;
        //           ...
        //       }
        //    }
        //
        // The let bindings are valid till the end of block so all we have to do is to pop all
        // the let-scopes at the end.
        //
        // First we build all the statements in the block.
        let mut let_scope_stack = Vec::with_capacity(8);
        let outer_source_scope = this.source_scope;
        // This scope information is kept for breaking out of the parent remainder scope in case
        // one let-else pattern matching fails.
        // By doing so, we can be sure that even temporaries that receive extended lifetime
        // assignments are dropped, too.
        let mut last_remainder_scope = region_scope;

        let source_info = this.source_info(span);
        for stmt in stmts {
            let Stmt { ref kind } = this.thir[*stmt];
            match kind {
                StmtKind::Expr { scope, expr } => {
                    this.block_context.push(BlockFrame::Statement { ignores_expr_result: true });
                    let si = (*scope, source_info);
                    block = this
                        .in_scope(si, LintLevel::Inherited, |this| {
                            this.stmt_expr(block, *expr, Some(*scope))
                        })
                        .into_block();
                }
                StmtKind::Let {
                    remainder_scope,
                    init_scope,
                    pattern,
                    initializer: Some(initializer),
                    lint_level,
                    else_block: Some(else_block),
                    span: _,
                } => {
                    // When lowering the statement `let <pat> = <expr> else { <else> };`,
                    // the `<else>` block is nested in the parent scope enclosing this statement.
                    // That scope is usually either the enclosing block scope,
                    // or the remainder scope of the last statement.
                    // This is to make sure that temporaries instantiated in `<expr>` are dropped
                    // as well.
                    // In addition, even though bindings in `<pat>` only come into scope if
                    // the pattern matching passes, in the MIR building the storages for them
                    // are declared as live any way.
                    // This is similar to `let x;` statements without an initializer expression,
                    // where the value of `x` in this example may or may be assigned,
                    // because the storage for their values may not be live after all due to
                    // failure in pattern matching.
                    // For this reason, we declare those storages as live but we do not schedule
                    // any drop yet- they are scheduled later after the pattern matching.
                    // The generated MIR will have `StorageDead` whenever the control flow breaks out
                    // of the parent scope, regardless of the result of the pattern matching.
                    // However, the drops are inserted in MIR only when the control flow breaks out of
                    // the scope of the remainder scope associated with this `let .. else` statement.
                    // Pictorial explanation of the scope structure:
                    // ┌─────────────────────────────────┐
                    // │  Scope of the enclosing block,  │
                    // │  or the last remainder scope    │
                    // │  ┌───────────────────────────┐  │
                    // │  │  Scope for <else> block   │  │
                    // │  └───────────────────────────┘  │
                    // │  ┌───────────────────────────┐  │
                    // │  │  Remainder scope of       │  │
                    // │  │  this let-else statement  │  │
                    // │  │  ┌─────────────────────┐  │  │
                    // │  │  │ <expr> scope        │  │  │
                    // │  │  └─────────────────────┘  │  │
                    // │  │  extended temporaries in  │  │
                    // │  │  <expr> lives in this     │  │
                    // │  │  scope                    │  │
                    // │  │  ┌─────────────────────┐  │  │
                    // │  │  │ Scopes for the rest │  │  │
                    // │  │  └─────────────────────┘  │  │
                    // │  └───────────────────────────┘  │
                    // └─────────────────────────────────┘
                    // Generated control flow:
                    //          │ let Some(x) = y() else { return; }
                    //          │
                    // ┌────────▼───────┐
                    // │ evaluate y()   │
                    // └────────┬───────┘
                    //          │              ┌────────────────┐
                    // ┌────────▼───────┐      │Drop temporaries│
                    // │Test the pattern├──────►in y()          │
                    // └────────┬───────┘      │because breaking│
                    //          │              │out of <expr>   │
                    // ┌────────▼───────┐      │scope           │
                    // │Move value into │      └───────┬────────┘
                    // │binding x       │              │
                    // └────────┬───────┘      ┌───────▼────────┐
                    //          │              │Drop extended   │
                    // ┌────────▼───────┐      │temporaries in  │
                    // │Drop temporaries│      │<expr> because  │
                    // │in y()          │      │breaking out of │
                    // │because breaking│      │remainder scope │
                    // │out of <expr>   │      └───────┬────────┘
                    // │scope           │              │
                    // └────────┬───────┘      ┌───────▼────────┐
                    //          │              │Enter <else>    ├────────►
                    // ┌────────▼───────┐      │block           │ return;
                    // │Continue...     │      └────────────────┘
                    // └────────────────┘

                    let ignores_expr_result = matches!(pattern.kind, PatKind::Wild);
                    this.block_context.push(BlockFrame::Statement { ignores_expr_result });

                    // Lower the `else` block first because its parent scope is actually
                    // enclosing the rest of the `let .. else ..` parts.
                    let else_block_span = this.thir[*else_block].span;
                    // This place is not really used because this destination place
                    // should never be used to take values at the end of the failure
                    // block.
                    let dummy_place = this.temp(this.tcx.types.never, else_block_span);
                    let failure_entry = this.cfg.start_new_block();
                    let failure_block;
                    failure_block = this
                        .ast_block(
                            dummy_place,
                            failure_entry,
                            *else_block,
                            this.source_info(else_block_span),
                        )
                        .into_block();
                    this.cfg.terminate(
                        failure_block,
                        this.source_info(else_block_span),
                        TerminatorKind::Unreachable,
                    );

                    // Declare the bindings, which may create a source scope.
                    let remainder_span = remainder_scope.span(this.tcx, this.region_scope_tree);
                    this.push_scope((*remainder_scope, source_info));
                    let_scope_stack.push(remainder_scope);

                    let visibility_scope =
                        Some(this.new_source_scope(remainder_span, LintLevel::Inherited));

                    let initializer_span = this.thir[*initializer].span;
                    let scope = (*init_scope, source_info);
                    let failure_and_block = this.in_scope(scope, *lint_level, |this| {
                        this.declare_bindings(
                            visibility_scope,
                            remainder_span,
                            pattern,
                            None,
                            Some((Some(&destination), initializer_span)),
                        );
                        this.visit_primary_bindings(
                            pattern,
                            UserTypeProjections::none(),
                            &mut |this, _, _, node, span, _, _| {
                                this.storage_live_binding(
                                    block,
                                    node,
                                    span,
                                    OutsideGuard,
                                    ScheduleDrops::Yes,
                                );
                            },
                        );
                        let else_block_span = this.thir[*else_block].span;
                        let (matching, failure) =
                            this.in_if_then_scope(last_remainder_scope, else_block_span, |this| {
                                this.lower_let_expr(
                                    block,
                                    *initializer,
                                    pattern,
                                    None,
                                    initializer_span,
                                    DeclareLetBindings::No,
                                    EmitStorageLive::No,
                                )
                            });
                        matching.and(failure)
                    });
                    let failure = unpack!(block = failure_and_block);
                    this.cfg.goto(failure, source_info, failure_entry);

                    if let Some(source_scope) = visibility_scope {
                        this.source_scope = source_scope;
                    }
                    last_remainder_scope = *remainder_scope;
                }
                StmtKind::Let { init_scope, initializer: None, else_block: Some(_), .. } => {
                    span_bug!(
                        init_scope.span(this.tcx, this.region_scope_tree),
                        "initializer is missing, but else block is present in this let binding",
                    )
                }
                StmtKind::Let {
                    remainder_scope,
                    init_scope,
                    ref pattern,
                    initializer,
                    lint_level,
                    else_block: None,
                    span: _,
                } => {
                    let ignores_expr_result = matches!(pattern.kind, PatKind::Wild);
                    this.block_context.push(BlockFrame::Statement { ignores_expr_result });

                    // Enter the remainder scope, i.e., the bindings' destruction scope.
                    this.push_scope((*remainder_scope, source_info));
                    let_scope_stack.push(remainder_scope);

                    // Declare the bindings, which may create a source scope.
                    let remainder_span = remainder_scope.span(this.tcx, this.region_scope_tree);

                    let visibility_scope =
                        Some(this.new_source_scope(remainder_span, LintLevel::Inherited));

                    // Evaluate the initializer, if present.
                    if let Some(init) = *initializer {
                        let initializer_span = this.thir[init].span;
                        let scope = (*init_scope, source_info);

                        block = this
                            .in_scope(scope, *lint_level, |this| {
                                this.declare_bindings(
                                    visibility_scope,
                                    remainder_span,
                                    pattern,
                                    None,
                                    Some((None, initializer_span)),
                                );
                                this.expr_into_pattern(block, &pattern, init)
                                // irrefutable pattern
                            })
                            .into_block();
                    } else {
                        let scope = (*init_scope, source_info);
                        let _: BlockAnd<()> = this.in_scope(scope, *lint_level, |this| {
                            this.declare_bindings(
                                visibility_scope,
                                remainder_span,
                                pattern,
                                None,
                                None,
                            );
                            block.unit()
                        });

                        debug!("ast_block_stmts: pattern={:?}", pattern);
                        this.visit_primary_bindings(
                            pattern,
                            UserTypeProjections::none(),
                            &mut |this, _, _, node, span, _, _| {
                                this.storage_live_binding(
                                    block,
                                    node,
                                    span,
                                    OutsideGuard,
                                    ScheduleDrops::Yes,
                                );
                                this.schedule_drop_for_binding(node, span, OutsideGuard);
                            },
                        )
                    }

                    // Enter the visibility scope, after evaluating the initializer.
                    if let Some(source_scope) = visibility_scope {
                        this.source_scope = source_scope;
                    }
                    last_remainder_scope = *remainder_scope;
                }
                StmtKind::BindContext { remainder_scope, init_scope, bundle, span, self_id: _ } => {
                    assert!(this.ctx_restrictions.is_none());

                    this.block_context.push(BlockFrame::Statement { ignores_expr_result: false });

                    // Enter the remainder scope, i.e., the bindings' destruction scope.
                    this.push_scope((*remainder_scope, source_info));
                    let_scope_stack.push(remainder_scope);

                    // Declare the bindings, which may create a source scope.
                    let remainder_span = remainder_scope.span(this.tcx, this.region_scope_tree);
                    let visibility_scope =
                        Some(this.new_source_scope(remainder_span, LintLevel::Inherited));

                    // Evaluate the initializer, if present.
                    let bundle_span = this.thir[*bundle].span;
                    let scope = (*init_scope, source_info);

                    block = this
                        .in_scope(scope, LintLevel::Inherited, |this| {
                            let mut block = block;

                            let tcx = this.tcx;
                            let bundle_ty = this.thir.exprs[*bundle].ty;
                            let bundle_reified = tcx.reified_bundle((bundle_ty, MirBuilding));

                            // Lower bundle expression
                            let bundle_place = unpack!(block = this.as_place(block, *bundle));

                            // Reborrow the fields that we need
                            let borrowed_fields = bundle_reified
                                .fields
                                .iter()
                                .map(|(did, fields)| {
                                    let field = &fields[0];

                                    let reborrow = this.local_decls.push(LocalDecl::new(
                                        tcx.erase_regions(field.location.ty()),
                                        bundle_span,
                                    ));

                                    this.cfg.push_assign(
                                        block,
                                        source_info,
                                        Place::from(reborrow),
                                        Rvalue::Ref(
                                            tcx.lifetimes.re_erased,
                                            match field.mutability {
                                                Mutability::Not => BorrowKind::Shared,
                                                Mutability::Mut => BorrowKind::Mut {
                                                    kind: MutBorrowKind::Default,
                                                }
                                            },
                                            field.location.project_place(
                                                tcx,
                                                bundle_place,
                                                [PlaceElem::Deref],
                                            ),
                                        ),
                                    );

                                    (*did, reborrow)
                                })
                                .collect::<Vec<_>>();

                            // Limit lifetimes
                            let lt_limiter = this.new_lt_limiter_static(block, source_info);
                            let relate_refs = [lt_limiter].into_iter()
                                .chain(borrowed_fields
                                    .iter()
                                    .map(|&(_did, local)| Place::from(local))
                                )
                                .collect::<Vec<_>>();

                            let relate_csts = (1..relate_refs.len())
                                .map(|binder| (binder, 0))
                                .collect::<Vec<_>>();

                            this.relate_lifetimes(block, source_info, &relate_refs, &relate_csts);

                            // Assign context
                            let deref_proj = this.tcx.mk_place_elems(&[PlaceElem::Deref]);
                            for &(did, local) in &borrowed_fields {
                                let did_ty = this.tcx.erase_regions(this.tcx.context_ptr_ty(did));

                                // Save old value
                                let old_ptr = this.local_decls.push(LocalDecl::new(
                                    did_ty,
                                    bundle_span,
                                ));

                                this.cfg.push_assign(
                                    block,
                                    source_info,
                                    Place::from(old_ptr),
                                    Rvalue::ContextRef(did),
                                );

                                this.schedule_drop(
                                    *span,
                                    *remainder_scope,
                                    old_ptr,
                                    // TODO: Use more precise modes
                                    DropKind::RestoreBind(
                                        did,
                                        AssignContextKind::MaybeClobber,
                                    ),
                                );

                                // Assign to new value
                                let new_ptr = this.local_decls.push(LocalDecl::new(
                                    did_ty,
                                    bundle_span,
                                ));

                                let new_ptr_imm = this.local_decls.push(LocalDecl::new(
                                    tcx.erase_regions(
                                        this.tcx.context_ptr_ty_with_muta(did, Mutability::Not),
                                    ),
                                    bundle_span,
                                ));

                                this.cfg.push_assign(
                                    block,
                                    source_info,
                                    Place::from(new_ptr_imm),
                                    Rvalue::RawPtr(
                                        Mutability::Not,
                                        Place {
                                            local,
                                            projection: deref_proj,
                                        },
                                    ),
                                );

                                this.cfg.push_assign(
                                    block,
                                    source_info,
                                    Place::from(new_ptr),
                                    Rvalue::Cast(
                                        CastKind::PtrToPtr,
                                        Operand::Copy(new_ptr_imm.into()),
                                        did_ty,
                                    ),
                                );

                                this.cfg.push(block, Statement {
                                    source_info,
                                    kind: StatementKind::AssignContext(Box::new((
                                        did,
                                        Operand::Copy(new_ptr.into()),
                                        AssignContextKind::DoesClobber,
                                    ))),
                                });
                            }

                            // Prepare context
                            this.init_and_borrow_context_binder_locals(
                                block,
                                source_info,
                                ty::ContextBinder::LocalBinder(*stmt),
                                lt_limiter,
                            );

                            block.unit()
                        })
                        .into_block();

                    // Enter the visibility scope, after evaluating the initializer.
                    if let Some(source_scope) = visibility_scope {
                        this.source_scope = source_scope;
                    }
                    last_remainder_scope = *remainder_scope;
                },
            }

            // We bind the context after this statement has been visited to ensure that it isn't
            // visible to expressions in the statement.
            this.ctx_bind_tracker.bind_from_stmt(this.tcx, MirBuilding, &this.thir, &this.thir[*stmt]);

            let popped = this.block_context.pop();
            assert!(popped.is_some_and(|bf| bf.is_statement()));
        }

        // Then, the block may have an optional trailing expression which is a “return” value
        // of the block, which is stored into `destination`.
        let tcx = this.tcx;
        let destination_ty = destination.ty(&this.local_decls, tcx).ty;
        if let Some(expr_id) = expr {
            let expr = &this.thir[expr_id];
            let tail_result_is_ignored =
                destination_ty.is_unit() || this.block_context.currently_ignores_tail_results();
            this.block_context
                .push(BlockFrame::TailExpr { tail_result_is_ignored, span: expr.span });

            block = this.expr_into_dest(destination, block, expr_id).into_block();
            let popped = this.block_context.pop();

            assert!(popped.is_some_and(|bf| bf.is_tail_expr()));
        } else {
            // If a block has no trailing expression, then it is given an implicit return type.
            // This return type is usually `()`, unless the block is diverging, in which case the
            // return type is `!`. For the unit type, we need to actually return the unit, but in
            // the case of `!`, no return value is required, as the block will never return.
            // Opaque types of empty bodies also need this unit assignment, in order to infer that their
            // type is actually unit. Otherwise there will be no defining use found in the MIR.
            if destination_ty.is_unit()
                || matches!(destination_ty.kind(), ty::Alias(ty::Opaque, ..))
            {
                // We only want to assign an implicit `()` as the return value of the block if the
                // block does not diverge. (Otherwise, we may try to assign a unit to a `!`-type.)
                this.cfg.push_assign_unit(block, source_info, destination, this.tcx);
            }
        }
        // Finally, we pop all the let scopes before exiting out from the scope of block
        // itself.
        for scope in let_scope_stack.into_iter().rev() {
            block = this.pop_scope((*scope, source_info), block).into_block();
        }
        // Restore the original source scope.
        this.source_scope = outer_source_scope;
        block.unit()
    }
}
