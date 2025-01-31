use rustc_ast::{BindContextKind, Block, BlockCheckMode, Local, LocalKind, Stmt, StmtKind};
use rustc_hir as hir;
use smallvec::SmallVec;

use crate::{ImplTraitContext, ImplTraitPosition, LoweringContext};

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    pub(super) fn lower_block(
        &mut self,
        b: &Block,
        targeted_by_break: bool,
    ) -> &'hir hir::Block<'hir> {
        self.arena.alloc(self.lower_block_noalloc(b, targeted_by_break))
    }

    pub(super) fn lower_block_noalloc(
        &mut self,
        b: &Block,
        targeted_by_break: bool,
    ) -> hir::Block<'hir> {
        let (stmts, expr) = self.lower_stmts(&b.stmts);
        let rules = self.lower_block_check_mode(&b.rules);
        let hir_id = self.lower_node_id(b.id);
        hir::Block { hir_id, stmts, expr, rules, span: self.lower_span(b.span), targeted_by_break }
    }

    fn lower_stmts(
        &mut self,
        mut ast_stmts: &[Stmt],
    ) -> (&'hir [hir::Stmt<'hir>], Option<&'hir hir::Expr<'hir>>) {
        let mut stmts = SmallVec::<[hir::Stmt<'hir>; 8]>::new();
        let mut expr = None;
        while let [s, tail @ ..] = ast_stmts {
            match &s.kind {
                StmtKind::Let(local) => {
                    let hir_id = self.lower_node_id(s.id);
                    let local = self.lower_local(local);
                    self.alias_attrs(hir_id, local.hir_id);
                    let kind = hir::StmtKind::Let(local);
                    let span = self.lower_span(s.span);
                    stmts.push(hir::Stmt { hir_id, kind, span });
                }
                StmtKind::BindContext(bind) => {
                    let hir_id = self.lower_node_id(bind.id);

                    let bundle = match &bind.kind {
                        BindContextKind::Single(ty, expr) => {
                            let ty = self.lower_ty(
                                ty,
                                ImplTraitContext::Disallowed(ImplTraitPosition::Variable),
                            );
                            let expr = self.lower_expr_mut(expr);
                            let expr_span = self.lower_span(expr.span);

                            let infer_hir_id = self.next_id();
                            let bundle_ctor = self.make_lang_item_path(
                                hir::LangItem::SingleItemBundleCtor,
                                bind.span,
                                Some(self.arena.alloc(hir::GenericArgs {
                                    args: self.arena.alloc_from_iter([
                                        hir::GenericArg::Type(ty),
                                        hir::GenericArg::Infer(hir::InferArg {
                                            hir_id: infer_hir_id,
                                            span: expr_span,
                                        })
                                    ]),
                                    constraints: self.arena.alloc_from_iter([]),
                                    parenthesized: hir::GenericArgsParentheses::No,
                                    span_ext: expr_span,
                                })),
                            );

                            let bundle_ctor = self.arena.alloc(self.expr(
                                expr_span,
                                hir::ExprKind::Path(hir::QPath::Resolved(None, bundle_ctor)),
                            ));

                            self.arena.alloc(self.expr(expr_span, hir::ExprKind::Call(
                                bundle_ctor,
                                self.arena.alloc_from_iter([expr]),
                            )))
                        }
                        BindContextKind::Bundle(expr) => self.lower_expr(expr),
                    };

                    let span = self.lower_span(bind.span);
                    let kind = hir::StmtKind::BindContext(self.arena.alloc(hir::BindContextStmt {
                        hir_id,
                        bundle,
                        span,
                    }));

                    stmts.push(hir::Stmt { hir_id, kind, span });
                },
                StmtKind::Item(it) => {
                    stmts.extend(self.lower_item_ref(it).into_iter().enumerate().map(
                        |(i, item_id)| {
                            let hir_id = match i {
                                0 => self.lower_node_id(s.id),
                                _ => self.next_id(),
                            };
                            let kind = hir::StmtKind::Item(item_id);
                            let span = self.lower_span(s.span);
                            hir::Stmt { hir_id, kind, span }
                        },
                    ));
                }
                StmtKind::Expr(e) => {
                    let e = self.lower_expr(e);
                    if tail.is_empty() {
                        expr = Some(e);
                    } else {
                        let hir_id = self.lower_node_id(s.id);
                        self.alias_attrs(hir_id, e.hir_id);
                        let kind = hir::StmtKind::Expr(e);
                        let span = self.lower_span(s.span);
                        stmts.push(hir::Stmt { hir_id, kind, span });
                    }
                }
                StmtKind::Semi(e) => {
                    let e = self.lower_expr(e);
                    let hir_id = self.lower_node_id(s.id);
                    self.alias_attrs(hir_id, e.hir_id);
                    let kind = hir::StmtKind::Semi(e);
                    let span = self.lower_span(s.span);
                    stmts.push(hir::Stmt { hir_id, kind, span });
                }
                StmtKind::Empty => {}
                StmtKind::MacCall(..) => panic!("shouldn't exist here"),
            }
            ast_stmts = tail;
        }
        (self.arena.alloc_from_iter(stmts), expr)
    }

    fn lower_local(&mut self, l: &Local) -> &'hir hir::LetStmt<'hir> {
        let ty = l
            .ty
            .as_ref()
            .map(|t| self.lower_ty(t, ImplTraitContext::Disallowed(ImplTraitPosition::Variable)));
        let init = l.kind.init().map(|init| self.lower_expr(init));
        let hir_id = self.lower_node_id(l.id);
        let pat = self.lower_pat(&l.pat);
        let els = if let LocalKind::InitElse(_, els) = &l.kind {
            Some(self.lower_block(els, false))
        } else {
            None
        };
        let span = self.lower_span(l.span);
        let source = hir::LocalSource::Normal;
        self.lower_attrs(hir_id, &l.attrs);
        self.arena.alloc(hir::LetStmt { hir_id, ty, pat, init, els, span, source })
    }

    fn lower_block_check_mode(&mut self, b: &BlockCheckMode) -> hir::BlockCheckMode {
        match *b {
            BlockCheckMode::Default => hir::BlockCheckMode::DefaultBlock,
            BlockCheckMode::Unsafe(u) => {
                hir::BlockCheckMode::UnsafeBlock(self.lower_unsafe_source(u))
            }
        }
    }
}
