use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_middle::ty::{self, Ty, Region, RegionKind};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;

use crate::{lints, LateContext, LateLintPass, LintContext};

declare_lint! {
    /// TODO: Document
    pub UNNECESSARY_BUNDLE_ITEM,
    Warn,
    "context item could be passed by implicit passing rather than by bundle",
}

declare_lint_pass!(
    UnnecessaryBundleItem => [UNNECESSARY_BUNDLE_ITEM]
);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryBundleItem {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        _fn_kind: FnKind<'_>,
        decl: &'tcx hir::FnDecl<'_>,
        _body: &'tcx hir::Body<'_>,
        _span: Span,
        def_id: LocalDefId,
    ) {
        let tcx = cx.tcx;

        // Skip over closures and const blocks of proper function definitions since they can
        // never inherit context from their caller anyways.
        let typeck_root_def_id = tcx.typeck_root_def_id(def_id.to_def_id()).expect_local();
        if typeck_root_def_id != def_id {
            assert!(ty::def_can_borrow_context(tcx, def_id.to_def_id()).is_err());
            return;
        }

        // If this function cannot inherit context from its caller, the bundle cannot be redundant.
        if ty::def_can_borrow_context(tcx, def_id.to_def_id()).is_err() {
            return;
        }

        // Otherwise, let's look for redundant context items!

        // `fn_sig` gives us a signature where all late-bound lifetimes are bound lifetimes
        // represented using Debrujin indices. We need to track which lifetime parameters are
        // tainted in the `tainted_lifetimes` set and Debrujin indices are less-than-ideal for that
        // sort of work since a given parameter could be mentioned in many ways. Hence, we normalize
        // late-bound regions to the `ReLateBound` variant using `liberate_late_bound_regions` and
        // differentiate late-bound regions in the set by their `LateParamRegion`.
        //
        // This also means that we no longer have to deal with `Binder`s, which is nice.
        let sig = tcx.liberate_late_bound_regions(
            def_id.to_def_id(),
            tcx.fn_sig(def_id).instantiate_identity(),
        );

        let mut tainted_lifetimes = Option::<FxHashSet<ty::LateParamRegion>>::None;

        for (i, arg) in sig.inputs().iter().enumerate() {
            // We only care about arguments we know, for sure, to be bundles. "Surprise" bundles
            // from generic parameters are probably necessary.
            let Some(arg) = arg.opt_bundle_item_set(tcx) else {
                continue;
            };

            // Determine the set of concrete context items which appear in the bundle.
            let concrete = {
                let mut visitor = ConcreteItemVisitor::default();
                visitor.visit(arg);
                visitor.concrete
            };

            // Concrete context items may or may not be unnecessary depending on whether their
            // lifetime is shared with non-bundle types. For example, this is unnecessary:
            //
            // ```
            // #[context]
            // static FOO: u32;
            //
            // #[context]
            // static BAR: u32;
            //
            // fn demo<'a>(cx: Bundle<(&'a FOO, &'a BAR)>);
            // ```
            //
            // ...but this isn't:
            //
            // ```
            // fn demo_1<'a>(cx: Bundle<(&'a FOO, &'a BAR)>) -> &'a u32;
            //
            // fn demo_2<'a>(cx: Bundle<(&'a FOO, &'a BAR)>, f: OutputHelper<'a>);
            // ```
            //
            // Technically, we're searching for lifetimes which have the potential of being leaked
            // by the function—that is, lifetimes which appear in non-bundle positions and lifetimes
            // that must outlive those—but computing this precisely would require an analysis
            // similar to that performed in `rustc_borrowck`'s `free_region_relations` solver, which
            // is quite a bit of work for a simple lint.
            //
            // Instead, we employ a conservative solution which assumes that...
            //
            // 1) Types in `Bundle`'s cannot introduce additional region outlives constraints.
            // 2) In order for a type to introduce an implicit outlives constraint, it must mention
            //    the region in the type.
            //
            // These let us taint regions which either...
            //
            // - Are mentioned outside of a bundle which is root-most in the argument.
            //   The root-level requirement means that `arg: Foo<Bundle<&'a u32>>` will cause `'a`
            //   to be tainted under the assumption that `Foo` could potentially introduce
            //   constraints on `'a`.
            //
            // - Mentioned in the (inherited) predicates of the function.
            //
            // The second requirement can be easily satisfied by considering all early-bound
            // parameters automatically tainted. Recall that a region is late-bound if it a) is
            // constrained by an argument type and b) does not appear in a where clause. Hence, if
            // we find an early bound region, we know that it either is not constrained by an
            // argument type—in which case the lint won't trigger on it since we only check bundles
            // in the input position—or is mentioned in a where clause. Hence, all early regions are
            // always tainted.
            let tainted_lifetimes = tainted_lifetimes.get_or_insert_with(|| {
                let mut tainted_lifetimes = FxHashSet::default();

                for ty in sig.inputs_and_output {
                    if ty.opt_bundle_item_set(tcx).is_some() {
                        continue;
                    }

                    tcx.for_each_free_region(&ty, |re| match re.kind() {
                        RegionKind::ReLateParam(late) => {
                            tainted_lifetimes.insert(late);
                        }
                        RegionKind::ReEarlyParam(..) | RegionKind::ReStatic => {
                            // (these are already implicitly tainted)
                        }
                        RegionKind::ReBound(..)
                        | RegionKind::ReVar(..)
                        | RegionKind::RePlaceholder(..)
                        | RegionKind::ReErased => unreachable!(),
                        RegionKind::ReError(..) => {
                            // (don't emit lints for code with errors)
                        }
                    });
                }

                tainted_lifetimes
            });

            let offending = concrete.into_iter()
                .filter(|(re, _ty)| match re.kind() {
                    RegionKind::ReLateParam(late) => {
                        !tainted_lifetimes.contains(&late)
                    }
                    RegionKind::ReEarlyParam(..) | RegionKind::ReStatic => {
                        // (definitely tainted, see comment above)
                        false
                    }
                    RegionKind::ReBound(..)
                    | RegionKind::ReVar(..)
                    | RegionKind::RePlaceholder(..)
                    | RegionKind::ReErased => unreachable!(),
                    RegionKind::ReError(..) => {
                        // Don't emit lints for code with errors.
                        false
                    }
                })
                .map(|(_re, ty)| ty)
                .collect::<Vec<_>>();

            // Emit the lint!
            for ty in offending {
                cx.emit_span_lint(
                    UNNECESSARY_BUNDLE_ITEM,
                    decl.inputs[i].span,
                    lints::UnnecessaryBundleItem { ty },
                );
            }
        }
    }
}


#[derive(Default)]
struct ConcreteItemVisitor<'tcx> {
    concrete: Vec<(Region<'tcx>, Ty<'tcx>)>,
}

impl<'tcx> ConcreteItemVisitor<'tcx> {
    fn visit(&mut self, ty: Ty<'tcx>) {
        match ty::ReifiedBundleItemSet::decode(ty) {
            ty::ReifiedBundleItemSet::Ref(re, _muta, _did) => {
                self.concrete.push((re, ty));
            }
            ty::ReifiedBundleItemSet::Tuple(list) => {
                for ty in list {
                    self.visit(ty);
                }
            }
            ty::ReifiedBundleItemSet::GenericRef(..)
            | ty::ReifiedBundleItemSet::GenericSet(..)
            | ty::ReifiedBundleItemSet::InferSet(..) => {
                // (necessary)
            }
            ty::ReifiedBundleItemSet::Error(..) => {
                // (cannot lint)
            }
        }
    }
}
