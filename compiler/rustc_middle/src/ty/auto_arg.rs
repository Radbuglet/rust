use super::{TyCtxt, Ty, list::List};
use rustc_hir::def_id::DefId;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_span::Span;

use std::fmt::Write as _;

#[derive(Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct AutoArg<'tcx> {
    /// The value being produced.
    pub ty: Ty<'tcx>,

    /// How that value will be produced.
    pub kind: AutoArgKind,

    /// Diagnostic information about the call-site for which this argument was generated.
    pub origin: AutoArgOrigin<'tcx>,
}

impl<'tcx> AutoArg<'tcx> {
    pub fn of(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        origin: AutoArgOrigin<'tcx>
    ) -> Option<AutoArg<'tcx>> {
        AutoArgKind::of(tcx, ty).map(|kind| AutoArg {
            kind,
            ty,
            origin,
        })
    }
}

#[derive(Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub enum AutoArgKind {
    PackBundle,
}

impl AutoArgKind {
    pub fn of<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<AutoArgKind> {
        if ty.opt_bundle_item_set(tcx).is_some() {
            return Some(AutoArgKind::PackBundle);
        }

        None
    }
}

#[derive(Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct AutoArgOrigin<'tcx> {
    /// The function we're calling.
    pub fn_def_id: Option<DefId>,

    /// The arguments of that function's signature.
    pub fn_args: &'tcx List<Ty<'tcx>>,

    /// The span of the function call followed by the spans of all the explicitly provided
    /// arguments.
    pub spans: &'tcx [Span],

    /// The index of the argument that was synthesized.
    pub arg_idx: u32,

    /// Whether this auto-arg was produced by an overloaded operation.
    pub overloaded: bool,
}

impl<'tcx> AutoArgOrigin<'tcx> {
    pub fn intro_span(&self) -> Span {
        self.spans[0]
    }

    pub fn arg_spans(&self) -> &'tcx [Span] {
        &self.spans[1..]
    }

    pub fn suggest_intro(&self, tcx: TyCtxt<'tcx>) -> Option<String> {
        let mut text = String::new();

        if self.overloaded {
            todo!();
        }

        self.suggest_intro_args(tcx, &mut text)?;

        Some(text)
    }

    fn suggest_intro_args(&self, tcx: TyCtxt<'tcx>, text: &mut String) -> Option<()> {
        let source_map = tcx.sess.source_map();

        text.push('(');

        // Print out the provided arguments
        let mut first_arg = true;

        for &arg in self.arg_spans() {
            if !first_arg {
                text.push_str(", ");
            }
            first_arg = false;

            let arg = normalize_span_for_arg(self.intro_span(), arg);
            text.push_str(&source_map.span_to_snippet(arg).ok()?);
        }

        // Print out the placeholders
        for i in self.arg_spans().len()..=(self.arg_idx as usize) {
            if !first_arg {
                text.push_str(", ");
            }
            first_arg = false;

            let arg_ty = self.fn_args[i];
            write!(text, "/* {arg_ty} */").unwrap();
        }

        // Print closing parenthesis.
        text.push(')');

        Some(())
    }
}

pub fn strip_span_to_open_paren(tcx: TyCtxt<'_>, span: Span) -> Span {
    tcx.sess.source_map().span_until_char(span, '(').shrink_to_hi().to(span.shrink_to_hi())
}

// Copied from `report_arg_errors` in `hir_typeck`, which has to do something similar to us.
fn normalize_span_for_arg(error_span: Span, span: Span) -> Span {
    let normalized_span = span.find_ancestor_inside_same_ctxt(error_span).unwrap_or(span);
    // Sometimes macros mess up the spans, so do not normalize the
    // arg span to equal the error span, because that's less useful
    // than pointing out the arg expr in the wrong context.
    if normalized_span.source_equal(error_span) { span } else { normalized_span }
}
