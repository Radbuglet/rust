use super::{TyCtxt, Ty, list::List};
use rustc_hir::def_id::DefId;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_span::Span;

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
        if ty.ty_adt_def().map(|def| def.did()) == tcx.lang_items().bundle() {
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
