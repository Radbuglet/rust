use super::{TyCtxt, Ty};
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};

#[derive(Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct AutoArg<'tcx> {
    pub kind: AutoArgKind,
    pub ty: Ty<'tcx>,
}

#[derive(Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub enum AutoArgKind {
    PackBundle,
}

impl<'tcx> AutoArg<'tcx> {
    pub fn of(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<AutoArg<'tcx>> {
        AutoArgKind::of(tcx, ty).map(|kind| AutoArg {
            kind,
            ty,
        })
    }
}

impl AutoArgKind {
    pub fn of<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<AutoArgKind> {
        if ty.ty_adt_def().map(|def| def.did()) == tcx.lang_items().bundle() {
            return Some(AutoArgKind::PackBundle);
        }

        None
    }
}
