use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_macros::extension;
use rustc_middle::mir::{Body, LocalInfo, Mutability, Place, ProjectionElem};
use rustc_middle::ty::{self, TyCtxt};
use tracing::debug;

use crate::borrow_set::LocalsStateAtExit;

#[extension(pub trait PlaceExt<'tcx>)]
impl<'tcx> Place<'tcx> {
    /// Returns `Some` if the given place points to a local initialized by `Rvalue::ContextRef`.
    /// The first tuple element contains the `DefId` of the context item being borrowed and the
    /// second element contains the mutability.
    fn as_context_borrow(&self, body: &Body<'tcx>) -> Option<(DefId, hir::Mutability)> {
        let local_info = body.local_decls[self.local]
            .local_info
            .as_ref()
            .assert_crate_local();

        let &LocalInfo::ContextRef { def_id, mutability } = &**local_info else {
            return None;
        };

        Some((def_id, mutability))
    }

    /// Returns `true` if we can safely ignore borrows of this place.
    /// This is true whenever there is no action that the user can do
    /// to the place `self` that would invalidate the borrow. This is true
    /// for borrows of raw pointer dereferents as well as shared references.
    fn ignore_borrow(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        locals_state_at_exit: &LocalsStateAtExit,
    ) -> bool {
        // HACK: We do not want to ignore reborrows of `ContextRef` since these reborrows are used
        // as a proxy for borrows of what could be considered a mutable place. See note in
        // `MirBorrowckCtxt::check_context_ref_borrow` for details on this awful hack.
        if self.as_context_borrow(body).is_some() {
            return false;
        }

        // If a local variable is immutable, then we only need to track borrows to guard
        // against two kinds of errors:
        // * The variable being dropped while still borrowed (e.g., because the fn returns
        //   a reference to a local variable)
        // * The variable being moved while still borrowed
        //
        // In particular, the variable cannot be mutated -- the "access checks" will fail --
        // so we don't have to worry about mutation while borrowed.
        if let LocalsStateAtExit::SomeAreInvalidated { has_storage_dead_or_moved } =
            locals_state_at_exit
        {
            let ignore = !has_storage_dead_or_moved.contains(self.local)
                && body.local_decls[self.local].mutability == Mutability::Not;
            debug!("ignore_borrow: local {:?} => {:?}", self.local, ignore);
            if ignore {
                return true;
            }
        }

        for (i, (proj_base, elem)) in self.iter_projections().enumerate() {
            if elem == ProjectionElem::Deref {
                let ty = proj_base.ty(body, tcx).ty;
                match ty.kind() {
                    ty::Ref(_, _, hir::Mutability::Not) if i == 0 => {
                        // For references to thread-local statics, we do need
                        // to track the borrow.
                        if body.local_decls[self.local].is_ref_to_thread_local() {
                            continue;
                        }
                        return true;
                    }
                    ty::RawPtr(..) | ty::Ref(_, _, hir::Mutability::Not) => {
                        // For both derefs of raw pointers and `&T`
                        // references, the original path is `Copy` and
                        // therefore not significant. In particular,
                        // there is nothing the user can do to the
                        // original path that would invalidate the
                        // newly created reference -- and if there
                        // were, then the user could have copied the
                        // original path into a new variable and
                        // borrowed *that* one, leaving the original
                        // path unborrowed.
                        return true;
                    }
                    _ => {}
                }
            }
        }

        false
    }
}
