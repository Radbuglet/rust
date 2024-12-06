use std::fmt;
use std::path::PathBuf;

use rustc_errors::codes::*;
use rustc_errors::{DiagArgName, DiagArgValue, DiagMessage};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

use crate::ty::Ty;

#[derive(Diagnostic)]
#[diag(middle_drop_check_overflow, code = E0320)]
#[note]
pub struct DropCheckOverflow<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub overflow_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(middle_opaque_hidden_type_mismatch)]
pub struct OpaqueHiddenTypeMismatch<'tcx> {
    pub self_ty: Ty<'tcx>,
    pub other_ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub other_span: Span,
    #[subdiagnostic]
    pub sub: TypeMismatchReason,
}

#[derive(Subdiagnostic)]
pub enum TypeMismatchReason {
    #[label(middle_conflict_types)]
    ConflictType {
        #[primary_span]
        span: Span,
    },
    #[note(middle_previous_use_here)]
    PreviousUse {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(middle_limit_invalid)]
pub struct LimitInvalid<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub value_span: Span,
    pub error_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(middle_recursion_limit_reached)]
#[help]
pub struct RecursionLimitReached<'tcx> {
    pub ty: Ty<'tcx>,
    pub suggested_limit: rustc_session::Limit,
}

#[derive(Diagnostic)]
#[diag(middle_const_eval_non_int)]
pub struct ConstEvalNonIntError {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(middle_strict_coherence_needs_negative_coherence)]
pub(crate) struct StrictCoherenceNeedsNegativeCoherence {
    #[primary_span]
    pub span: Span,
    #[label]
    pub attr_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(middle_requires_lang_item)]
pub(crate) struct RequiresLangItem {
    #[primary_span]
    pub span: Option<Span>,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(middle_const_not_used_in_type_alias)]
pub(super) struct ConstNotUsedTraitAlias {
    pub ct: String,
    #[primary_span]
    pub span: Span,
}

pub struct CustomSubdiagnostic<'a> {
    pub msg: fn() -> DiagMessage,
    pub add_args: Box<dyn FnOnce(&mut dyn FnMut(DiagArgName, DiagArgValue)) + 'a>,
}

impl<'a> CustomSubdiagnostic<'a> {
    pub fn label(x: fn() -> DiagMessage) -> Self {
        Self::label_and_then(x, |_| {})
    }
    pub fn label_and_then<F: FnOnce(&mut dyn FnMut(DiagArgName, DiagArgValue)) + 'a>(
        msg: fn() -> DiagMessage,
        f: F,
    ) -> Self {
        Self { msg, add_args: Box::new(move |x| f(x)) }
    }
}

impl fmt::Debug for CustomSubdiagnostic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomSubdiagnostic").finish_non_exhaustive()
    }
}

#[derive(Diagnostic)]
pub enum LayoutError<'tcx> {
    #[diag(middle_unknown_layout)]
    Unknown { ty: Ty<'tcx> },

    #[diag(middle_values_too_big)]
    Overflow { ty: Ty<'tcx> },

    #[diag(middle_cannot_be_normalized)]
    NormalizationFailure { ty: Ty<'tcx>, failure_ty: String },

    #[diag(middle_cycle)]
    Cycle,

    #[diag(middle_layout_references_error)]
    ReferencesError,
}

#[derive(Diagnostic)]
#[diag(middle_adjust_for_foreign_abi_error)]
pub struct UnsupportedFnAbi {
    pub arch: Symbol,
    pub abi: &'static str,
}

#[derive(Diagnostic)]
#[diag(middle_erroneous_constant)]
pub struct ErroneousConstant {
    #[primary_span]
    pub span: Span,
}

/// Used by `rustc_const_eval`
pub use crate::fluent_generated::middle_adjust_for_foreign_abi_error;

#[derive(Diagnostic)]
#[diag(middle_type_length_limit)]
#[help(middle_consider_type_length_limit)]
pub struct TypeLengthLimit {
    #[primary_span]
    pub span: Span,
    pub shrunk: String,
    #[note(middle_written_to_path)]
    pub was_written: bool,
    pub path: PathBuf,
    pub type_length: usize,
}

#[derive(Diagnostic)]
#[diag(middle_reified_fn_using_ctx)]
pub(crate) struct ReifiedFnUsingCtx {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(middle_ambiguous_origin_for_context_item)]
pub(crate) struct AmbiguousOriginForContextItem<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ctx_ty: Ty<'tcx>,
    pub(crate) bundle_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(middle_ambiguous_origin_for_generic_item)]
pub(crate) struct AmbiguousOriginForGenericItem<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ctx_ty: Ty<'tcx>,
    pub(crate) bundle_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[note(middle_dependency_originates_from_infer_bundle_hint)]
pub(crate) struct DependencyOriginatesFromInferBundleHint<'tcx> {
    pub(crate) opaque_ty: Ty<'tcx>,
    pub(crate) concrete_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(middle_missing_context_item)]
pub(crate) struct MissingContextItem<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) missing_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[suggestion(middle_missing_context_add_env_suggestion, code = "@env, ")]
pub(crate) struct MissingContextAddEnvSuggestion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[note(middle_missing_item_lhs_type_hint)]
pub(crate) struct MissingItemLhsTypeHint<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) index: usize,
    pub(crate) expr_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(middle_missing_generic_item)]
pub(crate) struct MissingGenericItem<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) missing_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[note(middle_env_cannot_provide_generic)]
pub(crate) struct EnvCannotProvideGeneric {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[note(middle_originates_from_auto_arg_def)]
pub(crate) struct OriginatesFromAutoArgDef {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) arg_name: Symbol,
}

#[derive(Subdiagnostic)]
#[note(middle_originates_from_auto_arg_anon)]
pub(crate) struct OriginatesFromAutoArgAnon<'tcx> {
    pub(crate) arg_num: u32,
    pub(crate) callee_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[suggestion(middle_auto_arg_should_be_explicit, code = "{replacement}", style = "verbose", applicability = "has-placeholders")]
pub(crate) struct AutoArgShouldBeExplicit {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) replacement: String,
}

#[derive(Diagnostic)]
#[diag(middle_ambiguous_early_pack_resolution)]
pub(crate) struct AmbiguousEarlyPackResolution<'tcx> {
    #[primary_span]
    #[label]
    pub(crate) infer_span: Span,

    #[label(middle_ambiguous_early_pack_resolution_env_label)]
    pub(crate) env_span: Span,

    pub(crate) req_ty: Ty<'tcx>,
    pub(crate) infer_ty: Ty<'tcx>,

    #[subdiagnostic]
    pub(crate) note: AmbiguousEarlyPackResolutionNote,
}

#[derive(Subdiagnostic)]
#[note(middle_ambiguous_early_pack_resolution_note)]
pub(crate) struct AmbiguousEarlyPackResolutionNote {}

pub(crate) use crate::fluent_generated::{
    middle_entry_fn_uses_ctx,
    middle_extern_fn_uses_ctx,
    middle_async_fn_uses_ctx,
    middle_closure_uses_ctx,
    middle_trait_member_uses_ctx,
};
