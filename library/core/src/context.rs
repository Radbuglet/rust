#![allow(missing_docs)]  // TODO

#[cfg(not(bootstrap))]
#[lang = "context_marker"]
#[rustc_deny_explicit_impl(implement_via_object = false)]
#[unstable(feature = "context_injection", issue = "none")]
pub trait ContextMarker {
    #[unstable(feature = "context_injection", issue = "none")]
    type Item: ?Sized;
}
