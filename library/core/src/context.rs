#![allow(missing_docs)]  // TODO
#![cfg(not(bootstrap))]

#[lang = "context_marker"]
#[rustc_deny_explicit_impl(implement_via_object = false)]
#[unstable(feature = "context_injection", issue = "none")]
pub trait ContextMarker {
    #[unstable(feature = "context_injection", issue = "none")]
    type Item: ?Sized;
}

#[unstable(feature = "context_injection", issue = "none")]
pub trait BundleItem {
    #[unstable(feature = "context_injection", issue = "none")]
    type Item;

    #[unstable(feature = "context_injection", issue = "none")]
    type Context: ContextMarker;
}

#[unstable(feature = "context_injection", issue = "none")]
impl<'a, T: ContextMarker> BundleItem for &'a T {
    type Item = &'a T::Item;
    type Context = T;
}

#[unstable(feature = "context_injection", issue = "none")]
impl<'a, T: ContextMarker> BundleItem for &'a mut T {
    type Item = &'a mut T::Item;
    type Context = T;
}

#[unstable(feature = "context_injection", issue = "none")]
pub trait ContextMarkerSet {}

#[unstable(feature = "context_injection", issue = "none")]
pub trait BundleItemSet {
    #[unstable(feature = "context_injection", issue = "none")]
    type ItemSet;

    #[unstable(feature = "context_injection", issue = "none")]
    type Context: ContextMarkerSet;
}

// This blanket impl is safe to write alongside the tuple implementation because no tuple can
// implement `ContextMarker`.
#[unstable(feature = "context_injection", issue = "none")]
impl<T: ContextMarker> ContextMarkerSet for T {}

#[unstable(feature = "context_injection", issue = "none")]
impl<T: BundleItem> BundleItemSet for T {
    type ItemSet = T::Item;
    type Context = T::Context;
}

macro_rules! tuple {
    ($($para:ident)*) => {
        #[unstable(feature = "context_injection", issue = "none")]
        impl<$($para: ContextMarkerSet,)*> ContextMarkerSet for ($($para,)*) {}

        #[unstable(feature = "context_injection", issue = "none")]
        impl<$($para: BundleItemSet,)*> BundleItemSet for ($($para,)*) {
            type ItemSet = ($($para::ItemSet,)*);
            type Context = ($($para::Context,)*);
        }

        tuple!(@peel $($para)*);
    };
    (@peel) => {};
    (@peel $first:ident $($rest:ident)*) => {
        tuple!($($rest)*);
    };
}

tuple!(A B C D E F G H I J K L);

#[derive(Debug)]
#[unstable(feature = "context_injection", issue = "none")]
pub struct Bundle<T: BundleItemSet>(T::ItemSet);
