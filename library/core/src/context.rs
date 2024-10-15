#![allow(missing_docs)]  // TODO
#![cfg(not(bootstrap))]

#[lang = "context_item"]
#[rustc_deny_explicit_impl(implement_via_object = false)]
#[unstable(feature = "context_injection", issue = "none")]
pub trait ContextItem {
    #[unstable(feature = "context_injection", issue = "none")]
    type Item: ?Sized;
}

#[unstable(feature = "context_injection", issue = "none")]
pub trait BundleItem {
    #[unstable(feature = "context_injection", issue = "none")]
    type Item;

    #[unstable(feature = "context_injection", issue = "none")]
    type Context: ContextItem;
}

#[unstable(feature = "context_injection", issue = "none")]
impl<'a, T: ContextItem> BundleItem for &'a T {
    type Item = &'a T::Item;
    type Context = T;
}

#[unstable(feature = "context_injection", issue = "none")]
impl<'a, T: ContextItem> BundleItem for &'a mut T {
    type Item = &'a mut T::Item;
    type Context = T;
}

#[unstable(feature = "context_injection", issue = "none")]
pub trait ContextItemSet {}

#[unstable(feature = "context_injection", issue = "none")]
pub trait BundleItemSet {
    #[unstable(feature = "context_injection", issue = "none")]
    type ItemSet;

    #[unstable(feature = "context_injection", issue = "none")]
    type Context: ContextItemSet;
}

// This blanket impl is safe to write alongside the tuple implementation because no tuple can
// implement `ContextItem`.
#[unstable(feature = "context_injection", issue = "none")]
impl<T: ContextItem> ContextItemSet for T {}

#[unstable(feature = "context_injection", issue = "none")]
impl<T: BundleItem> BundleItemSet for T {
    type ItemSet = T::Item;
    type Context = T::Context;
}

macro_rules! tuple {
    ($($para:ident)*) => {
        #[unstable(feature = "context_injection", issue = "none")]
        impl<$($para: ContextItemSet,)*> ContextItemSet for ($($para,)*) {}

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
