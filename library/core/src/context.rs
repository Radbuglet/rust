#![allow(missing_docs)]  // TODO
#![cfg(not(bootstrap))]

#[lang = "context_item"]
#[rustc_deny_explicit_impl(implement_via_object = false)]
pub trait ContextItem: Sized + 'static {
    type Item: ?Sized + 'static;
}

pub trait BundleItem {
    type Item;
    type Context: ContextItem;
}

impl<'a, T: ContextItem> BundleItem for &'a T {
    type Item = &'a T::Item;
    type Context = T;
}

impl<'a, T: ContextItem> BundleItem for &'a mut T {
    type Item = &'a mut T::Item;
    type Context = T;
}

pub trait ContextItemSet {}

pub trait BundleItemSet {
    type ItemSet;
    type Context: ContextItemSet;
}

// This blanket impl is safe to write alongside the tuple implementation because no tuple can
// implement `ContextItem`.
impl<T: ContextItem> ContextItemSet for T {}

impl<T: BundleItem> BundleItemSet for T {
    type ItemSet = T::Item;
    type Context = T::Context;
}

macro_rules! tuple {
    ($($para:ident)*) => {
        impl<$($para: ContextItemSet,)*> ContextItemSet for ($($para,)*) {}

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

#[lang = "bundle"]
#[derive(Debug)]
pub struct Bundle<T: BundleItemSet>(T::ItemSet);

impl<T: BundleItemSet> Bundle<T> {
    pub const fn new(items: T::ItemSet) -> Self {
        Self(items)
    }
}

mod make_single_item_bundle {
    use super::*;

    #[lang = "single_item_bundle_ctor"]
    const fn make_single_item_bundle<Anno, Ref>(val: Ref) -> Bundle<Ref::BundleItem>
    where
        Anno: ContextItem,
        Ref: BundleRef<Anno>,
    {
        Bundle::new(val)
    }

    trait BundleRef<Ctx: ContextItem> {
        type BundleItem: BundleItem<Item = Self>;
    }

    impl<'a, Ctx: ContextItem> BundleRef<Ctx> for &'a Ctx::Item {
        type BundleItem = &'a Ctx;
    }

    impl<'a, Ctx: ContextItem> BundleRef<Ctx> for &'a mut Ctx::Item {
        type BundleItem = &'a mut Ctx;
    }
}
