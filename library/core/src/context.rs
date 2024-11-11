#![allow(missing_docs)]  // TODO

macro_rules! tuple {
    ($mac:path) => {
        tuple!(@inner $mac, A B C D E F G H I J K L);
    };
    (@inner $mac:path, $($first:ident $($rest:ident)*)?) => {
        $mac!($($first $($rest)*)?);
        $(tuple!(@inner $mac, $($rest)*);)?
    };
}

// === ContextItem(Set) === //

#[lang = "context_item"]
#[rustc_deny_explicit_impl(implement_via_object = false)]
pub trait ContextItem: Sized + 'static {
    type Item: ?Sized + 'static;
}

pub trait ContextItemSet {}

// This blanket impl is safe to write alongside the tuple implementations because no tuple can
// implement `ContextItem`.
impl<T: ContextItem> ContextItemSet for T {}

macro_rules! impl_context_item_set {
    ($($para:ident)*) => {
        impl<$($para: ContextItemSet,)*> ContextItemSet for ($($para,)*) {}
    }
}

tuple!(impl_context_item_set);

// === BundleItemSet === //

#[lang = "infer_bundle"]
// TODO: `#[rustc_deny_explicit_impl(implement_via_object = false)]`, but make it work with negative impls.
pub trait InferBundle: Sized {}

impl<T: ?Sized> !InferBundle for &'_ T {}

impl<T: ?Sized> !InferBundle for &'_ mut T {}

pub trait BundleItemSet {
    #[lang = "bundle_item_set_values"]
    type Values;
}

pub trait KnownBundleItemSet: BundleItemSet {
    type Contexts: ContextItemSet;
}

impl<T: InferBundle> BundleItemSet for T {
    type Values = T;
}

// This works because `InferBundle` has a negative impl for `&'_ T`.
impl<'a, T: ContextItem> BundleItemSet for &'a T {
    type Values = &'a T::Item;
}

impl<'a, T: ContextItem> KnownBundleItemSet for &'a T {
    type Contexts = T;
}

// This works because `InferBundle` has a negative impl for `&'_ mut T`.
impl<'a, T: ContextItem> BundleItemSet for &'a mut T {
    type Values = &'a mut T::Item;
}

impl<'a, T: ContextItem> KnownBundleItemSet for &'a mut T {
    type Contexts = T;
}

macro_rules! impl_bundle_items {
    ($($para:ident)*) => {
        impl<$($para: BundleItemSet,)*> BundleItemSet for ($($para,)*) {
            type Values = ($($para::Values,)*);
        }

        impl<$($para: KnownBundleItemSet,)*> KnownBundleItemSet for ($($para,)*) {
            type Contexts = ($($para::Contexts,)*);
        }
    }
}

tuple!(impl_bundle_items);

// === Bundle === //

#[lang = "bundle"]
#[derive(Debug)]
pub struct Bundle<T: BundleItemSet>(T::Values);

impl<T: BundleItemSet> Bundle<T> {
    pub const fn new(items: T::Values) -> Self {
        Self(items)
    }

    pub const fn unwrap(self) -> T::Values {
        self.0
    }
}

mod make_single_item_bundle {
    use super::*;

    #[lang = "single_item_bundle_ctor"]
    pub const fn make_single_item_bundle<Anno, Ref>(val: Ref) -> Bundle<Ref::BundleItem>
    where
        Anno: ContextItem,
        Ref: BundleRef<Anno>,
    {
        Bundle::new(val)
    }

    pub trait BundleRef<Ctx: ContextItem> {
        type BundleItem: BundleItemSet<Values = Self>;
    }

    impl<'a, Ctx: ContextItem> BundleRef<Ctx> for &'a Ctx::Item {
        type BundleItem = &'a Ctx;
    }

    impl<'a, Ctx: ContextItem> BundleRef<Ctx> for &'a mut Ctx::Item {
        type BundleItem = &'a mut Ctx;
    }
}

#[allow_internal_unstable(builtin_syntax)]
pub macro pack {
    (@env $(,$src:expr)* $(,)? $(=> $ty:ty)?) => {
        {builtin # pack(allow_env $($src),* $(=> $ty)?) }
    },
    ($($src:expr),+$(,)? $(=> $ty:ty)?) => {
        {builtin # pack(deny_env $($src),* $(=> $ty)?) }
    },
}

pub macro unpack {
    (@env $(,$src:expr)* $(,)? => $ty:ty) => {
        pack!(@env $(,$src)* => Bundle<$ty>).unwrap()
    },
    ($($src:expr),+$(,)? => $ty:ty) => {
        pack!($($src),* => Bundle<$ty>).unwrap()
    },
}

#[allow_internal_unstable(builtin_syntax)]
pub macro infer_bundle {
    ($lt:lifetime) => { builtin # infer_bundle($lt) },
}
