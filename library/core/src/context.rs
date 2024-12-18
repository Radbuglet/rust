#![allow(missing_docs)]  // TODO

use crate::{
    any::{TypeId, type_name},
    fmt,
    intrinsics,
    marker::PhantomData,
    mem::MaybeUninit,
    slice,
};

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

#[lang = "infer_bundle_for"]
// TODO: `#[rustc_deny_explicit_impl(implement_via_object = false)]`, but make it work with negative impls.
pub trait InferBundleFor<'a>: InferBundle {}

impl<T: ?Sized> !InferBundleFor<'_> for &'_ T {}

impl<T: ?Sized> !InferBundleFor<'_> for &'_ mut T {}

pub trait BundleItemSet {
    #[lang = "bundle_item_set_values"]
    type Values;
}

pub trait BundleItemSetFor<'a>: BundleItemSet {}

pub trait KnownBundleItemSet: BundleItemSet {
    type Contexts: ContextItemSet;
}

impl<T: InferBundle> BundleItemSet for T {
    type Values = T;
}

impl<'a, T: InferBundleFor<'a>> BundleItemSetFor<'a> for T {}

// This works because `InferBundle` has a negative impl for `&'_ T`.
impl<'a, T: ContextItem> BundleItemSet for &'a T {
    type Values = &'a T::Item;
}

impl<'a, T: ContextItem> BundleItemSetFor<'a> for &'a T {}

impl<'a, T: ContextItem> KnownBundleItemSet for &'a T {
    type Contexts = T;
}

// This works because `InferBundle` has a negative impl for `&'_ mut T`.
impl<'a, T: ContextItem> BundleItemSet for &'a mut T {
    type Values = &'a mut T::Item;
}

impl<'a, T: ContextItem> BundleItemSetFor<'a> for &'a mut T {}

impl<'a, T: ContextItem> KnownBundleItemSet for &'a mut T {
    type Contexts = T;
}

macro_rules! impl_bundle_items {
    ($($para:ident)*) => {
        impl<$($para: BundleItemSet,)*> BundleItemSet for ($($para,)*) {
            type Values = ($($para::Values,)*);
        }

        impl<'a, $($para: BundleItemSetFor<'a>,)*> BundleItemSetFor<'a> for ($($para,)*) {}

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

// === Contextual Operations === //

#[lang = "deref_cx"]
#[doc(alias = "*")]
#[doc(alias = "&*")]
pub trait DerefCx<'i, 'o> {
    #[lang = "deref_cx_context_ref"]
    type ContextRef: BundleItemSet;

    #[lang = "deref_cx_target"]
    type TargetCx: ?Sized;

    #[must_use]
    fn deref_cx(&'i self, cx: Bundle<Self::ContextRef>) -> &'o Self::TargetCx;
}

#[lang = "deref_cx_mut"]
#[doc(alias = "*")]
pub trait DerefCxMut<'i, 'o>: DerefCx<'i, 'o> {
    type ContextMut: BundleItemSet;

    #[must_use]
    fn deref_cx_mut(&'i mut self, cx: Bundle<Self::ContextMut>) -> &'o mut Self::TargetCx;
}

// === Bundle Auto Construction === //

impl<T: BundleItemSet> Bundle<T> {
    pub fn layout() -> &'static [BundleItemLayout] {
        let elems = intrinsics::bundle_layout::<T>();

        unsafe {
            slice::from_raw_parts(elems.as_ptr().cast(), elems.len())
        }
    }

    pub fn try_new_auto<'a, F, E>(mut f: F) -> Result<Self, E>
    where
        F: for<'m> FnMut(BundleItemRequest<'a, 'm>) -> Result<BundleItemResponse<'m>, E>,
        T: BundleItemSetFor<'a>,
    {
        let mut out = MaybeUninit::<Self>::uninit();

        for (index, item) in Self::layout().iter().enumerate() {
            f(BundleItemRequest {
                _ty: PhantomData,
                resp: BundleItemResponse {
                    _invariant: PhantomData,
                },
                index,
                item,
                write_to: unsafe {
                    out.as_mut_ptr().cast::<u8>().add(item.offset())
                },
            })?;
        }

        Ok(unsafe { out.assume_init() })
    }

    pub fn new_auto<'a, F>(mut f: F) -> Self
    where
        F: for<'m> FnMut(BundleItemRequest<'a, 'm>) -> BundleItemResponse<'m>,
        T: BundleItemSetFor<'a>,
    {
        Self::try_new_auto::<_, !>(|req| Ok(f(req))).unwrap()
    }
}

#[derive(Debug)]
pub struct BundleItemRequest<'a, 'm> {
    // Ensure that `'a`, the lifetime of the bundle we're producing, is contravariant i.e. that `'a`
    // can be made to live longer but not shorter.
    _ty: PhantomData<fn(&'a ())>,

    // This is the marker ZST we use to provide proof that `provide_mut` or `provide_ref` was called.
    // This makes `'m`, a universally quantified lifetime acting as a token, as invariant.
    resp: BundleItemResponse<'m>,

    // The index of this request in the layout array.
    index: usize,

    // The item we're trying to access.
    item: &'static BundleItemLayout,

    // The location to which we write our value.
    write_to: *mut u8,
}

impl<'a, 'm> BundleItemRequest<'a, 'm> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn item(&self) -> &'static BundleItemLayout {
        self.item
    }

    pub fn marker_type_id(&self) -> TypeId {
        self.item.marker_type_id()
    }

    pub fn pointee_type_id(&self) -> TypeId {
        self.item.pointee_type_id()
    }

    pub fn is_mut(&self) -> bool {
        self.item.is_mut()
    }

    pub fn is_ref(&self) -> bool {
        self.item.is_ref()
    }

    pub fn provide_mut<T: ?Sized + 'static>(self, value: &'a mut T) -> BundleItemResponse<'m> {
        assert!(
            self.pointee_type_id() == TypeId::of::<T>(),
            "expected `{}` (for context item `{}`), got `{}`",
            self.item.pointee_name(),
            self.item.marker_name(),
            type_name::<&T>(),
        );

        unsafe {
            self.write_to.cast::<&'a mut T>().write(value);
        }

        self.resp
    }

    pub fn provide_ref<T: ?Sized + 'static>(self, value: &'a T) -> BundleItemResponse<'m> {
        assert!(
            self.pointee_type_id() == TypeId::of::<T>() && self.is_ref(),
            "expected `{}` (for context item `{}`), got `{}`",
            self.item.pointee_name(),
            self.item.marker_name(),
            type_name::<&T>(),
        );

        unsafe {
            self.write_to.cast::<&'a T>().write(value);
        }

        self.resp
    }
}

pub struct BundleItemResponse<'m> {
    _invariant: PhantomData<fn(&'m ()) -> &'m ()>,
}

impl fmt::Debug for BundleItemResponse<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BundleItemResponse").finish_non_exhaustive()
    }
}

#[repr(transparent)]
pub struct BundleItemLayout((
    /* marker name */ &'static str,
    /* pointee name */ &'static str,
    /* marker type */ u128,
    /* pointee type */ u128,
    /* is mutable */ bool,
    /* offset */ usize,
));

impl fmt::Debug for BundleItemLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BundleItemLayout")
            .field("marker_name", &self.marker_name())
            .field("pointee_name", &self.pointee_name())
            .field("marker_type_id", &self.marker_type_id())
            .field("pointee_type_id", &self.pointee_type_id())
            .field("is_mut", &self.is_mut())
            .field("offset", &self.offset())
            .finish()
    }
}

impl BundleItemLayout {
    pub fn marker_name(&self) -> &'static str {
        (self.0).0
    }

    pub fn pointee_name(&self) -> &'static str {
        (self.0).1
    }

    pub fn marker_type_id(&self) -> TypeId {
        TypeId::from_u128((self.0).2)
    }

    pub fn pointee_type_id(&self) -> TypeId {
        TypeId::from_u128((self.0).3)
    }

    pub fn is_mut(&self) -> bool {
        (self.0).4
    }

    pub fn is_ref(&self) -> bool {
        !(self.0).4
    }

    fn offset(&self) -> usize {
        (self.0).5
    }
}
