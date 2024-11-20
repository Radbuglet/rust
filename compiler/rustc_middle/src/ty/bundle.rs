use std::ops::Deref;
use std::panic::Location;

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet, IndexEntry};
use rustc_data_structures::graph::{DirectedGraph, Successors, scc};
use rustc_data_structures::sync::Lrc;
use rustc_errors::DiagMessage;
use rustc_hir as hir;
use rustc_hir::{def_id::{DefId, LocalDefId, LocalDefIdMap}, def::DefKind};
use rustc_index::IndexVec;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, Encodable, Decodable};
use rustc_target::spec::abi::Abi;

use crate::mir;
use crate::thir::{self, visit as thir_visit};
use crate::ty::{self, Mutability, list::RawList, Ty, TyCtxt};
use crate::query::Providers;

// TODO: needed for `delay_bug`; upstream versions of rustc have this as an inherent method
use ty::Interner as _;
use thir_visit::Visitor as _;

use crate::error as errors;

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        reified_bundle,
        components_borrowed_local,
        components_borrowed_graph,
        components_borrowed,
        components_borrowed_fn_reify_check,
        ..*providers
    };
}

// === ContextSolveStage === //

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[derive(HashStable, Encodable, Decodable)]
pub enum ContextSolveStage {
    ClosureUpVars,
    GraphSolving,
    MirBuilding,
}

impl ContextSolveStage {
    pub fn fully_resolved(self) -> bool {
        self > Self::GraphSolving
    }

    #[track_caller]
    pub fn err_once<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
        during: ContextSolveStage,
        emit: impl FnOnce() -> ty::ErrorGuaranteed,
    ) -> ty::ErrorGuaranteed {
        if self == during {
            emit()
        } else {
            tcx.delay_bug(format!(
                "a later context solving stage should have reported the error at {}",
                Location::caller(),
            ))
        }
    }

    #[track_caller]
    pub fn err_during_graph<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
        emit: impl FnOnce() -> ty::ErrorGuaranteed,
    ) -> ty::ErrorGuaranteed {
        self.err_once(tcx, Self::GraphSolving, emit)
    }

    #[track_caller]
    pub fn err_during_mir<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
        emit: impl FnOnce() -> ty::ErrorGuaranteed,
    ) -> ty::ErrorGuaranteed {
        self.err_once(tcx, Self::MirBuilding, emit)
    }
}

// === ReifiedBundle === //

#[derive(Debug, Clone)]
pub struct ReifiedBundle<'tcx> {
    pub original_bundle: Ty<'tcx>,
    pub value_ty: Ty<'tcx>,
    pub fields: FxIndexMap<DefId, ReifiedBundleMemberList<'tcx>>,
    pub generic_fields: FxIndexMap<Ty<'tcx>, ReifiedBundleMemberList<'tcx>>,
    pub generic_sets: FxIndexMap<Ty<'tcx>, ReifiedBundleMemberList<'tcx>>,
    pub infer_sets: FxIndexMap<DefId, ReifiedBundleMemberList<'tcx>>,
}

impl<'tcx> ReifiedBundle<'tcx> {
    pub fn concrete_items(&self) -> impl Iterator<Item = (DefId, Mutability)> + use<'_, 'tcx> {
        self.fields.iter().map(|(def_id, list)| {
            (*def_id, list.iter().map(|v| v.mutability).max().unwrap())
        })
    }

    pub fn generic_types(&self) -> FxIndexSet<Ty<'tcx>> {
        // We expect to only run this during MirBuilding.
        debug_assert!(self.infer_sets.is_empty());

        self.generic_fields
            .keys()
            .copied()
            .chain(self.generic_sets.keys().copied())
            .collect()
    }
}

pub type ReifiedBundleMemberList<'tcx> = smallvec::SmallVec<[ReifiedBundleMember<'tcx>; 1]>;

#[derive(Debug, Clone)]
pub struct ReifiedBundleMember<'tcx> {
    pub location: ReifiedBundleProjs<'tcx>,
    pub mutability: Mutability,
}

#[derive(Debug, Copy, Clone, HashStable)]
pub struct ReifiedBundleProjs<'tcx>(pub &'tcx [ReifiedBundleProj<'tcx>]);

#[derive(Debug, Copy, Clone, HashStable)]
pub struct ReifiedBundleProj<'tcx> {
    pub field: ty::FieldIdx,
    pub ty: Ty<'tcx>,
}

impl<'tcx> ReifiedBundleProjs<'tcx> {
    pub fn ty(&self) -> Ty<'tcx> {
        self.0.last().unwrap().ty
    }

    pub fn project_place(
        &self,
        tcx: TyCtxt<'tcx>,
        place: mir::Place<'tcx>,
        deeper: impl IntoIterator<Item = mir::PlaceElem<'tcx>>,
    ) -> mir::Place<'tcx> {
        let projection = place.projection
            .iter()
            .chain(self.0.iter().map(|elem| {
                mir::PlaceElem::Field(elem.field, elem.ty)
            }))
            .chain(deeper);

        mir::Place {
            local: place.local,
            projection: tcx.mk_place_elems_from_iter(projection)
        }
    }
}

fn reified_bundle<'tcx>(
    tcx: TyCtxt<'tcx>,
    (ty, stage): (Ty<'tcx>, ContextSolveStage),
) -> &'tcx ReifiedBundle<'tcx> {
    tcx.arena.alloc(reified_bundle_owned(tcx, ty, stage))
}

pub fn reified_bundle_owned<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    stage: ContextSolveStage,
) -> ReifiedBundle<'tcx> {
    // Extract the inner type.
    let bundle_arg = ty.bundle_item_set(tcx);

    // Extract the fields.
    let mut walker = ReifiedBundleWalker {
        tcx,
        stage,
        proj_stack: Vec::new(),
        bundle_item_set_to_value: FxHashMap::default(),
        fields: FxIndexMap::default(),
        generic_fields: FxIndexMap::default(),
        generic_sets: FxIndexMap::default(),
        infer_sets: FxIndexMap::default(),
    };
    let value_ty = walker.bundle_item_set_to_value(bundle_arg);
    walker.proj_stack.push(ReifiedBundleProj {
        field: ty::FieldIdx::ZERO,
        ty: value_ty,
    });
    walker.collect_fields_in_bundle_item_set(bundle_arg);

    ReifiedBundle {
        original_bundle: ty,
        value_ty,
        fields: walker.fields,
        generic_fields: walker.generic_fields,
        generic_sets: walker.generic_sets,
        infer_sets: walker.infer_sets,
    }
}

struct ReifiedBundleWalker<'tcx> {
    tcx: TyCtxt<'tcx>,
    stage: ContextSolveStage,
    proj_stack: Vec<ReifiedBundleProj<'tcx>>,
    bundle_item_set_to_value: FxHashMap<Ty<'tcx>, Ty<'tcx>>,

    fields: FxIndexMap<DefId, ReifiedBundleMemberList<'tcx>>,
    generic_fields: FxIndexMap<Ty<'tcx>, ReifiedBundleMemberList<'tcx>>,
    generic_sets: FxIndexMap<Ty<'tcx>, ReifiedBundleMemberList<'tcx>>,
    infer_sets: FxIndexMap<DefId, ReifiedBundleMemberList<'tcx>>,
}

impl<'tcx> ReifiedBundleWalker<'tcx> {
    /// Transforms a type implementing `BundleItemSet` (e.g. `(&mut FOO, &BAR)`) into the
    /// corresponding bundle value types (e.g. `(&mut FOO::Item, &BAR::Item)`). Returns `Error` if
    /// this cannot be fully reified.
    fn bundle_item_set_to_value(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let Some(&out) = self.bundle_item_set_to_value.get(&ty) {
            return out;
        }

        let out = match ReifiedBundleItemSet::decode(ty) {
            ReifiedBundleItemSet::Ref(re, muta, did) => {
                Ty::new_ref(
                    self.tcx,
                    re,
                    self.tcx.context_ty(did),
                    muta,
                )
            }
            ReifiedBundleItemSet::Tuple(fields) => {
                Ty::new_tup_from_iter(
                    self.tcx,
                    fields.iter().map(|field| self.bundle_item_set_to_value(field)),
                )
            }
            ReifiedBundleItemSet::GenericRef(re, muta, ty) => {
                let context_item_trait_def_id =
                    self.tcx.require_lang_item(ty::LangItem::ContextItemTrait, None);

                let item_assoc_para_def_id =
                    self.tcx.associated_item_def_ids(context_item_trait_def_id)[0];

                let value_ty = Ty::new_alias(
                    self.tcx,
                    ty::AliasTyKind::Projection,
                    ty::AliasTy::new(self.tcx, item_assoc_para_def_id, [ty]),
                );
                Ty::new_ref(self.tcx, re, value_ty, muta)
            }
            ReifiedBundleItemSet::GenericSet(ty) => {
                Ty::new_alias(
                    self.tcx,
                    ty::AliasTyKind::Projection,
                    ty::AliasTy::new(
                        self.tcx,
                        self.tcx.lang_items().bundle_item_set_values().unwrap(),
                        [ty],
                    ),
                )
            }
            ReifiedBundleItemSet::InferSet(..) => {
                ty
            }
            ReifiedBundleItemSet::Error(err) => {
                Ty::new_error(self.tcx, err)
            }
        };

        self.bundle_item_set_to_value.insert(ty, out);
        out
    }

    fn collect_fields_in_bundle_item_set(&mut self, ty: Ty<'tcx>) {
        match ReifiedBundleItemSet::decode(ty) {
            ReifiedBundleItemSet::Ref(_re, muta, did) => {
                self.fields.entry(did).or_default().push(ReifiedBundleMember {
                    location: ReifiedBundleProjs(self.tcx.arena.alloc_from_iter(
                        self.proj_stack.iter().copied(),
                    )),
                    mutability: muta,
                });
            }
            ReifiedBundleItemSet::Tuple(fields) => {
                for (i, field) in fields.iter().enumerate() {
                    let field_values = self.bundle_item_set_to_value(field);
                    self.proj_stack.push(ReifiedBundleProj {
                        field: ty::FieldIdx::from_usize(i),
                        ty: field_values,
                    });
                    self.collect_fields_in_bundle_item_set(field);
                    self.proj_stack.pop();
                }
            }
            ReifiedBundleItemSet::GenericRef(_re, muta, ty) => {
                self.generic_fields.entry(ty).or_default().push(ReifiedBundleMember {
                    location: ReifiedBundleProjs(self.tcx.arena.alloc_from_iter(
                        self.proj_stack.iter().copied(),
                    )),
                    mutability: muta,
                });
            }
            ReifiedBundleItemSet::GenericSet(ty) => {
                self.generic_sets.entry(ty).or_default().push(ReifiedBundleMember {
                    location: ReifiedBundleProjs(self.tcx.arena.alloc_from_iter(
                        self.proj_stack.iter().copied(),
                    )),
                    mutability: Mutability::Mut,
                });
            }
            ReifiedBundleItemSet::InferSet(did, re) => {
                if self.stage.fully_resolved() {
                    let inner_ty = resolve_infer_bundle_set(self.tcx, did, re);
                    let inner_val_ty = resolve_infer_bundle_values(self.tcx, did, re);

                    self.proj_stack.push(ReifiedBundleProj {
                        field: ty::FieldIdx::ZERO,
                        ty: inner_val_ty,
                    });
                    self.collect_fields_in_bundle_item_set(inner_ty);
                    self.proj_stack.pop();
                } else {
                    self.infer_sets.entry(did).or_default().push(ReifiedBundleMember {
                        location: ReifiedBundleProjs(self.tcx.arena.alloc_from_iter(
                            self.proj_stack.iter().copied(),
                        )),
                        mutability: Mutability::Mut,
                    });
                }
            }
            ReifiedBundleItemSet::Error(_err) => {}
        }
    }
}

#[derive(Copy, Clone)]
pub enum ReifiedBundleItemSet<'tcx> {
    /// A reference `&{mut?} T` where `T` is known and implements `ContextItem`.
    Ref(ty::Region<'tcx>, Mutability, DefId),

    /// A tuple `(A, B, C, ...)` where each `A_k` implements `BundleItemSet`.
    Tuple(&'tcx RawList<(), Ty<'tcx>>),

    /// A reference `&{mut?} T` where `T` is generic but implements `BundleItemSet`.
    GenericRef(ty::Region<'tcx>, Mutability, Ty<'tcx>),

    /// A type `T` where `T` is generic but implements `BundleItemSet`.
    GenericSet(Ty<'tcx>),

    /// An inferred bundle set generated by `infer_bundle!('re)`. Does not show up after the
    /// `GraphSolving` solving stage.
    InferSet(DefId, ty::Region<'tcx>),

    /// The directly underlying type is a `TyKind::Error`.
    Error(ty::ErrorGuaranteed),
}

impl<'tcx> ReifiedBundleItemSet<'tcx> {
    pub fn decode(ty: Ty<'tcx>) -> Self {
        match ty.kind() {
            &ty::Ref(re, inner, muta) => {
                match ReifiedContextItem::decode(inner) {
                    ReifiedContextItem::Reified(did) => Self::Ref(re, muta, did),
                    ReifiedContextItem::Generic(ty) => Self::GenericRef(re, muta, ty),
                    ReifiedContextItem::Error(err) => Self::Error(err),
                }
            }
            ty::Tuple(fields) => {
                Self::Tuple(fields)
            }
            ty::Alias(..) | ty::Param(..) => {
                Self::GenericSet(ty)
            }
            &ty::InferBundle(did, re) => {
                Self::InferSet(did, re)
            }
            &ty::Error(err) => {
                Self::Error(err)
            }

            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Adt(..)
            | ty::Foreign(..)
            | ty::Str
            | ty::Array(..)
            | ty::Pat(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::ContextMarker(..) => {
                bug!("expected {ty} to implement `BundleItem` but it doesn't");
            }

            ty::Infer(..)
            | ty::Placeholder(..)
            | ty::Bound(..) => {
                bug!("{ty} should not occur in a type being resolved by `reified_bundle`");
            }
        }
    }
}

#[derive(Copy, Clone)]
pub enum ReifiedContextItem<'tcx> {
    Reified(DefId),
    Generic(Ty<'tcx>),
    Error(ty::ErrorGuaranteed),
}

impl<'tcx> ReifiedContextItem<'tcx> {
    pub fn decode(ty: Ty<'tcx>) -> Self {
        match ty.kind() {
            &ty::ContextMarker(did) => {
                Self::Reified(did)
            }
            &ty::Error(err) => {
                Self::Error(err)
            }
            ty::Alias(..)
            | ty::Param(..) => {
                Self::Generic(ty)
            }

            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Adt(..)
            | ty::Foreign(..)
            | ty::Str
            | ty::Array(..)
            | ty::Pat(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::InferBundle(..)
            | ty::Tuple(..) => {
                bug!("expected {ty} to implement `ContextItem` but it doesn't");
            }

            ty::Placeholder(..)
            | ty::Infer(..)
            | ty::Bound(..) => {
                bug!("{ty} should not occur in a type being resolved by `reified_bundle`");
            }
        }
    }
}

// === PackShape Parsing === //

#[derive(Clone, Debug, HashStable)]
pub enum PackShape<'tcx> {
    /// Fetches a single context item from the environment.
    ExtractEnv(Mutability, DefId),

    /// Fetches a bunch of context items from the environment. Does not show up after the
    /// `GraphSolving` solving stage.
    ExtractEnvInfer(DefId),

    /// Fetches a single context item by reborrowing from a field.
    ExtractLocalRef(Mutability, usize, ty::ReifiedBundleProjs<'tcx>),

    /// Fetches a single context item by moving/copying a set.
    ExtractLocalMove(usize, ty::ReifiedBundleProjs<'tcx>),

    /// A placeholder for local infer bundle reborrows. Does not show up after the `GraphSolving`
    /// solving stage.
    ExtractLocalInferPlaceholder(usize),

    /// Constructs a tuple of sub-shapes.
    MakeTuple(Box<[PackShape<'tcx>]>),

    /// Constructs an inference bundle containing the sub-values. Does not show up until after the
    /// `GraphSolving` stage.
    MakeInfer {
        bundle: DefId,
        inner_ty: Ty<'tcx>,
        inner_shape: Box<PackShape<'tcx>>,
    },

    /// An unrecoverable error ocurred and this field has no useful way to assign a value.
    Error(ty::ErrorGuaranteed),
}

type PackShapeStoreMap<'tcx> = IndexVec<thir::PackExprIndex, Option<Lrc<PackShape<'tcx>>>>;

pub struct PackShapeStore<'tcx> {
    map: Option<PackShapeStoreMap<'tcx>>,
}

impl<'tcx> PackShapeStore<'tcx> {
    pub fn new_ignore() -> Self {
        Self { map: None }
    }

    pub fn new_store() -> Self {
        Self { map: Some(IndexVec::new()) }
    }

    pub fn resolve(
        &mut self,
        tcx: TyCtxt<'tcx>,
        stage: ContextSolveStage,
        body: &thir::Thir<'tcx>,
        expr: &thir::Expr<'tcx>,
    ) -> PackShapeStoreRes<'tcx> {
        let thir::ExprKind::Pack { index, flags, exprs } = &expr.kind else {
            bug!("expected `Pack` expression, got {expr:?}");
        };

        let entry = self.map.as_mut().map(|map| map.ensure_contains_elem(*index, || None));

        if let Some(Some(entry)) = entry {
            return PackShapeStoreRes::Loan(entry.clone());
        }

        let ty = expr.ty.bundle_item_set(tcx);
        let bundles = exprs.iter()
            .map(|&expr| tcx.reified_bundle((body[expr].ty, stage)))
            .collect::<Vec<_>>();

        let shape = make_bundle_pack_shape(
            tcx,
            stage,
            *flags,
            &bundles,
            ty,
        );

        if let Some(entry) = entry {
            let shape = Lrc::new(shape);
            PackShapeStoreRes::Loan(entry.insert(shape).clone())
        } else {
            PackShapeStoreRes::Owned(shape)
        }
    }
}

pub enum PackShapeStoreRes<'tcx> {
    Owned(PackShape<'tcx>),
    Loan(Lrc<PackShape<'tcx>>),
}

impl<'tcx> PackShapeStoreRes<'tcx> {
    pub fn into_owned(self) -> PackShape<'tcx> {
        match self {
            Self::Owned(v) => v,
            Self::Loan(_) => bug!("`into_owned` called on `PackShapeStoreRes::Loan`"),
        }
    }

    pub fn into_rc(self) -> Lrc<PackShape<'tcx>> {
        match self {
            Self::Owned(v) => Lrc::new(v),
            Self::Loan(v) => v,
        }
    }
}

impl<'tcx> Deref for PackShapeStoreRes<'tcx> {
    type Target = PackShape<'tcx>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(v) => v,
            Self::Loan(v) => v,
        }
    }
}

pub fn make_bundle_pack_shape<'tcx>(
    tcx: TyCtxt<'tcx>,
    stage: ContextSolveStage,
    flags: ty::PackFlags,
    bundles: &[&ty::ReifiedBundle<'tcx>],
    ty: Ty<'tcx>,
) -> PackShape<'tcx> {
    match ty::ReifiedBundleItemSet::decode(ty) {
        ty::ReifiedBundleItemSet::Ref(_re, muta, def_id) => {
            for (i, bundle) in bundles.iter().enumerate() {
                if !bundle.infer_sets.is_empty() && flags.allows_env() {
                    return PackShape::Error(stage.err_during_graph(tcx, || {
                        todo!("unclear whether the reference should come from the environment or the infer set");
                    }));
                }

                let Some(members) = bundle.fields.get(&def_id) else {
                    continue;
                };

                if members.len() > 1 {
                    return PackShape::Error(stage.err_during_mir(tcx, || {
                        todo!("multiple possible origins for the context item");
                    }));
                }

                let Some(member) = members.get(0) else { unreachable!() };

                return PackShape::ExtractLocalRef(muta, i, member.location);
            }

            if flags.allows_env() {
                return PackShape::ExtractEnv(muta, def_id)
            }

            PackShape::Error(stage.err_during_mir(tcx, || {
                todo!("component not provided");
            }))
        }
        ty::ReifiedBundleItemSet::Tuple(items) => {
            let items = items.iter()
                .map(|item| make_bundle_pack_shape(tcx, stage, flags, bundles, item))
                .collect();

            PackShape::MakeTuple(items)
        }
        ty::ReifiedBundleItemSet::GenericSet(ty) => {
            for (i, bundle) in bundles.iter().enumerate() {
                let Some(members) = bundle.generic_sets.get(&ty) else {
                    continue;
                };

                if members.len() > 1 {
                    return PackShape::Error(stage.err_during_mir(tcx, || {
                        todo!("multiple possible origins for the generic set");
                    }));
                }

                let Some(member) = members.get(0) else { unreachable!() };

                return PackShape::ExtractLocalMove(i, member.location);
            }

            PackShape::Error(stage.err_during_mir(tcx, || {
                todo!("component not provided");
            }))
        }
        ty::ReifiedBundleItemSet::GenericRef(_re, muta, ty) => {
            for (i, bundle) in bundles.iter().enumerate() {
                let Some(members) = bundle.generic_fields.get(&ty) else {
                    continue;
                };

                if members.len() > 1 {
                    return PackShape::Error(stage.err_during_mir(tcx, || {
                        todo!("multiple possible origins for the generic item");
                    }));
                }

                let Some(member) = members.get(0) else { unreachable!() };

                return PackShape::ExtractLocalRef(muta, i, member.location);
            }

            PackShape::Error(stage.err_during_mir(tcx, || {
                todo!("component not provided");
            }))
        }
        ty::ReifiedBundleItemSet::InferSet(did, re) => {
            if stage.fully_resolved() {
                let inner_ty = resolve_infer_bundle_set(tcx, did, re);
                let inner_values_ty = resolve_infer_bundle_values(tcx, did, re);
                let inner_shape = make_bundle_pack_shape(tcx, stage, flags, bundles, inner_ty);

                PackShape::MakeInfer {
                    bundle: did,
                    inner_ty: inner_values_ty,
                    inner_shape: Box::new(inner_shape)
                }
            } else {
                for (i, bundle) in bundles.iter().enumerate() {
                    // Ambiguity errors will be handled by regular reference ambiguity semantics with
                    // tge desugaring of these infer bundles to regular component sets.
                    if bundle.infer_sets.contains_key(&did) {
                        return PackShape::ExtractLocalInferPlaceholder(i);
                    };
                }

                PackShape::ExtractEnvInfer(did)
            }
        }
        ty::ReifiedBundleItemSet::Error(err) => PackShape::Error(err),
    }
}

// === THIR Parsing === //

#[derive(Debug, Clone)]
pub struct ContextUsedByExpr {
    pub concrete: Vec<(DefId, Mutability)>,
    pub infer: Option<DefId>,
}

pub fn context_used_by_expr<'tcx>(
    tcx: TyCtxt<'tcx>,
    stage: ContextSolveStage,
    pack_shape_store: &mut PackShapeStore<'tcx>,
    body: &thir::Thir<'tcx>,
    expr: &thir::Expr<'tcx>,
) -> ContextUsedByExpr {
    let mut uses = ContextUsedByExpr {
        concrete: Vec::new(),
        infer: None,
    };
    context_used_by_expr_inner(
        tcx,
        stage,
        pack_shape_store,
        body,
        expr,
        &mut uses,
    );
    uses
}

fn context_used_by_expr_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    stage: ContextSolveStage,
    pack_shape_store: &mut PackShapeStore<'tcx>,
    body: &thir::Thir<'tcx>,
    expr: &thir::Expr<'tcx>,
    uses: &mut ContextUsedByExpr,
) {
    use thir::ExprKind::*;

    match &expr.kind {
        &ContextRef { item, muta } => {
            uses.concrete.push((item, muta));
        }
        Pack { .. } => {
            let shape = pack_shape_store.resolve(tcx, stage, body, expr);
            context_used_by_pack_shape(tcx, &shape, uses);
        }
        &Call { ty, .. } => {
            if
                stage == ContextSolveStage::MirBuilding &&
                let Some(def_id) = extract_static_callee_for_context(tcx, ty)
            {
                uses.concrete.extend(tcx.components_borrowed(def_id).iter());
            }
        }

        Scope { .. }
        | Box { .. }
        | If { .. }
        | Deref { .. }
        | Binary { .. }
        | LogicalOp { .. }
        | Unary { .. }
        | Cast { .. }
        | Use { .. }
        | NeverToAny { .. }
        | PointerCoercion { .. }
        | Loop { .. }
        | Let { .. }
        | Match { .. }
        | Block { .. }
        | Assign { .. }
        | AssignOp { .. }
        | Field { .. }
        | Index { .. }
        | VarRef { .. }
        | UpvarRef { .. }
        | Borrow { .. }
        | RawBorrow { .. }
        | Break { .. }
        | Continue { .. }
        | Return { .. }
        | Become { .. }
        | ConstBlock { .. }
        | Repeat { .. }
        | Array { .. }
        | Tuple { .. }
        | Adt { .. }
        | PlaceTypeAscription { .. }
        | ValueTypeAscription { .. }
        | Closure { .. }
        | Literal { .. }
        | NonHirLiteral { .. }
        | ZstLiteral { .. }
        | NamedConst { .. }
        | ConstParam { .. }
        | StaticRef { .. }
        | InlineAsm { .. }
        | OffsetOf { .. }
        | ThreadLocalRef { .. }
        | Yield { .. } => {
            // (no context users directly introduced by expression)
        }
    }
}

pub fn context_used_by_pack_shape<'tcx>(
    tcx: TyCtxt<'tcx>,
    shape: &PackShape<'tcx>,
    uses: &mut ContextUsedByExpr,
) {
    match shape {
        &PackShape::ExtractEnv(muta, item) => {
            uses.concrete.push((item, muta));
        }
        &PackShape::ExtractEnvInfer(item) => {
            uses.infer = Some(item);
        }
        PackShape::MakeTuple(fields) => {
            for field in fields {
                context_used_by_pack_shape(tcx, field, uses);
            }
        }
        PackShape::MakeInfer {
            bundle: _,
            inner_ty: _,
            inner_shape,
        } => {
            context_used_by_pack_shape(tcx, inner_shape, uses);
        }
        PackShape::ExtractLocalRef(..)
        | PackShape::ExtractLocalMove(..)
        | PackShape::ExtractLocalInferPlaceholder(..)
        | PackShape::Error(..) => {
            // (does not directly introduce context uses)
        }
    }
}

pub fn context_binds_by_stmt<'tcx>(
    tcx: TyCtxt<'tcx>,
    stage: ContextSolveStage,
    body: &thir::Thir<'tcx>,
    stmt: &thir::Stmt<'tcx>,
) -> Option<(ContextBinder, &'tcx ReifiedBundle<'tcx>)> {
    use thir::StmtKind::*;

    match &stmt.kind {
        &BindContext { self_id, bundle, .. } => {
            let bundle_ty = body.exprs[bundle].ty;
            let reified = tcx.reified_bundle((bundle_ty, stage));

            Some((
                ContextBinder::LocalBinder(self_id),
                reified,
            ))
        }
        Expr { .. } | Let { .. } => {
            None
        }
    }
}

pub fn has_components_borrowed_entry<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    tcx.def_kind(def_id) == DefKind::InferBundle
        || can_participate_in_context_solving(tcx, def_id)
}

pub fn can_participate_in_context_solving<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    tcx.is_mir_available(def_id)
        // The non-const condition is load bearing as it helps avoid recursive query calls.
        && !tcx.is_const_fn_raw(def_id)
}

pub fn extract_static_callee_for_context<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<DefId> {
    match ty.kind() {
        &ty::FnDef(def_id, ..) => {
            let include = can_participate_in_context_solving(tcx, def_id)
                && def_can_borrow_context(tcx, def_id).is_ok();

            include.then_some(def_id)
        }
        ty::Bool
        | ty::Char
        | ty::Int(..)
        | ty::Uint(..)
        | ty::Float(..)
        | ty::Adt(..)
        | ty::Foreign(..)
        | ty::Str
        | ty::Array(..)
        | ty::Pat(..)
        | ty::Slice(..)
        | ty::RawPtr(..)
        | ty::Ref(..)
        | ty::FnPtr(..)
        | ty::Dynamic(..)
        | ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Coroutine(..)
        | ty::CoroutineWitness(..)
        | ty::Never
        | ty::Tuple(..)
        | ty::Alias(..)
        | ty::Param(..)
        | ty::Bound(..)
        | ty::Placeholder(..)
        | ty::Infer(..)
        | ty::ContextMarker(..)
        | ty::InferBundle(..)
        | ty::Error(..) => {
            None
        }
    }
}

// === ContextBindTracker === //

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, HashStable)]
pub enum ContextBinder {
    FuncEnv,
    LocalBinder(thir::StmtId),
}

impl ContextBinder {
    pub fn from_info(info: Option<ContextBindInfo>) -> ContextBinder {
        match info {
            Some(info) => ContextBinder::LocalBinder(info.binder),
            None => ContextBinder::FuncEnv,
        }
    }

    pub fn is_env(self) -> bool {
        self == ContextBinder::FuncEnv
    }

    pub fn unwrap_local(self) -> thir::StmtId {
        match self {
            ContextBinder::FuncEnv => panic!("expected function-local context binder"),
            ContextBinder::LocalBinder(stmt) => stmt,
        }
    }
}

#[derive(Default)]
pub struct ContextBindTracker {
    curr_local_binders: FxHashMap<DefId, ContextBindInfo>,
    old_binders: Vec<(DefId, Option<ContextBindInfo>)>,
}

#[derive(Debug, Copy, Clone)]
pub struct ContextBindInfo {
    pub binder: thir::StmtId,
    pub muta: Mutability,
}

impl ContextBindTracker {
    pub fn push_scope(&self) -> ContextBindScope {
        ContextBindScope(self.old_binders.len())
    }

    pub fn pop_scope(&mut self, scope: ContextBindScope) {
        for (item, binder) in self.old_binders.drain((scope.0)..) {
            match binder {
                Some(old) => {
                    self.curr_local_binders.insert(item, old);
                },
                None => {
                    self.curr_local_binders.remove(&item);
                },
            }
        }
    }

    pub fn bind(&mut self, item: DefId, info: ContextBindInfo) {
        let old_binder = self.curr_local_binders.insert(item, info);
        self.old_binders.push((item, old_binder));
    }

    pub fn bind_from_stmt<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        stage: ContextSolveStage,
        body: &thir::Thir<'tcx>,
        stmt: &thir::Stmt<'tcx>,
    ) {
        let Some((binder, reified)) = context_binds_by_stmt(tcx, stage, body, stmt) else {
            return;
        };

        for (item, muta) in reified.concrete_items() {
            self.bind(item, ContextBindInfo {
                binder: binder.unwrap_local(),
                muta,
            });
        }
    }

    pub fn resolve(&self, item: DefId) -> ContextBinder {
        match self.curr_local_binders.get(&item) {
            Some(info) => ContextBinder::LocalBinder(info.binder),
            None => ContextBinder::FuncEnv,
        }
    }

    pub fn resolve_rich(&self, item: DefId) -> Option<ContextBindInfo> {
        self.curr_local_binders.get(&item).copied()
    }
}

#[must_use]
pub struct ContextBindScope(usize);

// === ContextSet === //

#[derive(Debug, Default, HashStable, Eq, PartialEq, Clone, TyEncodable, TyDecodable)]
pub struct ContextSet(FxIndexMap<DefId, Mutability>);

impl ContextSet {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn add(&mut self, id: DefId, muta: Mutability) -> bool {
        match self.0.entry(id) {
            IndexEntry::Vacant(entry) => {
                entry.insert(muta);
                true
            }
            IndexEntry::Occupied(mut entry) => {
                let entry = entry.get_mut();

                if *entry < muta {
                    *entry = muta;
                    true
                } else {
                    false
                }

            }
        }
    }

    pub fn has(&self, item: DefId) -> bool {
        self.0.contains_key(&item)
    }

    pub fn get(&self, item: DefId) -> Option<Mutability> {
        self.0.get(&item).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (DefId, Mutability)> + '_ {
        self.0.iter().map(|(&k, &v)| (k, v))
    }
}

// === Context Graph === //

#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct ContextBorrowsLocal<'tcx> {
    /// Function calls and local borrows affecting the borrow set of the function itself.
    pub entry: ContextBorrowsLocalNode<'tcx>,

    /// Function calls and local borrows affecting the borrow set of each `InferBundle` binding.
    pub infers: Vec<(DefId, ContextBorrowsLocalNode<'tcx>)>,
}

#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct ContextBorrowsLocalNode<'tcx> {
    /// The context being borrowed directly by the function (or the infer bundle binding) from
    /// its environment.
    pub local: ContextSet,

    /// The set of functions called (or infer bundle sets inherited) directly by the function (or
    /// the infer bundle binding) from its environment.
    pub calls: Vec<IndirectContextCall<'tcx>>,
}

#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct IndirectContextCall<'tcx> {
    /// The `DefId` of the function being called (or the infer bundle being inherited).
    pub target: DefId,

    /// The set of context items absorbed by `let static` bindings between the call (or the infer
    /// bundle being inherited) and the function entry (or the infer bundle binding).
    pub absorbs: &'tcx ContextSet,
}

impl<'tcx> ContextBorrowsLocal<'tcx> {
    pub fn nodes(&self) -> impl Iterator<Item = (Option<DefId>, &ContextBorrowsLocalNode<'tcx>)> {
        [(None, &self.entry)]
            .into_iter()
            .chain(self.infers.iter().map(|(k, v)| (Some(*k), v)))
    }
}

// TODO: Make queries public (stop glob-importing `bundle` in `ty`)
fn components_borrowed_local<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> &'tcx ty::ContextBorrowsLocal<'tcx> {
    let empty_set = tcx.arena.alloc(ContextSet::default());

    let Ok((thir, entry)) = tcx.thir_body(def_id) else {
        return tcx.arena.alloc(ContextBorrowsLocal {
            entry: ContextBorrowsLocalNode {
                local: ContextSet::default(),
                calls: Vec::new(),
            },
            infers: Vec::new(),
        });
    };

    let thir = thir.borrow();
    let thir = &thir;

    let mut visitor = ComponentsBorrowedLocalVisitor {
        tcx,
        thir,
        bind_tracker: ContextBindTracker::default(),
        curr_absorb: empty_set,
        borrows_nodes: vec![
            (None, ContextBorrowsLocalNode {
                local: ContextSet::default(),
                calls: Vec::new(),
            }),
        ],
        curr_borrows_node: 0,
    };
    visitor.visit_expr(&visitor.thir.exprs[entry]);

    let mut borrow_nodes_iter = visitor.borrows_nodes.into_iter();

    tcx.arena.alloc(ContextBorrowsLocal {
        entry: borrow_nodes_iter.next().unwrap().1,
        infers: borrow_nodes_iter
            .map(|(did, info)| (did.unwrap(), info))
            .collect(),
    })
}

struct ComponentsBorrowedLocalVisitor<'thir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: &'thir thir::Thir<'tcx>,
    bind_tracker: ContextBindTracker,
    curr_absorb: &'tcx ContextSet,
    borrows_nodes: Vec<(Option<DefId>, ContextBorrowsLocalNode<'tcx>)>,
    curr_borrows_node: usize,
}

impl<'thir, 'tcx> thir_visit::Visitor<'thir, 'tcx> for ComponentsBorrowedLocalVisitor<'thir, 'tcx> {
    fn thir(&self) -> &'thir thir::Thir<'tcx> {
        self.thir
    }

    fn visit_block(&mut self, block: &'thir thir::Block) {
        let old_absorb = self.curr_absorb;
        let old_scope = self.bind_tracker.push_scope();
        let old_borrows_node = self.curr_borrows_node;

        thir_visit::walk_block(self, block);

        self.curr_absorb = old_absorb;
        self.bind_tracker.pop_scope(old_scope);
        self.curr_borrows_node = old_borrows_node;
    }

    fn visit_stmt(&mut self, stmt: &'thir thir::Stmt<'tcx>) {
        thir_visit::walk_stmt(self, stmt);

        // Update the set of binds to which the rest of this block is subjected.
        //
        // We do this after visiting the statement to ensure that expressions used in the statement
        // don't see the bindings to which they're subjected.
        if let Some((_binder, reified)) = context_binds_by_stmt(
            self.tcx,
            ContextSolveStage::GraphSolving,
            self.thir,
            stmt,
        ) {
            // TODO: Consider enforcing no-generics and no-ambiguity rules here instead of in
            // THIR building.

            // See if we have an infer set binder.
            if !reified.infer_sets.is_empty() {
                // TODO: Ensure that there is only one origin.

                // We begin by clearing the `curr_absorb` list since the infer set now collects all
                // of these unabsorbed borrows.
                self.curr_absorb = self.tcx.arena.alloc(ContextSet::default());

                // Start collecting to a new `ContextBorrowsLocalNode`
                self.curr_borrows_node = self.borrows_nodes.len();
                self.borrows_nodes.push((
                    Some(*reified.infer_sets.get_index(0).unwrap().0),
                    ContextBorrowsLocalNode {
                        local: ContextSet::default(),
                        calls: Vec::new(),
                    },
                ));
            }

            // Update the set of absorbers we're subject to.
            let mut absorb = self.curr_absorb.clone();

            for (item, muta) in reified.concrete_items() {
                absorb.add(item, muta);
            }

            self.curr_absorb = self.tcx.arena.alloc(absorb);
        }

        // We apply regular binds after infer set binds because the inverse would cause those binds
        // to be immediately shadowed by the inference set.
        self.bind_tracker.bind_from_stmt(
            self.tcx,
            ContextSolveStage::GraphSolving,
            self.thir,
            stmt,
        );
    }

    fn visit_expr(&mut self, expr: &'thir thir::Expr<'tcx>) {
        use thir::ExprKind::*;

        let uses = context_used_by_expr(
            self.tcx,
            ContextSolveStage::GraphSolving,
            &mut PackShapeStore::new_ignore(),
            self.thir,
            expr,
        );

        for (item, muta) in uses.concrete {
            if self.bind_tracker.resolve(item).is_env() {
                self.borrows_nodes[self.curr_borrows_node]
                    .1
                    .local
                    .add(item, muta);
            }
        }

        if let Some(infer) = uses.infer {
            self.borrows_nodes[self.curr_borrows_node]
                .1
                .calls
                .push(IndirectContextCall {
                    target: infer,
                    absorbs: self.curr_absorb,
                });
        }

        match &expr.kind {
            &Call { ty, .. } => {
                if let Some(def_id) = extract_static_callee_for_context(self.tcx, ty) {
                    self.borrows_nodes[self.curr_borrows_node]
                        .1
                        .calls
                        .push(IndirectContextCall {
                            target: def_id,
                            absorbs: self.curr_absorb,
                        });
                }
            }

            Scope { .. }
            | Box { .. }
            | If { .. }
            | Deref { .. }
            | Binary { .. }
            | LogicalOp { .. }
            | Unary { .. }
            | Cast { .. }
            | Use { .. }
            | NeverToAny { .. }
            | PointerCoercion { .. }
            | Loop { .. }
            | Let { .. }
            | Match { .. }
            | Block { .. }
            | Assign { .. }
            | AssignOp { .. }
            | Field { .. }
            | Index { .. }
            | VarRef { .. }
            | UpvarRef { .. }
            | Borrow { .. }
            | RawBorrow { .. }
            | Break { .. }
            | Continue { .. }
            | Return { .. }
            | Become { .. }
            | ConstBlock { .. }
            | Repeat { .. }
            | Array { .. }
            | Tuple { .. }
            | Adt { .. }
            | PlaceTypeAscription { .. }
            | ValueTypeAscription { .. }
            | Closure { .. }
            | Literal { .. }
            | NonHirLiteral { .. }
            | ZstLiteral { .. }
            | NamedConst { .. }
            | ConstParam { .. }
            | StaticRef { .. }
            | InlineAsm { .. }
            | OffsetOf { .. }
            | ThreadLocalRef { .. }
            | ContextRef { .. }
            | Pack { .. }
            | Yield { .. } => {
                // (do not result in another function borrowing context being called)
            }
        }

        thir_visit::walk_expr(self, expr);
    }
}

rustc_index::newtype_index! {
    #[derive(Ord, PartialOrd)]
    struct FuncIdx {}
}

rustc_index::newtype_index! {
    #[derive(Ord, PartialOrd)]
    struct SccIdx {}
}

rustc_index::newtype_index! {
    #[derive(Ord, PartialOrd)]
    struct SubSccIdx {}
}

rustc_index::newtype_index! {
    #[derive(Ord, PartialOrd)]
    struct SubFuncIdx {}
}

struct ComponentsBorrowedGraph<'tcx> {
    nodes: FxIndexMap<DefId, FuncNodeWeight<'tcx>>,
}

struct FuncNodeWeight<'tcx> {
    local: &'tcx ContextSet,
    calls: Vec<FuncEdgeWeight<'tcx>>,
}

struct FuncEdgeWeight<'tcx> {
    target: FuncIdx,
    absorbs: &'tcx ContextSet,
}

impl<'tcx> DirectedGraph for ComponentsBorrowedGraph<'tcx> {
    type Node = FuncIdx;

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl<'tcx> Successors for ComponentsBorrowedGraph<'tcx> {
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.nodes[node.as_usize()].calls.iter().map(|v| v.target)
    }
}

struct ComponentsBorrowedSccSubgraph<'a, 'tcx> {
    graph: &'a ComponentsBorrowedGraph<'tcx>,
    members: scc::SccMemberSet<'a, FuncIdx, SubFuncIdx>,
    remove: DefId,
}

impl<'a, 'tcx> DirectedGraph for ComponentsBorrowedSccSubgraph<'a, 'tcx> {
    type Node = SubFuncIdx;

    fn num_nodes(&self) -> usize {
        self.members.len()
    }
}

impl<'a, 'tcx> Successors for ComponentsBorrowedSccSubgraph<'a, 'tcx> {
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.graph
            .nodes[self.members.member_to_node(node).as_usize()]
            .calls
            .iter()
            .filter(|v| v.absorbs.has(self.remove))
            .filter_map(|v| self.members.node_to_member(v.target))
    }
}

fn def_can_borrow_context<'tcx>(
    tcx: TyCtxt<'tcx>,
    func: DefId,
) -> Result<(), DiagMessage> {
    let kind = tcx.def_kind(func);

    if kind == DefKind::InferBundle {
        return Ok(());
    }

    if tcx.entry_fn(()).is_some_and(|(did, _)| did == func) {
        return Err(errors::middle_entry_fn_uses_ctx);
    }

    if
        kind == DefKind::AssocFn
        && let DefKind::Impl { of_trait: true } | DefKind::Trait = tcx.def_kind(tcx.parent(func))
    {
        return Err(errors::middle_trait_member_uses_ctx);
    }

    if tcx.is_closure_like(func) {
        return Err(errors::middle_closure_uses_ctx);
    }

    if tcx.is_coroutine(func) {
        return Err(errors::middle_async_fn_uses_ctx);
    }

    if kind == DefKind::Fn && tcx.fn_sig(func).skip_binder().abi() != Abi::Rust {
        return Err(errors::middle_extern_fn_uses_ctx);
    }

    Ok(())
}

fn components_borrowed_graph<'tcx>(
    tcx: TyCtxt<'tcx>,
    _: (),
) -> &'tcx LocalDefIdMap<&'tcx ContextSet> {
    // Build a graph from all the functions.
    let mut graph = ComponentsBorrowedGraph {
        nodes: FxIndexMap::default(),
    };

    // `can_participate_in_context_solving` contains a check for `is_mir_available`, which uses
    // `tcx.mir_keys` to make its determination. Hence, this will not miss any important
    // `LocalDefId`s.
    for &def_id in tcx.mir_keys(()) {
        if !can_participate_in_context_solving(tcx, def_id.to_def_id()) {
            continue;
        }

        for (node_def_id, info) in tcx.components_borrowed_local(def_id).nodes() {
            let node_def_id = node_def_id.unwrap_or(def_id.to_def_id());

            let calls = info.calls.iter()
                .map(|call| {
                    let callee_entry = graph.nodes.entry(call.target);
                    let callee = callee_entry.index();

                    if call.target.is_local() {
                        callee_entry.or_insert(FuncNodeWeight {
                            // Initialize to some bogus data to reserve the entry.
                            local: &info.local,
                            calls: Vec::new(),
                        });
                    } else {
                        // We won't be initializing this callee later in the loop so we have to
                        // initialize it immediately. In any case, the way these are initialized is
                        // fundamentally different from the way we initialize local nodes.
                        callee_entry.or_insert(FuncNodeWeight {
                            local: tcx.components_borrowed(call.target),
                            // We can treat it as if this foreign function borrowed all of its context
                            // items directly since this graph is not directly used for diagnostics.
                            calls: Vec::new(),
                        });
                    }

                    FuncEdgeWeight {
                        target: FuncIdx::from_usize(callee),
                        absorbs: &call.absorbs,
                    }
                })
                .collect();

            match graph.nodes.entry(node_def_id) {
                IndexEntry::Vacant(entry) => {
                    entry.insert(FuncNodeWeight {
                        local: &info.local,
                        calls,
                    });
                }
                IndexEntry::Occupied(mut entry) => {
                    let entry = entry.get_mut();

                    let mut local = entry.local.clone();
                    for (comp, muta) in info.local.iter() {
                        local.add(comp, muta);
                    }
                    entry.local = tcx.arena.alloc(local);
                    entry.calls.extend(calls);
                }
            }
        }
    }

    // Iterate through strongly connected components in dependency order. The main algorithm for
    // dealing with acyclic portions of the graph runs in O(n) w.r.t the number of nodes whereas the
    // algorithm for dealing with strongly-connected components runs in O(nm) where `n` is the number
    // of functions in the cluster and `m` is the number of unique component sets absorbed.
    let sccs = scc::Sccs::<FuncIdx, SccIdx>::new(&graph);
    let members = scc::SccMembers::<FuncIdx, SccIdx, SubFuncIdx>::new(&sccs);

    for scc in sccs.all_sccs() {
        let members = members.of(scc);

        // We begin by seeding local sets with their target components.
        for &member in members.iter() {
            let member_data = &graph.nodes[FuncIdx::as_usize(member)];
            let mut set = member_data.local.clone();

            for call in &member_data.calls {
                // Ignore component-internal calls for now since their local sets haven't yet been
                // set up correctly.
                if sccs.scc(call.target) == scc {
                    continue;
                }

                for (item, muta) in graph.nodes[FuncIdx::as_usize(call.target)].local.iter() {
                    if !call.absorbs.has(item) {
                        set.add(item, muta);
                    }
                }
            }

            graph.nodes[FuncIdx::as_usize(member)].local = tcx.arena.alloc(set);
        }

        // If this was a single-member component, we've properly populated the entire SCC with the
        // previous step and can move on to the next component.
        if members.len() <= 1 {
            continue;
        }

        // Collect the set of all context items absorbed by a function in the SCC.
        let mut absorbed_comps = FxHashSet::<DefId>::default();

        for &member in members.iter() {
            for call in &graph.nodes[FuncIdx::as_usize(member)].calls {
                if sccs.scc(call.target) != scc {
                    continue;
                }

                absorbed_comps.extend(call.absorbs.iter().map(|(item, _muta)| item));
            }
        }

        // Determine the mutabilities of each absorbed component for every node in the SCC.
        for &absorbed in &absorbed_comps {
            let sub_graph = ComponentsBorrowedSccSubgraph {
                graph: &graph,
                members,
                remove: absorbed,
            };

            let sub_sccs = scc::Sccs::<SubFuncIdx, SubSccIdx>::new(&sub_graph);
            let sub_members = scc::SccMembers::<SubFuncIdx, SubSccIdx, u32>::new(&sub_sccs);
            let mut scc_results = IndexVec::<SubSccIdx, Option<Mutability>>::from_elem_n(
                None,
                sub_sccs.num_sccs(),
            );

            // We know that every SCC in the subgraph will have the same set of borrows.
            for sub_scc in sub_sccs.all_sccs() {
                // Determine the mutability of the current `absorbed` component throughout this
                // sub-SCC. This is determined by two things:
                //
                // 1. The local borrow sets of the components in this SCC.
                // 2. The borrows of sub-SCCs at least one of our members points into.
                //
                // This is sufficient because...
                //
                // - Borrows introduced by their callees outside of the main-SCC have already been
                //   seeded into the local sets of nodes in the main-SCC.
                //
                // - Borrows introduced by members in the same sub-SCC will count towards `comp_muta`
                //   when we iterate over them.
                //
                // - Borrows introduced by members in the same main-SCC but in a different sub-SCC
                //   will be counted by the borrows of sub-SCCs at least one of our members points
                //   into.
                //
                // We know these will properly reflect absorption semantics because:
                //
                // 1) Local borrow-set seeding properly checks this field.
                // 2) `sub_graph` and `sub_sccs` properly obey absorptions for `absorbed`.
                //

                let sub_members = sub_members.of(sub_scc);
                let mut comp_muta = None;
                let mut update_comp_muta = |muta: Option<Mutability>| {
                    let Some(muta) = muta else { return };
                    let comp_muta = comp_muta.get_or_insert(Mutability::Not);
                    *comp_muta = (*comp_muta).max(muta);
                };

                for &suc_scc in sub_sccs.successors(sub_scc) {
                    update_comp_muta(scc_results[suc_scc]);
                }

                for &sub_member in sub_members.iter() {
                    let member = members[sub_member];
                    let member_data = &graph.nodes[FuncIdx::as_usize(member)];
                    update_comp_muta(member_data.local.get(absorbed));
                }

                scc_results[sub_scc] = comp_muta;

                let Some(comp_muta) = comp_muta else {
                    continue;
                };

                // If this absorbed component is used by any member of the sub-SCC after removal of
                // the relevant absorption edges, we know the entire SCC will share this same borrow.
                // Write it back!
                for &sub_member in sub_members.iter() {
                    let member = members[sub_member];
                    let member_data = &mut graph.nodes[FuncIdx::as_usize(member)];
                    let mut set = member_data.local.clone();
                    set.add(absorbed, comp_muta);
                    member_data.local = tcx.arena.alloc(set);
                }
            }
        }

        // Determine the borrows sets for every other context item not involved in an absorption.
        let mut union_set = ContextSet::default();

        for &member in members.iter() {
            for (item, muta) in graph.nodes[FuncIdx::as_usize(member)].local.iter() {
                if !absorbed_comps.contains(&item) {
                    union_set.add(item, muta);
                }
            }
        }

        for &member in members.iter() {
            let member_data = &mut graph.nodes[FuncIdx::as_usize(member)];
            let mut set = member_data.local.clone();

            for (item, muta) in union_set.iter() {
                set.add(item, muta);
            }

            member_data.local = tcx.arena.alloc(set);
        }
    }

    // Summarize the graph as an ID map.
    let mut map = LocalDefIdMap::<&'tcx ContextSet>::default();

    for (&node_def_id, weight) in &graph.nodes {
        let Some(node_def_id) = node_def_id.as_local() else {
            continue;
        };
        map.insert(node_def_id, weight.local);

        if
            !weight.local.is_empty()
            && let Err(msg) = def_can_borrow_context(tcx, node_def_id.to_def_id())
        {
            let span = tcx.span_of_impl(node_def_id.to_def_id()).unwrap();
            let diag = tcx.dcx().struct_span_err(span, msg);
            // TODO: Add much more context
            diag.emit();
        }
    }

    tcx.arena.alloc(map)
}

fn components_borrowed<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx ContextSet {
    assert!(has_components_borrowed_entry(tcx, def_id.to_def_id()));

    tcx.components_borrowed_graph(())
        .get(&def_id)
        .copied()
        .unwrap_or_else(|| {
            assert_eq!(
                tcx.def_kind(def_id.to_def_id()),
                DefKind::InferBundle,
                "only infer bundles can lack an entry in the context borrow graph, got {def_id:?}",
            );
            tcx.arena.alloc(ContextSet::default())
        })
}

pub fn resolve_infer_bundle_set<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    re: ty::Region<'tcx>,
) -> Ty<'tcx> {
    let orig_items = tcx.components_borrowed(def_id);
    let mut items = orig_items
        .iter()
        .map(|(item, muta)| Ty::new_ref(tcx, re, Ty::new_context_marker(tcx, item), muta));

    if orig_items.len() == 1 {
        items.next().unwrap()
    } else {
        Ty::new_tup_from_iter(tcx, items)
    }
}

pub fn resolve_infer_bundle_values<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    re: ty::Region<'tcx>,
) -> Ty<'tcx> {
    let orig_items = tcx.components_borrowed(def_id);
    let mut items = orig_items
        .iter()
        .map(|(item, muta)| Ty::new_ref(tcx, re, tcx.context_ty(item), muta));

    if orig_items.len() == 1 {
        items.next().unwrap()
    } else {
        Ty::new_tup_from_iter(tcx, items)
    }
}

fn components_borrowed_fn_reify_check<'tcx>(tcx: TyCtxt<'tcx>, (): ()) {
    for &def_id in tcx.mir_keys(()) {
        // Ignore definitions whose type-check results come from another function.
        let typeck_root_def_id = tcx.typeck_root_def_id(def_id.to_def_id());
        if typeck_root_def_id != def_id.to_def_id() {
            continue;
        }

        // Ignore functions without type-check results.
        if !tcx.has_typeck_results(def_id) {
            continue;
        }
        let owner_id = hir::OwnerId { def_id };
        let typeck_results = tcx.typeck(def_id);

        // Go through each adjustment and ensure that it doesn't unsize a function borrowing
        // context.
        for (node_local_id, adjustments) in typeck_results.adjustments().items_in_stable_order() {
            let node_hir_id = hir::HirId {
                owner: owner_id,
                local_id: node_local_id,
            };
            let mut curr_ty = typeck_results.node_type(node_hir_id);

            for adjustment in adjustments {
                use ty::adjustment::{Adjust::*, PointerCoercion::*};

                // Figure out the type of the expression before the adjustment is applied.
                let pre_adjust_ty = curr_ty;
                curr_ty = adjustment.target;

                // Determine whether the adjustment unsizes a function.
                let unsized_def_id = match &adjustment.kind {
                    Pointer(ReifyFnPointer) => {
                        match pre_adjust_ty.kind() {
                            ty::FnDef(def_id, _) => *def_id,
                            _ => bug!("ReifyFnPointer was not expected to be used against {pre_adjust_ty:?}"),
                        }
                    },
                    Pointer(ClosureFnPointer(_)) => {
                        // Although this does potentially turn a `FnTy` into a `FnPtr`, closures
                        // are already prevented from borrowing any context so no need to emit
                        // additional diagnostics.
                        continue;
                    }

                    Pointer(UnsafeFnPointer)
                    | Pointer(MutToConstPointer)
                    | Pointer(ArrayToPointer)
                    | Pointer(Unsize)
                    | Pointer(DynStar)
                    | NeverToAny
                    | Deref(..)
                    | Borrow(..)
                    | ReborrowPin(..) => {
                        // Cannot unsize fns to fn-pointers
                        continue;
                    }
                };

                // See whether the unsized function borrowed any context.
                let comps = tcx.components_borrowed(unsized_def_id);
                if comps.is_empty() {
                    continue;
                }

                // If it did, report the error!
                let span = tcx.hir().span(node_hir_id);
                tcx.dcx().emit_err(errors::ReifiedFnUsingCtx { span });
            }
        }
    }
}
