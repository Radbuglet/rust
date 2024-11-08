use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet, IndexEntry};
use rustc_data_structures::graph::{DirectedGraph, Successors, scc};
use rustc_hir::def_id::{DefId, LocalDefId, LocalDefIdMap};
use rustc_index::IndexVec;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};

use crate::mir;
use crate::thir::{self, visit as thir_visit};
use crate::ty::{self, Mutability, list::RawList, Ty, TyCtxt};
use crate::query::Providers;

use thir_visit::Visitor as _;

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        reified_bundle,
        components_borrowed_local,
        components_borrowed_graph,
        components_borrowed,
        ..*providers
    };
}

// === ReifiedBundle === //

#[derive(Debug, Clone)]
pub struct ReifiedBundle<'tcx> {
    pub original_bundle: Ty<'tcx>,
    pub value_ty: Ty<'tcx>,
    pub fields: FxIndexMap<DefId, ReifiedBundleMemberList<'tcx>>,
    pub generic_fields: FxIndexMap<Ty<'tcx>, ReifiedBundleMemberList<'tcx>>,
    pub generic_sets: FxIndexMap<Ty<'tcx>, ReifiedBundleMemberList<'tcx>>,
}

impl<'tcx> ReifiedBundle<'tcx> {
    pub fn generic_types(&self) -> FxIndexSet<Ty<'tcx>> {
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

fn reified_bundle<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> &'tcx ReifiedBundle<'tcx> {
    // Extract the inner type.
    let bundle_arg = ty.bundle_item_set(tcx);

    // Extract the fields.
    let mut walker = ReifiedBundleWalker {
        tcx,
        proj_stack: Vec::new(),
        bundle_item_set_to_value: FxHashMap::default(),
        fields: FxIndexMap::default(),
        generic_fields: FxIndexMap::default(),
        generic_sets: FxIndexMap::default(),
    };
    let value_ty = walker.bundle_item_set_to_value(bundle_arg);
    walker.proj_stack.push(ReifiedBundleProj {
        field: ty::FieldIdx::ZERO,
        ty: value_ty,
    });
    walker.collect_fields_in_bundle_item_set(bundle_arg);

    tcx.arena.alloc(ReifiedBundle {
        original_bundle: ty,
        value_ty,
        fields: walker.fields,
        generic_fields: walker.generic_fields,
        generic_sets: walker.generic_sets,
    })
}

struct ReifiedBundleWalker<'tcx> {
    tcx: TyCtxt<'tcx>,
    proj_stack: Vec<ReifiedBundleProj<'tcx>>,
    bundle_item_set_to_value: FxHashMap<Ty<'tcx>, Ty<'tcx>>,

    fields: FxIndexMap<DefId, ReifiedBundleMemberList<'tcx>>,
    generic_fields: FxIndexMap<Ty<'tcx>, ReifiedBundleMemberList<'tcx>>,
    generic_sets: FxIndexMap<Ty<'tcx>, ReifiedBundleMemberList<'tcx>>,
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
            &ty::Error(err) => {
                Self::Error(err)
            }

            ty::InferBundle(..) => todo!(),

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

// === Visit Context Uses === //

pub fn visit_context_used_by_expr<'tcx>(
    tcx: TyCtxt<'tcx>,
    expr: &thir::Expr<'tcx>,
    visit_calls: bool,
    f: &mut impl FnMut(DefId, Mutability),
) {
    use thir::ExprKind::*;

    match &expr.kind {
        &ContextRef { item, muta } => {
            f(item, muta);
        }
        Pack { shape, .. } => {
            visit_context_uses_by_pack_shape(tcx, shape, f);
        }
        &Call { ty, .. } => {
            if
                visit_calls &&
                let Some(def_id) = extract_static_callee_for_context(tcx, ty)
            {
                for (item, muta) in tcx.components_borrowed(def_id).iter() {
                    f(item, muta);
                }
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

pub fn visit_context_uses_by_pack_shape<'tcx>(
    tcx: TyCtxt<'tcx>,
    shape: &thir::PackShape<'tcx>,
    f: &mut impl FnMut(DefId, Mutability),
) {
    match shape {
        &thir::PackShape::ExtractEnv(muta, item) => {
            f(item, muta);
        }
        thir::PackShape::Tuple(fields) => {
            for field in fields {
                visit_context_uses_by_pack_shape(tcx, field, f);
            }
        }
        thir::PackShape::ExtractLocalRef(..)
        | thir::PackShape::ExtractLocalMove(..)
        | thir::PackShape::Error(..) => {
            // (does not directly introduce context uses)
        }
    }
}

pub fn visit_context_binds_by_stmt<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &thir::Thir<'tcx>,
    stmt: &thir::Stmt<'tcx>,
    f: &mut impl FnMut(ContextBinder, DefId, Mutability),
) {
    use thir::StmtKind::*;

    match &stmt.kind {
        &BindContext { self_id, bundle, .. } => {
            let bundle_ty = body.exprs[bundle].ty;
            let reified = tcx.reified_bundle(bundle_ty);

            for (&item, members) in &reified.fields {
                let muta = members.iter()
                    .map(|member| member.mutability)
                    .max()
                    .unwrap();

                f(ContextBinder::LocalBinder(self_id), item, muta);
            }
        }
        Expr { .. } | Let { .. } => {
            // (does not directly introduce context binds)
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
        body: &thir::Thir<'tcx>,
        stmt: &thir::Stmt<'tcx>,
    ) {
        visit_context_binds_by_stmt(tcx, body, stmt, &mut |binder, item, muta| {
            self.bind(item, ContextBindInfo {
                binder: binder.unwrap_local(),
                muta,
            });
        })
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

// === Context Graph === //

#[derive(Debug, Default, HashStable, Eq, PartialEq, Clone, TyEncodable, TyDecodable)]
pub struct ContextSet(FxIndexMap<DefId, Mutability>);

impl ContextSet {
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

#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct ContextBorrowsLocal<'tcx> {
    local: ContextSet,
    calls: Vec<IndirectContextCall<'tcx>>,
}

#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
struct IndirectContextCall<'tcx> {
    target: DefId,
    absorbs: &'tcx ContextSet,
}

pub fn can_def_borrow_extern_context<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    tcx.def_kind(def_id) == ty::DefKind::Fn
        && tcx.is_mir_available(def_id)
        // The non-const condition is load bearing as it helps avoid recursive query calls.
        && !tcx.is_const_fn_raw(def_id)
}

pub fn extract_static_callee_for_context<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<DefId> {
    match ty.kind() {
        &ty::FnDef(def_id, ..) => {
            can_def_borrow_extern_context(tcx, def_id).then_some(def_id)
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

// TODO: Make queries public (stop glob-importing `bundle` in `ty`)
fn components_borrowed_local<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> &'tcx ty::ContextBorrowsLocal<'tcx> {
    let empty_set = tcx.arena.alloc(ContextSet::default());

    let Ok((thir, entry)) = tcx.thir_body(def_id) else {
        return tcx.arena.alloc(ContextBorrowsLocal {
            local: ContextSet::default(),
            calls: Vec::new(),
        });
    };

    let thir = thir.borrow();
    let thir = &thir;

    let mut visitor = ComponentsBorrowedLocalVisitor {
        tcx,
        thir,
        bind_tracker: ContextBindTracker::default(),
        local: ContextSet::default(),
        calls: Vec::new(),
        curr_absorb: empty_set,
    };
    visitor.visit_expr(&visitor.thir.exprs[entry]);

    tcx.arena.alloc(ContextBorrowsLocal {
        local: visitor.local,
        calls: visitor.calls,
    })
}

struct ComponentsBorrowedLocalVisitor<'thir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: &'thir thir::Thir<'tcx>,
    bind_tracker: ContextBindTracker,
    local: ContextSet,
    calls: Vec<IndirectContextCall<'tcx>>,
    curr_absorb: &'tcx ContextSet,
}

impl<'thir, 'tcx> thir_visit::Visitor<'thir, 'tcx> for ComponentsBorrowedLocalVisitor<'thir, 'tcx> {
    fn thir(&self) -> &'thir thir::Thir<'tcx> {
        self.thir
    }

    fn visit_block(&mut self, block: &'thir thir::Block) {
        let old_absorb = self.curr_absorb;
        let old_scope = self.bind_tracker.push_scope();
        thir_visit::walk_block(self, block);
        self.curr_absorb = old_absorb;
        self.bind_tracker.pop_scope(old_scope);
    }

    fn visit_stmt(&mut self, stmt: &'thir thir::Stmt<'tcx>) {
        self.bind_tracker.bind_from_stmt(self.tcx, self.thir, stmt);

        visit_context_binds_by_stmt(self.tcx, self.thir, stmt, &mut |_binder, item, muta| {
            let mut absorb = self.curr_absorb.clone();
            absorb.add(item, muta);
            self.curr_absorb = self.tcx.arena.alloc(absorb);
        });

        thir_visit::walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'thir thir::Expr<'tcx>) {
        use thir::ExprKind::*;

        visit_context_used_by_expr(self.tcx, expr, false, &mut |item, muta| {
            if self.bind_tracker.resolve(item).is_env() {
                self.local.add(item, muta);
            }
        });

        match &expr.kind {
            &Call { ty, .. } => {
                if let Some(def_id) = extract_static_callee_for_context(self.tcx, ty) {
                    self.calls.push(IndirectContextCall {
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
    calls: Box<[FuncEdgeWeight<'tcx>]>,
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

fn components_borrowed_graph<'tcx>(
    tcx: TyCtxt<'tcx>,
    _: (),
) -> &'tcx LocalDefIdMap<&'tcx ContextSet> {
    // Build a graph from all the functions.
    let mut graph = ComponentsBorrowedGraph {
        nodes: FxIndexMap::default(),
    };

    // `can_def_borrow_extern_context` contains a check for `is_mir_available`, which uses
    // `tcx.mir_keys` to make its determination. Hence, this will not miss any important `LocalDefId`s.
    for &def_id in tcx.mir_keys(()) {
        if !can_def_borrow_extern_context(tcx, def_id.to_def_id()) {
            continue;
        }

        let info = tcx.components_borrowed_local(def_id);
        let calls = info.calls.iter()
            .map(|call| {
                let callee_entry = graph.nodes.entry(call.target);
                let callee = callee_entry.index();

                if call.target.is_local() {
                    callee_entry.or_insert(FuncNodeWeight {
                        // Initialize to some bogus data to reserve the entry.
                        local: &info.local,
                        calls: Box::new([]),
                    });
                } else {
                    // We won't be initializing this callee later in the loop so we have to
                    // initialize it immediately. In any case, the way these are initialized is
                    // fundamentally different from the way we initialize local nodes.
                    callee_entry.or_insert(FuncNodeWeight {
                        local: tcx.components_borrowed(call.target),
                        // We can treat it as if this foreign function borrowed all of its context
                        // items directly since this graph is not directly used for diagnostics.
                        calls: Box::new([]),
                    });
                }

                FuncEdgeWeight {
                    target: FuncIdx::from_usize(callee),
                    absorbs: &call.absorbs,
                }
            })
            .collect();

        graph.nodes.insert(def_id.to_def_id(), FuncNodeWeight {
            local: &info.local,
            calls,
        });
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
    }

    tcx.arena.alloc(map)
}

fn components_borrowed<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx ty::ContextSet {
    tcx.components_borrowed_graph(())[&def_id]
}
