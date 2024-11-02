#![expect(unused)]  // TODO

use rustc_data_structures::fx::{FxHashMap, FxIndexMap, IndexEntry};
use rustc_data_structures::graph::{DirectedGraph, Successors, scc};
use rustc_hir::def_id::{DefId, DefIndex, LocalDefId, LocalDefIdMap};
use rustc_index::{Idx, IndexVec};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use smallvec::SmallVec;

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
    pub fields: FxIndexMap<DefId, SmallVec<[ReifiedBundleMember<'tcx>; 1]>>,
    pub generic_types: Vec<Ty<'tcx>>,
}

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
        generic_types: Vec::new(),
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
        generic_types: walker.generic_types,
    })
}

struct ReifiedBundleWalker<'tcx> {
    tcx: TyCtxt<'tcx>,
    proj_stack: Vec<ReifiedBundleProj<'tcx>>,
    bundle_item_set_to_value: FxHashMap<Ty<'tcx>, Ty<'tcx>>,

    fields: FxIndexMap<DefId, SmallVec<[ReifiedBundleMember<'tcx>; 1]>>,
    generic_types: Vec<Ty<'tcx>>,
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
            ReifiedBundleItemSet::Generic(ty) => {
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
            ReifiedBundleItemSet::Generic(ty) => {
                self.generic_types.push(ty);
            }
            ReifiedBundleItemSet::Error(_err) => {}
        }
    }
}

#[derive(Copy, Clone)]
pub enum ReifiedBundleItemSet<'tcx> {
    Ref(ty::Region<'tcx>, Mutability, DefId),
    Tuple(&'tcx RawList<(), Ty<'tcx>>),
    Generic(Ty<'tcx>),
    Error(ty::ErrorGuaranteed),
}

impl<'tcx> ReifiedBundleItemSet<'tcx> {
    pub fn decode(ty: Ty<'tcx>) -> Self {
        match ty.kind() {
            &ty::Ref(re, inner, muta) => {
                match ReifiedContextItem::decode(inner) {
                    ReifiedContextItem::Reified(did) => Self::Ref(re, muta, did),
                    ReifiedContextItem::Generic(ty) => Self::Generic(ty),
                    ReifiedContextItem::Error(err) => Self::Error(err),
                }
            }
            ty::Tuple(fields) => {
                Self::Tuple(fields)
            }
            ty::Alias(..) | ty::Param(..) => {
                Self::Generic(ty)
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

// === Context Graph === //

#[derive(Debug, Default, HashStable, Eq, PartialEq, Clone, TyEncodable, TyDecodable)]
pub struct ContextSet(FxIndexMap<DefId, Mutability>);

impl ContextSet {
    pub fn add(&mut self, id: DefId, muta: Mutability) -> bool {
        match self.0.entry(id) {
            IndexEntry::Vacant(mut entry) => {
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

pub fn is_valid_static_callee_for_context<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    tcx.def_kind(def_id) == ty::DefKind::Fn
        && tcx.is_mir_available(def_id)
        // The non-const condition is load bearing as it helps avoid recursive query calls.
        && !tcx.is_const_fn_raw(def_id)
}

pub fn extract_static_callee_for_context<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<DefId> {
    match ty.kind() {
        &ty::FnDef(def_id, ..) => {
            is_valid_static_callee_for_context(tcx, def_id).then_some(def_id)
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
    local: ContextSet,
    calls: Vec<IndirectContextCall<'tcx>>,
    curr_absorb: &'tcx ContextSet,
}

impl<'thir, 'tcx> ComponentsBorrowedLocalVisitor<'thir, 'tcx> {
    fn visit_pack_shape(&mut self, shape: &'thir thir::PackShape<'tcx>) {
        match shape {
            &thir::PackShape::ExtractEnv(muta, item, binder) => {
                if binder.is_env() {
                    self.local.add(item, muta);
                }
            }
            thir::PackShape::Tuple(fields) => {
                for field in fields {
                    self.visit_pack_shape(field);
                }
            }
            thir::PackShape::ExtractLocal(..) | thir::PackShape::Error(..) => {
                // (does not introduce context uses)
            }
        }
    }
}

impl<'thir, 'tcx> thir_visit::Visitor<'thir, 'tcx> for ComponentsBorrowedLocalVisitor<'thir, 'tcx> {
    fn thir(&self) -> &'thir thir::Thir<'tcx> {
        self.thir
    }

    fn visit_block(&mut self, block: &'thir thir::Block) {
        let old_absorb = self.curr_absorb;
        thir_visit::walk_block(self, block);
        self.curr_absorb = old_absorb;
    }

    fn visit_stmt(&mut self, stmt: &'thir thir::Stmt<'tcx>) {
        use thir::StmtKind::*;

        match &stmt.kind {
            &BindContext { bundle, .. } => {
                let bundle_ty = self.thir.exprs[bundle].ty;
                let reified = self.tcx.reified_bundle(bundle_ty);

                let mut set = self.curr_absorb.clone();
                let mut changed = false;

                for (def_id, members) in &reified.fields {
                    for member in members {
                        changed |= set.add(*def_id, member.mutability);
                    }
                }

                if changed {
                    self.curr_absorb = self.tcx.arena.alloc(set);
                }
            }
            Expr { .. } | Let { .. } => {
                thir_visit::walk_stmt(self, stmt);
            }
        }
    }

    fn visit_expr(&mut self, expr: &'thir thir::Expr<'tcx>) {
        use thir::ExprKind::*;

        match &expr.kind {
            &ContextRef { item, muta, binder, .. } => {
                if binder.is_env() {
                    self.local.add(item, muta);
                }
            }
            Pack { shape, .. } => {
                self.visit_pack_shape(shape);
            }
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
            | Yield { .. } => {
                // (no context users directly introduced by expression)
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

struct SccMembers<N: Idx, S: Idx> {
    comp_range_ends: IndexVec<S, u32>,
    node_buf: Box<[N]>,
}

impl<N: Idx, S: Idx + Ord> SccMembers<N, S> {
    fn new<A: scc::Annotation>(sccs: &scc::Sccs<N, S, A>) -> Self {
        // Accumulate the number of elements in each SCC
        let mut comp_range_ends = IndexVec::from_elem_n(0, sccs.num_sccs());
        let mut node_buf = (0..sccs.scc_indices().len())
            .map(|_| N::new(0))
            .collect::<Box<_>>();

        for (node, &scc) in sccs.scc_indices().iter_enumerated() {
            comp_range_ends[scc] += 1;
        }

        // Transform these into fill-start indices
        let mut next_start = 0;
        for end in comp_range_ends.iter_mut() {
            let count = *end;
            *end = next_start;
            next_start += count;
        }

        // Fill up the buffer
        for (node, &scc) in sccs.scc_indices().iter_enumerated() {
            let fill_idx = &mut comp_range_ends[scc];
            node_buf[*fill_idx as usize] = node;
            *fill_idx += 1;
        }

        Self {
            comp_range_ends,
            node_buf,
        }
    }

    fn of(&self, comp: S) -> SccMemberSet<'_, N> {
        let slice = self.of_raw(comp);
        if slice.len() == 1 {
            SccMemberSet::Single(slice[0])
        } else {
            SccMemberSet::Many(slice)
        }
    }

    fn of_raw(&self, comp: S) -> &[N] {
        let start = comp.index()
            .checked_sub(1)
            .map(|idx| self.comp_range_ends[S::new(idx)])
            .unwrap_or(0) as usize;
        let end = self.comp_range_ends[comp] as usize;

        &self.node_buf[start..end]
    }
}

#[derive(Debug, Copy, Clone)]
enum SccMemberSet<'a, N> {
    Single(N),
    Many(&'a [N]),
}

fn components_borrowed_graph<'tcx>(
    tcx: TyCtxt<'tcx>,
    _: (),
) -> &'tcx LocalDefIdMap<&'tcx ContextSet> {
    // Build a graph from all the functions.
    let mut graph = ComponentsBorrowedGraph {
        nodes: FxIndexMap::default(),
    };

    // `is_valid_static_callee_for_context` contains a check for `is_mir_available`, which uses
    // `tcx.mir_keys` to make its determination. Hence, this will not miss any important `LocalDefId`s.
    for &def_id in tcx.mir_keys(()) {
        if !is_valid_static_callee_for_context(tcx, def_id.to_def_id()) {
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
    let mut sccs = scc::Sccs::<FuncIdx, SccIdx>::new(&graph);
    let members = SccMembers::new(&sccs);

    for scc in sccs.all_sccs() {
        match members.of(scc) {
            SccMemberSet::Single(node) => {
                let node_data = &graph.nodes[node.index()];
                let mut set = node_data.local.clone();

                for call in &node_data.calls {
                    for (comp, muta) in graph.nodes[call.target.index()].local.iter() {
                        if call.absorbs.0.contains_key(&comp) {
                            continue;
                        }

                        set.add(comp, muta);
                    }
                }

                graph.nodes[node.index()].local = tcx.arena.alloc(set);
            }
            SccMemberSet::Many(nodes) => {
                todo!();
            }
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
