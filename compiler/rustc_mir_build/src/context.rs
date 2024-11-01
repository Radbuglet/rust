use std::collections::hash_map::Entry;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::thir::{ContextBinder, StmtId};

#[derive(Default)]
pub(crate) struct ContextBindTracker {
    curr_local_binders: FxHashMap<DefId, StmtId>,
    old_binders: Vec<(DefId, ContextBinder)>,
}

impl ContextBindTracker {
    pub(crate) fn push_scope(&self) -> ContextBindScope {
        ContextBindScope(self.old_binders.len())
    }

    pub(crate) fn pop_scope(&mut self, scope: ContextBindScope) {
        for (item, binder) in self.old_binders.drain((scope.0)..) {
            match binder {
                ContextBinder::FuncEnv => {
                    self.curr_local_binders.remove(&item);
                },
                ContextBinder::LocalBinder(old_stmt) => {
                    self.curr_local_binders.insert(item, old_stmt);
                },
            }
        }
    }

    pub(crate) fn bind(&mut self, item: DefId, stmt: StmtId) {
        let old_binder = match self.curr_local_binders.entry(item) {
            Entry::Occupied(mut entry) => ContextBinder::LocalBinder(entry.insert(stmt)),
            Entry::Vacant(entry) => {
                entry.insert(stmt);
                ContextBinder::FuncEnv
            }
        };

        self.old_binders.push((item, old_binder));
    }

    pub(crate) fn resolve(&self, item: DefId) -> ContextBinder {
        match self.curr_local_binders.get(&item) {
            Some(stmt) => ContextBinder::LocalBinder(*stmt),
            None => ContextBinder::FuncEnv,
        }
    }
}

#[must_use]
pub(crate) struct ContextBindScope(usize);
