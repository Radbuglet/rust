use rustc_middle::ty::{self, Mutability, ParamEnv, ScalarInt, Ty, TyCtxt, ValTree};
use rustc_middle::mir::ConstValue;
use rustc_target::abi::FieldsShape;

use crate::const_eval::valtree_to_const_value;
use crate::interpret::Allocation;

pub fn bundle_layout<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> ConstValue<'tcx> {
    let mut items = Vec::new();

    let elem_ty = Ty::new_tup(tcx, &[
        Ty::new_static_str(tcx),
        Ty::new_static_str(tcx),
        tcx.types.u128,
        tcx.types.u128,
        tcx.types.bool,
        tcx.types.usize,
    ]);

    make_bundle_layout_vt(
        tcx,
        param_env,
        ty,
        0,
        &mut ty::BundleItemValueResolver::default(),
        &mut items,
    );

    let arr = valtree_to_const_value(
        tcx,
        param_env.and(Ty::new_array(
            tcx,
            elem_ty,
            items.len() as u64,
        )),
        ValTree::Branch(tcx.arena.alloc_slice(&items)),
    );

    match arr {
        ConstValue::ZeroSized => {
            let layout = tcx.layout_of(param_env.and(elem_ty)).unwrap().layout;
            ConstValue::Slice {
                data: tcx.mk_const_alloc(Allocation::from_bytes(
                    &[],
                    layout.align().pref,
                    Mutability::Mut,
                )),
                meta: 0,
            }
        }
        ConstValue::Indirect { alloc_id, .. } => {
            ConstValue::Slice {
                data: tcx.global_alloc(alloc_id).unwrap_memory(),
                meta: items.len() as u64,
            }
        }
        _ => unreachable!(),
    }
}

fn type_id<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> ValTree<'tcx> {
    ValTree::from_scalar_int(tcx.type_id_hash(ty).as_u128().into())
}

fn type_name<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> ValTree<'tcx> {
    let name = super::type_name::type_name(tcx, ty);

    ValTree::Branch(tcx.arena.alloc_from_iter(
        name.bytes().map(|by| ValTree::from_scalar_int(by.into())),
    ))
}

fn make_bundle_layout_vt<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    base_offset: u64,
    item_values: &mut ty::BundleItemValueResolver<'tcx>,
    children: &mut Vec<ValTree<'tcx>>,
) {
    match ty::ReifiedBundleItemSet::decode(ty) {
        ty::ReifiedBundleItemSet::Ref(_re, muta, did) => {
            let tup_fields = tcx.arena.alloc_slice(&[
                // marker name (&'static str)
                type_name(tcx, ty),
                // pointee name (&'static str)
                type_name(
                    tcx,
                    Ty::new_ref(
                        tcx,
                        tcx.lifetimes.re_erased,
                        tcx.context_ty(did),
                        muta,
                    ),
                ),
                // marker type (u128)
                type_id(tcx, Ty::new_context_marker(tcx, did)),
                // pointee type (u128)
                type_id(tcx, tcx.context_ty(did)),
                // is mutable
                ValTree::from_scalar_int(muta.is_mut().into()),
                // offset
                ValTree::from_scalar_int(
                    ScalarInt::try_from_target_usize(base_offset, tcx).unwrap(),
                ),
            ]);
            children.push(ValTree::Branch(tup_fields));
        }
        ty::ReifiedBundleItemSet::Tuple(tys) => {
            let value_ty = item_values.resolve(tcx, ty);
            let FieldsShape::Arbitrary { offsets, .. } = tcx.layout_of(param_env.and(value_ty))
                .unwrap()
                .layout
                .fields()
            else {
                unreachable!();
            };

            for (ty, offset) in tys.iter().zip(offsets) {
                make_bundle_layout_vt(
                    tcx,
                    param_env,
                    ty,
                    base_offset + offset.bytes(),
                    item_values,
                    children,
                );
            }
        }
        ty::ReifiedBundleItemSet::InferSet(did, re) => {
            make_bundle_layout_vt(
                tcx,
                param_env,
                ty::resolve_infer_bundle_set(tcx, did, re),
                base_offset,
                item_values,
                children,
            );
        }
        ty::ReifiedBundleItemSet::GenericRef(_, _, _)
        | ty::ReifiedBundleItemSet::GenericSet(_) => {
            unreachable!("type is insufficiently monomorphic");
        },
        ty::ReifiedBundleItemSet::Error(_) => {
            unreachable!("evaluating a body which failed type-checking");
        },
    }
}
