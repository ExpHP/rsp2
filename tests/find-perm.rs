// HACK: serde derives needed by `shared`
extern crate serde;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate rsp2_assert_close;
#[macro_use] extern crate rsp2_util_macros;

use rsp2_soa_ops::Permute;
extern crate rsp2_soa_ops;

use rsp2_structure::{find_perm};
extern crate rsp2_structure;

//use shared::filetypes::PrimitiveNew;
use shared::filetypes::Primitive;

mod shared;
//
//#[test]
//fn test_fix() {
//    let Primitive {
//        cart_rots: _, frac_ops, coords,  masses, displacements,
//    } = Primitive::load("tests/resources/primitive/graphene.json").unwrap();
//
//    let cart_ops = frac_ops.iter().map(|f| f.to_rot().to_cart_op_with_frac_trans(f.to_trans().frac(), coords.lattice())).collect();
//
//    PrimitiveNew {
//        cart_ops, coords, masses, displacements,
//    }.save("tests/resources/primitive/graphene-new.json").unwrap();
//}

#[test]
fn test_graphene() {
    let Primitive {
        cart_ops, coords, ..
    } = Primitive::load("tests/resources/primitive/graphene.json").unwrap();

    let perms = find_perm::of_spacegroup(&coords, &cart_ops, 1e-2).unwrap();
    for (op, perm) in zip_eq!(cart_ops, perms) {
        let transformed = op.transform(&coords);
        let permuted = coords.clone().permuted_by(&perm);

        transformed.check_same_cell_and_order(&permuted, 1e-2 * (1.0 + 1e-7)).unwrap();
    }
}

#[test]
#[should_panic(expected = "looney")]
fn validation_can_fail() {
    let Primitive {
        cart_ops, coords, ..
    } = Primitive::load("tests/resources/primitive/graphene.json").unwrap();

    let mut perms = find_perm::of_spacegroup(&coords, &cart_ops, 1e-2).unwrap();

    // make the result incorrect
    perms[5] = perms[5].clone().shift_right(1);

    for (op, perm) in zip_eq!(cart_ops, perms) {
        let transformed = op.transform(&coords);
        let permuted = coords.clone().permuted_by(&perm);

        transformed.check_same_cell_and_order(&permuted, 1e-2 * (1.0 + 1e-7)).expect("looney");
    }
}
