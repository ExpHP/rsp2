#[macro_use]
extern crate rsp2_util_macros;

use rsp2_soa_ops::Permute;
extern crate rsp2_soa_ops;

use rsp2_structure::{find_perm};
extern crate rsp2_structure;

use rsp2_integration_test::filetypes::Primitive;
extern crate rsp2_integration_test;

#[test]
fn test_graphene() {
    let Primitive {
        cart_ops, coords, ..
    } = Primitive::load("tests/resources/primitive/graphene.json").unwrap();

    let coperms = find_perm::spacegroup_coperms(&coords, &cart_ops, 1e-2).unwrap();
    for (op, coperm) in zip_eq!(cart_ops, coperms) {
        let transformed = op.transform(&coords);
        let permuted = coords.clone().permuted_by(&coperm);

        transformed.check_same_cell_and_order(&permuted, 1e-2 * (1.0 + 1e-7)).unwrap();
    }
}

#[test]
#[should_panic(expected = "looney")]
fn validation_can_fail() {
    let Primitive {
        cart_ops, coords, ..
    } = Primitive::load("tests/resources/primitive/graphene.json").unwrap();

    let mut coperms = find_perm::spacegroup_coperms(&coords, &cart_ops, 1e-2).unwrap();

    // make the result incorrect
    coperms[5] = coperms[5].clone().shift_right(1);

    for (op, coperm) in zip_eq!(cart_ops, coperms) {
        let transformed = op.transform(&coords);
        let permuted = coords.clone().permuted_by(&coperm);

        transformed.check_same_cell_and_order(&permuted, 1e-2 * (1.0 + 1e-7)).expect("looney");
    }
}
