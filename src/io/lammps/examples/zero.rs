extern crate rsp2_lammps_wrap;
extern crate rsp2_structure;
extern crate rsp2_array_types;
extern crate failure;

use rsp2_structure::{Coords, CoordsKind, Lattice};
use rsp2_array_types::V3;

fn main() -> Result<(), failure::Error> {
    let coords = CoordsKind::Fracs(vec![V3([0.5; 3])]);
    let coords = Coords::new(Lattice::eye(), coords);
    let value = {
        ::rsp2_lammps_wrap::Builder::new()
            .build(
                ::rsp2_lammps_wrap::INSTANCE_LOCK.lock().unwrap(),
                ::rsp2_lammps_wrap::potential::None,
                coords,
                (),
            ).unwrap()
            .compute_value()?
    };
    assert_eq!(value, 0.0);
    Ok(())
}
