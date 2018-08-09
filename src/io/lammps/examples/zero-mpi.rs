extern crate rsp2_lammps_wrap;
extern crate rsp2_structure;
extern crate rsp2_array_types;
extern crate failure;
extern crate mpi;

use rsp2_structure::{Coords, CoordsKind, Lattice};
use rsp2_array_types::V3;

fn main() -> Result<(), failure::Error> {
    let _universe = ::mpi::initialize().expect("failed to initialize MPI");

    let coords = CoordsKind::Fracs(vec![V3([0.5; 3])]);
    let coords = Coords::new(Lattice::eye(), coords);

    let result = {
        let lock = ::rsp2_lammps_wrap::INSTANCE_LOCK.lock().unwrap();
        ::rsp2_lammps_wrap::LammpsOnDemand::install(|on_demand| {
            let potential = ::rsp2_lammps_wrap::potential::None;
            ::rsp2_lammps_wrap::Builder::new()
                .on_demand(on_demand)
                .stdout(true)
                .build(lock, potential, coords, ()).unwrap()
                .compute_value()
        })
    };

    if let Some(result) = result {
        // root thread
        let value = result?;
        assert_eq!(value, 0.0);
    }
    Ok(())
}
