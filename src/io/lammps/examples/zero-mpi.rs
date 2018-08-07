extern crate rsp2_lammps_wrap;
extern crate rsp2_structure;
extern crate rsp2_array_types;
extern crate failure;
extern crate mpi;

use rsp2_structure::{Coords, CoordsKind, Lattice};
use rsp2_array_types::V3;

fn main() -> Result<(), failure::Error> {

    let coords = CoordsKind::Fracs(vec![V3([0.5; 3])]);
    let coords = Coords::new(Lattice::eye(), coords);

    let result = {
        use ::mpi::traits::Communicator;

        let universe = ::mpi::initialize().expect("failed to initialize MPI");
        let world = universe.world();
        let root = world.process_at_rank(0);

        let lock = ::rsp2_lammps_wrap::INSTANCE_LOCK.lock().unwrap();
        let potential = ::rsp2_lammps_wrap::potential::None;
        ::rsp2_lammps_wrap::Builder::new()
            .with_mpi_event_loop(root, |builder| {
                builder
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
