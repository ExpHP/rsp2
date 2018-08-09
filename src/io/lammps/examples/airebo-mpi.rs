extern crate rsp2_lammps_wrap;
extern crate rsp2_structure;
extern crate rsp2_array_types;
extern crate failure;
#[macro_use]
extern crate rsp2_assert_close;
extern crate mpi;

use rsp2_structure::{supercell, Coords, CoordsKind, Lattice};
use rsp2_array_types::V3;

#[path = "potentials/airebo.rs"]
mod airebo;
use airebo::Airebo;

fn main() -> Result<(), failure::Error> {
    let _universe = ::mpi::initialize().expect("failed to initialize MPI");

    let unit_coords = Coords::new(
        Lattice::orthorhombic(4.2, 4.2, 2.5579182965),
        CoordsKind::Carts(vec![
            V3([ 0.0000000000,  0.0000000000, 0.0000000000]),
            V3([ 0.9010066786, -0.6310205743, 0.0000000000]),
            V3([-0.9010066786, -0.6310205743, 0.0000000000]),
            V3([ 0.0000000000,  0.8470061967, 1.2789591482]),
            V3([ 0.9010066786,  1.4780267710, 1.2789591482]),
            V3([-0.9010066786,  1.4780267710, 1.2789591482]),
        ]),
    );
    let unit_meta = vec!["C", "H", "H", "C", "H", "H"];

    let (super_coords, sc) = supercell::diagonal([5, 5, 5]).build(&unit_coords);
    let super_meta = sc.replicate(&unit_meta);

    let result = {
        let lock = ::rsp2_lammps_wrap::INSTANCE_LOCK.lock().unwrap();
        ::rsp2_lammps_wrap::LammpsOnDemand::install(|on_demand| {
            ::rsp2_lammps_wrap::Builder::new()
                .on_demand(on_demand)
                .stdout(true)
                .build(lock, Airebo, super_coords, super_meta).unwrap()
                .compute_value()
        })

    };

    if let Some(result) = result {
        // root process
        let value = result?;
        assert_close!(value, 4.798422121393545 * sc.num_cells() as f64);
    }

    Ok(())
}

