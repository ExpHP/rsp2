use rsp2_structure::{Coords};
use rsp2_lammps_wrap::{Potential, AtomType, PairStyle, PairCoeff, InitInfo};

pub struct Airebo;
impl Potential for Airebo {
    type Meta = Vec<&'static str>;

    fn molecule_ids(&self, _: &Coords, _: &Vec<&'static str>) -> Option<Vec<usize>> { None }

    fn atom_types(&self, _: &Coords, meta: &Vec<&'static str>) -> Vec<AtomType> {
        meta.iter().map(|&elem| match elem {
            "C" => AtomType::new(1),
            "H" => AtomType::new(2),
            sym => panic!("Unexpected element in Airebo: {}", sym),
        }).collect()
    }

    fn init_info(&self, _: &Coords, _: &Vec<&'static str>) -> InitInfo {
        InitInfo {
            masses: vec![12.01, 1.00794],
            pair_style: PairStyle::named("airebo").arg(3.0).arg(1).arg(1),
            pair_coeffs: vec![
                PairCoeff::new(.., ..).args(&["CH.airebo", "H", "C"]),
            ],
        }
    }
}
