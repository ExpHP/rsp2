
// This is a reincarnation of assemble.py, in the form of
// a rust library function rather than a CLI utility.

use ::Result;

use ::rsp2_structure::{Coords, Lattice, CoordStructure};
use ::std::path::Path;

pub fn load_layers_yaml<P: AsRef<Path>>(path: P)
-> Result<CoordStructure>
{
    let file = ::std::fs::File::open(path.as_ref())?;
    let cereal = ::serde_yaml::from_reader(file)?;
    assemble_from_cereal(cereal)
}

mod cereal {
    #[derive(Debug, Clone)]
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct Root {
        #[serde(default = "defaults::a")]
        pub a: f64,
        pub lattice: [[f64; 2]; 2],
        pub layer: Vec<Layer>,
        pub layer_sep: Either<f64, Vec<f64>>,
        #[serde(default = "defaults::vacuum_sep")]
        pub vacuum_sep: f64,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    #[serde(untagged)]
    pub enum Either<A, B> { A(A), B(B) }

    #[derive(Debug, Clone)]
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct Layer {
        // NOTE: units of layer lattice
        pub frac_sites: Option<Vec<[f64; 2]>>,
        pub cart_sites: Option<Vec<[f64; 2]>>,
        // NOTE: units of superlattice
        pub frac_lattice: Option<[[f64; 2]; 2]>,
        pub cart_lattice: Option<[[f64; 2]; 2]>,
        #[serde(default = "defaults::layer::transform")]
        pub transform: [[f64; 2]; 2],
        // Number of unique images to generate along each layer lattice vector
        #[serde(default = "defaults::layer::repeat")]
        pub repeat: [i32; 2],
        // Common translation for all positions in layer.
        // NOTE: units of layer lattice
        #[serde(default = "defaults::layer::shift")]
        pub shift: [f64; 2],
    }

    mod defaults {
        pub fn a() -> f64 { 1.0 }
        pub fn vacuum_sep() -> f64 { 10.0 }
        pub mod layer {
            pub fn transform() -> [[f64; 2]; 2] { [[1.0, 0.0], [0.0, 1.0]] }
            pub fn repeat() -> [i32; 2] { [1, 1] }
            pub fn shift() -> [f64; 2] { [0.0, 0.0] }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Units { Cart, Frac }
#[derive(Debug, Clone)]
struct Layer {
    lattice: (Units, Lattice),
    sites: (Units, Vec<[f64; 3]>),
    transform: [[f64; 3]; 3],
    repeat: [i32; 2],
    shift: [f64; 3],
}



fn assemble_from_cereal(cereal: self::cereal::Root) -> Result<CoordStructure>
{Ok({

    // destructure cereal into more useful data
    // FIXME block is too big
    let (full_lattice, layers, layer_zs) = {
        let self::cereal::Root {
            a: lattice_a,
            layer: layers,
            layer_sep, lattice, vacuum_sep,
        } = cereal;

        let layer_seps = match layer_sep {
            self::cereal::Either::A(x) => vec![x; layers.len()],
            self::cereal::Either::B(xs) => {
                ensure!(xs.len() == layers.len() - 1, "wrong number of layer seps");
                xs
            },
        };

        let full_lattice = mat_22_to_33(&lattice);
        let layers = layers.into_iter().map(|layer| {Ok({
            let self::cereal::Layer {
                frac_lattice, frac_sites,
                cart_lattice, cart_sites,
                transform, repeat, shift,
            } = layer;

            fn resolve_units<T>(name: &str, cart: Option<T>, frac: Option<T>) -> Result<(Units, T)>
            {Ok(match (cart, frac) {
                (None, None) => bail!("layer needs one of: frac-{0}, cart-{0}", name),
                (Some(_), Some(_)) => bail!("layer cannot have both of: frac-{0}, cart-{0}", name),
                (None, Some(x)) => (Units::Frac, x),
                (Some(x), None) => (Units::Cart, x),
            })}

            let lattice = match resolve_units("lattice", cart_lattice, frac_lattice)? {
                (units, lattice) => (units, Lattice::new(&mat_22_to_33(&lattice))),
            };
            let sites = match resolve_units("sites", cart_sites, frac_sites)? {
                (units, sites) => (units, vec_2_to_3(&sites)),
            };
            let transform = mat_22_to_33(&transform);
            let shift = [shift[0], shift[1], 0.0];
            Layer { lattice, sites, transform, repeat, shift }
        })}).collect::<Result<Vec<_>>>()?;

        let lattice_c = layer_seps.iter().sum::<f64>() + vacuum_sep;
        let full_lattice = &Lattice::diagonal(&[lattice_a, lattice_a, lattice_c]) * &full_lattice;
        let layer_zs = get_z_positions(&layer_seps, vacuum_sep);
        (full_lattice, layers, layer_zs)
    }; // let (full_lattice, layers, layer_zs) = { ... }


    let mut full_carts = vec![];
    for (layer, layer_z_cart) in layers.into_iter().zip(layer_zs) {
        // cartesian layer lattice
        let lattice = match layer.lattice {
            (Units::Frac, x) => &x * &full_lattice,
            (Units::Cart, x) => x,
        };

        // cartesian sites
        let mut sites = match layer.sites {
            (Units::Frac, x) => Coords::Fracs(x).into_carts(&lattice),
            (Units::Cart, x) => x,
        };

        let mut structure = CoordStructure::new_coords(lattice, Coords::Carts(sites));
        structure.translate_frac(&layer.shift);
        structure.translate_cart(&[0.0, 0.0, layer_z_cart]);
        structure.transform(&layer.transform);

        // generate all unique sites in this layer
        let sc_vec = (layer.repeat[0] as u32, layer.repeat[1] as u32, 1);
        let (structure, _) = ::rsp2_structure::supercell::diagonal(sc_vec, structure);

        full_carts.extend(structure.to_carts());
    }

    CoordStructure::new_coords(full_lattice, Coords::Carts(full_carts))
})}

fn get_z_positions(layer_seps: &[f64], vacuum_sep: f64) -> Vec<f64>
{
    let mut zs = vec![0.0];
    zs.extend(layer_seps.iter().cloned());
    for i in 1..zs.len() {
        zs[i] += zs[i-1];
    }

    // put half of the vacuum separation on either side
    // (i.e. shift by 1/2 vacuum sep)
    for z in &mut zs {
        *z += 0.5 * vacuum_sep;
    }
    zs
}

fn mat_22_to_33(mat: &[[f64; 2]; 2]) -> [[f64; 3]; 3]
{[
    [mat[0][0], mat[0][1], 0.0],
    [mat[1][0], mat[1][1], 0.0],
    [0.0, 0.0, 1.0],
]}

fn vec_2_to_3(xs: &[[f64; 2]]) -> Vec<[f64; 3]>
{ xs.iter().map(|v| [v[0], v[1], 0.0]).collect() }

// fn zip_eq<As, Bs>(a: As, b: Bs) -> ::std::iter::Zip<As::IntoIter, Bs::IntoIter>
// where
//     As: IntoIterator, As::IntoIter: ExactSizeIterator,
//     Bs: IntoIterator, Bs::IntoIter: ExactSizeIterator,
// {
//     let (a, b) = (a.into_iter(), b.into_iter());
//     assert_eq!(a.len(), b.len());
//     a.zip(b)
// }
