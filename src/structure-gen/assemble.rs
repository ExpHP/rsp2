
// This is a reincarnation of assemble.py, in the form of
// a rust library function rather than a CLI utility.

use ::{Result, ok};

use ::rsp2_structure::{Coords, Lattice, CoordStructure};
use ::rsp2_array_utils::inv;
use ::std::io::Read;

pub fn load_layers_yaml<R: Read>(file: R) -> Result<Assemble>
{
    let cereal = ::serde_yaml::from_reader(file)?;
    assemble_from_cereal(cereal).map(|a| a)
}

// FIXME this really doesn't belong here, but it's the easiest reuse of code
pub fn layer_sc_info_from_layers_yaml<R: Read>(file: R) -> Result<Vec<([[i32; 3]; 3], [u32; 3], usize)>>
{
    let cereal = ::serde_yaml::from_reader(file)?;
    layer_sc_info_from_cereal(cereal).map(|a| a)
}

/// A partially assembled structure for which
/// some parameters are still configurable
pub struct Assemble {
    /// scales the x and y axes
    pub scale: f64,
    /// separation across periodic boundary
    pub vacuum_sep: f64,
    // a lattice with a dummy z length, and without 'scale' taken into account
    lattice: [[f64; 2]; 2],
    frac_sites: Vec<Vec<[f64; 2]>>,
    layer_seps: Vec<f64>,
}

impl Assemble {
    pub fn layer_seps(&mut self) -> &mut [f64]
    { &mut self.layer_seps }

    pub fn assemble(&self) -> CoordStructure
    {
        let lattice = {
            let scales = [self.scale, self.scale, self.get_z_length()];
            let lattice = Lattice::new(&mat_22_to_33(&self.lattice));
            &Lattice::diagonal(&scales) * &lattice
        };
        let layer_zs = self.get_z_positions();

        let mut full_carts = vec![];
        for (xy_fracs, z_cart) in self.frac_sites.iter().zip(layer_zs) {
            let mut structure =
                CoordStructure::new_coords(lattice.clone(), Coords::Fracs(vec_2_to_3(xy_fracs)));

            structure.translate_cart(&[0.0, 0.0, z_cart]);
            full_carts.extend(structure.to_carts());
        }

        CoordStructure::new_coords(lattice, Coords::Carts(full_carts))
    }

    fn get_z_length(&self) -> f64
    { self.layer_seps.iter().sum::<f64>() + self.vacuum_sep }

    fn get_z_positions(&self) -> Vec<f64>
    {
        let mut zs = vec![0.0];
        zs.extend(self.layer_seps.iter().cloned());
        for i in 1..zs.len() {
            zs[i] += zs[i-1];
        }

        // put half of the vacuum separation on either side
        // (i.e. shift by 1/2 vacuum sep)
        for z in &mut zs {
            *z += 0.5 * self.vacuum_sep;
        }
        zs
    }
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
        #[serde(default = "defaults::layer_sep")]
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
        pub repeat: [u32; 2],
        // Common translation for all positions in layer.
        // NOTE: units of layer lattice
        #[serde(default = "defaults::layer::shift")]
        pub shift: [f64; 2],
    }

    mod defaults {
        use super::*;
        pub fn a() -> f64 { 1.0 }
        pub fn vacuum_sep() -> f64 { 10.0 }
        pub fn layer_sep() -> Either<f64, Vec<f64>> { Either::A(1.0) }
        pub mod layer {
            pub fn transform() -> [[f64; 2]; 2] { [[1.0, 0.0], [0.0, 1.0]] }
            pub fn repeat() -> [u32; 2] { [1, 1] }
            pub fn shift() -> [f64; 2] { [0.0, 0.0] }
        }
    }
}

// intermediate form of data that is easier to work with than cereal
mod middle {
    #[derive(Debug, Clone)]
    pub struct Layers {
        pub full_lattice: [[f64; 3]; 3],
        pub layers: Vec<Layer>,
        pub layer_seps: Vec<f64>,
        pub vacuum_sep: f64,
        pub lattice_a: f64,
    }

    #[derive(Debug, Clone)]
    pub struct Layer {
        pub frac_lattice: ::rsp2_structure::Lattice,
        pub cart_lattice: ::rsp2_structure::Lattice,
        pub cart_sites: Vec<[f64; 3]>,
        pub transform: [[f64; 3]; 3],
        pub repeat: [u32; 3],
        pub shift: [f64; 3],
    }
}

fn interpret_cereal(cereal: self::cereal::Root) -> Result<middle::Layers>
{Ok({
    let self::cereal::Root {
        a: lattice_a,
        layer: layers,
        lattice: full_lattice,
        layer_sep, vacuum_sep,
    } = cereal;
    let full_lattice = mat_22_to_33(&full_lattice);

    let layer_seps = match layer_sep {
        self::cereal::Either::A(x) => vec![x; layers.len() - 1],
        self::cereal::Either::B(xs) => {
            ensure!(xs.len() == layers.len() - 1, "wrong number of layer seps");
            xs
        },
    };

    let layers = layers.into_iter().map(|layer| {Ok({
        let self::cereal::Layer {
            frac_lattice, frac_sites,
            cart_lattice, cart_sites,
            transform, repeat, shift,
        } = layer;

        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub enum Units { Cart, Frac }

        fn resolve_units<T>(name: &str, cart: Option<T>, frac: Option<T>)
        -> Result<(Units, T)>
        {Ok(match (cart, frac) {
            (None, None) => bail!("layer needs one of: frac-{0}, cart-{0}", name),
            (Some(_), Some(_)) => bail!("layer cannot have both of: frac-{0}, cart-{0}", name),
            (None, Some(x)) => (Units::Frac, x),
            (Some(x), None) => (Units::Cart, x),
        })}

        let (cart_lattice, frac_lattice) = match resolve_units("lattice", cart_lattice, frac_lattice)? {
            (units, x) => {
                let x = Lattice::new(&mat_22_to_33(&x));
                match units {
                    Units::Frac => (&x * &full_lattice, x),
                    Units::Cart => (x.clone(), &x * &inv(&full_lattice)),
                }
            }
        };
        let cart_sites = match resolve_units("sites", cart_sites, frac_sites)? {
            (units, x) => {
                let x = vec_2_to_3(&x);
                match units {
                    Units::Frac => Coords::Fracs(x).into_carts(&cart_lattice),
                    Units::Cart => x,
                }
            }
        };

        let transform = mat_22_to_33(&transform);
        let shift = [shift[0], shift[1], 0.0];
        let repeat = [repeat[0], repeat[1], 1];
        middle::Layer { cart_lattice, frac_lattice, cart_sites, transform, repeat, shift }
    })}).collect::<Result<Vec<_>>>()?;

    middle::Layers { lattice_a, full_lattice, layers, layer_seps, vacuum_sep }
})}

fn assemble_from_cereal(cereal: self::cereal::Root) -> Result<Assemble>
{Ok({

    let middle::Layers {
        lattice_a, layers, full_lattice, vacuum_sep, layer_seps,
    } = interpret_cereal(cereal)?;

    let mut frac_sites = vec![];
    for layer in layers.into_iter() {
        let lattice = layer.cart_lattice.clone();
        let sites = layer.cart_sites.clone();

        let mut structure = CoordStructure::new_coords(lattice, Coords::Carts(sites));
        structure.translate_frac(&layer.shift);
        structure.transform(&layer.transform);

        // generate all unique sites in this layer
        let sc_vec = (layer.repeat[0], layer.repeat[1], layer.repeat[2]);
        let (structure, _) = ::rsp2_structure::supercell::diagonal(sc_vec, structure);

        // put them in frac coords for the full lattice
        let mut structure = CoordStructure::new_coords(
            Lattice::new(&full_lattice),
            Coords::Carts(structure.to_carts()),
        );
        structure.reduce_positions();
        frac_sites.push(vec_3_to_2(&structure.to_fracs()));
    }

    Assemble {
        scale: lattice_a,
        lattice: mat_33_to_22(&full_lattice),
        frac_sites, layer_seps, vacuum_sep,
    }
})}

// FIXME this really doesn't belong here, but it's the easiest reuse of code
fn layer_sc_info_from_cereal(cereal: cereal::Root) -> Result<Vec<([[i32; 3]; 3], [u32; 3], usize)>>
{Ok({

    let middle::Layers {
        lattice_a: _, vacuum_sep: _, layer_seps: _, full_lattice: _,
        layers,
    } = interpret_cereal(cereal)?;

    layers.into_iter().map(|layer| ok({
        let matrix = *layer.frac_lattice.inverse_matrix();
        let matrix = ::rsp2_array_utils::try_map_mat(matrix, |x| ok({
            let r = x.round();
            ensure!((x - r).abs() <= 1e-3,
                "layers file does not look like a true supercell of each layer (error est: {:e})",
                (x - r).abs());
            r as i32
        }))?;
        let periods = layer.repeat;
        let primitive_atom_count = layer.cart_sites.len();

        (matrix, periods, primitive_atom_count)
    })).collect::<Result<Vec<_>>>()?
})}

fn mat_22_to_33(mat: &[[f64; 2]; 2]) -> [[f64; 3]; 3]
{[
    [mat[0][0], mat[0][1], 0.0],
    [mat[1][0], mat[1][1], 0.0],
    [0.0, 0.0, 1.0],
]}

fn mat_33_to_22(mat: &[[f64; 3]; 3]) -> [[f64; 2]; 2]
{
    assert_eq!(mat[0][2], 0.0);
    assert_eq!(mat[1][2], 0.0);
    assert_eq!(mat[2][0], 0.0);
    assert_eq!(mat[2][1], 0.0);
    assert_eq!(mat[2][2], 1.0);
    [
        [mat[0][0], mat[0][1]],
        [mat[1][0], mat[1][1]],
    ]
}

fn vec_2_to_3(xs: &[[f64; 2]]) -> Vec<[f64; 3]>
{ xs.iter().map(|v| [v[0], v[1], 0.0]).collect() }

fn vec_3_to_2(xs: &[[f64; 3]]) -> Vec<[f64; 2]>
{ xs.iter().map(|v| {
    assert_eq!(v[2], 0.0);
    [v[0], v[1]]
}).collect() }

// fn zip_eq<As, Bs>(a: As, b: Bs) -> ::std::iter::Zip<As::IntoIter, Bs::IntoIter>
// where
//     As: IntoIterator, As::IntoIter: ExactSizeIterator,
//     Bs: IntoIterator, Bs::IntoIter: ExactSizeIterator,
// {
//     let (a, b) = (a.into_iter(), b.into_iter());
//     assert_eq!(a.len(), b.len());
//     a.zip(b)
// }
