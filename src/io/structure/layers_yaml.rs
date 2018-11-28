/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

// This is a reincarnation of assemble.py, in the form of
// a rust library function rather than a CLI utility.

use crate::{FailResult, FailOk};
use crate::assemble::{RawAssemble, Assemble};

use ::rsp2_structure::{CoordsKind, Lattice, Coords};
use ::std::io::Read;

use ::rsp2_array_types::{M22, M33, mat, V2, V3, inv, Unvee};

pub fn load(mut file: impl Read) -> FailResult<Assemble>
{ _load(&mut file) }

// Monomorphized to ensure YAML parsing code is generated in this crate
fn _load(file: &mut Read) -> FailResult<Assemble>
{
    let cereal = ::serde_yaml::from_reader(file)?;
    assemble_from_cereal(cereal).map(|a| a)
}

// FIXME this really doesn't belong here, but it's the easiest reuse of code
pub fn load_layer_sc_info(mut file: impl Read) -> FailResult<Vec<(M33<i32>, [u32; 3], usize)>>
{ _load_layer_sc_info(&mut file) }

// Monomorphized to ensure YAML parsing code is generated in this crate
fn _load_layer_sc_info(file: &mut Read) -> FailResult<Vec<(M33<i32>, [u32; 3], usize)>>
{
    let cereal = ::serde_yaml::from_reader(file)?;
    layer_sc_info_from_cereal(cereal)
}

mod cereal {
    use super::*;

    #[derive(Debug, Clone)]
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct Root {
        #[serde(default = "defaults::a")]
        pub a: f64,
        pub lattice: M22,
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
        pub frac_sites: Option<Vec<V2>>,
        pub cart_sites: Option<Vec<V2>>,
        // NOTE: units of superlattice
        pub frac_lattice: Option<M22>,
        pub cart_lattice: Option<M22>,
        #[serde(default = "defaults::layer::transform")]
        pub transform: M22,
        // Number of unique images to generate along each layer lattice vector
        #[serde(default = "defaults::layer::repeat")]
        pub repeat: [u32; 2],
        // Common translation for all positions in layer.
        // NOTE: units of layer lattice
        #[serde(default = "defaults::layer::shift")]
        pub shift: V2,
    }

    mod defaults {
        use super::*;

        pub fn a() -> f64 { 1.0 }
        pub fn vacuum_sep() -> f64 { 10.0 }
        pub fn layer_sep() -> Either<f64, Vec<f64>> { Either::A(1.0) }

        pub mod layer {
            use super::*;

            pub fn transform() -> M22 { M22::eye() }
            pub fn repeat() -> [u32; 2] { [1, 1] }
            pub fn shift() -> V2 { V2([0.0, 0.0]) }
        }
    }
}

// intermediate form of data that is easier to work with than cereal
mod middle {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Layers {
        pub full_lattice: M33,
        pub layers: Vec<Layer>,
        pub layer_seps: Vec<f64>,
        pub vacuum_sep: f64,
        pub lattice_a: f64,
    }

    #[derive(Debug, Clone)]
    pub struct Layer {
        pub frac_lattice: ::rsp2_structure::Lattice,
        pub cart_lattice: ::rsp2_structure::Lattice,
        pub cart_sites: Vec<V3>,
        pub transform: M33,
        pub repeat: [u32; 3],
        pub shift: V3,
    }
}

fn interpret_cereal(cereal: self::cereal::Root) -> FailResult<middle::Layers>
{Ok({
    let self::cereal::Root {
        a: lattice_a,
        layer: layers,
        lattice: full_lattice,
        layer_sep, vacuum_sep,
    } = cereal;
    let full_lattice = m22_to_m33(&full_lattice);

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
        -> FailResult<(Units, T)>
        {Ok(match (cart, frac) {
            (None, None) => bail!("layer needs one of: frac-{0}, cart-{0}", name),
            (Some(_), Some(_)) => bail!("layer cannot have both of: frac-{0}, cart-{0}", name),
            (None, Some(x)) => (Units::Frac, x),
            (Some(x), None) => (Units::Cart, x),
        })}

        let (cart_lattice, frac_lattice) = match resolve_units("lattice", cart_lattice, frac_lattice)? {
            (units, x) => {
                let x = Lattice::new(&m22_to_m33(&x));
                match units {
                    Units::Frac => (&x * &full_lattice, x),
                    Units::Cart => (x.clone(), &x * &inv(&full_lattice)),
                }
            }
        };
        let cart_sites = match resolve_units("sites", cart_sites, frac_sites)? {
            (units, x) => {
                let x = v2_to_v3(&x);
                match units {
                    Units::Frac => CoordsKind::Fracs(x).into_carts(&cart_lattice),
                    Units::Cart => x,
                }
            }
        };

        let transform = m22_to_m33(&transform);
        let shift = V3([shift[0], shift[1], 0.0]);
        let repeat = [repeat[0], repeat[1], 1];
        middle::Layer { cart_lattice, frac_lattice, cart_sites, transform, repeat, shift }
    })}).collect::<FailResult<Vec<_>>>()?;

    middle::Layers { lattice_a, full_lattice, layers, layer_seps, vacuum_sep }
})}

fn assemble_from_cereal(cereal: self::cereal::Root) -> FailResult<Assemble>
{Ok({

    let middle::Layers {
        lattice_a, layers, full_lattice, vacuum_sep, layer_seps,
    } = interpret_cereal(cereal)?;

    let mut fracs_in_plane = vec![];
    for layer in layers.into_iter() {
        let lattice = layer.cart_lattice.clone();
        let sites = layer.cart_sites.clone();

        let mut structure = Coords::new(lattice, CoordsKind::Carts(sites));
        structure.translate_frac(&layer.shift);
        structure.transform(&layer.transform);

        // generate all unique sites in this layer
        // FIXME this causes a different-but-equivalent diagonal supercell
        //       to be used for the layer in some places even though the cell we *use* is
        //       not diagonal.  This is a huge footgun, and may even have been what caused
        //       the band unfolding code to give such miserable results at K.
        //
        //       it would be better to invert/transpose the fractional lattice in the file
        //       to get the integer sc matrices, and somehow verify that the 'repeat' field
        //       is correct.  Or just ignore the 'repeat' field and do HNF reduction to find
        //       a set of periods (but that feels wasteful).
        let (superstructure, _) = ::rsp2_structure::supercell::diagonal(layer.repeat).build(&structure);

        // put them in frac coords for the full lattice
        let mut superstructure = Coords::new(
            Lattice::new(&full_lattice),
            CoordsKind::Carts(superstructure.to_carts()),
        );
        // FIXME this reduction is just a bandaid for the above-mentioned issue.
        //       (taking unique positions in the diagonal layer supercells and mapping
        //        them into the cell that we generally use for the structure)
        superstructure.reduce_positions();
        fracs_in_plane.push(superstructure.to_fracs());
    }

    let carts_along_normal = {
        fracs_in_plane.iter()
            .map(|vec| vec![0.0; vec.len()]) // uniform value for zero width
            .collect()
    };

    let raw = RawAssemble {
        normal_axis: 2,
        lattice: Lattice::new(&full_lattice),
        initial_scale: Some([lattice_a, lattice_a, 0.0]),
        fracs_in_plane,
        carts_along_normal,
        initial_layer_seps: layer_seps,
        initial_vacuum_sep: vacuum_sep,
        check_intralayer_distance: None,
        part: None,
    };

    // FIXME: some of the possible errors produced by `from_raw` here are
    //        really indicative of bugs in this function, and should panic
    //        instead of being propagated
    Assemble::from_raw(raw)?
})}

// FIXME this really doesn't belong here, but it's the easiest reuse of code
fn layer_sc_info_from_cereal(cereal: cereal::Root) -> FailResult<Vec<(M33<i32>, [u32; 3], usize)>>
{Ok({

    let middle::Layers {
        lattice_a: _, vacuum_sep: _, layer_seps: _, full_lattice: _,
        layers,
    } = interpret_cereal(cereal)?;

    layers.into_iter().map(|layer| FailOk({
        let matrix = *layer.frac_lattice.inverse_matrix();
        let matrix = matrix.try_map(|x| FailOk({
            let r = x.round();
            ensure!((x - r).abs() <= 1e-3,
                "layers file does not look like a true supercell of each layer (error est: {:e})",
                (x - r).abs());
            r as i32
        }))?;
        let periods = layer.repeat;
        let primitive_atom_count = layer.cart_sites.len();

        (matrix, periods, primitive_atom_count)
    })).collect::<FailResult<Vec<_>>>()?
})}

fn m22_to_m33(mat: &M22) -> M33
{
    let [[m00, m01], [m10, m11]] = mat.unvee();
    mat::from_array([
        [m00, m01, 0.0],
        [m10, m11, 0.0],
        [0.0, 0.0, 1.0],
    ])
}

fn v2_to_v3(xs: &[V2]) -> Vec<V3>
{ xs.unvee().iter().map(|&[x, y]| V3([x, y, 0.0])).collect() }
