
// This is a reincarnation of assemble.py, in the form of
// a rust library function rather than a CLI utility.

use ::{FailResult, FailOk};

use ::rsp2_structure::{CoordsKind, Lattice, Coords};
use ::std::io::Read;

use ::rsp2_array_types::{M22, M33, V2, V3, mat, inv, Unvee};

pub fn load_layers_yaml(mut file: impl Read) -> FailResult<Assemble>
{ _load_layers_yaml(&mut file) }

// Monomorphized to ensure YAML parsing code is generated in this crate
fn _load_layers_yaml(file: &mut Read) -> FailResult<Assemble>
{
    let cereal = ::serde_yaml::from_reader(file)?;
    assemble_from_cereal(cereal).map(|a| a)
}

// FIXME this really doesn't belong here, but it's the easiest reuse of code
pub fn layer_sc_info_from_layers_yaml(mut file: impl Read) -> FailResult<Vec<(M33<i32>, [u32; 3], usize)>>
{ _layer_sc_info_from_layers_yaml(&mut file) }

// Monomorphized to ensure YAML parsing code is generated in this crate
fn _layer_sc_info_from_layers_yaml(file: &mut Read) -> FailResult<Vec<(M33<i32>, [u32; 3], usize)>>
{
    let cereal = ::serde_yaml::from_reader(file)?;
    layer_sc_info_from_cereal(cereal)
}

/// A partially assembled structure for which
/// some parameters are still configurable
pub struct Assemble {
    /// scales the x and y axes
    pub scale: f64,
    /// separation across periodic boundary
    pub vacuum_sep: f64,
    // a lattice with a dummy z length, and without 'scale' taken into account
    lattice: M22,
    frac_sites: Vec<Vec<V2>>,
    layer_seps: Vec<f64>,
}

impl Assemble {
    pub fn layer_seps(&mut self) -> &mut [f64]
    { &mut self.layer_seps }

    pub fn assemble(&self) -> Coords
    {
        let lattice = {
            let scales = [self.scale, self.scale, self.get_z_length()];
            let lattice = Lattice::new(&m22_to_m33(&self.lattice));
            &Lattice::diagonal(&scales) * &lattice
        };
        let layer_zs = self.get_z_positions();

        let mut full_carts = vec![];
        for (xy_fracs, z_cart) in self.frac_sites.iter().zip(layer_zs) {
            let mut structure =
                Coords::new(lattice.clone(), CoordsKind::Fracs(v2_to_v3(xy_fracs)));

            structure.translate_cart(&V3([0.0, 0.0, z_cart]));
            full_carts.extend(structure.to_carts());
        }

        Coords::new(lattice, CoordsKind::Carts(full_carts))
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

            pub fn transform() -> M22 { mat::eye() }
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

    let mut frac_sites = vec![];
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
        let (structure, _) = ::rsp2_structure::supercell::diagonal(layer.repeat).build(structure);

        // put them in frac coords for the full lattice
        let mut structure = Coords::new(
            Lattice::new(&full_lattice),
            CoordsKind::Carts(structure.to_carts()),
        );
        // FIXME this reduction is just a bandaid for the above-mentioned issue.
        //       (taking unique positions in the diagonal layer supercells and mapping
        //        them into the cell that we generally use for the structure)
        structure.reduce_positions();
        frac_sites.push(v3_to_v2(&structure.to_fracs()));
    }

    Assemble {
        scale: lattice_a,
        lattice: m33_to_m22(&full_lattice),
        frac_sites, layer_seps, vacuum_sep,
    }
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

fn m33_to_m22(mat: &M33) -> M22
{
    let [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = mat.unvee();
    assert_eq!(m02, 0.0);
    assert_eq!(m12, 0.0);
    assert_eq!(m20, 0.0);
    assert_eq!(m21, 0.0);
    assert_eq!(m22, 1.0);
    mat::from_array([
        [m00, m01],
        [m10, m11],
    ])
}

fn v2_to_v3(xs: &[V2]) -> Vec<V3>
{ xs.unvee().iter().map(|&[x, y]| V3([x, y, 0.0])).collect() }

fn v3_to_v2(xs: &[V3]) -> Vec<V2>
{ xs.unvee().iter().map(|&[x, y, z]| {
    assert_eq!(z, 0.0);
    V2([x, y])
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
