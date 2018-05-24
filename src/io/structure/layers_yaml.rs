
// This is a reincarnation of assemble.py, in the form of
// a rust library function rather than a CLI utility.

use ::{FailResult, FailOk};

use ::rsp2_structure::{CoordsKind, Lattice, Coords};
use ::std::io::Read;

use ::rsp2_array_utils::map_arr;
use ::rsp2_array_types::{M22, M33, V2, V3, dot, mat, inv, Unvee};

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

/// A partially assembled structure for which
/// some parameters are still configurable
pub struct Assemble {
    // currently, only normals along a lattice vector are supported, and that
    // lattice vector must be orthogonal to the others.
    normal_axis: usize,

    /// scales each lattice vector.  The value on the normal axis is ignored.
    pub scale: [f64; 3],

    // a lattice where lattice[normal_axis] is a unit vector and the others
    // are orthogonal to it.
    lattice: M33,
    // These are all zero along the normal axis
    fracs_in_plane: Vec<Vec<V3>>, // [layer][atom_in_layer]
    carts_along_normal: Vec<Vec<f64>>, // [layer][atom_in_layer]

    /// separation across periodic boundary (as the center-center distance
    /// between the first layer encountered on either side of the boundary)
    pub vacuum_sep: f64,
    layer_seps: Vec<f64>, // [layer] (length: nlayer - 1)
}

impl Assemble {
    /// Allows setting layer separations (as center-center distances)
    pub fn layer_seps(&mut self) -> &mut [f64]
    { &mut self.layer_seps }

    pub fn num_layer_seps(&self) -> usize
    { self.layer_seps.len() }

    pub fn normal_axis(&self) -> usize
    { self.normal_axis }

    pub fn assemble(&self) -> Coords
    {
        let lattice = {
            let mut scales = self.scale;
            scales[self.normal_axis] = self.get_z_length();

            // (assumption in our use of `scales` below)
            assert!(f64::abs(self.lattice[self.normal_axis].sqnorm() - 1.0) < 1e-5);

            let lattice = Lattice::new(&self.lattice);
            &Lattice::diagonal(&scales) * &lattice
        };
        let layer_zs = self.get_z_positions();

        let mut full_carts = vec![];

        // each layer
        let it = zip_eq!(&self.fracs_in_plane, &self.carts_along_normal, layer_zs);
        for (plane_fracs, z_carts, z_offset) in it {
            // convert the two fractional coords into cartesian
            let mut carts = CoordsKind::Fracs(plane_fracs).to_carts(&lattice);

            // add in the final coordinate, which is already cartesian
            let unit_z = self.lattice[self.normal_axis].unit();
            for (v, z) in zip_eq!(&mut carts, z_carts) {
                *v += (z + z_offset) * unit_z;
            }

            full_carts.extend(carts);
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
        fracs_in_plane.push(structure.to_fracs());
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
    };

    // FIXME: some of the possible errors produced by `from_raw` here are
    //        really indicative of bugs in this function, and should panic
    //        instead of being propagated
    Assemble::from_raw(raw)?
})}

pub struct RawAssemble {
    /// The axis normal to the layers, as the index of a lattice vector.
    pub normal_axis: usize,

    /// A lattice in which the lattice vector for `normal_axis` is orthogonal
    /// to the other two.  The length of the `normal_axis` vector is ignored.
    pub lattice: Lattice,

    /// Fractional coordinates for each layer.  Must all be zero in the normal axis.
    pub fracs_in_plane: Vec<Vec<V3>>, // [layer][atom_in_layer]

    /// Cartesian coordinates along the normal within each layer.  This allows layers
    /// to have a "thickness" to them.
    ///
    /// Only the variation in value within each layer matters.  It is important that,
    /// within each layer, these coordinates form a contiguous image.
    pub carts_along_normal: Vec<Vec<f64>>, // [layer][atom_in_layer]

    // NOTE: bulk has no vacuum sep, and 'nlayer' layer seps.
    //       The current API does not present a nice way of handling this.
    /// Initial separation across periodic boundary.
    pub initial_vacuum_sep: f64,
    /// Initial separations between layers.
    pub initial_layer_seps: Vec<f64>, // [layer] (length: nlayer - 1)

    /// Initial scale factors.  The value on `normal_axis` is ignored.
    pub initial_scale: Option<[f64; 3]>,

    /// Adds a sanity check that each pair of successive atoms along a layer lie within
    /// this distance from each other (with a little extra bit of fuzz). Construction
    /// of an `Assemble` will fail if this is untrue.
    ///
    /// This is to help catch bugs related to accidentally wrapping `carts_along_normal`
    /// across a periodic boundary.
    pub check_intralayer_distance: Option<f64>,
}

impl Assemble {
    fn from_raw(raw: RawAssemble) -> FailResult<Self> {
        let RawAssemble {
            normal_axis, lattice, fracs_in_plane, carts_along_normal,
            initial_vacuum_sep, initial_layer_seps, initial_scale,
            check_intralayer_distance,
        } = raw;
        assert!(normal_axis < 3);

        let other_axes = (0..3).filter(|&k| k != normal_axis).collect::<Vec<_>>();

        // selected vector must be orthogonal to the others.
        // It is normalized so that it can be easily scaled later.
        let lattice = {
            let units = map_arr(lattice.vectors().clone(), |v| v.unit());
            for &k in &other_axes {
                let d = dot(&units[normal_axis], &units[k]);
                ensure!(
                    d.abs() < 1e-6,
                    "Normal vector of Assemble must be orthogonal to the others \
                    (got normalized dot product of {})", d,
                );
            }

            let mut matrix = lattice.matrix().clone();
            matrix[normal_axis] = units[normal_axis];
            matrix
        };

        // while arbitrary vectors are supported *in spirit,* this code is almost
        // always only ever run for cases where it is along the corresponding
        // cartesian axis.  Warn about undertested code paths.
        if other_axes.iter().any(|&k| f64::abs(lattice[normal_axis][k]) > 1e-9) {
            warn_once!(
                "Your layer normal is along a nontrivial direction.  In *theory* rsp2 should \
                be able to handle this, but this functionality is seldom used so it is poorly \
                tested.  Ping Michael if the structure files created by rsp2 don't look right."
            );
        }

        for vs in &fracs_in_plane {
            for v in vs {
                ensure!(v[normal_axis] == 0.0, "Frac positions along normal must be zero!");
            }
        }

        // layer_seps and vacuum_sep will be interpreted as distance from
        // center to center.  Maybe this could be made configurable later.
        //
        // This is accomplished by simply recentering the coordinates prior
        // to construction of the Assemble, which will continue to treat them
        // as basically zero-width when computing layer offsets.
        let mut carts_along_normal = carts_along_normal;
        for v in &mut carts_along_normal {
            if let Some(mut thresh) = check_intralayer_distance {
                thresh *= 1.0 + 1e-8;
                let mut sorted = v.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for w in sorted.windows(2) {
                    let distance = w[1] - w[0];
                    ensure!(
                        distance < thresh,
                        "check_intralayer_distance failed! (distance = {}) Was a layer \
                        wrapped across the periodic boundary?", distance);
                }
            }

            use ::std::f64::INFINITY;
            let min = v.iter().cloned().fold(INFINITY, f64::min);
            let max = v.iter().cloned().fold(-INFINITY, f64::max);
            let center = 0.5 * (min + max);
            assert!(center.is_finite());
            for x in v {
                *x -= center;
            }
        }

        let scale = initial_scale.unwrap_or([1.0; 3]);
        let vacuum_sep = initial_vacuum_sep;
        let layer_seps = initial_layer_seps;
        Ok(Assemble {
            normal_axis, scale, lattice, fracs_in_plane, carts_along_normal,
            vacuum_sep, layer_seps,
        })
    }
}

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
