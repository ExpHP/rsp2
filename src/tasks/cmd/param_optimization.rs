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

use crate::{FailResult};
use crate::potential::{CommonMeta, PotentialBuilder, DiffFn, BondGrad};
use crate::meta::{prelude::*};

use rsp2_minimize::exact_ls::{Value, Golden};
use rsp2_structure::{Lattice, CoordsKind, Coords};
use rsp2_structure::layer::{LayersPerUnitCell, require_simple_axis_normal};
use rsp2_structure_io::assemble::{Assemble, RawAssemble};
use rsp2_array_types::{V3};
use rsp2_tasks_config as cfg;
use stack::{ArrayVec, Vector as StackVector};
use slice_of_array::prelude::*;

pub(crate) fn optimize_layer_parameters(
    settings: &cfg::ScaleRanges,
    pot: &PotentialBuilder,
    mut coords_builder: ScalableCoords,
    meta: CommonMeta,
) -> FailResult<ScalableCoords>
{Ok({
    // Gather a bunch of setter functions and search ranges
    let scalables = {
        let mut scalables = vec![];
        for cfg in &settings.scalables {
            add_scalables(
                cfg,
                &coords_builder,
                |s| scalables.push(s),
            )?;
        }
        scalables
    };

    // Set reasonable values for first iteration.
    for Scalable { spec, setter, .. } in &scalables {
        match *spec {
            cfg::ScalableRange::Exact { value } |
            cfg::ScalableRange::Search { range: _, guess: Some(value) } => {
                setter(&mut coords_builder, value);
            },
            // no guess ==> use whatever the structure had when we got it
            cfg::ScalableRange::Search { range: _, guess: None } => {},
        }
    }

    // optimize them one-by-one.
    //
    // Repeat the whole process repeat_count times.
    // In future iterations, parameters other than the one currently being
    // relaxed may be set to different, better values, which may in turn
    // cause different values to be chosen for the earlier parameters.
    for _ in 0..settings.repeat_count {
        for &Scalable { ref name, ref spec, ref setter } in &scalables {
            trace!("Optimizing {}", name);

            let best = match *spec {
                cfg::ScalableRange::Exact { value } => value,
                cfg::ScalableRange::Search { guess: _, range } => {
                    let best = Golden::new()
                        .stop_condition(&from_json!({"interval-size": 1e-7}))
                        .run(range, |a| {
                            setter(&mut coords_builder, a);

                            // FIXME: I'm not sure why this is a one-off computation; I should try
                            //        reusing a single lammps instance and see what happens.
                            //
                            //        (maybe I had issues with "lost atoms" errors or something)
                            pot.one_off()
                                .compute_value(&coords_builder.construct(), meta.sift())
                                .map(Value)

                        // note: result is Result<Result<_, E>, GoldenSearchError>
                        })??; // ?!??!!!?

                    if let Some(thresh) = settings.warn_threshold {
                        macro_rules! tell {
                            ($($t:tt)*) => {
                                if settings.fail { error!($($t)*); }
                                else { warn!($($t)*); }
                            }
                        }

                        // use signed differences so that all values outside violate the threshold
                        let lo = range.0.min(range.1);
                        let hi = range.0.max(range.1);
                        if (best - range.0).min(range.1 - best) / (range.1 - range.0) < thresh {
                            tell!("Relaxed value of '{}' is suspiciously close to limits!", name);
                            tell!("  lo: {:e}", lo);
                            tell!(" val: {:e}", best);
                            tell!("  hi: {:e}", hi);
                            if settings.fail {
                                bail!("Parameter optimization failed with 'fail = true'");
                            }
                        }
                    }

                    info!("Optimized {}: {} (from {:?})", name, best, range);
                    best
                },
            }; // let best = { ... }

            setter(&mut coords_builder, best);
        } // for ... in optimizables
    } // for ... in repeat_count

    coords_builder
})}

struct Scalable {
    setter: Box<Fn(&mut ScalableCoords, f64)>,
    name: String,
    spec: cfg::ScalableRange,
}

pub enum ScalableCoords {
    KnownLayers {
        layer_builder: Assemble,
    },
    UnknownLayers {
        scales: [f64; 3],
        lattice: Lattice,
        fracs: Vec<V3>,
    },
}

fn add_scalables(
    cfg: &cfg::Scalable,
    structure: &ScalableCoords,
    mut emit: impl FnMut(Scalable),
) -> FailResult<()> {
    // validate
    match (cfg, structure) {
        (
            &cfg::Scalable::Param { axis_mask, .. },
            &ScalableCoords::KnownLayers { ref layer_builder, .. },
        ) => {
            // (this is unfair to bulk...)
            ensure!(
                !axis_mask[layer_builder.normal_axis()].0,
                "Cannot scale the layer axis normal direction with a 'param'"
            );
        },

        (
            &cfg::Scalable::UniformLayerSep { .. },
            &ScalableCoords::UnknownLayers { .. },
        ) | (
            &cfg::Scalable::LayerSeps { .. },
            &ScalableCoords::UnknownLayers { .. },
        ) => bail!("cannot scale layer separations when layers have not been determined"),

        _ => {},
    }

    // obtain data to be used by some scalables
    let mut n_layer_seps = None;
    match structure {
        ScalableCoords::KnownLayers { ref layer_builder, .. } => {
            n_layer_seps = Some(layer_builder.num_layer_seps());
        },
        _ => {},
    }

    match cfg {
        &cfg::Scalable::Param { axis_mask, ref range } => {
            emit(Scalable {
                // FIXME these should get unique names
                name: format!("lattice param"),
                setter: Box::new(move |s, val| {
                    let scales = match s {
                        ScalableCoords::KnownLayers { layer_builder, .. } => &mut layer_builder.scale,
                        ScalableCoords::UnknownLayers { scales, .. } => scales,
                    };

                    for k in 0..3 {
                        if axis_mask[k].0 {
                            scales[k] = val;
                        }
                    }
                }),
                spec: range.clone(),
            });
        },

        &cfg::Scalable::UniformLayerSep { ref range } => {
            // one scalable for all layers
            emit(Scalable {
                name: format!("a uniform layer separation"),
                setter: Box::new(|s, val| match s {
                    ScalableCoords::UnknownLayers { .. } => unreachable!(),
                    ScalableCoords::KnownLayers { layer_builder, .. } => {
                        for x in layer_builder.layer_seps() {
                            *x = val;
                        }
                    },
                }),
                spec: range.clone(),
            });
        },

        &cfg::Scalable::LayerSeps { ref range } => {
            // separate scalables for each layer
            let n_layer_seps = n_layer_seps.expect("BUG!");
            for k in 0..n_layer_seps {
                emit(Scalable {
                    name: format!("layer separation {}", k),
                    setter: Box::new(move |s, val| match s {
                        ScalableCoords::UnknownLayers { .. } => unreachable!(),
                        ScalableCoords::KnownLayers { layer_builder, .. } => {
                            layer_builder.layer_seps()[k] = val;
                        },
                    }),
                    spec: range.clone(),
                });
            }
        },
    }
    Ok(())
}

impl ScalableCoords {
    pub fn construct(&self) -> Coords {
        match self {
            &ScalableCoords::KnownLayers {
                ref layer_builder,
            } => {
                layer_builder.assemble()
            },

            &ScalableCoords::UnknownLayers {
                ref scales, ref lattice, ref fracs,
            } => {
                Coords::new(
                    &Lattice::diagonal(scales) * lattice,
                    CoordsKind::Fracs(fracs.to_vec()),
                )
            },
        }
    }

    pub fn from_unlayered(
        coords: Coords,
    ) -> Self {
        let scales = [1.0; 3];
        let fracs = coords.to_fracs();
        let lattice = coords.lattice().clone();
        ScalableCoords::UnknownLayers { scales, fracs, lattice }
    }

    pub fn from_layer_search_results(
        coords: Coords,
        cfg: &cfg::LayerSearch,
        layers: &LayersPerUnitCell,
    ) -> Self {
        let mut gaps = layers.gaps.clone();
        assert!(
            // LayersPerUnitCell has some important invariants that prevent us
            // from being able to reorder its layers to move the vacuum separation
            // to the end.  It will need to already be there, or the positions will
            // have to be translated. (or LayersPerUnitCell will need to lose some
            // of its invariants)
            gaps.iter().all(|&x| x <= gaps.last().unwrap() * 1.25),
            "This currently only supports vacuum sep being the last gap.",
        );
        let initial_vacuum_sep = gaps.pop().unwrap();
        let initial_layer_seps = gaps;

        let lattice = coords.lattice().clone();
        let axis = require_simple_axis_normal(V3(cfg.normal), &lattice).unwrap();

        let (mut fracs_in_plane, mut carts_along_normal) = (vec![], vec![]);
        for layer_coords in layers.partition_into_contiguous_layers(V3(cfg.normal), coords) {
            let mut fracs = layer_coords.to_fracs();
            for v in &mut fracs {
                v[axis] = 0.0;
            }
            fracs_in_plane.push(fracs);

            let carts = layer_coords.to_carts().into_iter().map(|v| v[axis]).collect();
            carts_along_normal.push(carts);
        }

        let layer_builder = Assemble::from_raw(RawAssemble {
            normal_axis: axis,
            lattice,
            fracs_in_plane,
            carts_along_normal,
            initial_vacuum_sep,
            initial_layer_seps,
            initial_scale: Some([1.0; 3]),
            part: Some(layers.get_part()),
            check_intralayer_distance: Some(cfg.threshold),
        }).unwrap();
        ScalableCoords::KnownLayers { layer_builder }
    }
}

//----------------------------------------------------------------

/// Maps coordinates into a flat representation that includes up to three elements representing
/// the lattice parameters, so that the parameters can be optimized as part of a relaxation.
pub struct RelaxationOptimizationHelper {
    param_primary_axis: ArrayVec<[usize; 3]>,
    axis_param: [Option<usize>; 3],
    // Original input lattice, with the vectors each scaled so that the "primary" axes
    // have a norm of 1.0, and so that axes with the same parameter have vectors that
    // maintain their original proportions with respect to each other.
    normalized_lattice: Lattice,
}

impl RelaxationOptimizationHelper {
    pub fn new(params: &cfg::Parameters, original_lattice: &Lattice) -> Self {
        use rsp2_array_utils::arr_from_fn;

        let mut map = std::collections::BTreeMap::new();
        let mut param_primary_axis = ArrayVec::<[usize; 3]>::new();
        let axis_param: [_; 3] = arr_from_fn(|axis| match params[axis] {
            cfg::Parameter::Param(c) => {
                let param_index = *map.entry(c).or_insert_with(|| {
                    let out = param_primary_axis.len();
                    param_primary_axis.push(axis);
                    out
                });
                Some(param_index)
            },
            cfg::Parameter::One |
            cfg::Parameter::NotPeriodic => None,
        });

        let original_params = Self::_read_lattice_params(&param_primary_axis, original_lattice);
        let mut axis_vectors = *original_lattice.vectors();
        for axis in 0..3 {
            if let Some(param) = axis_param[axis] {
                axis_vectors[axis] /= original_params[param];
            }
        }
        let normalized_lattice = Lattice::from_vectors(&axis_vectors);
        RelaxationOptimizationHelper { param_primary_axis, axis_param, normalized_lattice }
    }

    pub fn num_params(&self) -> usize { self.param_primary_axis.len() }

    /// Produce a flat representation of the coords, with lattice params included as elements.
    pub fn flatten_coords(&self, coords: &Coords) -> Vec<f64> {
        let mut out = coords.to_fracs().flat().to_vec();
        out.extend(self.read_lattice_params(coords.lattice()));
        out
    }

    /// Recover a Coords from the flattened representation
    pub fn unflatten_coords(&self, flat: &[f64]) -> Coords {
        let (flat_fracs, params) = flat.split_at(flat.len() - self.num_params());

        Coords::new(
            self.lattice_with_params(params),
            CoordsKind::Fracs(flat_fracs.nest().to_vec()),
        )
    }

    /// Get the effective gradient on the coordinates in the flattened representation
    /// given gradients with respect to the cartesian bond vectors.
    pub fn flatten_grad(&self, flat: &[f64], bond_grads: &[BondGrad]) -> Vec<f64> {
        #![allow(bad_style)]

        let mut out = vec![0.0; flat.len()];
        { // FIXME: Block will be unnecessary once NLL lands
            let divider = flat.len() - self.num_params();
            let (_, params) = flat.split_at(divider);
            let (d_site_frac, d_param) = out.split_at_mut(divider);
            let d_site_frac: &mut [V3] = d_site_frac.nest_mut();

            let ref lattice = self.lattice_with_params(params);
            let reciprocal_lattice = self.normalized_lattice.reciprocal();
            let diag = V3::from_fn(|axis| self.axis_param[axis].map_or(1.0, |param| params[param]));

            for item in bond_grads {
                // We have a term of the potential that can be expressed as V(cart), where
                //
                // cart =     frac    * lattice
                //      = frac * diag * normalized_lattice
                //      =    scaled   * normalized_lattice
                //
                // cart is the cartesian vector of a bond.
                // frac is the vector in fractional coordinates.
                // diag is a diagonal 3x3 matrix formed from the lattice params.
                //
                // and we want to reformulate it as V(frac, params)

                // using the notation from rsp2-potentials (each row in a_J_b is a gradient of
                // an element of a with respect to b, so that products compose sensibly)
                let cart = item.cart_vector;
                let frac = cart / lattice;
                let term_d_cart = item.grad;

                let cart_J_scaled = reciprocal_lattice.inverse_matrix(); // = L^T

                let term_d_scaled = term_d_cart * cart_J_scaled;

                // scaled_J_frac = diag.t()
                let term_d_frac = term_d_scaled.mul_diag(&diag);

                // frac = fracs[plus_site] - fracs[minus_site]
                d_site_frac[item.plus_site] += term_d_frac;
                d_site_frac[item.minus_site] -= term_d_frac;

                // scaled_J_param = frac * (a diagonal matrix of 1s and 0s for each param)
                for axis in 0..3 {
                    if let Some(param) = self.axis_param[axis] {
                        d_param[param] += term_d_scaled[axis] * frac[axis];
                    }
                }
            }
        }

        out
    }

    fn read_lattice_params(&self, lattice: &Lattice) -> ArrayVec<[f64; 3]> {
        Self::_read_lattice_params(&self.param_primary_axis, lattice)
    }

    fn _read_lattice_params(param_primary_axis: &[usize], lattice: &Lattice) -> ArrayVec<[f64; 3]> {
        let mut out = ArrayVec::<[f64; 3]>::new();
        for &axis in param_primary_axis {
            out.push(lattice.vectors()[axis].norm());
        }
        out
    }

    fn lattice_with_params(&self, params: &[f64]) -> Lattice {
        assert_eq!(params.len(), self.num_params());

        let mut vectors = *self.normalized_lattice.vectors();
        for axis in 0..3 {
            if let Some(param) = self.axis_param[axis] {
                vectors[axis] *= params[param];
            }
        }
        Lattice::from_vectors(&vectors)
    }
}
