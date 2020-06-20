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
use crate::potential::{CommonMeta, PotentialBuilder, DiffFn, BondGrad, BondDiffFn};
use crate::meta::{prelude::*};

use rsp2_minimize::exact_ls::{Value, Golden};
use rsp2_structure::{Lattice, CoordsKind, Coords};
use rsp2_structure::layer::{LayersPerUnitCell, require_simple_axis_normal};
use rsp2_structure_io::assemble::{Assemble, RawAssemble};
use rsp2_array_types::{V3, M22};
use rsp2_tasks_config as cfg;
use stack::{ArrayVec, Vector as StackVector};
use slice_of_array::prelude::*;

pub(crate) fn optimize_layer_parameters(
    settings: &cfg::ScaleRanges,
    pot: &dyn PotentialBuilder,
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
            let best = match *spec {
                cfg::ScalableRange::Exact { value } => {
                    trace!("Fixing {} at {}", name, value);
                    value
                },
                cfg::ScalableRange::Search { guess: _, range } => {
                    trace!("Optimizing {}", name);
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
    setter: Box<dyn Fn(&mut ScalableCoords, f64)>,
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

#[derive(Clone)]
pub enum RelaxationOptimizationHelper {
    ParamBased(ParamOptimizationHelper),
    LatticeBased(LatticeOptimizationHelper),
}

/// Maps coordinates into a flat representation that includes up to three elements representing
/// the lattice parameters, so that the parameters can be optimized as part of a relaxation.
#[derive(Clone)]
pub struct ParamOptimizationHelper {
    param_primary_axis: ArrayVec<[usize; 3]>,
    axis_param: [Option<usize>; 3],
    // Original input lattice, with the vectors each scaled so that the "primary" axes
    // have a norm of 1.0, and so that axes with the same parameter have vectors that
    // maintain their original proportions with respect to each other.
    normalized_lattice: Lattice,
}

#[derive(Clone)]
pub struct LatticeOptimizationHelper {
    original_lattice: Lattice,
}

impl RelaxationOptimizationHelper {
    /// Produce a flat representation of the coords, with lattice params included as elements.
    pub fn flatten_coords(&self, coords: &Coords) -> Vec<f64> {
        match self {
            RelaxationOptimizationHelper::ParamBased(helper) => helper.flatten_coords(coords),
            RelaxationOptimizationHelper::LatticeBased(helper) => helper.flatten_coords(coords),
        }
    }

    /// Recover a Coords from the flattened representation
    pub fn unflatten_coords(&self, flat: &[f64]) -> Coords {
        match self {
            RelaxationOptimizationHelper::ParamBased(helper) => helper.unflatten_coords(flat),
            RelaxationOptimizationHelper::LatticeBased(helper) => helper.unflatten_coords(flat),
        }
    }

    /// Get the effective gradient on the coordinates in the flattened representation
    /// given gradients with respect to the cartesian bond vectors.
    pub fn flatten_grad(&self, flat: &[f64], bond_grads: &[BondGrad]) -> Vec<f64> {
        match self {
            RelaxationOptimizationHelper::ParamBased(helper) => helper.flatten_grad(flat, bond_grads),
            RelaxationOptimizationHelper::LatticeBased(helper) => helper.flatten_grad(flat, bond_grads),
        }
    }

    /// Get the gradient with respect to positions in cartesian space,
    /// and with respect to each lattice parameter.
    pub fn unflatten_grad(&self, coords: &[f64], grad: &[f64]) -> (Vec<V3>, ArrayVec<[f64; 4]>) {
        match self {
            RelaxationOptimizationHelper::ParamBased(helper) => helper.unflatten_grad(coords, grad),
            RelaxationOptimizationHelper::LatticeBased(helper) => helper.unflatten_grad(coords, grad),
        }
    }
}

impl ParamOptimizationHelper {
    pub fn new(params: &cfg::Parameters, original_lattice: &Lattice) -> Self {
        let mut map = std::collections::BTreeMap::new();
        let mut param_primary_axis = ArrayVec::<[usize; 3]>::new();
        let axis_param: [_; 3] = V3::from_fn(|axis| match params[axis] {
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
        }).0;

        let original_params = Self::_read_lattice_params(&param_primary_axis, original_lattice);
        let mut axis_vectors = *original_lattice.vectors();
        for axis in 0..3 {
            if let Some(param) = axis_param[axis] {
                axis_vectors[axis] /= original_params[param];
            }
        }
        let normalized_lattice = Lattice::from_vectors(&axis_vectors);
        ParamOptimizationHelper { param_primary_axis, axis_param, normalized_lattice }
    }

    pub fn num_params(&self) -> usize { self.param_primary_axis.len() }

    /// Split a slice into the parts corresponding to fractional coords, and the part corresponding
    /// to the parameters.
    pub fn split<'b, T>(&self, slice: &'b [T]) -> (&'b [V3<T>], &'b [T]) {
        let (frac, params) = slice.split_at(slice.len() - self.num_params());
        (frac.nest(), params)
    }

    pub fn split_mut<'b, T>(&self, slice: &'b mut [T]) -> (&'b mut [V3<T>], &'b mut [T]) {
        let div = slice.len() - self.num_params();
        let (frac, params) = slice.split_at_mut(div);
        (frac.nest_mut(), params)
    }

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
        let mut out = vec![0.0; flat.len()];
        let (_, params) = self.split(flat);
        let (d_site_frac, d_param) = self.split_mut(&mut out);

        let ref lattice = self.lattice_with_params(params);
        let ref recip_normalized_lattice = self.normalized_lattice.reciprocal();
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

            // gradients transform by the reciprocal lattice
            let term_d_scaled = term_d_cart / recip_normalized_lattice;

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

        out
    }

    /// Get the gradient with respect to positions in cartesian space,
    /// and with respect to each lattice parameter.
    pub fn unflatten_grad(&self, coords: &[f64], grad: &[f64]) -> (Vec<V3>, ArrayVec<[f64; 4]>) {
        let (_, params) = self.split(coords);
        let (d_fracs, d_params) = self.split(grad);

        // gradients transform as reciprocal vectors
        let recip_lattice = self.lattice_with_params(params).reciprocal();
        let d_carts = CoordsKind::Fracs(d_fracs).to_carts(&recip_lattice);
        (d_carts, d_params.iter().cloned().collect())
    }

    fn read_lattice_params(&self, lattice: &Lattice) -> ArrayVec<[f64; 4]> {
        Self::_read_lattice_params(&self.param_primary_axis, lattice)
    }

    fn _read_lattice_params(param_primary_axis: &[usize], lattice: &Lattice) -> ArrayVec<[f64; 4]> {
        let mut out = ArrayVec::<[f64; 4]>::new();
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

impl LatticeOptimizationHelper {
    pub fn new(params: &cfg::LatticeRelax, original_lattice: &Lattice) -> Self {
        // no config for now
        let cfg::LatticeRelax {} = params;
        let original_lattice = original_lattice.clone();

        LatticeOptimizationHelper { original_lattice }
    }

    pub fn num_params(&self) -> usize { 4 }

    /// Split a slice into the parts corresponding to fractional coords, and the part corresponding
    /// to the parameters.
    pub fn split<'b, T>(&self, slice: &'b [T]) -> (&'b [V3<T>], &'b [T]) {
        let (frac, params) = slice.split_at(slice.len() - self.num_params());
        (frac.nest(), params)
    }

    pub fn split_mut<'b, T>(&self, slice: &'b mut [T]) -> (&'b mut [V3<T>], &'b mut [T]) {
        let div = slice.len() - self.num_params();
        let (frac, params) = slice.split_at_mut(div);
        (frac.nest_mut(), params)
    }

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
        let mut out = vec![0.0; flat.len()];
        let (_, params) = self.split(flat);
        let (d_site_frac, d_param) = self.split_mut(&mut out);

        let ref lattice = self.lattice_with_params(params);
        let ref recip_lattice = lattice.reciprocal();

        let d_params_as_22: &mut M22 = d_param.nest_mut().as_mut_array();

        for item in bond_grads {
            // We have a term of the potential that can be expressed as V(cart), where
            //
            // cart = frac * lattice
            //
            // and we want to reformulate it as V(frac, lattice)

            // using the notation from rsp2-potentials (each row in a_J_b is a gradient of
            // an element of a with respect to b, so that products compose sensibly)
            let cart = item.cart_vector;
            let frac = cart / lattice;
            let term_d_cart = item.grad;

            // gradients transform by the reciprocal lattice
            let term_d_frac = term_d_cart / recip_lattice;

            // frac = fracs[plus_site] - fracs[minus_site]
            d_site_frac[item.plus_site] += term_d_frac;
            d_site_frac[item.minus_site] -= term_d_frac;

            for r in 0..2 {
                for c in 0..2 {
                    d_params_as_22[r][c] += term_d_cart[c] * frac[r];
                }
            }
        }

        out
    }

    /// Get the gradient with respect to positions in cartesian space,
    /// and with respect to each lattice parameter.
    pub fn unflatten_grad(&self, coords: &[f64], flat_grad: &[f64]) -> (Vec<V3>, ArrayVec<[f64; 4]>) {
        let (_, params) = self.split(coords);
        let (d_fracs, d_params) = self.split(flat_grad);

        // gradients transform as reciprocal vectors
        let recip_lattice = self.lattice_with_params(params).reciprocal();
        let d_carts = CoordsKind::Fracs(d_fracs).to_carts(&recip_lattice);
        (d_carts, d_params.iter().cloned().collect())
    }

    fn read_lattice_params(&self, lattice: &Lattice) -> ArrayVec<[f64; 4]> {
        Self::_read_lattice_params(lattice)
    }

    fn _read_lattice_params(lattice: &Lattice) -> ArrayVec<[f64; 4]> {
        // flatten xy 2x2
        let mut out = ArrayVec::<[f64; 4]>::new();
        for r in 0..2 {
            for c in 0..2 {
                out.push(lattice.matrix()[r][c]);
            }
        }
        out
    }

    fn lattice_with_params(&self, params: &[f64]) -> Lattice {
        assert_eq!(params.len(), self.num_params());

        let mut matrix = self.original_lattice.matrix().clone();
        let params_as_22: &M22 = params.nest().as_array();
        for r in 0..2 {
            for c in 0..2 {
                matrix[r][c] = params_as_22[r][c];
            }
        }
        Lattice::new(&matrix)
    }
}

//---------------------------

pub struct OptimizingDiffFn {
    pub helper: std::rc::Rc<RelaxationOptimizationHelper>,
    pub bond_diff_fn: Box<dyn BondDiffFn<CommonMeta>>,
    pub meta: CommonMeta,
}

impl rsp2_minimize::cg::DiffFn for OptimizingDiffFn {
    type Error = failure::Error;

    fn compute(&mut self, flat_coords: &[f64]) -> Result<(f64, Vec<f64>), failure::Error> {
        let OptimizingDiffFn { ref helper, ref mut bond_diff_fn, ref meta } = *self;
        let ref coords = helper.unflatten_coords(flat_coords);
        let (value, bond_grads) = bond_diff_fn.compute(coords, meta.clone())?;
        let flat_grad = helper.flatten_grad(flat_coords, &bond_grads);
        Ok((value, flat_grad))
    }

    fn check(&mut self, flat_coords: &[f64]) -> Result<(), failure::Error> {
        let OptimizingDiffFn { ref helper, ref mut bond_diff_fn, ref meta } = *self;
        let ref coords = helper.unflatten_coords(flat_coords);
        bond_diff_fn.check(coords, meta.clone())
    }
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potential::{PotentialBuilder, CommonMeta};
    use crate::meta::Element;
    use std::rc::Rc;
    use rsp2_minimize::cg::DiffFn;
    use rsp2_minimize::numerical;
    use rsp2_array_types::{mat, M33};

    pub(super) fn modified_graphene() -> (Coords, CommonMeta) {
        let half_r3 = 0.5 * f64::sqrt(3.0);
        let orig_params = V3([2.3, 2.3, 1.0]); // slightly smaller than ideal size of 2.4
        let mut coords = Coords::new(
            Lattice::new(&(M33::from_diag(orig_params) * &mat::from_array([
                [ 1.0,     0.0, 0.0],
                [-0.5, half_r3, 0.0],
                [ 0.0,     0.0, 20.0],
            ]))),
            CoordsKind::Fracs(vec![
                V3([0.0, 0.0, 0.5]),
                V3([1./3., 1./3., 0.5]),
            ]),
        );

        let elements: Rc<[_]> = vec![Element::CARBON; 2].into();
        let masses: Rc<[_]> = vec![crate::common::default_element_mass(Element::CARBON).unwrap(); 2].into();
        let bonds = Rc::new(rsp2_structure::bonds::FracBonds::compute(&coords, 1.8).unwrap());
        let meta = hlist![elements, masses, Some(bonds)];

        let carts = coords.carts_mut();
        carts[0][0] -= 0.1;
        carts[0][2] += 0.05;
        carts[1][2] -= 0.05;
        (coords, meta.sift())
    }

    #[test]
    fn param_optimization_helper() {
        let (coords, meta) = modified_graphene();
        let cfg = [
            cfg::Parameter::Param('a'),
            cfg::Parameter::Param('a'),
            cfg::Parameter::One,
        ];
        let helper = RelaxationOptimizationHelper::ParamBased({
            ParamOptimizationHelper::new(&cfg, coords.lattice())
        });
        test_helper(helper, coords, meta)
    }

    #[test]
    fn lattice_optimization_helper() {
        let (coords, meta) = modified_graphene();
        let cfg = cfg::LatticeRelax {};
        let helper = RelaxationOptimizationHelper::LatticeBased({
            LatticeOptimizationHelper::new(&cfg, coords.lattice())
        });
        test_helper(helper, coords, meta)
    }

    fn test_helper(helper: RelaxationOptimizationHelper, coords: Coords, meta: CommonMeta) {
        let pot = PotentialBuilder::from_config_parts(
            None,
            None,
            &cfg::Threading::Serial,
            &from_json!({ }),
            &from_json!({ "rebo-nonreactive": {"params": "brenner"} }),
        ).unwrap();
        let bond_diff_fn = pot.initialize_bond_diff_fn(&coords, meta.sift()).unwrap().unwrap();
        let mut diff_fn = OptimizingDiffFn {
            helper: Rc::new(helper.clone()), bond_diff_fn, meta,
        };
        let init_flat_coords = helper.flatten_coords(&coords);

        let num_grad = numerical::gradient(
            1e-4, Some(numerical::DerivativeKind::Stencil(7)),
            &init_flat_coords, |flat_coords| diff_fn.compute(flat_coords).unwrap().0,
        );
        let analytic_grad = diff_fn.compute(&init_flat_coords).unwrap().1;

        assert_close!(rel=1e-7, abs=1e-7, &analytic_grad[..], &num_grad[..]);
    }
}
