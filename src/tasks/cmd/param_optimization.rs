use ::{FailResult};
use ::cmd::potential::{PotentialBuilder, DiffFn};
use ::hlist_aliases::*;

use ::rsp2_minimize::exact_ls::{Value, Golden};
use ::rsp2_structure::{Lattice, CoordsKind, Element, Coords};
use ::rsp2_structure::layer::{LayersPerUnitCell, require_simple_axis_normal};
use ::rsp2_structure_io::assemble::{Assemble, RawAssemble};
use ::rsp2_array_types::{V3};
use ::rsp2_tasks_config as cfg;

pub(crate) fn optimize_layer_parameters(
    settings: &cfg::ScaleRanges,
    pot: &PotentialBuilder,
    mut coords_builder: ScalableCoords,
    meta: HList1<
        &[Element],
    >,
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
                                .compute_value(&::compat(&coords_builder.construct(), meta.sculpt().0))
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
