use ::{FailResult, FailOk};
use ::cmd::potential::{PotentialBuilder, DiffFn};

use ::rsp2_minimize::exact_ls::{Value, Golden};
use ::rsp2_structure::{Lattice, Structure, CoordsKind};
use ::rsp2_soa_ops::{Perm, Permute};
use ::rsp2_structure_io::layers_yaml::Assemble;
use ::rsp2_array_types::{V3};
use ::rsp2_tasks_config as cfg;

pub(crate) fn optimize_layer_parameters<M: Clone + 'static>(
    settings: &cfg::ScaleRanges,
    pot: &PotentialBuilder<M>,
    mut structure_builder: ScalableStructure<M>,
) -> FailResult<ScalableStructure<M>>
{Ok({

    let cfg::ScaleRanges {
        scalables: ref scalable_cfgs,
        warn_threshold,
        fail,
        repeat_count,
    } = *settings;

    // Gather a bunch of setter functions and search ranges
    let scalables = {
        let mut scalables = vec![];
        for cfg in scalable_cfgs {
            add_scalables(
                cfg,
                &structure_builder,
                |s| scalables.push(s),
            );
        }
        scalables
    };

    // Set reasonable values for first iteration.
    for Scalable { spec, setter, .. } in &scalables {
        match *spec {
            cfg::ScalableRange::Exact { value } |
            cfg::ScalableRange::Search { range: _, guess: Some(value) } => {
                setter(&mut structure_builder, value);
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
    for _ in 0..repeat_count {
        for &Scalable { ref name, spec, ref setter } in &scalables {
            trace!("Optimizing {}", name);

            let best = match spec {
                cfg::ScalableRange::Exact { value } => value,
                cfg::ScalableRange::Search { guess: _, range } => {
                    let best = Golden::new()
                        .stop_condition(&from_json!({"interval-size": 1e-7}))
                        .run(range, |a| {
                            setter(&mut structure_builder, a);

                            // FIXME: I'm not sure why this is a one-off computation; I should try
                            //        reusing a single lammps instance and see what happens.
                            //
                            //        (maybe I had issues with "lost atoms" errors or something)
                            pot.one_off()
                                .compute_value(&structure_builder.construct())
                                .map(Value)

                        // note: result is Result<Result<_, E>, GoldenSearchError>
                        })??; // ?!??!!!?

                    if let Some(thresh) = warn_threshold {
                        // use signed differences so that all values outside violate the threshold
                        let lo = range.0.min(range.1);
                        let hi = range.0.max(range.1);
                        if (best - range.0).min(range.1 - best) / (range.1 - range.0) < thresh {
                            warn!("Relaxed value of '{}' is suspiciously close to limits!", name);
                            warn!("  lo: {:e}", lo);
                            warn!(" val: {:e}", best);
                            warn!("  hi: {:e}", hi);
                        }
                    }

                    info!("Optimized {}: {} (from {:?})", name, best, range);
                    best
                },
            }; // let best = { ... }

            setter(&mut structure_builder, best);
        } // for ... in optimizables
    } // for ... in repeat_count

    structure_builder
})}

struct Scalable<M> {
    setter: Box<Fn(&mut ScalableStructure<M>, f64)>,
    name: String,
    spec: cfg::ScalableRange,
}

pub enum ScalableStructure<M> {
    KnownLayers {
        layer_builder: Assemble,

        // permutation that restores the true order of the atoms
        // from the output of Assemble.
        perm_from_assembled: Perm,

        // composite meta, after the perm has already been applied
        meta: Vec<M>,
    },
    UnknownLayers {
        scales: [f64; 3],
        lattice: Lattice,
        fracs: Vec<V3>,
        meta: Vec<M>,
    },
}

impl<M: Clone> ScalableStructure<M> {
    fn construct(&self) -> Structure<M> {
        match self {
            &ScalableStructure::KnownLayers {
                ref layer_builder, ref perm_from_assembled, ref meta,
            } => {
                layer_builder.assemble()
                    .permuted_by(perm_from_assembled)
                    .with_metadata(meta.to_vec())
            },
            &ScalableStructure::UnknownLayers {
                ref scales, ref lattice, ref fracs, ref meta,
            } => {
                Structure::new(
                    &Lattice::diagonal(scales) * lattice,
                    CoordsKind::Fracs(fracs.to_vec()),
                    meta.to_vec(),
                )
            },
        }
    }
}

fn add_scalables<M>(
    cfg: &cfg::Scalable,
    structure: &ScalableStructure<M>,
    emit: impl FnMut(Scalable<M>),
) -> FailResult<()> {
    // validate
    match (cfg, structure) {
        (
            &cfg::Scalable::Param { axis_mask, .. },
            &ScalableStructure::KnownLayers { ref layer_builder, .. },
        ) => {
            // (this is unfair to bulk...)
            ensure!(
                !axis_mask[layer_builder.normal_axis()].0,
                "Cannot scale the layer axis normal direction with a 'param'"
            );
        },

        (
            &cfg::Scalable::UniformLayerSep { .. },
            &ScalableStructure::UnknownLayers { .. },
        ) | (
            &cfg::Scalable::LayerSeps { .. },
            &ScalableStructure::UnknownLayers { .. },
        ) => bail!("cannot scale layer separations when layers have not been determined"),
    }

    // obtain data to be used by some scalables
    let mut n_layer_seps = None;
    match structure {
        ScalableStructure::KnownLayers { ref layer_builder, .. } => {
            n_layer_seps = Some(layer_builder.num_layer_seps());
        },
        _ => {},
    }

    match cfg {
        &cfg::Scalable::Param { axis_mask, ref range } => {
            emit(Scalable {
                // FIXME these should get unique names
                name: format!("lattice param"),
                setter: Box::new(|s, val| {
                    let scales = match s {
                        ScalableStructure::KnownLayers { layer_builder, .. } => &mut layer_builder.scale,
                        ScalableStructure::UnknownLayers { scales, .. } => scales,
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
                    ScalableStructure::UnknownLayers { .. } => unreachable!(),
                    ScalableStructure::KnownLayers { layer_builder, .. } => {
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
                    setter: Box::new(|s, val| match s {
                        ScalableStructure::UnknownLayers { .. } => unreachable!(),
                        ScalableStructure::KnownLayers { layer_builder, .. } => {
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
