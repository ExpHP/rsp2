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

use ::FailResult;
use ::ui::color::{ColorByRange, PaintAs, NullPainter};
use ::ui::cfg_merging::{no_summary, merge_summaries, make_nested_mapping};
use ::math::basis::Basis3;
use ::math::bands::{GammaUnfolder};
#[allow(unused)] // compiler bug
use ::itertools::Itertools;
use ::rsp2_tasks_config::Settings;

#[allow(unused)] // compiler bug
use ::rsp2_soa_ops::{Part, Partition};

use ::std::fmt;
use ::serde_yaml::Value as YamlValue;
#[allow(unused)] // compiler bug
use ::frunk::hlist::Sculptor;
use ::threading::Threading;

use super::acoustic_search;

// NOTE: All inputs are wrapped in distinct types for the sake of type-based indexing.
//       However, some inputs are of a sufficiently unique type that adding another newtype
//       wrapper seems pointless, so we simply reexport those directly.
//
// NOTE: Since a lot of the wrapped types are just vectors, the naming convention
//       is to prefix the name with each thing they are indexed over (in order)
// FIXME: Y'know, we DO have `Indexed` now...
pub use ::rsp2_structure::Coords as SiteCoordinates;
pub use ::meta::SiteLayers;
pub use ::meta::SiteElements;
pub use ::meta::SiteMasses;
pub use ::meta::LayerScMatrices;
#[derive(Debug, Clone)] pub struct EvClassifications(pub Vec<acoustic_search::ModeKind>);
#[derive(Debug, Clone)] pub struct EvFrequencies(pub Vec<f64>);
#[derive(Debug, Clone)] pub struct EvEigenvectors(pub Basis3);
#[derive(Debug, Clone)] pub struct Bonds(pub ::math::bonds::CartBonds);

pub use self::gamma_system_analysis::GammaSystemAnalysis;
pub mod gamma_system_analysis {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Input {
        // "Le gasp," I know.  Using Options for everything tosses away a lot of would-be static
        // guarantees and basically throws us into the land of Hoare's "billion dollar mistake."
        //
        // Basically what happens here is that values in the Output are selectively computed
        // based on whether all of the inputs they depend on are available. The logic for handling
        // this is in a macro, which is why everything must be Option.
        //
        // The point is to allow properties specific to certain types of materials to be
        // selectively computed (and displayed, and written to output files...).
        //
        pub site_coords:        Option<SiteCoordinates>,
        pub site_layers:        Option<SiteLayers>,
        pub site_elements:      Option<SiteElements>,
        pub site_masses:        Option<SiteMasses>,
        pub layer_sc_mats:      Option<LayerScMatrices>,
        pub ev_classifications: Option<EvClassifications>,
        pub ev_frequencies:     Option<EvFrequencies>,
        pub ev_eigenvectors:    Option<EvEigenvectors>,
        pub bonds:              Option<Bonds>,
    }

    pub struct GammaSystemAnalysis {
        pub ev_classifications:    Option<EvClassifications>,
        pub ev_frequencies:        Option<EvFrequencies>,
        pub ev_acousticness:       Option<EvAcousticness>,
        pub ev_polarization:       Option<EvPolarization>,
        pub ev_layer_acousticness: Option<EvLayerAcousticness>,
        pub ev_raman_tensors:      Option<EvRamanTensors>,
        pub layer_sc_mats:         Option<LayerScMatrices>,
        pub unfold_probs:          Option<UnfoldProbs>,
    }

    impl Input {
        pub fn compute(&self) -> FailResult<GammaSystemAnalysis>
        {Ok({
            let Input {
                site_coords, site_layers, site_elements, site_masses,
                layer_sc_mats, ev_frequencies, ev_eigenvectors, bonds,
                ev_classifications,
            } = self;

            // since our inputs are all uniquely typed, we can let HList
            // take care of finding all the right function arguments.
            let grab_bag = hlist![
                site_coords, site_layers, site_elements, site_masses,
                layer_sc_mats, ev_frequencies, ev_eigenvectors, bonds,
            ];

            let (args, _) = grab_bag.sculpt();
            let ev_acousticness = ev_acousticness::maybe_compute(args)?;

            let (args, _) = grab_bag.sculpt();
            let ev_polarization = ev_polarization::maybe_compute(args)?;

            let (args, _) = grab_bag.sculpt();
            let ev_layer_acousticness = ev_layer_acousticness::maybe_compute(args)?;

            let (args, _) = grab_bag.sculpt();
            let unfold_probs = unfold_probs::maybe_compute(args)?;

            let (args, _) = grab_bag.sculpt();
            let ev_raman_tensors = ev_raman_tensors::maybe_compute(args)?;

            let ev_frequencies = ev_frequencies.clone();
            let ev_classifications = ev_classifications.clone();
            let layer_sc_mats = layer_sc_mats.clone();

            GammaSystemAnalysis {
                ev_classifications,
                ev_frequencies,
                ev_acousticness,
                ev_polarization,
                layer_sc_mats,
                unfold_probs,
                ev_layer_acousticness,
                ev_raman_tensors,
            }
        })}
    }
}

macro_rules! wrap_maybe_compute {
    ( // '= path' syntax for delegating to a function defined outside the macro
        pub $struct_or_enum:tt $Thing:ident
            $(   { $($Thing_body_if_brace:tt)* }      )*
            $($( ( $($Thing_body_if_paren:tt)*  ) )* ; )*

        fn $thing:ident(
            $($arg:ident : & $Arg:ident),* $(,)*
        ) -> FailResult<_>
        = $fn_path:path;
    ) => {
        wrap_maybe_compute! {
            pub $struct_or_enum $Thing
                $(   { $($Thing_body_if_brace)* }      )*
                $($( ( $($Thing_body_if_paren)* ) )* ; )*

            fn $thing(
                $($arg : & $Arg),*
            ) -> FailResult<_>
            { $fn_path($($arg),*) }
        }
    };

    ( // function body defined inside the macro
        pub $struct_or_enum:tt $Thing:ident
            $(   { $($Thing_body_if_brace:tt)* }      )*
            $($( ( $($Thing_body_if_paren:tt)*  ) )* ; )*

        fn $thing:ident(
            $($arg:ident : & $Arg:ident),* $(,)*
        ) -> FailResult<_>
        $fn_body:block
    ) => {
        pub use self::$thing::$Thing;
        pub mod $thing {
            use super::*;

            pub $struct_or_enum $Thing
            $(   { $($Thing_body_if_brace)* }      )*
            $($( ( $($Thing_body_if_paren)* ) )* ; )*

            pub fn maybe_compute(
                list: Hlist![$(&Option<$Arg>),*],
            ) -> FailResult<Option<$Thing>>
            {
                let hlist_pat![$($arg),*] = list;
                $(
                    let $arg = match $arg {
                        Some(x) => x,
                        None => {
                            trace!{
                                "not computing {} due to missing requirement {}",
                                stringify!($Thing), stringify!($Arg),
                            }
                            return Ok(None);
                        },
                    };
                )*
                trace!{"computing {}", stringify!($Thing)}
                Ok(Some(compute($($arg),*)?))
            }

            fn compute($($arg: &$Arg),*) -> FailResult<$Thing>
            $fn_body
        }
    }
}

// Example of what the macro expands into.
pub use self::ev_acousticness::EvAcousticness;
pub mod ev_acousticness {
    use super::*;
    use ::frunk::hlist::{HCons, HNil};

    pub struct EvAcousticness(pub Vec<f64>);

    pub fn maybe_compute(list: HCons<&Option<EvEigenvectors>, HNil>)
    -> FailResult<Option<EvAcousticness>>
    {
        let hlist_pat![ev_eigenvectors] = list;
        let ev_eigenvectors = match ev_eigenvectors {
            Some(x) => x,
            None => {
                trace!{"not computing EvAcousticness due to missing requirement EvEigenvectors"}
                return Ok(None);
            },
        };
        trace!{"computing EvAcousticness"}
        Ok(Some(compute(ev_eigenvectors)?))
    }

    fn compute(ev_eigenvectors: &EvEigenvectors) -> FailResult<EvAcousticness> {
        Ok(EvAcousticness((ev_eigenvectors.0).0.iter().map(|ket| ket.acousticness()).collect()))
    }
}

wrap_maybe_compute! {
    pub struct EvPolarization(pub Vec<[f64; 3]>);
    fn ev_polarization(ev_eigenvectors: &EvEigenvectors) -> FailResult<_> {
        Ok(EvPolarization((ev_eigenvectors.0).0.iter().map(|ket| ket.polarization()).collect()))
    }
}

wrap_maybe_compute! {
    pub struct EvLayerAcousticness(pub Vec<f64>);
    fn ev_layer_acousticness(
        site_layers: &SiteLayers,
        ev_eigenvectors: &EvEigenvectors,
    ) -> FailResult<_> {
        let part = Part::from_ord_keys(site_layers.iter());
        Ok(EvLayerAcousticness({
            (ev_eigenvectors.0).0.iter().map(|ket| {
                ket.clone()
                    .into_unlabeled_partitions(&part)
                    .map(|ev| ev.acousticness())
                    .sum()
            }).collect()
        }))
    }
}

wrap_maybe_compute! {
    pub struct UnfoldProbs {
        pub layer_unfolders: Vec<GammaUnfolder>,
        pub layer_ev_q_probs: Vec<Vec<Vec<f64>>>,
    }
    fn unfold_probs(
        site_layers: &SiteLayers,
        site_coords: &SiteCoordinates,
        layer_sc_mats: &LayerScMatrices,
        ev_eigenvectors: &EvEigenvectors,
    ) -> FailResult<_>
    = _unfold_probs;
}

impl UnfoldProbs {
    fn layer_ev_gamma_probs(&self) -> Vec<Vec<f64>> {
        let UnfoldProbs { layer_unfolders, layer_ev_q_probs } = self;

        zip_eq!(layer_unfolders, layer_ev_q_probs)
            .map(|(unfolder, ev_q_probs)| {
                ev_q_probs.iter().map(|probs| {
                    zip_eq!(unfolder.q_indices(), probs.iter().cloned())
                        .find(|&(idx, _)| idx == &[0, 0, 0])
                        .unwrap().1
                }).collect()
            }).collect()
    }
}

fn _unfold_probs(
    site_layers: &SiteLayers,
    site_coords: &SiteCoordinates,
    layer_sc_mats: &LayerScMatrices,
    ev_eigenvectors: &EvEigenvectors,
) -> FailResult<UnfoldProbs> {
    let part = Part::from_ord_keys(site_layers.iter());
    let layer_partial_coords = site_coords
        .clone()
        .into_unlabeled_partitions(&part)
        .collect_vec();

    let ev_layer_partial_evs = (ev_eigenvectors.0).0.iter().map(|ket| {
        ket.clone().into_unlabeled_partitions(&part)
    });
    let layer_partial_evs = ::util::transpose_iter_to_vec(ev_layer_partial_evs);

    let (layer_unfolders, layer_ev_q_probs) = {
        zip_eq!(layer_partial_coords, layer_partial_evs, &layer_sc_mats[..])
            .map(|(partial_structure, partial_evs, sc_mat)| {
                // precompute data applicable to all kets
                let unfolder = GammaUnfolder::from_config(
                    &from_json!({
                        "fbz": "reciprocal-cell",
                        "sampling": { "plain": [4, 4, 1] },
                    }),
                    Threading::Parallel,
                    &partial_structure,
                    sc_mat,
                );

                let ev_q_probs = partial_evs.into_iter().map(|ket| {
                    unfolder.unfold_phonon(Threading::Parallel, ket.to_ket().as_ref())
                }).collect();
                (unfolder, ev_q_probs)
            }).unzip()
    };

    Ok(UnfoldProbs { layer_unfolders, layer_ev_q_probs })
}

wrap_maybe_compute! {
    pub struct EvRamanTensors(pub Vec<::math::bond_polarizability::RamanTensor>);
    fn ev_raman_tensors(
        bonds: &Bonds,
        site_masses: &SiteMasses,
        site_elements: &SiteElements,
        ev_frequencies: &EvFrequencies,
        ev_eigenvectors: &EvEigenvectors,
    ) -> FailResult<_>
    = _ev_raman_tensors;
}

fn _ev_raman_tensors(
    bonds: &Bonds,
    site_masses: &SiteMasses,
    site_elements: &SiteElements,
    ev_frequencies: &EvFrequencies,
    ev_eigenvectors: &EvEigenvectors,
) -> FailResult<EvRamanTensors> {
    use ::math::bond_polarizability::{Input};

    Input {
        temperature: 0.0,
        site_masses: &site_masses,
        site_elements: &site_elements,
        ev_eigenvectors: &ev_eigenvectors.0,
        ev_frequencies: &ev_frequencies.0,
        bonds: &bonds.0,
    }.compute_ev_raman_tensors()
        .map(EvRamanTensors)
}

macro_rules! format_columns {
    (
        $header_fmt: expr,
        $entry_fmt: expr,
        $columns1:ident $(,)*
    ) => {
        Columns {
            header: format!($header_fmt, $columns1.header),
            entries: {
                $columns1.entries.into_iter()
                    .map(|$columns1| format!($entry_fmt, $columns1))
                    .collect()
            },
        }
    };
    (
        $header_fmt: expr,
        $entry_fmt: expr,
        $columns1:ident, $($columns:ident,)+ $(,)*
    ) => {
        Columns {
            header: format!($header_fmt, $columns1.header, $($columns.header),+),
            entries: {
                izip!($columns1.entries, $(&$columns.entries),+)
                    .map(|($columns1, $($columns),*)| format!($entry_fmt, $columns1, $($columns),*))
                    .collect()
            },
        }
    };
}

pub enum ColumnsMode {
    ForHumans,
    ForMachines,
}

impl GammaSystemAnalysis {
    pub fn make_columns(&self, mode: ColumnsMode) -> Option<Columns> {
        use self::Color::{Colorful, Colorless};
        use self::columns::{quick_column, fixed_prob_column, display_prob_column};
        use self::columns::aligned_dot_column;

        let mut columns = vec![];

        let fix1 = |c, title: &str, data: &_| fixed_prob_column(c, Precision(1), title, data);
        let fix2 = |c, title: &str, data: &_| fixed_prob_column(c, Precision(2), title, data);
        let dp = display_prob_column;

        if let Some(data) = &self.ev_classifications {
            columns.push(Columns {
                header: "(C)".to_string(),
                entries: data.0.iter().map(|&kind| {
                    match mode {
                        ColumnsMode::ForHumans => format!("({})", ::cmd::acoustic_search::Colorful(kind)),
                        ColumnsMode::ForMachines => format!("({})", kind),
                    }
                }).collect(),
            })
        }

        if let Some(data) = &self.ev_frequencies {
            columns.push(aligned_dot_column("Frequency(cm-1)", &data.0));
        };

        if let Some(data) = &self.ev_acousticness {
            columns.push(match mode {
                ColumnsMode::ForHumans => dp(Colorful, "Acoust.", &data.0),
                ColumnsMode::ForMachines => fix2(Colorless, "Acou", &data.0),
            })
        };

        if let Some(EvRamanTensors(tensors)) = &self.ev_raman_tensors {
            let raman_column = |name: &str, data: &[f64]| {
                // NOTE: intensity can be "negative" for negative modes.  This clearly is
                //       not physical, but we also don't care about intensities of such modes,
                //       so simply set them to zero.
                let data = data.iter().map(|&x| f64::max(0.0, x)).collect_vec();
                // NOTE: raman_intensities are currently missing some scale factors
                //       so they are just normalized for now.
                let max = data.iter().fold(0.0, |acc, &x| f64::max(acc, x));
                let data = if max == 0.0 {
                    vec![0.0; data.len()]
                } else {
                    data.into_iter().map(|x| x / max).collect_vec()
                };
                let painter: Box<PaintAs<_, f64>> = match mode {
                    ColumnsMode::ForHumans => Box::new({
                        use ::ansi_term::Colour::*;
                        ColorByRange::new(vec![
                            ( 1e-0, Cyan.bold()),
                            ( 1e-1, Cyan.normal()),
                            ( 1e-5, Yellow.normal()),
                            (1e-10, Red.bold()),
                            (1e-25, Red.normal()),
                        ],          Black.normal()) // make zeros "disappear"
                    }),
                    ColumnsMode::ForMachines => Box::new(NullPainter),
                };
                quick_column(
                    &*painter, name, &data, 5,
                    |&value| ShortExp { value, cutoff_exp: -45 },
                )
            };

            use ::math::bond_polarizability::LightPolarization::*;
            columns.push(raman_column(
                "RamnA",
                &tensors.iter().map(|t| t.integrate_intensity(&Average)).collect_vec(),
            ));
            columns.push(raman_column(
                "RamnB",
                &tensors.iter().map(|t| t.integrate_intensity(&BackscatterZ)).collect_vec(),
            ));
        };

        if let Some(data) = &self.ev_layer_acousticness {
            columns.push(match mode {
                ColumnsMode::ForHumans => dp(Colorful, "Layer", &data.0),
                ColumnsMode::ForMachines => fix2(Colorless, "Lay.", &data.0),
            })
        };

        if let Some(data) = &self.ev_polarization {
            let data = |k| data.0.iter().map(|v| v[k]).collect_vec();
            let name = |k| "XYZ".chars().nth(k).unwrap().to_string();

            columns.push(match mode {
                ColumnsMode::ForHumans => {
                    let axis = |k| fix2(Colorful, &name(k), &data(k));
                    let (x, y, z) = (axis(0), axis(1), axis(2));
                    format_columns! {
                        "[{}, {}, {}]",
                        "[{}, {}, {}]",
                        x, y, z,
                    }
                },
                ColumnsMode::ForMachines => {
                    let axis = |k| fix2(Colorless, &name(k), &data(k));
                    let (x, y, z) = (axis(0), axis(1), axis(2));
                    format_columns! {
                        "{} {} {}",
                        "{} {} {}",
                        x, y, z,
                    }
                },
            });
        }

        if let Some(obj) = &self.unfold_probs {
            let data = obj.layer_ev_gamma_probs();
            for (n, probs) in data.iter().enumerate() {
                columns.push(match mode {
                    ColumnsMode::ForHumans   => fix1(Colorful,  &format!("G{:02}", n+1), &probs),
                    ColumnsMode::ForMachines => fix1(Colorless, &format!("G{:02}", n+1), &probs),
                })
            }
        }

        match columns.len() {
            0 => None,
            _ => Some({
                let joined = columns::join(&columns, " ");
                format_columns!(
                    "# {}", // "comment out" the header
                    "  {}",
                    joined,
                )
            }),
        }
    }
}

impl GammaSystemAnalysis {
    pub fn make_summary(&self, settings: &Settings) -> YamlValue {
        let GammaSystemAnalysis {
            ev_acousticness, ev_polarization,
            ev_frequencies, unfold_probs,
            ev_layer_acousticness,
            ev_raman_tensors: _,
            ev_classifications: _,
            layer_sc_mats: _,
        } = self;

        // This is where the newtypes start to get in the way;
        // turn as many things as we can into Option<Vec<T>> (indexed by ket)
        // for ease of composition.
        let frequency = ev_frequencies.as_ref().map(|d| d.0.to_vec());
        let acousticness = ev_acousticness.as_ref().map(|d| d.0.to_vec());
        let polarization = ev_polarization.as_ref().map(|d| d.0.to_vec());
        let layer_acousticness = ev_layer_acousticness.as_ref().map(|d| d.0.to_vec());
        //let raman_intensities = ev_raman_intensities.as_ref().map(|d| d.0.to_vec());

        // Work with Option<Vec<A>> as an applicative functor (for fixed length Vec)
        fn map1<A, R, F>(a: &Option<Vec<A>>, mut f: F) -> Option<Vec<R>>
        where F: FnMut(&A) -> R,
        {
            let a = a.as_ref()?;
            Some(a.iter().map(|a| f(a)).collect())
        }

        // (haskell's LiftA2)
        fn map2<B, A, R, F>(a: &Option<Vec<A>>, b: &Option<Vec<B>>, mut f: F) -> Option<Vec<R>>
        where F: FnMut(&A, &B) -> R,
        {
            Some({
                zip_eq!(a.as_ref()?, b.as_ref()?)
                    .map(|(a, b)| f(a, b))
                    .collect()
            })
        }

        // form a miniature DSL to reduce chances of accidental
        // discrepancy between closure args and parameters (
        fn at_least(thresh: f64, a: &Option<Vec<f64>>) -> Option<Vec<bool>>
        { map1(a, |&a| thresh <= a) }

        fn and(a: &Option<Vec<bool>>, b: &Option<Vec<bool>>) -> Option<Vec<bool>>
        { map2(a, b, |&a, &b| a && b) }

        fn zip<A: Clone, B: Clone>(a: &Option<Vec<A>>, b: &Option<Vec<B>>) -> Option<Vec<(A, B)>>
        { map2(a, b, |a, b| (a.clone(), b.clone())) }

        fn not(a: &Option<Vec<bool>>) -> Option<Vec<bool>>
        { map1(a, |a| !a) }

        fn enumerate<T>(a: &Option<Vec<T>>) -> Option<Vec<(usize, &T)>>
        { a.as_ref().map(|a| a.iter().enumerate().collect()) }

        fn select<T: Clone>(pred: &Option<Vec<bool>>, values: &Option<Vec<T>>) -> Option<Vec<T>>
        {
            Some({
                zip_eq!(pred.as_ref()?, values.as_ref()?)
                    .filter_map(|(&p, v)| if p { Some(v.clone()) } else { None })
                    .collect()
            })
        }

        let z_polarization = map1(&polarization, |p| p[2]);
        let xy_polarization = map1(&polarization, |p| p[0] + p[1]);

        let is_acoustic = at_least(0.95, &acousticness);
        let is_z_polarized = at_least(0.9, &z_polarization);
        let is_xy_polarized = at_least(0.9, &xy_polarization);
        let is_layer_acoustic = and(&at_least(0.95, &layer_acousticness), &not(&is_acoustic));

        let is_shear = and(&is_layer_acoustic, &is_xy_polarized);
        let is_layer_breathing = and(&is_layer_acoustic, &is_z_polarized);

        let mut out = vec![];

        if let Some(freqs) = select(&is_acoustic, &frequency) {
            let value = ::serde_yaml::to_value(freqs).unwrap();
            out.push(make_nested_mapping(&["acoustic"], value));
        }

        if let Some(freqs) = select(&is_shear, &frequency) {
            let value = ::serde_yaml::to_value(freqs).unwrap();
            out.push(make_nested_mapping(&["shear"], value));
        }

        if let Some(freqs) = select(&is_layer_breathing, &frequency) {
            let value = ::serde_yaml::to_value(freqs).unwrap();
            out.push(make_nested_mapping(&["layer-breathing"], value));
        }

        // For gamma probs, don't bother with all layers; just a couple.
        let layer_ev_gamma_probs = unfold_probs.as_ref().map(|u| u.layer_ev_gamma_probs());
        [0, 1].iter().for_each(|&layer_n| {

            let probs = layer_ev_gamma_probs.as_ref().map(|d| d[layer_n].to_vec());
            let layer_key = format!("layer-{}", layer_n + 1);

            let pred = at_least(settings.layer_gamma_threshold, &probs);
            if let Some(tuples) = select(&pred, &enumerate(&zip(&frequency, &probs))) {
                #[derive(Serialize)]
                struct Item {
                    index: usize,
                    frequency: f64,
                    probability: f64,
                }

                let items = tuples.into_iter().map(|(index, &(frequency, probability))| {
                    Item { index, frequency, probability }
                }).collect_vec();
                let value = ::serde_yaml::to_value(&items).unwrap();
                out.push(make_nested_mapping(&["layer-gammas", &layer_key], value));
            }
        });

        match out.len() {
            0 => no_summary(),
            _ => {
                let yaml = out.into_iter().fold(no_summary(), merge_summaries);
                make_nested_mapping(&["modes", "gamma"], yaml)
            }
        }
    }
}

// Color range used by most columns that contain probabilities in [0, 1]
fn default_prob_color_range() -> ColorByRange<f64> {
    use ::ansi_term::Colour::*;
    ColorByRange::new(vec![
        (0.999, Cyan.bold()),
        (0.9,   Cyan.normal()),
        (0.1,   Yellow.normal()),
        (1e-4,  Red.bold()),
        (1e-10, Red.normal()),
    ],          Black.normal()) // make zeros "disappear"
}

/// Simple Display impl for probabilities (i.e. from 0 to 1).
///
/// Shows a float at dynamically-chosen fixed precision.
#[derive(Debug, Copy, Clone)]
pub struct FixedProb(f64, usize);
impl fmt::Display for FixedProb {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    { write!(f, "{:width$.prec$}", self.0, prec = self.1, width = self.1 + 2) }
}

/// Simple Display impl for positive numbers of wildly-varying magnitude.
///
/// Shows a float at dynamically-chosen fixed precision.
///
/// Printed width is always 5 characters, assuming cutoff_exp >= -99
#[derive(Debug, Copy, Clone)]
pub struct ShortExp {
    value: f64,
    cutoff_exp: i32,
}
impl fmt::Display for ShortExp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let ShortExp { value, cutoff_exp } = *self;
        assert!(value >= 0.0);
        if value == 0.0 || (value.log10().ceil() as i32) < cutoff_exp {
            write!(f, "{:<5}", "0")
        } else {
            write!(f, "{:<5.0e}", value)
        }
    }
}

/// Specialized display impl for probabilities (i.e. from 0 to 1)
/// which may be extremely close to either 0 or 1.
///
/// This should only be used on a value which is computed from a sum of
/// non-negative values; if it were computed from a sum where cancellation may
/// occur, then magnitudes close to 0 or 1 would be too noisy to be meaningful.
///
/// Always displays with a width of 7 characters.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct DisplayProb(f64);
impl fmt::Display for DisplayProb {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let log10_1p = |x: f64| x.ln_1p() / ::std::f64::consts::LN_10;

        // NOTE: This used to deliberately reject values precisely equal to zero,
        //       but I could not recall why, so I loosened the restriction.
        //       (that said, the assertion never failed when it did reject zeros...)
        //
        // NOTE: It still rejects slightly negative values ("numerical zero")
        //       because it should not be used on sums where cancellation may occur.
        assert!(
            0.0 <= self.0 && self.0 < 1.0 + 1e-5,
            "bad probability: {}", self.0,
        );

        if self.0 >= 1.0 {
            write!(f, "{:>7}", 1.0)
        } else if self.0 == 0.0 {
            write!(f, "{:>7}", 0.0) // don't do log of 0
        } else if self.0 - 1e-3 <= 0.0 {
            write!(f, "  1e{:03}", self.0.log10().round())
        } else if self.0 + 1e-3 >= 1.0 {
            write!(f, "1-1e{:03}", log10_1p(-self.0).round())
        } else {
            write!(f, "{:<7.5}", self.0)
        }
    }
}

use self::columns::{Columns, Color, Precision};
mod columns {
    use super::*;
    use std::iter::{Chain, Once, once};

    #[derive(Debug, Clone)]
    pub struct Columns<T = String> {
        pub header: String,
        pub entries: Vec<T>,
    }

    impl<T: fmt::Display> IntoIterator for Columns<T> {
        type IntoIter = Chain<Once<String>, ::std::vec::IntoIter<String>>;
        type Item = String;

        fn into_iter(self) -> Self::IntoIter
        {
            // HACK: so many unnecessary allocations...
            let entries = self.entries.into_iter().map(|x| x.to_string()).collect_vec();
            once(self.header).chain(entries)
        }
    }

    /// Join columns side-by-side.
    pub fn join(columns: &[Columns], sep: &str) -> Columns {
        let mut columns = columns.iter();
        let mut out = columns.next().expect("can't join 0 columns").clone();
        for column in columns {
            out.header = out.header + sep + &column.header;
            for (dest, src) in zip_eq!(&mut out.entries, &column.entries) {
                *dest += sep;
                *dest += src;
            }
        }
        out
    }

    /// Factors out logic common between the majority of columns.
    ///
    /// Values are string-formatted by some mapping function, and optionally colorized
    /// according to the magnitude of their value.
    pub fn quick_column<C, D, F>(
        painter: &PaintAs<D, C>, header: &str, values: &[C],
        width: usize, mut show: F,
    ) -> Columns
    where
        C: PartialOrd,
        F: FnMut(&C) -> D,
        D: fmt::Display,
    { Columns {
        header: format!("{:^width$}", header, width = width),
        entries: values.iter().map(|x| format!("{}", painter.paint_as(x, show(x)))).collect(),
    }}

    pub struct Precision(pub usize);
    pub enum Color {
        Colorful,
        Colorless,
    }

    pub fn aligned_dot_column(header: &str, values: &[f64]) -> Columns
    {
        // do it the dumb way
        // print each one to a string
        let mut strings: Vec<_> = values.iter().map(|x| x.to_string()).collect();
        for s in &mut strings {
            if !s.contains(".") {
                *s += ".0";
            }
        }

        let period_indices: Vec<_> = {
            strings.iter()
                .map(|s| s.find(".").expect("we just guaranteed a '.'"))
                .collect()
        };

        let &max_period = period_indices.iter().max().expect("no lines!?");
        let padding_source = " ".repeat(max_period);
        for (period, s) in zip_eq!(period_indices, &mut strings) {
            match max_period - period {
                0 => {},
                n => {
                    let new = padding_source[..n].to_string();
                    let old = ::std::mem::replace(s, new);
                    *s += &old;
                },
            }
        }

        let max_len = strings.iter().map(|s| s.len()).max().expect("no lines!?");
        let pad_right = |s| format!("{:<len$}", s, len=max_len);

        // start the column header at the digit in the ones place
        let header = padding_source[..max_period.saturating_sub(1)].to_string() + header;
        let header = pad_right(header);

        let entries = strings.into_iter().map(pad_right).collect();

        Columns { header, entries }
    }

    pub fn fixed_prob_column(color: Color, precision: Precision, header: &str, values: &[f64]) -> Columns
    {
        let painter: Box<PaintAs<_, f64>> = match color {
            Color::Colorful  => Box::new(default_prob_color_range()),
            Color::Colorless => Box::new(NullPainter),
        };
        quick_column(&*painter, header, values, precision.0 + 2, |&x| FixedProb(x, precision.0))
    }

    pub fn display_prob_column(color: Color, header: &str, values: &[f64]) -> Columns
    {
        let painter: Box<PaintAs<_, f64>> = match color {
            Color::Colorful  => Box::new(default_prob_color_range()),
            Color::Colorless => Box::new(NullPainter),
        };
        quick_column(&*painter, header, values, 7, |&x| DisplayProb(x))
    }
}
