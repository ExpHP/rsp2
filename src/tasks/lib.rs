// HERE BE DRAGONS

extern crate rsp2_lammps_wrap;
extern crate rsp2_minimize;
extern crate rsp2_structure;
extern crate rsp2_structure_io;
extern crate rsp2_phonopy_io;
extern crate rsp2_array_utils;
extern crate rsp2_slice_math;
extern crate rsp2_tempdir;
extern crate rsp2_eigenvector_classify;
#[macro_use] extern crate rsp2_util_macros;

extern crate rand;
extern crate env_logger;
extern crate slice_of_array;
extern crate serde;
extern crate ansi_term;
extern crate serde_json;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate itertools;

const THZ_TO_WAVENUMBER: f64 = 33.35641;

use ::rsp2_array_utils::vec_from_fn;
use ::slice_of_array::prelude::*;
use ::rsp2_structure::{supercell, Coords, CoordStructure, Lattice};
use ::rsp2_lammps_wrap::Lammps;
use ::std::path::Path;

type LmpError = ::rsp2_lammps_wrap::Error;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Settings {
    supercell_relax: SupercellSpec,
    supercell_phonopy: SupercellSpec,
    displacement_distance: f64, // 1e-3
    neg_frequency_threshold: f64, // 1e-3
    hack_scale: [f64; 3], // HACK
    cg: ::rsp2_minimize::acgsd::Settings,
}

// fn array_sum(arrs: &[[f64; 3]]) -> [f64; 3] {
//     let mut acc = [0.0, 0.0, 0.0];
//     for arr in arrs {
//         acc = vec_from_fn(|k| acc[k] + arr[k]);
//     }
//     acc
// }

// fn array_mean(arrs: &[[f64; 3]]) -> [f64; 3] {
//     assert!(arrs.len() > 0);
//     let out = array_sum(arrs);
//     vec_from_fn(|k| out[k] / arrs.len() as f64)
// }

// fn remove_mean_shift(a: &mut [[f64; 3]], b: &[[f64; 3]]) {
//     let shifts = v(a.flat()) - v(b.flat());
//     let mean = array_mean(shifts.nest());
//     for row in a {
//         *row = vec_from_fn(|k| row[k] - mean[k]);
//     }
// }

fn lammps_flat_diff_fn<'a>(lmp: &'a mut Lammps)
-> Box<FnMut(&[f64]) -> Result<(f64, Vec<f64>), LmpError> + 'a>
{
    Box::new(move |pos| {
        lmp.set_carts(pos.nest())?;
        lmp.compute().map(|(v,g)| (v, g.flat().to_vec()))
    })
}

pub enum Panic {}
impl<E: ::std::fmt::Debug> From<E> for Panic {
    fn from(e: E) -> Panic { Err::<(),_>(e).unwrap(); unreachable!() }
}

// fn numerical_lattice_param_slope(structure: &CoordStructure, mask: [f64; 3]) -> [f64; 3]
// {
//     vec![
//         -f64::exp2(-47.),
//         -f64::exp2(-48.),
//         -f64::exp2(-49.),
//         -f64::exp2(-50.),
//         -f64::exp2(-51.),
//         f64::exp2(-51.),
//         f64::exp2(-50.),
//         f64::exp2(-49.),
//         f64::exp2(-48.),
//         f64::exp2(-47.),
//     ].into_iter().map(|x| {
//         let scales = [1.0 + mask[0] * x, 1.0 + mask[1] * x, 1.0 + mask[2] * x];
//         let mut structure = structure.clone();
//         structure.scale_vecs(&scales);
//         // scale_vecs.
//         panic!("TODO")
//     });

//     *out.as_array().unwrap()
// }

pub fn run_relax_with_eigenvectors<P, Q>(settings: &Settings, input: P, outdir: Q) -> Result<(), Panic>
where P: AsRef<Path>, Q: AsRef<Path>,
{
    use ::std::io::prelude::*;
    use ::rsp2_phonopy_io as p;
    use ::rsp2_structure_io::poscar;
    use ::rsp2_slice_math::{v, V, vdot};
    use ::std::fs::File;

    let relax = |structure: CoordStructure| -> Result<CoordStructure, Panic> {
        let sc_dims = tup3(settings.supercell_relax.dim_for_unitcell(structure.lattice()));
        let (supercell, sc_token) = supercell::diagonal(sc_dims, structure);

        // FIXME confusing for Lammps::new_carbon to take initial position
        let mut lmp = Lammps::new_carbon(supercell.clone())?;
        let relaxed_flat = ::rsp2_minimize::acgsd(
            &settings.cg,
            &supercell.to_carts().flat(),
            &mut *lammps_flat_diff_fn(&mut lmp),
        )?.position;

        let supercell = supercell.with_coords(Coords::Carts(relaxed_flat.nest().to_vec()));
        Ok(multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?)
    };

    use ::rsp2_structure::supercell::{SupercellToken, DeconstructionError};
    fn multi_threshold_deconstruct(
        sc_token: SupercellToken,
        warn: f64,
        fail: f64,
        supercell: CoordStructure,
    ) -> Result<CoordStructure, DeconstructionError>
    {
        match sc_token.deconstruct(warn, supercell.clone()) {
            Ok(x) => Ok(x),
            Err(e) => {
                warn!("Suspiciously broad deviations in supercell: {:?}", e);
                Ok(sc_token.deconstruct(fail, supercell)?)
            }
        }
    }

    let diagonalize = |structure: CoordStructure| -> Result<_, Panic> {
        let conf = collect![
            (format!("DISPLACEMENT_DISTANCE"), format!("{:e}", settings.displacement_distance)),
            (format!("DIM"), {
                let (a, b, c) = tup3(settings.supercell_phonopy.dim_for_unitcell(structure.lattice()));
                format!("{} {} {}", a, b, c)
            }),
            (format!("HDF5"), format!(".TRUE.")),
            (format!("DIAG"), format!(".FALSE.")), // maybe?
        ];

        let (superstructure, displacements, disp_token)
            = p::cmd::phonopy_displacements_carbon(&conf, structure)?;

        let mut lmp = Lammps::new_carbon(superstructure.clone())?;
        println!();
        let mut i = 0;
        let force_sets = p::force_sets::compute_from_grad(
            superstructure.clone(), // FIXME only a borrow is needed later, and only for natom (dumb)
            &displacements,
            |s| {
                i += 1;
                print!("\rdisp {} of {}", i, displacements.len());
                ::std::io::stdout().flush().unwrap();
                lmp.set_structure(s.clone())?;
                lmp.compute().map(|diff| diff.1)
            }
        )?;
        println!();

        let (eval, evec) = p::cmd::phonopy_gamma_eigensystem(&conf, force_sets, &disp_token)?;
        let V(eval) = THZ_TO_WAVENUMBER * v(eval);
        Ok((eval, evec))
    };

    let minimize_evec = |structure: CoordStructure, evec: &[[f64; 3]]| -> Result<(f64, CoordStructure), Panic> {
        let sc_dims = tup3(settings.supercell_relax.dim_for_unitcell(structure.lattice()));
        let (structure, sc_token) = supercell::diagonal(sc_dims, structure);
        let evec = sc_token.replicate(evec);
        let mut lmp = Lammps::new_carbon(structure.clone())?;

        let from_structure = structure;
        let direction = &evec[..];
        let from_pos = from_structure.to_carts();
        let pos_at_alpha = |alpha| {
            let V(pos) = v(from_pos.flat()) + alpha * v(direction.flat());
            pos
        };
        let alpha = ::rsp2_minimize::exact_ls::<LmpError, _>(0.0, 1e-4, |alpha| {
            let gradient = lammps_flat_diff_fn(&mut lmp)(&pos_at_alpha(alpha))?.1;
            let slope = vdot(&gradient[..], direction.flat());
            Ok(::rsp2_minimize::exact_ls::Slope(slope))
        })??.alpha;
        let pos = pos_at_alpha(alpha);
        let structure = from_structure.with_coords(Coords::Carts(pos.nest().to_vec()));

        Ok((alpha, multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, structure)?))
    };

    let write_eigen_info = |f: &mut Write, einfos: &EigenInfo| -> Result<_, Panic>
    {
        use ::ansi_term::Colour::{Red, Cyan, Yellow, Black};
        use display_util::{ColorByRange, DisplayProb};

        let color_range = ColorByRange::new(vec![
            (0.999, Cyan.bold()),
            (0.9, Cyan.normal()),
            (0.1, Yellow.normal()),
            (1e-4, Red.bold()),
            (1e-10, Black.normal()),
        ], Red.normal());
        let dp = |x: f64| color_range.paint_as(&x, DisplayProb(x));
        let pol = |x: f64| color_range.paint(x);

        writeln!(f, "{:27}  {:^7}  {:^7} [{:^4}, {:^4}, {:^4}]",
            "# Frequency (cm^-1)", "Acoustc", "Layer", "X", "Y", "Z")?;
        for item in einfos.iter() {
            let eval = item.frequency;
            let acou = dp(item.acousticness);
            let layer = dp(item.layer_acousticness);
            let (x, y, z) = tup3(item.polarization);
            writeln!(f, "{:27}  {}  {} [{:4.2}, {:4.2}, {:4.2}]",
                eval, acou, layer, pol(x), pol(y), pol(z))?;
        }
        Ok(())
    };

    let mut original = poscar::load_carbon(File::open(input)?)?;
    original.scale_vecs(&settings.hack_scale); // HACK
    let original = original;

    ::std::fs::create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;

        let mut from_structure = original;
        // HACK to stop one iteration AFTER all non-acoustics are positive
        let mut iteration = 1;
        let mut all_ok_count = 0;
        let (structure, evals, _evecs) = loop { // NOTE: we use break with value
            let structure = relax(from_structure)?;
            let (evals, evecs) = diagonalize(structure.clone())?;

            println!("============================");
            println!("Finished relaxation # {}", iteration);

            let (layers, _nlayer) = ::rsp2_structure::assign_layers(&structure, &[0, 0, 1], 0.25);
            eprintln!("{:?}", layers);
            let einfos = get_eigensystem_info(&evals, &evecs, &layers[..]);
            write_eigen_info(
                &mut File::create(format!("eigenvalues.{:02}", iteration))?,
                &einfos,
            )?;
            {
                let out = ::std::io::stdout();
                write_eigen_info(&mut out.lock(), &einfos)?;
            }

            let mut all_ok = true;
            let mut structure = structure;
            for (i, info, evec) in izip!(1.., &einfos, &evecs) {
                if info.frequency < 0.0 && info.acousticness < 0.95 {
                    if all_ok {
                        println!("!! Optimizing along bad bands");
                        all_ok = false;
                    }
                    let (alpha, new_structure) = minimize_evec(structure, &evec[..])?;
                    println!("!! Optimized along band {} ({}), a = {:e}", i, info.frequency, alpha);

                    structure = new_structure;
                }
            }

            if all_ok {
                all_ok_count += 1;
                if all_ok_count >= 3 {
                    break (structure, evals, evecs);
                }
            }

            println!("============================");

            from_structure = structure;
            iteration += 1;
        }; // (structure, evals, evecs)

        {
            let mut f = File::create("eigenvalues.final")?;
            for &x in &evals {
                writeln!(f, "{:?}", x)?;
            }
        }

        poscar::dump_carbon(File::create("./final.vasp")?, "", &structure)?;

        cwd_guard.pop()?;
    }

    Ok(())
}


macro_rules! each_fmt_trait {
    ($mac:ident!)
    => {
        $mac!(::std::fmt::Display);
        $mac!(::std::fmt::Octal);
        $mac!(::std::fmt::LowerHex);
        $mac!(::std::fmt::UpperHex);
        $mac!(::std::fmt::Pointer);
        $mac!(::std::fmt::Binary);
        $mac!(::std::fmt::LowerExp);
        $mac!(::std::fmt::UpperExp);
    }
}

mod display_util {
    use ::std::fmt;
    use ::ansi_term::Style;


    /// Specialized display impl for numbers that from 0 to 1 and may be
    /// extremely close to either 0 or 1
    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
    pub struct DisplayProb(pub f64);
    impl fmt::Display for DisplayProb {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let log10_1p = |x: f64| x.ln_1p() / ::std::f64::consts::LN_10;
            assert!(0.0 < self.0 && self.0 < 1.0 + 1e-5,
                "bad probability: {}", self.0);

            if self.0 >= 1.0 {
                write!(f, "{:>7}", 1.0)
            } else if self.0 <= 1e-3 {
                write!(f, "  1e{:03}", self.0.log10().round())
            } else if self.0 + 1e-3 >= 1.0 {
                write!(f, "1-1e{:03}", log10_1p(-self.0).round())
            } else {
                write!(f, "{:<7.5}", self.0)
            }
        }
    }

    pub struct ColorByRange<T> {
        pub divs: Vec<(T, Style)>,
        pub lowest: Style,
    }
    impl<T> ColorByRange<T> {
        pub fn new(divs: Vec<(T, Style)>, lowest: Style) -> ColorByRange<T> {
            ColorByRange { divs, lowest }
        }

        fn style_of(&self, x: &T) -> Style
        where T: PartialOrd,
        {
            for &(ref pivot, style) in &self.divs {
                if x > pivot { return style; }
            }
            return self.lowest;
        }

        pub fn paint<'a, U>(&self, x: U) -> super::term::Wrapper<U, T>
        where
            T: PartialOrd + 'a,
            U: ::std::borrow::Borrow<T> + 'a,
        {
            super::term::gpaint(self.style_of(x.borrow()), x)
        }

        pub fn paint_as<'a, U>(&self, compare_me: &T, show_me: U) -> super::term::Wrapper<U, U>
        where T: PartialOrd,
        {
            super::term::paint(self.style_of(compare_me), show_me)
        }
    }
}

mod term {
    // hack for type  inference issues
    pub fn paint<T>(
        style: ::ansi_term::Style,
        value: T,
    ) -> Wrapper<T, T>
    { gpaint(style, value) }

    pub fn gpaint<U, T>(
        style: ::ansi_term::Style,
        value: U,
    ) -> Wrapper<U, T>
    { Wrapper { style, value, _target: Default::default() } }

    /// A wrapper for colorizing all formatting traits like `Display`.
    ///
    /// Except `Debug`.
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Wrapper<U, T=U> {
        style: ::ansi_term::Style,
        value: U,
        _target: ::std::marker::PhantomData<T>,
    }

    use std::fmt;

    macro_rules! derive_fmt_impl {
        ($Trait:path)
        => {
            impl<U, T> $Trait for Wrapper<U, T>
            where
                U: ::std::borrow::Borrow<T>,
                T: $Trait,
            {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "{}", self.style.prefix())?;
                    T::fmt(self.value.borrow(), f)?;
                    write!(f, "{}", self.style.suffix())?;
                    Ok(())
                }
            }
        };
    }

    each_fmt_trait!{derive_fmt_impl!}
}


pub type EigenInfo = Vec<eigen_info::Item>;

pub fn get_eigensystem_info(
    evals: &[f64],
    evecs: &[Vec<[f64; 3]>],
    layers: &[::rsp2_structure::Layer],
) -> EigenInfo
{
    use ::rsp2_eigenvector_classify::{keyed_acoustic_basis, polarization};

    let layers: Vec<_> = layers.iter().cloned().map(Some).collect();
    let layer_acoustics = keyed_acoustic_basis(&layers[..], &[1,1,1]);
    let acoustics = keyed_acoustic_basis(&vec![Some(()); evecs[0].len()], &[1,1,1]);

    let mut out = vec![];
    for (&eval, evec) in izip!(evals, evecs) {
        out.push(eigen_info::Item {
            frequency: eval,
            acousticness: acoustics.probability(&evec),
            layer_acousticness: layer_acoustics.probability(&evec),
            polarization: polarization(&evec[..]),
        })
    }
    out
}

pub mod eigen_info {
    pub struct Item {
        pub frequency: f64,
        pub acousticness: f64,
        pub layer_acousticness: f64,
        pub polarization: [f64; 3],
    }
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum SupercellSpec {
    Target([f64; 3]),
    Dim([u32; 3]),
}
impl SupercellSpec {
    fn dim_for_unitcell(&self, prim: &Lattice) -> [u32; 3] {
        match *self {
            SupercellSpec::Dim(d) => d,
            SupercellSpec::Target(targets) => {
                let unit_lengths = prim.lengths();
                vec_from_fn(|k| {
                    (targets[k] / unit_lengths[k]).ceil().max(1.0) as u32
                })
            },
        }
    }
}

// HACK
fn tup3<T:Copy>(arr: [T; 3]) -> (T, T, T) { (arr[0], arr[1], arr[2]) }

use util::push_dir;
mod util {
    use ::std::path::{Path, PathBuf};
    use ::std::io::Result as IoResult;

    /// RAII type to temporarily enter a directory.
    ///
    /// The recommended usage is actually not to rely on the implicit destructor
    /// (which panics on failure), but to instead explicitly call `.pop()`.
    /// The advantage of doing so over just manually calling 'set_current_dir'
    /// is the unused variable lint can help remind you to call `pop`.
    ///
    /// Usage is highly discouraged in multithreaded contexts where
    /// another thread may need to access the filesystem.
    #[must_use]
    pub struct PushDir(Option<PathBuf>);
    pub fn push_dir<P: AsRef<Path>>(path: P) -> IoResult<PushDir> {
        let old = ::std::env::current_dir()?;
        ::std::env::set_current_dir(path)?;
        Ok(PushDir(Some(old)))
    }

    impl PushDir {
        /// Explicitly destroy the PushDir.
        ///
        /// This lets you handle the IO error, and has an advantage over
        /// manual calls to 'env::set_current_dir' in that the compiler will
        pub fn pop(mut self) -> IoResult<()> {
            ::std::env::set_current_dir(self.0.take().unwrap())
        }
    }

    impl Drop for PushDir {
        fn drop(&mut self) {
            if let Some(d) = self.0.take() {
                if let Err(e) = ::std::env::set_current_dir(d) {
                    // uh oh.
                    panic!("automatic popdir failed: {}", e);
                }
            }
        }
    }
}
