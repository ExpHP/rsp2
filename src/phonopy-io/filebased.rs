//! Filesystem-based API.
//!
//! This API is centered around two types of objects:
//!
//! - Wrappers around directory types (which are generic in the
//!   directory type and may contain TempDirs, PathBufs, or even
//!   borrowed &Paths), representing a directory where some
//!   computation was run.
//! - Builders for configuring the next computation.
//!
//! The directory wrapper types allow the user to be able to
//! explicitly "keep" a phonopy output directory, and potentially
//! even reuse it in a future computation. The inability to do these
//! things was the biggest limitation of the previous API, which never
//! exposed its own temporary directories.

use ::{Result, IoResult, ErrorKind};
use ::As3;

use ::filetypes::{conf, args, q_positions};
use ::filetypes::{symmetry_yaml, disp_yaml, force_sets};

use ::rsp2_structure_io::poscar;
use ::std::io::prelude::BufRead;
use ::std::process::Command;
use ::std::path::{Path, PathBuf};
use ::rsp2_fs_util::{open, open_text, create, copy, hard_link, mv, rm_rf};
use ::tempdir::TempDir;
use ::std::collections::HashMap;

use ::rsp2_kets::Basis;
use ::rsp2_structure::{CoordStructure, ElementStructure};
use ::rsp2_structure::{FracRot, FracTrans, FracOp};

use ::slice_of_array::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Builder {
    symprec: Option<f64>,
    conf: HashMap<String, String>,
}

impl Builder {
    pub fn new() -> Self
    { Default::default() }

    pub fn symmetry_tolerance(mut self, x: f64) -> Self
    { self.symprec = Some(x); self }

    pub fn conf<K: AsRef<str>, V: AsRef<str>>(mut self, key: K, value: V) -> Self
    { self.conf.insert(key.as_ref().to_owned(), value.as_ref().to_owned()); self }

    /// Read configuration from a phonopy .conf file,
    /// overwriting existing values.
    // FIXME: Result<Self>... oh, THAT's why the general recommendation is for &mut Self
    pub fn conf_from_file<R: BufRead>(self, file: R) -> Result<Self>
    {Ok({
        let mut me = self;
        for (key, value) in ::filetypes::conf::read(file)? {
            me = me.conf(key, value);
        }
        me
    })}

    pub fn supercell_dim<V: As3<u32>>(self, dim: V) -> Self
    {
        self.conf("DIM", {
            let (a, b, c) = dim.as_3();
            format!("{} {} {}", a, b, c)
        })
    }

    fn args_from_settings(&self) -> Vec<String>
    {
        let mut out = vec![];
        if let Some(tol) = self.symprec {
            out.push(format!("--tolerance"));
            out.push(format!("{:e}", tol));
        }
        out
    }
}

impl Builder {
    pub fn displacements(
        &self,
        structure: &ElementStructure,
    ) -> Result<DirWithDisps<TempDir>>
    {Ok({
        let dir = TempDir::new("rsp2")?;
        {
            let dir = dir.path();
            trace!("Displacement dir: '{}'...", dir.display());

            let extra_args = self.args_from_settings();
            conf::write(create(dir.join("disp.conf"))?, &self.conf)?;
            poscar::dump(create(dir.join("POSCAR"))?, "blah", &structure)?;
            args::write(create(dir.join("disp.args"))?, &extra_args)?;

            trace!("Calling phonopy for displacements...");
            {
                let mut command = Command::new("phonopy");
                command
                    .args(&extra_args)
                    .arg("disp.conf")
                    .arg("--displacement")
                    .current_dir(&dir);

                log_stdio_and_wait(command)?;
            }
        };

        DirWithDisps::from_existing(dir)?
    })}

    // FIXME: Should return a new DirWithSymmetry type
    pub fn symmetry(
        &self,
        structure: &ElementStructure,
    ) -> Result<(Vec<FracOp>)>
    {Ok({
        let tmp = TempDir::new("rsp2")?;
        let tmp = tmp.path();
        trace!("Entered '{}'...", tmp.display());

        conf::write(create(tmp.join("phonopy.conf"))?, &self.conf)?;

        poscar::dump(create(tmp.join("POSCAR"))?, "blah", &structure)?;

        trace!("Calling phonopy for symmetry...");
        check_status(Command::new("phonopy")
            .args(self.args_from_settings())
            .arg("phonopy.conf")
            .arg("--sym")
            .current_dir(&tmp)
            .stdout(create(tmp.join("symmetry.yaml"))?)
            .status()?)?;

        trace!("Done calling phonopy");

        // check if input structure was primitive
        {
            let prim = poscar::load(open(tmp.join("PPOSCAR"))?)?;

            let ratio = structure.lattice().volume() / prim.lattice().volume();
            let ratio = round_checked(ratio, 1e-4)?;

            // sorry, supercells are just not supported... yet.
            //
            // (In the future we may be able to instead return an object
            //  which will allow the spacegroup operators of the primitive
            //  to be applied in meaningful ways to the superstructure.)
            ensure!(ratio == 1, ErrorKind::NonPrimitiveStructure);
        }

        let yaml = symmetry_yaml::read(open(tmp.join("symmetry.yaml"))?)?;
        yaml.space_group_operations.into_iter()
            .map(|op| Ok({
                let rotation = FracRot::new(&op.rotation);
                let translation = FracTrans::from_floats(&op.translation)?;
                FracOp::new(&rotation, &translation)
            }))
            .collect::<Result<_>>()?
    })}
}

// Declares a type whose body is hidden behind an Option,
// which can be taken to "poison" (consume) the value.
//
// This is a design pattern for &mut Self-based builders.
macro_rules! declare_poison_pair {
    (
        generics: { $($generics:tt)* }
        where: { $($bounds:tt)* }
        type: {
            #[derive($($derive:ident),*)]
            pub struct $Type:ident<...>(Option<_>);
            struct $Impl:ident<...> { $($body:tt)* }
        }
        poisoned: $poisoned:block
    ) => {
        #[derive($($derive),*)]
        pub struct $Type<$($generics)*>(Option<$Impl<$($generics)*>>)
        where $($bounds)*;

        #[derive($($derive),*)]
        struct $Impl<$($generics)*>
        where $($bounds)*
        { $($body)* }

        impl<$($generics)*> $Type<$($generics)*> where $($bounds)*
        {
            // modify self if not poisoned
            fn inner_mut(&mut self) -> &mut $Impl<$($generics)*>
            { match self.0 {
                Some(ref mut inner) => inner,
                None => $poisoned,
            }}

            // poisons self
            fn into_inner(&mut self) -> $Impl<$($generics)*>
            { match self.0.take() {
                Some(inner) => inner,
                None => $poisoned,
            }}
        }

    }
}

/// Represents a directory with the following data:
/// - `POSCAR`: The input structure
/// - `disp.yaml`: Phonopy file with displacements
/// - configuration settings which impact the selection of displacements
///   - `--tol`, `--dim`
///
/// Generally, the next step is to supply the force sets, turning this
/// into a DirWithForces.
///
/// # Note
///
/// Currently, the implementation is rather optimistic that files in
/// the directory have not been tampered with since its creation.
/// As a result, some circumstances which probably should return `Error`
/// may instead cause a panic, or may not be detected as early as possible.
#[derive(Debug, Clone)]
pub struct DirWithDisps<P: AsPath> {
    pub(crate) dir: P,
    // These are cached in memory from `disp.yaml` due to the likelihood
    // that code using `DirWithDisps` will need them.
    // FIXME: Should be ElementStructure, but this needs `try_map_metadata_to` or
    //        similar.  I guess we could also expose the DispYaml metadata type,
    //        but it'd be unnecessarily annoying to work with in user code,
    //        and we're already commited to ElementStructure as an input type.
    pub(crate) superstructure: CoordStructure,
    pub(crate) displacements: Vec<(usize, [f64; 3])>,
}

impl<P: AsPath> DirWithDisps<P> {
    pub fn from_existing(dir: P) -> Result<Self>
    {Ok({
        for name in &[
            "POSCAR",
            "disp.yaml",
            "disp.conf",
            "disp.args",
        ] {
            let path = dir.as_path().join(name);
            ensure!(path.exists(),
                ErrorKind::MissingFile("DirWithDisps", dir.as_path().to_owned(), name.to_string()));
        }

        trace!("Parsing disp.yaml...");
        let disp_yaml::DispYaml {
            displacements, structure: superstructure
        } = disp_yaml::read(open(dir.as_path().join("disp.yaml"))?)?;
        let superstructure = superstructure.map_metadata_into(|_| ());

        DirWithDisps { dir, superstructure, displacements }
    })}

    /// FIXME: This should be an ElementStructure.
    pub fn superstructure(&self) -> &CoordStructure
    { &self.superstructure }
    pub fn displacements(&self) -> &[(usize, [f64; 3])]
    { &self.displacements }

    /// Due to the "StreamingIterator problem" this iterator must return
    /// clones of the structure... though it seems unlikely that this cost
    /// is anything to worry about compared to the cost of computing the
    /// forces on said structure.
    ///
    /// FIXME: This should give ElementStructure.
    pub fn displaced_structures<'a>(&'a self) -> Box<Iterator<Item=CoordStructure> + 'a>
    {Box::new({
        use ::filetypes::disp_yaml::apply_displacement;
        self.displacements
            .iter()
            .map(move |&disp| apply_displacement(&self.superstructure, disp))
    })}

    /// Write FORCE_SETS to create a `DirWithForces`
    /// (which may serve as a template for one or more band computations).
    ///
    /// This variant creates a new temporary directory.
    pub fn make_force_dir<V>(self, forces: &[V]) -> Result<DirWithForces<TempDir>>
    where V: AsRef<[[f64; 3]]>,
    {Ok({
        let out = TempDir::new("rsp2")?;
        trace!("Force sets dir: '{}'...", out.path().display());

        self.make_force_dir_in_dir(&forces, out)?
    })}

    /// Write FORCE_SETS to create a `DirWithForces`
    /// (which may serve as a template for one or more band computations).
    ///
    /// This variant uses the specified path.
    pub fn make_force_dir_in_dir<V, Q>(self, forces: &[V], path: Q) -> Result<DirWithForces<Q>>
    where
        V: AsRef<[[f64; 3]]>,
        Q: AsPath,
    {Ok({
        let forces: Vec<_> = forces.iter().map(|x: &_| x.as_ref()).collect();
        self.prepare_force_dir(&forces, path.as_path())?;
        DirWithForces::from_existing(path)?
    })}

    fn prepare_force_dir(self, forces: &[&[[f64; 3]]], path: &Path) -> Result<()>
    {Ok({
        let disp_dir = self.path();

        for name in &["POSCAR", "disp.yaml", "disp.conf", "disp.args"] {
            copy(disp_dir.join(name), path.join(name))?;
        }

        trace!("Writing FORCE_SETS...");
        force_sets::write(
            create(path.join("FORCE_SETS"))?,
            &self.superstructure,
            &self.displacements,
            &forces,
        )?;
    })}
}

/// Represents a directory with the following data:
/// - `POSCAR`: The input structure
/// - `FORCE_SETS`: Phonopy file with forces for displacements
/// - configuration settings which impact the selection of displacements
///   - `--tol`, `--dim`
///
/// It may also contain a force constants file.
///
/// Generally, one produces a `DirWithBands` from this by selecting some
/// q-points to sample.
///
/// # Note
///
/// Currently, the implementation is rather optimistic that files in
/// the directory have not been tampered with since its creation.
/// As a result, some circumstances which probably should return `Error`
/// may instead cause a panic, or may not be detected as early as possible.
#[derive(Debug, Clone)]
pub struct DirWithForces<P: AsPath> {
    dir: P,
    cache_force_constants: bool,
}

impl<P: AsPath> DirWithForces<P> {
    pub fn from_existing(dir: P) -> Result<Self>
    {Ok({
        // Sanity check
        for name in &[
            "POSCAR",
            "FORCE_SETS",
            "disp.args",
            "disp.conf",
        ] {
            let path = dir.as_path().join(name);
            if !path.exists() {
                ensure!(path.exists(),
                    ErrorKind::MissingFile("DirWithForces", dir.as_path().to_owned(), name.to_string()));
            }
        }
        DirWithForces { dir, cache_force_constants: true }
    })}

    pub fn structure(&self) -> Result<ElementStructure>
    { Ok(poscar::load(open_text(self.path().join("POSCAR"))?)?) }

    /// Enable/disable caching of force constants.
    ///
    /// When enabled (the default), force constants are copied back from
    /// the first successful `DirWithBands` created, and are reused in
    /// subsequent band computations. (these copies are hard links if possible)
    pub fn cache_force_constants(&mut self, b: bool) -> &mut Self
    { self.cache_force_constants = b; self }

    /// Compute bands in a temp directory.
    ///
    /// Returns an object used to configure the computation.
    pub fn build_bands(&self) -> BandsBuilder<P, TempDir>
    { BandsBuilder::init(self, MaybeDeferred::Deferred(make_temp_bands_dir)) }

    // FIXME experimental API, but `relocate` may be safer/saner
    /// Compute bands, running in a given directory.
    ///
    /// Returns an object used to configure the computation.
    ///
    /// Beware! Existing files in the given directory may be overwritten.
    pub fn build_bands_in_dir<Q: AsPath>(&self, dir: Q) -> BandsBuilder<P, Q>
    { BandsBuilder::init(self, MaybeDeferred::Value(dir)) }
}

#[derive(Debug, Clone)]
enum MaybeDeferred<T> {
    Value(T),
    Deferred(fn() -> Result<T>),
}

fn make_temp_bands_dir() -> Result<TempDir>
{Ok({
    let tmp = TempDir::new("rsp2")?;
    trace!("Bands directory: {}", tmp.path().display());
    tmp
})}

declare_poison_pair! {
    generics: {'p, P, Q}
    where: {
        P: AsPath + 'p,
        Q: AsPath,
    }
    type: {
        #[derive(Debug, Clone)]
        pub struct BandsBuilder<...>(Option<_>);
        struct BandsBuilderImpl<...> {
            dir_with_forces: &'p DirWithForces<P>,
            directory: MaybeDeferred<Q>,
            eigenvectors: bool,
        }
    }
    poisoned: { panic!("This BandsBuilder has already been used!"); }
}

impl<'p, P: AsPath, Q: AsPath> BandsBuilder<'p, P, Q> {
    // Q must be provided in advance to dodge issues with type inference
    fn init(dir_with_forces: &'p DirWithForces<P>, directory: MaybeDeferred<Q>) -> Self
    { BandsBuilder(Some(BandsBuilderImpl {
        dir_with_forces,
        directory: directory,
        eigenvectors: false,
    })) }

    pub fn eigenvectors(&mut self, b: bool) -> &mut Self
    { self.inner_mut().eigenvectors = b; self }

    pub fn compute(&mut self, q_points: &[[f64; 3]]) -> Result<DirWithBands<Q>>
    {Ok({
        let me = self.into_inner();
        let dir = match me.directory {
            MaybeDeferred::Value(d) => d,
            MaybeDeferred::Deferred(f) => f()?,
        };

        let fc_filename = "force_constants.hdf5";
        {
            let src = me.dir_with_forces.as_path();
            let dir = dir.as_path();

            copy(src.join("POSCAR"), dir.join("POSCAR"))?;
            copy(src.join("disp.conf"), dir.join("disp.conf"))?;
            copy(src.join("disp.args"), dir.join("disp.args"))?;

            // NOTE: this rm_rf carries with it the limitation
            //        limitation that the force dir cannot be the band dir.
            //       Not sure whether that's a reasonable use case or not...
            rm_rf(dir.join("FORCE_SETS"))?;
            hard_link(src.join("FORCE_SETS"), dir.join("FORCE_SETS"))?;

            if dir.join(fc_filename).exists() {
                rm_rf(dir.join(fc_filename))?;
                hard_link(src.join(fc_filename), dir.join(fc_filename))?;
            }

            q_positions::write(create(dir.join("q-positions.json"))?, q_points)?;

            // band.conf
            {
                // Carry over settings from displacements.
                let mut conf = conf::read(open_text(dir.join("disp.conf"))?)?;

                // Append a dummy qpoint so that each of our points begin a line segment.
                conf.insert("BAND".to_string(), band_string(q_points) + " 0 0 0");
                // NOTE: using 1 band point does result in a (currently inconsequential) divide-by-zero
                //       warning in phonopy, but cuts the filesize of band output by half compared to 2
                //       points. Considering how large the band output is, I'll take my chances!
                //          - ML
                conf.insert("BAND_POINTS".to_string(), "1".to_string());
                conf.insert("EIGENVECTORS".to_string(), match me.eigenvectors {
                    true => ".TRUE.".to_string(),
                    false => ".FALSE.".to_string(),
                });
                conf::write(create(dir.join("band.conf"))?, &conf)?;
            }

            trace!("Calling phonopy for bands...");
            {
                let mut command = Command::new("phonopy");
                command
                    .args(args::read(open(dir.join("disp.args"))?)?)
                    .arg("band.conf")
                    .arg("--hdf5")
                    .arg(match dir.join(fc_filename).exists() {
                        true => "--readfc",
                        false => "--writefc",
                    })
                    .env("EIGENVECTOR_NPY_HACK", "1")
                    .current_dir(&dir);

                log_stdio_and_wait(command)?;
            }

            if me.dir_with_forces.cache_force_constants {
                cache_link(dir.join(fc_filename), src.join(fc_filename))?;
            }
        }

        DirWithBands::from_existing(dir)?
    })}
}

/// Represents a directory with the following data:
/// - input structure
/// - q-points
/// - eigenvalues (and possibly eigenvectors)
///
/// # Note
///
/// Currently, the implementation is rather optimistic that files in
/// the directory have not been tampered with since its creation.
/// As a result, some circumstances which probably should return `Error`
/// may instead cause a panic, or may not be detected as early as possible.
#[derive(Debug, Clone)]
pub struct DirWithBands<P: AsPath> {
    dir: P,
}

impl<P: AsPath> DirWithBands<P> {
    pub fn from_existing(dir: P) -> Result<Self>
    {Ok({
        // Sanity check
        for name in &[
            "POSCAR",
            "FORCE_SETS",
            "eigenvalue.npy",
            "q-positions.json",
        ] {
            let path = dir.as_path().join(name);
            if !path.exists() {
                ensure!(path.exists(),
                    ErrorKind::MissingFile("DirWithBands", dir.as_path().to_owned(), name.to_string()));
            }
        }

        DirWithBands { dir }
    })}

    pub fn structure(&self) -> Result<ElementStructure>
    { Ok(poscar::load(open_text(self.path().join("POSCAR"))?)?) }

    pub fn q_positions(&self) -> Result<Vec<[f64; 3]>>
    { Ok(q_positions::read(open(self.path().join("q-positions.json"))?)? )}

    /// This will be `None` if `.eigenvectors(true)` was not set prior
    /// to the band computation.
    pub fn eigenvectors(&self) -> Result<Option<Vec<Basis>>>
    {Ok({
        let path = self.path().join("eigenvector.npy");
        if path.exists() {
            trace!("Reading eigenvectors...");
            Some(::npy::read_eigenvector_npy(open(path)?)?)
        } else { None }
    })}

    pub fn eigenvalues(&self) -> Result<Vec<Vec<f64>>>
    {
        trace!("Reading eigenvectors...");
        let file = open(self.path().join("eigenvalue.npy"))?;
        ::npy::read_eigenvalue_npy(file)
    }

    // ick. Sloppy design, but it's the best we can do without
    // having a separate directory type just for Gamma computations
    /// Read the eigensystem at Gamma.
    ///
    /// There are a couple of reasons this could (partially) fail:
    /// * It must read several files, which may produce Err.
    /// * One of the q-positions requested must have been at gamma!
    ///   Otherwise, the outer value will be `None`.
    /// * The eigenvector will be `None` if eigenvectors were not computed.
    pub fn gamma_eigensystem(&self)
    -> Result<Option<(Vec<f64>, Option<Vec<Vec<[f64; 3]>>>)>>
    {Ok({
        // search for gamma.
        let index = self.q_positions()?.iter().position(|&x| x == [0_f64; 3]);
        let index = match index {
            None => { return Ok(None); }
            Some(i) => i,
        };

        // read the *entire* set of eigensystems at *all* q-points,
        //  just to extract the one at gamma.
        // ...What can I say? We've painted ourselves into a corner here.  _/o\_
        let eigenvalues = self.eigenvalues()?[index].clone();
        let eigenvectors = self.eigenvectors()?.map(|bases| {
            trace!("Getting real...");
            bases[index].iter().map(|ev| Ok({
                ev.iter().map(|c| {
                    ensure!(c.imag == 0.0, "non-real gamma eigenvector");
                    Ok(c.real)
                }).collect::<Result<Vec<_>>>()?.nest().to_vec()
            })).collect::<Result<_>>()
        });

        let eigenvectors = match eigenvectors {
            None => None,
            Some(Err(e)) => Err(e)?,
            Some(Ok(v)) => Some(v),
        };

        Some((eigenvalues, eigenvectors))
    })}
}

//-----------------------------

fn band_string(ks: &[[f64; 3]]) -> String
{
    ks.flat().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

fn round_checked(x: f64, tol: f64) -> Result<i32>
{Ok({
    let r = x.round();
    ensure!((r - x).abs() < tol, "not nearly integral: {}", x);
    r as i32
})}

pub(crate) fn log_stdio_and_wait(mut cmd: ::std::process::Command) -> Result<()>
{Ok({
    use ::std::process::Stdio;
    use ::std::io::{BufRead, BufReader};

    debug!("$ {:?}", cmd);

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout_worker = {
        let f = BufReader::new(child.stdout.take().unwrap());
        ::std::thread::spawn(move || -> Result<()> {Ok({
            for line in f.lines() {
                ::stdout::log(&(line?[..]));
            }
        })})
    };

    let stderr_worker = {
        let f = BufReader::new(child.stderr.take().unwrap());
        ::std::thread::spawn(move || -> Result<()> {Ok({
            for line in f.lines() {
                ::stderr::log(&(line?[..]));
            }
        })})
    };

    check_status(child.wait()?)?;

    let _ = stdout_worker.join();
    let _ = stderr_worker.join();
})}

fn check_status(status: ::std::process::ExitStatus) -> Result<()>
{Ok({
    ensure!(status.success(), ErrorKind::PhonopyFailed(status));
})}

// Wrapper around `hard_link` which:
// - falls back to copying if the destination is on another filesystem.
// - does not fail if the destination exists (it is simply left behind)
//
// The use case is where this may be used on many identical source files
// for the same destination, possibly concurrently, in order to cache the
// file for further reuse.
pub fn cache_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> Result<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    hard_link(src, dest)
        .map(|_| ())
        .or_else(|_| Ok({
            // if the file already existed, the link will have failed.
            // Check this before continuing because we don't want to
            //   potentially overwrite a link with a copy.
            if dest.exists() {
                return Ok(());
            }

            // assume the error was due to being on a different filesystem.
            // (Even if not, we will probably just get the same error)
            copy(src, dest)?;
        }))
}

//-----------------------------

/// AsRef<Path> with more general impls on smart pointer types.
///
/// (for instance, `Box<AsPath>` and `Rc<TempDir>` both implement
///  the trait)
pub trait AsPath {
    fn as_path(&self) -> &Path;
}

macro_rules! as_path_impl {
    (@AsRef [$($generics:tt)*] $Type:ty)
    => {
        impl<$($generics)*> AsPath for $Type {
            fn as_path(&self) -> &Path { self.as_ref() }
        }
    };
    (@Deref [$($generics:tt)*] $Type:ty)
    => {
        impl<$($generics)*> AsPath for $Type {
            fn as_path(&self) -> &Path { (&**self).as_path() }
        }
    };
    ( $(
        (by $tag:tt) [$($generics:tt)*] $Type:ty;
    )+ )
    => {
        $( as_path_impl!{@$tag [$($generics)*] $Type} )*
    };
}

as_path_impl!{
    (by AsRef) [] Path;
    (by AsRef) [] PathBuf;
    (by AsRef) [] TempDir;
    (by AsRef) [] ::std::ffi::OsString;
    (by AsRef) [] ::std::ffi::OsStr;
    (by AsRef) [] str;
    (by AsRef) [] String;
    (by AsRef) ['a] ::std::path::Iter<'a>;
    (by AsRef) ['a] ::std::path::Components<'a>;
    (by Deref) ['p, P: AsPath + ?Sized] &'p mut P;
    (by Deref) [P: AsPath + ?Sized] Box<P>;
    (by Deref) [P: AsPath + ?Sized] ::std::rc::Rc<P>;
    (by Deref) [P: AsPath + ?Sized] ::std::sync::Arc<P>;
    (by Deref) ['p, P: AsPath + ToOwned + ?Sized] ::std::borrow::Cow<'p, P>;
}

impl<'p, P: AsPath + ?Sized> AsPath for &'p P
{ fn as_path(&self) -> &Path { P::as_path(self) } }



/// Trait for types that own a temporary directory, which can be
/// released (to prevent automatic deletion) or explicitly closed
/// to catch IO errors (which would be ignored on drop).
///
/// This is really just an implementation detail, and you should not
/// worry about it. All types that implement this expose it through
/// the `close()` and `into_path()` inherent methods, so you do not
/// need to import it.
pub trait HasTempDir: AsPath {
    /// Provides `close()` in generic contexts
    fn temp_dir_close(self) -> IoResult<()>;
    /// Provides `into_path()` in generic contexts
    fn temp_dir_into_path(self) -> PathBuf;
}

macro_rules! impl_dirlike_boilerplate {
    (
        type: {$Type:ident<_>}
        member: self.$member:ident
        other_members: [$(self.$other_members:ident),*]
    ) => {
        // all dir-likes implement HasTempDir if possible
        impl<P: HasTempDir> HasTempDir for $Type<P> {
            fn temp_dir_close(self) -> IoResult<()>
            { self.$member.temp_dir_close() }

            fn temp_dir_into_path(self) -> PathBuf
            { self.$member.temp_dir_into_path() }
        }

        // all dir-likes implement AsPath
        impl<P: AsPath> AsPath for $Type<P> {
            fn as_path(&self) -> &Path { self.dir.as_path() }
        }

        // all dir-likes expose inherent methods that are aliases
        // for the HasTempDir and AsPath methods
        impl<P: HasTempDir> $Type<P> {
            /// Explicitly close the temporary directory, deleting it.
            ///
            /// This also happens when the object is dropped, but in that
            /// case it is not possible to detect errors.
            pub fn close(self) -> IoResult<()> { self.temp_dir_close() }

            /// Convert into a PathBuf, disabling this object's destructor.
            ///
            /// To retain the ability to call the other methods on this type,
            /// see the `keep()` method.
            pub fn into_path(self) -> PathBuf { self.temp_dir_into_path() }

            /// Move the directory to the given path, which must not exist.
            ///
            /// Currently, there is no recourse if the operation fails;
            /// the directory is simply lost. In the future, this may take
            /// '&mut self' and poison the object once the move has succeeded.
            pub fn relocate<Q: AsPath>(self, path: Q)
            -> Result<$Type<PathBuf>>
            {Ok({
                // (use something that supports cross-filesystem moves)
                mv(self.path(), path.as_path())?;

                self.map_dir(|old| {
                    // forget the TempDir
                    let _ = old.temp_dir_into_path();
                    // store the new path
                    path.as_path().to_owned()
                })
            })}
        }

        impl<P: AsPath> $Type<P> {
            pub fn path(&self) -> &Path { self.as_path() }

            /// Apply a function to change the type of the directory.
            /// For example, when `P = TempDir`, one could use `.map_dir(Rc::new)`
            ///  to enable cloning of the object.
            pub fn map_dir<Q, F>(self, f: F) -> $Type<Q>
            where
                Q: AsPath,
                F: FnOnce(P) -> Q,
            {
                let $member = f(self.$member);
                $(let $other_members = self.$other_members;)*
                $Type { $member, $($other_members),* }
            }
        }
    };
}

impl HasTempDir for TempDir {
    fn temp_dir_close(self) -> IoResult<()> { self.close() }
    fn temp_dir_into_path(self) -> PathBuf { self.into_path() }
}

impl_dirlike_boilerplate!{
    type: {DirWithDisps<_>}
    member: self.dir
    other_members: [self.displacements, self.superstructure]
}

impl_dirlike_boilerplate!{
    type: {DirWithForces<_>}
    member: self.dir
    other_members: [self.cache_force_constants]
}

impl_dirlike_boilerplate!{
    type: {DirWithBands<_>}
    member: self.dir
    other_members: []
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;

    // check use cases I need to work
    #[allow(unused)]
    fn things_expected_to_impl_aspath() {
        use ::std::rc::Rc;
        use ::std::sync::Arc;
        panic!("This is a compiletest; it should not be executed.");

        // clonable TempDirs
        let x: DirWithBands<TempDir> = panic!();
        let x = x.map_dir(Rc::new);
        let _ = x.clone();

        // sharable TempDirs
        let x: DirWithBands<TempDir> = panic!();
        let x = x.map_dir(Arc::new);
        let _: &(Send + Sync) = &x;

        // erased types, for conditional deletion
        let _: DirWithBands<Box<AsPath>> = x.map_dir(|e| Box::new(e) as _);
    }
}

