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

use ::std::ops::Deref;
use ::std::path::Path;
use ::std::io::prelude::*;
use ::itertools::Itertools;
use ::std::collections::{BTreeSet, BTreeMap};

use ::util;

use ::CheckFile;

pub use ::failure::Error;

use ::path_abs::{FileRead, FileWrite};
use ::rsp2_structure::{Coords, CartOp};
use ::rsp2_array_types::{V3, M33, Unvee};

#[macro_export]
macro_rules! impl_json {
    (($Type:ty) [$($methods:ident),*]) => {
        $( impl_json!{@one ($Type) [$methods]} )*
    };
    (@one ($Type:ty) [load]) => {
        #[allow(unused)]
        impl $Type {
            pub fn load(path: impl AsRef<::std::path::Path>) -> Result<Self, $crate::filetypes::Error> {
                $crate::filetypes::load_json(path)
            }
        }
    };
    (@one ($Type:ty) [save]) => {
        #[allow(unused)]
        impl $Type {
            pub fn save(&self, path: impl AsRef<::std::path::Path>) -> Result<(), $crate::filetypes::Error> {
                $crate::filetypes::save_json(path, self)
            }
        }
    }
}

#[derive(Debug, PartialEq)]
#[derive(Deserialize, Serialize)]
pub struct Frequencies(pub Vec<f64>);
impl Deref for Frequencies {
    type Target = Vec<f64>;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl_json!{ (Frequencies)[save, load] }

// for "keyword arguments"
#[derive(Debug, Clone, Copy)]
pub struct FrequencyTolerances {
    pub max_acoustic: f64,
    pub rel_tol: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RamanJsonTolerances {
    pub frequency: FrequencyTolerances,
    /// Intensities less than `intensity_zero_thresh * intensity.max()` are compared as "zerolike".
    /// Only their locations must correspond.
    pub intensity_nonzero_thresh: f64,
    /// Used for non-zerolike intensities
    pub intensity_nonzero_rel_tol: f64,
}

#[derive(Debug, PartialEq)]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct RamanJson {
    pub frequency: Frequencies,
    pub average_3d: Vec<f64>,
    pub backscatter: Vec<f64>,
}
impl_json!{ (RamanJson)[save, load] }

impl Frequencies {
    pub fn check_against(&self, expected: &Self, tol: FrequencyTolerances) {
        // Acoustic mode frequencies; magnitude is irrelevant as long as it is not large.
        assert!(expected.0[..3].iter().all(|x| x.abs() < tol.max_acoustic), "bad expected acoustics!");
        assert!(self.0[..3].iter().all(|x| x.abs() < tol.max_acoustic), "bad acoustics!");

        for (a, b) in zip_eq!(&self.0[3..], &expected.0[3..]) {
            assert_close!(rel=tol.rel_tol, a, b);
        }
    }
}

/// Implements a comparison which requires that either both values are less than
/// some absolute value ("zerolike"), or that one is within a relative tolerance
/// of the other.
#[derive(Copy, Clone)]
pub struct MaybeZerolike(pub f64);
impl_newtype_debug!(MaybeZerolike);

#[derive(Debug, Copy, Clone)]
pub struct MaybeZerolikeTolerances {
    pub negative_ok: bool,
    pub zero_thresh: f64,
    pub rel_tol: f64,
}

impl MaybeZerolike {
    pub fn check_against(&self, expected: &Self, tol: MaybeZerolikeTolerances) {
        if !tol.negative_ok {
            assert!(expected.0 >= 0.0, "expected is negative! ({})", expected.0);
            assert!(self.0 >= 0.0, "value is negative! ({})", self.0);
        }

        // Succeed if either of the following are true:
        // * The lesser magnitude is within rel tol times the larger one
        // * Both are zerolike
        if f64::max(self.0.abs(), expected.0.abs()) < tol.zero_thresh {
            return;
        }
        assert_close!(rel=tol.rel_tol, self.0, expected.0);
    }
}

// data affected by choice of primitive structure
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Primitive {
    pub cart_ops: Vec<CartOp>,
    pub masses: Vec<f64>,
    #[serde(rename = "structure")]
    pub coords: Coords,
    // FIXME move to ForceConstants test
    pub displacements: Vec<(usize, V3)>, // [disp] -> (prim, v)
}
impl_json!{ (Primitive)[save, load] }

impl CheckFile for RamanJson {
    type OtherArgs = RamanJsonTolerances;

    fn read_file(path: &Path) -> Result<Self, Error> { Self::load(path) }

    fn check_against(&self, expected: &RamanJson, tol: RamanJsonTolerances) {
        self.frequency.check_against(&expected.frequency, tol.frequency);
        for &(actual, expected) in &[
            (&self.average_3d, &expected.average_3d),
            (&self.backscatter, &expected.backscatter),
        ] {
            // Even though there is no catastrophic cancellation in the sums
            // that produce raman intensities, very small values are still more sensitive
            // to subtle changes in the input than others, and those extremely close to
            // zero have been observed to differ by a factor of 2 or more.
            //
            // Hence we use a MaybeZerolike comparison, branching on magnitude relative to max.
            let &max = util::partial_max(expected).unwrap();
            let zero_thresh = max * tol.intensity_nonzero_thresh;
            for (&a, &b) in zip_eq!(actual, expected) {
                let a = MaybeZerolike(a);
                let b = MaybeZerolike(b);
                a.check_against(&b, MaybeZerolikeTolerances {
                    negative_ok: false,
                    zero_thresh: zero_thresh,
                    rel_tol: tol.intensity_nonzero_rel_tol,
                });
            }
        }
    }
}

#[derive(Debug, PartialEq)]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Dynmat {
    pub complex_blocks: Vec<(M33, M33)>,
    pub col: Vec<usize>,
    pub row_ptr: Vec<usize>,
    pub dim: (usize, usize),
}

impl Dynmat {
    #[allow(unused)]
    pub fn load(path: impl AsRef<::std::path::Path>) -> Result<Self, Error> {
        // we can't read NPZ in rust
        let _guard = ::rsp2_python::add_to_python_path();
        let _tmp = ::fsx::TempDir::new("rsp2")?;
        let json_path = _tmp.path().join("tmp.json");
        // FIXME awkward as heck to be using process::Command for this, should the rust wrapper
        //       maybe be public in rsp2_python rather than private in rsp2_tasks?
        assert!({ // FIXME: Error instead of panic?
            ::std::process::Command::new("python3")
                .arg("-m").arg("rsp2.cli.convert_dynmat")
                .arg("--keep")
                .arg(path.as_ref())
                .arg("--output").arg(&json_path)
                .status()?
                .success()
        });
        load_json(json_path)
    }
}

impl Dynmat {
    #[allow(unused)]
    pub fn save(&self, path: impl AsRef<::std::path::Path>) -> Result<(), Error> {
        // we can't write NPZ in rust
        let _guard = ::rsp2_python::add_to_python_path();

        let _tmp = ::fsx::TempDir::new("rsp2")?;
        let json_path = _tmp.path().join("tmp.json");
        save_json(&json_path, self)?;

        assert!({ // FIXME: Error instead of panic?
            ::std::process::Command::new("python3")
                .arg("-m").arg("rsp2.cli.convert_dynmat")
                .arg(json_path)
                .arg("--output").arg(path.as_ref())
                .status()?
                .success()
        });
        Ok(())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DynmatTolerances {
    pub abs_tol: f64,
    pub rel_tol: f64,
}

impl Dynmat {
    fn to_map(&self) -> BTreeMap<(usize, usize), (M33, M33)> {
        let mut out = BTreeMap::new();
        let ends_1 = self.row_ptr.iter().cloned();
        let ends_2 = self.row_ptr.iter().cloned().dropping(1);
        for (row, (from, to)) in ends_1.zip(ends_2).enumerate() {
            let blocks_in_row = self.complex_blocks[from..to].iter().cloned();
            let cols_in_row = self.col[from..to].iter().cloned();
            for (col, block) in cols_in_row.zip(blocks_in_row) {
                out.insert((row, col), block);
            }
        }
        out
    }

    fn zip_nonzero_blocks(&self, other: &Dynmat) -> impl Iterator<Item=((M33, M33), (M33, M33))> {
        let mut our_map = self.to_map();
        let our_set: BTreeSet<_> = our_map.keys().cloned().collect();
        let mut their_map = other.to_map();
        let their_set: BTreeSet<_> = our_map.keys().cloned().collect();

        for key in &our_set - &their_set {
            assert!(their_map.insert(key, (M33::zero(), M33::zero())).is_none(), "bug in unit test");
        }
        for key in &their_set - &our_set {
            assert!(our_map.insert(key, (M33::zero(), M33::zero())).is_none(), "bug in unit test");
        }
        assert!(our_map.keys().eq(their_map.keys()), "bug in unit test");

        our_map.values().cloned().collect_vec().into_iter()
            .zip(their_map.values().cloned().collect_vec())
    }
}

impl CheckFile for Dynmat {
    type OtherArgs = DynmatTolerances;

    fn read_file(path: &Path) -> Result<Self, Error> { Self::load(path) }

    fn check_against(&self, expected: &Dynmat, tol: DynmatTolerances) {
        assert_eq!(self.dim, expected.dim);
        for (actual, expected) in self.zip_nonzero_blocks(expected) {
            let (actual_real, actual_imag) = actual;
            let (expected_real, expected_imag) = expected;
            assert_close!{
                abs=tol.abs_tol,
                rel=tol.rel_tol,
                [actual_real.unvee(), actual_imag.unvee()],
                [expected_real.unvee(), expected_imag.unvee()],
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------

pub fn load_json<T>(path: impl AsRef<Path>) -> Result<T, Error>
where T: ::serde::de::DeserializeOwned,
{
    let file = FileRead::read(path)?;
    Ok(::serde_json::from_reader(file)?)
}

pub fn save_json<T>(path: impl AsRef<Path>, obj: &T) -> Result<(), Error>
where T: ::serde::Serialize,
{
    let mut file = FileWrite::create(path)?;
    ::serde_json::to_writer(&mut file, obj)?;
    writeln!(file)?;
    Ok(())
}
