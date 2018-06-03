use ::std::ops::Deref;
use ::std::path::Path;
use ::std::io::prelude::*;

use ::shared::util;

use self::rsp2_integration_test::CheckFile;
extern crate rsp2_integration_test;

pub use self::failure::Error;
extern crate failure;

use self::path_abs::{FileRead, FileWrite};
extern crate path_abs;

use self::rsp2_structure::{Coords, CartOp};
extern crate rsp2_structure;

extern crate serde;

extern crate serde_json;

use self::rsp2_array_types::{V3};
extern crate rsp2_array_types;

macro_rules! impl_json {
    (($Type:ty) [$($methods:ident),*]) => {
        $( impl_json!{@one ($Type) [$methods]} )*
    };
    (@one ($Type:ty) [load]) => {
        #[allow(unused)]
        impl $Type {
            pub fn load(path: impl AsRef<::std::path::Path>) -> Result<Self, $crate::shared::filetypes::Error> {
                $crate::shared::filetypes::load_json(path)
            }
        }
    };
    (@one ($Type:ty) [save]) => {
        #[allow(unused)]
        impl $Type {
            pub fn save(&self, path: impl AsRef<::std::path::Path>) -> Result<(), $crate::shared::filetypes::Error> {
                $crate::shared::filetypes::save_json(path, self)
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

// ----------------------------------------------------------------------------------------------

pub fn load_json<T>(path: impl AsRef<Path>) -> Result<T, Error>
where T: serde::de::DeserializeOwned,
{
    let file = FileRead::read(path)?;
    Ok(serde_json::from_reader(file)?)
}

pub fn save_json<T>(path: impl AsRef<Path>, obj: &T) -> Result<(), Error>
where T: ::serde::Serialize,
{
    let mut file = FileWrite::create(path)?;
    serde_json::to_writer(&mut file, obj)?;
    writeln!(file)?;
    Ok(())
}
