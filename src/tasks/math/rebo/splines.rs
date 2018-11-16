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

//! Implementations of splines used by REBO.

// FIXME: The current implementation of REBO never has fractional weights
//        and thus the bicubic/tricubic spline implementations are unnecessary.
//
//        They should be removed if these circumstances persist.

use ::FailResult;
#[allow(unused)] // https://github.com/rust-lang/rust/issues/45268
use ::slice_of_array::prelude::*;
use ::std::ops::RangeInclusive;
#[allow(unused)] // https://github.com/rust-lang/rust/issues/45268
use ::std::borrow::Borrow;
use ::rsp2_array_types::{V2, V3};

// Until we get const generics, it's too much trouble to be generic over lengths,
// so we'll just use one fixed dimension.
pub const MAX_I: usize = 4;
pub const MAX_J: usize = 4;
pub const MAX_K: usize = 9;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum EvalKind { Fast, Slow }

pub mod P {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct SplineSet {
        pub CC: BicubicGrid,
        pub CH: BicubicGrid, // P_CH only. (P_HC is zero)
    }

    #[allow(unused)]
    lazy_static!{
        /// The only fully correct choice for PCC in second-generation REBO.
        ///
        /// Brenner (2002), Table 8.
        pub static ref BRENNER: SplineSet = SplineSet {
            CC: brenner_CC_input().solve().unwrap(),
            CH: brenner_CH(),
        };

        /// Suitable for AIREBO. (with the torsion term enabled)
        ///
        /// * Stuart (2000) Table VIII
        /// * LAMMPS, `pair_style airebo`.
        /// * LAMMPS, `pair_style rebo` prior ot 05Oct2016.
        ///
        /// Modifies a few of the terms to counteract AIREBO's torsional
        /// forces in unsaturated systems. (e.g. graphene)
        ///
        /// The rounding of values is chosen to match Brenner where
        /// available, and Stuart otherwise.
        pub static ref STUART: SplineSet = SplineSet {
            CC: stuart_CC(),
            CH: brenner_CH(),
        };

        /// Used by LAMMPS 05Oct2016–current (09Nov2018) in `pair_style rebo`.
        ///
        /// In 2016, Favata et. al reported that `pair_style rebo` erroneously
        /// used a parameter from AIREBO. LAMMPS was updated accordingly.
        ///
        /// However, there are actually three parameters that change, and
        /// this update only corrected one of them. Hence, this spline is not
        /// fully correct for neither REBO nor AIREBO.
        pub static ref FAVATA: SplineSet = SplineSet {
            CC: favata_CC(),
            CH: brenner_CH(),
        };
    }

    // * Brenner Table 8
    fn brenner_CC_input() -> bicubic::Input {
        let mut input = bicubic::Input::default();

        // NOTE: In the paper, Table 8 has the columns for i and j flipped.
        input.value[1][1] = 0.003_026_697_473_481; // (CH3)HC=CH(CH3)
        input.value[0][2] = 0.007_860_700_254_745; // C2H4
        input.value[0][3] = 0.016_125_364_564_267; // C2H6
        input.value[2][1] = 0.003_179_530_830_731; // i-C4H10
        input.value[1][2] = 0.006_326_248_241_119; // c-c6H12
        input
    }

    // * Brenner (Table 8)
    // * Stuart (Table VIII)  (at much lower precision)
    // * LAMMPS REBO/AIREBO  (rounded only sightly differently)
    fn brenner_CH() -> BicubicGrid {
        let mut input =  bicubic::Input::default();
        input.value[0][1] = 0.209_336_732_825_0380;  // CH2
        input.value[0][2] = -0.064_449_615_432_525;  // CH3
        input.value[0][3] = -0.303_927_546_346_162;  // CH4
        input.value[1][0] = 0.01;                    // C2H2
        input.value[2][0] = -0.122_042_146_278_2555; // (CH3)HC=CH(CH3)
        input.value[1][1] = -0.125_123_400_628_7090; // C2H4
        input.value[1][2] = -0.298_905_245_783;      // C2H6
        input.value[3][0] = -0.307_584_705_066;      // i-C4H10
        input.value[2][1] = -0.300_529_172_406_7579; // c-C6H12
        input.solve().unwrap()
    }

    fn stuart_CC() -> BicubicGrid {
        let mut input = brenner_CC_input();

        // Terms modified to counteract AIREBO's torsion.
        input.value[1][1] = -0.010_960;
        input.value[0][2] = -0.000_500;
        input.value[2][0] = -0.027_603;
        input.solve().unwrap()
    }

    fn favata_CC() -> BicubicGrid {
        let mut input = brenner_CC_input();

        // Beginning from the AIREBO params, Favata fixes one of the terms to match REBO
        // while leaving the other two. (because we're starting from REBO, we change the other two)
        input.value[1][1] = -0.010_960;
        input.value[0][2] = -0.000_500;
        // [2][0] is the one that was fixed.

        input.solve().unwrap()
    }
}

pub mod T {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct SplineSet {
        pub CC: TricubicGrid,
    }

    lazy_static!{
        /// The TCC spline found in:
        ///
        /// * The 2nd-gen REBO paper (Brenner, 2002)
        ///
        /// This differs from Stuart's in a manner which *seems* that it could
        /// possibly be a typo. (I have not yet looked further into this.)
        ///
        /// (namely, a value defined by Brenner at only `Tij(2,2,9)` is defined
        ///  by Stuart on `Tij(2,2,2..=9)` without any ceremony).
        pub static ref BRENNER: SplineSet = SplineSet {
            CC: brenner_CC(),
        };

        /// The TCC spline found in:
        ///
        /// * The AIREBO paper (Stuart, 2000)
        /// * The LAMMPS implementation of REBO and AIREBO
        ///
        /// It seems plausible that this is a "bugfix" of Brenner's table.
        pub static ref STUART: SplineSet = SplineSet {
            CC: stuart_CC(),
        };
    }

    /// Brenner, Table 5\
    fn brenner_CC() -> TricubicGrid {
        let mut input = tricubic::Input::default();
        input.value.assign((2, 2, 1), -0.070_280_085); // Ethane
        input.value.assign((2, 2, 9), -0.008_096_75);  // "Solid state carbon." (Graphene/graphite)

        let input = input.scale(0.5); // The values in Brenner's table are doubled
        input.solve().unwrap()
    }

    /// Stuart, Table X
    fn stuart_CC() -> TricubicGrid {
        let mut input = tricubic::Input::default();
        input.value.assign((2, 2, 1), -0.035_140);
        input.value.assign((2, 2, 2..=9), -0.004_048); // NOTE: different from Brenner

        input.solve().unwrap()
    }
}

pub mod F {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct SplineSet {
        pub CC: TricubicGrid,
        pub CH: TricubicGrid, // F_CH and F_HC.
        pub HH: TricubicGrid,
    }

    lazy_static! {
        /// Brenner (2002), Tables 4, 6, and 9.
        ///
        /// This has NOT been thoroughly checked against Stuart and LAMMPS, although
        /// all of what I have checked so far against Stuart matches.  (LAMMPS takes
        /// some creative freedom with the values of the splines near some of the
        /// boundaries, and I haven't yet figured out precisely what it does)
        ///
        /// **Caution:** This very likely contains errors, and using it is currently
        /// inadvisable.
        pub static ref BRENNER: SplineSet = SplineSet {
            CC: brenner_CC(),
            HH: brenner_HH(),
            CH: brenner_CH(),
        };
    }

    // Brenner, Table 4
    fn brenner_CC() -> TricubicGrid {
        let mut input = tricubic::Input::default();

        // NOTE: LAMMPS flattens out some values at high coordinates rather than having them
        //       go to zero. This might be a good idea for an alternate parameterization.

        input.value.assign((1, 1, 1), 0.105_000); // Acetylene
        input.value.assign((1, 1, 2), -0.004_177_5); // H2C=C=CH
        input.value.assign((1, 1, 3..=9), -0.016_085_6); // C4
        input.value.assign((2, 2, 1), 0.094_449_57); // (CH3)2C=C(CH3)2
        input.value.assign((2, 2, 2), 0.022_000_00); // Benzene

        // !!!!!!!!!!!!
        // FIXME: Are these correct?
        //
        // These are the exact values written in the paper, but it describes them as
        // "Average from difference F(2, 2, 2) to difference F(2, 2, 9)".
        //
        // They do have a constant difference, but if we were really starting from
        // the value of F[2][2][2], then that difference should be around 0.00314, not 0.00662.
        // (notice how F[2][2][3] > F[2][2][2])
        //
        // NOTE: The corresponding derivative is written as dF[2, 2, 4..=8]/dk, which is
        //       consistent with the values. Perhaps simply the comment in the paper is wrong.
        //
        // NOTE: Stuart and LAMMPS both also use these values.
        // !!!!!!!!!!!!
        input.value.assign((2, 2, 3), 0.039_705_87);
        input.value.assign((2, 2, 4), 0.033_088_22);
        input.value.assign((2, 2, 5), 0.026_470_58);
        input.value.assign((2, 2, 6), 0.019_852_93);
        input.value.assign((2, 2, 7), 0.013_235_29);
        input.value.assign((2, 2, 8), 0.006_617_64);
        input.value.assign((2, 2, 9), 0.0);

        input.value.assign((0, 1, 1), 0.043_386_99); // C2H

        input.value.assign((0, 1, 2), 0.009_917_2158); // C3
        input.value.assign((0, 2, 1), 0.049_397_6637); // CCH2
        input.value.assign((0, 2, 2), -0.011_942_669); // CCH(CH2)
        input.value.assign((0, 3, 1..=9), -0.119_798_935); // H3CC

        input.value.assign((1, 2, 1), 0.009_649_5698); // H2CCH
        input.value.assign((1, 2, 2), 0.030); // H2C=C=CH2
        input.value.assign((1, 2, 3), -0.0200); // C6H5

        // "Average from F(1,2,3) to F(1,2,6)".
        // At least this time, the description checks out.
        input.value.assign((1, 2, 4), -0.023_377_8774);
        input.value.assign((1, 2, 5), -0.026_755_7548);

        input.value.assign((1, 2, 6..=9), -0.030_133_632); // Graphite vacancy
        input.value.assign((1, 3, 2..=9), -0.124_836_752); // H3C–CCH
        input.value.assign((2, 3, 1..=9), -0.044_709_383); // Diamond vacancy

        // --------------------------
        // Derivatives

        input.di.assign((2, 1, 1), -0.052_500);
        input.di.assign((2, 1, 5..=9), -0.054_376);
        input.di.assign((2, 3, 1), 0.000_00);

        // NOTE: another oddity. These two ranges are written separately
        //       in the paper even though they could be a single range 2..=9.
        //       Does one contain an error?
        input.di.assign((2, 3, 2..=6), 0.062_418);
        input.di.assign((2, 3, 7..=9), 0.062_418);

        // !!!!!!!!!!!!!!!!!!
        // NOTE
        //
        // This derivative is related to the seemingly problematic values
        // in F[2][2][3..=8]
        // !!!!!!!!!!!!!!!!!!
        input.dk.assign((2, 2, 4..=8), -0.006_618);

        input.dk.assign((1, 1, 2), -0.060_543);

        // !!!!!!!!!!!!!!!!!!
        // FIXME
        //
        // This is highly suspicious; it seems this is intended to be the slope
        // from the linear equation describing F[1][2][3..=6], but it is at least
        // an order of magnitude off!
        //
        // This apparent error is unfortunately replicated in Stuart (2000) as
        // well as LAMMPS.
        // !!!!!!!!!!!!!!!!!!
        input.dk.assign((1, 2, 4), -0.020_044);
        input.dk.assign((1, 2, 5), -0.020_044);

        // symmetrize
        let input = input.symmetrize();

        // The values in Brenner (2002) are actually 2 * F.
        let input = input.scale(0.5);
        input.solve().unwrap()
    }

    // Brenner, Table 6
    // TODO: Check against Stuart and LAMMPS
    fn brenner_HH() -> TricubicGrid {
        let mut input = tricubic::Input::default();
        input.value.assign((1, 1, 1), 0.249_831_916);

        // The values in Brenner (2002) are actually 2 * F.
        let input = input.scale(0.5);
        input.solve().unwrap()
    }

    // Brenner, Table 9
    // TODO: Check against Stuart and LAMMPS
    fn brenner_CH() -> TricubicGrid {
        let mut input = tricubic::Input::default();

        input.value.assign((0, 2, 5..=9), -0.009_047_787_516_128_8110); // C6H6
        input.value.assign((1, 2, 1..=9), -0.25);  // Equations (23)–(25)
        input.value.assign((1, 3, 1..=9), -0.213); // Equations (23)–(25)
        input.value.assign((1, 1, 1..=9), -0.5);   // Equations (23)–(25)

        // Brenner's paper says that "F is symmetric", and Table 9 explicitly says that
        // "F(i, j, k) = F(j, i, k)", but it still seems open to interpretation whether
        // he merely meant that `F_ij(i,j,k) = F_ji(j,i,k)`, or if he further meant
        // that `F_ij(i,j,k) = F_ij(i,j,k)`. (in turn implying that `F_CH == F_HC`)
        //
        // We will assume he meant the latter, stronger condition, based on the following:
        //
        // * Stuart is more explicit and says `F_ij(i,j,k) = F_ij(j,i,k)`
        // * The LAMMPS implementation does it this way.
        // * All rows in Brenner's table suggestively have `i <= j`.
        // * The table appears to imply that `F_CH(0, 2, 5-9)` was fitted to C6H6, when
        //   the shape of the molecule implies that we should be fitting `F_CH(2, 0, 5-9)`.
        //   (I would call this "killer evidence" were it not for the fact that he also
        //    flipped the columns in the table right above this one ;P)
        let input = input.symmetrize();

        // The values in Brenner (2002) are actually 2 * F.
        let input = input.scale(0.5);
        input.solve().unwrap()
    }
}

//-----------------------------------------------

pub mod G {
    use super::*;

    // Spline coeffs were precomputed with:
    //
    // (FIXME: would be better to do this in Rust so they can be configured)
    /*
import numpy as np
from math import radians

# Construct a bunch of terms representing the boundary conditions.
# (the nth derivative of G at x0 equals some y0)

# produces a row in the matrix to be multiplied against
# the column vector [c0, c1, ..., c5] of polynomial coeffs
def matrix_row(term):
    x, order, _value = term
    coeffs = np.polyder([1]*6, order).tolist() + [0] * order
    powers = np.arange(6).tolist()[:6-order][::-1] + [0] * order
    return np.array(x) ** powers * coeffs

def solve_spline(terms):
    matrix = np.array(list(map(matrix_row, terms)))
    b = [[value] for (_, _, value) in terms]

    coeffs, = np.linalg.solve(matrix, b).T
    for (x, order, value) in terms:
        assert abs(np.polyval(np.polyder(coeffs, order), x) - value) < 1e-13
    return coeffs

# Data from Donald W Brenner et al 2002 J. Phys.: Condens. Matter 14 783
# Table 3 (C) and Table 6 (H)

# Terms for G(x) = y, G'(x) = yp, G''(x) = ypp
def terms_at(x, ys):
    y, yp, ypp = ys
    return [(x, 0, y), (x, 1, yp), (x, 2, ypp)]

cterms_1 = terms_at(  -1, (-0.00100, 0.10400, 0.00000)) # x = cos(pi)
cterms_2 = terms_at(-1/2, ( 0.05280, 0.17000, 0.37000)) # x = cos(2/3 pi)
cterms_3 = terms_at(-1/3, ( 0.09733, 0.40000, 1.98000)) # x = cos(0.6081 pi)
cterms_4_G = [
    (0.0, 0, 0.37545), # x = cos(pi/2)
    (0.5, 0, 2.0014),  # x = cos(pi/3)
    (1.0, 0, 8.0),     # x = cos(0)
]
cterms_4_gamma = [
    (0.0, 0, 0.271856), # x = cos(pi/2)
    (0.5, 0, 0.416335), # x = cos(pi/3)
    (1.0, 0, 1.0),      # x = cos(0)
]
hterms = [
    (np.cos(radians(  0)), 0, 19.991787),
    (np.cos(radians( 60)), 0, 19.704059),
    (np.cos(radians( 90)), 0, 19.065124),
    (np.cos(radians(120)), 0, 16.811574),
    (np.cos(radians(150)), 0, 12.164186),
    (np.cos(radians(180)), 0, 11.235870),
]
pieces = [
    ("C_COEFFS_1", "Segment 1: -1 to -1/2  (pi to 2pi/3)", cterms_1 + cterms_2),
    ("C_COEFFS_2", "Segment 2: -1/2 to -1/3  (2pi/3 to 109.47°)", cterms_2 + cterms_3),
    ("C_COEFFS_3_HIGH_COORDINATION", "Segment 3 (G): -1/3 to +1  (109.47° to 0°)", cterms_3 + cterms_4_G),
    ("C_COEFFS_3_LOW_COORDINATION", "Segment 3 (gamma): -1/3 to +1  (109.47° to 0°)", cterms_3 + cterms_4_gamma),
    ("H_COEFFS", "Full curve for hydrogen", hterms),
]

print("/*")
print(open(__file__).read(), end='')
print("*/")
print("// Coeffs listed from x**5 to x**0")
for (i, xval) in enumerate(["-1.0", "-0.5", "-1.0/3.0", "1.0"]):
    print(f"const C_X_{i}: f64 = {xval};")

for (name, heading, terms) in pieces:
    print()
    print(f"// {heading}")
    print(f"const {name}: &'static [f64] = &[")
    for x in solve_spline(terms):
        print(f"{x},")
    print(f"];")
*/

    /// A piecewise polynomial, optimized for the use case of only having a few segments.
    ///
    /// Between each two elements of x_div, it uses a polynomial from `coeffs`.
    #[derive(Debug, Clone)]
    pub struct SmallSpline1d<Array: Borrow<[f64]> + 'static> {
        pub x_div: &'static [f64],
        /// Polynomials between each two points in `x_div`, with coefficients in
        /// descending order.
        pub poly: &'static [Polynomial1d<Array>],
    }

    #[derive(Debug, Clone)]
    pub struct SplineSet {
        pub low_coord: f64,
        pub high_coord: f64,
        pub carbon_high_coord: SmallSpline1d<[f64; 6]>,
        pub carbon_low_coord: SmallSpline1d<[f64; 6]>,
        pub hydrogen: SmallSpline1d<[f64; 6]>,
    }

    impl SplineSet {
        #[cfg(test)]
        pub fn all_splines(&self) -> Vec<SmallSpline1d<[f64; 6]>> {
            vec![
                self.carbon_high_coord.clone(),
                self.carbon_low_coord.clone(),
                self.hydrogen.clone(),
            ]
        }
    }

    /// Splines produced by fitting the data in Brenner Table 3.
    pub const BRENNER: SplineSet = SplineSet {
        low_coord: 3.2,
        high_coord: 3.7,
        carbon_high_coord: SmallSpline1d {
            x_div: &[-1.0, -0.5, -1.0/3.0, 1.0],
            poly: &[Polynomial1d([
                // Segment 1: -1 to -1/2  (pi to 2pi/3)
                -1.342399999999925, -4.927999999999722, -6.829999999999602,
                -4.3459999999997265, -1.0979999999999095, 0.002600000000011547,
            ]), Polynomial1d([
                // Segment 2: -1/2 to -1/3  (2pi/3 to 109.47°)
                35.3116800000094, 69.87600000001967, 55.94760000001625,
                23.43200000000662, 5.544400000001327, 0.6966900000001047,
            ]), Polynomial1d([
                // Segment 3 (G): -1/3 to +1  (109.47° to 0°)
                0.5064259725000047, 1.4271989062499966, 2.028821591249997,
                2.254920828750001, 1.4071827012500007, 0.37545,
            ])],
        },
        carbon_low_coord: SmallSpline1d {
            x_div: &[-1.0, -0.5, -1.0/3.0, 1.0],
            poly: &[Polynomial1d([
                // Segment 1: -1 to -1/2  (pi to 2pi/3)
                -1.342399999999925, -4.927999999999722, -6.829999999999602,
                -4.3459999999997265, -1.0979999999999095, 0.002600000000011547,
            ]), Polynomial1d([
                // Segment 2: -1/2 to -1/3  (2pi/3 to 109.47°)
                35.3116800000094, 69.87600000001967, 55.94760000001625,
                23.43200000000662, 5.544400000001327, 0.6966900000001047,
            ]), Polynomial1d([
                // Segment 3 (G): -1/3 to +1  (109.47° to 0°)
                -0.03793074749999925, 1.2711119062499994, -0.5613989287500004,
                -0.4328552912499998, 0.4892170612500001, 0.271856,
            ])],
        },
        hydrogen: SmallSpline1d {
            x_div: &[-1.0, 1.0],
            poly: &[Polynomial1d([
                -9.287290931116942, -0.29608733333332005, 13.589744997229507,
                -3.1552081666666805, 0.0755044338874331, 19.065124,
            ])],
        },
    };

    /// From CH.airebo.
    ///
    /// These appear to have been produced by fitting the data in the AIREBO paper. (Stuart 2000)
    ///
    /// My current understanding is that it is okay to use these for REBO, and that they are
    /// simply an improvement upon the curves provided in Brenner (2002) that goes hand-in-hand
    /// with the modifications to `lambda_ijk`.
    ///
    /// ...however, the coefficients here are rounded to dangerously low precision, which
    /// might introduce discontinuities at the switch points (most troublingly so at 120°)
    /// that could ruin optimization algorithms.
    ///
    /// TODO: Build our own splines without such insane rounding errors
    pub const LAMMPS: SplineSet = SplineSet {
        low_coord: 3.2,
        high_coord: 3.7,
        carbon_high_coord: SmallSpline1d {
            x_div: &[-1.0, -0.6666666667, -0.5, -0.3333333333, 1.0],
            poly: &[Polynomial1d([
                0.3862485000, 1.5544035000, 2.5334145000,
                2.1363075000, 1.0627430000, 0.2816950000,
            ]), Polynomial1d([
                0.4025160000, 1.6019100000, 2.5885710000,
                2.1681365000, 1.0718770000, 0.2827390000,
            ]), Polynomial1d([
                34.7051520000, 68.6124000000, 54.9086400000,
                23.0108000000, 5.4601600000, 0.6900250000,
            ]), Polynomial1d([
                0.5063519355, 1.4269207324, 2.0288747461,
                2.2551320117, 1.4072691309, 0.3754514434,
            ])],
        },
        carbon_low_coord: SmallSpline1d {
            x_div: &[-1.0, -0.6666666667, -0.5, -0.3333333333, 1.0],
            poly: &[Polynomial1d([
                0.3862485000, 1.5544035000, 2.5334145000,
                2.1363075000, 1.0627430000, 0.2816950000,
            ]), Polynomial1d([
                0.4025160000, 1.6019100000, 2.5885710000,
                2.1681365000, 1.0718770000, 0.2827390000,
            ]), Polynomial1d([
                34.7051520000, 68.6124000000, 54.9086400000,
                23.0108000000, 5.4601600000, 0.6900250000,
            ]), Polynomial1d([
                -0.0375008379, 1.2708702246, -0.5616817383,
                -0.4328177539, 0.4892740137, 0.2718560918,
            ])],
        },
        hydrogen: SmallSpline1d {
            x_div: &[-1.0, -0.8333333333, -0.5, 1.0],
            poly: &[Polynomial1d([
                630.6336000042, 2721.4308000191, 4582.1544000348,
                3781.7719000316, 1549.6358000143, 270.4568000026,
            ]), Polynomial1d([
                -94.9946400000, -229.8471299999, -210.6432299999,
                -102.4683000000, -21.0823875000, 16.9534406250,
            ]), Polynomial1d([
                0.8376699753, -2.6535615062, 3.2913322346,
                -2.5664219198, 2.0177562840, 19.0650249321,
            ])],
        },
    };

    impl<Array: Borrow<[f64]> + 'static> SmallSpline1d<Array> {
        pub fn evaluate(&self, x: f64) -> (f64, f64) {
            // NOTE: This linear search will *almost always* stop at one of the first two
            //       elements.  Large cosine means small angles, which are rare.
            for (i, &div) in self.x_div.iter().skip(1).enumerate() {
                if x <= div {
                    return self.poly[i].evaluate(x);
                }
            }

            // tolerate fuzz
            let high = *self.x_div.last().unwrap();
            let width = high - self.x_div[0];
            assert!(x < high + width * 1e-8);

            self.poly.last().unwrap().evaluate(x)
        }
    }

    /// A polynomial with coefficients listed in decreasing order
    #[derive(Debug, Clone)]
    pub struct Polynomial1d<Array>(pub Array);

    impl<Array: Borrow<[f64]>> Polynomial1d<Array> {
        pub fn evaluate(&self, x: f64) -> (f64, f64) {
            let poly_coeffs = self.0.borrow().iter().cloned();
            let deriv_coeffs = polyder_dec(self.0.borrow().iter().cloned());
            (_polyval_dec(poly_coeffs, x), _polyval_dec(deriv_coeffs, x))
        }

        #[cfg(test)]
        pub fn derivative(&self) -> Polynomial1d<Vec<f64>> {
            Polynomial1d(polyder_dec(self.0.borrow().iter().cloned()).collect())
        }
    }

    fn polyder_dec(
        coeffs: impl DoubleEndedIterator<Item=f64> + ExactSizeIterator + Clone,
    ) -> impl DoubleEndedIterator<Item=f64> + ExactSizeIterator + Clone
    { coeffs.rev().skip(1).enumerate().map(|(n, x)| (n + 1) as f64 * x).rev() }

    #[inline(always)]
    fn _polyval_dec(coeffs: impl Iterator<Item=f64>, x: f64) -> f64 {
        coeffs.fold(0.0, |acc, c| acc * x + c)
    }
}

//-----------------------------------------------

// uniform interface for assigning a single element or a range
pub trait ArrayAssignExt<I> {
    fn assign(&mut self, i: I, fill: f64);
}

impl ArrayAssignExt<(usize, usize, usize)> for tricubic::EndpointGrid<f64> {
    fn assign(&mut self, (i, j, k): (usize, usize, usize), fill: f64) {
        self[i][j][k] = fill;
    }
}

impl ArrayAssignExt<(usize, usize, RangeInclusive<usize>)> for tricubic::EndpointGrid<f64> {
    fn assign(&mut self, (i, j, k): (usize, usize, RangeInclusive<usize>), fill: f64) {
        for x in &mut self[i][j][k] {
            *x = fill;
        }
    }
}

//-----------------------------------------------

pub use self::tricubic::TricubicGrid;
pub mod tricubic {
    use super::*;

    /// A grid of "fencepost" values.
    pub type EndpointGrid<T> = nd![T; MAX_I+1; MAX_J+1; MAX_K+1];
    /// A grid of "fence segment" values.
    pub type Grid<T> = nd![T; MAX_I; MAX_J; MAX_K];
    pub type Input = _Input<EndpointGrid<f64>>;

    /// The values and derivatives that are fitted to produce a tricubic spline.
    ///
    /// NOTE: not all constraints are explicitly listed;
    /// We also place implicit constraints that `d^2/didj`, `d^2/didk`,
    /// `d^2/djdk`, and `d^3/didjdk` are zero at all integer points.
    ///
    /// (why these particular derivatives?  It turns out that these are the
    ///  ones that produce linearly independent equations. See Lekien.)
    ///
    /// # References
    ///
    /// F. Lekien and J. Marsden, Tricubic interpolation in three dimensions,
    /// Int. J. Numer. Meth. Engng 2005; 63:455–471
    #[derive(Debug, Clone, Default)]
    pub struct _Input<G> {
        pub value: G,
        pub di: G,
        pub dj: G,
        pub dk: G,
    }

    //------------------------------------

    #[derive(Debug, Clone)]
    pub struct TricubicGrid {
        pub(super) fit_params: Box<Input>,
        pub(super) polys: Box<Grid<(TriPoly3, V3<TriPoly3>)>>,
    }

    impl TricubicGrid {
        pub fn evaluate(&self, point: V3) -> (f64, V3) { self._evaluate(point).1 }

        pub(super) fn _evaluate(&self, point: V3) -> (EvalKind, (f64, V3)) {
            // We assume the splines are flat with constant value outside the fitted regions.
            let point = clip_point(point);

            let indices = point.map(|x| x as usize);

            if point == indices.map(|x| x as f64) {
                // Fast path (integer point)

                let V3([i, j, k]) = indices;
                let value = self.fit_params.value[i][j][k];
                let di = self.fit_params.di[i][j][k];
                let dj = self.fit_params.dj[i][j][k];
                let dk = self.fit_params.dk[i][j][k];
                (EvalKind::Fast, (value, V3([di, dj, dk])))
            } else {
                // Slow path.
                //
                // It is only ever possible to take this path when a reaction is occurring.
                warn!("untested codepath: 70dfe923-e1af-45f1-8dc6-eb50ae4ce1cc");

                // Indices must now be constrained to the smaller range that is valid
                // for the polynomials. (i.e. the max index is no longer valid)
                //
                // (Yes, we must account for this even though we clipped the point; if the
                //  point is only out of bounds along one axis, the others may still be
                //  fractional and thus the slow path could still be taken)
                let V3([mut i, mut j, mut k]) = indices;
                i = i.min(MAX_I - 1);
                j = j.min(MAX_J - 1);
                k = k.min(MAX_K - 1);

                let frac_point = point - V3([i, j, k]).map(|x| x as f64);
                let (value_poly, diff_polys) = &self.polys[i][j][k];
                let value = value_poly.evaluate(point);
                let diff = V3::from_fn(|axis| diff_polys[axis].evaluate(frac_point));
                (EvalKind::Slow, (value, diff))
            }
        }
    }

    impl<A> _Input<A> {
        fn map_grids<B>(&self, mut func: impl FnMut(&A) -> B) -> _Input<B> {
            _Input {
                value: func(&self.value),
                di: func(&self.di),
                dj: func(&self.dj),
                dk: func(&self.dk),
            }
        }
    }

    impl Input {
        pub fn solve(&self) -> FailResult<TricubicGrid> {
            use ::rsp2_array_utils::{try_arr_from_fn, arr_from_fn};
            self.verify_clipping_is_valid()?;

            let polys = Box::new({
                try_arr_from_fn(|i| {
                    try_arr_from_fn(|j| {
                        try_arr_from_fn(|k| -> FailResult<_> {
                            // Gather the 8 points describing this region.
                            // (ni,nj,nk = 0 or 1)
                            let poly_input: TriPoly3Input = self.map_grids(|grid| {
                                arr_from_fn(|ni| {
                                    arr_from_fn(|nj| {
                                        arr_from_fn(|nk| {
                                            grid[i + ni][j + nj][k + nk]
                                        })
                                    })
                                })
                            });
                            let value_poly = poly_input.solve()?;
                            let diff_polys = V3::from_fn(|axis| value_poly.axis_derivative(axis));
                            Ok((value_poly, diff_polys))
                        })
                    })
                })?
            });

            let fit_params = Box::new(self.clone());
            Ok(TricubicGrid { fit_params, polys })
        }

        pub fn scale(mut self, factor: f64) -> Self {
            { // FIXME: block will be unnecessary once NLL lands
                let Input { value, di, dj, dk } = &mut self;
                for &mut &mut ref mut array in &mut[value, di, dj, dk] {
                    for plane in array {
                        for row in plane {
                            for x in row {
                                *x *= factor;
                            }
                        }
                    }
                }
            }
            self
        }

        /// Symmetrize a function so that `F(i, j, k) = F(j, i, k)`.
        ///
        /// The input function must be triangular. (`dj` must be uniformly zero,
        /// and it must have `value[i][j] = dk[i][j] = 0` for `i > j`)
        pub fn symmetrize(mut self) -> Self {
            let n = self.value.len();
            for upper in 0..n {
                for lower in 0..upper {
                    for k in 0..self.value[0][0].len() {
                        assert_eq!(self.value[upper][lower][k], 0.0, "input was not triangular");
                        assert_eq!(self.dk[upper][lower][k], 0.0, "input was not triangular");
                        self.value[upper][lower][k] = self.value[lower][upper][k];
                        self.dk[upper][lower][k] = self.dk[lower][upper][k];
                    }
                }
            }
            for i in 0..n {
                for j in 0..n {
                    assert!(self.dj[i][j].iter().all(|&x| x == 0.0), "input was not triangular");
                    self.dj[i][j].copy_from_slice(&self.di[j][i]);
                }
            }
            self
        }

        #[cfg(test)]
        pub fn random(scale: f64) -> Self {
            Input {
                value: ::rand::random(),
                di: ::rand::random(),
                dj: ::rand::random(),
                dk: ::rand::random(),
            }.scale(scale).ensure_clipping_is_valid()
        }
    }

    impl Input {
        // To make clipping always valid, we envision that the spline is flat outside of
        // the fitted region.  For C1 continuity, this means the derivatives at these
        // boundaries must be zero.
        pub fn verify_clipping_is_valid(&self) -> FailResult<()> {
            let Input { value: _, di, dj, dk } = self;

            macro_rules! check {
                ($iter:expr) => {
                    ensure!(
                        $iter.into_iter().all(|&x| x == 0.0),
                        "derivatives must be zero at the endpoints of the spline"
                    )
                };
            }

            check!(di[0].flat());
            check!(di.last().unwrap().flat());
            check!(dj.iter().flat_map(|plane| &plane[0]));
            check!(dj.iter().flat_map(|plane| plane.last().unwrap()));
            check!(dk.iter().flat_map(|plane| plane.iter().map(|row| &row[0])));
            check!(dk.iter().flat_map(|plane| plane.iter().map(|row| row.last().unwrap())));
            Ok(())
        }

        // useful for tests
        #[cfg(test)]
        pub(super) fn ensure_clipping_is_valid(mut self) -> Self {
            { // FIXME block is unnecessary once NLL lands
                let Input { value: _, di, dj, dk } = &mut self;
                fn zero<'a>(xs: impl IntoIterator<Item=&'a mut f64>) {
                    for x in xs { *x = 0.0; }
                }

                zero(di[0].flat_mut());
                zero(di.last_mut().unwrap().flat_mut());
                zero(dj.iter_mut().flat_map(|plane| &mut plane[0]));
                zero(dj.iter_mut().flat_map(|plane| plane.last_mut().unwrap()));
                zero(dk.iter_mut().flat_map(|plane| plane.iter_mut().map(|row| &mut row[0])));
                zero(dk.iter_mut().flat_map(|plane| plane.iter_mut().map(|row| row.last_mut().unwrap())));
            }
            self
        }
    }

    pub fn clip_point(point: V3) -> V3 {
        let mut point = point.map(|x| f64::max(x, 0.0));
        point[0] = point[0].min(MAX_I as f64);
        point[1] = point[1].min(MAX_J as f64);
        point[2] = point[2].min(MAX_K as f64);
        point
    }

    //------------------------------------

    /// A third-order polynomial in three variables.
    #[derive(Debug, Clone)]
    pub struct TriPoly3 {
        /// coeffs along each index are listed in order of increasing power
        coeff: Box<nd![f64; 4; 4; 4]>,
    }

    pub type TriPoly3Input = _Input<nd![f64; 2; 2; 2]>;
    impl TriPoly3Input {
        fn solve(&self) -> FailResult<TriPoly3> {
            let b_vec: nd![f64; 8; 2; 2; 2] = [
                self.value,
                self.di, self.dj, self.dk,
                Default::default(), // constraints on didj
                Default::default(), // constraints on didk
                Default::default(), // constraints on djdk
                Default::default(), // constraints on didjdk
            ];
            let b_vec: &[[f64; 1]] = b_vec.flat().flat().flat().nest();
            let b_vec: ::rsp2_linalg::CMatrix = b_vec.into();

            let coeff = ::rsp2_linalg::lapacke_linear_solve(ZERO_ONE_CMATRIX.clone(), b_vec)?;
            Ok(TriPoly3 {
                coeff: Box::new(coeff.c_order_data().nest().nest().to_array()),
            })
        }
    }

    impl TriPoly3 {
        pub fn zero() -> Self {
            TriPoly3 { coeff: Box::new(<nd![f64; 4; 4; 4]>::default()) }
        }

        pub fn evaluate(&self, point: V3) -> f64 {
            let V3([i, j, k]) = point;

            let powers = |x| [1.0, x, x*x, x*x*x];
            let i_pows = powers(i);
            let j_pows = powers(j);
            let k_pows = powers(k);

            let mut acc = 0.0;
            for (coeff_plane, &i_pow) in zip_eq!(&self.coeff[..], &i_pows) {
                for (coeff_row, &j_pow) in zip_eq!(coeff_plane, &j_pows) {
                    let row_sum = zip_eq!(coeff_row, &k_pows).map(|(&a, &b)| a * b).sum::<f64>();
                    acc += i_pow * j_pow * row_sum;
                }
            }
            acc
        }

        #[inline(always)]
        fn coeff(&self, (i, j, k): (usize, usize, usize)) -> f64 { self.coeff[i][j][k] }
        #[inline(always)]
        fn coeff_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut f64 { &mut self.coeff[i][j][k] }

        pub fn axis_derivative(&self, axis: usize) -> Self {
            let mut out = Self::zero();
            for scan_idx_1 in 0..4 {
                for scan_idx_2 in 0..4 {
                    let get_pos = |i| match axis {
                        0 => (i, scan_idx_1, scan_idx_2),
                        1 => (scan_idx_1, i, scan_idx_2),
                        2 => (scan_idx_1, scan_idx_2, i),
                        _ => panic!("invalid axis: {}", axis),
                    };
                    for i in 1..4 {
                        *out.coeff_mut(get_pos(i-1)) = i as f64 * self.coeff(get_pos(i));
                    }
                }
            }
            out
        }
    }

    lazy_static! {
        // The matrix representing the system of equations that must be solved for
        // a piece of a tricubic spline with boundaries at zero and one.
        //
        // Indices are, from slowest to fastest:
        // - row (8x2x2x2 = 64), broken into two levels:
        //   - constraint kind (8: [value, di, dj, dk, didj, didk, djdk, didjdk])
        //   - constraint location (2x2x2: [i=0, i=1] x [j=0, j=1] x [k=0, k=1])
        // - col (4x4x4 = 64), where each axis is the power of one of the variables
        //   for the coefficient belonging to this column
        static ref ZERO_ONE_MATRIX: nd![f64; 8; 2; 2; 2; 4; 4; 4] = compute_zero_one_matrix();
        static ref ZERO_ONE_CMATRIX: ::rsp2_linalg::CMatrix = {
            ZERO_ONE_MATRIX
                .flat().flat().flat().flat()
                .flat().flat().nest::<[_; 64]>()
                .into()
        };
    }

    fn compute_zero_one_matrix() -> nd![f64; 8; 2; 2; 2; 4; 4; 4] {
        use ::rsp2_array_utils::{arr_from_fn, map_arr};

        // we build a system of equations from our constraints
        //
        // we end up with an equation of the form  M a = b,
        // where M is a square matrix whose elements are products of the end-point coords
        // raised to various powers.

        #[derive(Debug, Copy, Clone)]
        struct Monomial {
            coeff: f64,
            powers: [u32; 3],
        }
        impl Monomial {
            fn axis_derivative(mut self, axis: usize) -> Self {
                self.coeff *= self.powers[axis] as f64;
                if self.powers[axis] > 0 {
                    self.powers[axis] -= 1;
                }
                self
            }

            fn evaluate(&self, point: V3) -> f64 {
                let mut out = self.coeff;
                for i in 0..3 {
                    out *= point[i].powi(self.powers[i] as i32);
                }
                out
            }
        }

        // Polynomials here are represented as values to be multiplied against each coefficient.
        //
        // e.g. [1, x, x^2, x^3, y, y*x, y*x^2, y*x^3, ... ]
        let derive = |poly: &[Monomial], axis| -> Vec<Monomial> {
            poly.iter().map(|m| m.axis_derivative(axis)).collect()
        };

        let value_poly: nd![Monomial; 4; 4; 4] = {
            arr_from_fn(|i| {
                arr_from_fn(|j| {
                    arr_from_fn(|k| {
                        Monomial { coeff: 1.0, powers: [i as u32, j as u32, k as u32] }
                    })
                })
            })
        };
        let value_poly = value_poly.flat().flat().to_vec();
        let di_poly = derive(&value_poly, 0);
        let dj_poly = derive(&value_poly, 1);
        let dk_poly = derive(&value_poly, 2);
        let didj_poly = derive(&di_poly, 1);
        let didk_poly = derive(&di_poly, 2);
        let djdk_poly = derive(&dj_poly, 2);
        let didjdk_poly = derive(&didj_poly, 2);

        map_arr([
                    value_poly, di_poly, dj_poly, dk_poly,
                    didj_poly, didk_poly, djdk_poly, didjdk_poly,
                ], |poly| {
            // coords of each corner (0 or 1)
            arr_from_fn(|i| {
                arr_from_fn(|j| {
                    arr_from_fn(|k| {
                        // powers
                        let poly: &nd![_; 4; 4; 4] = poly.nest().nest().as_array();
                        arr_from_fn(|ei| {
                            arr_from_fn(|ej| {
                                arr_from_fn(|ek| {
                                    poly[ei][ej][ek].evaluate(V3([i, j, k]).map(|x| x as f64))
                                })
                            })
                        })
                    })
                })
            })
        })
    }

    //------------------------------------
    // tests

    #[test]
    fn test_spline_fast_path() -> FailResult<()> {
        let fit_params = Input::random(1.0);
        let spline = fit_params.solve()?;

        // every valid integer point should be evaluated quickly
        for i in 0..=MAX_I {
            for j in 0..=MAX_J {
                for k in 0..=MAX_K {
                    let (kind, output) = spline._evaluate(V3([i, j, k]).map(|x| x as f64));
                    let (value, V3([di, dj, dk])) = output;
                    assert_eq!(kind, EvalKind::Fast);
                    assert_eq!(value, fit_params.value[i][j][k]);
                    assert_eq!(di, fit_params.di[i][j][k]);
                    assert_eq!(dj, fit_params.dj[i][j][k]);
                    assert_eq!(dk, fit_params.dk[i][j][k]);
                }
            }
        }

        // points outside the boundaries should also be evaluated quickly if the
        // remaining coords are integers
        let base_point = V3([2.0, 2.0, 2.0]);
        let base_index = V3([2, 2, 2]);
        for axis in 0..3 {
            for do_right_side in vec![false, true] {
                let mut input_point = base_point;
                let mut expected_index = base_index;
                match do_right_side {
                    false => {
                        input_point[axis] = -1.2;
                        expected_index[axis] = 0;
                    },
                    true => {
                        input_point[axis] = [MAX_I, MAX_J, MAX_K][axis] as f64 + 3.2;
                        expected_index[axis] = [MAX_I, MAX_J, MAX_K][axis];
                    }
                }

                let (kind, output) = spline._evaluate(input_point);
                let (value, V3([di, dj, dk])) = output;

                let V3([i, j, k]) = expected_index;
                assert_eq!(kind, EvalKind::Fast);
                assert_eq!(value, fit_params.value[i][j][k]);
                assert_eq!(di, fit_params.di[i][j][k]);
                assert_eq!(dj, fit_params.dj[i][j][k]);
                assert_eq!(dk, fit_params.dk[i][j][k]);
            }
        }
        Ok(())
    }

    #[test]
    fn test_spline_fit_accuracy() -> FailResult<()> {
        for _ in 0..3 {
            let fit_params = Input::random(1.0);
            let spline = fit_params.solve()?;

            // index of a polynomial
            for i in 0..MAX_I {
                for j in 0..MAX_J {
                    for k in 0..MAX_K {
                        // index of a corner of the polynomial
                        for ni in 0..2 {
                            for nj in 0..2 {
                                for nk in 0..2 {
                                    // index of the point of evaluation
                                    let V3([pi, pj, pk]) = V3([i + ni, j + nj, k + nk]);
                                    let frac_point = V3([ni, nj, nk]).map(|x| x as f64);

                                    let (value_poly, diff_polys) = &spline.polys[i][j][k];
                                    let V3([di_poly, dj_poly, dk_poly]) = diff_polys;
                                    assert_close!(rel=1e-8, abs=1e-8, value_poly.evaluate(frac_point), fit_params.value[pi][pj][pk]);
                                    assert_close!(rel=1e-8, abs=1e-8, di_poly.evaluate(frac_point), fit_params.di[pi][pj][pk]);
                                    assert_close!(rel=1e-8, abs=1e-8, dj_poly.evaluate(frac_point), fit_params.dj[pi][pj][pk]);
                                    assert_close!(rel=1e-8, abs=1e-8, dk_poly.evaluate(frac_point), fit_params.dk[pi][pj][pk]);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_poly3_evaluate() {
        for _ in 0..1 {
            let point = V3::from_fn(|_| uniform(-1.0, 1.0));
            let poly = TriPoly3 {
                coeff: Box::new({
                    ::std::iter::repeat_with(|| uniform(-5.0, 5.0)).take(64).collect::<Vec<_>>()
                        .nest().nest().to_array()
                }),
            };

            let expected = {
                // brute force
                let mut acc = 0.0;
                for i in 0..4 {
                    for j in 0..4 {
                        for k in 0..4 {
                            acc += {
                                poly.coeff[i][j][k]
                                    * point[0].powi(i as i32)
                                    * point[1].powi(j as i32)
                                    * point[2].powi(k as i32)
                            };
                        }
                    }
                }
                acc
            };
            assert_close!(poly.evaluate(point), expected);
        }
    }

    #[test]
    fn test_poly3_numerical_deriv() -> () {
        for _ in 0..20 {
            let value_poly = TriPoly3 {
                coeff: Box::new(::rand::random()),
            };
            let grad_polys = V3::from_fn(|axis| value_poly.axis_derivative(axis));

            let point = V3::from_fn(|_| uniform(-6.0, 6.0));

            let computed_grad = grad_polys.map(|poly| poly.evaluate(point));
            let numerical_grad = num_grad_v3(1e-6, point, |p| value_poly.evaluate(p));

            // This can fail pretty bad if the polynomial produces lots of cancellation
            // in one of the derivatives.  We must accept either abs or rel tolerance.
            assert_close!(rel=1e-5, abs=1e-5, computed_grad.0, numerical_grad.0)
        }
    }
} // mod tricubic

//------------------------------------
// bicubic

pub use self::bicubic::BicubicGrid;
pub mod bicubic {
    use super::*;

    /// A grid of "fencepost" values.
    pub type EndpointGrid<T> = nd![T; MAX_I+1; MAX_J+1];

    /// Input for a bicubic spline.
    ///
    /// Not included is an implicit constraint that `d^2/didj = 0` at all integer points.
    #[derive(Default)]
    pub struct Input {
        pub value: EndpointGrid<f64>,
        pub di: EndpointGrid<f64>,
        pub dj: EndpointGrid<f64>,
    }

    #[derive(Debug, Clone)]
    pub struct BicubicGrid {
        // "Do the simplest thing that will work."
        tricubic: TricubicGrid,
    }

    impl BicubicGrid {
        pub fn evaluate(&self, point: V2) -> (f64, V2) { self._evaluate(point).1 }

        fn _evaluate(&self, point: V2) -> (EvalKind, (f64, V2)) {
            let V2([i, j]) = point;

            let (kind, (value, V3([di, dj, dk]))) = self.tricubic._evaluate(V3([i, j, 0.0]));
            assert_eq!(dk, 0.0);

            (kind, (value, V2([di, dj])))
        }

        #[cfg(test)]
        fn lookup_poly(&self, V2([i, j]): V2<usize>) -> (tricubic::TriPoly3, V2<tricubic::TriPoly3>){
            let (value, V3([di, dj, _])) = &self.tricubic.polys[i][j][0];
            (value.clone(), V2([di.clone(), dj.clone()]))
        }
    }

    impl Input {
        pub fn solve(&self) -> FailResult<BicubicGrid> {
            let tricubic = self.to_tricubic_input().solve()?;
            Ok(BicubicGrid { tricubic })
        }

        fn to_tricubic_input(&self) -> tricubic::Input {
            use ::rsp2_array_utils::{map_arr};
            let Input { value, di, dj } = *self;

            // make everything constant along the k axis
            let extend = |arr| map_arr(arr, |row| map_arr(row, |x| [x; MAX_K+1]));
            tricubic::Input {
                value: extend(value),
                di: extend(di),
                dj: extend(dj),
                dk: Default::default(),
            }
        }

        #[cfg(test)]
        fn from_tricubic_input(input: &tricubic::Input) -> Self {
            use ::rsp2_array_utils::{map_arr};

            let tricubic::Input { value, di, dj, dk } = *input;

            let unextend = |arr| map_arr(arr, |plane| map_arr(plane, |row: [_; MAX_K+1]| row[0]));

            assert_eq!(unextend(dk), unextend(<tricubic::EndpointGrid<f64>>::default()));
            Input {
                value: unextend(value),
                di: unextend(di),
                dj: unextend(dj),
            }
        }

        #[cfg(test)]
        pub fn random(scale: f64) -> Self {
            Self::from_tricubic_input(&tricubic::Input::random(scale))
        }
    }

    //------------------------------------
    // tests

    #[test]
    fn test_spline_fast_path() -> FailResult<()> {
        let fit_params = Input::random(1.0);
        let spline = fit_params.solve()?;

        // every valid integer point should be evaluated quickly
        for i in 0..=MAX_I {
            for j in 0..=MAX_J {
                let (kind, output) = spline._evaluate(V2([i, j]).map(|x| x as f64));
                let (value, V2([di, dj])) = output;
                assert_eq!(kind, EvalKind::Fast);
                assert_eq!(value, fit_params.value[i][j]);
                assert_eq!(di, fit_params.di[i][j]);
                assert_eq!(dj, fit_params.dj[i][j]);
            }
        }

        // points outside the boundaries should also be evaluated quickly if the
        // remaining coords are integers
        let base_point = V2([2.0, 2.0]);
        let base_index = V2([2, 2]);
        for axis in 0..2 {
            for do_right_side in vec![false, true] {
                let mut input_point = base_point;
                let mut expected_index = base_index;
                match do_right_side {
                    false => {
                        input_point[axis] = -1.2;
                        expected_index[axis] = 0;
                    },
                    true => {
                        input_point[axis] = [MAX_I, MAX_J][axis] as f64 + 3.2;
                        expected_index[axis] = [MAX_I, MAX_J][axis];
                    }
                }

                let (kind, output) = spline._evaluate(input_point);
                let (value, V2([di, dj])) = output;

                let V2([i, j]) = expected_index;
                assert_eq!(kind, EvalKind::Fast);
                assert_eq!(value, fit_params.value[i][j]);
                assert_eq!(di, fit_params.di[i][j]);
                assert_eq!(dj, fit_params.dj[i][j]);
            }
        }
        Ok(())
    }

    #[test]
    fn test_spline_fit_accuracy() -> FailResult<()> {
        for _ in 0..3 {
            let fit_params = Input::random(1.0);
            let spline = fit_params.solve()?;

            // index of a polynomial
            for i in 0..MAX_I {
                for j in 0..MAX_J {
                    // index of a corner of the polynomial
                    for ni in 0..2 {
                        for nj in 0..2 {
                            // index of the point of evaluation
                            let V2([pi, pj]) = V2([i + ni, j + nj]);
                            let frac_point = V3([ni, nj, 0]).map(|x| x as f64);

                            let (value_poly, diff_polys) = &spline.lookup_poly(V2([i, j]));
                            let V2([di_poly, dj_poly]) = diff_polys;
                            assert_close!(rel=1e-8, abs=1e-8, value_poly.evaluate(frac_point), fit_params.value[pi][pj]);
                            assert_close!(rel=1e-8, abs=1e-8, di_poly.evaluate(frac_point), fit_params.di[pi][pj]);
                            assert_close!(rel=1e-8, abs=1e-8, dj_poly.evaluate(frac_point), fit_params.dj[pi][pj]);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
fn uniform(a: f64, b: f64) -> f64 { ::rand::random::<f64>() * (b - a) + a }

#[cfg(test)]
fn num_grad_v3(
    interval: f64,
    point: V3,
    mut value_fn: impl FnMut(V3) -> f64,
) -> V3 {
    use ::rsp2_minimize::numerical;
    numerical::gradient(interval, None, &point.0, |v| value_fn(v.to_array())).to_array()
}
