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
use ::potential::{PotentialBuilder};

use ::meta::{self, prelude::*};
use ::rsp2_tasks_config as cfg;

use ::math::basis::{Basis3, EvDirections};

use ::rsp2_slice_math::{v, V, vdot, vnormalize, BadNorm};

use ::slice_of_array::prelude::*;
use ::rsp2_structure::{Coords};
use ::hlist_aliases::*;

use ::std::fmt;
use ::std::rc::Rc;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ModeKind {
    /// Uniform translations of the entire structure.
    ///
    /// Any given structure has three.
    /// (*technically*, fewer may be found if there are multiple
    ///  non-interacting parts in the structure; but currently,
    ///  the acoustic searcher explicitly does not support such
    ///  structures, because it has no strategy for identifying
    ///  piecewise translational modes)
    Translational,

    // FIXME: This seems unreliable.
    /// A mode where the force is not only at a zero, but also
    /// at an inflection point.
    ///
    /// There are at most three, depending on the dimensionality of
    /// the structure.
    Rotational,

    /// An imaginary mode that is not acoustic! Bad!
    Imaginary,

    /// An acoustic mode identified as such only because we were
    /// told that there might be some acoustic modes that are hard
    /// to spot.
    OtherAcoustic,

    Vibrational,
}

pub struct Colorful(pub ModeKind);

impl fmt::Display for ModeKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            ModeKind::Translational => "T",
            ModeKind::Rotational    => "R",
            ModeKind::Imaginary     => "‼",
            ModeKind::OtherAcoustic => "A",
            ModeKind::Vibrational   => "-",
        })
    }
}

impl fmt::Display for Colorful {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let color = match self.0 {
            ModeKind::Translational => ::ansi_term::Colour::Yellow.bold(),
            ModeKind::Rotational    => ::ansi_term::Colour::Purple.bold(),
            ModeKind::Imaginary     => ::ansi_term::Colour::Red.bold(),
            ModeKind::OtherAcoustic => ::ansi_term::Colour::Green.bold(),
            ModeKind::Vibrational   => ::ansi_term::Colour::White.normal(),
        };
        write!(f, "{}", ::ui::color::gpaint(color, self.0))
    }
}

pub(crate) fn perform_acoustic_search(
    pot: &PotentialBuilder,
    eigenvalues: &[f64],
    eigenvectors: &Basis3,
    coords: &Coords,
    meta: HList2<
        meta::SiteElements,
        meta::SiteMasses,
    >,
    settings: &cfg::AcousticSearch,
) -> FailResult<Rc<[ModeKind]>>
{Ok({
    let ev_directions = {
        EvDirections::from_eigenvectors(eigenvectors, meta.sift())
            .normalized()
    };

    let &cfg::AcousticSearch {
        expected_non_translations,
        displacement_distance,
        rotational_fdot_threshold,
        imaginary_fdot_threshold,
    } = settings;

    let zero_index = eigenvalues.iter().position(|&x| x >=  0.0).unwrap_or(eigenvalues.len());

    let mut kinds = vec![None; eigenvalues.len()];

    { // quickly spot translational modes

        let stop_index = {
            // We want to search a little bit past the negative eigenvalues, but not *too* far.
            // Surely, the frequencies of the acoustic modes must be less than this:
            const HARD_LIMIT: f64 = 10.0;

            eigenvalues.iter().position(|&x| x >= HARD_LIMIT).unwrap_or(eigenvalues.len())
        };

        let mut t_end = zero_index;
        for (i, direction) in ev_directions.0.iter().take(stop_index).enumerate() {
            if direction.acousticness() >= 0.95 {
                t_end = i + 1;
                kinds[i] = Some(ModeKind::Translational);
            }
        }

        // if there's more than three then the eigenbasis clearly isn't even orthonormal
        //
        // (NOTE: The above statement was originally written under the false assumption that
        //        the eigenvectors are the directions, but I think it is still correct?
        //        Haven't worked it out.)
        ensure!(
            kinds.iter().filter(|&x| x == &Some(ModeKind::Translational)).count() <= 3,
            "More than 3 pure translational modes! These eigenvectors are garbage!");

        // Everything after the last translational or negative mode is vibrational.
        kinds.truncate(t_end);
        kinds.resize(eigenvalues.len(), Some(ModeKind::Vibrational));
    }

    // look at the negative eigenvectors for rotations and true imaginary modes
    let mut diff_at_pos = pot.parallel(true).initialize_flat_diff_fn(coords, meta.sift())?;

    let pos_0 = coords.to_carts();
    let grad_0 = diff_at_pos(pos_0.flat())?.1;

    let mut rotational_count = 0;
    let mut uncertain_indices = vec![];
    for (i, direction) in ev_directions.0.iter().take(zero_index).enumerate() {
        if kinds[i].is_some() {
            continue;
        }

        let direction = direction.as_real_checked();
        let V(pos_l) = v(pos_0.flat()) - displacement_distance * v(direction.flat());
        let V(pos_r) = v(pos_0.flat()) + displacement_distance * v(direction.flat());
        let grad_l = diff_at_pos(&pos_l[..])?.1;
        let grad_r = diff_at_pos(&pos_r[..])?.1;

        let mut d_grad_l = v(&grad_0) - v(grad_l);
        let mut d_grad_r = v(grad_r) - v(&grad_0);

        // for rotational modes, the two d_grads should be anti-parallel.
        // for true imaginary modes, the two d_grads are.... uh, "pro-parallel".
        // for non-pure translational modes, the two d_grads could be just noise
        //   (they should be zero, but we're about to normalize them)
        //   which means they could also masquerade as one of the other types.
        for d_grad in vec![&mut d_grad_l, &mut d_grad_r] {
            *d_grad = match vnormalize(&*d_grad) {
                Err(BadNorm(_)) => {
                    // use a zero vector; it'll be classified as suspicious
                    d_grad.clone()
                },
                Ok(v) => v,
            };
        }

        let ddot = vdot(&d_grad_l, &d_grad_r);
        trace!("Examining mode {} ({:.7}) (ddot = {:.6})...", i + 1, eigenvalues[i], ddot);
        match ddot {
            dot if dot < -1.001 || 1.001 < dot => panic!("bad unit vector dot"),
            dot if dot <= -rotational_fdot_threshold => {
                kinds[i] = Some(ModeKind::Rotational);
                rotational_count += 1;
            },
            dot => {
                if dot < imaginary_fdot_threshold {
                    // This mode *could* be piecewise translational, which we don't support.
                    warn!(
                        "Could not classify mode at frequency {} (fdot = {:.6})!",
                        eigenvalues[i], dot,
                    );
                }
                uncertain_indices.push(i);
            },
        }
    }

    let fill = {
        if let Some(expected) = expected_non_translations {
            if expected - rotational_count >= uncertain_indices.len() {
                ModeKind::OtherAcoustic
            } else { ModeKind::Imaginary }
        } else { ModeKind::Imaginary }
    };

    for i in uncertain_indices {
        kinds[i] = Some(fill);
    }

    kinds.into_iter()
        .map(|opt| opt.expect("bug! every index should have been accounted for"))
        .collect::<Vec<_>>()
        .into()
})}
