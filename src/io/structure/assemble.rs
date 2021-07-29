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

use rsp2_structure::{CoordsKind, Lattice, Coords, Element};

use rsp2_soa_ops::{Part, Perm, Permute};
use rsp2_array_types::{M33, V3, dot};

/// A partially assembled structure for which
/// some parameters are still configurable
pub struct Assemble {
    // currently, only normals along a lattice vector are supported, and that
    // lattice vector must be orthogonal to the others.
    normal_axis: usize,

    /// scales each lattice vector.  The value on the normal axis is ignored.
    pub scale: [f64; 3],

    // Permutation that restores the original order of the atoms after the
    // positions from each layer have been concatenated.
    perm: Perm,

    // a lattice where lattice[normal_axis] is a unit vector and the others
    // are orthogonal to it.
    lattice: M33,
    // These are all zero along the normal axis
    fracs_in_plane: Vec<Vec<V3>>, // [layer][atom_in_layer]
    carts_along_normal: Vec<Vec<f64>>, // [layer][atom_in_layer]

    /// separation across periodic boundary (as the center-center distance
    /// between the first layer encountered on either side of the boundary)
    pub vacuum_sep: f64,
    layer_seps: Vec<f64>, // [layer] (length: nlayer - 1)
    pub atoms: Vec<Element>
}

impl Assemble {
    /// Allows setting layer separations (as center-center distances)
    pub fn layer_seps(&mut self) -> &mut [f64]
    { &mut self.layer_seps }

    pub fn num_layer_seps(&self) -> usize
    { self.layer_seps.len() }

    pub fn normal_axis(&self) -> usize
    { self.normal_axis }

    pub fn num_atoms(&self) -> usize
    { self.fracs_in_plane.iter().fold(0, |acc, v| acc + v.len()) }

    pub fn atom_layers(&self) -> Vec<usize>
    {
        use std::iter::repeat;
        let mut vec = vec![];
        for (layer, data) in self.fracs_in_plane.iter().enumerate() {
            vec.extend(repeat(layer).take(data.len()));
        }
        vec.permuted_by(&self.perm)
    }

    pub fn assemble(&self) -> Coords
    {
        let lattice = {
            let mut scales = self.scale;
            scales[self.normal_axis] = self.get_z_length();

            // (assumption in our use of `scales` below)
            assert!(f64::abs(self.lattice[self.normal_axis].sqnorm() - 1.0) < 1e-5);

            let lattice = Lattice::new(&self.lattice);
            &Lattice::diagonal(&scales) * &lattice
        };
        let layer_zs = self.get_z_positions();

        let mut full_carts = vec![];

        // each layer
        let it = zip_eq!(&self.fracs_in_plane, &self.carts_along_normal, layer_zs);
        for (plane_fracs, z_carts, z_offset) in it {
            // convert the two fractional coords into cartesian
            let mut carts = CoordsKind::Fracs(plane_fracs).to_carts(&lattice);

            // add in the final coordinate, which is already cartesian
            let unit_z = self.lattice[self.normal_axis].unit();
            for (v, z) in zip_eq!(&mut carts, z_carts) {
                *v += (z + z_offset) * unit_z;
            }

            full_carts.extend(carts);
        }

        Coords::new(lattice, CoordsKind::Carts(full_carts))
            .permuted_by(&self.perm)
    }

    fn get_z_length(&self) -> f64
    { self.layer_seps.iter().sum::<f64>() + self.vacuum_sep }

    fn get_z_positions(&self) -> Vec<f64>
    {
        let mut zs = vec![0.0];
        zs.extend(self.layer_seps.iter().cloned());
        for i in 1..zs.len() {
            zs[i] += zs[i-1];
        }

        // put half of the vacuum separation on either side
        // (i.e. shift by 1/2 vacuum sep)
        for z in &mut zs {
            *z += 0.5 * self.vacuum_sep;
        }
        zs
    }
}

pub struct RawAssemble {
    /// The axis normal to the layers, as the index of a lattice vector.
    pub normal_axis: usize,

    /// A lattice in which the lattice vector for `normal_axis` is orthogonal
    /// to the other two.  The length of the `normal_axis` vector is ignored.
    pub lattice: Lattice,

    /// Fractional coordinates for each layer.  Must all be zero in the normal axis.
    pub fracs_in_plane: Vec<Vec<V3>>, // [layer][atom_in_layer]

    /// Cartesian coordinates along the normal within each layer.  This allows layers
    /// to have a "thickness" to them.
    ///
    /// Only the variation in value within each layer matters.  It is important that,
    /// within each layer, these coordinates form a contiguous image.
    pub carts_along_normal: Vec<Vec<f64>>, // [layer][atom_in_layer]

    // NOTE: bulk has no vacuum sep, and 'nlayer' layer seps.
    //       The current API does not present a nice way of handling this.
    /// Initial separation across periodic boundary.
    pub initial_vacuum_sep: f64,
    /// Initial separations between layers.
    pub initial_layer_seps: Vec<f64>, // [layer] (length: nlayer - 1)

    /// Initial scale factors.  The value on `normal_axis` is ignored.
    pub initial_scale: Option<[f64; 3]>,

    /// The partition that decided the order of the atoms in `fracs_in_plane`.
    ///
    /// If provided, it is used to make sure that the order of the atoms in the assembled
    /// structure matches the original order of the structure that was partitioned.
    pub part: Option<Part<usize>>,

    /// Adds a sanity check that each pair of successive atoms along a layer lie within
    /// this distance from each other (with a little extra bit of fuzz). Construction
    /// of an `Assemble` will fail if this is untrue.
    ///
    /// This is to help catch bugs related to accidentally wrapping `carts_along_normal`
    /// across a periodic boundary.
    pub check_intralayer_distance: Option<f64>,
    pub atoms: Vec<Element>

}

impl Assemble {
    pub fn from_raw(raw: RawAssemble) -> FailResult<Self> {
        let RawAssemble {
            normal_axis, lattice, fracs_in_plane, carts_along_normal,
            initial_vacuum_sep, initial_layer_seps, initial_scale,
            check_intralayer_distance, part, atoms
        } = raw;
        assert!(normal_axis < 3);

        let other_axes = (0..3).filter(|&k| k != normal_axis).collect::<Vec<_>>();

        // selected vector must be orthogonal to the others.
        // It is normalized so that it can be easily scaled later.
        let lattice = {
            let units = V3(lattice.vectors().clone()).map(|v| v.unit()).0;
            for &k in &other_axes {
                let d = dot(&units[normal_axis], &units[k]);
                ensure!(
                    d.abs() < 1e-6,
                    "Normal vector of Assemble must be orthogonal to the others \
                    (got normalized dot product of {})", d,
                );
            }

            let mut matrix = lattice.matrix().clone();
            matrix[normal_axis] = units[normal_axis];
            matrix
        };

        // while arbitrary vectors are supported *in spirit,* this code is almost
        // always only ever run for cases where it is along the corresponding
        // cartesian axis.  Warn about undertested code paths.
        if other_axes.iter().any(|&k| f64::abs(lattice[normal_axis][k]) > 1e-9) {
            warn_once!(
                "Your layer normal is along a nontrivial direction.  In *theory* rsp2 should \
                be able to handle this, but this functionality is seldom used so it is poorly \
                tested.  Ping Michael if the structure files created by rsp2 don't look right."
            );
        }

        for vs in &fracs_in_plane {
            for v in vs {
                ensure!(v[normal_axis] == 0.0, "Frac positions along normal must be zero!");
            }
        }

        // layer_seps and vacuum_sep will be interpreted as distance from
        // center to center.  Maybe this could be made configurable later.
        //
        // This is accomplished by simply recentering the coordinates prior
        // to construction of the Assemble, which will continue to treat them
        // as basically zero-width when computing layer offsets.
        let mut carts_along_normal = carts_along_normal;
        for v in &mut carts_along_normal {
            if let Some(mut thresh) = check_intralayer_distance {
                thresh *= 1.0 + 1e-8;
                let mut sorted = v.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for w in sorted.windows(2) {
                    let distance = w[1] - w[0];
                    ensure!(
                        distance < thresh,
                        "check_intralayer_distance failed! (distance = {}) Was a layer \
                        wrapped across the periodic boundary?", distance);
                }
            }

            use std::f64::INFINITY;
            let min = v.iter().cloned().fold(INFINITY, f64::min);
            let max = v.iter().cloned().fold(-INFINITY, f64::max);
            let center = 0.5 * (min + max);
            assert!(center.is_finite());
            for x in v {
                *x -= center;
            }
        }

        let perm = match part {
            None => Perm::eye(fracs_in_plane.iter().fold(0, |acc, v| acc + v.len())),
            Some(part) => {
                warn_once!(
                    "Entering undertested situation; ideally, rsp2 should restore the \
                    original order of the atoms after optimizing parameters.  If output \
                    files reflect a different ordering, this is a bug (let Michael know)!"
                );
                part.restoring_perm()
            },
        };

        let scale = initial_scale.unwrap_or([1.0; 3]);
        let vacuum_sep = initial_vacuum_sep;
        let layer_seps = initial_layer_seps;
        Ok(Assemble {
            normal_axis, scale, lattice, fracs_in_plane, carts_along_normal,
            vacuum_sep, layer_seps, perm, atoms
        })
    }
}
