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

//! All of the post-processing that occurs after config merging is written here.
//!
//! It mostly deals with mapping deprecated forms to their new correct form.

use crate::config::*;
use failure::Error;
use std::collections::HashMap;

impl Settings {
    pub fn validate(mut self) -> Result<ValidatedSettings, Error> {
        fill_lammps_from_deprecated(
            &mut self.lammps,
            &mut self._deprecated_lammps_settings,
        );
        fix_version(&mut self.version)?;

        if let Some(phonons) = &mut self.phonons {
            fix_deprecated_eigensolver(&mut phonons.eigensolver);
            check_phonons(&phonons, &self.potential)?;
        }

        Ok(ValidatedSettings(self))
    }
}

impl EnergyPlotSettings {
    pub fn validate(mut self) -> Result<ValidatedEnergyPlotSettings, Error> {
        fill_lammps_from_deprecated(
            &mut self.lammps,
            &mut self._deprecated_lammps_settings,
        );
        fix_version(&mut self.version)?;

        Ok(ValidatedEnergyPlotSettings(self))
    }
}

impl Potential {
    pub fn validate(self) -> Result<ValidatedPotential, Error> {
        let mut found_deprecated = false;

        // Replace all deprecated potentials.
        let potentials: Vec<_> = {
            self.into_vec().into_iter().map(|pot| match pot {
                PotentialKind::OldLammpsRebo(inner) => {
                    found_deprecated = true;
                    PotentialKind::Lammps(LammpsPotentialKind::Rebo(inner))
                },

                PotentialKind::OldLammpsAirebo(inner) => {
                    found_deprecated = true;
                    PotentialKind::Lammps(LammpsPotentialKind::Airebo(inner))
                },

                PotentialKind::OldLammpsKolmogorovCrespiZ(inner) => {
                    found_deprecated = true;
                    PotentialKind::Lammps(LammpsPotentialKind::KolmogorovCrespiZ(inner))
                },

                PotentialKind::OldLammpsKolmogorovCrespiFull(inner) => {
                    found_deprecated = true;
                    PotentialKind::Lammps(LammpsPotentialKind::KolmogorovCrespiFull(inner))
                },

                PotentialKind::OldReboNew(inner) => {
                    found_deprecated = true;
                    PotentialKind::ReboNonreactive(inner)
                },

                PotentialKind::OldKolmogorovCrespiZ(inner) => {
                    found_deprecated = true;

                    let OldPotentialKolmogorovCrespiZ {
                        cutoff_begin, cutoff_transition_dist, skin_depth, skin_check_frequency,
                    } = inner;
                    PotentialKind::KolmogorovCrespi(PotentialKolmogorovCrespi {
                        cutoff_begin, cutoff_transition_dist, skin_depth, skin_check_frequency,
                        normals: KolmogorovCrespiNormals::Z {},
                        params: KolmogorovCrespiParams::Original,
                    })
                },

                pot@PotentialKind::ReboNonreactive(..) |
                pot@PotentialKind::KolmogorovCrespi(..) |
                pot@PotentialKind::DftbPlus(..) |
                pot@PotentialKind::Lammps(..) |
                pot@PotentialKind::TestZero |
                pot@PotentialKind::TestChainify => pot,
            }).collect()
        };

        let out = Potential(potentials);

        if found_deprecated {
            let yaml: HashMap<String, Potential> = from_json!({
                "potential": out.clone(),
            });

            warn!("\
                Found deprecated items in `potential` config! \
                The equivalent config is:\n{}\
            ", ::serde_yaml::to_string(&yaml).expect("should not fail"));
        }

        if out.as_slice().iter().filter(|x| matches!(PotentialKind::DftbPlus(_), x)).count() > 1 {
            bail!("The `dftb+` potential may only be listed at most once!");
        }

        if out.as_slice().iter().filter(|x| matches!(PotentialKind::Lammps(_), x)).count() > 1 {
            bail!("The `lammps` potential may only be listed at most once!");
        }

        if matches!([PotentialKind::KolmogorovCrespi(_)], out.as_slice()) {
            warn!("\
                You are using the Kolmogorov/Crespi potential alone, with no intralayer term. \
                This is a bit unusual; did you mean to add a REBO term? (e.g. `nonreactive-rebo`)\
            ");
        }

        Ok(ValidatedPotential(out))
    }
}

fn fix_version(it: &mut Option<u32>) -> Result<(), Error> {
    match *it {
        Some(x) if x == 0 || x > MAX_VERSION => {
            bail!("`version: {}` is invalid. (1 <= version <= {})", x, MAX_VERSION);
        },
        None => {
            warn!("\
                Settings file has no `version` field! Assuming `version: 1`. \
                (the latest is version {})\
            ", MAX_VERSION);
            *it = Some(1);
        },
        _ => {},
    };

    Ok(())
}

fn fix_deprecated_eigensolver(it: &mut PhononEigensolver) {
    match *it {
        PhononEigensolver::Phonopy(AlwaysFail(never, _)) => match never {},
        PhononEigensolver::Rsp2 { dense: true, .. } => {
            warn!("`phonon.eigensolver: rsp2 {{ dense: true }}` is deprecated. Use the `dense` eigensolver.");
            *it = PhononEigensolver::Dense {};
        },
        PhononEigensolver::Rsp2 { dense: false, shift_invert_attempts, how_many } => {
            warn!("`phonon.eigensolver: rsp2 {{ dense: false }}` is deprecated. Use the `sparse` eigensolver.");
            *it = PhononEigensolver::Sparse { shift_invert_attempts, how_many };
        },
        PhononEigensolver::Dense { .. } => {},
        PhononEigensolver::Sparse { .. } => {},
    };
}

fn fill_lammps_from_deprecated(
    new: &mut Lammps,
    old: &mut DeprecatedLammpsSettings,
) {
    let Lammps { processor_axis_mask, update_style } = new;

    if let Some(value) = old.lammps_processor_axis_mask.take() {
        warn!("\
            `lammps-processor-axis-mask` is deprecated. \
            It now lives at `lammps.processor-axis-mask`.\
        ");
        processor_axis_mask.0.get_or_insert(value);
    }
    processor_axis_mask.0.get_or_insert([true; 3]);

    if let Some(value) = old.lammps_update_style.take() {
        warn!("\
            `lammps-update-style` is deprecated. \
            It now lives at `lammps.update-style`.\
        ");
        update_style.0.get_or_insert(value);
    }
    update_style.0.get_or_insert_with(Default::default);
}

fn check_phonons(phonons: &Phonons, potential: &ValidatedPotential) -> Result<(), Error> {
    let ValidatedPotential(Potential(kinds)) = potential;

    if phonons.analytic_hessian {
        for kind in kinds {
            match kind {
                PotentialKind::OldKolmogorovCrespiZ(_) => {},
                _ => bail!{"The chosen potential does not support analytic-hessian mode."},
            }
        }
    } else {
        // numerical hessian
        if phonons.symmetry_tolerance.is_none() {
            bail!("phonons.symmetry-tolerance is required.");
        }
        if phonons.displacement_distance.is_none() {
            bail!("phonons.displacement-distance is required.");
        }
    }

    Ok(())
}
