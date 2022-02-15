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

#[macro_use] extern crate serde;
#[macro_use] extern crate rsp2_clap;
#[macro_use] extern crate log;
#[macro_use] extern crate failure;
#[macro_use] extern crate rsp2_config_utils;

use failure::ResultExt;

use rsp2_structure::{Element, bonds::FracBonds};
use rsp2_structure_io::Poscar;
use rsp2_array_types::M33;
use rsp2_dynmat::DynamicalMatrix;
use rsp2_fs_util as fsx;
use rsp2_config_utils::merge::ConfigSources;

pub type FailResult<T> = Result<T, failure::Error>; // FIXME forced by usage in other rsp2 crates

pub fn main() {
    _main().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });
}

pub fn _main() -> FailResult<()> {
    env_logger::init();
    let app = {
        clap::App::new("rsp2-bond-polarizability")
            .about("Computes the dynamical matrix at a qpoint.")
            .args(&[
                arg!( input=STRUCTURE "Input structure (POSCAR)"),
                arg!(*dynmat [--dynmat]=DYNMAT "Dynamical matrix at gamma. (.json.gz)"), // TODO DOCUMENT FORMAT
                arg!( bond_radius [--bond-radius]=RADIUS "Bond radius (A)"),
                arg!( temperature [--temperature]=TEMPERATURE "Temperature (K)"),
                arg!( settings [--config][-c]=SETTINGS... "Settings yaml file"),
                arg!(*out_path [--output][-o]=OUTPATH "Output JSON path"),
            ])
    };
    let matches = app.get_matches();
    let structure_path = matches.value_of("input").unwrap();
    let dynmat_path = matches.value_of("dynmat").unwrap();
    let out_path = matches.value_of("out_path").unwrap();
    let bond_radius = matches.value_of("bond_radius").unwrap_or("1.8").parse::<f64>().with_context(|e| format!("in bond-radius: {}", e))?;
    let configs = ConfigSources::resolve_from_args(matches.values_of_lossy("settings").unwrap_or(vec![]))?;
    let settings = configs.deserialize::<Settings>()?;
    let temperature = matches.value_of("temperature").unwrap_or("0").parse::<f64>().with_context(|e| format!("in temperature: {}", e))?;
    let temperatures = vec![temperature];  // TODO: way of specifying multiple temperatures

    let poscar = Poscar::from_reader(fsx::open(structure_path)?).with_context(|e| format!("reading {}: {}", structure_path, e))?;
    trace!("Computing bond graph...");
    let frac_bonds = FracBonds::compute(&poscar.coords, bond_radius)?;
    let cart_bonds = frac_bonds.to_cart_bonds(&poscar.coords);

    let dynmat = read_dynmat(dynmat_path).with_context(|e| format!("while reading {}: {}", dynmat_path, e))?;
    let (eigenvalues, eigenvectors) = dynmat.compute_eigensolutions_dense_gamma();

    let frequencies: Vec<f64> = eigenvalues.eigenvalues.into_iter().map(eigenvalue_to_frequency).collect::<Vec<_>>();

    let tensor_objects = rsp2_bond_polarizability::Input {
        /// Normal mode frequencies, in cm^-1.
        ev_frequencies: &frequencies,
        /// Normal mode eigenvectors, normalized.
        ev_eigenvectors: eigenvectors.iter().map(|v| &v[..]),
        /// Element of each site.  Used to determine bond polarizability coefficients.
        site_elements: &poscar.elements,
        /// Masses of each site, in AMU.
        site_masses: &poscar.elements.iter().map(|&elem| element_mass(elem)).collect::<FailResult<Vec<_>>>()?,
        bonds: &cart_bonds,
        settings: &settings.bond_polarizability,
    }.compute_ev_raman_tensors()?;
    let raman_tensors = tensor_objects.iter().map(|tensor| tensor.tensor().clone()).collect();
    let raman_intensity_prefactors = temperatures.iter().map(|&temperature| {
        tensor_objects.iter().map(|tensor| tensor.intensity_prefactor(temperature)).collect()
    }).collect();

    let out_file = fsx::create(out_path)?;
    serde_json::to_writer(out_file, &Output {
        frequencies,
        temperatures,
        raman_intensity_prefactors,
        raman_tensors,
    })?;
    Ok(())
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Settings {
    #[serde(default)]
    bond_polarizability: rsp2_bond_polarizability::Settings,
}

derive_yaml_read! { Settings }

fn read_dynmat(path: &str) -> FailResult<DynamicalMatrix> {
    let lower_path = path.to_ascii_lowercase();
    if lower_path.ends_with(".npz") {
        let stem = &path[..path.len() - ".npz".len()];
        bail!("\
rsp2-bond-polarizability does not support .npz format for dynmats, try the following command:
(<rsp2_dir>/src/python must be on your PYTHONPATH!)

    python3 -m rsp2.cli.convert_dynmat {} --output {}.json.gz\
", path, stem);
    }
    let mut reader: Box<dyn std::io::Read> = Box::new(fsx::open(path)?);
    if path.to_ascii_lowercase().ends_with(".gz") {
        reader = Box::new(flate2::read::GzDecoder::new(reader));
    }
    let cereal = serde_json::from_reader(reader)?;
    DynamicalMatrix::from_cereal(cereal)
}

fn element_mass(elem: Element) -> FailResult<f64>
{Ok({
    match elem {
        Element::HYDROGEN => 1.00794,
        Element::CARBON => 12.0107,
        _ => failure::bail!("no default mass for element {}.", elem.symbol()),
    }
})}

// ================================================

/// All output data.
#[derive(serde::Serialize)]
struct Output {
    /// [eigensol] -> frequency
    frequencies: Vec<f64>,
    /// [temperature] -> temperature
    temperatures: Vec<f64>,
    /// [temperature][eigensol] -> (bose_occupation / frequency) for each phonon.
    ///
    /// Multiply this against `abs(unit_in * tensor * unit_out)**2` when computing intensity
    /// to ensure you have the correct frequency dependence.
    raman_intensity_prefactors: Vec<Vec<f64>>,
    /// [eigensol][ax_1][ax_2] -> derivative of polarizability
    raman_tensors: Vec<M33>,
}

// ================================================

// Conversion factor phonopy uses to scale the eigenvalues to THz angular momentum.
//    = sqrt(eV/amu)/angstrom/(2*pi)/THz
const SQRT_EIGENVALUE_TO_THZ: f64 = 15.6333043006705;
//    = THz / (c / cm)
const THZ_TO_WAVENUMBER: f64 = 33.3564095198152;
const SQRT_EIGENVALUE_TO_WAVENUMBER: f64 = SQRT_EIGENVALUE_TO_THZ * THZ_TO_WAVENUMBER;
fn eigenvalue_to_frequency(val: f64) -> f64 {
    f64::sqrt(f64::abs(val)) * f64::signum(val) * SQRT_EIGENVALUE_TO_WAVENUMBER
}
