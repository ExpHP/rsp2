use ::errors::{Result, ok};
use ::rsp2_tasks_config::YamlRead;

use ::cmd::trial::TheOnlyGlobalTrial;
use ::cmd::trial::{CliDeserialize};
use ::util::{canonicalize};
use ::util::ArgMatchesExt;

/// Uses a precompiled from_value in a separate crate to avoid unnecessary codegen.
pub fn from_value<T: YamlRead>(yaml: ::serde_yaml::Value) -> Result<T>
{ Ok(YamlRead::from_value(yaml)?) }

fn wrap_result_main<F>(main: F)
where F: FnOnce() -> Result<()>
{ main().unwrap_or_else(|e| panic!("{}", e.display_chain())); }

pub fn rsp2() {
    use ::cmd::CliArgs;

    impl CliDeserialize for CliArgs {
        fn _augment_clap_app<'a, 'b>(app: ::clap::App<'a, 'b>) -> ::clap::App<'a, 'b> {
            app.args(&[
                arg!( save_bands [--save-bands] "save phonopy directory with bands at gamma"),
            ])
        }

        fn _resolve_args(m: &::clap::ArgMatches) -> Result<Self>
        { Ok(CliArgs {
            save_bands: m.is_present("save_bands"),
        })}
    }

    wrap_result_main(|| {
        let (app, de) = CliDeserialize::augment_clap_app({
            app_from_crate!(", ")
                .args(&[
                    arg!( input=POSCAR "POSCAR"),
                ])
        });
        let matches = app.get_matches();
        let (dir_args, extra_args) = de.resolve_args(&matches)?;

        let input = canonicalize(matches.expect_value_of("input"))?;

        TheOnlyGlobalTrial::run_in_new_dir(dir_args, |trial, yaml| ok({
            let settings = from_value(yaml)?;
            trial.run_relax_with_eigenvectors(&settings, &input, extra_args)?;
        }))
    });
}

pub fn shear_plot() {
    wrap_result_main(|| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-shear-plot")
                .version("negative 0.00.3-734.bubbles")
                .author(crate_authors!{", "})
                .about("blah")
                .args(&[
                    arg!( input=FORCES_DIR "phonopy forces dir (try --save-bands in main script)"),
                ])
        });
        let matches = app.get_matches();
        let dir_args = de.resolve_args(&matches)?;

        let input = canonicalize(matches.expect_value_of("input"))?;

        TheOnlyGlobalTrial::run_in_new_dir(dir_args, |trial, yaml| ok({
            let settings = from_value(yaml)?;
            trial.run_energy_surface(&settings, &input)?;
        }))
    });
}

pub fn save_bands_after_the_fact() {
    wrap_result_main(|| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-shear-plot")
                .version("negative 0.00.3-734.bubbles")
                .author(crate_authors!{", "})
                .about("blah")
        });
        let matches = app.get_matches();
        let dir_args = de.resolve_args(&matches)?;

        TheOnlyGlobalTrial::run_in_existing_dir(dir_args, |trial, yaml| ok({
            let settings = from_value(yaml)?;
            trial.run_save_bands_after_the_fact(&settings)?;
        }))
    });
}
