use ::errors::{Result, ok};
use ::rsp2_tasks_config::YamlRead;

use ::cmd::trial::{CommonArgs, TheOnlyGlobalTrial};
use ::util::canonicalize;

/// Uses a precompiled from_value in a separate crate to avoid unnecessary codegen.
pub fn from_value<T: YamlRead>(yaml: ::serde_yaml::Value) -> Result<T>
{ Ok(YamlRead::from_value(yaml)?) }

fn wrap_result_main<F>(main: F)
where F: FnOnce() -> Result<()>
{ main().unwrap_or_else(|e| panic!("{}", e.display_chain())); }

pub fn rsp2() {
    wrap_result_main(|| {
        let (app, resolve_common_args) = CommonArgs::augment_clap_app({
            app_from_crate!(", ")
                .args(&[
                    arg!( input=POSCAR "POSCAR"),
                    arg!( save_bands [--save-bands] "save phonopy directory with bands at gamma"),
                ])
        });
        let matches = app.get_matches();
        let common_args = resolve_common_args(&matches)?;

        let input = matches.value_of("input").expect("(BUG) input was required!");
        let input = canonicalize(input)?;

        let args = ::cmd::CliArgs {
            save_bands: matches.is_present("save_bands"),
        };

        TheOnlyGlobalTrial::from_args(common_args)
            .will_now_commence(|trial, yaml| ok({
                let settings = from_value(yaml)?;
                trial.run_relax_with_eigenvectors(&settings, &input, args)?;
            }))
    });
}

pub fn shear_plot() {
    wrap_result_main(|| {
        let (app, resolve_common_args) = CommonArgs::augment_clap_app({
            ::clap::App::new("rsp2-shear-plot")
                .version("negative 0.00.3-734.bubbles")
                .author(crate_authors!{", "})
                .about("blah")
                .args(&[
                    arg!( input=FORCES_DIR "phonopy forces dir (try --save-bands in main script)"),
                ])
        });
        let matches = app.get_matches();
        let common_args = resolve_common_args(&matches)?;

        let input = matches.value_of("input").expect("(BUG) input was required!");
        let input = canonicalize(input)?;

        TheOnlyGlobalTrial::from_args(common_args)
            .will_now_commence(|trial, yaml| ok({
                let settings = from_value(yaml)?;
                trial.run_energy_surface(&settings, &input)?;
            }))
    });
}
