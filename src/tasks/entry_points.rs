use ::errors::{Result, ok};
use ::rsp2_tasks_config::YamlRead;

use ::cmd::trial::{CommonArgs, TheOnlyGlobalTrial};
use ::util::canonicalize;

/// Uses a precompiled from_value in a separate crate to avoid unnecessary codegen.
pub fn from_value<T: YamlRead>(yaml: ::serde_yaml::Value) -> Result<T>
{ Ok(YamlRead::from_value(yaml)?) }

pub fn rsp2() {
    fn inner() -> Result<()> {
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

        let args = ::relax_with_eigenvectors::CliArgs {
            save_bands: matches.is_present("save_bands"),
        };

        TheOnlyGlobalTrial::from_args(common_args)
            .will_now_commence(|trial, yaml| ok({
                let settings = from_value(yaml)?;
                trial.run_relax_with_eigenvectors(&settings, &input, args)?;
            }))
    }

    if let Err(e) = inner() {
        panic!("{}", e.display_chain());
    };
}
