use ::errors::{Result};

use ::clap;
use ::cmd::trial::{TrialDir, NewTrialDirArgs};
use ::cmd::StructureFileType;
use ::path_abs::{PathDir, PathFile};
use ::ui::logging::init_global_logger;
use ::ui::cfg_merging::ConfigSources;
use ::ui::cli_deserialize::CliDeserialize;
use ::util::ext_traits::{ArgMatchesExt};

fn wrap_result_main<F>(main: F)
    where F: FnOnce() -> Result<()>,
{
    main().unwrap_or_else(|e| {
        error!("{}", e.display_chain());
        ::std::process::exit(1);
    });
}

impl CliDeserialize for NewTrialDirArgs {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!(*trial_dir [-o][--output]=OUTDIR "output trial directory"),
            arg!( force [-f][--force] "replace existing output directories"),
            arg!(*config [-c][--config]=CONFIG... "\
                config yaml, provided as either a filepath, or as an embedded literal \
                (via syntax described below). \
                When provided multiple times, the configs are merged according to some fairly \
                dumb strategy, with preference to the values supplied in later arguments. \
                \n\n\
                Literals are written as '--config [NESTED_KEY]:VALID_YAML', \
                where NESTED_KEY is an optional '.'-separated sequence of string keys, \
                and the ':' is a literal colon. When provided, NESTED_KEY constructs a nested \
                mapping (so `--config a.b.c:[2]` is equivalent to `--config :{a: {b: {c: [2]}}}`.\
                \n\n\
                Note that detection of filepaths versus literals is based solely \
                on the presence of a colon, and no means of escaping one in a path \
                are currently provided.\
            "),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches) -> Result<Self>
    { Ok(NewTrialDirArgs {
        config_sources: ConfigSources::resolve_from_args(m.expect_values_of("config"))?,
        err_if_existing: !m.is_present("force"),
        // FIXME factor out 'absolute()'
        trial_dir: PathDir::current_dir()?.join(m.expect_value_of("trial_dir")),
    })}
}

// (not sure why `impl CliDeserialize for Option<StructureFileType>` isn't good enough
//  but rustc says Option<_> doesn't impl CliDeserialize, even when it ought to be
//  inferrable that the _ is StructureFileType)
pub struct OptionalFileType(Option<StructureFileType>);

impl CliDeserialize for OptionalFileType {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!( structure_type [--structure-type]=STYPE "Structure filetype. \
                [choices: poscar, layers, guess] [default: guess]\
            "),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches) -> Result<Self> {
        Ok(OptionalFileType({
            if let Some(s) = m.value_of("structure_type") {
                match s {
                    "poscar" => Some(StructureFileType::Poscar),
                    "layers" => Some(StructureFileType::LayersYaml),
                    "guess" => None,
                    _ => bail!("invalid setting for --structure-type"),
                }
            } else { None }
        }))
    }
}

impl StructureFileType {
    pub fn guess(path: &PathFile) -> StructureFileType {
        if let Some(ext) = path.extension() {
            match ext.to_string_lossy().as_ref() {
                "yaml" => return StructureFileType::LayersYaml,
                _ => {},
            }
        }
        StructureFileType::Poscar
    }
}

// -------------------------------------------------------------------------------------

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
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            app_from_crate!(", ")
                .args(&[
                    arg!( input=STRUCTURE "input file for structure"),
                ])
        });
        let matches = app.get_matches();
        let (dir_args, (filetype, extra_args)) = de.resolve_args(&matches)?;

        let input = PathFile::new(matches.expect_value_of("input"))?;

        let OptionalFileType(filetype) = filetype;
        let filetype = filetype.unwrap_or_else(|| StructureFileType::guess(&input));

        let trial = TrialDir::create_new(dir_args)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        trial.run_relax_with_eigenvectors(&settings, filetype, &input, extra_args)
    });
}

pub fn shear_plot() {
    wrap_result_main(|| {
        let logfile = init_global_logger()?;

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

        let input = PathDir::new(matches.expect_value_of("input"))?;

        let trial = TrialDir::create_new(dir_args)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        // HACK: Incomplete refactoring; canonicalize() and CanonicalPath should
        //       no longer exist, and this function should take TrialDir
        trial.run_energy_surface(&settings, &input)
    });
}

pub fn save_bands_after_the_fact() {
    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-shear-plot")
                .version("negative 0.00.3-734.bubbles")
                .author(crate_authors!{", "})
                .about("blah")
                .args(&[
                    arg!( trial_dir=TRIAL_DIR "existing trial directory"),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        let trial = PathDir::new(matches.expect_value_of("trial_dir"))?;
        let trial = TrialDir::from_existing(&trial)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        trial.run_save_bands_after_the_fact(&settings)
    });
}

pub fn bond_test() {
    wrap_result_main(|| {
        let _logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-bond-test")
                .version("negative 0.00.3-734.bubbles")
                .author(crate_authors!{", "})
                .about("blah")
                .args(&[
                    arg!( input=STRUCTURE ""),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        use ::rsp2_structure_gen::load_layers_yaml;

        let input = PathFile::new(matches.expect_value_of("input"))?;
        let structure = {
            let mut builder = load_layers_yaml(input.read()?)?;
            builder.scale = 2.46;
            for sep in builder.layer_seps() {
                *sep = 3.38;
            }
            builder.assemble()
        };

        let bonds = ::math::bonds::Bonds::from_brute_force_very_dumb(&structure, 1.8);

        println!("{:?}", bonds);
        Ok(())
    });
}
