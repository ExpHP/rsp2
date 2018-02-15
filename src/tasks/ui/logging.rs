use ::errors::{Result, ok};

use ::std::fmt;
use ::std::fs::File;
use ::std::path::{Path};
use ::log::{LogLevel, LogRecord};
use ::fern::FernLog;

pub use self::fern::{DelayedLogFile, GLOBAL_LOGFILE};
mod fern {
    use super::*;
    use ::std::sync::RwLock;
    use ::std::io::prelude::*;

    /// A log file for fern that can be created *after* logger initialization.
    #[derive(Debug, Default)]
    pub struct DelayedLogFile {
        // (the lock is only to make it safe to insert or replace the file)
        // (the hungarian is because 'file.read()' would be misleading)
        file_rw: RwLock<Option<File>>,
    }

    lazy_static! {
        /// The DelayedLogFile given to fern.
        pub static ref GLOBAL_LOGFILE: DelayedLogFile = Default::default();
    }

    impl DelayedLogFile {
        pub fn start<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            // (note: the Err case here is PoisonError)
            if let Ok(mut file) = self.file_rw.write() {
                if file.is_some() {
                    bail!("The logfile has already been created!");
                }
                *file = Some(::fern::log_file(path)?);
            }
            Ok(())
        }
    }

    impl FernLog for &'static DelayedLogFile {
        fn log_args(&self, payload: &fmt::Arguments, _original: &LogRecord) {
            // (note: the Err case here is PoisonError)
            if let Ok(file) = self.file_rw.read() {
                if let Some(mut file) = (*file).as_ref() {
                    let _ = writeln!(file, "{}", payload);
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Verbosity { Default, Loud }

impl Default for Verbosity {
    fn default() -> Self { Verbosity::Default }
}

impl Verbosity {
    /// Any integer will be accepted; the level will be truncated
    /// to the most extreme value supported.
    fn from_int(level: i32) -> Self
    { match level > 0 {
        true => Verbosity::Loud,
        false => Verbosity::Default,
    }}

    pub fn from_env() -> Result<Self>
    { ::env::verbosity().map(Self::from_int) }
}

/// Set the global logger, enabling the use of `log!()` macros.
/// This can only be done once.
pub fn init_global_logger() -> Result<()>
{ok({
    use ::std::time::Instant;
    use ::log::LogLevelFilter as LevelFilter;

    let verbosity = Verbosity::from_env()?;

    let start = Instant::now();
    let mut fern = ::fern::Dispatch::new();
    fern = fern.format(move |out, message, record| {
            let t = start.elapsed();
            out.finish(format_args!("[{:>4}.{:03}s][{}][{}] {}",
                t.as_secs(),
                t.subsec_nanos() / 1_000_000,
                record.target(),
                ColorizedLevel(record.level()),
                message))
        })
        .level(LevelFilter::Debug)
        .level_for("rsp2_tasks", LevelFilter::Trace)
        .level_for("rsp2_minimize", LevelFilter::Trace)
        .level_for("rsp2_phonopy_io", LevelFilter::Trace)
        .level_for("rsp2_minimize::exact_ls", match verbosity {
            Verbosity::Default => LevelFilter::Debug,
            Verbosity::Loud => LevelFilter::Trace,
        })
        // Yes, this really is deliberately boxing a reference (a 'static one).
        // The reason is simply because chain asks for a Box.
        .chain(Box::new(&*GLOBAL_LOGFILE) as Box<FernLog>)
        .chain(::std::io::stdout());

    fern.apply()?;
})}

#[derive(Debug, Copy, Clone)]
pub struct ColorizedLevel(pub LogLevel);
impl fmt::Display for ColorizedLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let style = match self.0 {
            LogLevel::Error => ::ansi_term::Colour::Red.bold(),
            LogLevel::Warn  => ::ansi_term::Colour::Red.normal(),
            LogLevel::Info  => ::ansi_term::Colour::Cyan.bold(),
            LogLevel::Debug => ::ansi_term::Colour::Yellow.dimmed(),
            LogLevel::Trace => ::ansi_term::Colour::Cyan.normal(),
        };
        write!(f, "{}", ::ui::color::gpaint(style, self.0))
    }
}
