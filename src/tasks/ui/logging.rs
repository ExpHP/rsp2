use ::errors::{Result, ok};

use ::std::fmt;
use ::std::path::{Path, PathBuf};
use ::log::LogLevel;

/// Builder-style setup for logging
#[derive(Debug, Clone, Default)]
pub struct GlobalLogger {
    path: Option<PathBuf>,
    verbosity: Verbosity,
}

impl GlobalLogger {
    /// NOTE: Relative paths will not be resolved until apply() is called.
    pub fn path<P: AsRef<Path>>(&mut self, path: P) -> &mut Self
    { self.path = Some(path.as_ref().to_owned()); self }

    /// Any integer will be accepted; the level will be truncated
    /// to the most extreme value supported.
    pub fn verbosity(&mut self, level: i32) -> &mut Self
    {
        self.verbosity = match level > 0 {
            true => Verbosity::Loud,
            false => Verbosity::Default,
        };
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Verbosity { Default, Loud }

impl Default for Verbosity {
    fn default() -> Self { Verbosity::Default }
}

impl GlobalLogger {
    /// NOTE: I'm not sure what happens (or don't particularly care)
    ///       if this is called multiple times. It won't be UB, but
    ///       it probably also won't make sense.
    pub fn apply(&mut self) -> Result<()>
    {ok({
        use ::std::time::Instant;
        use ::log::LogLevelFilter as LevelFilter;

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
            .level_for("rsp2_minimize::exact_ls", match self.verbosity {
                Verbosity::Default => LevelFilter::Debug,
                Verbosity::Loud => LevelFilter::Trace,
            })
            .chain(::std::io::stdout());

        if let Some(path) = self.path.as_ref() {
            fern = fern.chain(::fern::log_file(path)?);
        }

        fern.apply()?;
    })}
}

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
