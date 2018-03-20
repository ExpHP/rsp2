use ::errors::{Result, ok};

use ::std::fmt;
use ::std::time;
use ::log::{LogLevel, LogRecord};
use ::fern::FernLog;
use ::path_abs::{FileWrite, PathFile};

pub use self::fern::{DelayedLogFile, GLOBAL_LOGFILE};
mod fern {
    use super::*;
    use ::std::sync::RwLock;
    use ::std::io::prelude::*;

    /// A log file for fern that can be created *after* logger initialization.
    #[derive(Debug, Default)]
    pub struct DelayedLogFile {
        file_rw: RwLock<Option<FileWrite>>,
    }

    lazy_static! {
        /// The DelayedLogFile given to fern.
        pub static ref GLOBAL_LOGFILE: DelayedLogFile = Default::default();
    }

    impl DelayedLogFile {
        pub fn start(&self, path: PathFile) -> Result<()> {
            if let Ok(mut file) = self.file_rw.write() {
                if file.is_some() {
                    bail!("The logfile has already been set!");
                }
                *file = Some(path.append()?);
            } else {
                // PoisonError. In the highly unlikely event this occurs, something else
                // will probably catch it. If we try to be a hero, we risk a double-panic.
            }
            Ok(())
        }
    }

    impl FernLog for &'static DelayedLogFile {
        fn log_args(&self, payload: &fmt::Arguments, _original: &LogRecord) {
            if let Ok(mut file) = self.file_rw.write() {
                if let Some(mut file) = (*file).as_mut() {
                    let _ = writeln!(file, "{}", payload);
                }
            } else {
                // PoisonError. In the highly unlikely event this occurs, something else
                // will probably catch it. If we try to be a hero, we risk a double-panic.
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
/// This can be done at any time, but it can only be done once.
///
/// It returns an object for setting up `GLOBAL_LOGFILE`, leveraging the
/// "unused variable" lint to help remind you to do this once possible.
pub fn init_global_logger() -> Result<SetGlobalLogfile>
{ok({
    use ::log::LogLevelFilter as LevelFilter;

    let verbosity = Verbosity::from_env()?;

    let start = time::Instant::now();
    let mut fern = ::fern::Dispatch::new();
    fern =
        fern.format(move |out, message, record| {
            let message = fmt_log_message_lines(message, record, start.elapsed());

            out.finish(format_args!("{}", message))
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

    SetGlobalLogfile(())
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
        write!(f, "[{:^5}]", ::ui::color::gpaint(style, self.0))
    }
}

/// Returned by `init_global_logger` to remind you to set the logfile once possible.
/// (which can be done by calling the `start` method)
#[must_use = "The logfile has not been set up!"]
pub struct SetGlobalLogfile(());
impl SetGlobalLogfile {
    pub fn start(self, path: PathFile) -> Result<()>
    { GLOBAL_LOGFILE.start(path) }
}

fn fmt_log_message_lines(
    message: &fmt::Arguments,
    record: &LogRecord,
    elapsed: time::Duration,
) -> String {
    // (yes, I know, we're ruining the entire point of fmt::Arguments by printing
    //  it to a String, boo hoo.  We want to inspect its contents.)
    let buf = message.to_string();
    let mut lines = buf.lines();

    // Break into lines, format the first, and pad the rest.
    let first = lines.next().unwrap_or("");
    let mut out = vec![format!(
        "[{:>4}.{:03}s][{}]{} {}",
        elapsed.as_secs(),
        elapsed.subsec_nanos() / 1_000_000,
        record.target(),
        ColorizedLevel(record.level()),
        first,
    )];

    let len_secs = format!("{:4}", elapsed.as_secs()).len();
    let len_target = record.target().len();
    let len_level = record.level().to_string().len();
    out.extend(lines.map(|line| {
        format!(
            "\n {:len_secs$} {:3}   {:len_target$}  {:6}| {}",
            "", "", "", "", line,
            len_secs=len_secs,
            len_target=len_target,
        )
    }));
    out.concat()
}
