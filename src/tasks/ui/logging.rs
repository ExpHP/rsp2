use ::FailResult;
use ::std::fmt;
use ::std::time;
use ::log::{LogLevel, LogRecord};
use ::fern::FernLog;
use ::path_abs::{FileWrite, PathFile};

pub use self::fern::{CapturableStderr, DelayedLogFile, GLOBAL_LOGFILE};
mod fern {
    use super::*;
    use ::std::sync::RwLock;
    use ::std::io::prelude::*;

    /// Fern logger that uses `eprintln!`.
    ///
    /// This is NOT the same as using `::std::io::stderr()`, because that would
    /// not get captured properly by the unit test harness.
    pub struct CapturableStderr;

    impl FernLog for CapturableStderr {
        fn log_args(&self, payload: &fmt::Arguments, _original: &LogRecord) {
            eprintln!("{}", payload);
        }
    }

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
        pub fn start(&self, path: PathFile) -> FailResult<()> {
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
enum Verbosity { Default, Loud, MyEarsHurt }

impl Default for Verbosity {
    fn default() -> Self { Verbosity::Default }
}

impl Verbosity {
    /// Any integer will be accepted; the level will be truncated
    /// to the most extreme value supported.
    fn from_int(level: i32) -> Self
    { match level {
        level if level < 1 => Verbosity::Default,
        1 => Verbosity::Loud,
        _ => Verbosity::MyEarsHurt,
    }}

    pub fn from_env() -> FailResult<Self>
    { ::env::verbosity().map(Self::from_int) }
}

/// Set the global logger, enabling the use of `log!()` macros.
/// This can be done at any time, but it can only be done once.
///
/// It returns an object for setting up `GLOBAL_LOGFILE`, leveraging the
/// "unused variable" lint to help remind you to do this once possible.
pub fn init_global_logger() -> FailResult<SetGlobalLogfile>
{
    use ::log::LogLevelFilter as L;
    use self::Verbosity as V;

    let verbosity = V::from_env()?;
    let log_mod_setting = ::env::log_mod()?;

    let start = time::Instant::now();
    let mut fern = ::fern::Dispatch::new();
    fern =
        fern.format(move |out, message, record| {
            let message = fmt_log_message_lines(message, record, start.elapsed(), log_mod_setting);

            out.finish(format_args!("{}", message))
        })
        .level(L::Debug)
        .level_for("rsp2_tasks", L::Trace)
        .level_for("rsp2_minimize", L::Trace)
        .level_for("rsp2_phonopy_io", L::Trace)
        .level_for("rsp2_minimize::hager_ls", match verbosity {
            V::Default => L::Debug,
            V::Loud |
            V::MyEarsHurt => L::Trace,
        })
        .level_for("rsp2_minimize::exact_ls", match verbosity {
            V::Default |
            V::Loud => L::Debug,
            V::MyEarsHurt => L::Trace,
        })
        // Yes, this really is deliberately boxing a reference (a 'static one).
        // The reason is simply because chain asks for a Box.
        .chain(Box::new(&*GLOBAL_LOGFILE) as Box<FernLog>)
        .chain(Box::new(CapturableStderr) as Box<FernLog>);

    fern.apply()?;

    Ok(SetGlobalLogfile(()))
}

/// Called by tests to have log output written to the captured stderr, for easier debugging.
///
/// Be aware that once one test function calls this, it remains active for the rest.
/// As a result, it may *appear* that you do not need to call this to see log output
/// (but you should!).
///
/// Tests do not rely on the specific logging framework and output format used by rsp2.
/// At least, they shouldn't.  (As a rule of thumb, a test should remain just as effective
/// if all calls to this function were removed.)
#[cfg(test)]
pub fn init_test_logger() {
    let _ = init_global_logger();
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
        write!(f, "[{:>5}]", ::ui::color::gpaint(style, self.0))
    }
}

/// Returned by `init_global_logger` to remind you to set the logfile once possible.
/// (which can be done by calling the `start` method)
#[must_use = "The logfile has not been set up!"]
pub struct SetGlobalLogfile(());
impl SetGlobalLogfile {
    pub fn start(self, path: PathFile) -> FailResult<()>
    { GLOBAL_LOGFILE.start(path).map_err(Into::into) }
}

fn fmt_log_message_lines(
    message: &fmt::Arguments,
    record: &LogRecord,
    elapsed: time::Duration,
    log_mod_setting: bool,
) -> String {
    // (yes, I know, we're ruining the entire point of fmt::Arguments by printing
    //  it to a String, boo hoo.  We want to inspect its contents.)
    let buf = message.to_string();
    let mut lines = buf.lines();

    // Break into lines, format the first, and pad the rest.
    let first = lines.next().unwrap_or("");
    let mut out = vec![format!(
        "[{:>4}.{:03}s]{}{} {}",
        elapsed.as_secs(),
        elapsed.subsec_nanos() / 1_000_000,
        match log_mod_setting {
            true => format!("[{}]", record.target()),
            false => format!(""),
        },
        ColorizedLevel(record.level()),
        first,
    )];

    let len_secs = format!("{:4}", elapsed.as_secs()).len();
    let len_target = match log_mod_setting {
        true => record.target().len() + 2,
        false => 0,
    };
    out.extend(lines.map(|line| {
        format!(
            // [############.####s][############][####]
            "\n {:len_secs$} {:3}  {:len_target$} {:5}| {}",
            "", "", "", "", line,
            len_secs=len_secs,
            len_target=len_target,
        )
    }));
    out.concat()
}
