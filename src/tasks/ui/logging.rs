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

use ::FailResult;
use ::std::fmt;
use ::std::time;
use ::log::{Log, Level, Record};
use ::path_abs::{FileWrite, PathFile};

pub use self::fern::{CapturableStderr, DelayedLogFile, GLOBAL_LOGFILE};
mod fern {
    use super::*;
    use ::std::sync::RwLock;
    use ::std::io::prelude::*;

    /// Logger that uses `eprintln!`.
    ///
    /// This is NOT the same as handing `::std::io::stderr()` to `fern::Dispatch::chain`,
    /// because that would not get captured properly by the unit test harness.
    pub struct CapturableStderr;

    impl Log for CapturableStderr {
        fn enabled(&self, _: &::log::Metadata) -> bool { true }
        fn log(&self, record: &Record) { eprintln!("{}", record.args()); }
        fn flush(&self) {}
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
                // PoisonError. In the highly unlikely event that this occurs, something else will
                // probably catch it. And if we try to be a hero, we risk a double-panic,
                // especially since Drop impls may use the logging facilities.
            }
            Ok(())
        }
    }

    impl Log for DelayedLogFile {
        fn enabled(&self, _: &::log::Metadata) -> bool { true }

        fn log(&self, record: &Record) {
            if let Ok(mut file) = self.file_rw.write() {
                if let Some(mut file) = (*file).as_mut() {
                    // ignore error for same reasons as ignoring PoisonError (see above).
                    //
                    // (`Display::fmt` might also double-panic, but that's the Drop impl's fault
                    //  for trying to format something that can panic!)
                    let _ = writeln!(file, "{}", record.args());
                }
            } // ignore PoisonError silently for reasons documented above
        }

        fn flush(&self) {
            if let Ok(mut file) = self.file_rw.write() {
                if let Some(mut file) = (*file).as_mut() {
                    // ignore error for same reasons as ignoring PoisonError (see above)
                    let _ = file.flush();
                }
            } // ignore PoisonError silently for reasons documented above
        }
    }
}

/// Set the global logger, enabling the use of `log!()` macros.
/// This can be done at any time, but it can only be done once.
///
/// It returns an object for setting up `GLOBAL_LOGFILE`, leveraging the
/// "unused variable" lint to help remind you to do this once possible.
pub fn init_global_logger() -> FailResult<SetGlobalLogfile>
{
    let log_mod_setting = ::env::log_mod()?;

    // NOTE
    // It might seem silly that we are setting up the initial logging filter by
    // tweaking the env_logger filter rather than calling the `level` and `level_for`
    // methods on fern::Dispatch.  The reason is because fern's filter uses AND logic
    // (a message must successfully pass through ALL filters individually), but I want
    // RUST_LOG to be capable of *increasing* the verbosity.
    //
    // env_logger::filter does not expose a list of targets and level filters, so this
    // is our only option at the moment.
    //
    // FIXME ...at this point, it seems to me that we are no longer deriving any
    //       benefit at all from using fern, and thus might as well just write a
    //       single implementor of Log.  That's a yak to be shaved another time...
    let env_filter = {
        ::env_logger::filter::Builder::new()
            .parse("debug")
            .parse("rsp2_tasks=trace")
            .parse("rsp2_minimize=trace")
            .parse("rsp2_phonopy_io=trace")
            .parse("rsp2_minimize::hager_ls=debug")
            .parse("rsp2_minimize::exact_ls=debug")
            .parse(&::env::rust_log()?)
            .build()
    };

    let start = time::Instant::now();
    let fern = ::fern::Dispatch::new()
        .format(move |out, message, record| {
            let message = fmt_log_message_lines(message, record, start.elapsed(), log_mod_setting);

            out.finish(format_args!("{}", message))
        })
        .filter(move |metadata| env_filter.enabled(metadata))
        .chain(&*GLOBAL_LOGFILE as &Log)
        .chain(Box::new(CapturableStderr) as Box<Log>)
        ;

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
pub struct ColorizedLevel(pub Level);
impl fmt::Display for ColorizedLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let style = match self.0 {
            Level::Error => ::ansi_term::Colour::Red.bold(),
            Level::Warn  => ::ansi_term::Colour::Red.normal(),
            Level::Info  => ::ansi_term::Colour::Cyan.bold(),
            Level::Debug => ::ansi_term::Colour::Yellow.dimmed(),
            Level::Trace => ::ansi_term::Colour::Cyan.normal(),
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
    record: &Record,
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
