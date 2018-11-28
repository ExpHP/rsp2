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

use crate::FailResult;
use std::fmt;
use std::time;
use log::{Log, Record};
use path_abs::{FileWrite, PathFile};
use fern::colors;

pub use self::loggers::{CapturableStderr, DelayedLogFile, GLOBAL_LOGFILE};
mod loggers {
    use super::*;
    use std::sync::RwLock;
    use std::io::prelude::*;

    /// Logger that uses `eprintln!`.
    ///
    /// This is NOT the same as handing `::std::io::stderr()` to `fern::Dispatch::chain`,
    /// because that would not get captured properly by the unit test harness.
    pub struct CapturableStderr;

    impl Log for CapturableStderr {
        fn enabled(&self, _: &::log::Metadata<'_>) -> bool { true }
        fn log(&self, record: &Record<'_>) { eprintln!("{}", record.args()); }
        fn flush(&self) {}
    }

    /// A log file for fern that can be created *after* logger initialization.
    #[derive(Debug)]
    pub struct DelayedLogFile {
        rw: RwLock<DelayedLogFileInner>,
    }

    // States:
    //                 file     delayed_messages
    // - initial:      None        Some(...)
    // - started:    Some(...)       None
    // - disabled:     None          None
    #[derive(Debug)]
    struct DelayedLogFileInner {
        file: Option<FileWrite>,

        // messages logged via `log` macros before the logfile was initialized,
        // so that they can be written to it once it is open.
        delayed_messages: Option<Vec<String>>,
    }

    lazy_static! {
        /// The DelayedLogFile given to fern.
        pub static ref GLOBAL_LOGFILE: DelayedLogFile = DelayedLogFile {
            rw: RwLock::new(DelayedLogFileInner {
                file: None,
                delayed_messages: Some(vec![]),
            })
        };
    }

    impl DelayedLogFile {
        pub(in crate::ui::logging) fn start(&self, path: PathFile) -> FailResult<()> {
            if let Ok(mut inner) = self.rw.write() {
                // This situation should be impossible since SetGlobalLogFile (the only API
                // visible outside the `ui` module) has methods that take `self`.
                assert!(inner.delayed_messages.is_some(), "(bug) The logfile was set twice!");
                assert!(inner.file.is_none(), "(bug) impossible state!");

                let mut file = path.append()?;
                for line in inner.delayed_messages.take().unwrap() {
                    writeln!(file, "{}", line)?;
                }

                inner.file = Some(file);
            } else {
                // PoisonError. In the highly unlikely event that this occurs, something else will
                // probably catch it. And if we try to be a hero, we risk a double-panic,
                // especially since Drop impls may use the logging facilities.
            }
            Ok(())
        }

        pub(in crate::ui::logging) fn disable(&self) {
            if let Ok(mut inner) = self.rw.write() {
                // This situation should be impossible since SetGlobalLogFile (the only API
                // visible outside the `ui` module) has methods that take `self`.
                assert!(inner.delayed_messages.is_some(), "(bug) The logfile was set twice!");
                assert!(inner.file.is_none(), "(bug) impossible state!");

                // stop recording messages
                inner.delayed_messages.take();
            } // ignore PoisonError silently for reasons documented above
        }
    }

    impl Log for DelayedLogFile {
        fn enabled(&self, _: &::log::Metadata<'_>) -> bool { true }

        fn log(&self, record: &Record<'_>) {
            if let Ok(mut inner) = self.rw.write() {
                let inner = &mut *inner;
                match (inner.file.as_mut(), inner.delayed_messages.as_mut()) {
                    (Some(file), _) => {
                        // ignore error for same reasons as ignoring PoisonError (see above).
                        //
                        // (`Display::fmt` might also double-panic, but that's the Drop impl's fault
                        //  for trying to format something that can panic!)
                        let _ = writeln!(file, "{}", record.args());
                    },
                    // not yet started
                    (None, Some(delayed_messages)) => {
                        delayed_messages.push(format!("{}", record.args()));
                    },
                    // disabled
                    (None, None) => {}
                }
            } // ignore PoisonError silently for reasons documented above
        }

        fn flush(&self) {
            if let Ok(mut inner) = self.rw.write() {
                if let Some(file) = inner.file.as_mut() {
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
    let log_mod_setting = crate::env::log_mod()?;

    let colors = colors::ColoredLevelConfig {
        error: colors::Color::BrightRed,
        warn:  colors::Color::Yellow, // non-bright yellow is orange in many color schemes
        info:  colors::Color::BrightCyan,
        debug: colors::Color::BrightYellow,
        trace: colors::Color::Cyan,
    };

    // NOTE
    // It might seem silly that we are setting up the initial logging filter by
    // tweaking the env_logger filter rather than calling the `level` and `level_for`
    // methods on fern::Dispatch.  The reason is because fern's filter uses AND logic
    // (a message must successfully pass through ALL filters individually), but I want
    // RUST_LOG to be capable of *increasing* the verbosity.
    //
    // env_logger::filter does not expose a list of targets and level filters, so this
    // is our only option at the moment.
    let env_filter = {
        env_logger::filter::Builder::new()
            .parse("debug")
            .parse("rsp2_tasks=trace")
            .parse("rsp2_tasks::ev_analysis=debug")
            .parse("rsp2_minimize=trace")
            .parse("rsp2_phonopy_io=trace")
            .parse("rsp2_minimize::hager_ls=debug")
            .parse("rsp2_minimize::exact_ls=debug")
            // not an actual module. Some debug output files are generated if you enable `trace` for
            // specific paths under this. (don't set the whole module to `trace`; you will quickly
            // fill your hard drive!)
            .parse("rsp2_tasks::special=info")
            .parse(&crate::env::rust_log()?)
            .build()
    };

    let start = time::Instant::now();
    let fern = fern::Dispatch::new()
        .format(move |out, message, record| {
            let message = fmt_log_message_lines(message, &colors, record, start.elapsed(), log_mod_setting);

            out.finish(format_args!("{}", message))
        })
        .filter(move |metadata| env_filter.enabled(metadata))
        .chain(&*GLOBAL_LOGFILE as &dyn Log)
        .chain(Box::new(CapturableStderr) as Box<dyn Log>)
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

/// Returned by `init_global_logger` to remind you to set the logfile once possible,
/// or to disable its logging. (which can be done by calling the `start` or `disable` method)
pub struct SetGlobalLogfile(());
impl SetGlobalLogfile {
    pub fn start(self, path: PathFile) -> FailResult<()> {
        let result = GLOBAL_LOGFILE.start(path).map_err(Into::into);
        std::mem::forget(self);
        result
    }

    pub fn disable(self) {
        GLOBAL_LOGFILE.disable();
        std::mem::forget(self);
    }
}

impl Drop for SetGlobalLogfile {
    // NOTE: this should never be called in successful calls to the program, as the entry point
    //       ought to call either 'start' or 'disable' (or else risk unbounded memory usage due
    //       to `delayed_messages`).
    //
    //       ...however, we can't just make this method complain, because there are many ways that
    //       an entry point could fail before it ever even reaches the call to 'start'/'disable'.
    fn drop(&mut self)
    { GLOBAL_LOGFILE.disable() }
}

fn fmt_log_message_lines(
    message: &fmt::Arguments<'_>,
    colors: &::fern::colors::ColoredLevelConfig,
    record: &Record<'_>,
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
        "[{:>4}.{:03}s]{}[{:>5}] {}",
        elapsed.as_secs(),
        elapsed.subsec_nanos() / 1_000_000,
        match log_mod_setting {
            true => format!("[{}]", record.target()),
            false => format!(""),
        },
        colors.color(record.level()),
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
