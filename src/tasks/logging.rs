use ::Result;

use ::std::fmt;
use ::std::path::Path;
use ::log::LogLevel;

pub(crate) fn setup_global_logger(path: Option<&AsRef<Path>>) -> Result<()>
{Ok({
    use ::std::time::Instant;

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
        .level(::log::LogLevelFilter::Debug)
        .level_for("rsp2_tasks", ::log::LogLevelFilter::Trace)
        .level_for("rsp2_minimize", ::log::LogLevelFilter::Trace)
        .level_for("rsp2_phonopy_io", ::log::LogLevelFilter::Trace)
        .level_for("rsp2_minimize::exact_ls", ::log::LogLevelFilter::Debug)
        .chain(::std::io::stdout());

    if let Some(path) = path {
        fern = fern.chain(::fern::log_file(path)?);
    }

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
        write!(f, "{}", ::color::gpaint(style, self.0))
    }
}
