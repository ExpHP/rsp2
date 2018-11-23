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

#[macro_use]
extern crate clap;
extern crate atty;
extern crate ordered_float;
#[macro_use]
extern crate rsp2_clap;
use rsp2_clap::arg;
use ordered_float::NotNan;

use std::{
    cmp::Ordering,
    path::Path,
    fs::File,
    io::{self, BufReader, BufRead},
};

fn main() -> io::Result<()> {
    let app = {
        app_from_crate!()
            .subcommand({
                clap::SubCommand::with_name("cat")
                    .arg(arg!(?input=INPUT... "input files"))
                    .help("\
                        extract the debug lines from potential files, concatenate them, \
                        and sort them. The sort ordering is a form of natural sort; \
                        numerical values surrounded by whitespace (or line boundaries) will \
                        be sorted by numerical value. +0.0 and -0.0 will obey the IEEE-754 \
                        standard of comparing equal, however NaNs will take one of the extreme \
                        positions in the ordering so that they are all collected in one location.\
                    ")
            })
            .subcommand({
                clap::SubCommand::with_name("uniq")
                    .arg(arg!(?input=INPUT "input file"))
                    .help("\
                        Filters and sorts the input like the `cat` subcommand, then removes \
                        duplicates.\
                    ")
            })

            // NOTE: This cannot be achieved by trying to compose our `sort` with UNIX diff, sadly,
            //       the issue being that `diff` may show the same line as both "added" and
            //       "removed" (from different locations) if that produces a smaller diff.
            //       It also doesn't know of our comparison scheme.
            .subcommand({
                clap::SubCommand::with_name("diff")
                    .arg(arg!( file_a=FILEA "input a"))
                    .arg(arg!( file_b=FILEB "input b"))
                    .arg(arg!( equal [-e][--equal] "show equal lines"))
                    .arg({
                        arg!( color [--color]=COLORMODE "when to show color")
                            .possible_values(&["auto", "always", "never"])
                    })
                    .help("\
                        Performs a multiset diff on the debug lines from two output files. \
                        Lines are written in natural sort order and appear once for each individual \
                        occurrence in the file.\
                    ")
            })
    };
    let matches = app.get_matches();

    if let Some(ref matches) = matches.subcommand_matches("cat") {
        let files = match matches.values_of("input") {
            Some(paths) => {
                paths.map(box_open).collect::<io::Result<Vec<_>>>()?
            },
            None => {
                if atty::is(atty::Stream::Stdin) {
                    return Err(io::Error::new(io::ErrorKind::Other, "No input provided"));
                }
                vec![Box::new(io::stdin()) as Box<dyn io::Read>]
            },
        };

        cmd_cat(files)

    } else if let Some(ref matches) = matches.subcommand_matches("uniq") {
        let file = match matches.value_of("input") {
            Some(path) => box_open(path)?,
            None => {
                if atty::is(atty::Stream::Stdin) {
                    return Err(io::Error::new(io::ErrorKind::Other, "No input provided"));
                }
                Box::new(io::stdin())
            },
        };

        cmd_uniq(file)

    } else if let Some(ref matches) = matches.subcommand_matches("diff") {
        let file_a = box_open(matches.value_of("file_a").unwrap())?;
        let file_b = box_open(matches.value_of("file_b").unwrap())?;
        let equal = matches.is_present("equal");
        let color = ColorMode::parse(matches.value_of("color").unwrap_or("auto")).unwrap();
        cmd_diff(file_a, file_b, equal, color)
    } else {
        unreachable!("missing branch for subcommand")
    }
}

//--------------------------------------------------------------------

fn cmd_cat(files: Vec<Box<dyn io::Read>>) -> io::Result<()> {
    let mut all_line_strings = vec![];
    for file in files {
        all_line_strings.extend(read_file_filtered(file)?);
    }
    let all_line_strings = all_line_strings;

    let mut all_lines: Vec<_> = all_line_strings.iter().map(|s| Line::parse(s)).collect();
    all_lines.sort_by(Line::cmp_function);

    for line in all_lines {
        println!("{}", line.original_text);
    }
    Ok(())
}

fn cmd_uniq(file: Box<dyn io::Read>) -> io::Result<()> {
    let all_line_strings = read_file_filtered(file)?;

    let mut all_lines: Vec<_> = all_line_strings.iter().map(|s| Line::parse(s)).collect();
    all_lines.sort_by(Line::cmp_function);

    all_lines.dedup_by(|a, b| Line::cmp_function(a, b) == Ordering::Equal);

    for line in all_lines {
        println!("{}", line.original_text);
    }
    Ok(())
}

fn cmd_diff(
    file_a: Box<dyn io::Read>,
    file_b: Box<dyn io::Read>,
    show_equal: bool,
    color_mode: ColorMode,
) -> io::Result<()> {
    let all_line_strings_a = read_file_filtered(file_a)?;
    let all_line_strings_b = read_file_filtered(file_b)?;

    let mut all_lines_a: Vec<_> = all_line_strings_a.iter().map(|s| Line::parse(s)).collect();
    let mut all_lines_b: Vec<_> = all_line_strings_b.iter().map(|s| Line::parse(s)).collect();
    all_lines_a.sort_by(Line::cmp_function);
    all_lines_b.sort_by(Line::cmp_function);

    let stylize = color_mode.stdout_diff_styler();
    sorted_diff_each_by(
        all_lines_a, all_lines_b,
        Line::cmp_function,
        |side, line| {
            if (side, show_equal) == (Ordering::Equal, false) {
                return;
            }
            stylize(side, line.original_text);
        },
    );
    Ok(())
}

//--------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ColorMode { Never, Auto, Always }

const ANSI_FG_RED: &'static str = "\x1b[31m";
const ANSI_FG_GREEN: &'static str = "\x1b[32m";
const ANSI_SGR0: &'static str = "\x1b[0m";
impl ColorMode {
    fn parse(s: &str) -> Option<ColorMode> {
        match s {
            "auto" => Some(ColorMode::Auto),
            "always" => Some(ColorMode::Always),
            "never" => Some(ColorMode::Never),
            _ => None,
        }
    }

    fn stdout_diff_styler(&self) -> (fn(Ordering, &str)) {
        match self {
            ColorMode::Never => |c, s| match c {
                Ordering::Less => println!("< {}", s),
                Ordering::Greater => println!("> {}", s),
                Ordering::Equal => println!("= {}", s),
            },
            ColorMode::Always => |c, s| match c {
                Ordering::Less => println!("{}< {}{}", ANSI_FG_RED, s, ANSI_SGR0),
                Ordering::Greater => println!("{}> {}{}", ANSI_FG_GREEN, s, ANSI_SGR0),
                Ordering::Equal => println!("= {}", s),
            },
            ColorMode::Auto => match atty::is(atty::Stream::Stdout) {
                true => ColorMode::Always.stdout_diff_styler(),
                false => ColorMode::Never.stdout_diff_styler(),
            }
        }
    }
}

//--------------------------------------------------------------------

/// Has an Ord impl useful for natural ordering
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Token<'a> {
    Number(Option<NotNan<f64>>),
    Str(&'a str),
}

impl<'a> Token<'a> {
    fn parse(s: &'a str) -> Token<'a> {
        if let Ok(x) = s.parse::<f64>() {
            Token::Number(NotNan::new(x).ok())
        } else {
            Token::Str(s)
        }
    }
}

#[derive(Debug, Clone)]
struct Line<'a> {
    tokens: Vec<Token<'a>>,
    original_text: &'a str,
}

impl<'a> Line<'a> {
    pub fn parse(line: &'a str) -> Line<'a> {
        Line {
            tokens: line.split_whitespace().map(Token::parse).collect(),
            original_text: line,
        }
    }

    // For use in sort_by etc.
    pub fn cmp_function(a: &Self, b: &Self) -> Ordering { a.tokens.cmp(&b.tokens) }
}

//--------------------------------------------------------------------

fn box_open(path: impl AsRef<Path>) -> io::Result<Box<dyn io::Read>> {
    File::open(path).map(|x| Box::new(x) as _)
}

fn read_file_filtered(file: impl io::Read) -> io::Result<Vec<String>> {
    let mut out = vec![];
    for line in BufReader::new(file).lines() {
        let line = line?;

        // Must be flush against the margin
        if let Some(c) = line.chars().next() {
            if c.is_ascii_whitespace() {
                continue;
            }
        }

        // First word must end with ":"
        match line.split_whitespace().next() {
            None => continue, // No words!
            Some(first_word) => {
                if !first_word.ends_with(":") {
                    continue;
                }

                if PREFIX_BLACKLIST.contains(&first_word) {
                    continue;
                }
            },
        };
        out.push(line)
    }
    Ok(out)
}

const PREFIX_BLACKLIST: &'static [&'static str] = &[
    // lammps
    "Nlocal:",
    "Histogram:",
    "Nghost:",
    "Histogram:",
    "Neighs:",
    "Histogram:",
    "FullNghs:",
    "Histogram:",
    "WARNING:",
    // rust
    "note:",
    "right:",
    "failures:",
];

//--------------------------------------------------------------------

// Assuming `a` and `b` are sorted according to `cmp`, calls `emit` on each line
// from one of the inputs with an `Ordering` that indicates whether the line is only
// in `a` (`Less`), only in `b` (`Greater`), or in both (`Equal`).
//
// `emit` will be called at least `max(a.len(), b.len())` times (when one input is a
// subset of the other), and at most `a.len() + b.len()` times (when the inputs are
// disjoint).
fn sorted_diff_each_by<T>(
    a: impl IntoIterator<Item=T>,
    b: impl IntoIterator<Item=T>,
    mut cmp: impl FnMut(&T, &T) -> Ordering,
    mut emit: impl FnMut(Ordering, T),
) {
    let mut iter_a = a.into_iter().fuse();
    let mut iter_b = b.into_iter().fuse();

    let mut next_a = iter_a.next();
    let mut next_b = iter_b.next();
    while let (Some(_), Some(_)) = (&next_a, &next_b) {
        let a = next_a.unwrap();
        let b = next_b.unwrap();
        match cmp(&a, &b) {
            side@Ordering::Less => {
                emit(side, a);
                next_a = iter_a.next();
                next_b = Some(b);
            },
            side@Ordering::Equal => {
                emit(side, a);
                next_a = iter_a.next();
                next_b = iter_b.next();
            },
            side@Ordering::Greater => {
                emit(side, b);
                next_a = Some(a);
                next_b = iter_b.next();
            },
        }
    }

    for x in next_a.into_iter().chain(iter_a) {
        emit(Ordering::Less, x);
    }
    for x in next_b.into_iter().chain(iter_b) {
        emit(Ordering::Greater, x);
    }
}
