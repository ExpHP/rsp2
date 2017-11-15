use ::Result;
use ::std::ffi::OsStr;
use ::std::process::{Command, Stdio};
use ::std::io::Read;

/// Move a file or directory, possibly across filesystems.
///
/// Properties:
/// * Moves files or folders alike.
/// * The destination must not exist.
/// * The destination can be on a different filesystem.
/// * The operation is O(1) and preserves hard-links if
///   the destination is on the same filesystem.
/// * Symlinks retain their textual path targets.
/// * Permissions and attributes are carried over on a best
///   effort basis.
/// * aaaah why am I even documenting this
///
/// Alright, look. It calls `mv -T`, okay?
/// *It literally calls `mv -T`.*
///
/// (who ever knew so much complexity was secretly hiding in
///  /usr/bin/mv?)
pub fn mv<P, Q>(src: P, dest: Q) -> Result<()>
where
    P: AsRef<OsStr>,
    Q: AsRef<OsStr>,
{ Move::new().one(src, dest) }

/// Calls `cp -aT`.
pub fn cp_a<P, Q>(src: P, dest: Q) -> Result<()>
where
    P: AsRef<OsStr>,
    Q: AsRef<OsStr>,
{ Copy::new().one(src, dest) }

/// Builder for configuring a `mv` command.
#[derive(Debug, Clone)]
pub struct Move(MoveCmd);

/// Builder for configuring a `cp -a` command.
#[derive(Debug, Clone)]
pub struct Copy(MoveCmd);

impl Move {
    pub fn new() -> Self
    { Move(MoveCmd::new_move()) }
}

impl Copy {
    pub fn new() -> Self
    { Copy(MoveCmd::new_copy()) }
}

impl Default for Move {
    fn default() -> Self
    { Self::new() }
}

impl Default for Copy {
    fn default() -> Self
    { Self::new() }
}

macro_rules! impl_move_cmd_wrapper {
    ($Type:ty)
    => {
        impl $Type {
            /// Allow existing files to be replaced.
            ///
            /// Default: true.
            pub fn clobber(&mut self, value: bool) -> &mut Self
            { self.0.clobber = value; self }

            /// When the destination exists and is a directory,
            /// `one` will move *into* the directory instead.
            ///
            /// The default setting is false, which corresponds to the
            /// `-T`/`--no-target-directory` option.
            pub fn magic_directory(&mut self, value: bool) -> &mut Self
            { self.0.magic_directory = value; self }

            /// `mv src dest`
            pub fn one<P, Q>(&self, src: P, dest: Q) -> Result<()>
            where
                P: AsRef<OsStr>,
                Q: AsRef<OsStr>,
            { self.0.one(src, dest) }

            /// `mv -t dest src1 [src2...]`
            pub fn many<P, Q, Qs>(&self, target: P, sources: Qs) -> Result<()>
            where
                P: AsRef<OsStr>,
                Q: AsRef<OsStr>,
                Qs: IntoIterator<Item=Q>,
            { self.0.many(target, sources) }
        }
    };
}

impl Copy {
    /// Try to create hard links instead of copying, when possible.
    ///
    /// The default setting is false.  Enabling it corresponds to
    /// the "--link" argument.
    pub fn link(&mut self, value: bool) -> &mut Self
    { self.0.link = value; self }

    /// Copy the referent of links rather than the links themselves.
    ///
    /// The default setting is false.  Enabling it corresponds to
    /// the "--dereference" argument.
    pub fn follow_links(&mut self, value: bool) -> &mut Self
    { self.0.dereference = value; self }
}

impl_move_cmd_wrapper!{ Move }
impl_move_cmd_wrapper!{ Copy }

// Calling 'cp -a' and calling 'mv' are such similar tasks that
// it is easy enough to just write one implementation that does both.
#[derive(Debug, Clone)]
struct MoveCmd {
    // is this `cp -a` or `mv`?
    copy: bool,
    // enables "--link" argument (cp only)
    link: bool,
    // enables "--dereference" argument (cp only)
    dereference: bool,
    // disables "--no-clobber" argument
    clobber: bool,
    // disables "--no-target-directory" argument in `one`
    magic_directory: bool,
}

impl MoveCmd {
    fn new_move() -> Self
    { MoveCmd {
        copy: false,
        link: false,
        dereference: false,
        clobber: true,
        magic_directory: false,
    }}

    fn new_copy() -> Self
    { MoveCmd {
        copy: true,
        ..Self::new_move()
    }}

    // args that do not depend on whether we are calling "one" or "many"
    fn base_cmd(&self) -> Command
    {
        // args specific to copy vs move, but valid for both "one and many"
        let mut cmd = match self.copy {
            true => {
                let mut cmd = Command::new("/usr/bin/cp");
                cmd.arg("--archive");

                if self.link { cmd.arg("--link"); }
                if self.dereference { cmd.arg("--dereference"); }
                cmd
            },
            false => {
                let cmd = Command::new("/usr/bin/mv");

                // no API should have been provided to set these
                assert!(!self.link);
                assert!(!self.dereference);
                cmd
            },
        };

        // args valid for both copy and move, and both one and many
        if !self.clobber { cmd.arg("--no-clobber"); }
        cmd
    }

    fn one<P, Q>(&self, src: P, dest: Q) -> Result<()>
    where
        P: AsRef<OsStr>,
        Q: AsRef<OsStr>,
    {Ok({
        let mut cmd = self.base_cmd();

        if !self.magic_directory { cmd.arg("--no-target-directory"); }

        cmd.arg("--").arg(src).arg(dest);

        Self::call_and_check(cmd)?;
    })}

    fn many<P, Q, Qs>(&self, target: P, sources: Qs) -> Result<()>
    where
        P: AsRef<OsStr>,
        Q: AsRef<OsStr>,
        Qs: IntoIterator<Item=Q>,
    {Ok({
        let mut cmd = self.base_cmd();
        cmd.arg("-t").arg(target).arg("--").args(sources);

        Self::call_and_check(cmd)?;
    })}

    fn call_and_check(mut cmd: Command) -> Result<()>
    {Ok({
        let mut child = cmd
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;

        let stderr = {
            let mut stderr = child.stderr.take().unwrap();
            let mut s = String::new();
            stderr.read_to_string(&mut s)?;
            s
        };

        let code = child.wait()?;
        if !code.success() {
            bail!("{:?} failed with code {} and message: {}", cmd, code, stderr);
        }
    })}
}
