
use ::clap;
use ::Result;

/// Trait used to factor out code for adding arguments to a binary and parsing them,
/// leveraging type inference to help reduce boilerplate.
pub trait CliDeserialize: Sized {
    fn augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> (clap::App<'a, 'b>, ClapDeserializer<Self>)
    {
        let app = Self::_augment_clap_app(app);
        let token = ClapDeserializer(Default::default());
        (app, token)
    }

    /// Don't use this. Call 'augment_clap_app' instead.
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b>;
    /// Don't use this. Call 'resolve_args' on the ClapDeserializer instead.
    fn _resolve_args(matches: &clap::ArgMatches) -> Result<Self>;
}

/// Token of "proof" that a clap app was augmented to be capable of deserializing A.
///
/// (note this requirement can be easily circumvented; it's just a speed bump to
///  catch stupid mistakes)
pub struct ClapDeserializer<A>(::std::marker::PhantomData<A>);

impl<A> ClapDeserializer<A>
where A: CliDeserialize,
{
    /// Deserialize the arguments.  This may perform IO such as eagerly reading input files.
    ///
    /// (that said, implementations of the trait are highly discouraged from doing things
    ///  that would cause the behavior of arg resolution to depend on the order in which
    ///  multiple CliDeserialize instances are handled)
    pub fn resolve_args(self, matches: &clap::ArgMatches) -> Result<A>
    { A::_resolve_args(matches) }
}

// Trivial impl so that an entry point with no CliDeserialize args can still
// use the same boilerplate or generic utility function.
impl CliDeserialize for () {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b>
    { app }

    fn _resolve_args(_: &clap::ArgMatches) -> Result<Self>
    { Ok(()) }
}

// Tuple as product combinator
impl<A, B> CliDeserialize for (A, B)
where
    A: CliDeserialize,
    B: CliDeserialize,
{
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b>
    {
        let app = A::_augment_clap_app(app);
        let app = B::_augment_clap_app(app);
        app
    }

    fn _resolve_args(matches: &clap::ArgMatches) -> Result<Self>
    { Ok((A::_resolve_args(matches)?, B::_resolve_args(matches)?)) }
}
