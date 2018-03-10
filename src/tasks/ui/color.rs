use ::std::fmt;
use ::ansi_term::Style;

macro_rules! each_fmt_trait {
    ($mac:ident!)
    => {
        $mac!(::std::fmt::Display);
        $mac!(::std::fmt::Octal);
        $mac!(::std::fmt::LowerHex);
        $mac!(::std::fmt::UpperHex);
        $mac!(::std::fmt::Pointer);
        $mac!(::std::fmt::Binary);
        $mac!(::std::fmt::LowerExp);
        $mac!(::std::fmt::UpperExp);
    }
}


/// A thing that might colorize text based on a value.
pub trait PaintAs<D, C> {
    fn paint_as(&self, compared: &C, displayed: D) -> String;
}

pub struct ColorByRange<T> {
    pub divs: Vec<(T, Style)>,
    pub lowest: Style,
}

impl<T> ColorByRange<T> {
    pub fn new(divs: Vec<(T, Style)>, lowest: Style) -> ColorByRange<T>
    { ColorByRange { divs, lowest } }

    fn style_of(&self, x: &T) -> Style
    where T: PartialOrd,
    {
        for &(ref pivot, style) in &self.divs {
            if x > pivot { return style; }
        }
        return self.lowest;
    }

    // pub fn paint<'a, U>(&self, x: U) -> Wrapper<U, T>
    // where
    //     T: PartialOrd + 'a,
    //     U: ::std::borrow::Borrow<T> + 'a,
    // { gpaint(self.style_of(x.borrow()), x) }
}

impl<D, C> PaintAs<D, C> for ColorByRange<C>
  where C: PartialOrd, D: fmt::Display,
{
    fn paint_as(&self, compared: &C, displayed: D) -> String
    { paint(self.style_of(compared), displayed).to_string() }
}

/// Does not colorize.
pub struct NullPainter;

impl<D, C> PaintAs<D, C> for NullPainter
  where C: PartialOrd, D: fmt::Display,
{
    fn paint_as(&self, _: &C, displayed: D) -> String
    { displayed.to_string() }
}

// hack for type inference issues
pub fn paint<T>(
    style: ::ansi_term::Style,
    value: T,
) -> Wrapper<T, T>
{ gpaint(style, value) }

pub fn gpaint<U, T>(
    style: ::ansi_term::Style,
    value: U,
) -> Wrapper<U, T>
{ Wrapper { style, value, _target: Default::default() } }

/// A wrapper for colorizing all formatting traits like `Display`.
///
/// It has two parameters so that it can `borrow()` `U` as `T` when it wants to.
/// (otherwise, it would have to store `&T`, making it virtually impossible to
///  return one of these from a function)
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Wrapper<U, T=U> {
    style: ::ansi_term::Style,
    value: U,
    _target: ::std::marker::PhantomData<T>,
}

macro_rules! derive_fmt_impl {
    ($Trait:path)
    => {
        impl<U, T> $Trait for Wrapper<U, T>
        where
            U: ::std::borrow::Borrow<T>,
            T: $Trait,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", self.style.prefix())?;
                T::fmt(self.value.borrow(), f)?;
                write!(f, "{}", self.style.suffix())?;
                Ok(())
            }
        }
    };
}

each_fmt_trait!{derive_fmt_impl!}
