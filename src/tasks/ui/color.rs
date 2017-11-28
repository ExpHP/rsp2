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

/// Specialized display impl for numbers that from 0 to 1 and may be
/// extremely close to either 0 or 1
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct DisplayProb(pub f64);
impl fmt::Display for DisplayProb {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let log10_1p = |x: f64| x.ln_1p() / ::std::f64::consts::LN_10;
        assert!(0.0 < self.0 && self.0 < 1.0 + 1e-5,
            "bad probability: {}", self.0);

        if self.0 >= 1.0 {
            write!(f, "{:>7}", 1.0)
        } else if self.0 <= 1e-3 {
            write!(f, "  1e{:03}", self.0.log10().round())
        } else if self.0 + 1e-3 >= 1.0 {
            write!(f, "1-1e{:03}", log10_1p(-self.0).round())
        } else {
            write!(f, "{:<7.5}", self.0)
        }
    }
}

pub struct ColorByRange<T> {
    pub divs: Vec<(T, Style)>,
    pub lowest: Style,
}
impl<T> ColorByRange<T> {
    pub fn new(divs: Vec<(T, Style)>, lowest: Style) -> ColorByRange<T> {
        ColorByRange { divs, lowest }
    }

    fn style_of(&self, x: &T) -> Style
    where T: PartialOrd,
    {
        for &(ref pivot, style) in &self.divs {
            if x > pivot { return style; }
        }
        return self.lowest;
    }

    pub fn paint<'a, U>(&self, x: U) -> Wrapper<U, T>
    where
        T: PartialOrd + 'a,
        U: ::std::borrow::Borrow<T> + 'a,
    {
        gpaint(self.style_of(x.borrow()), x)
    }

    pub fn paint_as<'a, U>(&self, compare_me: &T, show_me: U) -> Wrapper<U, U>
    where T: PartialOrd,
    {
        paint(self.style_of(compare_me), show_me)
    }
}

// hack for type  inference issues
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
/// Except `Debug`.
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
