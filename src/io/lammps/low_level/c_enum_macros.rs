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

macro_rules! c_enums {
    (
        $(
            [$($vis:tt)*] enum $Type:ident {
                // tt so it can double as expr and pat
                $($Variant:ident = $value:tt,)+
            }
        )+
    ) => {
        $(
            #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
            $($vis)* enum $Type {
                $($Variant = $value,)+
            }

            impl $Type {
                #[allow(unused)]
                pub fn from_int(x: u32) -> crate::FailResult<$Type>
                { match x {
                    $($value => Ok($Type::$Variant),)+
                    _ => bail!("Invalid value {} for {}", x, stringify!($Type)),
                }}
            }
        )+
    };
}

macro_rules! derive_into_from_as_cast {
    ($($A:ty as $B:ty;)*)
    => { $(
        impl From<$A> for $B {
            fn from(a: $A) -> $B { a as $B }
        }
    )* };
}
