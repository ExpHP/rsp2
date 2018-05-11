#[macro_use] extern crate failure;
extern crate rand;

#[cfg(test)]
macro_rules! assert_matches {
    ($pat:pat, $expr:expr,)
    => { assert_matches!($pat, $expr) };
    ($pat:pat, $expr:expr)
    => { assert_matches!($pat, $expr, "actual {:?}", $expr) };
    ($pat:pat, $expr:expr, $($arg:expr),+ $(,)*)
    => {
        match $expr {
            $pat => {},
            _ => panic!(
                "assertion failed: {} ({})",
                stringify!(assert_matches!($pat, $expr)),
                format_args!($($arg),+))
        }
    };
}

pub mod helper {
    // NOTE: it's worth noting that, to date, this has never been used outside
    //       of this crate.  But it was *intended* to be public, and I guess it
    //       could still be useful if some veclike data were not stored in a Vec
    //       for some reason. (like a dense matrix?)
    pub use ::part::composite_perm_for_part_lifo;
}

pub use self::perm::{Perm, Permute};
pub use self::perm::InvalidPermutationError;
mod perm;

pub use self::part::{Part, Parted, Partition, Unlabeled};
pub use self::part::InvalidPartitionError;
mod part;

mod util;
