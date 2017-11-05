pub(crate) mod layer;
pub(crate) mod supercell;

// these are allow(unused) because they contain
// untested/incomplete code
#[allow(unused)] pub(crate) mod rotations;
#[allow(unused)] pub(crate) mod reduction;
pub(crate) mod find_perm;

// these are tested but not yet part of public APIs
#[cfg_attr(not(test), allow(unused))]
pub(crate) mod group;

#[cfg(test)]
mod tests;
