//! # What is this?
//!
//! A doc comment clarifying the difference between row-centric
//! and row-major.
//!
//! # Why keep it?
//!
//! It remains as an accurate description of pretty much any API
//! I design that involves matrices. Also, I have a tendency to
//! waste lots of time time rewording stuff like this (and I did),
//! so it'd be nice if I could just remember that this is sitting
//! around the next time I feel the need to document this somewhere.



//! # A note on conventions
//!
//! There are two independent dimensions by which we describe the layout
//! of a matrix-like data structure.
//!
//! The first is the mathematical convention:
//!
//! * **column-centric** -
//!    This is the traditional convention that you probably learned in school.
//!    Most things we care about are represented by column vectors.
//!    Matrix operators are applied on the left of their "argument" (like functions).
//!    Matrix products read most naturally from right to left.
//! * **row-centric** -
//!    Most things we care about are represented by row vectors.
//!    Matrix operators are applied on the right of their "argument" (like methods),
//!    Matrix products read most naturally from left to right.
//!
//! The second is the storage convention:
//!
//! * **row-major** - The slower (outer) index is ascribed to rows.
//! * **column-major** - The slower (outer) index is ascribed to columns.
//!
//! This API uses **row-major layout _exclusively_.** The only reason we even bring
//! it up is to help clarify the distinction between row-centric and column-centric.
//!
//! Using row-major layout exclusively means that the textual rows in literal and
//! string-formatted matrices always corresponds to the rows in the mathematical object.
//! You don't need to transpose anything in your head; that would be _heartless!_
//! This decision in turn makes the row-centric formalism even more attractive,
//! because it is easy to access rows as `&[f64]` or `&[f64; 3]` in row-major layout,
//! and impossible to access columns.
//!
//! Therefore, **almost all matrices are row-centric.**  The primary exceptions are
//! *cartesian operators* and *eigenvector matrices*, each of whose column-centeredness
//! is so deeply ingrained into the mathematical community that the cost of betraying
//! convention far outweighs the benefit of easy access to each constituent vector.
