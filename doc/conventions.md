# Conventions

This documentation page exists to clarify the conventions that `rsp2` usually follows which are not always easy to describe in concise terms.

## Matrices

- Data-layout is **C order**, *without exception.*  I.e., the first index always has the greatest stride.
- All matrices are **row-major**, *without exception.*  I.e., if there is a data structure indexed by two indices, we call the first index the "row" and the second index the "column."
- The formalism is almost entirely **row-centric**. This is to say that, when you see an equation written in comments, the matrices therein are typically composed of row vectors.  Matrix multiplication is almost always between a row vector on the left and a matrix on the right, and a product of matrices is typically read from left to right.

These are actually all independent axes of design.  For instance, Fortran is Fortran order, row major, and most Fortran code uses a column-centric formalism.  WebGL code is Fortran order, column major, and column-centric.  When most people write code in C-order languages like C, C++, and Rust, it is typically row-major and column-centric.

Row-centric formalism is uncommon, but RSP2 uses it because it yields many of the benefits of Fortran order without the obvious disadvantages. Also, left-to-right composition of matrices feels natural and is highly amenable to method syntax.

There are a couple of exceptions:

* **Cartesian transformations** are **column-centric**, i.e. composed of three column vectors.  Typically, to operate on a row vector `x` with a rotation `R`, we write `x * R.t()`.
* When **Bra-Ket notation** is used, the formalism is **column-centric** (i.e. the state vector is typically a ket rather than a bra). Furthermore, after diagonalizing a matrix, one *could* say that the eigenvectors are stored in a column-major layout (as the first index selects an eigenket).  However, comments will describe this as "a matrix where each row is a column eigenvector," because it is simply easier to just always call the first index the "row."

## Reciprocal space and factors of 2π

RSP2 does **not** include factors of 2π in the definition of reciprocal space.  This is often called the crystallographer's convention.  RSP2 uses this convention *not* because it is written by crystallographers (it isn't!), but rather because it's the only convention that makes sense.

Using this convention, the form of an equation involving `k . r` looks exactly the same regardless of whether `k` and `r` are both fractional or both cartesian, making it easier to write bug-free code (whereas in the physicist convention, `k_c . r_c = 2π k_f . r_f`).  Needless to say, the definition of the reciprocal lattice and of miller index plane spacings are also simpler.

In fact, at the time of writing, *every single reference* to `pi` or `PI` in RSP2 is either part of a unit test, or it appears inside a computation of `exp(i 2π * ...)`.

## Phase convention for Q points away from gamma

RSP2 uses the same phase convention as Phonopy for normal mode displacements:

The normal modes of vibration computed at a point Q in reciprocal space contain an `exp(i 2π Q . x)` factor.  This means that, when a normal mode is translated by a lattice point R, this introduces a uniform phase factor of `exp(-i 2π Q . R)` (since it is equivalent to evaluating the original function at `x' = x - R`).

This affects the definition of the dynamical matrix, and thereby its eigenvectors and the normal mode displacements as well.

## Permutations
### Permutations as sequences of indices

#### The intuitive representation

`Perm::from_vec` and `Perm::into_vec` follow a convention that permutations __*pull* data into place.__

This is to say, applying the permutation `[1, 3, 2, 0]` to the sequence `[A, B, C, D]` will produce `[B, D, C, A]`. Basically:

* The value at index 1 (`B`) is _pulled into_ the first spot.
* The value at index 3 (`D`) is _pulled into_ the second spot.
* ... and etc.

This representation is generally easy to reason about, because the permutation vector transforms like the data it permutes. However, there's something funny about this format which becomes evident if you were to make it generic over index type:

* Let `IndexVec<I, X>` be a wrapped form of `Vec<X>` which uses type `I` (a newtype around `usize`) for its indices.
* Let `Perm<Src, Dest>` be an object that transforms data indexed by `Src` into data indexed by `Dest`.  E.g. `IndexVec<Src, X>` into `IndexVec<Dest, X>`.
* Then you can see that the above "intuitive" representation of `Perm<Src, Dest>` is actually `IndexVec<Dest, Src>`.  That is to say, given a `Dest` index, we can readily recover the `Src` index, but not the other way around.

This explains why rsp2 prefers to actually *store* the permutation in a different format.

#### The actual representation

rsp2 *actually* stores the inverse of the above representation.  If we were to make it generic over index types, one could say that `Perm<Src, Dest>` is stored as `IndexVec<Src, Dest>`.

In this format, a permutation _pushes_ elements to the written locations. For example, applying `[1, 3, 2, 0]` to `[A, B, C, D]` would produce `[D, A, C, B]`, py *pushing* the `A` to position 1, *pushing* the `B` to position 3, and etc.

Both representations are equally efficient at permuting data stored in a dense format.  The advantage of *this* format is that it can also efficiently permute sparse data:

* Let `type SparseVec<I, X> = Vec<(I, X)>` represent a sparse vector of index type `I`.
* Then you can see how the actual representation of `Perm<Src, Dest>` (which is `IndexVec<Src, Dest>`) can efficiently transform `SparseVec<Src, X>` into `SparseVec<Dest, X>`, while the intuitive representation (which is `IndexVec<Dest, Src>`) would require searching for each index (or depending on the circumstances, precomputing the inverse).

### Permutations representing symmetry operators

rsp2 picks a convention here that might seem unusual.  Let me explain the choices here, and explain which choice rsp2 uses, and why.

#### The copermutation of a symmetry operator

Suppose you have a structure described by `coords` (which has `N` rows of `[f64; 3]` position data) and `lattice`.  Let `oper` be a symmetry operator (which may be translational, rotational, or both) on this structure.

There must exist a permutation `perm` such that applying `oper` to `coords` should have the same effect as permuting by `perm`; rsp2 calls this the **copermutation** (short for "coordinate permutation") of a symmetry operator:

```text
 coords.transformed_by(oper) ~~ coords.permuted_by(perm)
```

where `~~` is an equivalence relation[^1] which tests that, for all indices `i`, the `i`th position in the LHS is equivalent to the `i`th position in the RHS under the translational symmetry of `lattice`.

After taking a moment to digest the above definition, it might seem that I am simply stating the obvious.  But, you see, *now* comes the tricky part, because it turns out these permutations have some very surprising properties.

#### Why copermutations of symmetry operators suck

The root of all trouble is the following:

*Copermutations compose in the reverse order of the operators they describe.*

Basically, if you have operators `R1` and `R2` and corresponding copermutations `P1` and `P2`, then the effect of transforming first by `R1` then by `R2` is equivalent to permuting the coordinates *first* by `P2` and then by `P1`.  Weird, right?  But it makes sense if you think about it. After all, the copermutation of an operator depends on what order the coords are initially arranged:

(**note:** Here I am showing the intuitive representation of permutations, not the format actually stored by rsp2)

```
     x coords       ==>  copermutation that mirrors along x
  [ 1,  2, -1, -2]  ==>          [ 2, 3, 0, 1]
  [-1, -2,  1,  2]  ==>          [ 2, 3, 0, 1]
  [ 1, -1,  2, -2]  ==>          [ 1, 0, 3, 2]
  [ 1, -2,  2, -1]  ==>          [ 3, 2, 1, 0]

```

So lets say that after computing a bunch of copermutations for various operators, you apply one to `coords` in order to simulate performing that operation.  By putting `coords` into a different order, *you just invalidated all of the copermutations you computed,* and you now must correct each one by applying a similarity transform.

What all of this tells us is that permuting `coords` is something we generally want to *avoid doing* when doing symmetry-related stuff.  Which is too bad, because it's the use case that copermutations were optimized for!

#### The _depermutation_ of a symmetry operator

Because permutations of symmetry operations are so troublesome conceptually, you might see local bindings, method names, and comments refer to another concept called the **depermutation** of a symmetry operator.  For a given `coords` and `oper`, it is the unique permutation `deperm` that satisfies

```
 coords.transformed_by(oper).permuted_by(deperm) ~~ coords
```

On other words, it's just the inverse of `perm`.  But the terms "copermutation" and "depermutation" were invented to deliberately avoid use of the term "inverse," because it has the potential to create confusion.

In rsp2, `perm` and `deperm` are both considered to be representations of `oper`. *Neither* represents the inverse of `oper` (if we wanted that, we would have called them `deperm_inv` or `perm_inv`!). The difference between the two is in *how they are used.*

#### Depermutations permute metadata in a composable manner

Suppose we have a supercell of a 1D structure along the Z axis with two atoms in the primitive cell. Suppose also that we have metadata that labels each site according to its primitive atom and unit cell index.

 ```
  Fractional z: [0.01, 0.02, 0.26, 0.27, 0.51, 0.52, 0.76, 0.77]
        Labels: ["0a", "0b", "1a", "1b", "2a", "2b", "3a", "3b"]
 ```

`Fractional z` is in units of the supercell lattice.  This supercell has 4 equivalent cells along Z, giving it a translational symmetry operator with vector `(0, 0, 0.25)`. The copermutation of this operator is `coperm = [2, 3, 4, 5, 6, 7, 0, 1]`, and the depermutation is `deperm = [6, 7, 0, 1, 2, 3, 4, 5]`.

Both of these permutations can be used to create a structure where the each label appears to have been displaced by a vector of `(0, 0, 0.25)`.  You can **apply `coperm` to the coordinates:** (leaving metadata untouched)

```
  Fractional z: [0.26, 0.27, 0.51, 0.52, 0.76, 0.77, 0.01, 0.02]
        Labels: ["0a", "0b", "1a", "1b", "2a", "2b", "3a", "3b"]
```

...or you can **apply `deperm` to the metadata:** (leaving the coordinates untouched)

```
  Fractional z: [0.01, 0.02, 0.26, 0.27, 0.51, 0.52, 0.76, 0.77]
        Labels: ["3a", "3b", "0a", "0b", "1a", "1b", "2a", "2b"]
```

The advantage of using `deperm` is that it composes properly; because the coords were left untouched, all previously-computed coperms and deperms are still valid descriptions of their corresponding operators.

## Footnotes

[^1]: I am ignoring issues of floating point precision here.  In reality, `equiv` must use a tolerance, and thus fails to be an equivalence relation in the strictest mathematical sense as it is not transitive.
