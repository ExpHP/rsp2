# Conventions

This documentation page exists to clarify the conventions that `rsp2` usually follows which are not always easy to describe in concise terms.

## Matrices

Matrices are row-major, the formalism is row-centric (with the exception of cartesian space operators), and data-layout is C order.

(TODO: It's easy to conflate these terms, but they're really three distinct axes of design.  A while back I wrote up something somewhere that clarified the subtle differences, but can't seem to find it now...)

## Permutations
### Permutations as sequences of indices

Although method docs (and this document) may often speak as though permutations _push_ data around, the convention followed by `rsp2` is really such that they __*pull* data into place.__

This is to say, applying the permutation `[1, 3, 2, 0]` to the sequence `[A, B, C, D]` will produce `[B, D, C, A]`. Basically:

* The value at index 1 (`B`) is _pulled into_ the first spot.
* The value at index 3 (`D`) is _pulled into_ the second spot.
* ... and etc.

Contrast this with another possible convention, where applying the perm _pushes_ elements to the given locations. (where applying `[1, 3, 2, 0]` to `[A, B, C, D]` would produce `[D, A, C, B]`.

rsp2's convention matches how numpy works when arrays are indexed by integer arrays.  In this same vein, rsp2's permutation operations can be seen as a special case of a more general operation of "indexing by an array," where e.g. indexing `[A, B, C, D]` by `[1, 1, 1, 3, 1]` would produce `[B, B, B, D, B]`.  Note however that the design of the `Permute` trait deliberately prevents such usage, and that construction of a `Perm` from a sequence like `[1, 1, 1, 3, 1]` is forbidden in order to allow `Permute` impls to be written without `Clone` bounds.

Most code does not need even need to be aware of this convention thanks to the `Perm` type and `Permute` trait that largely abstract it away.  However, functions that produce permutations through means other than composition are generally expected to do so through `Perm::from_vec`, which requires one to be aware of this convention.

### The permutation _of a symmetry operator_

This is a phrase you may encounter.  In rsp2, the permutation of a symmetry operator on positions **tries to represent that operator as a permutation on the positions.** You might be thinking, "well, _duh_," but there's actually a bit of nuance here you should be careful about.

In fact, some of the consequences are so counterintuitive that I am considering switching to a different convention.

#### Convention used by rsp2

Suppose you have a structure described by `coords` (which has `N` rows of `[f64; 3]` position data) and `lattice`.  Let `oper` be a symmetry operator (which may be translational, rotational, or both) on this structure.

The permutation `perm` of a symmetry operator `oper` satisfies

```text
 coords.transformed_by(oper) ~~ coords.permuted_by(perm)
```

where `~~` is an equivalence relation[^get-real] which tests that, for all indices `i`, the `i`th position in the LHS is equivalent to the `i`th position in the RHS under the translational symmetry of `lattice`.

After taking a moment to digest the above paragraphs, you may find yourself again thinking that this is all obvious and that I've just wasted your time.

*Now* comes the tricky part, because it turns out this definition has some very surprising consequences.  It's possible that rsp2 actually made a _very poor choice_ here, and this definition might even be reconsidered!  (hopefully soon, before too much code depends on it!)

[^get-real]: I am ignoring issues of floating point precision here.  In reality, `equiv` must use a tolerance, and thus fails to be an equivalence relation in the strictest mathematical sense as it is not transitive.

#### Meaning of this convention _in practice_

**Tl;dr:** Challenge your instincts and think carefully before applying a permutation derived from symmetry to anything other than coordinate data. (it may be the case that you need to apply the inverse instead)

For simplicity of providing graphical descriptions, this example will use a pure translation operator (these show up in supercells).[^order-2] Suppose we have a supercell of a 1D structure along the Z axis with two atoms in the primitive cell. Suppose also that we have metadata that labels each site according to its primitive atom and unit cell index.

 ```
  Fractional z: [0.01, 0.02, 0.26, 0.27, 0.51, 0.52, 0.76, 0.77]
        Labels: ["0a", "0b", "1a", "1b", "2a", "2b", "3a", "3b"]
 ```

`Fractional z` is in units of the supercell lattice.  This supercell has 4 equivalent cells along Z, giving it a translational symmetry operator for the vector `(0, 0, 0.25)`. The corresponding permutation is `[2, 3, 4, 5, 6, 7, 0, 1]`.

As stated before, applying the perm **only to the coordinates** has the same effect as applying the translation, causing each atom to appear to move by `(0, 0, 0.25)`:

```
  Fractional z: [0.26, 0.27, 0.51, 0.52, 0.76, 0.77, 0.01, 0.02]
        Labels: ["0a", "0b", "1a", "1b", "2a", "2b", "3a", "3b"]
 ```

However, if we instead applied the perm **only to the labels**, it would appear to have the opposite effect; each label will appear to have moved by a fractional displacement of `(0, 0, −0.25)`!

```
  Fractional z: [0.01, 0.02, 0.26, 0.27, 0.51, 0.52, 0.76, 0.77]
        Labels: ["1a", "1b", "2a", "2b", "3a", "3b", "0a", "0b"]
```

There'd be a snag of some sort here no matter what convention we used; but in our case, **it gets worse.**

[^order-2]: The only space group operators that are easy to depict in 1D are all of order 2, but the effect demonstrated in this example is only visible in operators whose order is at least 3.

#### How permutations of symmetry operators compose

**Tl;dr:** Thanks to the convention used by rsp2, permutations of symmetry operators appear to compose _in the wrong direction._

Basically, it comes down to this:

* Symmetry operators act in positional space.  Writing positions as `Nx3` matrices, symmetry operators are `3x3` matrices applied (in transpose) on the RHS.
* Permutations act in index space.  Writing positions as `Nx3` matrices, permutations are `NxN` matrices applied on the LHS.

So if `X` is an `Nx3` matrix of positions, and we have two symmetry operators `R1` and `R2` with known permutations `P1` and `P2`.  Let's see how they compose:

```
Suppose  P1 X = X R1.T
    and  P2 X = X R2.T

   then:    P2 X inv(R2.T) = X
         P1 P2 X inv(R2.T) = X R1.T
                   P1 P2 X = X R1.T R2.T
```

In other words, `x.transformed_by(r1).transformed_by(r2)` is equivalent to `x.permuted_by(p2).permuted_by(p1)`.  The permutations need to be applied in reverse order!

#### Another possible convention

It turns out there is another way to look at things: What if we instead worked with the permutation that *undoes* a symmetry operation?

```
  coords.transformed_by(oper).permuted_by(perm) ~~ coords
```

This definition would eliminate each of the surprising consequences previously described, and the most surprising effect it introduces is easily summed up by the word "undoes."  As far as I can tell, this is a much nicer concept to work with.
