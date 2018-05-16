# Conventions

This documentation page exists to clarify the conventions that `rsp2` usually follows which are not always easy to describe in concise terms.

## Matrices

Matrices are row-major, the formalism is row-centric (with the exception of cartesian space operators), and data-layout is C order.

(TODO: It's easy to conflate these terms, but they're really three distinct axes of design.  A while back I wrote up something somewhere that clarified the subtle differences, but can't seem to find it now...)

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

#### The permutation of a symmetry operator

Suppose you have a structure described by `coords` (which has `N` rows of `[f64; 3]` position data) and `lattice`.  Let `oper` be a symmetry operator (which may be translational, rotational, or both) on this structure.

There must exist a permutation `perm` such that applying `oper` to `coords` should have the same effect as permuting by `perm`.  We call this the **permutation of a symmetry operator.**

```text
 coords.transformed_by(oper) ~~ coords.permuted_by(perm)
```

where `~~` is an equivalence relation[^1] which tests that, for all indices `i`, the `i`th position in the LHS is equivalent to the `i`th position in the RHS under the translational symmetry of `lattice`.

After taking a moment to digest the above definition, it might seem that I am simply stating the obvious.  But, you see, *now* comes the tricky part, because it turns out these permutations have some very surprising properties.

#### Why permutations of symmetry operators suck

The root of all trouble is the following:

*These permutations compose in the reverse order compared to the operators they describe.*

Basically, if you have operators `R1` and `R2` and corresponding permutations `P1` and `P2`, then the effect of transforming first by `R1` then by `R2` is equivalent to permuting the coordinates *first* by `P2` and then by `P1`.  Weird, right?  But it makes sense if you think about it. After all, the permutation representation of an operator depends on what order the coords are initially arranged:

(**note:** Here I am showing the intuitive representation of permutations, not the format actually stored by rsp2)

```
     x coords       ==>  permutation that mirrors along x
  [ 1,  2, -1, -2]  ==>          [ 2, 3, 0, 1]
  [-1, -2,  1,  2]  ==>          [ 2, 3, 0, 1]
  [ 1, -1,  2, -2]  ==>          [ 1, 0, 3, 2]
  [ 1, -2,  2, -1]  ==>          [ 3, 2, 1, 0]

```

So lets say that after computing a bunch of permutations for various operators, you apply one of your permutations to `coords` in order to simulate performing that operation.  By putting `coords` into a different order, *you just invalidated all of the permutations you computed,* and you now must correct each one by applying a similarity transform.

What all of this tells us is that applying permutations to `coords` is something we generally want to *avoid doing.*  Which is too bad, because it's the use case that our definition was optimized for!

#### The _depermutation_ of a symmetry operator

Because permutations of symmetry operations are so troublesome conceptually, you might see local bindings, method names, and comments refer to another concept called the **depermutation of a symmetry operator.**  For a given `coords` and `oper`, it is the unique permutation `deperm` that satisfies

```
 coords.transformed_by(oper).permuted_by(deperm) ~~ coords
```

On other words, it's just the inverse of `perm`.  But the term "depermutation" was invented to deliberately avoid use of the term "inverse," because it has the potential to cause confusion.

In rsp2, the `deperm` of an operator `oper` is considered to represent `oper` itself, not the inverse of `oper`! The difference between `perm` is in *how they are used.*

#### Depermutations permute metadata in a composable manner

Suppose we have a supercell of a 1D structure along the Z axis with two atoms in the primitive cell. Suppose also that we have metadata that labels each site according to its primitive atom and unit cell index.

 ```
  Fractional z: [0.01, 0.02, 0.26, 0.27, 0.51, 0.52, 0.76, 0.77]
        Labels: ["0a", "0b", "1a", "1b", "2a", "2b", "3a", "3b"]
 ```

`Fractional z` is in units of the supercell lattice.  This supercell has 4 equivalent cells along Z, giving it a translational symmetry operator with vector `(0, 0, 0.25)`. The permutation of this operator is `perm = [2, 3, 4, 5, 6, 7, 0, 1]`, and the depermutation is `deperm = [6, 7, 0, 1, 2, 3, 4, 5]`.

Both of these permutations can be used to create a structure where the each label appears to have been displaced by a vector of `(0, 0, 0.25)`.  You can **apply `perm` to the coordinates:** (leaving metadata untouched)

```
  Fractional z: [0.26, 0.27, 0.51, 0.52, 0.76, 0.77, 0.01, 0.02]
        Labels: ["0a", "0b", "1a", "1b", "2a", "2b", "3a", "3b"]
```

...or you can **apply `deperm` to the metadata:** (leaving the coordinates untouched)

```
  Fractional z: [0.01, 0.02, 0.26, 0.27, 0.51, 0.52, 0.76, 0.77]
        Labels: ["3a", "3b", "0a", "0b", "1a", "1b", "2a", "2b"]
```

The advantage of using `deperm` is that it composes properly; because the coords were left untouched, all previously-computed perms and deperms are still valid descriptions of their corresponding operators.

## Footnotes

[^1]: I am ignoring issues of floating point precision here.  In reality, `equiv` must use a tolerance, and thus fails to be an equivalence relation in the strictest mathematical sense as it is not transitive.
