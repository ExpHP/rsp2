# FAQ

Designing a generic API in Rust can sometimes be trickier than you might at first expect. This document is intended to help remind myself about the design decisions involved in writing these generic functions, and to aid others in maintaining the crate if necessary.

## What's the easiest way to find the full API of these types?

For a full list of methods and free functions, just go to `methods_v.rs` or `methods_m.rs` and just look at the regions marked off as "PUBLIC API." I did this because I knew the docs would come out like crap.

As for operator impls, here's a (possibly outdated) list: (for a vector `v`, scalar `x`, and matrix `M`)

* `-{v,M}`
* `v {+,-} v`
* `x * {v,M}`
* `{v,M} {*,/} x`
* `M * M`
* `M * v`, `v * M`

Matrix arguments must always be shared borrows (`&`).

## Why are some methods just stubs for traits?

```rust
gen_each!{
    @{Vn}
    impl_v_inherent_wrappers!(
        {$Vn:ident}
    ) => {
        impl<X> $Vn<X> {

            /// Get the inner product of two vectors.
            #[inline(always)]
            pub fn dot(&self, other: &Self) -> ScalarT<Self>
            where Self: Dot,
            { Dot::dot(self, other) }

        }
    }
}
```

There are a number of possible reasons for making a trait for a method:

#### Free functions

`dot(a, b)` is arguably nicer than `a.dot(b)` in some scenarios, so `dot` is also exposed as a free function:
  ```rust
  pub fn dot<V>(a: &V, b: &V) -> ScalarT<V>
  where V: Dot,
  { a.dot(b) }
  ```
  Notice that this free function works for `V2`s as well as `V3`s and `V4`s. This would not be possible if we could not write a bound like `V: Dot`.

#### Complicated output type

For functions like `{M2,M3,M4}::t`, we probably at least want a `Transpose` trait just so that we can have an associated type for the transpose of a given matrix.  Might as well put the method there itself as well.


#### Encapsulation of awkward trait bounds

This is a big one.

If an inherent method `foo()` calls an inherent method `bar()`, then `foo` is required to include all of the necessary `where` bounds for `bar`.  In contrast, if an inherent method `foo()` calls a trait method `bar()`, then `Self: Bar` usually suffices.

Extending this argument to arbitrary downstream code; generic functions in downstream code need to have appropriate `where` bounds, and in this respect, individual traits per method are more stable. Normally this might be cited as an argument for wrapping *every single function* into its own trait... however, generally speaking, this point does not matter much to rsp2 since *all of the downstream code is also part of rsp2.*

What it all basically boils down to is creating a balance:

- *If every single method has its own trait,* then any function generic over vectors basically needs to duplicate its entire implementation within its `where` bounds to make sure that each method it calls is available. This is, of course, *really friggin dumb.*

- *If any method does not have its own trait,* then it might be able to keep a much simpler and more widely-applicable set of bounds (like `X: Semiring`).  However, *any change to its implementation which affects its `where` bounds will affect all code that uses it in a generic context.*

## Why is `X` used in some places while `ScalarT<Self>` is used in others? Aren't they interchangeable?

...ehhh. No. They really aren't. There will be times when Rust will not be able to unify `X` with `ScalarT<Self>`.

Sometimes, `ScalarT<Self>` is the only choice; for instance, any method wrapped in a trait is unable to use `X` as its output type because it is not in scope at the trait level (only in the impls).  For this reason, `ScalarT<Self>` is sometimes preferred over `X` as a return type even in cases where `X` could work, as this will create less churn if the method ever needs to be pulled out into a trait.
