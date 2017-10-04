//! Math utils for variable length contiguous vectors.
 
// Currently restricted to 'f64' to make the design tractible.
// Generic traits would be nice and I would kill for a functional-style
// maps and folds, but issues concerning borrowed data and trait bounds
// get ugly fast.
 
#[derive(Debug,Copy,Clone,PartialEq,PartialOrd)]
pub struct BadNorm(f64);

/// Implements element-wise operations.
///
/// Use the lowercase [`v`] to construct.
/// 
/// [`v`]: TODO-LINK-FN
#[derive(Debug,Copy,Clone,PartialEq,PartialOrd)]
pub struct V<T: AllowedV>(pub T);
pub type VOwn = V<Vec<f64>>;
pub type VRef<'a> = V<&'a [f64]>;
pub type VMut<'a> = V<&'a mut [f64]>;

/// This exists to give you better errors when you accidentally construct a `V<&Vec<f64>>`.
///
/// Hint: use `v(value)` instead of `V(value)` to construct V.
pub trait AllowedV {}
impl AllowedV for Vec<f64> {}
impl<'a> AllowedV for &'a [f64] {}
impl<'a> AllowedV for &'a mut [f64] {}


//------------------------
// Math ops

use ::std::ops::{
    Add, AddAssign, Sub, SubAssign,
    Mul, MulAssign, Div, DivAssign,
    Rem, RemAssign, Neg,
    Deref, DerefMut,
};

macro_rules! impl_binary {
    ($Op:ident::$op:ident, $OpAssign:ident::$op_assign:ident)
    => {
        // vector + scalar
        impl $Op<VOwn> for f64 {
            type Output = VOwn;
            fn $op(self, mut u: VOwn) -> VOwn {
                for x in &mut u.0 { *x = self.$op(*x); }
                u
            }
        }

        impl<'a> $Op<VRef<'a>> for f64 {
            type Output = VOwn;
            fn $op(self, u: VRef<'a>) -> VOwn {
                V(u.0.iter().cloned().map(|x| self.$op(x)).collect())
            }
        }

        // scalar + vector
        impl $Op<f64> for VOwn {
            type Output = VOwn;
            fn $op(mut self, s: f64) -> VOwn {
                for x in &mut self.0 { *x = x.$op(s); }
                self
            }
        }

        impl<'a> $Op<f64> for VRef<'a> {
            type Output = VOwn;
            fn $op(self, s: f64) -> VOwn {
                V(self.0.iter().cloned().map(|x| x.$op(s)).collect())
            }
        }

        // vector + vector
        impl $Op<VOwn> for VOwn {
            type Output = VOwn;
            fn $op(self, x: VOwn) -> VOwn { v(&self.0).$op(v(&x.0)) }
        }

        impl<'b> $Op<VRef<'b>> for VOwn {
            type Output = VOwn;
            fn $op(self, x: VRef<'b>) -> VOwn { v(&self.0).$op(x) }
        }

        impl<'a> $Op<VOwn> for VRef<'a> {
            type Output = VOwn;
            fn $op(self, x: VOwn) -> VOwn { v(self.0).$op(v(&x.0)) }
        }

        impl<'a,'b> $Op<VRef<'b>> for VRef<'a> {
            type Output = VOwn;
            fn $op(self, u: VRef<'b>) -> VOwn {
                assert_eq!(self.len(), u.len());
                V(self.0.into_iter().zip(u.0).map(|(a,b)| a.$op(b)).collect())
            }
        }

        // These don't work as dreamed.
        // v(&mut x) is not an lvalue so you can't do assign-ops.
        
        /*
        // vector += scalar
        impl<'a> $OpAssign<f64> for VMut<'a> {
            fn $op_assign(&mut self, s: f64) {
                for x in &mut *self.0 {
                    x.$op_assign(s);
                }
            }
        }

        // vector += vector
        impl<'a> $OpAssign<VOwn> for VMut<'a> {
            fn $op_assign(&mut self, u: VOwn) {
                assert_eq!(self.len(), u.len());
                for (x,y) in self.iter_mut().zip(u) {
                    x.$op_assign(y);
                }
            }
        }

        impl<'a,'b> $OpAssign<VRef<'b>> for VMut<'a> {
            fn $op_assign(&mut self, u: VRef<'b>) {
                assert_eq!(self.len(), u.len());
                for (x,&y) in self.iter_mut().zip(u.0) {
                    x.$op_assign(y);
                }
            }
        }
        */
    }
}

impl_binary!(Add::add, AddAssign::add_assign);
impl_binary!(Sub::sub, SubAssign::sub_assign);
impl_binary!(Mul::mul, MulAssign::mul_assign);
impl_binary!(Div::div, DivAssign::div_assign);
impl_binary!(Rem::rem, RemAssign::rem_assign);

impl Neg for VOwn {
    type Output = VOwn;
    fn neg(self) -> VOwn { -1.0 * self }
}

impl<'a> Neg for VRef<'a> {
    type Output = VOwn;
    fn neg(self) -> VOwn { -1.0 * self }
}

//------------------------
// Let &V coerce to &[f64] for the sake of `vdot` and `vnorm` below

impl Deref for VOwn {
    type Target = [f64];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<'a> Deref for VRef<'a> {
    type Target = [f64];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<'a> Deref for VMut<'a> {
    type Target = [f64];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<'a> DerefMut for VMut<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target { self.0 }
}

// override some functionality of VOwn to behave more like Vec than slices
impl IntoIterator for VOwn {
    type IntoIter = ::std::vec::IntoIter<f64>;
    type Item = f64;
    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

//------------------------
// v() function.
//
// This is all so you can write `v(&some_vec)` instead of `V(&some_vec[..])`
// (`V(&some_vec)` doesn't make the right type since the generic type
//  parameter in `V` inhibits reference coercions.)

pub fn v<W: MakeV>(w: W) -> W::Output { w.make_v() }

/// Implementation detail of [`v`].
/// 
/// [`v`]: TODO-LINK-FN
pub trait MakeV {
    type Output;
    fn make_v(self) -> Self::Output;
}

impl MakeV for Vec<f64> {
    type Output = VOwn;
    fn make_v(self) -> VOwn { V(self) }
}

impl<'a> MakeV for &'a [f64] {
    type Output = VRef<'a>;
    fn make_v(self) -> VRef<'a> { V(self) }
}

impl<'a> MakeV for &'a Vec<f64> {
    type Output = VRef<'a>;
    fn make_v(self) -> VRef<'a> { V(self) }
}

impl<'a> MakeV for &'a mut [f64] {
    type Output = VMut<'a>;
    fn make_v(self) -> VMut<'a> { V(self) }
}

impl<'a> MakeV for &'a mut Vec<f64> {
    type Output = VMut<'a>;
    fn make_v(self) -> VMut<'a> { V(self) }
}

//------------------------
// Math

pub fn vsqnorm(u: &[f64]) -> f64 { vdot(u, u) }
pub fn vnorm(u: &[f64]) -> f64 { vdot(u, u).sqrt() }

pub fn vnormalize(u: &[f64]) -> Result<VOwn, BadNorm> {
    let norm = vnorm(u);
    let recip = norm.recip();
    if !recip.is_normal() {
        return Err(BadNorm(norm));
    }

    Ok(recip * v(u))
}

pub fn vdot(u: &[f64], w: &[f64]) -> f64 {
    (v(u) * v(w)).into_iter().sum()
}

//---------------------------

#[cfg(test)]
mod tests {
    use super::{v,vdot,SubAssign};

    // TODO: test panics on incorrect length for:
    //  - each vector-vector binop impl
    //  - vdot

    #[test]
    fn vown_overridden_methods() {
        // Ensure IntoIter::Item is by-value
        let _: Box<Iterator<Item=f64>> = Box::new(v(vec![]).into_iter());
    }

    #[test]
    fn vdot_works() {
        assert_eq!(7.0, vdot(&[4.0, 2.0, 1.0], &[2.0, -1.0, 1.0]));
    }

    #[test]
    fn operand_order() {
        // exercise each separate impl with a non-commutative operator
        //  to make sure it puts the operands in the correct order
        let ua = vec![1.0, 2.0, 3.0];
        let ub = vec![4.0, 1.0, 1.0];
        let diff_ua_ub = v(vec![-3.0, 1.0, 2.0]);
        let diff_ua_2 = v(vec![-1.0, 0.0, 1.0]);
        let diff_2_ua = v(vec![1.0, 0.0, -1.0]);

        assert_eq!(diff_2_ua, 2.0 - v(ua.clone()));
        assert_eq!(diff_2_ua, 2.0 - v(&ua));
        assert_eq!(diff_ua_2, v(ua.clone()) - 2.0);
        assert_eq!(diff_ua_2, v(&ua) - 2.0);
        assert_eq!(diff_ua_ub, v(ua.clone()) - v(ub.clone()));
        assert_eq!(diff_ua_ub, v(ua.clone()) - v(&ub));
        assert_eq!(diff_ua_ub, v(&ua) - v(ub.clone()));
        assert_eq!(diff_ua_ub, v(&ua) - v(&ub));

/*
        assert_eq!(diff_ua_2, {
            let mut tmp = ua.clone();
            v(&mut tmp) -= 2.0;
            v(tmp.clone())
        });

        assert_eq!(diff_ua_ub, {
            let mut tmp = ua.clone();
            v(&mut tmp).sub_assign(v(&ub));
            v(tmp.clone())
        });

        assert_eq!(diff_ua_ub, {
            let mut tmp = ua.clone();
            v(&mut tmp).sub_assign(v(ub.clone()));
            v(tmp.clone())
        });
*/

    }
}