mod vec {
	use ::std::fmt;

	#[derive(Copy,Clone,Hash,PartialEq,Eq,PartialOrd,Ord,Debug)]
	pub struct Vec2<T>(pub T, pub T);
	#[derive(Copy,Clone,Hash,PartialEq,Eq,PartialOrd,Ord,Debug)]
	pub struct Vec3<T>(pub T, pub T, pub T);

	pub type Mat2<T> = Vec2<Vec2<T>>;
	pub type Mat3<T> = Vec3<Vec3<T>>;

	impl<T> Vec2<T> {
		pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Vec2<U> {
			Vec2(f(self.0), f(self.1))
		}
	}
	impl<T> Vec3<T> {
		pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Vec3<U> {
			Vec3(f(self.0), f(self.1), f(self.2))
		}
	}

	use ::std::ops::{Add,Mul,Neg};
	impl<T> Vec2<Vec2<T>> where T: Add<T,Output=T> + Mul<T,Output=T> + Neg<Output=T> + Copy {
		pub fn det(self) -> T {
			let Vec2(Vec2(m00,m01), Vec2(m10,m11)) = self;
			m00 * m11 + -(m10 * m01)
		}
	}
	impl<T> Vec3<Vec3<T>> where T: Add<T,Output=T> + Mul<T,Output=T> + Neg<Output=T> + Copy {
		pub fn det(self) -> T {
			let Vec3(Vec3(m00,m01,m02), Vec3(m10,m11,m12), Vec3(m20,m21,m22)) = self;
			(   m00 * m11 * m22
			+   m01 * m12 * m20
			+   m02 * m10 * m21
			+ -(m00 * m12 * m21)
			+ -(m01 * m10 * m22)
			+ -(m02 * m11 * m20)
			)
		}
	}

	impl<T: fmt::Display> fmt::Display for Vec2<T> {
		fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
			write!(f, "[{}, {}]", &self.0, &self.1)
		}
	}
	impl<T: fmt::Display> fmt::Display for Vec3<T> {
		fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
			write!(f, "[{}, {}, {}]", &self.0, &self.1, &self.2)
		}
	}
}

mod ratio {
    //! A highly-specialized ratio type that only implements
    //! the operations required by the HNF search.
	use ::numtheory::{gcd, extended_gcd, GcdData};
	use super::Int;

	#[derive(Copy,Clone,Hash,PartialEq,Eq,PartialOrd,Ord,Debug)]
	pub struct Ratio {
		numer: Int,
		denom: Int,
	}

	impl ::std::fmt::Display for Ratio {
		fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
			match self.denom {
				1 => write!(f, "{}", self.numer),
				_ => write!(f, "{}/{}", self.numer, self.denom),
			}
		}
	}

	fn coprime_parts(a: i32, b: i32) -> (i32,i32) {
		let GcdData { quotients: (a, b), .. } = extended_gcd(a, b);
		(a, b)
	}

	impl Ratio {
		pub fn new(numer: i32, denom: i32) -> Ratio {
			debug_assert!(denom != 0, "divide by zero");
			let (numer, denom) = (numer * denom.signum(), denom * denom.signum());
			let (numer, denom) = coprime_parts(numer, denom);
			Ratio::new_unchecked(numer, denom)
		}
		pub fn numer(self) -> i32 { self.numer }
		/// Denominator in simplest form
		pub fn denom(self) -> i32 { self.denom }
		pub fn zero() -> Ratio { Ratio { numer: 0, denom: 1 } }
		pub fn one()  -> Ratio { Ratio { numer: 1, denom: 1 } }

		fn new_unchecked(numer: i32, denom: i32) -> Ratio {
			// we'll still check in debug mode...
			debug_assert!(gcd(numer, denom) == 1, "numer and denom not coprime");
			debug_assert!(denom >= 1, "denom not positive");
			Ratio { numer, denom }
		}

		pub fn to_integer(self) -> Option<i32> {
			match self.denom {
				1 => Some(self.numer),
				_ => None,
			}
		}
	}

	impl ::std::ops::Add<Ratio> for Ratio {
		type Output = Ratio;
		fn add(self, other: Ratio) -> Ratio {
			Ratio::new(
				self.numer * other.denom + self.denom * other.numer,
				self.denom * other.denom,
			)
		}
	}

	/// Ratios can be multiplied directly by integers,
	/// at a potentially cheaper cost.
	/// TODO: Bench?
	impl ::std::ops::Mul<i32> for Ratio {
		type Output = Ratio;
		fn mul(self, b: i32) -> Ratio {
			let (new_denom, new_b) = coprime_parts(self.denom, b);
			Ratio::new_unchecked(self.numer * new_b, new_denom)
		}
	}

	impl ::std::ops::Mul<Ratio> for Ratio {
		type Output = Ratio;
		fn mul(self, other: Ratio) -> Ratio {
			Ratio::new(
				self.numer * other.numer,
				self.denom * other.denom,
			)
		}
	}

	impl ::std::ops::Neg for Ratio {
		type Output = Ratio;
		fn neg(self) -> Ratio {
			Ratio::new_unchecked(-self.numer, self.denom)
		}
	}

	/// Check if two ratios add to an integer.
	/// Theoretically cheaper than `(a + b).to_integer().is_some()`,
	///  which would needlessly compute a reduced form of the sum.
	/// TODO: Empirical benchmark?
	pub fn sum_is_integer(a: Ratio, b: Ratio) -> bool {
		(a.denom == b.denom) && (a.numer + b.numer) % a.denom == 0
	}
}

mod hnf {
	use super::Int;
	use ::vec::{Mat2,Vec2};
	use ::ratio::Ratio;
	use numtheory::lcm;

	pub fn search_hnf_prefactor_mat2(mat: Mat2<Ratio>) -> Mat2<Int> {
		let Vec2(
			Vec2(m00,m01),
			Vec2(m10,m11),
		) = mat;

		let c00 = lcm(m00.denom(), m01.denom());

		let c10;
		let c11;
		let mut y = 0i32;
		'y: loop {
			y += 1;
			let sums = Vec2(m10, m11).map(|m| m * y);
			for x in 0..c00 {
				if ::ratio::sum_is_integer(sums.0, m00 * x) {
					if ::ratio::sum_is_integer(sums.1, m01 * x) {
						c10 = x;
						c11 = y;
						break 'y;
					}
				}
			}
		}
		Vec2(Vec2(c00,0),Vec2(c10,c11))
	}
}
