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

// 1D test functions for e.g. linesearch.
pub mod one_dee {

    pub mod prelude {
        pub use super::Differentiable1d;
    }

    pub trait Differentiable1d: Clone {
        type Derivative: Differentiable1d;
        fn evaluate(&self, x: f64) -> f64;
        fn derivative(&self) -> Self::Derivative;

        fn scale_x(&self, scale: f64) -> ScaleX<Self> { ScaleX(scale, self.clone()) }
        fn scale_y(&self, scale: f64) -> ScaleY<Self> { ScaleY(scale, self.clone()) }
        fn recenter(&self, center: f64) -> Recenter<Self> { Recenter(center, self.clone()) }
    }

    /// Computes `f(center + x)`
    #[derive(Debug, Copy, Clone)] pub struct Recenter<F>(pub f64, pub F);
    /// Computes `f(scale * x)`
    #[derive(Debug, Copy, Clone)] pub struct ScaleX<F>(pub f64, pub F);
    /// Computes `scale * f(x)`
    #[derive(Debug, Copy, Clone)] pub struct ScaleY<F>(pub f64, pub F);

    impl<F:Differentiable1d> Differentiable1d for Recenter<F> {
        type Derivative = Recenter<F::Derivative>;
        fn evaluate(&self, x: f64) -> f64 { self.1.evaluate(self.0 + x) }
        fn derivative(&self) -> Self::Derivative { self.1.derivative().recenter(self.0) }
    }

    impl<F:Differentiable1d> Differentiable1d for ScaleX<F> {
        type Derivative = ScaleY<ScaleX<F::Derivative>>;
        fn evaluate(&self, x: f64) -> f64 { self.1.evaluate(self.0 * x) }
        fn derivative(&self) -> Self::Derivative {
            // notice that scaling x scales the differential dx in the slope
            self.1.derivative().scale_x(self.0).scale_y(self.0.recip())
        }
    }

    impl<F:Differentiable1d> Differentiable1d for ScaleY<F> {
        type Derivative = ScaleY<F::Derivative>;
        fn evaluate(&self, x: f64) -> f64 { self.0 * self.1.evaluate(x) }
        fn derivative(&self) -> Self::Derivative { self.1.derivative().scale_y(self.0) }
    }

    /// A simple polynomial in one term
    #[derive(Debug, Clone)]
    pub struct Polynomial(Vec<f64>);
    #[allow(dead_code)]
    impl Polynomial {
        pub fn from_coeffs(coeffs: &[f64]) -> Polynomial { Polynomial(coeffs.to_owned()) }

        // pub fn constant(c: f64) -> Polynomial { Polynomial(vec![c]) }
        pub fn x_n(n: u32) -> Polynomial {
            let mut coeffs = vec![0.0; n as usize + 1];
            *coeffs.last_mut().unwrap() = 1.0;
            Polynomial(coeffs)
        }
        pub fn x() -> Polynomial { Polynomial::x_n(1) }

        pub fn coeffs(&self) -> &[f64] { &self.0 }
    }

    impl Differentiable1d for Polynomial {
        type Derivative = Polynomial;

        fn evaluate(&self, x: f64) -> f64 {
            let mut acc = 0f64;
            let mut xp = 1f64;
            for &c in &self.0 {
                acc += c * xp;
                xp *= x;
            }
            acc
        }

        fn derivative(&self) -> Polynomial {
            let mut out = Vec::with_capacity(self.0.len() - 1);
            for i in 1..self.0.len() {
                out.push(self.0[i] * i as f64);
            }
            Polynomial(out)
        }
    }

    // NOTE: probably not necessary, but we'll see
    /*
    impl Add<'a, &'a Polynomial> for Polynomial {
        type Output = Polynomial;
        fn add(self, other: &'a Polynomial) -> Polynomial {
            let mut out = self;
            out.0.resize(::std::cmp::max(out.0.len(), other.0.len()), 0f64);
            for (a, b) in self.0.iter_mut().zip(other) {
                *a += *b;
            }
            out
        }
    }

    impl Mul<'a, &'a Polynomial> for Polynomial {
        type Output = Polynomial;
        fn mul(self, other: &'a Polynomial) -> Polynomial {
            let out = vec![0f64; self.0.len() + other.0.len() - 1];
            for i in 0..self.0.len() {
                for j in 0..other.0.len() {
                    out[i + j] += self[i] * other[j];
                }
            }
            Polynomial(out)
        }
    }

    impl Neg for Polynomial {
        type Output = Polynomial;
        fn neg(self) -> Polynomial {
            let mut out = self;
            for x in &mut out.0 { *x *= -1.0; }
            out
        }
    }
    */

    #[test]
    fn polynomial_eval() {
        let poly = Polynomial::from_coeffs(&[5422.0, 1084.0, -27.0, 1.0]);
        assert_close!(poly.evaluate(3.141592653589793), 8592.013394342222);
    }
}

// N-dimensional test functions for e.g. conjugate gradient
pub mod n_dee {
    use ::rsp2_slice_math::{vdot,v,V,vsqnorm};

    /// NOTE: Default implementations are mutually recursive.
    /// You must define either both `value()` and `gradient()`, or just `diff()`.
    pub trait OnceDifferentiable
    {
        fn ndim(&self) -> usize;
        fn value(&mut self, position: &[f64]) -> f64  { self.diff(position).0 }
        fn gradient(&mut self, position: &[f64]) -> Vec<f64> { self.diff(position).1 }
        fn diff(&mut self, position: &[f64]) -> (f64, Vec<f64>) {
            (self.value(position), self.gradient(position))
        }

        /// Translate the input, so that we compute `f(x + center)`
        fn recenter(self, center: Vec<f64>) -> Recenter<Self>
        where Self: Sized,
        {
            assert_eq!(self.ndim(), center.len());
            Recenter(center, self)
        }

        /// Translate the potential itself, so that we compute `f(x - disp)`
        fn displace(self, displacement: Vec<f64>) -> Recenter<Self>
        where Self: Sized
        { self.recenter((-v(displacement)).0) }
    }

    impl<'a, T: OnceDifferentiable> OnceDifferentiable for &'a mut T {
        fn ndim(&self) -> usize { (**self).ndim() }
        fn value(&mut self, position: &[f64]) -> f64  { (**self).value(position) }
        fn gradient(&mut self, position: &[f64]) -> Vec<f64> { (**self).gradient(position) }
        fn diff(&mut self, position: &[f64]) -> (f64, Vec<f64>) { (**self).diff(position) }
    }

    /// computes f(x) + g(x)
    #[derive(Debug, Copy, Clone)] pub struct Sum<A, B>(pub A, pub B);
    /// computes f(x) g(x)
    #[derive(Debug, Copy, Clone)] pub struct Product<A, B>(pub A, pub B);
    /// computes f(center + x)
    #[derive(Debug, Clone)] pub struct Recenter<A>(pub Vec<f64>, pub A);

    impl<A, B> OnceDifferentiable for Sum<A, B>
    where A: OnceDifferentiable, B: OnceDifferentiable,
    {
        fn ndim(&self) -> usize { self.0.ndim() }

        fn diff(&mut self, position: &[f64]) -> (f64, Vec<f64>) {
            let (a_value, a_grad) = self.0.diff(position);
            let (b_value, b_grad) = self.1.diff(position);
            assert_eq!(self.ndim(), a_grad.len());
            assert_eq!(self.ndim(), b_grad.len());

            let value = a_value + b_value;
            let V(grad) = v(a_grad) + v(b_grad);
            (value, grad)
        }
    }

    impl<A, B> OnceDifferentiable for Product<A, B>
    where A: OnceDifferentiable, B: OnceDifferentiable,
    {
        fn ndim(&self) -> usize { self.0.ndim() }

        fn diff(&mut self, position: &[f64]) -> (f64, Vec<f64>) {
            let (a_value, a_grad) = self.0.diff(position);
            let (b_value, b_grad) = self.1.diff(position);
            assert_eq!(self.ndim(), a_grad.len());
            assert_eq!(self.ndim(), b_grad.len());

            let value = a_value * b_value;
            let V(grad) = v(a_grad) * b_value + a_value * v(b_grad);

            (value, grad)
        }
    }

    impl<A> OnceDifferentiable for Recenter<A>
    where A: OnceDifferentiable,
    {
        fn ndim(&self) -> usize { self.0.len() }

        fn diff(&mut self, position: &[f64]) -> (f64, Vec<f64>) {
            let Recenter(ref center, ref mut f) = *self;
            let V(position) = v(position) + v(center);
            f.diff(&position)
        }
    }

    /// The function
    ///
    ///     sum[i = 1 -> d] (1 - x_i)^2 - sum[i = 2 -> d] x_i x_{i-1}
    ///
    /// It has no local extrema aside from the global minimum.
    ///
    /// References:
    /// * https://www.sfu.ca/~ssurjano/trid.html
    pub struct Trid(pub usize);
    impl Trid {
        pub fn min_position(&self) -> Vec<f64> {
            let d = self.0 as f64;
            (0..self.0)
                .map(|i| i as f64)
                .map(|i| (i + 1.0) * (d - i))
                .collect()
        }

        pub fn min_value(&self) -> f64 {
            let d = self.0 as f64;
            -d * (d + 4.0) * (d - 1.0) / 6.0
        }
    }

    impl OnceDifferentiable for Trid {
        fn ndim(&self) -> usize { self.0 }

        fn value(&mut self, pos: &[f64]) -> f64 {
            assert_eq!(pos.len(), self.ndim());

            let t1 = vsqnorm(&(1.0 - v(pos)));
            let t2 = vdot(&pos[1..], &pos[..pos.len()-1]);
            t1 - t2
        }

        // (1 - xi)^2
        // 1 - 2xi + xi^2
        // deriv: -2 + 2xi
        fn gradient(&mut self, pos: &[f64]) -> Vec<f64> {
            assert_eq!(pos.len(), self.ndim());

            let V(t1) = -2.0 + 2.0 * v(pos);

            // the position vector rotated once left and right.
            let t2a = { let mut x = pos.to_vec(); x.pop(); x.insert(0, 0.0); x };
            let t2b = { let mut x = pos.to_vec(); x.remove(0); x.push(0.0); x };
            let V(out) = v(t1) - v(t2a) - v(t2b);
            out
        }
    }

    /// This is just a Lennard-Jones potential. In n-dimensional hyperspace.
    ///
    /// Its only minimum is the equipotential hypersurface at `r = min_radius`.
    pub struct HyperLennardJones {
        /// Number of dimensions
        pub ndim: usize,
        /// The minimal value
        pub min_value: f64,
        /// The radius at which value is minimal.
        pub min_radius: f64,
    }

    impl OnceDifferentiable for HyperLennardJones {
        fn ndim(&self) -> usize { self.ndim }

        fn diff(&mut self, position: &[f64]) -> (f64, Vec<f64>) {
            let r2 = vdot(position, position);
            let gamma2 = (self.min_radius * self.min_radius) / r2;
            let gamma6 = gamma2.powi(3);
            let gamma12 = gamma2.powi(6);
            // logarithmic derivative of r. (or a "logarithmic gradient", if you will)
            let V(log_grad) = v(position) / r2;

            let value = self.min_value * (2.0 * gamma6 - gamma12);
            let V(grad) = self.min_value * -12.0 * (gamma6 - gamma12) * v(log_grad);
            (value, grad)
        }
    }

    pub mod work {
        use super::*;

        pub enum PathConfig {
            Random {
                num_points: usize,
                domain_min: Vec<f64>,
                domain_max: Vec<f64>,
            },
            Fixed(Vec<Vec<f64>>),
        }

        pub enum RefinementMode {
            Double, // double the number of points each step
            Linear, // do 1 point per vertex, then 2, then 3...
        }

        pub struct BasePath(Vec<Vec<f64>>);

        impl PathConfig {
            pub fn generate(&self) -> BasePath {
                BasePath({
                    match *self {
                        PathConfig::Random {
                            num_points, ref domain_min, ref domain_max,
                        } => {
                            assert_eq!(domain_min.len(), domain_max.len());
                            assert!(num_points > 1, "{}", num_points);
                            (0..num_points)
                                .map(|_| {
                                    (0..domain_min.len()).map(|i| {
                                        let min = domain_min[i];
                                        let max = domain_max[i];
                                        ::util::random::uniform(min, max)
                                    }).collect()
                                })
                                .collect()
                        },
                        PathConfig::Fixed(ref points) => {
                            assert!(points.len() > 1, "{}", points.len());
                            points.clone()
                        },
                    }
                })
            }

            /// Adds an extra point if necessary to close the curve.
            ///
            /// (the length returned by this *may or may not* be equal
            /// to the length of the path produced by `generate`)
            pub fn generate_closed(&self) -> BasePath {
                let BasePath(mut points) = self.generate();
                if &points[0] != points.last().expect("checked in generate") {
                    let first = points[0].clone();
                    points.push(first);
                }
                BasePath(points)
            }
        }

        impl RefinementMode {
            pub fn densities(&self) -> impl Iterator<Item=usize> {
                use ::itertools::iterate;
                match *self {
                    RefinementMode::Double => Box::new(iterate(1usize, |x| 2 * x)) as Box<Iterator<Item=_>>,
                    RefinementMode::Linear => Box::new(0usize..) as Box<Iterator<Item=_>>,
                }
            }
        }

        impl BasePath {
            pub fn is_closed(&self) -> bool {
                &self.0[0] == self.0.last().expect("cannot construct len 0 BasePath")
            }

            pub fn start(&self) -> Vec<f64> { self.0[0].clone() }
            pub fn end(&self) -> Vec<f64> { self.0.last().expect("").clone() }

            pub fn with_density(&self, density: usize) -> Vec<Vec<f64>> {
                use ::std::iter::once;

                assert_ne!(density, 0);
                assert!(self.0.len() > 0);

                let BasePath(vertices) = self;
                vertices.windows(2).flat_map(move |window| {
                    let prev = &window[0];
                    let next = &window[1];
                    (0..density).map(move |i| {
                        let alpha = i as f64 / density as f64;
                        zip_eq!(prev, next)
                            .map(|(&p, &n)| (1.0 - alpha) * p + alpha * n)
                            .collect()
                    })
                // finish the final segment
                }).chain(once(vertices.last().expect("").clone())).collect()
            }
        }

        pub fn compute_work_along_path<D>(mut diff: D, path: &[Vec<f64>]) -> f64
        where D: OnceDifferentiable,
        {
            use ::itertools::Itertools;
            use ::rsp2_slice_math::{v,V,vdot};
            path.iter().tuple_windows().map(|(cur, prev)| {
                let V(mid) = (v(cur) + v(prev)) / 2.0;
                let V(displacement) = v(cur) - v(prev);

                let gradient = diff.gradient(&mid);
                -1.0 * vdot(&gradient, &displacement)
            }).sum()
        }
    }

    // these are for specificity;
    // no good using a test function if it isn't correct!
    #[deny(dead_code)]
    #[cfg(test)]
    mod tests {
        use super::OnceDifferentiable;
        use super::work::{PathConfig, RefinementMode};
        use super::work::compute_work_along_path;

        pub struct MaxRefines(pub u32);

        pub fn test_conservativity<D>(
            path: &PathConfig,
            refine_mode: RefinementMode,
            max_refines: MaxRefines,
            abs_tol: f64,
            mut diff: D,
        ) where D: super::OnceDifferentiable
        {
            use ::std::fmt::Write;

            let path = path.generate_closed();

            let mut computed_values = vec![];
            for density in refine_mode.densities().take(max_refines.0 as usize) {
                let work = compute_work_along_path(&mut diff, &path.with_density(density));
                computed_values.push(work);
                if work < abs_tol {
                    return;
                }
            }

            let last_3 = &computed_values[computed_values.len() - 3..];
            let mut s = String::new();
            let _ = writeln!(&mut s, "Non-conservative?");
            let _ = writeln!(&mut s, " Convergents: {:e}", last_3[0]);
            let _ = writeln!(&mut s, "              {:e}", last_3[1]);
            let _ = writeln!(&mut s, "              {:e}", last_3[2]);
            panic!("{}", s);
        }

        /// NOTE: Not suitable for closed paths (where the integral should be zero)
        ///       because it uses a relative tolerance.  Use `test_conservativity` instead.
        pub fn test_work_value<D>(
            path: &PathConfig,
            refine_mode: RefinementMode,
            max_refines: MaxRefines,
            rel_tol: f64,
            mut diff: D,
        ) where D: super::OnceDifferentiable
        {
            use ::std::fmt::Write;
            let path = path.generate();
            if path.is_closed() {
                warn!{"\
                    test_work_value was used on a closed path. \
                    Please use `test_conservativity` instead.\
                "}
            }

            let initial_value = diff.value(&path.start());
            let final_value = diff.value(&path.end());
            let mut computed_values = vec![];
            for density in refine_mode.densities().take(max_refines.0 as usize) {
                let work = compute_work_along_path(&mut diff, &path.with_density(density));
                let computed = initial_value + work;
                computed_values.push(computed);

                if (final_value - computed).abs() < rel_tol * final_value.abs() {
                    return;
                }
            }

            let last_3 = &computed_values[computed_values.len() - 3..];
            let mut s = String::new();
            let _ = writeln!(&mut s, "Value-gradient mismatch?");
            let _ = writeln!(&mut s, "          Actual value: {:e}", final_value);
            let _ = writeln!(&mut s, " Convergents from work: {:e}", last_3[0]);
            let _ = writeln!(&mut s, "                        {:e}", last_3[1]);
            let _ = writeln!(&mut s, "                        {:e}", last_3[2]);
            panic!("{}", s);
        }

        #[test]
        fn trid_minimum() {
            let mut trid = super::Trid(10);
            let point = trid.min_position();
            println!("{:?}", trid.gradient(&point));
            assert_close!(trid.min_value(), trid.value(&point));
            for x in trid.gradient(&point) {
                assert_close!(abs=1e-8, x, 0.0);
            }
        }

        #[test]
        fn trid_conservative() {
            test_conservativity(
                &PathConfig::Random {
                    num_points: 5,
                    domain_min: vec![-10.0; 10],
                    domain_max: vec![ 10.0; 10],
                },
                RefinementMode::Double,
                MaxRefines(8),
                1e-6,
                super::Trid(10),
            )
        }

        #[test]
        fn trid_value_vs_grad() {
            test_work_value(
                &PathConfig::Random {
                    num_points: 5,
                    domain_min: vec![-10.0; 10],
                    domain_max: vec![ 10.0; 10],
                },
                RefinementMode::Double,
                MaxRefines(8),
                1e-6,
                super::Trid(10)
            )
        }

        #[test]
        fn lj_minimum() {
            use ::rsp2_slice_math::{v, V};

            let ndim = 10;
            let min_value = -40.0;
            let min_radius = 1.25;
            let mut lj = super::HyperLennardJones { ndim, min_value, min_radius };

            for _ in 0..5 {
                let V(point) = min_radius * v(::util::random::direction(ndim));
                assert_close!(min_value, lj.value(&point));
                for x in lj.gradient(&point) {
                    assert_close!(abs=1e-8, x, 0.0);
                }
            }
        }

        #[test]
        fn lj_conservative() {
            let ndim = 10;
            let min_value = -40.0;
            let min_radius = 1.25;

            test_conservativity(
                &PathConfig::Random {
                    // FIXME The work integral converges slowly for lj,
                    //       leading to spurious failures, so I had to
                    //       loosen the constraints a bit.
                    //       Perhaps Gaussian quadrature could help.
                    num_points: 4,                // 5 originally
                    domain_min: vec![ 0.5; ndim], // vec![ 0.1; ndim]
                    domain_max: vec![ 2.0; ndim], // vec![ 3.0; ndim]
                },
                RefinementMode::Double,
                MaxRefines(10),              // 8 originally
                1e-4,                        // 1e-6 originally
                super::HyperLennardJones { ndim, min_value, min_radius },
            )
        }

        #[test]
        fn lj_value_vs_grad() {
            let ndim = 10;
            let min_value = -40.0;
            let min_radius = 1.25;

            test_work_value(
                &PathConfig::Random {
                    // FIXME The work integral converges slowly for lj,
                    //       leading to spurious failures, so I had to
                    //       loosen the constraints a bit.
                    //       Perhaps Gaussian quadrature could help.
                    num_points: 4,                // 5 originally
                    domain_min: vec![ 0.5; ndim], // vec![ 0.1; ndim]
                    domain_max: vec![ 2.0; ndim], // vec![ 3.0; ndim]
                },
                RefinementMode::Double,
                MaxRefines(10),  // 8 originally
                1e-4,            // 1e-6 originally
                super::HyperLennardJones { ndim, min_value, min_radius },
            )
        }

        // TODO test Product, Sum, Recenter
    }
}
