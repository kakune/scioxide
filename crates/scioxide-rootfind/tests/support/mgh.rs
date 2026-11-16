//! MINPACK's fourteen square systems, differentiated by test-local dual arithmetic.
use core::cell::Cell;
use core::ops::{Add, Div, Mul, Neg, Sub};
use scioxide_core::{JacobianOperator, NumericScalar, RealScalar, ResidualOperator, SciOxideError};

use scioxide_rootfind::DenseJacobian;

pub fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
#[derive(Clone, Copy)]
struct Jet<S> {
    v: S,
    d: S,
}
impl<S: RealScalar> Jet<S> {
    fn c(x: f64) -> Self {
        Self {
            v: r(x),
            d: S::ZERO,
        }
    }
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::c(1.0);
        }
        Self {
            v: self.v.powi(n),
            d: r::<S>(n as f64) * self.v.powi(n - 1) * self.d,
        }
    }
    fn exp(self) -> Self {
        Self {
            v: self.v.exp(),
            d: self.v.exp() * self.d,
        }
    }
    fn sin(self) -> Self {
        Self {
            v: self.v.sin(),
            d: self.v.cos() * self.d,
        }
    }
    fn cos(self) -> Self {
        Self {
            v: self.v.cos(),
            d: -self.v.sin() * self.d,
        }
    }
    fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        Self {
            v,
            d: self.d / (r::<S>(2.0) * v),
        }
    }
    fn atan(self) -> Self {
        Self {
            v: self.v.atan(),
            d: self.d / (S::ONE + self.v * self.v),
        }
    }
}
impl<S: RealScalar> Add for Jet<S> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: self.d + b.d,
        }
    }
}
impl<S: RealScalar> Sub for Jet<S> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            v: self.v - b.v,
            d: self.d - b.d,
        }
    }
}
impl<S: RealScalar> Mul for Jet<S> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: self.d * b.v + self.v * b.d,
        }
    }
}
impl<S: RealScalar> Div for Jet<S> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        Self {
            v: self.v / b.v,
            d: (self.d - (self.v / b.v) * b.d) / b.v,
        }
    }
}
impl<S: RealScalar> Neg for Jet<S> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: -self.d,
        }
    }
}
fn equations<S: RealScalar>(p: usize, x: &[Jet<S>], f: &mut [Jet<S>]) {
    let c = Jet::<S>::c;
    let n = x.len();
    let one = c(1.0);
    let two = c(2.0);
    f.fill(c(0.0));
    match p {
        1 => {
            f[0] = one - x[0];
            f[1] = c(10.0) * (x[1] - x[0] * x[0]);
        }
        2 => {
            f[0] = x[0] + c(10.0) * x[1];
            f[1] = c(5.0).sqrt() * (x[2] - x[3]);
            f[2] = (x[1] - two * x[2]).powi(2);
            f[3] = c(10.0).sqrt() * (x[0] - x[3]).powi(2);
        }
        3 => {
            f[0] = c(1e4) * x[0] * x[1] - one;
            f[1] = (-x[0]).exp() + (-x[1]).exp() - c(1.0001);
        }
        4 => {
            let a = x[1] - x[0] * x[0];
            let b = x[3] - x[2] * x[2];
            f[0] = -c(200.0) * x[0] * a - (one - x[0]);
            f[1] = c(200.0) * a + c(20.2) * (x[1] - one) + c(19.8) * (x[3] - one);
            f[2] = -c(180.0) * x[2] * b - (one - x[2]);
            f[3] = c(180.0) * b + c(20.2) * (x[3] - one) + c(19.8) * (x[1] - one);
        }
        5 => {
            let theta = if x[0].v > S::ZERO {
                (x[1] / x[0]).atan() / c(2.0 * core::f64::consts::PI)
            } else if x[0].v < S::ZERO {
                (x[1] / x[0]).atan() / c(2.0 * core::f64::consts::PI) + c(0.5)
            } else {
                c(if x[1].v < S::ZERO { -0.25 } else { 0.25 })
            };
            f[0] = c(10.0) * (x[2] - c(10.0) * theta);
            f[1] = c(10.0) * ((x[0] * x[0] + x[1] * x[1]).sqrt() - one);
            f[2] = x[2];
        }
        6 => {
            for i in 1..=29 {
                let t = c(i as f64 / 29.0);
                let sum1 = x.iter().enumerate().skip(1).fold(c(0.0), |a, (j, &v)| {
                    a + c(j as f64) * t.powi(j as i32 - 1) * v
                });
                let sum2 = x
                    .iter()
                    .enumerate()
                    .fold(c(0.0), |a, (j, &v)| a + t.powi(j as i32) * v);
                let a = sum1 - sum2 * sum2 - one;
                for (k, out) in f.iter_mut().enumerate() {
                    *out = *out + t.powi(k as i32 - 1) * (c(k as f64) - two * t * sum2) * a;
                }
            }
            let a = x[1] - x[0] * x[0] - one;
            f[0] = f[0] + x[0] * (one - two * a);
            f[1] = f[1] + a;
        }
        7 => {
            for &v in x {
                let z = two * v - one;
                let mut previous = one;
                let mut current = z;
                for out in f.iter_mut() {
                    *out = *out + current;
                    let next = two * z * current - previous;
                    previous = current;
                    current = next;
                }
            }
            for (i, out) in f.iter_mut().enumerate() {
                *out = *out / c(n as f64);
                if (i + 1) % 2 == 0 {
                    *out = *out + one / c(((i + 1) * (i + 1) - 1) as f64);
                }
            }
        }
        8 => {
            let sum = x.iter().copied().fold(-c((n + 1) as f64), |a, b| a + b);
            for (out, &v) in f.iter_mut().zip(x) {
                *out = v + sum;
            }
            f[n - 1] = x.iter().copied().fold(one, |a, b| a * b) - one;
        }
        9 | 10 => {
            let h = one / c((n + 1) as f64);
            for (i, out) in f.iter_mut().enumerate() {
                let t = c((i + 1) as f64) * h;
                if p == 9 {
                    *out = two * x[i] + h * h / c(2.0) * (x[i] + t + one).powi(3);
                    if i > 0 {
                        *out = *out - x[i - 1];
                    }
                    if i + 1 < n {
                        *out = *out - x[i + 1];
                    }
                } else {
                    let (left, right) =
                        x.iter()
                            .enumerate()
                            .fold((c(0.0), c(0.0)), |(a, b), (j, &v)| {
                                let tj = c((j + 1) as f64) * h;
                                let value = (v + tj + one).powi(3);
                                if j <= i {
                                    (a + tj * value, b)
                                } else {
                                    (a, b + (one - tj) * value)
                                }
                            });
                    *out = x[i] + h / c(2.0) * ((one - t) * left + t * right);
                }
            }
        }
        11 => {
            let sum = x.iter().copied().fold(c(0.0), |a, b| a + b.cos());
            for (i, out) in f.iter_mut().enumerate() {
                *out = c((n + i + 1) as f64) - x[i].sin() - sum - c((i + 1) as f64) * x[i].cos();
            }
        }
        12 => {
            let sum = x
                .iter()
                .enumerate()
                .fold(c(0.0), |a, (i, &v)| a + c((i + 1) as f64) * (v - one));
            for (i, out) in f.iter_mut().enumerate() {
                *out = x[i] - one + c((i + 1) as f64) * sum * (one + two * sum * sum);
            }
        }
        13 => {
            for (i, out) in f.iter_mut().enumerate() {
                *out = (c(3.0) - two * x[i]) * x[i] + one;
                if i > 0 {
                    *out = *out - x[i - 1];
                }
                if i + 1 < n {
                    *out = *out - two * x[i + 1];
                }
            }
        }
        14 => {
            for (i, out) in f.iter_mut().enumerate() {
                *out = x[i] * (two + c(5.0) * x[i] * x[i]) + one;
                for (j, &v) in x
                    .iter()
                    .enumerate()
                    .take((i + 2).min(n))
                    .skip(i.saturating_sub(5))
                {
                    if j != i {
                        *out = *out - v * (one + v);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}

pub struct Mgh {
    pub problem: usize,
    pub n: usize,
    pub residual_calls: Cell<usize>,
    pub jacobian_calls: Cell<usize>,
    pub jvp_calls: Cell<usize>,
}
impl Mgh {
    pub fn new(problem: usize, n: usize) -> Self {
        Self {
            problem,
            n,
            residual_calls: Cell::new(0),
            jacobian_calls: Cell::new(0),
            jvp_calls: Cell::new(0),
        }
    }
    pub fn reset_counts(&self) {
        self.residual_calls.set(0);
        self.jacobian_calls.set(0);
        self.jvp_calls.set(0);
    }
    pub fn values<S: RealScalar>(&self, x: &[S], out: &mut [S]) {
        self.residual_calls.set(self.residual_calls.get() + 1);
        self.apply(x, None, out);
    }
    fn apply<S: RealScalar>(&self, x: &[S], direction: Option<&[S]>, out: &mut [S]) {
        let input: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| Jet {
                v,
                d: direction.map_or(S::ZERO, |d| d[i]),
            })
            .collect();
        let mut output = vec![Jet::c(0.0); self.n];
        equations(self.problem, &input, &mut output);
        for (out, j) in out.iter_mut().zip(output) {
            *out = if direction.is_some() { j.d } else { j.v };
        }
    }
    pub fn jacobian<S: RealScalar>(&self, x: &[S], out: &mut [S]) {
        self.jacobian_calls.set(self.jacobian_calls.get() + 1);
        let mut d = vec![S::ZERO; self.n];
        let mut column = vec![S::ZERO; self.n];
        for j in 0..self.n {
            d[j] = S::ONE;
            self.apply(x, Some(&d), &mut column);
            for (i, &v) in column.iter().enumerate() {
                out[i * self.n + j] = v;
            }
            d[j] = S::ZERO;
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> DenseJacobian<S> for Mgh {
    fn jacobian_into(&self, x: &[S], out: &mut [S], _: &mut [S]) -> Result<bool, SciOxideError> {
        self.jacobian(x, out);
        Ok(true)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> ResidualOperator<S> for Mgh {
    fn input_dimension(&self) -> usize {
        self.n
    }
    fn output_dimension(&self) -> usize {
        self.n
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[S], out: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        self.values(x, out);
        Ok(())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> JacobianOperator<S> for Mgh {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(&self, x: &[S], d: &[S], out: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        self.jvp_calls.set(self.jvp_calls.get() + 1);
        self.apply(x, Some(d), out);
        Ok(())
    }
}
