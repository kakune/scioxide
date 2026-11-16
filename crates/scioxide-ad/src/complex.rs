//! Elementary functions of complex values whose parts are number types.

use scioxide_core::{Complex, RealScalar};

use crate::{BaseReal, Real};

/// Elementary functions of [`Complex<R>`] for any number type `R`, named as the
/// inherent methods of `Complex<f64>`.
///
/// `+`, `−` and `×` of `Complex<R>` are already generic; division is
/// `z * w.inv()`. Each function is written in `R` arithmetic on the real and
/// imaginary parts, so derivatives are those of the parts (real-linear, not
/// holomorphic derivatives), in every mode. Branch cuts are the principal ones of
/// `Complex<f64>`, and on the negative real axis the sign of a zero imaginary part
/// picks the side: `ln` and `sqrt` take the lower side for `−0`. `inv` scales by
/// `max(|x|, |y|)` computed on base values, so it records no kinks; `ln`, `arg` and
/// `norm` go through `hypot` and `atan2` and count their kinks at the origin and
/// on the negative real axis.
///
/// ```
/// use scioxide_ad::ComplexElementary;
/// use scioxide_core::Complex;
/// let below = ComplexElementary::sqrt(Complex::new(-4.0_f64, -0.0));
/// assert_eq!((below.re, below.im), (0.0, -2.0));
/// assert_eq!(ComplexElementary::sqrt(Complex::new(-4.0_f64, 0.0)).im, 2.0);
/// ```
///
/// # Examples
///
/// ```
/// use scioxide_ad::{ComplexElementary, Jet};
/// use scioxide_core::Complex;
///
/// // exp((0.3 + s) + 2s·i) as a Taylor series in s: cₖ = e^0.3·(1 + 2i)ᵏ/k!.
/// let z = Complex::new(Jet::<f64, 4>::new([0.3, 1.0, 0.0, 0.0]), Jet::new([0.0, 2.0, 0.0, 0.0]));
/// let w = z.exp();
/// let e = 0.3_f64.exp();
/// assert!((w.re.coefficients()[2] - e * -1.5).abs() < 1e-14);
/// assert!((w.im.coefficients()[2] - e * 2.0).abs() < 1e-14);
/// ```
pub trait ComplexElementary<S: RealScalar = f64>: Sized {
    /// The type of the real and imaginary parts.
    type Part;
    /// Returns `1/z = z̄/|z|²`, scaled by the larger part against overflow.
    #[must_use]
    fn inv(self) -> Self;
    /// Returns `eᶻ = eˣ(cos y + i sin y)`.
    #[must_use]
    fn exp(self) -> Self;
    /// Returns the principal logarithm `ln |z| + i arg z`.
    #[must_use]
    fn ln(self) -> Self;
    /// Returns the principal square root, with non-negative real part.
    #[must_use]
    fn sqrt(self) -> Self;
    /// Returns `sin z = sin x cosh y + i cos x sinh y`.
    #[must_use]
    fn sin(self) -> Self;
    /// Returns `cos z = cos x cosh y − i sin x sinh y`.
    #[must_use]
    fn cos(self) -> Self;
    /// Returns `|z| = hypot(x, y)`.
    #[must_use]
    fn norm(self) -> Self::Part;
    /// Returns `arg z = atan2(y, x) ∈ [−π, π]`.
    #[must_use]
    fn arg(self) -> Self::Part;
}

impl<S: RealScalar, R: Real<S>> ComplexElementary<S> for Complex<R> {
    type Part = R;

    fn inv(self) -> Self {
        // The scale cancels exactly, so it is taken from base values and not
        // differentiated: the derivatives are those of `z̄/|z|²` and no kink of `abs`
        // or `max` is counted.
        let scale = R::passive(self.re.base().abs().max(self.im.base().abs()));
        let (re, im) = (self.re / scale, self.im / scale);
        // `norm ∈ [1, 2]`; dividing by the scale last avoids overflow near `MAX`.
        let norm = re * re + im * im;
        Complex::new(re / norm / scale, -im / norm / scale)
    }

    fn exp(self) -> Self {
        let magnitude = self.re.exp();
        let (sin, cos) = self.im.sin_cos();
        Complex::new(magnitude * cos, magnitude * sin)
    }

    fn ln(self) -> Self {
        let (x, y) = (self.re, self.im);
        // Near |z| = 1, `ln|z|` cancels: take `½·ln(1 + (x² + y² − 1))` with the
        // larger part's square formed as `(m − 1)(m + 1)`. The branch is chosen on
        // base values, so it counts no kinks.
        let (xb, yb) = (x.base().abs(), y.base().abs());
        let r = xb.hypot(yb);
        let near = r.ge(R::Base::constant(S::HALF)) & r.le(R::Base::constant(S::TWO));
        let shifted = R::select(
            xb.ge(yb),
            (x - S::ONE) * (x + S::ONE) + y * y,
            x * x + (y - S::ONE) * (y + S::ONE),
        );
        let re = R::select(near, shifted.ln_1p() * S::HALF, x.hypot(y).ln());
        // The rounded squares still erase a residual near |z| = 1: the value is that
        // of `Complex::ln` of the base lanes, which sums the products exactly, and the
        // formula above gives the derivatives.
        let exact = R::Base::from_fn(|k| {
            let lane = |v: R::Base| v.lane(k).unwrap_or(S::NAN);
            Complex::new(lane(x.base()), lane(y.base())).ln().re
        });
        let correction = exact - re.base();
        let finite = correction.is_finite();
        let re = re + R::passive(R::Base::select(finite, correction, R::Base::zero()));
        Complex::new(re, y.atan2(x))
    }

    fn sqrt(self) -> Self {
        // For x ≥ 0: a = √((|z| + x)/2), b = y/(2a); for x < 0 the roles swap, so
        // neither part cancels.
        let r = self.re.hypot(self.im);
        let half = S::HALF;
        let zero = R::zero();
        let a = ((r + self.re) * half).sqrt();
        let b = ((r - self.re) * half).sqrt();
        // The sign of b follows the sign bit of y, so y = −0 on the negative real
        // axis gives the lower side, as `Complex<f64>::sqrt` and `ln`.
        let y = self.im.base();
        let zero_base = <R::Base as Real<S>>::zero();
        let below = y.lt(zero_base) | (<R::Base as Real<S>>::one() / y).lt(zero_base);
        let b = R::select(below, -b, b);
        let nonnegative = self.re.ge(zero);
        let re = R::select(nonnegative, a, self.im / (b * S::TWO));
        let im = R::select(nonnegative, self.im / (a * S::TWO), b);
        let origin = r.le(zero);
        Complex::new(R::select(origin, zero, re), R::select(origin, self.im, im))
    }

    fn sin(self) -> Self {
        let (sin, cos) = self.re.sin_cos();
        Complex::new(sin * self.im.cosh(), cos * self.im.sinh())
    }

    fn cos(self) -> Self {
        let (sin, cos) = self.re.sin_cos();
        Complex::new(cos * self.im.cosh(), -(sin * self.im.sinh()))
    }

    fn norm(self) -> R {
        self.re.hypot(self.im)
    }

    fn arg(self) -> R {
        self.im.atan2(self.re)
    }
}
