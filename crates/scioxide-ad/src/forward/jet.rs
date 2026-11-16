//! Truncated Taylor series over any number type.

use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

use scioxide_core::{RealScalar, SciOxideError};

use super::series;
use crate::Real;
use crate::real::sealed::{self, Token};

/// The first `N` normalized Taylor coefficients of a function of one variable `t`,
/// `cₖ = (1/k!) dᵏx/dtᵏ` at `t = 0`, so that `x(t) = Σ_{k<N} cₖ tᵏ + O(tᴺ)`.
///
/// Arithmetic on jets is univariate Taylor arithmetic: every [`Real`] operation
/// returns the coefficients of the composed series through order `N − 1`. The
/// derivatives are `dᵏx/dtᵏ = k!·cₖ`; `Jet<V, 3>` carries a value, a first and a
/// second derivative (`2·c₂`). `c₀` is always the operation on `V`, bit for bit,
/// and `c₁` is the [`Dual`](crate::Dual) derivative, formed from the same partial
/// with an IEEE product: for a base `V` (`f64`, `Lanes`) it differs only in the
/// sign of a zero and where an infinite or NaN partial meets a zero coefficient,
/// which gives NaN here (so `sqrt(max(v, 0))` at `v < 0` has `c₁ = NaN`). For a
/// nested `V` such as `Dual<f64>` the partial is formed in `V` arithmetic, and the
/// derivative parts agree with `Dual<Dual<f64>>` to rounding. `V` is any [`Real`]:
/// `Jet<Lanes<f64, 8>, N>` expands eight lanes at once, and a nested jet
/// `Jet<Jet<V, M>, N>` holds mixed coefficients `c[i][j] = (1/(i! j!))
/// ∂ⁱ⁺ʲx/∂sⁱ∂tʲ` with `s` the outer and `t` the inner variable. The storage is
/// plain `Copy` data, `N` values of `V`; `N ≥ 1` is checked at compile time.
///
/// # Method and cost
///
/// Products and quotients are Cauchy products; `exp`, `ln`, `sqrt`, `cbrt`,
/// `recip`, `powi`, `powf` and their scaled forms use Euler-weighted recurrences
/// (Miller's recurrence for powers); `sin`/`cos` and `sinh`/`cosh` are coupled
/// pairs; `tan`, `tanh` and `sigmoid` run auxiliary series; inverse functions
/// solve `u·y′ = ±x′`; `hypot` follows its direction cosines. A nonlinear
/// operation costs `O(N²)` operations on `V`, a linear one `O(N)`. All products
/// are IEEE products with a separate multiply and add, and sums run in index
/// order, so results do not depend on FMA support.
///
/// Kinks follow the convention of the crate coefficient by coefficient: `abs`
/// multiplies every coefficient by `sign x₀` (0 at `x₀ = 0`); `min`, `max` and
/// `clamp` take the coefficients of the selected operand and average them at a tie
/// of `c₀`; rounding functions and `signum` are constant above `c₀`; `fract` and
/// `%` follow from their definitions. `powf` with an exponent whose coefficients
/// above `c₀` are all zero admits negative bases; at a zero base it has the
/// coefficients of `powi` for an integer exponent `y₀ ≥ 0`, and otherwise
/// coefficient `k ≥ 2` is 0 for `k < y₀` and NaN from `y₀` on; with a varying
/// exponent a zero base gives NaN above `c₁`. For a nested `V` these zero-base
/// cases need every component of `x₀` and `y₀` to be zero or constant; otherwise a
/// coefficient can be NaN. `atan2` and [`Real::combine`] are
/// exact to first order and `exprel`, `sinc`, `sincpi` and [`Real::map_scalar`] to
/// second order; their coefficients above are NaN.
///
/// # Accuracy
///
/// On arguments and coefficients that are zero or of magnitude in
/// `[√MIN_POSITIVE, √MAX]`, coefficient `k` has absolute error at most
/// `64 ε (k + 1) g` plus the smallest subnormal, with `g` the sum of the absolute
/// values of the terms of the exact composition, for `f32` and `f64` through order
/// 8 and for mixed coefficients through total order 8. Outside that range plain
/// IEEE evaluation can overflow or underflow.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Jet, Real};
/// // exp(sin t) = 1 + t + t²/2 + 0·t³ + O(t⁴)
/// let t = Jet::<f64, 4>::variable(0.0);
/// let c = t.sin().exp().coefficients();
/// for (actual, expected) in c.into_iter().zip([1.0, 1.0, 0.5, 0.0]) {
///     assert!((actual - expected).abs() < 1e-15);
/// }
/// // Second derivative of x³ at x = 2: 2·c₂ = 12.
/// let y = Jet::<f64, 3>::variable(2.0).powi(3);
/// assert!((2.0 * y.coefficients()[2] - 12.0).abs() < 1e-13);
/// ```
///
/// # References
///
/// - A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008,
///   ch. 13 (Taylor arithmetic).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jet<V, const N: usize> {
    coefficients: [V; N],
}

impl<V: Copy, const N: usize> Jet<V, N> {
    /// Builds the series with normalized coefficients `coefficients`.
    #[must_use]
    #[inline(always)]
    pub const fn new(coefficients: [V; N]) -> Self {
        const { assert!(N >= 1, "a jet has at least one coefficient") };
        Self { coefficients }
    }
    /// Returns the normalized coefficients `c₀, …, c_{N−1}`.
    #[must_use]
    #[inline(always)]
    pub const fn coefficients(self) -> [V; N] {
        self.coefficients
    }
    /// Returns `cₖ`, or `None` for `k ≥ N`.
    #[must_use]
    #[inline(always)]
    pub fn coefficient(self, k: usize) -> Option<V> {
        self.coefficients.get(k).copied()
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Jet<V, N> {
    /// Builds the independent variable `value + t`: `c₀ = value`, `c₁ = 1` and
    /// zero above.
    #[must_use]
    #[inline(always)]
    pub fn variable(value: V) -> Self {
        let mut coefficients = series::lift(value);
        if N > 1 {
            coefficients[1] = V::one();
        }
        Self::new(coefficients)
    }
}

impl<V: sealed::Value, const N: usize> sealed::Value for Jet<V, N> {
    type Elem = V::Elem;
}

impl<S: RealScalar, V: Real<S>, const N: usize> sealed::Map<S> for Jet<V, N> {
    #[inline]
    fn map_all<F: FnMut(S) -> [S; 3]>(token: Token, x: Self, rule: &mut F) -> [Self; 3] {
        let derivatives = V::map_all(token, x.coefficients[0], rule);
        series::map_all(&x.coefficients, derivatives).map(Self::new)
    }

    #[inline]
    fn select_lanes<T, P: Fn(&T) -> Self, I: FnMut(usize) -> usize>(
        token: Token,
        table: &[T],
        project: &P,
        index: &mut I,
    ) -> Self {
        Self::new(core::array::from_fn(|k| {
            V::select_lanes(token, table, &|entry| project(entry).coefficients[k], index)
        }))
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Add for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(core::array::from_fn(|k| {
            self.coefficients[k] + other.coefficients[k]
        }))
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Sub for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(core::array::from_fn(|k| {
            self.coefficients[k] - other.coefficients[k]
        }))
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Mul for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self::new(series::mul(&self.coefficients, &other.coefficients))
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Div for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        Self::new(series::div(&self.coefficients, &other.coefficients))
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Rem for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: Self) -> Self {
        Self::new(series::rem(&self.coefficients, &other.coefficients))
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Neg for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(self.coefficients.map(|c| -c))
    }
}

macro_rules! assign {
    ($($Assign:ident $assign:ident $Trait:ident $method:ident),*) => {
        $(
            impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> $Assign
                for Jet<V, N>
            {
                #[inline(always)]
                fn $assign(&mut self, other: Self) {
                    *self = $Trait::$method(*self, other);
                }
            }
            impl<S: RealScalar, V: Real<S>, const N: usize> $Assign<S> for Jet<V, N> {
                #[inline(always)]
                fn $assign(&mut self, other: S) {
                    *self = $Trait::$method(*self, other);
                }
            }
        )*
    };
}

assign!(
    AddAssign add_assign Add add,
    SubAssign sub_assign Sub sub,
    MulAssign mul_assign Mul mul,
    DivAssign div_assign Div div
);

impl<V: Copy, const N: usize> Jet<V, N> {
    /// Replaces `c₀` by `value(c₀)`.
    #[inline(always)]
    fn with_value(self, value: impl FnOnce(V) -> V) -> Self {
        let mut coefficients = self.coefficients;
        coefficients[0] = value(coefficients[0]);
        Self::new(coefficients)
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Add<S> for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: S) -> Self {
        self.with_value(|c| c + other)
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Sub<S> for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: S) -> Self {
        self.with_value(|c| c - other)
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Mul<S> for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: S) -> Self {
        Self::new(self.coefficients.map(|c| c * other))
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Div<S> for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: S) -> Self {
        Self::new(self.coefficients.map(|c| c / other))
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Rem<S> for Jet<V, N> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: S) -> Self {
        self.with_value(|c| c % other)
    }
}

macro_rules! scalar_left {
    ($($S:ty),*) => {
        $(
            impl<V: Real<$S>, const N: usize> Add<Jet<V, N>> for $S {
                type Output = Jet<V, N>;
                #[inline(always)]
                fn add(self, other: Jet<V, N>) -> Jet<V, N> {
                    other.with_value(|c| V::constant(self) + c)
                }
            }
            impl<V: Real<$S>, const N: usize> Sub<Jet<V, N>> for $S {
                type Output = Jet<V, N>;
                #[inline(always)]
                fn sub(self, other: Jet<V, N>) -> Jet<V, N> {
                    let mut coefficients = other.coefficients.map(|c| -c);
                    coefficients[0] = V::constant(self) - other.coefficients[0];
                    Jet::new(coefficients)
                }
            }
            impl<V: Real<$S>, const N: usize> Mul<Jet<V, N>> for $S {
                type Output = Jet<V, N>;
                #[inline(always)]
                fn mul(self, other: Jet<V, N>) -> Jet<V, N> {
                    other * self
                }
            }
            impl<V: Real<$S>, const N: usize> Div<Jet<V, N>> for $S {
                type Output = Jet<V, N>;
                #[inline(always)]
                fn div(self, other: Jet<V, N>) -> Jet<V, N> {
                    Jet::constant(self) / other
                }
            }
            impl<V: Real<$S>, const N: usize> Rem<Jet<V, N>> for $S {
                type Output = Jet<V, N>;
                #[inline(always)]
                fn rem(self, other: Jet<V, N>) -> Jet<V, N> {
                    Jet::constant(self) % other
                }
            }
        )*
    };
}

scalar_left!(f32, f64);

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Sum for Jet<V, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = Self::new([V::constant(S::NEG_ZERO); N]);
        iter.fold(zero, |sum, value| sum + value)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Product for Jet<V, N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |product, value| product * value)
    }
}

/// Emits one unary method per elementary function from [`series`].
macro_rules! jet_unary_methods {
    ($($(#[$doc:meta])* $name:ident,)*) => {
        $(
            #[inline]
            fn $name(self) -> Self {
                crate::forward::jet::jet_unary_method!($name, self)
            }
        )*
    };
}

macro_rules! jet_unary_method {
    (exprel, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Exprel)
    };
    (sinc, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Sinc)
    };
    (sincpi, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Sincpi)
    };
    (sin, $x:ident) => {
        Self::new(crate::forward::series::sin_cos(&$x.coefficients).0)
    };
    (cos, $x:ident) => {
        Self::new(crate::forward::series::sin_cos(&$x.coefficients).1)
    };
    (sinh, $x:ident) => {
        Self::new(crate::forward::series::sinh_cosh(&$x.coefficients).0)
    };
    (cosh, $x:ident) => {
        Self::new(crate::forward::series::sinh_cosh(&$x.coefficients).1)
    };
    (asin, $x:ident) => {
        crate::forward::jet::jet_unary_method!(@inverse Asin, $x)
    };
    (acos, $x:ident) => {
        crate::forward::jet::jet_unary_method!(@inverse Acos, $x)
    };
    (atan, $x:ident) => {
        crate::forward::jet::jet_unary_method!(@inverse Atan, $x)
    };
    (asinh, $x:ident) => {
        crate::forward::jet::jet_unary_method!(@inverse Asinh, $x)
    };
    (acosh, $x:ident) => {
        crate::forward::jet::jet_unary_method!(@inverse Acosh, $x)
    };
    (atanh, $x:ident) => {
        crate::forward::jet::jet_unary_method!(@inverse Atanh, $x)
    };
    (@inverse $kind:ident, $x:ident) => {
        Self::new(crate::forward::series::inverse(
            &$x.coefficients,
            crate::forward::series::Inverse::$kind,
        ))
    };
    (signum, $x:ident) => {
        Self::lift($x.coefficients[0].signum())
    };
    (floor, $x:ident) => {
        Self::lift($x.coefficients[0].floor())
    };
    (ceil, $x:ident) => {
        Self::lift($x.coefficients[0].ceil())
    };
    (round, $x:ident) => {
        Self::lift($x.coefficients[0].round())
    };
    (trunc, $x:ident) => {
        Self::lift($x.coefficients[0].trunc())
    };
    (fract, $x:ident) => {
        Self::new(crate::forward::series::linear(
            &$x.coefficients,
            $x.coefficients[0].fract(),
            S::ONE,
        ))
    };
    (to_degrees, $x:ident) => {
        Self::new(crate::forward::series::linear(
            &$x.coefficients,
            $x.coefficients[0].to_degrees(),
            S::ONE.to_degrees(),
        ))
    };
    (to_radians, $x:ident) => {
        Self::new(crate::forward::series::linear(
            &$x.coefficients,
            $x.coefficients[0].to_radians(),
            S::ONE.to_radians(),
        ))
    };
    ($name:ident, $x:ident) => {
        Self::new(crate::forward::series::$name(&$x.coefficients))
    };
}

pub(crate) use {jet_unary_method, jet_unary_methods};

impl<S: RealScalar, V: Real<S>, const N: usize> Real<S> for Jet<V, N> {
    type Base = V::Base;
    type Primal = V;
    type Mask = V::Mask;
    const LANES: usize = V::LANES;
    const LEVEL: usize = V::LEVEL + 1;

    #[inline(always)]
    fn constant(value: S) -> Self {
        Self::new(series::lift(V::constant(value)))
    }
    #[inline(always)]
    fn passive(base: V::Base) -> Self {
        Self::new(series::lift(V::passive(base)))
    }
    #[inline(always)]
    fn lift(primal: V) -> Self {
        Self::new(series::lift(primal))
    }
    #[inline(always)]
    fn primal(self) -> V {
        self.coefficients[0]
    }
    #[inline(always)]
    fn base(self) -> V::Base {
        self.coefficients[0].base()
    }
    #[inline(always)]
    fn lt(self, other: Self) -> V::Mask {
        self.coefficients[0].lt(other.coefficients[0])
    }
    #[inline(always)]
    fn le(self, other: Self) -> V::Mask {
        self.coefficients[0].le(other.coefficients[0])
    }
    #[inline(always)]
    fn gt(self, other: Self) -> V::Mask {
        self.coefficients[0].gt(other.coefficients[0])
    }
    #[inline(always)]
    fn ge(self, other: Self) -> V::Mask {
        self.coefficients[0].ge(other.coefficients[0])
    }
    #[inline(always)]
    fn is_finite(self) -> V::Mask {
        let [first, rest @ ..] = self.coefficients.as_slice() else {
            return <V::Mask as crate::Mask>::splat(true);
        };
        rest.iter()
            .fold(first.is_finite(), |finite, c| finite & c.is_finite())
    }
    #[inline(always)]
    fn select(mask: V::Mask, if_true: Self, if_false: Self) -> Self {
        Self::new(core::array::from_fn(|k| {
            V::select(mask, if_true.coefficients[k], if_false.coefficients[k])
        }))
    }

    crate::real::unary_functions!(crate::forward::jet::jet_unary_methods);

    #[inline]
    fn powf(self, other: Self) -> Self {
        Self::new(series::powf(&self.coefficients, &other.coefficients))
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::new(series::hypot(&self.coefficients, &other.coefficients))
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::new(series::atan2(&self.coefficients, &other.coefficients))
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        Self::new(series::extremum(
            &self.coefficients,
            &other.coefficients,
            false,
        ))
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        Self::new(series::extremum(
            &self.coefficients,
            &other.coefficients,
            true,
        ))
    }
    #[inline]
    fn log(self, other: Self) -> Self {
        Self::new(series::log(&self.coefficients, &other.coefficients))
    }
    #[inline]
    fn logaddexp(self, other: Self) -> Self {
        Self::new(series::logaddexp(&self.coefficients, &other.coefficients))
    }
    #[inline]
    fn xlogy(self, other: Self) -> Self {
        Self::new(series::xlogy(&self.coefficients, &other.coefficients))
    }
    #[inline]
    fn xlog1py(self, other: Self) -> Self {
        Self::new(series::xlog1py(&self.coefficients, &other.coefficients))
    }
    #[inline]
    fn powm1(self, other: Self) -> Self {
        Self::new(series::powm1(&self.coefficients, &other.coefficients))
    }
    #[inline]
    fn powi(self, exponent: i32) -> Self {
        Self::new(series::powi(&self.coefficients, exponent))
    }
    #[inline]
    fn clamp(self, minimum: Self, maximum: Self) -> Self {
        // `c₀` is the clamp in `V`, so a NaN stays NaN and `−0` keeps its sign; the
        // higher coefficients are those of `min(max(x, minimum), maximum)`.
        let mut coefficients = self.max(minimum).min(maximum).coefficients;
        let x0 = self.coefficients[0];
        coefficients[0] = x0.clamp(minimum.coefficients[0], maximum.coefficients[0]);
        let nan = !x0.ge(x0);
        let clamped = Self::select(nan, self, Self::new(coefficients));
        Self::select(minimum.le(maximum), clamped, Self::constant(S::NAN))
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = series::sin_cos(&self.coefficients);
        (Self::new(s), Self::new(c))
    }

    #[inline]
    fn azmul(self, other: Self) -> Self {
        Self::new(series::azmul(&self.coefficients, &other.coefficients))
    }

    fn combine(arguments: &[Self], value: V, partials: &[V]) -> Result<Self, SciOxideError> {
        crate::support::check_len(partials.len(), arguments.len())?;
        Ok(Self::new(series::combine(
            arguments.iter().map(|argument| argument.coefficients),
            value,
            partials,
        )))
    }
}
