//! The number contract shared by plain values, lanes and every derivative mode.

use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{
    Add, AddAssign, BitAnd, BitOr, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, Sub, SubAssign,
};

use scioxide_core::{RealScalar, SciOxideError};

pub(crate) mod sealed;

/// Lists the unary elementary methods of [`Real`] with their documentation.
macro_rules! unary_functions {
    ($($callback:tt)*) => {
        $($callback)*! {
            /// Returns the reciprocal `1/x`.
            recip,
            /// Returns the square root `√x`; NaN for `x < 0`.
            sqrt,
            /// Returns the cube root `∛x`.
            cbrt,
            /// Returns `eˣ`.
            exp,
            /// Returns `2ˣ`.
            exp2,
            /// Returns `eˣ − 1`, accurate near `x = 0`.
            exp_m1,
            /// Returns the natural logarithm `ln x`; NaN for `x < 0`.
            ln,
            /// Returns `ln(1 + x)`, accurate near `x = 0`.
            ln_1p,
            /// Returns `log₂ x`.
            log2,
            /// Returns `log₁₀ x`.
            log10,
            /// Returns `sin x` (radians).
            sin,
            /// Returns `cos x` (radians).
            cos,
            /// Returns `tan x` (radians).
            tan,
            /// Returns `arcsin x ∈ [−π/2, π/2]`; NaN for `|x| > 1`.
            asin,
            /// Returns `arccos x ∈ [0, π]`; NaN for `|x| > 1`.
            acos,
            /// Returns `arctan x ∈ (−π/2, π/2)`.
            atan,
            /// Returns `sinh x`.
            sinh,
            /// Returns `cosh x`.
            cosh,
            /// Returns `tanh x`.
            tanh,
            /// Returns `arsinh x`.
            asinh,
            /// Returns `arcosh x`; NaN for `x < 1`.
            acosh,
            /// Returns `artanh x`; NaN for `|x| > 1`.
            atanh,
            /// Returns `|x|`. The derivative is `sign x`, and 0 at the kink `x = 0`.
            abs,
            /// Returns `1` for `+0` and positive `x`, `−1` for `−0` and negative `x`
            /// (NaN for NaN); its derivative is 0.
            signum,
            /// Returns the largest integer `≤ x`; its derivative is 0.
            floor,
            /// Returns the smallest integer `≥ x`; its derivative is 0.
            ceil,
            /// Returns `x` rounded half away from zero; its derivative is 0.
            round,
            /// Returns the integer part of `x`; its derivative is 0.
            trunc,
            /// Returns `x − trunc(x)`; its derivative is 1.
            fract,
            /// Converts radians to degrees, `x·180/π`.
            to_degrees,
            /// Converts degrees to radians, `x·π/180`.
            to_radians,
            /// Returns `softplus(x) = ln(1 + eˣ)`, computed without overflow.
            softplus,
            /// Returns the logistic function `σ(x) = 1/(1 + e⁻ˣ)`.
            sigmoid,
            /// Returns `ln σ(x) = −softplus(−x)`.
            log_sigmoid,
            /// Returns `logit(x) = ln(x/(1 − x))` for `0 < x < 1`, and NaN elsewhere.
            logit,
            /// Returns `exprel(x) = (eˣ − 1)/x`, with `exprel(0) = 1`.
            exprel,
            /// Returns `cos x − 1`, accurate near `x = 0`.
            cosm1,
            /// Returns `sinc(x) = sin(x)/x`, with `sinc(0) = 1`.
            sinc,
            /// Returns the normalized `sinc(πx) = sin(πx)/(πx)`.
            sincpi,
            /// Returns `sin(πx)` with exact reduction in units of π.
            sin_pi,
            /// Returns `cos(πx)` with exact reduction in units of π.
            cos_pi,
            /// Returns `tan(πx)`; NaN at half-integers.
            tan_pi,
            /// Returns the sine of `x` degrees with exact reduction.
            sin_degrees,
            /// Returns the cosine of `x` degrees with exact reduction.
            cos_degrees,
            /// Returns the tangent of `x` degrees; NaN at odd multiples of 90.
            tan_degrees,
        }
    };
}

/// Lists the binary elementary methods of [`Real`] with their documentation.
macro_rules! binary_functions {
    ($($callback:tt)*) => {
        $($callback)*! {
            /// Returns `xʸ` (`self` is the base).
            powf,
            /// Returns `√(x² + y²)` without undue overflow.
            hypot,
            /// Returns the angle of the point `(other, self)`, `atan2(self, other) ∈ [−π, π]`.
            atan2,
            /// Returns the smaller operand; the other one if exactly one is NaN.
            min,
            /// Returns the larger operand; the other one if exactly one is NaN.
            max,
            /// Returns `log_b x` with `self = x` and `other = b`.
            log,
            /// Returns `ln(eˣ + eʸ)`, computed without overflow.
            logaddexp,
            /// Returns `x·ln y`, with `0·ln y = 0` for every finite `y` (NaN for an
            /// infinite or NaN `y`).
            xlogy,
            /// Returns `x·ln(1 + y)`, with `0·ln(1 + y) = 0` for every finite `y` (NaN
            /// for an infinite or NaN `y`).
            xlog1py,
            /// Returns `xʸ − 1`, accurate when `xʸ` is near 1.
            powm1,
        }
    };
}

pub(crate) use unary_functions;

macro_rules! declare_unary {
    ($($(#[$doc:meta])* $name:ident,)*) => {
        $(
            $(#[$doc])*
            #[must_use]
            fn $name(self) -> Self;
        )*
    };
}

macro_rules! declare_binary {
    ($($(#[$doc:meta])* $name:ident,)*) => {
        $(
            $(#[$doc])*
            #[must_use]
            fn $name(self, other: Self) -> Self;
        )*
    };
}

/// A real number type: `f32`, `f64`, [`Lanes`](crate::Lanes), [`Dual`](crate::Dual),
/// [`MultiDual`](crate::MultiDual), [`Jet`](crate::Jet) or [`Var`](crate::Var).
///
/// Code written once against `R: Real` runs on plain floats, on SIMD lanes, in
/// forward mode and in reverse mode; the caller chooses the number. `S` is the
/// IEEE scalar (`f64` by default). The trait is sealed.
///
/// Values follow IEEE 754 in a fixed operation order: no operation checks its
/// arguments or returns an error, and NaN and infinity propagate. Wrappers take
/// every value from their [`Primal`](Real::Primal), so a value never depends on the
/// derivative mode. Derivatives apply the chain rule with the partials of the
/// [crate rules](crate#rules-and-conventions), evaluated in `Primal` arithmetic, so
/// nesting differentiates the evaluated form. Kinks use the midpoint of the
/// one-sided derivatives.
///
/// Generic code puts the scalar on the right of an operator (`x * 2.0`,
/// `-x + 1.0`); concrete types also accept `f32` and `f64` on the left. A base value
/// (for example a random draw) enters arithmetic through [`Real::passive`]. Code that
/// needs a plain `if` bounds `R: Real<Mask = bool>`; otherwise it branches with
/// [`Real::select`].
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Dual, Lanes, Real};
///
/// fn sample<R: Real>(theta: R, noise: R::Base) -> R {
///     theta.sin() * R::passive(noise)
/// }
///
/// let plain = sample(0.5, 2.0);
/// let lanes = sample(Lanes::<f64, 4>::splat(0.5), Lanes::splat(2.0));
/// let tangent = sample(Dual::variable(0.5), 2.0);
/// assert_eq!(lanes.lane(3), Some(plain));
/// assert_eq!(tangent.value(), plain);
/// assert!((tangent.derivative() - 2.0 * 0.5_f64.cos()).abs() < 1e-15);
/// ```
pub trait Real<S: RealScalar = f64>:
    sealed::Value<Elem = S>
    + sealed::Map<S>
    + sealed::Linear<S>
    + sealed::Loop<S>
    + Copy
    + Debug
    + Sum
    + Product
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Neg<Output = Self>
    + Add<S, Output = Self>
    + Sub<S, Output = Self>
    + Mul<S, Output = Self>
    + Div<S, Output = Self>
    + Rem<S, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + AddAssign<S>
    + SubAssign<S>
    + MulAssign<S>
    + DivAssign<S>
{
    /// The innermost value type: `S` or [`Lanes<S, W>`](crate::Lanes).
    type Base: BaseReal<S, Mask = Self::Mask>;
    /// The type one derivative level down; `Self` for `S` and `Lanes`.
    type Primal: Real<S, Base = Self::Base, Mask = Self::Mask>;
    /// The result of comparisons: `bool` for scalar types, [`LaneMask<W>`](crate::LaneMask)
    /// for lane types.
    type Mask: Mask;
    /// The number of independent lanes: 1 or `W`.
    const LANES: usize;
    /// The number of derivative levels above `Base`.
    const LEVEL: usize;

    /// Returns the undifferentiated value `value` in every lane.
    #[must_use]
    fn constant(value: S) -> Self;
    /// Lifts a base value through every level, undifferentiated.
    #[must_use]
    fn passive(base: Self::Base) -> Self;
    /// Lifts a value of the next level down, undifferentiated.
    #[must_use]
    fn lift(primal: Self::Primal) -> Self;
    /// Returns the value one level down (the value component of a wrapper).
    #[must_use]
    fn primal(self) -> Self::Primal;
    /// Returns the innermost value.
    #[must_use]
    fn base(self) -> Self::Base;
    /// Returns the constant 0.
    #[must_use]
    fn zero() -> Self {
        Self::constant(S::ZERO)
    }
    /// Returns the constant 1.
    #[must_use]
    fn one() -> Self {
        Self::constant(S::ONE)
    }

    /// Compares base values lane by lane: `self < other` (false where either is NaN).
    #[must_use]
    fn lt(self, other: Self) -> Self::Mask;
    /// Compares base values lane by lane: `self ≤ other`.
    #[must_use]
    fn le(self, other: Self) -> Self::Mask;
    /// Compares base values lane by lane: `self > other`.
    #[must_use]
    fn gt(self, other: Self) -> Self::Mask;
    /// Compares base values lane by lane: `self ≥ other`.
    #[must_use]
    fn ge(self, other: Self) -> Self::Mask;
    /// Returns where the value and all its derivative components are finite.
    #[must_use]
    fn is_finite(self) -> Self::Mask;
    /// Picks, lane by lane, every component of `if_true` where `mask` holds and of
    /// `if_false` elsewhere. The derivative is that of the selected operand; a
    /// selection is not a kink.
    #[must_use]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    unary_functions!(declare_unary);
    binary_functions!(declare_binary);

    /// Returns `xⁿ` for an integer exponent.
    #[must_use]
    fn powi(self, exponent: i32) -> Self;
    /// Returns `min(max(self, minimum), maximum)`; NaN where `minimum > maximum` or a
    /// bound is NaN.
    #[must_use]
    fn clamp(self, minimum: Self, maximum: Self) -> Self;
    /// Returns `(sin x, cos x)`, equal to the two separate calls bit for bit.
    #[must_use]
    fn sin_cos(self) -> (Self, Self);

    /// Returns the absolute-zero product: `self · other`, except that an exactly zero
    /// operand gives `+0` (also against an infinite or NaN operand). Wrappers apply
    /// it to every inner product.
    #[must_use]
    fn azmul(self, other: Self) -> Self;

    /// Applies a scalar function given lane by lane as `[f(x), f′(x), f″(x)]`.
    ///
    /// The result is exact through total derivative order two (`Dual`, `MultiDual`,
    /// `Var`, `Var<Dual>`, `Dual<Var>`, `Dual<Dual>`, `Jet` up to `c₂`);
    /// deeper components are NaN.
    /// `rule` is called once per lane and must be deterministic.
    ///
    /// ```
    /// use scioxide_ad::{Dual, Real};
    /// let x = Dual::variable(0.3_f64);
    /// let y = x.map_scalar(|v| [v.sin(), v.cos(), -v.sin()]);
    /// assert_eq!(y, x.sin());
    /// ```
    #[must_use]
    fn map_scalar(self, mut rule: impl FnMut(S) -> [S; 3]) -> Self {
        let [value, _, _] = <Self as sealed::Map<S>>::map_all(sealed::Token(()), self, &mut rule);
        value
    }

    /// Applies a fallible scalar function as [`Real::map_scalar`] does.
    ///
    /// # Errors
    ///
    /// The first error returned by `rule`, unchanged.
    fn try_map_scalar(
        self,
        mut rule: impl FnMut(S) -> Result<[S; 3], SciOxideError>,
    ) -> Result<Self, SciOxideError> {
        let mut failure = None;
        let value = self.map_scalar(|x| match rule(x) {
            Ok(rule) => rule,
            Err(error) => {
                failure.get_or_insert(error);
                [S::NAN; 3]
            }
        });
        failure.map_or(Ok(value), Err)
    }

    /// Builds a value with primal `value` and partial `partials[k]` with respect to
    /// `arguments[k]`.
    ///
    /// The caller computes `value` and `partials` in `Primal` arithmetic, so the
    /// result is exact to the nesting order. `Jet` coefficients above `c₁` are NaN.
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` if `partials` and `arguments` differ in length.
    fn combine(
        arguments: &[Self],
        value: Self::Primal,
        partials: &[Self::Primal],
    ) -> Result<Self, SciOxideError>;

    /// Picks, lane by lane, the table entry `locate(key)`.
    ///
    /// The key is not differentiated. `locate` is called once per lane and once more
    /// per base component of the number type, so it must be deterministic.
    ///
    /// # Errors
    ///
    /// `OutOfBounds` if `locate` returns an index past the table.
    fn gather(
        table: &[Self],
        key: Self,
        mut locate: impl FnMut(S) -> usize,
    ) -> Result<Self, SciOxideError> {
        let key = key.base();
        let len = table.len();
        for lane in 0..Self::LANES {
            let index = locate(key.lane(lane).unwrap_or(S::NAN));
            if index >= len {
                return Err(SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("table"),
                    index,
                    len,
                ));
            }
        }
        let mut index = |lane: usize| locate(key.lane(lane).unwrap_or(S::NAN)).min(len - 1);
        Ok(<Self as sealed::Map<S>>::select_lanes(
            sealed::Token(()),
            table,
            &|entry: &Self| *entry,
            &mut index,
        ))
    }
}

/// An innermost value type: `f32`, `f64` or [`Lanes`](crate::Lanes). Sealed.
pub trait BaseReal<S: RealScalar = f64>: Real<S, Base = Self, Primal = Self> {
    /// Builds a value whose lane `k` is `value(k)`.
    #[must_use]
    fn from_fn(value: impl FnMut(usize) -> S) -> Self;
    /// Returns lane `lane`, or `None` past the last lane.
    #[must_use]
    fn lane(self, lane: usize) -> Option<S>;
}

/// The result of a lane-by-lane comparison: `bool` or [`LaneMask`](crate::LaneMask).
/// Sealed.
pub trait Mask:
    sealed::MaskLanes + Copy + Debug + BitAnd<Output = Self> + BitOr<Output = Self> + Not<Output = Self>
{
    /// Returns `value` in every lane.
    #[must_use]
    fn splat(value: bool) -> Self;
    /// Returns whether any lane holds.
    #[must_use]
    fn any(self) -> bool;
    /// Returns whether every lane holds.
    #[must_use]
    fn all(self) -> bool;
}
