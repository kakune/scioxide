//! [`Real`] for the IEEE scalars `f32` and `f64`.

use scioxide_core::{NumericScalar, RealScalar, SciOxideError, backend};

use crate::real::sealed::{self, Token};
use crate::stable::{self, Stable};
use crate::{BaseReal, Mask, Real};

impl sealed::MaskLanes for bool {
    #[inline(always)]
    fn count(self) -> u64 {
        u64::from(self)
    }
    #[inline(always)]
    fn count_first(self, lanes: usize) -> u64 {
        u64::from(self && lanes > 0)
    }
}

impl Mask for bool {
    #[inline(always)]
    fn splat(value: bool) -> Self {
        value
    }
    #[inline(always)]
    fn any(self) -> bool {
        self
    }
    #[inline(always)]
    fn all(self) -> bool {
        self
    }
}

impl<S: RealScalar> sealed::Value for S {
    type Elem = S;
}

impl<S: RealScalar> sealed::Map<S> for S {
    #[inline(always)]
    fn map_all<F: FnMut(S) -> [S; 3]>(_: Token, x: Self, rule: &mut F) -> [Self; 3] {
        rule(x)
    }

    #[inline(always)]
    fn select_lanes<T, P: Fn(&T) -> Self, I: FnMut(usize) -> usize>(
        _: Token,
        table: &[T],
        project: &P,
        index: &mut I,
    ) -> Self {
        table.get(index(0)).map_or(S::NAN, project)
    }
}

macro_rules! direct_unary {
    ($($name:ident),*) => {
        $(
            #[inline(always)]
            fn $name(self) -> Self {
                RealScalar::$name(self)
            }
        )*
    };
}

macro_rules! stable_unary {
    ($($name:ident => $function:ident),*) => {
        $(
            #[inline]
            fn $name(self) -> Self {
                stable::value(Stable::$function, self)
            }
        )*
    };
}

macro_rules! stable_binary {
    ($($name:ident => $kernel:ident),*) => {
        $(
            #[inline]
            fn $name(self, other: Self) -> Self {
                backend::$kernel(self, other).unwrap_or(S::NAN)
            }
        )*
    };
}

/// Returns the absolute-zero product of two scalars.
#[inline(always)]
pub(crate) fn azmul<S: RealScalar>(a: S, b: S) -> S {
    if a == S::ZERO || b == S::ZERO {
        S::ZERO
    } else {
        a * b
    }
}

impl<S: RealScalar> Real<S> for S {
    type Base = S;
    type Primal = S;
    type Mask = bool;
    const LANES: usize = 1;
    const LEVEL: usize = 0;

    #[inline(always)]
    fn constant(value: S) -> Self {
        value
    }
    #[inline(always)]
    fn passive(base: S) -> Self {
        base
    }
    #[inline(always)]
    fn lift(primal: S) -> Self {
        primal
    }
    #[inline(always)]
    fn primal(self) -> S {
        self
    }
    #[inline(always)]
    fn base(self) -> S {
        self
    }
    #[inline(always)]
    fn lt(self, other: Self) -> bool {
        self < other
    }
    #[inline(always)]
    fn le(self, other: Self) -> bool {
        self <= other
    }
    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self > other
    }
    #[inline(always)]
    fn ge(self, other: Self) -> bool {
        self >= other
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        RealScalar::is_finite(self)
    }
    #[inline(always)]
    fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }

    direct_unary!(
        recip, sqrt, cbrt, exp, exp2, exp_m1, ln, ln_1p, log2, log10, sin, cos, tan, asin, acos,
        atan, sinh, cosh, tanh, asinh, acosh, atanh, signum, floor, ceil, round, trunc, fract,
        to_degrees, to_radians
    );

    #[inline(always)]
    fn abs(self) -> Self {
        NumericScalar::abs(self)
    }

    stable_unary!(
        softplus => Softplus, sigmoid => Sigmoid, log_sigmoid => LogSigmoid, logit => Logit,
        exprel => Exprel, cosm1 => Cosm1, sinc => Sinc, sincpi => Sincpi, sin_pi => SinPi,
        cos_pi => CosPi, tan_pi => TanPi, sin_degrees => SinDegrees, cos_degrees => CosDegrees,
        tan_degrees => TanDegrees
    );

    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        RealScalar::powf(self, other)
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        RealScalar::hypot(self, other)
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        RealScalar::atan2(self, other)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        RealScalar::min(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        RealScalar::max(self, other)
    }
    #[inline(always)]
    fn log(self, other: Self) -> Self {
        RealScalar::log(self, other)
    }

    stable_binary!(
        logaddexp => real_logaddexp, xlogy => real_xlogy, xlog1py => real_xlog1py,
        powm1 => real_powm1
    );

    #[inline(always)]
    fn powi(self, exponent: i32) -> Self {
        RealScalar::powi(self, exponent)
    }
    #[inline(always)]
    fn clamp(self, minimum: Self, maximum: Self) -> Self {
        RealScalar::clamp(self, minimum, maximum)
    }
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        (RealScalar::sin(self), RealScalar::cos(self))
    }
    #[inline(always)]
    fn azmul(self, other: Self) -> Self {
        azmul(self, other)
    }

    fn combine(arguments: &[Self], value: S, partials: &[S]) -> Result<Self, SciOxideError> {
        crate::support::check_len(partials.len(), arguments.len())?;
        Ok(value)
    }
}

impl<S: RealScalar> BaseReal<S> for S {
    #[inline(always)]
    fn from_fn(mut value: impl FnMut(usize) -> S) -> Self {
        value(0)
    }
    #[inline(always)]
    fn lane(self, lane: usize) -> Option<S> {
        (lane == 0).then_some(self)
    }
}
