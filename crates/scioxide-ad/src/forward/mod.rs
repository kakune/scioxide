//! Forward-mode number types: tangents and Taylor coefficients carried with values.

use scioxide_core::RealScalar;

use crate::Real;
use crate::rules::Product;

mod dual;
mod jet;
mod multidual;
mod series;
mod transforms;

pub use dual::Dual;
pub use jet::Jet;
pub use multidual::MultiDual;
pub use transforms::{jet, jvp};

/// Applies a first partial to a tangent component with the product of its rule.
#[inline(always)]
fn apply<S: RealScalar, V: Real<S>>(partial: V, tangent: V, product: Product) -> V {
    match product {
        Product::Ieee => partial * tangent,
        Product::Absolute => partial.azmul(tangent),
    }
}

/// Emits the elementary methods of a first-order wrapper from [`crate::rules`].
macro_rules! chain_methods {
    () => {
        crate::real::unary_functions!(crate::forward::chain_unary_methods);

        #[inline(always)]
        fn powf(self, other: Self) -> Self {
            crate::rules::powf(self, other)
        }
        #[inline(always)]
        fn hypot(self, other: Self) -> Self {
            crate::rules::hypot(self, other)
        }
        #[inline(always)]
        fn atan2(self, other: Self) -> Self {
            crate::rules::atan2(self, other)
        }
        #[inline(always)]
        fn min(self, other: Self) -> Self {
            crate::rules::extremum(self, other, false)
        }
        #[inline(always)]
        fn max(self, other: Self) -> Self {
            crate::rules::extremum(self, other, true)
        }
        #[inline(always)]
        fn log(self, other: Self) -> Self {
            crate::rules::log(self, other)
        }
        #[inline(always)]
        fn logaddexp(self, other: Self) -> Self {
            crate::rules::logaddexp(self, other)
        }
        #[inline(always)]
        fn xlogy(self, other: Self) -> Self {
            crate::rules::xlogy(self, other)
        }
        #[inline(always)]
        fn xlog1py(self, other: Self) -> Self {
            crate::rules::xlog1py(self, other)
        }
        #[inline(always)]
        fn powm1(self, other: Self) -> Self {
            crate::rules::powm1(self, other)
        }
        #[inline(always)]
        fn powi(self, exponent: i32) -> Self {
            crate::rules::powi(self, exponent)
        }
        #[inline(always)]
        fn clamp(self, minimum: Self, maximum: Self) -> Self {
            crate::rules::clamp(self, minimum, maximum)
        }
        #[inline(always)]
        fn sin_cos(self) -> (Self, Self) {
            crate::rules::sin_cos(self)
        }
    };
}

/// Emits one unary method per elementary function; stable transforms go through
/// `map_scalar`.
macro_rules! chain_unary_methods {
    ($($(#[$doc:meta])* $name:ident,)*) => {
        $(
            #[inline(always)]
            fn $name(self) -> Self {
                crate::forward::chain_unary_method!($name, self)
            }
        )*
    };
}

macro_rules! chain_unary_method {
    (softplus, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Softplus)
    };
    (sigmoid, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Sigmoid)
    };
    (log_sigmoid, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::LogSigmoid)
    };
    (logit, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Logit)
    };
    (exprel, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Exprel)
    };
    (cosm1, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Cosm1)
    };
    (sinc, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Sinc)
    };
    (sincpi, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::Sincpi)
    };
    (sin_pi, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::SinPi)
    };
    (cos_pi, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::CosPi)
    };
    (tan_pi, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::TanPi)
    };
    (sin_degrees, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::SinDegrees)
    };
    (cos_degrees, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::CosDegrees)
    };
    (tan_degrees, $x:ident) => {
        crate::rules::stable($x, crate::stable::Stable::TanDegrees)
    };
    ($name:ident, $x:ident) => {
        crate::rules::$name($x)
    };
}

pub(crate) use {chain_methods, chain_unary_method, chain_unary_methods};
