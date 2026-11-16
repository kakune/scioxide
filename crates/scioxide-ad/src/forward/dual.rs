//! First-order dual numbers over any number type.

use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

use scioxide_core::{RealScalar, SciOxideError};

use super::apply;
use crate::Real;
use crate::real::sealed::{self, Token};
use crate::rules::{self, Chain, Class, Product as Apply};

/// A value `v` and one directional derivative `v̇`, the number `v + v̇ ε` with `ε² = 0`.
///
/// `V` is any [`Real`]: `Dual<f64>` is one tangent of a scalar, `Dual<Lanes<f64, 8>>`
/// one tangent of eight lanes, and `Dual<Dual<f64>>` a second derivative. Operations
/// apply the chain rule of the [crate rules](crate#rules-and-conventions) in `V`
/// arithmetic, with IEEE products for `×`, scalar factors, `fract` and the smooth
/// exponential and trigonometric family listed there, and [`Real::azmul`] for every
/// other partial. A forward
/// pass costs a small constant times the plain evaluation.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Dual, Real};
/// let x = Dual::variable(2.0_f64);
/// let y = x * x.sin();
/// assert_eq!(y.value(), 2.0 * 2.0_f64.sin());
/// assert!((y.derivative() - (2.0_f64.sin() + 2.0 * 2.0_f64.cos())).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<V> {
    value: V,
    derivative: V,
}

impl<V: Copy> Dual<V> {
    /// Builds `value + derivative·ε`.
    #[must_use]
    #[inline(always)]
    pub const fn new(value: V, derivative: V) -> Self {
        Self { value, derivative }
    }
    /// Returns the value `v`.
    #[must_use]
    #[inline(always)]
    pub const fn value(self) -> V {
        self.value
    }
    /// Returns the derivative `v̇`.
    #[must_use]
    #[inline(always)]
    pub const fn derivative(self) -> V {
        self.derivative
    }
    /// Returns the value and the derivative by reference.
    #[inline(always)]
    pub(crate) const fn parts(&self) -> (&V, &V) {
        (&self.value, &self.derivative)
    }
    /// Returns the value and the derivative for update in place.
    #[inline(always)]
    pub(crate) const fn parts_mut(&mut self) -> (&mut V, &mut V) {
        (&mut self.value, &mut self.derivative)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Dual<V> {
    /// Builds the independent variable `value + 1·ε`.
    #[must_use]
    #[inline(always)]
    pub fn variable(value: V) -> Self {
        Self::new(value, V::one())
    }
}

impl<S: RealScalar, V: Real<S>> Chain<S> for Dual<V> {
    #[inline(always)]
    fn is_active(self) -> bool {
        true
    }
    #[inline(always)]
    fn unary(self, value: V, partial: V, product: Apply, _: Class) -> Self {
        Self::new(value, apply(partial, self.derivative, product))
    }
    #[inline(always)]
    fn binary(self, other: Self, value: V, left: V, right: V, product: Apply, _: Class) -> Self {
        Self::new(
            value,
            apply(left, self.derivative, product) + apply(right, other.derivative, product),
        )
    }
    #[inline(always)]
    fn kinks(self, _: V::Mask) {}
    #[inline(always)]
    fn pair_kinks(self, _: Self, _: V::Mask) {}
}

impl<V: sealed::Value> sealed::Value for Dual<V> {
    type Elem = V::Elem;
}

impl<S: RealScalar, V: Real<S>> sealed::Map<S> for Dual<V> {
    #[inline]
    fn map_all<F: FnMut(S) -> [S; 3]>(token: Token, x: Self, rule: &mut F) -> [Self; 3] {
        let [f0, f1, f2] = V::map_all(token, x.value, rule);
        let d = x.derivative;
        [
            Self::new(f0, f1.azmul(d)),
            Self::new(f1, f2.azmul(d)),
            Self::new(f2, V::constant(S::NAN).azmul(d)),
        ]
    }

    #[inline]
    fn select_lanes<T, P: Fn(&T) -> Self, I: FnMut(usize) -> usize>(
        token: Token,
        table: &[T],
        project: &P,
        index: &mut I,
    ) -> Self {
        Self::new(
            V::select_lanes(token, table, &|entry| project(entry).value, index),
            V::select_lanes(token, table, &|entry| project(entry).derivative, index),
        )
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Add for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(self.value + other.value, self.derivative + other.derivative)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Sub for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(self.value - other.value, self.derivative - other.derivative)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Mul for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        rules::mul(self, other)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Div for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        rules::div(self, other)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Rem for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: Self) -> Self {
        rules::rem(self, other)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Neg for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.value, -self.derivative)
    }
}

macro_rules! assign {
    ($($Assign:ident $assign:ident $Trait:ident $method:ident),*) => {
        $(
            impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> $Assign for Dual<V> {
                #[inline(always)]
                fn $assign(&mut self, other: Self) {
                    *self = $Trait::$method(*self, other);
                }
            }
            impl<S: RealScalar, V: Real<S>> $Assign<S> for Dual<V> {
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

impl<S: RealScalar, V: Real<S>> Add<S> for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: S) -> Self {
        Self::new(self.value + other, self.derivative)
    }
}

impl<S: RealScalar, V: Real<S>> Sub<S> for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: S) -> Self {
        Self::new(self.value - other, self.derivative)
    }
}

impl<S: RealScalar, V: Real<S>> Mul<S> for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: S) -> Self {
        Self::new(self.value * other, self.derivative * other)
    }
}

impl<S: RealScalar, V: Real<S>> Div<S> for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: S) -> Self {
        Self::new(self.value / other, self.derivative / other)
    }
}

impl<S: RealScalar, V: Real<S>> Rem<S> for Dual<V> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: S) -> Self {
        Self::new(self.value % other, self.derivative)
    }
}

macro_rules! scalar_left {
    ($($S:ty),*) => {
        $(
            impl<V: Real<$S>> Add<Dual<V>> for $S {
                type Output = Dual<V>;
                #[inline(always)]
                fn add(self, other: Dual<V>) -> Dual<V> {
                    Dual::new(V::constant(self) + other.value, other.derivative)
                }
            }
            impl<V: Real<$S>> Sub<Dual<V>> for $S {
                type Output = Dual<V>;
                #[inline(always)]
                fn sub(self, other: Dual<V>) -> Dual<V> {
                    Dual::new(V::constant(self) - other.value, -other.derivative)
                }
            }
            impl<V: Real<$S>> Mul<Dual<V>> for $S {
                type Output = Dual<V>;
                #[inline(always)]
                fn mul(self, other: Dual<V>) -> Dual<V> {
                    other * self
                }
            }
            impl<V: Real<$S>> Div<Dual<V>> for $S {
                type Output = Dual<V>;
                #[inline(always)]
                fn div(self, other: Dual<V>) -> Dual<V> {
                    Dual::constant(self) / other
                }
            }
            impl<V: Real<$S>> Rem<Dual<V>> for $S {
                type Output = Dual<V>;
                #[inline(always)]
                fn rem(self, other: Dual<V>) -> Dual<V> {
                    Dual::constant(self) % other
                }
            }
        )*
    };
}

scalar_left!(f32, f64);

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Sum for Dual<V> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = V::constant(S::NEG_ZERO);
        iter.fold(Self::new(zero, zero), |sum, value| sum + value)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>> Product for Dual<V> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |product, value| product * value)
    }
}

impl<S: RealScalar, V: Real<S>> Real<S> for Dual<V> {
    type Base = V::Base;
    type Primal = V;
    type Mask = V::Mask;
    const LANES: usize = V::LANES;
    const LEVEL: usize = V::LEVEL + 1;

    #[inline(always)]
    fn constant(value: S) -> Self {
        Self::new(V::constant(value), V::zero())
    }
    #[inline(always)]
    fn passive(base: V::Base) -> Self {
        Self::new(V::passive(base), V::zero())
    }
    #[inline(always)]
    fn lift(primal: V) -> Self {
        Self::new(primal, V::zero())
    }
    #[inline(always)]
    fn primal(self) -> V {
        self.value
    }
    #[inline(always)]
    fn base(self) -> V::Base {
        self.value.base()
    }
    #[inline(always)]
    fn lt(self, other: Self) -> V::Mask {
        self.value.lt(other.value)
    }
    #[inline(always)]
    fn le(self, other: Self) -> V::Mask {
        self.value.le(other.value)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> V::Mask {
        self.value.gt(other.value)
    }
    #[inline(always)]
    fn ge(self, other: Self) -> V::Mask {
        self.value.ge(other.value)
    }
    #[inline(always)]
    fn is_finite(self) -> V::Mask {
        self.value.is_finite() & self.derivative.is_finite()
    }
    #[inline(always)]
    fn select(mask: V::Mask, if_true: Self, if_false: Self) -> Self {
        Self::new(
            V::select(mask, if_true.value, if_false.value),
            V::select(mask, if_true.derivative, if_false.derivative),
        )
    }

    crate::forward::chain_methods!();

    #[inline(always)]
    fn azmul(self, other: Self) -> Self {
        Self::new(
            self.value.azmul(other.value),
            self.derivative.azmul(other.value) + self.value.azmul(other.derivative),
        )
    }

    fn combine(arguments: &[Self], value: V, partials: &[V]) -> Result<Self, SciOxideError> {
        crate::support::check_len(partials.len(), arguments.len())?;
        let derivative = arguments
            .iter()
            .zip(partials)
            .fold(V::constant(S::NEG_ZERO), |sum, (argument, &partial)| {
                sum + partial.azmul(argument.derivative)
            });
        Ok(Self::new(value, derivative))
    }
}
