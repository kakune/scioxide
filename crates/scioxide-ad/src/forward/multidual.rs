//! Dual numbers with a fixed number of tangents over any number type.

use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

use scioxide_core::{RealScalar, SciOxideError};

use super::apply;
use crate::Real;
use crate::real::sealed::{self, Token};
use crate::rules::{self, Chain, Class, Product as Apply};

/// A value `v` and `N` directional derivatives `∇v = (v̇₀, …, v̇_{N−1})`, the number
/// `v + Σᵢ v̇ᵢ εᵢ` with `εᵢ εⱼ = 0`.
///
/// One forward pass gives the gradient with respect to `N` inputs, or `N` columns
/// of a Jacobian. Every operation applies the same chain rule as [`Dual`](crate::Dual)
/// to each component: the derivatives equal `N` separate `Dual` passes bit for bit,
/// at about `N` times the cost of a `Dual` operation on the derivative part and
/// one evaluation of the value. The storage is plain `Copy` data, `N + 1` values of
/// `V`, so `V = Lanes<f64, W>` carries `N` tangents for `W` lanes.
/// `Dual<MultiDual<V, N>>`, with the values seeded by [`MultiDual::variables`] and
/// the derivatives by `MultiDual::constant(vᵢ)`, gives the gradient and the `N`
/// components of one Hessian-vector product `H·v`; `K` products in one sweep come
/// from `Tape<MultiDual<V, K>>` with seeds `MultiDual::new(xᵢ, [v₁ᵢ, …, v_Kᵢ])`.
/// Forward mode pays off for a few inputs; beyond about four, reverse mode is
/// cheaper.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{MultiDual, Real};
/// let x = MultiDual::<f64, 2>::variable(1.5, 0)?;
/// let y = MultiDual::variable(0.5, 1)?;
/// let f = x * y.exp() + y.sin();
/// assert_eq!(f.value(), 1.5 * 0.5_f64.exp() + 0.5_f64.sin());
/// let [dx, dy] = f.gradient();
/// assert!((dx - 0.5_f64.exp()).abs() < 1e-15);
/// assert!((dy - (1.5 * 0.5_f64.exp() + 0.5_f64.cos())).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiDual<V, const N: usize> {
    value: V,
    gradient: [V; N],
}

impl<V: Copy, const N: usize> MultiDual<V, N> {
    /// Builds `value + Σᵢ gradient[i]·εᵢ`.
    #[must_use]
    #[inline(always)]
    pub const fn new(value: V, gradient: [V; N]) -> Self {
        Self { value, gradient }
    }
    /// Returns the value `v`.
    #[must_use]
    #[inline(always)]
    pub const fn value(self) -> V {
        self.value
    }
    /// Returns the derivatives `(v̇₀, …, v̇_{N−1})`.
    #[must_use]
    #[inline(always)]
    pub const fn gradient(self) -> [V; N] {
        self.gradient
    }
    /// Returns the value and the derivatives by reference.
    #[inline(always)]
    pub(crate) const fn parts(&self) -> (&V, &[V; N]) {
        (&self.value, &self.gradient)
    }
    /// Returns the value for update in place; the derivatives stay.
    #[inline(always)]
    pub(crate) const fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> MultiDual<V, N> {
    /// Builds the independent variable `value + 1·ε_direction`.
    ///
    /// # Errors
    ///
    /// `OutOfBounds` if `direction ≥ N`.
    #[inline]
    pub fn variable(value: V, direction: usize) -> Result<Self, SciOxideError> {
        if direction >= N {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("direction"),
                direction,
                N,
            ));
        }
        let mut gradient = [V::zero(); N];
        gradient[direction] = V::one();
        Ok(Self::new(value, gradient))
    }

    /// Returns `N` independent variables: element `i` has value `values[i]` and
    /// derivative 1 in direction `i` and 0 in the others.
    ///
    /// Element `i` equals `MultiDual::variable(values[i], i)` bit for bit; use
    /// [`MultiDual::variable`] to seed a subset of the directions.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_ad::{MultiDual, Real};
    /// let [x, y] = MultiDual::<f64, 2>::variables([1.5, 0.5]);
    /// let [dx, dy] = (x * y.exp()).gradient();
    /// assert!((dx - 0.5_f64.exp()).abs() < 1e-15 && (dy - 1.5 * 0.5_f64.exp()).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn variables(values: [V; N]) -> [Self; N] {
        let mut variables = [Self::new(V::zero(), [V::zero(); N]); N];
        for (i, (variable, value)) in variables.iter_mut().zip(values).enumerate() {
            variable.value = value;
            variable.gradient[i] = V::one();
        }
        variables
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Chain<S> for MultiDual<V, N> {
    #[inline(always)]
    fn is_active(self) -> bool {
        true
    }
    #[inline(always)]
    fn unary(self, value: V, partial: V, product: Apply, _: Class) -> Self {
        let mut gradient = self.gradient;
        for target in &mut gradient {
            *target = apply(partial, *target, product);
        }
        Self::new(value, gradient)
    }
    #[inline(always)]
    fn binary(self, other: Self, value: V, left: V, right: V, product: Apply, _: Class) -> Self {
        let mut gradient = self.gradient;
        for (target, &o) in gradient.iter_mut().zip(&other.gradient) {
            *target = apply(left, *target, product) + apply(right, o, product);
        }
        Self::new(value, gradient)
    }
    #[inline(always)]
    fn kinks(self, _: V::Mask) {}
    #[inline(always)]
    fn pair_kinks(self, _: Self, _: V::Mask) {}
}

impl<V: sealed::Value, const N: usize> sealed::Value for MultiDual<V, N> {
    type Elem = V::Elem;
}

impl<S: RealScalar, V: Real<S>, const N: usize> sealed::Map<S> for MultiDual<V, N> {
    #[inline]
    fn map_all<F: FnMut(S) -> [S; 3]>(token: Token, x: Self, rule: &mut F) -> [Self; 3] {
        let [f0, f1, f2] = V::map_all(token, x.value, rule);
        let nan = V::constant(S::NAN);
        [
            Self::new(f0, x.gradient.map(|g| f1.azmul(g))),
            Self::new(f1, x.gradient.map(|g| f2.azmul(g))),
            Self::new(f2, x.gradient.map(|g| nan.azmul(g))),
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
            crate::arrays::build(V::zero(), |i| {
                V::select_lanes(token, table, &|entry| project(entry).gradient[i], index)
            }),
        )
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Add for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(
            self.value + other.value,
            crate::arrays::zip(self.gradient, other.gradient, |a, b| a + b),
        )
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Sub for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(
            self.value - other.value,
            crate::arrays::zip(self.gradient, other.gradient, |a, b| a - b),
        )
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Mul for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        rules::mul(self, other)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Div for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        rules::div(self, other)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Rem for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: Self) -> Self {
        rules::rem(self, other)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Neg for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.value, self.gradient.map(|g| -g))
    }
}

macro_rules! assign {
    ($($Assign:ident $assign:ident $Trait:ident $method:ident),*) => {
        $(
            impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> $Assign
                for MultiDual<V, N>
            {
                #[inline(always)]
                fn $assign(&mut self, other: Self) {
                    *self = $Trait::$method(*self, other);
                }
            }
            impl<S: RealScalar, V: Real<S>, const N: usize> $Assign<S> for MultiDual<V, N> {
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

impl<S: RealScalar, V: Real<S>, const N: usize> Add<S> for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: S) -> Self {
        Self::new(self.value + other, self.gradient)
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Sub<S> for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: S) -> Self {
        Self::new(self.value - other, self.gradient)
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Mul<S> for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: S) -> Self {
        Self::new(self.value * other, self.gradient.map(|g| g * other))
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Div<S> for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: S) -> Self {
        Self::new(self.value / other, self.gradient.map(|g| g / other))
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Rem<S> for MultiDual<V, N> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: S) -> Self {
        Self::new(self.value % other, self.gradient)
    }
}

macro_rules! scalar_left {
    ($($S:ty),*) => {
        $(
            impl<V: Real<$S>, const N: usize> Add<MultiDual<V, N>> for $S {
                type Output = MultiDual<V, N>;
                #[inline(always)]
                fn add(self, other: MultiDual<V, N>) -> MultiDual<V, N> {
                    MultiDual::new(V::constant(self) + other.value, other.gradient)
                }
            }
            impl<V: Real<$S>, const N: usize> Sub<MultiDual<V, N>> for $S {
                type Output = MultiDual<V, N>;
                #[inline(always)]
                fn sub(self, other: MultiDual<V, N>) -> MultiDual<V, N> {
                    MultiDual::new(V::constant(self) - other.value, other.gradient.map(|g| -g))
                }
            }
            impl<V: Real<$S>, const N: usize> Mul<MultiDual<V, N>> for $S {
                type Output = MultiDual<V, N>;
                #[inline(always)]
                fn mul(self, other: MultiDual<V, N>) -> MultiDual<V, N> {
                    other * self
                }
            }
            impl<V: Real<$S>, const N: usize> Div<MultiDual<V, N>> for $S {
                type Output = MultiDual<V, N>;
                #[inline(always)]
                fn div(self, other: MultiDual<V, N>) -> MultiDual<V, N> {
                    MultiDual::constant(self) / other
                }
            }
            impl<V: Real<$S>, const N: usize> Rem<MultiDual<V, N>> for $S {
                type Output = MultiDual<V, N>;
                #[inline(always)]
                fn rem(self, other: MultiDual<V, N>) -> MultiDual<V, N> {
                    MultiDual::constant(self) % other
                }
            }
        )*
    };
}

scalar_left!(f32, f64);

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Sum for MultiDual<V, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = V::constant(S::NEG_ZERO);
        iter.fold(Self::new(zero, [zero; N]), |sum, value| sum + value)
    }
}

impl<S: RealScalar, V: sealed::Value<Elem = S> + Real<S>, const N: usize> Product
    for MultiDual<V, N>
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |product, value| product * value)
    }
}

impl<S: RealScalar, V: Real<S>, const N: usize> Real<S> for MultiDual<V, N> {
    type Base = V::Base;
    type Primal = V;
    type Mask = V::Mask;
    const LANES: usize = V::LANES;
    const LEVEL: usize = V::LEVEL + 1;

    #[inline(always)]
    fn constant(value: S) -> Self {
        Self::new(V::constant(value), [V::zero(); N])
    }
    #[inline(always)]
    fn passive(base: V::Base) -> Self {
        Self::new(V::passive(base), [V::zero(); N])
    }
    #[inline(always)]
    fn lift(primal: V) -> Self {
        Self::new(primal, [V::zero(); N])
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
        self.gradient
            .iter()
            .fold(self.value.is_finite(), |finite, g| finite & g.is_finite())
    }
    #[inline(always)]
    fn select(mask: V::Mask, if_true: Self, if_false: Self) -> Self {
        Self::new(
            V::select(mask, if_true.value, if_false.value),
            crate::arrays::build(V::zero(), |i| {
                V::select(mask, if_true.gradient[i], if_false.gradient[i])
            }),
        )
    }

    crate::forward::chain_methods!();

    #[inline(always)]
    fn azmul(self, other: Self) -> Self {
        Self::new(
            self.value.azmul(other.value),
            crate::arrays::build(V::zero(), |i| {
                self.gradient[i].azmul(other.value) + self.value.azmul(other.gradient[i])
            }),
        )
    }

    fn combine(arguments: &[Self], value: V, partials: &[V]) -> Result<Self, SciOxideError> {
        crate::support::check_len(partials.len(), arguments.len())?;
        let gradient = crate::arrays::build(V::zero(), |i| {
            arguments
                .iter()
                .zip(partials)
                .fold(V::constant(S::NEG_ZERO), |sum, (argument, &partial)| {
                    sum + partial.azmul(argument.gradient[i])
                })
        });
        Ok(Self::new(value, gradient))
    }
}
