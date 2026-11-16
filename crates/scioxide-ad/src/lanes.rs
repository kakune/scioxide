//! Fixed-width lane packs and their masks.

use core::iter::{Product, Sum};
use core::ops::{
    Add, AddAssign, BitAnd, BitOr, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, Sub, SubAssign,
};

use scioxide_core::{RealScalar, SciOxideError};

use crate::arrays;
use crate::real::sealed::{self, Token};
use crate::{BaseReal, Mask, Real};

mod kernels;

/// `W` independent values of `S` evaluated together.
///
/// Every [`Real`] operation is a loop over the fixed array, inlined in safe Rust, so
/// it vectorizes for the target features of the build: AVX2 needs
/// `-C target-cpu=x86-64-v3` (or `native`), while a default x86-64 build uses SSE2.
/// Each lane is computed by the same operation sequence, so results are identical
/// for every `W` and every target. `exp`, `exp_m1`, `ln`, `ln_1p`, `sin`, `cos`,
/// `tanh` and `powf` use the branch-free lane kernels of `scioxide_core::backend`
/// (evaluated in `f64` for `f32` lanes) and can differ from the scalar function in
/// the last ulps; every other operation equals the scalar function bit for bit.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Lanes, Real};
/// let x = Lanes::<f64, 4>::from_array([0.0, 1.0, 2.0, 3.0]);
/// let y = (x * 2.0 + 1.0).sqrt();
/// assert_eq!(y.to_array(), [1.0, 3.0_f64.sqrt(), 5.0_f64.sqrt(), 7.0_f64.sqrt()]);
/// assert!(x.gt(Lanes::splat(1.5)).any());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Lanes<S, const W: usize>([S; W]);

/// The result of a lane-by-lane comparison of [`Lanes<S, W>`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneMask<const W: usize>([bool; W]);

impl<const W: usize> LaneMask<W> {
    /// Returns `value` in every lane.
    #[must_use]
    pub const fn splat(value: bool) -> Self {
        Self([value; W])
    }
    /// Builds a mask from one flag per lane.
    #[must_use]
    pub const fn from_array(lanes: [bool; W]) -> Self {
        Self(lanes)
    }
    /// Returns one flag per lane.
    #[must_use]
    pub const fn to_array(self) -> [bool; W] {
        self.0
    }
    /// Returns whether any lane holds.
    #[must_use]
    pub fn any(self) -> bool {
        self.0.iter().any(|&lane| lane)
    }
    /// Returns whether every lane holds.
    #[must_use]
    pub fn all(self) -> bool {
        self.0.iter().all(|&lane| lane)
    }
}

impl<const W: usize> BitAnd for LaneMask<W> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        Self(arrays::zip(self.0, other.0, |a, b| a & b))
    }
}

impl<const W: usize> BitOr for LaneMask<W> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        Self(arrays::zip(self.0, other.0, |a, b| a | b))
    }
}

impl<const W: usize> Not for LaneMask<W> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(arrays::map(self.0, |lane| !lane))
    }
}

impl<const W: usize> sealed::MaskLanes for LaneMask<W> {
    #[inline(always)]
    fn count(self) -> u64 {
        self.0.iter().map(|&lane| u64::from(lane)).sum()
    }
    #[inline(always)]
    fn count_first(self, lanes: usize) -> u64 {
        self.0.iter().take(lanes).map(|&lane| u64::from(lane)).sum()
    }
}

impl<const W: usize> Mask for LaneMask<W> {
    #[inline(always)]
    fn splat(value: bool) -> Self {
        Self([value; W])
    }
    #[inline(always)]
    fn any(self) -> bool {
        LaneMask::any(self)
    }
    #[inline(always)]
    fn all(self) -> bool {
        LaneMask::all(self)
    }
}

impl<S: RealScalar, const W: usize> Lanes<S, W> {
    /// Returns `value` in every lane.
    #[must_use]
    #[inline(always)]
    pub const fn splat(value: S) -> Self {
        Self([value; W])
    }
    /// Builds a pack from one value per lane.
    #[must_use]
    #[inline(always)]
    pub const fn from_array(lanes: [S; W]) -> Self {
        Self(lanes)
    }
    /// Builds a pack whose lane `k` is `value(k)`.
    #[must_use]
    #[inline(always)]
    pub fn from_fn(value: impl FnMut(usize) -> S) -> Self {
        Self(arrays::build(S::ZERO, value))
    }
    /// Builds a pack from the `W` values of `lanes`.
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` if `lanes.len() != W`.
    pub fn from_slice(lanes: &[S]) -> Result<Self, SciOxideError> {
        crate::support::check_len(lanes.len(), W)?;
        Ok(Self(arrays::build(S::ZERO, |lane| lanes[lane])))
    }
    /// Returns one value per lane.
    #[must_use]
    #[inline(always)]
    pub const fn to_array(self) -> [S; W] {
        self.0
    }
    /// Returns lane `lane`, or `None` for `lane ≥ W`.
    #[must_use]
    #[inline(always)]
    pub fn lane(self, lane: usize) -> Option<S> {
        self.0.get(lane).copied()
    }
    /// Writes the `W` lanes to `out`.
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` if `out.len() != W`.
    pub fn write_to_slice(self, out: &mut [S]) -> Result<(), SciOxideError> {
        crate::support::check_len(out.len(), W)?;
        out.copy_from_slice(&self.0);
        Ok(())
    }

    /// Returns the lanes by reference.
    #[inline(always)]
    pub(crate) const fn as_array(&self) -> &[S; W] {
        &self.0
    }

    /// Returns the lanes for update in place.
    #[inline(always)]
    pub(crate) const fn as_mut_array(&mut self) -> &mut [S; W] {
        &mut self.0
    }

    #[inline(always)]
    fn map(self, f: impl Fn(S) -> S) -> Self {
        Self(arrays::map(self.0, f))
    }

    #[inline(always)]
    fn zip(self, other: Self, f: impl Fn(S, S) -> S) -> Self {
        Self(arrays::zip(self.0, other.0, f))
    }

    #[inline(always)]
    fn compare(self, other: Self, f: impl Fn(S, S) -> bool) -> LaneMask<W> {
        LaneMask(arrays::build(false, |lane| f(self.0[lane], other.0[lane])))
    }
}

macro_rules! lane_operator {
    ($Trait:ident, $method:ident, $Assign:ident, $assign:ident) => {
        impl<S: RealScalar, const W: usize> $Trait for Lanes<S, W> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, other: Self) -> Self {
                self.zip(other, $Trait::$method)
            }
        }
        impl<S: RealScalar, const W: usize> $Trait<S> for Lanes<S, W> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, other: S) -> Self {
                self.map(|a| $Trait::$method(a, other))
            }
        }
        impl<S: RealScalar, const W: usize> $Assign for Lanes<S, W> {
            #[inline(always)]
            fn $assign(&mut self, other: Self) {
                *self = $Trait::$method(*self, other);
            }
        }
        impl<S: RealScalar, const W: usize> $Assign<S> for Lanes<S, W> {
            #[inline(always)]
            fn $assign(&mut self, other: S) {
                *self = $Trait::$method(*self, other);
            }
        }
        impl<const W: usize> $Trait<Lanes<f64, W>> for f64 {
            type Output = Lanes<f64, W>;
            #[inline(always)]
            fn $method(self, other: Lanes<f64, W>) -> Lanes<f64, W> {
                $Trait::$method(Lanes::splat(self), other)
            }
        }
        impl<const W: usize> $Trait<Lanes<f32, W>> for f32 {
            type Output = Lanes<f32, W>;
            #[inline(always)]
            fn $method(self, other: Lanes<f32, W>) -> Lanes<f32, W> {
                $Trait::$method(Lanes::splat(self), other)
            }
        }
    };
}

lane_operator!(Add, add, AddAssign, add_assign);
lane_operator!(Sub, sub, SubAssign, sub_assign);
lane_operator!(Mul, mul, MulAssign, mul_assign);
lane_operator!(Div, div, DivAssign, div_assign);

impl<S: RealScalar, const W: usize> Rem for Lanes<S, W> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: Self) -> Self {
        self.zip(other, |a, b| a % b)
    }
}

impl<S: RealScalar, const W: usize> Rem<S> for Lanes<S, W> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: S) -> Self {
        self.map(|a| a % other)
    }
}

impl<const W: usize> Rem<Lanes<f64, W>> for f64 {
    type Output = Lanes<f64, W>;
    #[inline(always)]
    fn rem(self, other: Lanes<f64, W>) -> Lanes<f64, W> {
        Lanes::splat(self) % other
    }
}

impl<const W: usize> Rem<Lanes<f32, W>> for f32 {
    type Output = Lanes<f32, W>;
    #[inline(always)]
    fn rem(self, other: Lanes<f32, W>) -> Lanes<f32, W> {
        Lanes::splat(self) % other
    }
}

impl<S: RealScalar, const W: usize> Neg for Lanes<S, W> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.map(|a| -a)
    }
}

impl<S: RealScalar, const W: usize> Sum for Lanes<S, W> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::splat(S::NEG_ZERO), |sum, value| sum + value)
    }
}

impl<S: RealScalar, const W: usize> Product for Lanes<S, W> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::splat(S::ONE), |product, value| product * value)
    }
}

impl<S: RealScalar, const W: usize> sealed::Value for Lanes<S, W> {
    type Elem = S;
}

impl<S: RealScalar, const W: usize> sealed::Map<S> for Lanes<S, W> {
    #[inline]
    fn map_all<F: FnMut(S) -> [S; 3]>(_: Token, x: Self, rule: &mut F) -> [Self; 3] {
        let rules = x.0.map(rule);
        arrays::build(Self::splat(S::ZERO), |order| {
            Self(arrays::build(S::ZERO, |lane| rules[lane][order]))
        })
    }

    #[inline]
    fn select_lanes<T, P: Fn(&T) -> Self, I: FnMut(usize) -> usize>(
        _: Token,
        table: &[T],
        project: &P,
        index: &mut I,
    ) -> Self {
        Self(arrays::build(S::ZERO, |lane| {
            table
                .get(index(lane))
                .map_or(S::NAN, |entry| project(entry).0[lane])
        }))
    }
}

macro_rules! lanewise_unary {
    ($($name:ident),*) => {
        $(
            #[inline(always)]
            fn $name(self) -> Self {
                self.map(Real::$name)
            }
        )*
    };
}

macro_rules! lanewise_binary {
    ($($name:ident),*) => {
        $(
            #[inline(always)]
            fn $name(self, other: Self) -> Self {
                self.zip(other, Real::$name)
            }
        )*
    };
}

impl<S: RealScalar, const W: usize> Real<S> for Lanes<S, W> {
    type Base = Self;
    type Primal = Self;
    type Mask = LaneMask<W>;
    const LANES: usize = W;
    const LEVEL: usize = 0;

    #[inline(always)]
    fn constant(value: S) -> Self {
        Self::splat(value)
    }
    #[inline(always)]
    fn passive(base: Self) -> Self {
        base
    }
    #[inline(always)]
    fn lift(primal: Self) -> Self {
        primal
    }
    #[inline(always)]
    fn primal(self) -> Self {
        self
    }
    #[inline(always)]
    fn base(self) -> Self {
        self
    }
    #[inline(always)]
    fn lt(self, other: Self) -> LaneMask<W> {
        self.compare(other, |a, b| a < b)
    }
    #[inline(always)]
    fn le(self, other: Self) -> LaneMask<W> {
        self.compare(other, |a, b| a <= b)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> LaneMask<W> {
        self.compare(other, |a, b| a > b)
    }
    #[inline(always)]
    fn ge(self, other: Self) -> LaneMask<W> {
        self.compare(other, |a, b| a >= b)
    }
    #[inline(always)]
    fn is_finite(self) -> LaneMask<W> {
        LaneMask(arrays::build(false, |lane| {
            RealScalar::is_finite(self.0[lane])
        }))
    }
    #[inline(always)]
    fn select(mask: LaneMask<W>, if_true: Self, if_false: Self) -> Self {
        Self(arrays::build(S::ZERO, |lane| {
            if mask.0[lane] {
                if_true.0[lane]
            } else {
                if_false.0[lane]
            }
        }))
    }

    lanewise_unary!(
        recip,
        sqrt,
        cbrt,
        exp2,
        log2,
        log10,
        tan,
        asin,
        acos,
        atan,
        sinh,
        cosh,
        asinh,
        acosh,
        atanh,
        abs,
        signum,
        floor,
        ceil,
        round,
        trunc,
        fract,
        to_degrees,
        to_radians,
        softplus,
        sigmoid,
        log_sigmoid,
        logit,
        exprel,
        cosm1,
        sinc,
        sincpi,
        sin_pi,
        cos_pi,
        tan_pi,
        sin_degrees,
        cos_degrees,
        tan_degrees
    );

    #[inline(always)]
    fn exp(self) -> Self {
        Self(kernels::exp(self.0))
    }
    #[inline(always)]
    fn exp_m1(self) -> Self {
        Self(kernels::exp_m1(self.0))
    }
    #[inline(always)]
    fn ln(self) -> Self {
        Self(kernels::ln(self.0))
    }
    #[inline(always)]
    fn ln_1p(self) -> Self {
        Self(kernels::ln_1p(self.0))
    }
    #[inline(always)]
    fn sin(self) -> Self {
        Self(kernels::sin(self.0))
    }
    #[inline(always)]
    fn cos(self) -> Self {
        Self(kernels::cos(self.0))
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        Self(kernels::tanh(self.0))
    }
    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        Self(kernels::powf(self.0, other.0))
    }

    lanewise_binary!(
        hypot, atan2, min, max, log, logaddexp, xlogy, xlog1py, powm1
    );

    #[inline(always)]
    fn powi(self, exponent: i32) -> Self {
        self.map(|a| a.powi(exponent))
    }
    #[inline(always)]
    fn clamp(self, minimum: Self, maximum: Self) -> Self {
        Self(arrays::build(S::ZERO, |lane| {
            RealScalar::clamp(self.0[lane], minimum.0[lane], maximum.0[lane])
        }))
    }
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        (Real::sin(self), Real::cos(self))
    }
    #[inline(always)]
    fn azmul(self, other: Self) -> Self {
        self.zip(other, crate::scalar::azmul)
    }

    fn combine(arguments: &[Self], value: Self, partials: &[Self]) -> Result<Self, SciOxideError> {
        crate::support::check_len(partials.len(), arguments.len())?;
        Ok(value)
    }
}

impl<S: RealScalar, const W: usize> BaseReal<S> for Lanes<S, W> {
    #[inline(always)]
    fn from_fn(value: impl FnMut(usize) -> S) -> Self {
        Self(arrays::build(S::ZERO, value))
    }
    #[inline(always)]
    fn lane(self, lane: usize) -> Option<S> {
        self.0.get(lane).copied()
    }
}
