//! Reverse-mode variables recorded on a tape.

use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

use scioxide_core::{RealScalar, SciOxideError};

use super::TapeValue;
use super::recorder::{MIXED, Recorder};
use super::word::Kind;
use crate::Real;
use crate::real::sealed::{self, MaskLanes, Token};
use crate::rules::{self, Chain, Class, Product as Apply};
use crate::{BaseReal, Mask};

/// A value recorded on a [`Tape`](crate::Tape): its value and the slot of the
/// statement that produced it.
///
/// A transform creates the inputs of each recording and hands them to the
/// function; every operation with an active operand appends one statement (see
/// the [emission rules](crate#tapes)). An operation whose operands are all passive
/// records nothing and returns a passive value, as do [`Real::constant`] and
/// [`Real::passive`]. A `Var` is neither `Send` nor `Sync`, and one used outside its
/// recording does not compile:
///
/// ```compile_fail,E0521
/// use scioxide_ad::{Tape, TapeOptions, Var};
/// use scioxide_core::ExecutionContext;
///
/// let mut tape = Tape::<f64>::new(TapeOptions::new());
/// let mut kept = None;
/// let mut gradient = [0.0];
/// tape.gradient_into(
///     |x| {
///         kept = Some(x[0]);
///         Ok(x[0])
///     },
///     &[1.0],
///     &mut gradient,
///     ExecutionContext::sequential(),
/// )?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Var<'t, V: TapeValue> {
    value: V,
    slot: u32,
    recorder: Option<&'t Recorder<'t, V>>,
}

impl<V: TapeValue> fmt::Debug for Var<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Var")
            .field("value", &self.value)
            .field("active", &self.recorder.is_some())
            .finish()
    }
}

/// How two operands relate to recorders.
enum Pair<'t, V: TapeValue> {
    Passive,
    Left,
    Right,
    Both(&'t Recorder<'t, V>),
    Mixed,
}

impl<'t, V: TapeValue> Var<'t, V> {
    /// Returns the value.
    #[must_use]
    #[inline(always)]
    pub const fn value(self) -> V {
        self.value
    }

    /// Returns whether the value carries a slot of its recording.
    ///
    /// A value computed from active inputs is active, except the results of
    /// `floor`, `ceil`, `round`, `trunc` and `signum`, whose derivative is zero and
    /// which are passive.
    ///
    /// ```
    /// use scioxide_ad::{Real, grad};
    /// grad(|x| {
    ///     assert!(x[0].is_active() && !x[0].floor().is_active());
    ///     Ok(x[0] + x[0].floor())
    /// }, &[2.5])?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn is_active(self) -> bool {
        self.recorder.is_some()
    }

    #[inline(always)]
    pub(crate) const fn active(value: V, slot: u32, recorder: &'t Recorder<'t, V>) -> Self {
        Self {
            value,
            slot,
            recorder: Some(recorder),
        }
    }

    #[inline(always)]
    pub(crate) const fn passive_value(value: V) -> Self {
        Self {
            value,
            slot: 0,
            recorder: None,
        }
    }

    /// The slot of an active value, or the constant slot 0.
    #[inline(always)]
    pub(crate) const fn slot(self) -> u32 {
        self.slot
    }

    #[inline(always)]
    pub(crate) const fn recorder(self) -> Option<&'t Recorder<'t, V>> {
        self.recorder
    }

    #[inline(always)]
    fn with_value(self, value: V) -> Self {
        Self { value, ..self }
    }

    #[inline(always)]
    fn nan() -> Self {
        Self::passive_value(V::constant(<V::Scalar as RealScalar>::NAN))
    }

    #[inline(always)]
    fn pair(self, other: Self) -> Pair<'t, V> {
        match (self.recorder, other.recorder) {
            (None, None) => Pair::Passive,
            (Some(_), None) => Pair::Left,
            (None, Some(_)) => Pair::Right,
            (Some(r), Some(q)) if core::ptr::eq(r, q) => Pair::Both(r),
            (Some(r), Some(q)) => {
                r.fault(MIXED);
                q.fault(MIXED);
                Pair::Mixed
            }
        }
    }

    /// Records a unary statement on `self` with a full partial.
    #[inline(always)]
    fn full(self, value: V, partial: V, class: Class) -> Self {
        match self.recorder {
            Some(r) => {
                r.push_full(partial);
                Self::active(value, r.unary(Kind::Full, class, self.slot), r)
            }
            None => Self::passive_value(value),
        }
    }

    /// Records a unary statement on `self` with a scalar partial.
    #[inline(always)]
    fn scaled(self, value: V, partial: V::Scalar) -> Self {
        match self.recorder {
            Some(r) => {
                r.push_scalar(partial);
                Self::active(value, r.unary(Kind::Scalar, Class::Linear, self.slot), r)
            }
            None => Self::passive_value(value),
        }
    }

    /// Records `−self` with the given value.
    #[inline(always)]
    fn negated(self, value: V) -> Self {
        match self.recorder {
            Some(r) => Self::active(value, r.unary(Kind::MinusOne, Class::Linear, self.slot), r),
            None => Self::passive_value(value),
        }
    }

    /// Records a statement with full partials `left` and `right`, or a unary one when
    /// an operand is passive. A product with one passive operand is linear.
    #[inline(always)]
    fn pair_full(self, other: Self, value: V, left: V, right: V, class: Class) -> Self {
        let linear = |class| match class {
            Class::Bilinear => Class::Linear,
            class => class,
        };
        match self.pair(other) {
            Pair::Passive => Self::passive_value(value),
            Pair::Left => self.full(value, left, linear(class)),
            Pair::Right => other.full(value, right, linear(class)),
            Pair::Both(r) => {
                r.push_full(left);
                r.push_full(right);
                let slot = r.binary([Kind::Full, Kind::Full], class, [self.slot, other.slot]);
                Self::active(value, slot, r)
            }
            Pair::Mixed => Self::nan(),
        }
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Chain<S> for Var<'_, V> {
    #[inline(always)]
    fn is_active(self) -> bool {
        self.recorder.is_some()
    }
    #[inline(always)]
    fn unary(self, value: V, partial: V, _: Apply, class: Class) -> Self {
        self.full(value, partial, class)
    }
    #[inline(always)]
    fn binary(self, other: Self, value: V, left: V, right: V, _: Apply, class: Class) -> Self {
        self.pair_full(other, value, left, right, class)
    }
    #[inline(always)]
    fn kinks(self, hits: V::Mask) {
        if let Some(r) = self.recorder {
            let count = hits.count_first(r.valid_lanes());
            if count != 0 {
                r.add_kinks(count);
            }
        }
    }
    #[inline(always)]
    fn pair_kinks(self, other: Self, hits: V::Mask) {
        Self {
            recorder: self.recorder.or(other.recorder),
            ..self
        }
        .kinks(hits);
    }
}

impl<V: TapeValue> sealed::Value for Var<'_, V> {
    type Elem = V::Scalar;
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> sealed::Map<S> for Var<'_, V> {
    /// Records every supplied derivative so `Dual<Var>`, `MultiDual<Var>` and
    /// `Jet<Var>` retain mixed derivatives through total order two.
    #[inline]
    fn map_all<F: FnMut(S) -> [S; 3]>(token: Token, x: Self, rule: &mut F) -> [Self; 3] {
        let [f0, f1, f2] = V::map_all(token, x.value, rule);
        [
            x.full(f0, f1, Class::Nonlinear),
            x.full(f1, f2, Class::Nonlinear),
            x.full(f2, V::constant(S::NAN), Class::Nonlinear),
        ]
    }

    fn select_lanes<T, P: Fn(&T) -> Self, I: FnMut(usize) -> usize>(
        token: Token,
        table: &[T],
        project: &P,
        index: &mut I,
    ) -> Self {
        let value = V::select_lanes(token, table, &|entry| project(entry).value, index);
        let entry = |lane: usize, index: &mut I| table.get(index(lane)).map(project);
        if V::LANES == 1 {
            return entry(0, index).map_or_else(Self::nan, |e| e.with_value(value));
        }
        let mut recorder = None;
        for lane in 0..V::LANES {
            if let Some(Self {
                recorder: Some(r), ..
            }) = entry(lane, index)
            {
                match recorder {
                    None => recorder = Some(r),
                    Some(q) if core::ptr::eq(q, r) => {}
                    Some(q) => {
                        q.fault(MIXED);
                        r.fault(MIXED);
                        return Self::nan();
                    }
                }
            }
        }
        let Some(r) = recorder else {
            return Self::passive_value(value);
        };
        let one_hot = |lane: usize| {
            V::passive(<V as Real<S>>::Base::from_fn(|k| {
                if k == lane { S::ONE } else { S::ZERO }
            }))
        };
        let arguments = (0..V::LANES).map(|lane| {
            let slot = entry(lane, index).map_or(0, |e| e.slot);
            (slot, Some(one_hot(lane)))
        });
        let slot = r.extended(None, Class::Linear, V::LANES, arguments);
        Self::active(value, slot, r)
    }
}

impl<V: TapeValue> Add for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let value = self.value + other.value;
        match self.pair(other) {
            Pair::Passive => Self::passive_value(value),
            Pair::Left => self.with_value(value),
            Pair::Right => other.with_value(value),
            Pair::Both(r) => Self::active(
                value,
                r.binary(
                    [Kind::One, Kind::One],
                    Class::Linear,
                    [self.slot, other.slot],
                ),
                r,
            ),
            Pair::Mixed => Self::nan(),
        }
    }
}

impl<V: TapeValue> Sub for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        let value = self.value - other.value;
        match self.pair(other) {
            Pair::Passive => Self::passive_value(value),
            Pair::Left => self.with_value(value),
            Pair::Right => other.negated(value),
            Pair::Both(r) => Self::active(
                value,
                r.binary(
                    [Kind::One, Kind::MinusOne],
                    Class::Linear,
                    [self.slot, other.slot],
                ),
                r,
            ),
            Pair::Mixed => Self::nan(),
        }
    }
}

impl<V: TapeValue> Mul for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        let (x, y) = (self.value, other.value);
        self.pair_full(other, x * y, y, x, Class::Bilinear)
    }
}

impl<V: TapeValue> Div for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        let (x, y) = (self.value, other.value);
        let q = x / y;
        match self.pair(other) {
            Pair::Passive => Self::passive_value(q),
            // x/p has the constant partial 1/p.
            Pair::Left => self.full(q, y.recip(), Class::Linear),
            Pair::Right => other.full(q, -(q / y), Class::Nonlinear),
            Pair::Both(_) => self.pair_full(other, q, y.recip(), -(q / y), Class::Nonlinear),
            Pair::Mixed => Self::nan(),
        }
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Rem for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: Self) -> Self {
        rules::rem(self, other)
    }
}

impl<V: TapeValue> Neg for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.negated(-self.value)
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Add<S> for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: S) -> Self {
        self.with_value(self.value + other)
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Sub<S> for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: S) -> Self {
        self.with_value(self.value - other)
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Mul<S> for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: S) -> Self {
        self.scaled(self.value * other, other)
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Div<S> for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: S) -> Self {
        self.scaled(self.value / other, other.recip())
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Rem<S> for Var<'_, V> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: S) -> Self {
        rules::rem(self, Self::constant(other))
    }
}

macro_rules! assign {
    ($($Assign:ident $assign:ident $Trait:ident $method:ident),*) => {
        $(
            impl<S: RealScalar, V: TapeValue<Scalar = S>> $Assign for Var<'_, V> {
                #[inline(always)]
                fn $assign(&mut self, other: Self) {
                    *self = $Trait::$method(*self, other);
                }
            }
            impl<S: RealScalar, V: TapeValue<Scalar = S>> $Assign<S> for Var<'_, V> {
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

macro_rules! scalar_left {
    ($($S:ty),*) => {
        $(
            impl<'t, V: TapeValue<Scalar = $S>> Add<Var<'t, V>> for $S {
                type Output = Var<'t, V>;
                #[inline(always)]
                fn add(self, other: Var<'t, V>) -> Var<'t, V> {
                    other.with_value(V::constant(self) + other.value)
                }
            }
            impl<'t, V: TapeValue<Scalar = $S>> Sub<Var<'t, V>> for $S {
                type Output = Var<'t, V>;
                #[inline(always)]
                fn sub(self, other: Var<'t, V>) -> Var<'t, V> {
                    other.negated(V::constant(self) - other.value)
                }
            }
            impl<'t, V: TapeValue<Scalar = $S>> Mul<Var<'t, V>> for $S {
                type Output = Var<'t, V>;
                #[inline(always)]
                fn mul(self, other: Var<'t, V>) -> Var<'t, V> {
                    other.scaled(V::constant(self) * other.value, self)
                }
            }
            impl<'t, V: TapeValue<Scalar = $S>> Div<Var<'t, V>> for $S {
                type Output = Var<'t, V>;
                #[inline(always)]
                fn div(self, other: Var<'t, V>) -> Var<'t, V> {
                    Var::constant(self) / other
                }
            }
            impl<'t, V: TapeValue<Scalar = $S>> Rem<Var<'t, V>> for $S {
                type Output = Var<'t, V>;
                #[inline(always)]
                fn rem(self, other: Var<'t, V>) -> Var<'t, V> {
                    Var::constant(self) % other
                }
            }
        )*
    };
}

scalar_left!(f32, f64);

impl<S: RealScalar, V: TapeValue<Scalar = S>> Sum for Var<'_, V> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::constant(S::NEG_ZERO), |sum, value| sum + value)
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Product for Var<'_, V> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |product, value| product * value)
    }
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Real<S> for Var<'_, V> {
    type Base = V::Base;
    type Primal = V;
    type Mask = V::Mask;
    const LANES: usize = V::LANES;
    const LEVEL: usize = V::LEVEL + 1;

    #[inline(always)]
    fn constant(value: S) -> Self {
        Self::passive_value(V::constant(value))
    }
    #[inline(always)]
    fn passive(base: V::Base) -> Self {
        Self::passive_value(V::passive(base))
    }
    #[inline(always)]
    fn lift(primal: V) -> Self {
        Self::passive_value(primal)
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
        self.value.is_finite()
    }
    #[inline(always)]
    fn select(mask: V::Mask, if_true: Self, if_false: Self) -> Self {
        if mask.all() {
            return if_true;
        }
        if !mask.any() {
            return if_false;
        }
        let one = V::one();
        let zero = V::zero();
        if_true.pair_full(
            if_false,
            V::select(mask, if_true.value, if_false.value),
            V::select(mask, one, zero),
            V::select(mask, zero, one),
            Class::Linear,
        )
    }

    crate::forward::chain_methods!();

    #[inline(always)]
    fn azmul(self, other: Self) -> Self {
        let (x, y) = (self.value, other.value);
        self.pair_full(other, x.azmul(y), y, x, Class::Bilinear)
    }

    // A direct scalar map needs only the value statement. `Map::map_all` also
    // records derivative statements when a forward number wraps this variable.
    #[inline]
    fn map_scalar(self, mut rule: impl FnMut(S) -> [S; 3]) -> Self {
        let [value, partial, _] = V::map_all(Token(()), self.value, &mut rule);
        self.full(value, partial, Class::Nonlinear)
    }

    fn combine(arguments: &[Self], value: V, partials: &[V]) -> Result<Self, SciOxideError> {
        crate::support::check_len(partials.len(), arguments.len())?;
        let mut recorder = None;
        for argument in arguments {
            if let Some(r) = argument.recorder {
                match recorder {
                    None => recorder = Some(r),
                    Some(q) if core::ptr::eq(q, r) => {}
                    Some(q) => {
                        q.fault(MIXED);
                        r.fault(MIXED);
                        return Ok(Self::nan());
                    }
                }
            }
        }
        let Some(r) = recorder else {
            return Ok(Self::passive_value(value));
        };
        let pairs = arguments
            .iter()
            .zip(partials)
            .map(|(argument, &partial)| (argument.slot, Some(partial)));
        let slot = r.extended(None, Class::Nonlinear, arguments.len(), pairs);
        Ok(Self::active(value, slot, r))
    }
}
