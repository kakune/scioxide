//! The values a tape can record.

use scioxide_core::RealScalar;

use crate::{Dual, Lanes, Real};

mod sealed {
    pub trait Recordable {}
}

/// A number type a [`Tape`](crate::Tape) can record, naming its IEEE scalar: `f32`,
/// `f64`, [`Lanes`], and [`Dual`], [`MultiDual`](crate::MultiDual) and
/// [`Jet`](crate::Jet) over those. Sealed.
pub trait TapeValue:
    Real<<Self as TapeValue>::Scalar> + sealed::Recordable + Send + Sync + 'static
{
    /// The IEEE scalar.
    type Scalar: RealScalar;
}

impl<S: RealScalar> sealed::Recordable for S {}
impl<S: RealScalar> TapeValue for S {
    type Scalar = S;
}

impl<S: RealScalar, const W: usize> sealed::Recordable for Lanes<S, W> {}
impl<S: RealScalar, const W: usize> TapeValue for Lanes<S, W> {
    type Scalar = S;
}

impl<V: TapeValue> sealed::Recordable for Dual<V> {}
impl<V: TapeValue> TapeValue for Dual<V> {
    type Scalar = V::Scalar;
}

impl<V: TapeValue, const N: usize> sealed::Recordable for crate::MultiDual<V, N> {}
impl<V: TapeValue, const N: usize> TapeValue for crate::MultiDual<V, N> {
    type Scalar = V::Scalar;
}

impl<V: TapeValue, const N: usize> sealed::Recordable for crate::Jet<V, N> {}
impl<V: TapeValue, const N: usize> TapeValue for crate::Jet<V, N> {
    type Scalar = V::Scalar;
}
