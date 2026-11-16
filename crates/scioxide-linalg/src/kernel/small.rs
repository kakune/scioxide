//! Complete-pivoting solves through order eight, with ordinary or extended arithmetic.

use scioxide_core::{NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError};

/// Whether an exactly singular local equation may return a basic solution.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Singularity {
    Reject,
    /// Accept exact consistency and set free variables to zero (matrix roots).
    Consistent,
}

/// Arithmetic policy shared by the complete-pivoting and substitution loops.
pub(super) trait Arithmetic<S: NumericScalar> {
    type Value: Copy;
    type Magnitude: Copy;
    const ZERO: Self::Value;
    fn value(&self, x: S) -> Result<Self::Value, SciOxideError>;
    fn magnitude(&self, x: Self::Value) -> Result<Self::Magnitude, SciOxideError>;
    fn greater(&self, a: Self::Magnitude, b: Self::Magnitude) -> bool;
    fn zero(&self, x: Self::Value) -> bool;
    fn add(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value, SciOxideError>;
    fn mul(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value, SciOxideError>;
    fn div(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value, SciOxideError>;
    fn subtract_product(
        &self,
        a: Self::Value,
        b: Self::Value,
        c: Self::Value,
    ) -> Result<Self::Value, SciOxideError>;
}
pub(super) struct Ordinary<R> {
    lower: R,
    upper: R,
}
pub(super) struct ExtendedArithmetic<V>(pub(super) core::marker::PhantomData<V>);
impl<R: RealScalar> Ordinary<R> {
    pub(super) fn new() -> Result<Self, SciOxideError> {
        // Evaluate scale bounds once per leaf, never in the coupling loop.
        let bound = super::frames::range_frame_bound::<R>()?;
        Ok(Self {
            lower: R::ONE.scalbn(-bound),
            upper: R::ONE.scalbn(bound),
        })
    }
    pub(super) fn check<S: NumericScalar<Real = R>>(&self, value: S) -> Result<S, SciOxideError> {
        if [value.real_component(), value.imaginary_component()]
            .into_iter()
            .all(|v| v == R::ZERO || (v.abs() >= self.lower && v.abs() <= self.upper))
        {
            Ok(value)
        } else {
            Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "local equation needs a wider binary frame",
            ))
        }
    }
}
impl<S: NumericScalar> Arithmetic<S> for Ordinary<S::Real> {
    type Value = S;
    type Magnitude = S::Real;
    const ZERO: S = S::ZERO;
    fn value(&self, x: S) -> Result<S, SciOxideError> {
        self.check(x)
    }
    fn magnitude(&self, x: S) -> Result<S::Real, SciOxideError> {
        Ok(x.abs())
    }
    fn greater(&self, a: S::Real, b: S::Real) -> bool {
        a > b
    }
    fn zero(&self, x: S) -> bool {
        x == S::ZERO
    }
    fn add(&self, a: S, b: S) -> Result<S, SciOxideError> {
        self.check(a + b)
    }
    fn mul(&self, a: S, b: S) -> Result<S, SciOxideError> {
        self.check(a * b)
    }
    fn div(&self, a: S, b: S) -> Result<S, SciOxideError> {
        self.check(a / b)
    }
    fn subtract_product(&self, a: S, b: S, c: S) -> Result<S, SciOxideError> {
        self.check(a - b * c)
    }
}
impl<S: NumericScalar, V: NumericScalar<Real = S::Real>> Arithmetic<S> for ExtendedArithmetic<V> {
    type Value = ScaledValue<V>;
    type Magnitude = Scaled<S::Real>;
    const ZERO: Self::Value = ScaledValue::ZERO;
    fn value(&self, x: S) -> Result<Self::Value, SciOxideError> {
        ScaledValue::from_value(V::from_components(
            x.real_component(),
            x.imaginary_component(),
        ))
    }
    fn magnitude(&self, x: Self::Value) -> Result<Self::Magnitude, SciOxideError> {
        x.magnitude()
    }
    fn greater(&self, a: Self::Magnitude, b: Self::Magnitude) -> bool {
        !a.abs_le(b)
    }
    fn zero(&self, x: Self::Value) -> bool {
        x.is_zero()
    }
    fn add(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value, SciOxideError> {
        a.add(b)
    }
    fn mul(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value, SciOxideError> {
        a.mul(b)
    }
    fn div(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value, SciOxideError> {
        a.div(b)
    }
    fn subtract_product(
        &self,
        a: Self::Value,
        b: Self::Value,
        c: Self::Value,
    ) -> Result<Self::Value, SciOxideError> {
        a.sub(b.mul(c)?)
    }
}

/// One complete-pivoting solve for both ordinary and extended arithmetic.
pub(super) fn eliminate<S: NumericScalar, A: Arithmetic<S>>(
    arithmetic: &A,
    a: &mut [A::Value],
    b: &mut [A::Value],
    n: usize,
    singularity: Singularity,
) -> Result<bool, SciOxideError> {
    let stride = b.len();
    if n > 8 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("n"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "complete-pivoting solves have at most eight unknowns",
        ));
    }
    if n > stride {
        return Err(SciOxideError::dimension_mismatch([n], [stride]));
    }
    if a.len() < stride.saturating_mul(stride) {
        return Err(SciOxideError::dimension_mismatch(
            [stride.saturating_mul(stride)],
            [a.len()],
        ));
    }
    if n == 1 {
        if arithmetic.zero(a[0]) {
            return Ok(singularity == Singularity::Consistent && arithmetic.zero(b[0]));
        }
        b[0] = arithmetic.div(b[0], a[0])?;
        return Ok(true);
    }
    let mut rank = n;
    let mut permutation = [0, 1, 2, 3, 4, 5, 6, 7];
    for k in 0..n {
        let mut pivot = (k, k);
        let mut maximum = arithmetic.magnitude(A::ZERO)?;
        for i in k..n {
            for j in k..n {
                let value = arithmetic.magnitude(a[i * stride + j])?;
                if arithmetic.greater(value, maximum) {
                    maximum = value;
                    pivot = (i, j);
                }
            }
        }
        if arithmetic.zero(a[pivot.0 * stride + pivot.1]) {
            if singularity == Singularity::Reject || b[k..n].iter().any(|&v| !arithmetic.zero(v)) {
                return Ok(false);
            }
            b[k..n].fill(A::ZERO);
            rank = k;
            break;
        }
        for j in 0..n {
            a.swap(k * stride + j, pivot.0 * stride + j);
        }
        b.swap(k, pivot.0);
        for i in 0..n {
            a.swap(i * stride + k, i * stride + pivot.1);
        }
        permutation.swap(k, pivot.1);
        for i in k + 1..n {
            let factor = arithmetic.div(a[i * stride + k], a[k * stride + k])?;
            for j in k + 1..n {
                a[i * stride + j] =
                    arithmetic.subtract_product(a[i * stride + j], factor, a[k * stride + j])?;
            }
            b[i] = arithmetic.subtract_product(b[i], factor, b[k])?;
        }
    }
    for i in (0..rank).rev() {
        for j in i + 1..rank {
            b[i] = arithmetic.subtract_product(b[i], a[i * stride + j], b[j])?;
        }
        b[i] = arithmetic.div(b[i], a[i * stride + i])?;
    }
    let mut original = [A::ZERO; 8];
    original[..n].copy_from_slice(&b[..n]);
    for i in 0..n {
        b[permutation[i]] = original[i];
    }
    Ok(true)
}

pub(crate) fn small<V: NumericScalar>(
    a: &mut [ScaledValue<V>],
    b: &mut [ScaledValue<V>],
    n: usize,
    singularity: Singularity,
) -> Result<bool, SciOxideError> {
    eliminate::<V, ExtendedArithmetic<V>>(
        &ExtendedArithmetic(core::marker::PhantomData),
        a,
        b,
        n,
        singularity,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::kernel::frames::{extended, scalar};

    #[test]
    fn consistent_singular_system_undoes_column_pivots() {
        for policy in [Singularity::Reject, Singularity::Consistent] {
            for residual in [0., 1.] {
                let mut a = [0., 2., 0., 0.].map(|x: f64| extended(x).unwrap());
                let mut b = [6., residual].map(|x| extended(x).unwrap());
                let accepted = small(&mut a, &mut b, 2, policy).unwrap();
                assert_eq!(
                    accepted,
                    policy == Singularity::Consistent && residual == 0.
                );
                if accepted {
                    assert_eq!(scalar::<f64>(b[0]).unwrap(), 0.);
                    assert!((scalar::<f64>(b[1]).unwrap() - 3.).abs() <= f64::EPSILON);
                }
            }
        }
    }
}
