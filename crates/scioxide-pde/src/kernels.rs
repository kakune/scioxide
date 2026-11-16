//! Prepared range admission for the crate's short sparse stencil rows.

use alloc::vec::Vec;
use scioxide_core::{Accuracy, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError};

/// Native stencil columns use four-byte indices when the prepared shape fits.
/// Canonical CSR retains its public usize indices for assembly and replay.
pub(crate) fn compact_columns<V: NumericScalar>(
    matrix: &scioxide_sparse::CsrMatrix<V>,
) -> Result<Option<Vec<u32>>, SciOxideError> {
    if core::mem::size_of::<usize>() <= core::mem::size_of::<u32>()
        || u32::try_from(matrix.cols()).is_err()
    {
        return Ok(None);
    }
    let mut columns = scioxide_core::backend::try_with_capacity(
        matrix.nnz(),
        scioxide_core::quantity!("stencil_columns"),
    )?;
    // Every canonical column is below the admitted u32 dimension.
    columns.extend(matrix.col_indices().iter().map(|&i| i as u32));
    Ok(Some(columns))
}

pub(crate) trait StencilIndex: Copy {
    fn index(self) -> usize;
}
impl StencilIndex for usize {
    #[inline]
    fn index(self) -> usize {
        self
    }
}
impl StencilIndex for u32 {
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

/// Component bounds that prove every nonzero product is normal and every row
/// has accumulation headroom. Extreme inputs retain the checked row reduction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowBounds<R> {
    lower: R,
    upper: R,
}

impl<R: RealScalar> RowBounds<R> {
    pub(crate) fn new<V: NumericScalar<Real = R>>(values: &[V], pointers: &[usize]) -> Self {
        let mut minimum = R::MAX;
        let mut maximum = R::ZERO;
        for value in values {
            for component in [
                value.real_component().abs(),
                value.imaginary_component().abs(),
            ] {
                maximum = maximum.max(component);
                if component != R::ZERO {
                    minimum = minimum.min(component);
                }
            }
        }
        if maximum == R::ZERO {
            return Self {
                lower: R::ZERO,
                upper: R::MAX,
            };
        }
        let width = pointers.windows(2).map(|p| p[1] - p[0]).max().unwrap_or(0);
        // A power-of-two upper bound avoids narrowing the integer row width.
        // Three additional bits cover complex cross products and rounding.
        let growth = usize::BITS - width.max(1).leading_zeros() + 3;
        let mut headroom = R::MAX;
        for _ in 0..growth {
            headroom *= R::HALF;
        }
        Self {
            lower: (R::MIN_POSITIVE / minimum) * R::TWO,
            upper: (headroom / maximum).min(R::MAX),
        }
    }

    /// Validate every input, including entries unused by the stencil.
    pub(crate) fn admits<V: NumericScalar<Real = R>>(
        &self,
        input: &[V],
    ) -> Result<bool, SciOxideError> {
        let admitted = input.iter().all(|value| {
            let within = |component: R| {
                let magnitude = component.abs();
                magnitude == R::ZERO || (magnitude >= self.lower && magnitude <= self.upper)
            };
            within(value.real_component()) && (V::IS_REAL || within(value.imaginary_component()))
        });
        if !admitted {
            crate::validation::finite_slice(input, NonFiniteOrigin::Input)?;
        }
        Ok(admitted)
    }
}

/// One native stencil row, admitted by the prepared component bounds.
#[inline]
pub(crate) fn row<V: NumericScalar, I: StencilIndex>(
    prior: V,
    values: &[V],
    columns: &[I],
    input: &[V],
) -> V {
    // Pair sums expose independent arithmetic in the common 1D/2D/3D rows.
    // Slice patterns establish the fixed coefficient/index bounds first.
    let sum = match (values, columns) {
        ([a, b, c, d, e], [i, j, k, l, m]) => {
            (*a * input[i.index()] + *b * input[j.index()])
                + (*c * input[k.index()] + *d * input[l.index()])
                + *e * input[m.index()]
        }
        ([a, b, c], [i, j, k]) => {
            (*a * input[i.index()] + *b * input[j.index()]) + *c * input[k.index()]
        }
        ([a, b, c, d, e, f, g], [i, j, k, l, m, n, o]) => {
            ((*a * input[i.index()] + *b * input[j.index()])
                + (*c * input[k.index()] + *d * input[l.index()]))
                + ((*e * input[m.index()] + *f * input[n.index()]) + *g * input[o.index()])
        }
        ([a], [i]) => *a * input[i.index()],
        _ => values
            .iter()
            .zip(columns)
            .fold(V::ZERO, |sum, (&a, &j)| sum + a * input[j.index()]),
    };
    prior + sum
}

/// Range replay includes the original load, so cancellation with it is retained.
#[cold]
pub(crate) fn checked_row<V: NumericScalar>(
    prior: V,
    values: &[V],
    columns: &[usize],
    input: &[V],
) -> Result<V, SciOxideError> {
    scioxide_core::backend::standard::sum_products(
        core::iter::once((prior, V::ONE))
            .chain(values.iter().zip(columns).map(|(&a, &j)| (a, input[j]))),
        Accuracy::Standard,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::discretization::AssembledOperator;
    use alloc::vec;
    use scioxide_core::AlgebraicScalar;

    fn operator<V: NumericScalar>(values: &[V; 3]) -> AssembledOperator<V::Real, V> {
        let matrix = scioxide_sparse::CsrMatrix::from_raw(
            2,
            2,
            vec![0, 1, 3],
            vec![0, 0, 1],
            values.to_vec(),
        )
        .unwrap();
        AssembledOperator {
            bounds: RowBounds::new(matrix.values(), matrix.row_ptr()),
            compact_columns: compact_columns(&matrix).unwrap(),
            matrix,
            constrained: vec![false; 2],
            mass: vec![V::Real::ONE; 2],
            components: 1,
            constant_modes: false,
            entry_bound: 3,
        }
    }

    #[test]
    fn stop_at_commit_preserves_output() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        struct Stop(AtomicUsize);
        impl scioxide_core::Cancellation for Stop {
            fn is_cancelled(&self) -> bool {
                self.0.fetch_add(1, Ordering::Relaxed) != 0
            }
        }
        let stop = Stop(AtomicUsize::new(0));
        let execution = scioxide_core::ExecutionContext::sequential().with_cancellation(&stop);
        let plan = operator(&[1.0_f64, 1.0, -1.0]);
        let mut out = [7.0; 2];
        assert!(
            plan.apply(&[2.0, 1.0], &mut out, &mut [0.0; 2], Some(execution))
                .is_err()
        );
        assert_eq!(out, [7.0; 2]);
    }
}
