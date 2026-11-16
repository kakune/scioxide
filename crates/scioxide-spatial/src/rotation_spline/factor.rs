//! Inline order-three LU, matching the default linalg factor and solve arithmetic.
use crate::{math3::Mat, validation::finite};
use scioxide_core::{
    RealScalar, SciOxideError,
    backend::{KernelBackend, standard},
};

pub(super) struct Factor<S: RealScalar> {
    entries: Mat<S>,
    permutation: [usize; 3],
    rows: [i32; 3],
    columns: [i32; 3],
    maximum: S,
}

impl<S: RealScalar> Factor<S> {
    pub(super) fn new(mut entries: Mat<S>) -> Result<Self, SciOxideError> {
        let mut rows = [0; 3];
        let mut columns = [0; 3];
        let mut permutation = [0, 1, 2];
        for (row, exponent) in entries.iter_mut().zip(&mut rows) {
            let maximum = row.iter().fold(S::ZERO, |m, x| m.max(x.abs()));
            *exponent = power(maximum)?;
            *row = finite(row.map(|x| standard::scale(x, *exponent)))?;
        }
        for (column, exponent) in columns.iter_mut().enumerate() {
            let maximum = entries
                .iter()
                .fold(S::ZERO, |m, row| m.max(row[column].abs()));
            *exponent = power(maximum)?;
            for row in &mut entries {
                row[column] = standard::scale(row[column], *exponent);
            }
        }
        for column in 0..3 {
            let mut pivot = column;
            for row in column + 1..3 {
                if entries[row][column].abs() > entries[pivot][column].abs() {
                    pivot = row;
                }
            }
            if entries[pivot][column] == S::ZERO {
                return Err(SciOxideError::singular(None));
            }
            entries.swap(column, pivot);
            permutation.swap(column, pivot);
            for row in column + 1..3 {
                entries[row][column] /= entries[column][column];
            }
            for next in column + 1..3 {
                for row in column + 1..3 {
                    entries[row][next] -= entries[row][column] * entries[column][next];
                }
            }
        }
        for row in entries {
            finite(row)?;
        }
        let maximum = entries
            .iter()
            .flatten()
            .fold(S::ZERO, |m, x| m.max(x.abs()));
        Ok(Self {
            entries,
            permutation,
            rows,
            columns,
            maximum,
        })
    }

    pub(super) fn solve(
        &self,
        rhs: [S; 3],
        backend: KernelBackend,
    ) -> Result<[S; 3], SciOxideError> {
        let mut values: [S; 3] = finite(core::array::from_fn(|i| {
            let source = self.permutation[i];
            standard::scale(rhs[source], self.rows[source])
        }))?;
        for lower in [true, false] {
            let mut solved_maximum = S::ZERO;
            for step in 0..3 {
                let i = if lower { step } else { 2 - step };
                let known = if lower { 0..i } else { i + 1..3 };
                let products = known.map(|j| (self.entries[i][j], values[j]));
                let prior = values[i];
                let residual = residual(prior, products.clone(), backend);
                let pivot = if lower { S::ONE } else { self.entries[i][i] };
                let value = standard::finish_substitution(
                    prior,
                    pivot,
                    residual,
                    products,
                    Some((self.maximum, solved_maximum)),
                )?;
                values[i] = value;
                solved_maximum = solved_maximum.max(value.abs());
            }
        }
        finite(core::array::from_fn(|i| {
            standard::scale(values[i], self.columns[i])
        }))
    }
}

fn power<S: RealScalar>(maximum: S) -> Result<i32, SciOxideError> {
    if maximum == S::ZERO {
        Ok(0)
    } else {
        Ok(1 - standard::exponent(maximum)?)
    }
}

// The general triangular kernel has at most two known values in this system.
// Preserve both its portable lanes and its selected ISA's short-dot rounding.
fn residual<S: RealScalar>(
    prior: S,
    products: impl ExactSizeIterator<Item = (S, S)>,
    backend: KernelBackend,
) -> S {
    if products.len() == 0 {
        return prior;
    }
    if backend == KernelBackend::Scalar {
        let mut sums = [prior, S::ZERO, S::ZERO, S::ZERO];
        for (i, (a, b)) in products.enumerate() {
            sums[i] -= a * b;
        }
        (sums[0] + sums[1]) + (sums[2] + sums[3])
    } else {
        let vector_pair =
            backend == KernelBackend::Neon && S::MANTISSA_DIGITS == 53 && products.len() == 2;
        let sum = products.fold(S::ZERO, |sum, (a, b)| {
            if vector_pair {
                sum + a.mul_add(b, S::ZERO)
            } else {
                a.mul_add(b, sum)
            }
        });
        prior - sum
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        DirectSolveOptions, DirectSolveTermination, LinalgExecutionOptions, LinalgWorkspace,
        LuDecomposition, Matrix, RankPolicy, VectorView, VectorViewMut,
    };

    fn compare<S: RealScalar>(context: ExecutionContext<'_>) {
        let scalar = |x| S::try_from_f64_lossy(x).unwrap();
        let span = if S::MANTISSA_DIGITS == 24 { 100 } else { 900 };
        for frame in [[-span; 3], [0; 3], [span; 3], [-span, 0, span]] {
            for raw in [
                [[0., 2., -1.], [3., -1., 0.5], [1., 0.25, 4.]],
                [[4., -0.5, 0.25], [0.25, 3., 1.], [-0.25, 0.5, 2.]],
                [[1., 1., 0.], [1., 1.0001, 0.5], [0., 0.5, 3.]],
            ] {
                let entries = core::array::from_fn(|i| raw[i].map(|x| scalar(x).scalbn(frame[i])));
                let factor = Factor::new(entries).unwrap();
                let matrix =
                    Matrix::from_vec(3, 3, entries.into_iter().flatten().collect()).unwrap();
                let mut workspace = LinalgWorkspace::new();
                let mut reference = LuDecomposition::empty();
                LuDecomposition::factor_into(
                    &matrix,
                    &LinalgExecutionOptions::new(),
                    &mut reference,
                    &mut workspace,
                    context,
                )
                .unwrap();
                for (i, row) in factor.entries.iter().enumerate() {
                    for (j, value) in row.iter().enumerate() {
                        assert_eq!(
                            value.into_f64().to_bits(),
                            reference
                                .packed_factors()
                                .get(i, j)
                                .unwrap()
                                .into_f64()
                                .to_bits()
                        );
                    }
                }
                for rhs in [[1., -0.5, 0.25], [0., 0., 0.], [0.1, 0.2, -0.7]] {
                    let rhs = core::array::from_fn(|i| scalar(rhs[i]).scalbn(frame[i]));
                    let actual = factor
                        .solve(
                            rhs,
                            scioxide_core::backend::KernelDispatch::detect(context.processor())
                                .backend(),
                        )
                        .unwrap();
                    let mut expected = [S::ZERO; 3];
                    let report = reference
                        .solve_into(
                            VectorView::from_slice(&rhs),
                            &DirectSolveOptions::new(RankPolicy::Exact),
                            VectorViewMut::from_slice(&mut expected),
                            &mut workspace,
                            context,
                        )
                        .unwrap();
                    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
                    assert_eq!(
                        actual.map(|x| x.into_f64().to_bits()),
                        expected.map(|x| x.into_f64().to_bits())
                    );
                }
            }
        }
    }

    #[test]
    fn factors_and_solves_match_linalg_bits() {
        compare::<f32>(ExecutionContext::sequential());
        compare::<f64>(ExecutionContext::sequential());
        #[cfg(feature = "std")]
        {
            struct Processor(KernelBackend);
            impl scioxide_core::ProcessorCapabilities for Processor {
                fn supports(&self, feature: scioxide_core::ProcessorFeature) -> bool {
                    use scioxide_core::ProcessorFeature::*;
                    matches!(
                        (self.0, feature),
                        (KernelBackend::Avx2Fma, X86Avx2 | X86Fma)
                            | (
                                KernelBackend::Avx512,
                                X86Avx512F | X86Avx512Dq | X86Avx512Vl
                            )
                            | (KernelBackend::Neon, Aarch64Neon)
                    )
                }
            }
            for backend in KernelBackend::available(&scioxide_core::StdProcessor) {
                let processor = Processor(backend);
                let context = ExecutionContext::sequential().with_processor(&processor);
                compare::<f32>(context);
                compare::<f64>(context);
            }
        }
    }

    #[test]
    fn singular_and_nonfinite_systems_fail() {
        assert!(
            matches!(Factor::new([[0_f64; 3]; 3]), Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular)
        );
        assert!(Factor::new([[f64::NAN; 3]; 3]).is_err());
        let factor = Factor::new([[1_f64, 0., 0.], [0., 1., 0.], [0., 0., 1.]]).unwrap();
        assert!(
            factor
                .solve([f64::INFINITY, 0., 0.], KernelBackend::Scalar)
                .is_err()
        );
        let tiny = Factor::new([[f64::MIN_POSITIVE, 0., 0.], [0., 1., 0.], [0., 0., 1.]]).unwrap();
        assert!(
            tiny.solve([f64::MAX, 0., 0.], KernelBackend::Scalar)
                .is_err()
        );
    }
}
