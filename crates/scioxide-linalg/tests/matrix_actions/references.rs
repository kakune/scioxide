//! Independent non-normal action references and power-estimator coverage.
#![allow(clippy::unwrap_used)]
use crate::support::function_reference as reference;
use core::cell::Cell;
use scioxide_core::{
    AdjointOperator, BlockVectorView, BlockVectorViewMut, Complex, ExecutionContext,
    LinearOperator, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_linalg::{ExponentialAction, Matrix, SpectralTermination};

struct Counted<'a, S: NumericScalar> {
    matrix: scioxide_linalg::FiniteMatrixView<'a, S>,
    adjoints: Cell<usize>,
}
impl<S: NumericScalar> LinearOperator<S> for Counted<'_, S> {
    fn rows(&self) -> usize {
        self.matrix.rows()
    }
    fn cols(&self) -> usize {
        self.matrix.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_workspace_len()
    }
    fn apply_into(&self, x: &[S], y: &mut [S], work: &mut [S]) -> Result<(), SciOxideError> {
        self.matrix.apply_into(x, y, work)
    }
}
impl<S: NumericScalar> AdjointOperator<S> for Counted<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_adjoint_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        x: &[S],
        y: &mut [S],
        work: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.adjoints.set(self.adjoints.get() + 1);
        self.matrix.apply_adjoint_into(x, y, work)
    }
}

fn check<S: NumericScalar>() {
    let mut count = 0;
    for fields in reference::records::<S>(include_str!("../data/matrix_actions.tsv")) {
        assert_eq!(fields.len(), 11);
        let source: usize = fields[4].parse().unwrap();
        // An exact zero coordinate leaves the reference unchanged and takes
        // exp's norm estimator past its exhaustive order-four shortcut.
        let n = source + 1;
        let source_a = reference::values::<S>(fields[6]);
        let a = Matrix::<S>::from_vec(
            n,
            n,
            (0..n * n)
                .map(|i| {
                    if i / n < source && i % n < source {
                        source_a[i / n * source + i % n]
                    } else {
                        S::ZERO
                    }
                })
                .collect(),
        )
        .unwrap();
        let op = Counted {
            matrix: a.view().try_finite().unwrap(),
            adjoints: Cell::new(0),
        };
        let bound = (0..n)
            .map(|j| {
                (0..n)
                    .map(|i| a.as_slice()[i * n + j].abs().into_f64())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        // A conservative caller bound forces estimation for every scalar type
        // and phi index; the operator itself remains the independent fixture.
        let bound = if fields[2] == "overscaling" {
            4.0 * bound
        } else {
            bound
        };
        let bound = S::Real::try_from_f64_lossy(bound)
            .unwrap()
            .next_after(S::Real::INFINITY);
        let plan = ExponentialAction::new(bound).with_max_phi(3);
        let mut workspace = plan.workspace(&op, 2).unwrap();
        let rhs = Matrix::<S>::from_vec(source, 2, reference::values(fields[7])).unwrap();
        // BlockVectorView stores RHS columns consecutively.
        let input: Vec<_> = (0..2)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .map(|(i, j)| {
                if i < source {
                    rhs.as_slice()[i * 2 + j]
                } else {
                    S::ZERO
                }
            })
            .collect();
        let expected = reference::expected::<S>(fields[8], fields[9]);
        let expected: Vec<_> = (0..2)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .map(|(i, j)| {
                if i < source {
                    expected[i * 2 + j]
                } else {
                    S::ZERO
                }
            })
            .collect();
        let mut output = vec![S::ZERO; 2 * n];
        let report = plan
            .apply_into(
                &op,
                scioxide_linalg::PhiFunction::phi(
                    fields[3].parse().unwrap(),
                    reference::number(fields[5]),
                ),
                BlockVectorView::new(&input, n, 2).unwrap(),
                BlockVectorViewMut::new(&mut output, n, 2).unwrap(),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::Converged);
        reference::assert_close(&output, &expected, n, fields[10], fields[2]);
        if fields[2] == "overscaling" {
            assert!(op.adjoints.get() > 0, "power estimates were not exercised");
        }
        count += 1;
    }
    assert_eq!(count, 16);
}

#[test]
fn independent_non_normal_exponential_and_phi_actions() {
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}
