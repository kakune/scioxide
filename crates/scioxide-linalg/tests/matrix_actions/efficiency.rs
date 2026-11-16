//! Deterministic provider-work contracts; no timing measurements.
use super::*;
use scioxide_core::{LinearOperator, SciOxideError};
use scioxide_linalg::{
    LobpcgBuffer, LobpcgSolver, OperatorStructure, PartialSpectrumBuffer, PartialSpectrumSolver,
    SpectrumOperator,
};
use std::cell::Cell;

struct Diagonal {
    values: Vec<f64>,
    calls: Cell<usize>,
    constraint_calls: Cell<usize>,
}
impl Diagonal {
    fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            calls: Cell::new(0),
            constraint_calls: Cell::new(0),
        }
    }
}
impl LinearOperator<f64> for Diagonal {
    fn rows(&self) -> usize {
        self.values.len()
    }
    fn cols(&self) -> usize {
        self.values.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        if x[0] != 0. && x[1..].iter().all(|&v| v == 0.) {
            self.constraint_calls.set(self.constraint_calls.get() + 1);
        }
        for ((out, &x), &d) in y.iter_mut().zip(x).zip(&self.values) {
            *out = x * d;
        }
        Ok(())
    }
}
fn initial(n: usize) -> Vec<f64> {
    (0..3)
        .flat_map(|j| {
            (0..n).map(move |i| match j {
                0 => 1.,
                1 => ((i * 7) % 11) as f64 - 4.,
                _ => ((i * 11) % 17) as f64 - 8.,
            })
        })
        .collect()
}
#[test]
fn lobpcg_applies_only_new_directions_between_verifications() {
    let n = 60;
    let k = 3;
    for generalized in [false, true] {
        let a = Diagonal::new((1..=n).map(|v| v as f64).collect());
        let b = Diagonal::new(vec![2.; n]);
        let op = if generalized {
            scioxide_linalg::LobpcgOperator::new(&a).with_mass(&b)
        } else {
            scioxide_linalg::LobpcgOperator::new(&a)
        };
        let constraints = if generalized {
            let mut c = vec![0.; n];
            c[0] = 1.;
            c
        } else {
            vec![]
        };
        let plan = LobpcgSolver::new(k);
        let mut work = plan.workspace(op, usize::from(generalized)).unwrap();
        let mut out = LobpcgBuffer::new(n, k).unwrap();
        let initial = initial(n);
        let out_report = plan
            .solve_into(
                op,
                BlockVectorView::new(&initial, n, k).unwrap(),
                BlockVectorView::new(&constraints, n, usize::from(generalized)).unwrap(),
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            out_report.termination(),
            SpectralTermination::Converged,
            "{out:?}"
        );
        // Each iteration applies A to the k new residual directions and the
        // k previous directions; the Ritz block is applied only when verified.
        assert!(
            a.calls.get()
                <= 2 * k * out_report.iterations()
                    + k * (out_report.iterations() / 50 + 2)
                    + usize::from(generalized),
            "{} applications in {} iterations",
            a.calls.get(),
            out_report.iterations()
        );
        assert_eq!(out_report.applications(), a.calls.get() + b.calls.get());
        if generalized {
            assert_eq!(b.constraint_calls.get(), 1);
            assert_eq!(b.calls.get(), a.calls.get());
        }
        for j in 0..k {
            let expected =
                (j + 1 + usize::from(generalized)) as f64 / if generalized { 2. } else { 1. };
            assert!((out.eigenvalues()[j] - expected).abs() < 1e-7);
            let residual = (0..n)
                .map(|i| {
                    let x = out.eigenvectors().get(i, j).unwrap();
                    ((a.values[i] - out.eigenvalues()[j] * if generalized { 2. } else { 1. }) * x)
                        .powi(2)
                })
                .sum::<f64>()
                .sqrt();
            assert!((residual - out.residuals()[j]).abs() < 1e-12);
        }
    }
}
#[test]
fn general_symmetric_spectra_stay_within_ten_percent_of_hermitian_work() {
    let n = 60;
    let mut counts = Vec::new();
    for structure in [OperatorStructure::Hermitian, OperatorStructure::General] {
        let a = Diagonal::new((1..=n).map(|v| v as f64).collect());
        let plan = PartialSpectrumSolver::new(3)
            .with_basis(Budget::try_from(20).unwrap())
            .with_structure(structure);
        let mut work = plan.workspace(&a).unwrap();
        let mut out = PartialSpectrumBuffer::new(n, 3).unwrap();
        let initial = vec![1.; n];
        let out_report = plan
            .solve_into(
                &a,
                initial.as_slice(),
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            out_report.termination(),
            SpectralTermination::Converged,
            "{out:?}"
        );
        assert_eq!(a.calls.get(), out_report.applications());
        for (j, value) in out.eigenvalues().iter().enumerate() {
            assert!((value.re - (n - j) as f64).abs() < 1e-7);
            assert_eq!(value.im, 0.);
        }
        counts.push(out_report.applications());
    }
    assert!(
        10 * counts[1] <= 11 * counts[0],
        "Hermitian/general calls: {counts:?}"
    );
}
#[test]
fn real_shift_invert_ritz_vectors_use_one_original_probe() {
    for structure in [OperatorStructure::Hermitian, OperatorStructure::General] {
        let a = Diagonal::new(vec![1., 2., 3.]);
        let inverse = Diagonal::new(vec![1., 0.5, 1. / 3.]);
        let operator = SpectrumOperator::ShiftInvert {
            a: &a,
            inverse: &inverse,
            mass: None,
            shift: 0.,
        };
        let plan = PartialSpectrumSolver::new(1).with_structure(structure);
        let mut work = plan.workspace(operator).unwrap();
        let mut out = PartialSpectrumBuffer::new(3, 1).unwrap();
        let out_report = plan
            .solve_into(
                operator,
                [1.; 3].as_slice(),
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(out_report.termination(), SpectralTermination::Converged);
        assert_eq!(a.calls.get(), 1);
        assert_eq!(out_report.applications(), inverse.calls.get() + 1);
    }
}

#[test]
fn lobpcg_budget_stops_keep_the_last_complete_explicit_verification() {
    let n = 60;
    let a = Diagonal::new((1..=n).map(|v| v as f64).collect());
    let initial = initial(n);
    let mut reference: Option<LobpcgBuffer<f64>> = None;
    // Six calls complete the initial block; subsequent calls update the
    // recurrence but cannot publish it until the next complete verification.
    for applications in [6, 7, 8, 9, 10, 30] {
        a.calls.set(0);
        let plan = LobpcgSolver::new(3).with_applications(Budget::try_from(applications).unwrap());
        let mut work = plan.workspace(&a, 0).unwrap();
        let mut out = LobpcgBuffer::new(n, 3).unwrap();
        let out_report = plan
            .solve_into(
                &a,
                BlockVectorView::new(&initial, n, 3).unwrap(),
                BlockVectorView::new(&[], n, 0).unwrap(),
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            out_report.termination(),
            SpectralTermination::EvaluationLimit
        );
        assert_eq!(out_report.iterations(), 1);
        assert_eq!(out_report.applications(), applications);
        if let Some(reference) = &reference {
            assert_eq!(out.eigenvalues(), reference.eigenvalues());
            assert_eq!(out.residuals(), reference.residuals());
            assert_eq!(
                out.eigenvectors().as_contiguous_slice().unwrap(),
                reference.eigenvectors().as_contiguous_slice().unwrap()
            );
        } else {
            reference = Some(out);
        }
    }
}

#[test]
fn nonzero_imaginary_ritz_components_still_receive_both_real_probes() {
    struct Counted<'a> {
        op: &'a dyn LinearOperator<f64>,
        calls: Cell<usize>,
    }
    impl LinearOperator<f64> for Counted<'_> {
        fn rows(&self) -> usize {
            self.op.rows()
        }
        fn cols(&self) -> usize {
            self.op.cols()
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            self.op.apply_workspace_len()
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64], w: &mut [f64]) -> Result<(), SciOxideError> {
            self.calls.set(self.calls.get() + 1);
            self.op.apply_into(x, y, w)
        }
    }
    let matrix = Matrix::from_vec(3, 3, vec![0., -2., 0., 2., 0., 0., 0., 0., 1.]).unwrap();
    let view = matrix.view().try_finite().unwrap();
    let a = Counted {
        op: &view,
        calls: Cell::new(0),
    };
    let plan = PartialSpectrumSolver::new(2);
    let mut out = PartialSpectrumBuffer::new(3, 2).unwrap();
    let mut work = plan.workspace(&a).unwrap();
    let out_report = plan
        .solve_into(
            &a,
            [1., 0.5, 0.2].as_slice(),
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out_report.termination(), SpectralTermination::Converged);
    assert_eq!(a.calls.get(), 7); // Three Arnoldi images and two probes per Ritz vector.
    assert_eq!(out_report.applications(), a.calls.get());
    for value in out.eigenvalues() {
        assert!((value.im.abs() - 2.).abs() < 1e-12);
    }
}
