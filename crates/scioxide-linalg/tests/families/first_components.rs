//! First-component spectral measures, storage and transactional failures.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Budget, ExecutionContext, NumericScalar, RealScalar, SciOxideErrorKind};
use scioxide_linalg::{
    EigenIndexRange, EigenvalueAccuracy, EigenvalueOrder, HermitianEigenDecomposition,
    HermitianEigenFirstComponents, HermitianEigenOptions, HermitianEigenWorkspaceRequest,
    LinalgExecutionOptions, LinalgWorkspace,
};

fn spectral_measure<R: RealScalar + NumericScalar<Real = R>>() {
    let real = |x| R::try_from_f64_lossy(x).unwrap();
    let tolerance = real(2048.0) * R::EPSILON;
    for (d, e) in [
        (vec![], vec![]),
        (vec![3.0], vec![]),
        (vec![0.0, 0.0], vec![1.0]),
        (vec![0.0; 6], vec![1.0; 5]),
        (vec![2.0; 6], vec![1.0, 0.0, 1.0, 0.0, 1.0]),
        (vec![1.0, 2.0, -1.0, 0.0], vec![-0.5, 0.25, 0.75]),
    ] {
        let d: Vec<_> = d.into_iter().map(real).collect();
        let e: Vec<_> = e.into_iter().map(real).collect();
        for ordering in [EigenvalueOrder::Ascending, EigenvalueOrder::Descending] {
            let options = HermitianEigenOptions::default().with_ordering(ordering);
            let result =
                HermitianEigenFirstComponents::tridiagonal_with_options(&d, &e, &options).unwrap();
            let full =
                HermitianEigenDecomposition::tridiagonal_with_options(&d, &e, &options).unwrap();
            assert_eq!(result.eigenvalues().len(), d.len());
            for (&actual, &expected) in result.eigenvalues().iter().zip(full.eigenvalues()) {
                assert!((actual - expected).abs() <= tolerance * expected.abs().max(R::ONE));
            }
            // Degenerate eigenspaces need not choose the same basis: compare
            // their invariant spectral moments, not individual vector phases.
            let mut power = vec![R::ZERO; d.len()];
            if let Some(first) = power.first_mut() {
                *first = R::ONE;
            }
            let mut next = power.clone();
            for degree in 0..=6 {
                let actual = result
                    .eigenvalues()
                    .iter()
                    .zip(result.first_components())
                    .fold(R::ZERO, |sum, (&x, &z)| sum + x.powi(degree) * z * z);
                let expected = power.first().copied().unwrap_or(R::ZERO);
                assert!((actual - expected).abs() <= tolerance * expected.abs().max(R::ONE));
                for (i, value) in next.iter_mut().enumerate() {
                    *value = d[i] * power[i]
                        + if i > 0 {
                            e[i - 1] * power[i - 1]
                        } else {
                            R::ZERO
                        }
                        + if i + 1 < d.len() {
                            e[i] * power[i + 1]
                        } else {
                            R::ZERO
                        };
                }
                core::mem::swap(&mut power, &mut next);
            }
            assert!(result.report().is_some());
        }
    }
}

#[test]
fn scalar_native_spectral_measures() {
    spectral_measure::<f32>();
    spectral_measure::<f64>();
}

#[test]
fn scaling_and_first_component_accuracy() {
    for scale in [1e-300_f64, 1.0, 1e300] {
        let d = [2.0 * scale, 2.0 * scale];
        let result = HermitianEigenFirstComponents::tridiagonal(&d, &[scale]).unwrap();
        for ((&value, &component), expected) in result
            .eigenvalues()
            .iter()
            .zip(result.first_components())
            .zip([1.0, 3.0])
        {
            assert!((value / scale - expected).abs() < 2e-14);
            assert!((component * component - 0.5).abs() < 2e-14);
        }
    }
    // Tiny couplings must survive unsquared QL rotation generation.
    let tiny = f64::MIN_POSITIVE;
    let result = HermitianEigenFirstComponents::tridiagonal(&[0.0, 0.0], &[tiny]).unwrap();
    assert!((result.eigenvalues()[0] / tiny + 1.0).abs() < 1e-14);
    assert!((result.first_components()[0].powi(2) - 0.5).abs() < 1e-14);
}

#[test]
fn invalid_inputs_and_unsupported_jobs() {
    let options = HermitianEigenOptions::<f64>::default();
    for (d, e, kind) in [
        (
            &[1.0, 2.0][..],
            &[][..],
            SciOxideErrorKind::DimensionMismatch,
        ),
        (&[f64::NAN][..], &[][..], SciOxideErrorKind::NonFinite),
        (
            &[1.0, 2.0][..],
            &[f64::INFINITY][..],
            SciOxideErrorKind::NonFinite,
        ),
    ] {
        assert_eq!(
            HermitianEigenFirstComponents::tridiagonal(d, e)
                .unwrap_err()
                .kind(),
            kind
        );
    }
    for unsupported in [
        options.with_index_range(EigenIndexRange::new(0, 1).unwrap()),
        options.with_eigenvalue_accuracy(EigenvalueAccuracy::Relative),
    ] {
        assert_eq!(
            HermitianEigenFirstComponents::tridiagonal_with_options(
                &[1.0, 2.0],
                &[0.5],
                &unsupported
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    assert!(
        HermitianEigenFirstComponents::<f64>::workspace_requirements(
            &HermitianEigenWorkspaceRequest::new(2, &options)
        )
        .is_err()
    );
    assert!(
        HermitianEigenFirstComponents::<f64>::workspace_requirements(
            &HermitianEigenWorkspaceRequest::tridiagonal(usize::MAX, &options)
        )
        .is_err()
    );
    let limited = options.with_execution_options(LinalgExecutionOptions::new().with_byte_limit(1));
    assert_eq!(
        HermitianEigenFirstComponents::tridiagonal_with_options(&[1.0, 2.0], &[0.5], &limited)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let timed = options.with_execution_options(
        LinalgExecutionOptions::new().with_time_limit(core::time::Duration::from_secs(1)),
    );
    assert_eq!(
        HermitianEigenFirstComponents::tridiagonal_with_options(&[1.0, 2.0], &[0.5], &timed)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    let mut result = HermitianEigenFirstComponents::empty();
    let mut empty = [];
    let mut workspace = LinalgWorkspace::from_scalar_slice(&mut empty);
    assert_eq!(
        HermitianEigenFirstComponents::tridiagonal_into(
            &[1.0, 2.0],
            &[0.5],
            &options,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(result.eigenvalues().is_empty());
    assert!(result.report().is_none());
}

#[test]
fn first_component_sweeps_are_nested_in_the_callers_solve() {
    use scioxide_core::{
        AttemptOutcome, BudgetKind, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(2048),
        scioxide_core::algorithm_id!("test", "first_components", 1),
        0,
        0,
    );
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut state = SolveState::new();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut result = HermitianEigenFirstComponents::empty();
    let mut workspace = LinalgWorkspace::new();
    let report = HermitianEigenFirstComponents::tridiagonal_into(
        &[0.0; 8],
        &[0.5; 7],
        &HermitianEigenOptions::default(),
        &mut result,
        &mut workspace,
        &mut context,
    )
    .unwrap();
    assert!(report.iterations() > 1);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), report.iterations());
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
}
