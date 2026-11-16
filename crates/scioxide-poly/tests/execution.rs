//! Cancellation, transaction and retained execution configuration contracts.
#![allow(clippy::unwrap_used)]
use core::fmt::Debug;
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{
    Cancellation, CancellationToken, Complex, ExecutionContext, ExecutionPlan, ExecutionPolicy,
    Executor, Interval, NdView, PartitionFn, SciOxideError, SciOxideErrorView, SequentialExecutor,
    StridedLayout,
};
use scioxide_poly::PolynomialOperationRequest;
use scioxide_poly::*;

#[path = "support/exact_field.rs"]
mod exact_field;
use exact_field::Q;

struct CancelAt {
    calls: AtomicUsize,
    limit: usize,
}
impl CancelAt {
    fn new(limit: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            limit,
        }
    }
}
impl Cancellation for CancelAt {
    fn is_cancelled(&self) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed) + 1 >= self.limit
    }
}

// Exercise entry, an internal work boundary and the last publication boundary.
// The final case detects a missing check after staging has completed.
fn cancellation_boundaries<O: Clone + PartialEq + Debug, T>(
    mut seed: O,
    mut run: impl FnMut(ExecutionContext<'_>, &mut O) -> Result<T, SciOxideError>,
) {
    let counter = CancelAt::new(usize::MAX);
    run(
        ExecutionContext::sequential().with_cancellation(&counter),
        &mut seed.clone(),
    )
    .unwrap();
    let polls = counter.calls.load(Ordering::Relaxed);
    assert!(polls >= 2);
    for limit in [1, polls.div_ceil(2), polls] {
        let cancellation = CancelAt::new(limit);
        let mut output = seed.clone();
        let result = run(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            &mut output,
        );
        assert!(matches!(
            result.err().unwrap().view(),
            SciOxideErrorView::ExecutionStopped { .. }
        ));
        assert_eq!(
            output, seed,
            "publication after cancellation at poll {limit}/{polls}"
        );
    }
    // Failed workspaces remain reusable with the same caller capabilities.
    run(ExecutionContext::sequential(), &mut seed).unwrap();
}

#[test]
fn approximate_and_exact_coefficient_operations_publish_atomically() {
    let plan = PolynomialOperationPlan::<f64>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 2,
            right_len: 2,
        },
        PolynomialOperationRequest::Compose {
            outer_len: 3,
            inner_len: 2,
        },
        PolynomialOperationRequest::Compose {
            outer_len: 3,
            inner_len: 1,
        },
        PolynomialOperationRequest::TaylorShift { len: 3 },
        PolynomialOperationRequest::Multiply {
            left_len: 129,
            right_len: 129,
        },
    ])
    .unwrap();
    let mut ws = plan.workspace(plan.requests()).unwrap();
    cancellation_boundaries([99.; 3], |cx, out| {
        plan.multiply_into(&[1., 2.], &[3., 4.], out, &mut ws, cx)
    });
    cancellation_boundaries([99.; 3], |cx, out| {
        plan.compose_into(&[1., 2., 3.], &[1., 2.], out, &mut ws, cx)
    });
    cancellation_boundaries([99.; 1], |cx, out| {
        plan.compose_into(&[1., 2., 3.], &[0.5], out, &mut ws, cx)
    });
    cancellation_boundaries([99.; 1], |cx, out| {
        plan.compose_into(&[0., 0., f64::MAX], &[0.], out, &mut ws, cx)
    });
    cancellation_boundaries([99.; 3], |cx, out| {
        plan.taylor_shift_into(&[1., 2., 3.], 2., out, &mut ws, cx)
    });
    cancellation_boundaries([99.; 257], |cx, out| {
        plan.multiply_into(&[1.; 129], &[2.; 129], out, &mut ws, cx)
    });

    let mut ws = ExactPolynomialWorkspace::workspace(129).unwrap();
    cancellation_boundaries([Q(99, 1); 129], |cx, out| {
        exact_multiply_into(&[Q(1, 1); 65], &[Q(2, 1); 65], out, &mut ws, cx)
    });
    cancellation_boundaries([Q(99, 1); 95], |cx, out| {
        exact_multiply_into(&[Q(1, 1); 64], &[Q(2, 1); 32], out, &mut ws, cx)
    });
    cancellation_boundaries([Q(99, 1); 96], |cx, out| {
        exact_multiply_into(&[Q(2, 1); 32], &[Q(1, 1); 65], out, &mut ws, cx)
    });
    cancellation_boundaries([Q(99, 1); 3], |cx, out| {
        exact_compose_into(
            &[Q(1, 1), Q(2, 1), Q(3, 1)],
            &[Q(1, 1), Q(2, 1)],
            out,
            &mut ws,
            cx,
        )
    });
    cancellation_boundaries([Q(99, 1); 3], |cx, out| {
        exact_taylor_shift_into(&[Q(1, 1), Q(2, 1), Q(3, 1)], Q(2, 1), out, &mut ws, cx)
    });
}

#[test]
fn shared_composition_powers_and_range_retries_publish_atomically() {
    for n in [513, 1024] {
        let request = PolynomialOperationRequest::Compose {
            outer_len: n,
            inner_len: 3,
        };
        let plan = PolynomialOperationPlan::<f64>::new(&[request]).unwrap();
        let mut workspace = plan.workspace(&[request]).unwrap();
        cancellation_boundaries(vec![99.0; 2 * n - 1], |cx, out| {
            plan.compose_into(&vec![0.5; n], &[0.25, 0.5, 0.25], out, &mut workspace, cx)
        });
        let mut output = vec![99.0; 2 * n - 1];
        for q in [[f64::NAN, 0.5, 0.25], [f64::MAX, 1.0, 1.0]] {
            assert!(
                plan.compose_into(
                    &vec![1.0; n],
                    &q,
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .is_err()
            );
            assert!(output.iter().all(|&coefficient| coefficient == 99.0));
        }
    }
}

#[test]
fn division_basis_matrices_and_fits_publish_at_cancellation_boundaries() {
    let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Divide {
        dividend_len: 3,
        divisor_len: 2,
    }])
    .unwrap();
    let mut ws = plan.workspace(plan.requests()).unwrap();
    cancellation_boundaries(([99.; 2], [99.]), |cx, out| {
        plan.div_rem_into(
            &[-1., 0., 1.],
            &[-1., 1.],
            &mut out.0,
            &mut out.1,
            &mut ws,
            cx,
        )
    });
    let matrix = Vandermonde::<f64>::monomial(2).unwrap();
    let mut scratch = [0.; 6];
    cancellation_boundaries([99.; 6], |cx, out| {
        matrix.matrix_into(
            &[-1., 1.],
            scioxide_linalg::MatrixViewMut::from_slice(2, 3, out)?,
            &mut scratch,
            cx,
        )
    });
    let fit =
        PolynomialFitPlan::monomial(&[-1.0_f64, 0., 1.], 1, None, Default::default()).unwrap();
    let mut ws = fit.workspace(()).unwrap();
    cancellation_boundaries([99.; 2], |cx, out| {
        fit.fit_into(&[-1., 1., 3.], out, &mut ws, cx)
    });
    // Empty matrices still poll before publication.
    cancellation_boundaries([99.; 0], |cx, out| {
        matrix.matrix_into(
            &[],
            scioxide_linalg::MatrixViewMut::from_slice(0, 3, out)?,
            &mut [],
            cx,
        )
    });
}

#[test]
fn borrowed_output_forms_poll_cancellation_and_publish_atomically() {
    // Root linearization in both bases.
    let seed = scioxide_linalg::Matrix::from_vec(2, 2, vec![99.0_f64; 4]).unwrap();
    cancellation_boundaries(seed.clone(), |cx, out| {
        polynomial_root_matrix_into(&[2.0, -3.0, 1.0], PolynomialRootBasisRef::Monomial, out, cx)
    });
    let chebyshev = OrthogonalBasisSpec::new(
        OrthogonalFamily::ChebyshevT,
        BasisNormalization::Classical,
        0.0_f64,
        1.0,
    )
    .unwrap()
    .prepare(2, Default::default())
    .unwrap();
    cancellation_boundaries(seed, |cx, out| {
        polynomial_root_matrix_into(
            &[2.0, -3.0, 1.0],
            PolynomialRootBasisRef::Orthogonal(&chebyshev),
            out,
            cx,
        )
    });
    // Anchored orthogonal antiderivative.
    let primitive = chebyshev.antiderivative_plan().unwrap();
    let mut scratch = vec![0.0; primitive.workspace_requirements(()).unwrap().scratch_len()];
    cancellation_boundaries([99.0; 4], |cx, out| {
        primitive.apply_into(&[1.0, 2.0, 3.0], (0.0, 1.0), out, &mut scratch, cx)
    });
    // Borrowed tensor Bernstein evaluation of two components.
    let domains = [Interval::new(0.0, 1.0).unwrap()];
    let view = TensorBernsteinView::new(
        &[2],
        &domains,
        2,
        &[1.0, 2.0, 3.0, 5.0],
        PolynomialExecutionOptions::new(),
    )
    .unwrap();
    let mut scratch = vec![0.0; view.workspace_requirements(1).unwrap().scratch_len()];
    cancellation_boundaries([99.0; 2], |cx, out| {
        view.evaluate_into(&[0.5], out, &mut scratch, cx)
    });
    // Dense tensor export of a sparse polynomial.
    let sparse = ApproxSparsePolynomial::from_terms(
        2,
        MonomialOrder::Lexicographic,
        vec![
            (Monomial::new(&[1, 0]).unwrap(), 3.0),
            (Monomial::new(&[0, 1]).unwrap(), 5.0),
        ],
    )
    .unwrap();
    cancellation_boundaries([99.0; 4], |cx, out| {
        let layout = StridedLayout::new(&[2, 2], &[2, 1])?;
        sparse.to_tensor_into(scioxide_core::NdViewMut::new(out, layout)?, &[0, 1], cx)
    });
}

#[test]
fn elimination_partial_fractions_and_diagnostics_publish_atomically() {
    let policy = PolynomialApproximationPolicy::default();
    let plan = SylvesterElimination::new(2, 2, 0, policy).unwrap();
    let seed = approximate_resultant(&[1.], &[1.]).unwrap();
    let mut ws = plan.workspace().unwrap();
    cancellation_boundaries(seed, |cx, out| {
        plan.coefficient_scoped(&[1., 0., 1.], &[-1., 0., 1.], 0, &mut ws, cx)
            .map(|result| *out = result)
    });
    let d = SylvesterDiscriminant::new(2, policy).unwrap();
    let mut ws = d.workspace().unwrap();
    cancellation_boundaries(seed, |cx, out| {
        d.compute_scoped(&[1., 0., 1.], &mut ws, cx)
            .map(|result| *out = result)
    });
    // Constant identities also observe cancellation before returning.
    let constant = SylvesterElimination::new(0, 0, 0, policy).unwrap();
    let mut ws = constant.workspace().unwrap();
    cancellation_boundaries(seed, |cx, out| {
        constant
            .coefficient_scoped(&[1.], &[2.], 0, &mut ws, cx)
            .map(|result| *out = result)
    });

    let factor = ApproxPolynomial::monomial_with_options(vec![-1., 1.], policy).unwrap();
    let fractions = ApproxPartialFractionPlan::new(&factor, &[(&factor, 1)]).unwrap();
    let seed = fractions.decompose(&[1.]).unwrap();
    let mut ws = fractions.workspace(1).unwrap();
    cancellation_boundaries(seed, |cx, out| {
        fractions.decompose_into(&[3.], out, &mut ws, cx)
    });

    let coefficients = [2., -3., 1.];
    let roots = [Complex::new(1., 0.), Complex::new(2., 0.)];
    let seed = polynomial_root_diagnostics(
        &[6., -5., 1.],
        &[Complex::new(2., 0.), Complex::new(3., 0.)],
    )
    .unwrap();
    let mut ws = RootDiagnosticsWorkspace::workspace(2).unwrap();
    cancellation_boundaries(seed, |cx, out| {
        polynomial_root_diagnostics_into(&coefficients, &roots, out, &mut ws, cx)
    });
}

#[test]
fn all_batch_families_observe_cancellation_before_publication() {
    let options = PolynomialExecutionOptions::new();
    let terms_exponents = [1];
    let terms_coefficients = [2.];
    let sparse = SparseEvaluationPlan::new(
        1,
        MonomialOrder::Lexicographic,
        &terms_exponents,
        &terms_coefficients,
        options,
    )
    .unwrap();
    let points = [0., 0.25, 1.];
    let layout = StridedLayout::new(&[3, 1], &[1, 1]).unwrap();
    let mut scratch = vec![0.; sparse.workspace_requirements(3).unwrap().scratch_len()];
    cancellation_boundaries([99.; 3], |cx, out| {
        sparse.evaluate_batch_into(NdView::new(&points, layout).unwrap(), out, &mut scratch, cx)
    });
    let domains = [Interval::new(0., 1.).unwrap()];
    let tensor = TensorBernsteinView::new(&[2], &domains, 1, &[0., 2.], options).unwrap();
    let mut scratch = [0.; 5];
    cancellation_boundaries([99.; 3], |cx, out| {
        tensor.evaluate_batch_into(NdView::new(&points, layout).unwrap(), out, &mut scratch, cx)
    });
    let owned =
        TensorBernstein::with_options(&[1], domains.to_vec(), 1, vec![0., 2.], options).unwrap();
    cancellation_boundaries([99.; 3], |cx, out| {
        owned.as_view().evaluate_batch_into(
            NdView::new(&points, layout).unwrap(),
            out,
            &mut scratch,
            cx,
        )
    });
    let orthogonal = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Classical,
        0.,
        1.,
    )
    .unwrap()
    .prepare(1, options)
    .unwrap();
    cancellation_boundaries([99.; 3], |cx, out| {
        orthogonal.evaluate_batch_into(&[1., 2.], &points, out, &mut scratch[..3], cx)
    });
    let terms_exponents = [1];
    let terms_coefficients = [Q(2, 1)];
    let exact = ExactSparseView::new(
        1,
        MonomialOrder::Lexicographic,
        &terms_exponents,
        &terms_coefficients,
    )
    .unwrap();
    let points = [Q(0, 1), Q(1, 4), Q(1, 1)];
    let mut scratch = [Q(0, 1); 3];
    cancellation_boundaries([Q(99, 1); 3], |cx, out| {
        exact.evaluate_batch_into(NdView::new(&points, layout).unwrap(), out, &mut scratch, cx)
    });
}

#[test]
fn bernstein_batch_numerical_and_input_errors_preserve_output() {
    let domains = [Interval::new(0., 1.).unwrap()];
    let controls = [1.0_f64, f64::MAX];
    let tensor =
        TensorBernsteinView::new(&[2], &domains, 1, &controls, Default::default()).unwrap();
    let layout = StridedLayout::new(&[2, 1], &[1, 1]).unwrap();
    for (points, kind) in [
        ([0., f64::NAN], scioxide_core::SciOxideErrorKind::NonFinite),
        (
            [0., 2.],
            scioxide_core::SciOxideErrorKind::NumericalInstability,
        ),
    ] {
        let mut output = [99.; 2];
        let error = tensor
            .evaluate_batch_into(
                NdView::new(&points, layout).unwrap(),
                &mut output,
                &mut [0.; 4],
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(output, [99.; 2]);
    }
}

#[test]
fn substitution_has_real_term_limits_and_atomic_output() {
    let p = ApproxSparsePolynomial::from_terms(
        1,
        MonomialOrder::Lexicographic,
        vec![(Monomial::new(&[2]).unwrap(), 1.)],
    )
    .unwrap();
    let replacement = ApproxSparsePolynomial::from_terms(
        1,
        p.order(),
        vec![
            (Monomial::new(&[0]).unwrap(), 1.),
            (Monomial::new(&[1]).unwrap(), 1.),
        ],
    )
    .unwrap();
    let replacements = [replacement];
    cancellation_boundaries(p.clone(), |cx, out| {
        p.substitute_into(&replacements, 1, SparseExpansionLimits::new(3), out, cx)
    });
    let mut output = p.clone();
    let error = p
        .substitute_into(
            &replacements,
            1,
            SparseExpansionLimits::new(2),
            &mut output,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ResourceLimit { .. }
    ));
    assert_eq!(output, p);
    assert_eq!(SparseExpansionLimits::new(0).max_terms(), 0);
    let zero = ApproxSparsePolynomial::new(1, p.order(), vec![], vec![]).unwrap();
    assert_eq!(
        zero.substitute_with_options(&replacements, 1, SparseExpansionLimits::new(0))
            .unwrap(),
        zero
    );
}

struct FailAfterWork<'a> {
    token: &'a CancellationToken,
    fail: bool,
}
impl Executor for FailAfterWork<'_> {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::MIN.saturating_add(1)
    }
    fn execute(
        &self,
        cx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(cx, plan, operation)?;
        if self.fail {
            Err(SciOxideError::dimension_mismatch(1, 0))
        } else {
            self.token.cancel();
            Ok(())
        }
    }
}

#[test]
fn executor_failure_or_cancellation_after_work_preserves_batches() {
    let options = PolynomialExecutionOptions::new().with_execution_policy(
        ExecutionPolicy::Parallel(NonZeroUsize::MIN.saturating_add(1)),
    );
    let terms_exponents = [1];
    let terms_coefficients = [2.];
    let sparse = SparseEvaluationPlan::new(
        1,
        MonomialOrder::Lexicographic,
        &terms_exponents,
        &terms_coefficients,
        options,
    )
    .unwrap();
    let domains = [Interval::new(0., 1.).unwrap()];
    let tensor = TensorBernsteinView::new(&[2], &domains, 1, &[0., 2.], options).unwrap();
    let orthogonal = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Classical,
        0.,
        1.,
    )
    .unwrap()
    .prepare(1, options)
    .unwrap();
    let points = [0., 0.25, 0.5, 1.];
    let layout = StridedLayout::new(&[4, 1], &[1, 1]).unwrap();
    for fail in [true, false] {
        let token = CancellationToken::new();
        let executor = FailAfterWork {
            token: &token,
            fail,
        };
        let cx = ExecutionContext::new(&executor).with_cancellation(&token);
        let mut output = [99.; 4];
        let mut scratch = [0.; 8];
        assert_eq!(
            sparse
                .evaluate_batch_into(
                    NdView::new(&points, layout).unwrap(),
                    &mut output,
                    &mut scratch,
                    cx
                )
                .unwrap_err()
                .kind(),
            if fail {
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            } else {
                scioxide_core::SciOxideErrorKind::ExecutionStopped
            }
        );
        assert_eq!(output, [99.; 4]);
        token.reset();
        assert_eq!(
            tensor
                .evaluate_batch_into(
                    NdView::new(&points, layout).unwrap(),
                    &mut output,
                    &mut scratch,
                    cx
                )
                .unwrap_err()
                .kind(),
            if fail {
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            } else {
                scioxide_core::SciOxideErrorKind::ExecutionStopped
            }
        );
        assert_eq!(output, [99.; 4]);
        token.reset();
        assert_eq!(
            orthogonal
                .evaluate_batch_into(&[1., 2.], &points, &mut output, &mut scratch[..4], cx)
                .unwrap_err()
                .kind(),
            if fail {
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            } else {
                scioxide_core::SciOxideErrorKind::ExecutionStopped
            }
        );
        assert_eq!(output, [99.; 4]);
    }
}

#[test]
fn construction_validates_options_and_observes_cancellation_without_fft() {
    let options =
        PolynomialExecutionOptions::new().with_worker_bound(NonZeroUsize::MIN.saturating_add(64));
    assert!(
        PolynomialOperationPlan::<f64>::build(
            &[PolynomialOperationRequest::Multiply {
                left_len: 1,
                right_len: 1
            }],
            options,
            ExecutionContext::sequential()
        )
        .is_err_and(|error| error.kind() == scioxide_core::SciOxideErrorKind::InvalidInput)
    );
    assert!(
        SparseEvaluationPlan::<f64>::new(0, MonomialOrder::Lexicographic, &[], &[], options)
            .is_err_and(|error| error.kind() == scioxide_core::SciOxideErrorKind::InvalidInput)
    );
    assert!(
        TensorBernsteinView::<f64>::new(&[], &[], 1, &[1.], options)
            .is_err_and(|error| error.kind() == scioxide_core::SciOxideErrorKind::InvalidInput)
    );
    assert!(
        OrthogonalBasisSpec::new(
            OrthogonalFamily::Legendre,
            BasisNormalization::Classical,
            0.,
            1.
        )
        .unwrap()
        .prepare(0, options)
        .is_err_and(|error| error.kind() == scioxide_core::SciOxideErrorKind::InvalidInput)
    );
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        PolynomialOperationPlan::<f64>::build(
            &[PolynomialOperationRequest::Multiply {
                left_len: 1,
                right_len: 1
            }],
            PolynomialExecutionOptions::new(),
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
}

#[test]
fn empty_batches_still_observe_cancellation() {
    let options = PolynomialExecutionOptions::new();
    let sparse =
        SparseEvaluationPlan::<f64>::new(1, MonomialOrder::Lexicographic, &[], &[], options)
            .unwrap();
    let layout = StridedLayout::new(&[0, 1], &[1, 1]).unwrap();
    cancellation_boundaries([], |cx, out: &mut [f64; 0]| {
        sparse.evaluate_batch_into(NdView::new(&[], layout).unwrap(), out, &mut [], cx)
    });
    let domains = [Interval::new(0., 1.).unwrap()];
    let tensor = TensorBernsteinView::new(&[2], &domains, 1, &[0., 2.], options).unwrap();
    cancellation_boundaries([], |cx, out: &mut [f64; 0]| {
        tensor.evaluate_batch_into(NdView::new(&[], layout).unwrap(), out, &mut [0.; 2], cx)
    });
    let orthogonal = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Classical,
        0.,
        1.,
    )
    .unwrap()
    .prepare(1, options)
    .unwrap();
    cancellation_boundaries([], |cx, out: &mut [f64; 0]| {
        orthogonal.evaluate_batch_into(&[1., 2.], &[], out, &mut [], cx)
    });
    let exact = ExactSparseView::<Q>::new(1, MonomialOrder::Lexicographic, &[], &[]).unwrap();
    cancellation_boundaries([], |cx, out: &mut [Q; 0]| {
        exact.evaluate_batch_into(NdView::new(&[], layout).unwrap(), out, &mut [], cx)
    });
}

#[test]
fn incompatible_elimination_workspaces_return_dimension_mismatch() {
    let policy = PolynomialApproximationPolicy::default();
    let plan = SylvesterElimination::<f64>::new(2, 2, 0, policy).unwrap();
    let mut small = SylvesterElimination::new(1, 1, 0, policy)
        .unwrap()
        .workspace()
        .unwrap();
    let cx = ExecutionContext::sequential();
    let error = plan
        .coefficient_scoped(&[1., 0., 1.], &[-1., 0., 1.], 0, &mut small, cx)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    let d = SylvesterDiscriminant::new(2, policy).unwrap();
    let mut small = SylvesterDiscriminant::new(1, policy)
        .unwrap()
        .workspace()
        .unwrap();
    let error = d.compute_scoped(&[1., 0., 1.], &mut small, cx).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
}

struct BoundedExecutor {
    maximum: AtomicUsize,
}
impl Executor for BoundedExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::MIN.saturating_add(3)
    }
    fn execute(
        &self,
        cx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.maximum
            .fetch_max(plan.worker_count(), Ordering::Relaxed);
        SequentialExecutor.execute(cx, plan, operation)
    }
}

#[test]
fn fft_execution_respects_retained_worker_bound() {
    let executor = BoundedExecutor {
        maximum: AtomicUsize::new(0),
    };
    let cx = ExecutionContext::new(&executor);
    let options = PolynomialExecutionOptions::new()
        .with_execution_policy(ExecutionPolicy::Auto)
        .with_worker_bound(NonZeroUsize::MIN.saturating_add(1));
    // Packed-real FFTs use a half-length complex child. That child must reach
    // FFT's parallel gate before this test can observe the retained bound.
    let plan = PolynomialOperationPlan::<f64>::build(
        &[PolynomialOperationRequest::Multiply {
            left_len: 65536,
            right_len: 65536,
        }],
        options,
        cx,
    )
    .unwrap();
    let mut ws = plan.workspace(plan.requests()).unwrap();
    let a = vec![1.; 65536];
    let mut output = vec![0.; 131071];
    plan.multiply_into(&a, &a, &mut output, &mut ws, cx)
        .unwrap();
    assert!((output[65535] - 65536.).abs() < 1e-8);
    assert!(executor.maximum.load(Ordering::Relaxed) <= 2);
    assert!(executor.maximum.load(Ordering::Relaxed) > 1);
}
