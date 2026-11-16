//! Storage protocol: growable workspaces report requirements, check
//! and reserve retained capacity, and every undersized storage is a dimension
//! mismatch that leaves outputs unchanged.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;
use scioxide_core::{
    Capacity, Complex, ExecutionContext, ExecutionPolicy, Interval, LinearOperator, NdView,
    NdViewMut, SciOxideError, SciOxideErrorView, StridedLayout,
};
use scioxide_poly::*;

#[path = "support/exact_field.rs"]
mod exact_field;
use exact_field::Q;

fn mismatch<T: core::fmt::Debug>(result: Result<T, SciOxideError>) {
    let error = result.unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. }),
        "{error:?}"
    );
}

fn operation_plan_protocol<C: scioxide_core::NumericScalar>() {
    let one = C::ONE;
    let small = PolynomialOperationPlan::<C>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 2,
            right_len: 2,
        },
        PolynomialOperationRequest::Compose {
            outer_len: 3,
            inner_len: 2,
        },
    ])
    .unwrap();
    let large = PolynomialOperationPlan::<C>::new(&[PolynomialOperationRequest::Multiply {
        left_len: 100,
        right_len: 101,
    }])
    .unwrap();
    let small_requirements = small.workspace_requirements(small.requests()).unwrap();
    let large_requirements = large.workspace_requirements(large.requests()).unwrap();
    // Below the FFT crossover no transform storage is required.
    assert_eq!(small_requirements.fft_len(), 0);
    assert_eq!(small_requirements.real_len(), 0);
    assert!(large_requirements.fft_len() > 0);
    assert_eq!(large_requirements.real_len() > 0, C::IS_REAL);
    assert!(large_requirements.coefficient_len() > small_requirements.coefficient_len());
    let bytes = large_requirements.coefficient_len() * core::mem::size_of::<C>()
        + large_requirements.fft_len() * core::mem::size_of::<Complex<C::Real>>()
        + large_requirements.real_len() * core::mem::size_of::<C::Real>()
        + large_requirements.scaled_len() * core::mem::size_of::<scioxide_core::ScaledValue<C>>();
    assert_eq!(large_requirements.required_bytes().unwrap(), bytes);

    let mut workspace = PolynomialWorkspace::new();
    assert_eq!(
        workspace.check(&small_requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let mut output = [C::ZERO; 3];
    mismatch(small.multiply_into(
        &[one, one],
        &[one, one],
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    ));
    mismatch(small.compose_into(
        &[one, one, one],
        &[one, one],
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    ));
    assert_eq!(output, [C::ZERO; 3]);
    // A large product has no extended composition buffers. Reserve each
    // operation's requirements; growth retains the earlier FFT capacity.
    workspace.try_reserve(&large_requirements).unwrap();
    assert_eq!(
        workspace.check(&small_requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&small_requirements).unwrap();
    assert_eq!(
        workspace.check(&large_requirements).unwrap(),
        Capacity::Sufficient
    );
    assert_eq!(
        workspace.check(&small_requirements).unwrap(),
        Capacity::Sufficient
    );
    small
        .multiply_into(
            &[one, one],
            &[one, one],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [one, one + one, one]);
    small
        .compose_into(
            &[one, one, one],
            &[one, one],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [one + one + one, one + one + one, one]);
    // The smaller plan's workspace is undersized for the larger plan.
    let mut undersized = small.workspace(small.requests()).unwrap();
    assert_eq!(
        undersized.check(&small_requirements).unwrap(),
        Capacity::Sufficient
    );
    assert_eq!(
        undersized.check(&large_requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let mut large_output = [one; 200];
    let seed = large_output;
    mismatch(large.multiply_into(
        &[one; 100],
        &[one; 101],
        &mut large_output,
        &mut undersized,
        ExecutionContext::sequential(),
    ));
    assert_eq!(large_output, seed);
}

#[test]
fn operation_workspaces_follow_the_requirements_protocol() {
    operation_plan_protocol::<f32>();
    operation_plan_protocol::<f64>();
    operation_plan_protocol::<Complex<f64>>();
}

#[test]
fn exact_operation_workspaces_follow_the_requirements_protocol() {
    type Ws = ExactPolynomialWorkspace<Q>;
    let small_requirements = Ws::workspace_requirements(3).unwrap();
    let large_requirements = Ws::workspace_requirements(64).unwrap();
    assert!(large_requirements.coefficient_len() > small_requirements.coefficient_len());
    let mut workspace = Ws::new();
    assert_eq!(
        workspace.check(&small_requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let one = Q(1, 1);
    let mut output = [Q(9, 1); 3];
    let cx = ExecutionContext::sequential();
    mismatch(exact_multiply_into(
        &[one, one],
        &[one, one],
        &mut output,
        &mut workspace,
        cx,
    ));
    mismatch(exact_taylor_shift_into(
        &[one, one, one],
        one,
        &mut output,
        &mut workspace,
        cx,
    ));
    assert_eq!(output, [Q(9, 1); 3]);
    workspace.try_reserve(&large_requirements).unwrap();
    assert_eq!(
        workspace.check(&small_requirements).unwrap(),
        Capacity::Sufficient
    );
    exact_multiply_into(&[one, one], &[one, one], &mut output, &mut workspace, cx).unwrap();
    assert_eq!(output, [one, Q(2, 1), one]);
    let reserved = Ws::workspace(3).unwrap();
    assert_eq!(
        reserved.check(&large_requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    assert!(matches!(
        Ws::workspace_requirements(usize::MAX).unwrap_err().view(),
        SciOxideErrorView::SizeOverflow { .. }
    ));
    assert!(matches!(
        Ws::workspace_requirements(0).unwrap_err().view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn exact_workspace_for_a_capacity_serves_every_shorter_output() {
    let mut workspace = ExactPolynomialWorkspace::<Q>::workspace(8).unwrap();
    let cx = ExecutionContext::sequential();
    let q = |n: i64| Q(i128::from(n), 1);
    for len in 1..=8_i64 {
        // (1 + x)^(len − 1) as the product of (1 + x)^(len − 2) and (1 + x).
        let left: Vec<_> = (0..len - 1).map(|k| q(binomial(len - 2, k))).collect();
        let mut product = vec![q(0); len as usize];
        if len == 1 {
            exact_multiply_into(&[q(3)], &[q(2)], &mut product, &mut workspace, cx).unwrap();
            assert_eq!(product, [q(6)]);
        } else {
            exact_multiply_into(&left, &[q(1), q(1)], &mut product, &mut workspace, cx).unwrap();
            let expected: Vec<_> = (0..len).map(|k| q(binomial(len - 1, k))).collect();
            assert_eq!(product, expected);
        }
        // p(x + 1) with p = x^(len − 1) has the same binomial coefficients.
        let mut power = vec![q(0); len as usize];
        power[len as usize - 1] = q(1);
        let mut shifted = vec![q(0); len as usize];
        exact_taylor_shift_into(&power, q(1), &mut shifted, &mut workspace, cx).unwrap();
        let expected: Vec<_> = (0..len).map(|k| q(binomial(len - 1, k))).collect();
        assert_eq!(shifted, expected);
        let mut composed = vec![q(0); len as usize];
        exact_compose_into(&power, &[q(1), q(1)], &mut composed, &mut workspace, cx).unwrap();
        assert_eq!(composed, expected);
    }
    let mut nine = [q(7); 9];
    let left = [q(1); 5];
    let error = exact_multiply_into(&left, &left, &mut nine, &mut workspace, cx).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(nine, [q(7); 9]);
    let mut power = [q(0); 9];
    power[8] = q(1);
    let error = exact_taylor_shift_into(&power, q(1), &mut nine, &mut workspace, cx).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(nine, [q(7); 9]);
}

fn binomial(n: i64, k: i64) -> i64 {
    (0..k).fold(1, |value, i| value * (n - i) / (i + 1))
}

#[test]
fn prepared_polynomial_scratch_is_a_minimum_length() {
    let bernstein = ApproxPolynomial::bernstein(vec![1.0, 2.0, 4.0], 0.0, 1.0).unwrap();
    // Two points stage two values and two derivatives next to three controls.
    let required = bernstein.workspace_requirements(2).unwrap().scratch_len();
    assert_eq!(required, 7);
    let cx = ExecutionContext::sequential();
    let points = [0.25, 0.5];
    let (mut values, mut slopes) = ([9.0; 2], [9.0; 2]);
    let mut scratch = [0.0; 8];
    mismatch(bernstein.evaluate_batch_with_derivative_into(
        &points,
        &mut values,
        &mut slopes,
        &mut scratch[..required - 1],
        cx,
    ));
    // Values alone need `points + controls` entries.
    mismatch(bernstein.evaluate_batch_into(&points, &mut values, &mut scratch[..4], cx));
    assert_eq!((values, slopes), ([9.0; 2], [9.0; 2]));
    bernstein
        .evaluate_batch_into(&points, &mut values, &mut scratch[..5], cx)
        .unwrap();
    let exact = values;
    // Longer scratch is accepted and gives the same result.
    bernstein
        .evaluate_batch_into(&points, &mut values, &mut scratch, cx)
        .unwrap();
    assert_eq!(values, exact);
    bernstein
        .evaluate_batch_with_derivative_into(
            &points,
            &mut values,
            &mut slopes,
            &mut scratch[..required],
            cx,
        )
        .unwrap();
    assert_eq!(values, exact);
    let monomial = ApproxPolynomial::monomial(vec![1.0, 2.0]).unwrap();
    assert_eq!(monomial.workspace_requirements(2).unwrap().scratch_len(), 4);
    assert_eq!(monomial.evaluate(1.0).unwrap(), 3.0);
}

#[test]
fn tensor_batch_scratch_scales_with_active_workers() {
    let domains = [Interval::new(0.0, 1.0).unwrap()];
    let controls = [0.0, 1.0, 1.0, 0.0];
    let workers = NonZeroUsize::new(3).unwrap();
    let options =
        PolynomialExecutionOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers));
    let view = TensorBernsteinView::new(&[2], &domains, 2, &controls, options).unwrap();
    // Two components per point and one two-control lane per active worker.
    assert_eq!(view.workspace_requirements(0).unwrap().scratch_len(), 0);
    assert_eq!(
        view.workspace_requirements(1).unwrap().scratch_len(),
        2 + 2 * 2
    );
    assert_eq!(
        view.workspace_requirements(5).unwrap().scratch_len(),
        10 + 3 * 2
    );
    assert!(matches!(
        view.workspace_requirements(usize::MAX).unwrap_err().view(),
        SciOxideErrorView::SizeOverflow { .. }
    ));
    let points = [0.0, 0.5, 1.0];
    let layout = StridedLayout::new(&[3, 1], &[1, 1]).unwrap();
    let required = view.workspace_requirements(3).unwrap().scratch_len();
    let mut scratch = vec![0.0; required + 1];
    let mut output = [7.0; 6];
    mismatch(view.evaluate_batch_into(
        NdView::new(&points, layout).unwrap(),
        &mut output,
        &mut scratch[..required - 1],
        ExecutionContext::sequential(),
    ));
    assert_eq!(output, [7.0; 6]);
    for len in [required, required + 1] {
        view.evaluate_batch_into(
            NdView::new(&points, layout).unwrap(),
            &mut output,
            &mut scratch[..len],
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(output, [0.0, 1.0, 0.5, 0.5, 1.0, 0.0]);
    }
    let mut point = [9.0; 2];
    let single = view.workspace_requirements(1).unwrap().scratch_len();
    mismatch(view.evaluate_into(
        &[0.5],
        &mut point,
        &mut scratch[..single - 1],
        scioxide_core::ExecutionContext::sequential(),
    ));
    assert_eq!(point, [9.0; 2]);
    view.evaluate_into(
        &[0.5],
        &mut point,
        &mut scratch[..single],
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(point, [0.5, 0.5]);
}

#[test]
fn sparse_and_orthogonal_batches_stage_values_with_worker_scratch() {
    let options = PolynomialExecutionOptions::new();
    let terms_exponents = [2];
    let terms_coefficients = [1.0];
    let sparse = SparseEvaluationPlan::new(
        1,
        MonomialOrder::Lexicographic,
        &terms_exponents,
        &terms_coefficients,
        options,
    )
    .unwrap();
    let points = [1.0, 2.0, 3.0];
    let layout = StridedLayout::new(&[3, 1], &[1, 1]).unwrap();
    assert_eq!(sparse.workspace_requirements(3).unwrap().scratch_len(), 5);
    let mut output = [0.0; 3];
    let mut scratch = [0.0; 6];
    mismatch(sparse.evaluate_batch_into(
        NdView::new(&points, layout).unwrap(),
        &mut output,
        &mut scratch[..2],
        ExecutionContext::sequential(),
    ));
    assert_eq!(output, [0.0; 3]);
    sparse
        .evaluate_batch_into(
            NdView::new(&points, layout).unwrap(),
            &mut output,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [1.0, 4.0, 9.0]);

    let exact_terms_exponents = [1];
    let exact_terms_coefficients = [Q(2, 1)];
    let exact = ExactSparseView::new(
        1,
        MonomialOrder::Lexicographic,
        &exact_terms_exponents,
        &exact_terms_coefficients,
    )
    .unwrap();
    assert_eq!(exact.workspace_requirements(2).unwrap().scratch_len(), 2);
    let exact_points = [Q(1, 2), Q(3, 1)];
    let exact_layout = StridedLayout::new(&[2, 1], &[1, 1]).unwrap();
    let mut exact_output = [Q(0, 1); 2];
    let mut exact_scratch = [Q(0, 1); 3];
    mismatch(exact.evaluate_batch_into(
        NdView::new(&exact_points, exact_layout).unwrap(),
        &mut exact_output,
        &mut exact_scratch[..1],
        ExecutionContext::sequential(),
    ));
    exact
        .evaluate_batch_into(
            NdView::new(&exact_points, exact_layout).unwrap(),
            &mut exact_output,
            &mut exact_scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(exact_output, [Q(1, 1), Q(6, 1)]);

    let plan = OrthogonalBasisSpec::new(
        OrthogonalFamily::ChebyshevT,
        BasisNormalization::Classical,
        0.0,
        1.0,
    )
    .unwrap()
    .prepare(2, options)
    .unwrap();
    assert_eq!(plan.workspace_requirements(3).unwrap().scratch_len(), 3);
    let mut values = [5.0; 3];
    mismatch(plan.evaluate_batch_into(
        &[0.0, 0.0, 1.0],
        &points,
        &mut values,
        &mut scratch[..2],
        ExecutionContext::sequential(),
    ));
    assert_eq!(values, [5.0; 3]);
    // Longer scratch is accepted.
    plan.evaluate_batch_into(
        &[0.0, 0.0, 1.0],
        &points,
        &mut values,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(values, [1.0, 7.0, 17.0]);
}

#[test]
fn orthogonal_operators_report_their_borrowed_scratch() {
    let plan = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Classical,
        0.0_f64,
        1.0,
    )
    .unwrap()
    .prepare(2, PolynomialExecutionOptions::new())
    .unwrap();
    let transform = plan.monomial_operator().unwrap();
    let required = LinearOperator::<f64>::apply_workspace_len(&transform).unwrap();
    assert_eq!(required, LinearOperator::<f64>::rows(&transform));
    let mut output = [4.0; 3];
    let mut scratch = [0.0; 4];
    mismatch(transform.apply_into(&[0.0, 0.0, 1.0], &mut output, &mut scratch[..required - 1]));
    assert_eq!(output, [4.0; 3]);
    transform
        .apply_into(&[0.0, 0.0, 1.0], &mut output, &mut scratch)
        .unwrap();
    assert_eq!(output, [-0.5, 0.0, 1.5]);

    let derivative = plan.derivative_operator();
    assert_eq!(
        LinearOperator::<f64>::apply_workspace_len(&derivative).unwrap(),
        LinearOperator::<f64>::rows(&derivative)
    );
    let primitive = plan.antiderivative_plan().unwrap();
    assert_eq!(
        primitive.workspace_requirements(()).unwrap().scratch_len(),
        primitive.output_plan().degree() + 1
    );
}

#[test]
fn tensor_export_borrows_no_scratch_and_addresses_permuted_strided_tensors() {
    // 3 x + 5 y^2, exported with axis 0 holding y and axis 1 holding x.
    let sparse = ApproxSparsePolynomial::from_terms(
        2,
        MonomialOrder::Lexicographic,
        vec![
            (Monomial::new(&[1, 0]).unwrap(), 3.0),
            (Monomial::new(&[0, 2]).unwrap(), 5.0),
        ],
    )
    .unwrap();
    // Row stride 4 leaves two padding slots per row; they are never written.
    let mut tensor = [8.0; 10];
    let layout = StridedLayout::new(&[3, 2], &[4, 1]).unwrap();
    sparse
        .to_tensor_into(
            NdViewMut::new(&mut tensor, layout).unwrap(),
            &[1, 0],
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(tensor, [0.0, 3.0, 8.0, 8.0, 0.0, 0.0, 8.0, 8.0, 5.0, 0.0]);
    // Every exponent is checked before the tensor changes.
    let mut short = [8.0; 4];
    let error = sparse
        .to_tensor_into(
            NdViewMut::new(&mut short, StridedLayout::new(&[2, 2], &[2, 1]).unwrap()).unwrap(),
            &[1, 0],
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(short, [8.0; 4]);
}
