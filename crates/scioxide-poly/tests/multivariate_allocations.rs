//! Prepared sparse and Bernstein execution must not allocate, including first use.
#![allow(clippy::unwrap_used)]
#[cfg(feature = "std")]
use core::num::NonZeroUsize;

use scioxide_core::*;
use scioxide_poly::*;

#[test]
fn first_and_reused_numerical_execution() {
    let p =
        ApproxSparsePolynomial::from_terms(2, MonomialOrder::Lexicographic, vec![([1, 1], 2f64)])
            .unwrap();
    let tensor = TensorBernstein::new(
        &[1, 1],
        vec![Interval::new(0., 1.).unwrap(); 2],
        1,
        vec![0., 0., 0., 2.],
    )
    .unwrap();
    let points = [0.25, 0.5, 1., 1.];
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let mut output = [0.; 2];
    let mut work = [0.; 6];
    for _ in 0..2 {
        {
            p.evaluation_plan(scioxide_poly::PolynomialExecutionOptions::new())
                .unwrap()
                .evaluate_batch_into(
                    NdView::new(&points, layout).unwrap(),
                    &mut output,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(output, [0.25, 2.]);
            tensor
                .as_view()
                .evaluate_batch_into(
                    NdView::new(&points, layout).unwrap(),
                    &mut output,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(output, [0.25, 2.]);
        };
    }
}
#[test]
#[cfg(feature = "std")]
fn threaded_numerical_workers_have_no_hidden_allocations() {
    let workers = NonZeroUsize::new(3).unwrap();
    let executor = CheckedExecutor(ThreadExecutor::new(workers));
    let tensor = TensorBernstein::with_options(
        &[1],
        vec![Interval::new(0., 1.).unwrap()],
        2,
        vec![0., 1., 1., 0.],
        scioxide_poly::PolynomialExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Parallel(workers)),
    )
    .unwrap();
    let points = [0., 0.25, 0.5, 0.75, 1.];
    let layout = StridedLayout::new(&[5, 1], &[1, 1]).unwrap();
    let mut output = [0.; 10];
    let view = tensor.as_view();
    let mut scratch = vec![0.; view.workspace_requirements(5).unwrap().scratch_len()];
    {
        view.evaluate_batch_into(
            NdView::new(&points, layout).unwrap(),
            &mut output,
            &mut scratch,
            ExecutionContext::new(&executor),
        )
        .unwrap()
    }
    assert_eq!(output, [0., 1., 0.25, 0.75, 0.5, 0.5, 0.75, 0.25, 1., 0.]);
    let polynomial =
        ApproxSparsePolynomial::from_terms(1, MonomialOrder::Lexicographic, vec![([1], 2.0_f64)])
            .unwrap();
    let plan = polynomial
        .evaluation_plan(
            PolynomialExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Parallel(workers)),
        )
        .unwrap();
    let mut scratch = vec![0.; plan.workspace_requirements(5).unwrap().scratch_len()];
    let mut values = [0.; 5];
    for _ in 0..2 {
        plan.evaluate_batch_into(
            NdView::new(&points, layout).unwrap(),
            &mut values,
            &mut scratch,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        assert_eq!(values, [0., 0.5, 1., 1.5, 2.]);
    }
}

#[cfg(feature = "std")]
struct CheckedExecutor(ThreadExecutor);
#[cfg(feature = "std")]
impl Executor for CheckedExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        self.0.worker_budget()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.0.execute(context, plan, &|partition, (), scope| {
            operation(partition, (), scope)
        })
    }
}

#[test]
fn orthogonal_batches_allocate_no_metadata_on_first_or_repeated_use() {
    let plan = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Classical,
        0.0_f64,
        1.0,
    )
    .unwrap()
    .prepare(1, PolynomialExecutionOptions::new())
    .unwrap();
    let mut output = [0.; 3];
    let mut scratch = [0.; 3];
    for _ in 0..2 {
        {
            plan.evaluate_batch_into(
                &[1., 2.],
                &[0., 0.5, 1.],
                &mut output,
                &mut scratch,
                ExecutionContext::sequential(),
            )
            .unwrap();
        };
        assert_eq!(output, [1., 2., 3.]);
    }
}

#[test]
fn bernstein_value_derivative_and_fallback_execute() {
    let ordinary = ApproxPolynomial::bernstein(vec![1.0_f64; 65], -1., 1.).unwrap();
    let extreme =
        ApproxPolynomial::bernstein(vec![-f64::MAX, f64::MAX], -f64::MAX, f64::MAX).unwrap();
    let complex = ApproxPolynomial::bernstein(vec![Complex::new(1., 2.); 17], 0., 1.).unwrap();
    let complex_extreme =
        ApproxPolynomial::bernstein(vec![Complex::new(f64::MAX / 4., f64::MAX / 4.); 3], 0., 1.)
            .unwrap();
    let mut work = vec![0.; ordinary.workspace_requirements(1).unwrap().scratch_len()];
    let mut complex_work =
        vec![Complex::ZERO; complex.workspace_requirements(1).unwrap().scratch_len()];
    let (mut value, mut slope) = ([0.0_f64], [0.0_f64]);
    let (mut complex_value, mut complex_slope) = ([Complex::ZERO], [Complex::ZERO]);
    let cx = ExecutionContext::sequential();
    for _ in 0..2 {
        {
            ordinary
                .evaluate_batch_into(&[0.25], &mut value, &mut work, cx)
                .unwrap();
            assert_eq!(value, [1.]);
            extreme
                .evaluate_batch_with_derivative_into(&[0.], &mut value, &mut slope, &mut work, cx)
                .unwrap();
            assert_eq!(slope, [1.]);
            complex_extreme
                .evaluate_batch_with_derivative_into(
                    &[Complex::new(0.5, 0.)],
                    &mut complex_value,
                    &mut complex_slope,
                    &mut complex_work,
                    cx,
                )
                .unwrap();
            assert_eq!(complex_slope, [Complex::ZERO]);
            complex
                .evaluate_batch_with_derivative_into(
                    &[Complex::new(0.5, 0.25)],
                    &mut complex_value,
                    &mut complex_slope,
                    &mut complex_work,
                    cx,
                )
                .unwrap();
            std::hint::black_box((complex_value, complex_slope));
        };
    }
    // The owned Bernstein forms allocate their scratch.
    assert!(ordinary.root_condition(0.25).is_ok());
}

#[test]
fn sparse_scalar_batch_and_range_fallback_execute_on_first_use() {
    let p = ApproxSparsePolynomial::from_terms(
        2,
        MonomialOrder::GradedReverseLexicographic,
        (0..128).map(|i| ([i % 16, i / 16], 1.0_f64)),
    )
    .unwrap();
    let extreme = ApproxSparsePolynomial::from_terms(
        1,
        MonomialOrder::Lexicographic,
        vec![([2], 1e-300_f64)],
    )
    .unwrap();
    let complex = ApproxSparsePolynomial::from_terms(
        1,
        MonomialOrder::Lexicographic,
        vec![([2], Complex::new(1e-300_f64, 1e-300))],
    )
    .unwrap();
    let plan = p.evaluation_plan(Default::default()).unwrap();
    let mut work = vec![0.; plan.workspace_requirements(2).unwrap().scratch_len()];
    let complex_plan = complex.evaluation_plan(Default::default()).unwrap();
    let mut complex_work = vec![
        Complex::ZERO;
        complex_plan
            .workspace_requirements(1)
            .unwrap()
            .scratch_len()
    ];
    let extreme_plan = extreme.evaluation_plan(Default::default()).unwrap();
    let mut extreme_work = vec![
        0.;
        extreme_plan
            .workspace_requirements(1)
            .unwrap()
            .scratch_len()
    ];
    let mut output = [0.; 2];
    let points = [1.; 4];
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    for _ in 0..2 {
        {
            assert_eq!(plan.evaluate(&[1., 1.], &mut work).unwrap(), 128.);
            plan.evaluate_batch_into(
                NdView::new(&points, layout).unwrap(),
                &mut output,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(output, [128.; 2]);
            let extreme_value = extreme_plan.evaluate(&[1e200], &mut extreme_work).unwrap();
            assert!((extreme_value / 1e100 - 1.).abs() < 1e-14);
            let value = complex_plan
                .evaluate(&[Complex::new(1e200, 0.)], &mut complex_work)
                .unwrap();
            assert!((value.re / 1e100 - 1.).abs() < 1e-14);
            assert!((value.im / 1e100 - 1.).abs() < 1e-14);
        };
    }
    // The owned form allocates the scratch of the plan.
    assert_eq!(p.evaluate(&[1., 1.]).unwrap(), 128.);
}
