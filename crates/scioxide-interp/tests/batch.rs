//! Batch shell: requirements protocol, transactions, executors and allocation.
#![allow(clippy::unwrap_used)]

use core::num::NonZeroUsize;

use scioxide_core::{
    AffineValueMut, Capacity, ExecutionContext, ExecutionPolicy, Interval, NdView, NdViewMut,
    SciOxideError, SciOxideErrorKind, StridedLayout,
};
use scioxide_interp::{
    BatchEvaluator, BatchRequest, BicubicInterpolator, BilinearInterpolator, Differentiable,
    ExtrapolationPolicy, ExtrapolationReport, Interpolant, TwiceDifferentiable,
};

const REJECT: ExtrapolationPolicy<f64> = ExtrapolationPolicy::Reject;

/// `component c = (c + 1) * x` on `[0, 10]`, failing exactly at `fail_at`.
struct Ramp {
    dimension: usize,
    fail_at: f64,
}

impl Interpolant<f64, 1> for Ramp {
    type Value = Vec<f64>;
    type Workspace = Vec<f64>;
    fn domains(&self) -> [Interval<f64>; 1] {
        [Interval::new(0., 10.).unwrap()]
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<Vec<f64>, SciOxideError> {
        Ok(vec![0.; self.dimension])
    }
    fn workspace(&self) -> Result<Vec<f64>, SciOxideError> {
        Ok(vec![0.; self.dimension])
    }
    fn workspace_capacity(&self, workspace: &Vec<f64>) -> Capacity {
        if workspace.len() >= self.dimension {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        }
    }
    fn evaluate_into<O: AffineValueMut<Scale = f64> + ?Sized>(
        &self,
        point: [f64; 1],
        policy: ExtrapolationPolicy<f64>,
        out: &mut O,
        workspace: &mut Vec<f64>,
    ) -> Result<(), SciOxideError> {
        if self.workspace_capacity(workspace) != Capacity::Sufficient {
            return Err(SciOxideError::dimension_mismatch(
                [self.dimension],
                [workspace.len()],
            ));
        }
        let x = policy.resolve(self.domains(), point)?.effective()[0];
        if x == self.fail_at {
            return Err(SciOxideError::singular(None));
        }
        let mut scale = 0.;
        for slot in &mut workspace[..self.dimension] {
            scale += 1.;
            *slot = scale * x;
        }
        scioxide_core::write_value(&workspace[..self.dimension], out)
    }
}

fn ramp(dimension: usize) -> Ramp {
    Ramp {
        dimension,
        fail_at: f64::NAN,
    }
}

fn sentinel<const D: usize>() -> ExtrapolationReport<f64, D> {
    ExtrapolationPolicy::MethodDefined
        .resolve([Interval::new(0., 1.).unwrap(); D], [99.; D])
        .unwrap()
}

/// Only combined evaluation succeeds, detecting accidental separate dispatch.
struct CombinedOnly;

impl Interpolant<f64, 1> for CombinedOnly {
    type Value = f64;
    type Workspace = ();
    fn domains(&self) -> [Interval<f64>; 1] {
        [Interval::new(0., 1.).unwrap()]
    }
    fn value_dimension(&self) -> usize {
        1
    }
    fn zero_value(&self) -> Result<f64, SciOxideError> {
        Ok(0.)
    }
    fn workspace(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn workspace_capacity(&self, _: &()) -> Capacity {
        Capacity::Sufficient
    }
    fn evaluate_into<O: AffineValueMut<Scale = f64> + ?Sized>(
        &self,
        _: [f64; 1],
        _: ExtrapolationPolicy<f64>,
        _: &mut O,
        _: &mut (),
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::singular(None))
    }
    fn report(
        &self,
        _: [f64; 1],
        _: ExtrapolationPolicy<f64>,
        _: &mut (),
    ) -> Result<ExtrapolationReport<f64, 1>, SciOxideError> {
        Err(SciOxideError::singular(None))
    }
    fn evaluate_into_with_report<O: AffineValueMut<Scale = f64> + ?Sized>(
        &self,
        point: [f64; 1],
        policy: ExtrapolationPolicy<f64>,
        out: &mut O,
        _: &mut (),
    ) -> Result<ExtrapolationReport<f64, 1>, SciOxideError> {
        let report = policy.resolve(self.domains(), point)?;
        scioxide_core::write_value(&report.effective(), out)?;
        Ok(report)
    }
}

#[test]
fn reported_values_use_combined_dispatch_through_borrowed_models() {
    let model = &CombinedOnly;
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let mut workspace = plan
        .workspace(BatchRequest::values(3).with_reports())
        .unwrap();
    let mut storage = [99.; 3];
    let mut reports = [sentinel(); 3];
    for points in [[0.25, 0.5, 0.75], [0.25, f64::NAN, 0.75]] {
        let queries = NdView::new(&points, layout(&[3, 1], &[1, 1])).unwrap();
        let previous_values = storage;
        let previous_reports = reports;
        let mut out = NdViewMut::new(&mut storage, layout(&[3, 1], &[1, 1])).unwrap();
        let result = plan.evaluate_into(
            &queries,
            REJECT,
            &mut out,
            Some(&mut reports),
            &mut workspace,
            ExecutionContext::sequential_native(),
        );
        if points[1].is_nan() {
            assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
            assert_eq!(storage, previous_values);
            assert_eq!(reports, previous_reports);
        } else {
            result.unwrap();
            assert_eq!(storage, points);
            for (&report, point) in reports.iter().zip(points) {
                assert_eq!(report.effective(), [point]);
            }
        }
    }
    // Reusing the reserved reported batch allocates nothing.
    {
        let queries = NdView::new(&[0.25, 0.5, 0.75], layout(&[3, 1], &[1, 1])).unwrap();
        let mut out = NdViewMut::new(&mut storage, layout(&[3, 1], &[1, 1])).unwrap();
        plan.evaluate_into(
            &queries,
            REJECT,
            &mut out,
            Some(&mut reports),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    };
}

fn layout<'a>(extents: &'a [usize], strides: &'a [usize]) -> StridedLayout<'a> {
    StridedLayout::new(extents, strides).unwrap()
}

fn quadratic_surface() -> BicubicInterpolator<f64, Vec<f64>> {
    let x = vec![-1., 0., 0.5, 2.];
    let y = vec![-2., 0., 1.];
    let values = y
        .iter()
        .flat_map(|&b| {
            x.iter()
                .map(move |&a| vec![a * a + b * b + a * b, 3. * a * a - 2. * b * b + 4. * a * b])
        })
        .collect();
    BicubicInterpolator::new(x, y, values).unwrap()
}

#[test]
fn values_gradients_hessians_and_reports_match_point_forms() {
    let model = quadratic_surface();
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let points = [0.25, 0.5, 3.0, -1.5, -1.0, 0.0];
    let queries = NdView::new(&points, layout(&[3, 2], &[2, 1])).unwrap();
    let policy = ExtrapolationPolicy::Clamp;
    let cx = ExecutionContext::sequential_native();
    let mut point_workspace = model.workspace().unwrap();
    let mut workspace = plan
        .workspace(BatchRequest::hessians(3).with_reports())
        .unwrap();

    // Values into a padded strided view; padding stays untouched.
    let mut storage = [99.; 9];
    let mut reports = [sentinel(); 3];
    {
        let mut out = NdViewMut::new(&mut storage, layout(&[3, 2], &[3, 1])).unwrap();
        plan.evaluate_into(
            &queries,
            policy,
            &mut out,
            Some(&mut reports),
            &mut workspace,
            cx,
        )
        .unwrap();
    }
    for query in 0..3 {
        let point = [points[2 * query], points[2 * query + 1]];
        let mut expected = [0.; 2];
        model
            .evaluate_into(point, policy, &mut expected, &mut point_workspace)
            .unwrap();
        assert_eq!(storage[3 * query..3 * query + 2], expected);
        assert_eq!(storage[3 * query + 2], 99.);
        assert_eq!(
            reports[query],
            model.report(point, policy, &mut point_workspace).unwrap()
        );
    }
    assert_eq!(reports[1].effective(), [2.0, -1.5]);
    let rows = [[0.25, 0.5], [3.0, -1.5], [-1.0, 0.0]];
    let many = model.evaluate_many(&rows, policy).unwrap();
    for (query, value) in many.iter().enumerate() {
        assert_eq!(*value, model.evaluate(rows[query], policy).unwrap());
        assert_eq!(value[..], storage[3 * query..3 * query + 2]);
    }

    let mut gradients = [0.; 12];
    let mut out = NdViewMut::new(&mut gradients, layout(&[3, 2, 2], &[4, 2, 1])).unwrap();
    plan.gradient_into(&queries, policy, &mut out, None, &mut workspace, cx)
        .unwrap();
    let mut hessians = [0.; 24];
    let mut out = NdViewMut::new(&mut hessians, layout(&[3, 2, 2, 2], &[8, 4, 2, 1])).unwrap();
    plan.hessian_into(&queries, policy, &mut out, None, &mut workspace, cx)
        .unwrap();
    for query in 0..3 {
        let point = [points[2 * query], points[2 * query + 1]];
        let gradient = model.gradient(point, policy).unwrap();
        let hessian = model.hessian(point, policy).unwrap();
        assert_eq!(gradients[4 * query..4 * query + 4], *gradient.concat());
        let flat: Vec<f64> = hessian.iter().flatten().flatten().copied().collect();
        assert_eq!(hessians[8 * query..8 * query + 8], *flat);
    }
    // The clamped x axis of the second query has zero x derivatives.
    assert_eq!(gradients[4..6], [0., 0.]);
}

#[test]
fn preflight_rejections_leave_output_untouched() {
    let model = ramp(2);
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let mut workspace = plan.workspace(BatchRequest::values(1)).unwrap();
    let data = [1., 2., 3.];
    let mut storage = [99.; 6];
    let mut reports = [sentinel(); 3];
    let cx = ExecutionContext::sequential_native();
    let kind = |result: Result<(), SciOxideError>| result.unwrap_err().kind();
    let wrong_axes = NdView::new(&data, layout(&[1, 3], &[3, 1])).unwrap();
    let queries = NdView::new(&data, layout(&[3, 1], &[1, 1])).unwrap();
    {
        let mut out = NdViewMut::new(&mut storage, layout(&[3, 2], &[2, 1])).unwrap();
        assert_eq!(
            kind(plan.evaluate_into(&wrong_axes, REJECT, &mut out, None, &mut workspace, cx)),
            SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(
            kind(plan.evaluate_into(
                &queries,
                REJECT,
                &mut out,
                Some(&mut reports[..2]),
                &mut workspace,
                cx
            )),
            SciOxideErrorKind::DimensionMismatch
        );
        // The workspace was reserved for one query only.
        assert_eq!(
            kind(plan.evaluate_into(
                &queries,
                REJECT,
                &mut out,
                Some(&mut reports),
                &mut workspace,
                cx
            )),
            SciOxideErrorKind::DimensionMismatch
        );
    }
    workspace
        .try_reserve(
            &plan
                .workspace_requirements(BatchRequest::values(3))
                .unwrap(),
        )
        .unwrap();
    let mut wrong_width = NdViewMut::new(&mut storage, layout(&[3, 1], &[2, 1])).unwrap();
    assert_eq!(
        kind(plan.evaluate_into(&queries, REJECT, &mut wrong_width, None, &mut workspace, cx)),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(storage, [99.; 6]);
    assert_eq!(reports, [sentinel(); 3]);
}

#[test]
fn failing_queries_are_transactional_and_the_workspace_stays_reusable() {
    let model = Ramp {
        dimension: 2,
        fail_at: 2.,
    };
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let mut workspace = plan
        .workspace(BatchRequest::values(3).with_reports())
        .unwrap();
    let cx = ExecutionContext::sequential_native();
    let mut storage = [99.; 6];
    let mut reports = [sentinel(); 3];
    for failing in [[1., 2., 3.], [1., 11., 3.], [1., f64::NAN, 3.]] {
        let queries = NdView::new(&failing, layout(&[3, 1], &[1, 1])).unwrap();
        let mut out = NdViewMut::new(&mut storage, layout(&[3, 2], &[2, 1])).unwrap();
        assert!(
            plan.evaluate_into(
                &queries,
                REJECT,
                &mut out,
                Some(&mut reports),
                &mut workspace,
                cx
            )
            .is_err()
        );
        assert_eq!(storage, [99.; 6]);
        assert_eq!(reports, [sentinel(); 3]);
    }
    let queries = NdView::new(&[1., 3., 4.], layout(&[3, 1], &[1, 1])).unwrap();
    let mut out = NdViewMut::new(&mut storage, layout(&[3, 2], &[2, 1])).unwrap();
    plan.evaluate_into(
        &queries,
        REJECT,
        &mut out,
        Some(&mut reports),
        &mut workspace,
        cx,
    )
    .unwrap();
    assert_eq!(storage, [1., 2., 3., 6., 4., 8.]);
    assert_eq!(reports[2].original(), [4.]);
}

#[test]
fn evaluate_many_returns_values_in_order_or_the_first_error() {
    let model = Ramp {
        dimension: 2,
        fail_at: 2.,
    };
    assert_eq!(
        model.evaluate_many(&[[1.], [3.], [4.]], REJECT).unwrap(),
        [vec![1., 2.], vec![3., 6.], vec![4., 8.]]
    );
    assert!(model.evaluate_many(&[], REJECT).unwrap().is_empty());
    let first = |points: &[[f64; 1]]| model.evaluate_many(points, REJECT).unwrap_err().kind();
    assert_eq!(first(&[[1.], [2.], [11.]]), SciOxideErrorKind::Singular);
    assert_eq!(first(&[[1.], [11.], [2.]]), SciOxideErrorKind::DomainError);
    assert_eq!(first(&[[f64::NAN], [2.]]), SciOxideErrorKind::NonFinite);
    let clamped = model
        .evaluate_many(&[[-1.], [12.]], ExtrapolationPolicy::Clamp)
        .unwrap();
    assert_eq!(clamped, [vec![0., 0.], vec![10., 20.]]);
}

#[test]
fn reserved_batches_execute() {
    let model = quadratic_surface();
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let mut workspace = plan
        .workspace(BatchRequest::hessians(2).with_reports())
        .unwrap();
    let points = [0.25, 0.5, 1.0, -1.0];
    let queries = NdView::new(&points, layout(&[2, 2], &[2, 1])).unwrap();
    let mut values = [0.; 4];
    let mut hessians = [0.; 16];
    let mut reports = [sentinel(); 2];
    {
        let mut out = NdViewMut::new(&mut values, layout(&[2, 2], &[2, 1])).unwrap();
        plan.evaluate_into(
            &queries,
            REJECT,
            &mut out,
            Some(&mut reports),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        let mut out = NdViewMut::new(&mut hessians, layout(&[2, 2, 2, 2], &[8, 4, 2, 1])).unwrap();
        plan.hessian_into(
            &queries,
            REJECT,
            &mut out,
            None,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    };
    assert!(values.iter().all(|value| value.is_finite()));
}

#[test]
fn requirements_protocol_grows_in_place_and_shares_workspaces() {
    let (small, wide) = (ramp(1), ramp(3));
    let small_plan = BatchEvaluator::new(&small, ExecutionPolicy::Sequential);
    let wide_plan = BatchEvaluator::new(&wide, ExecutionPolicy::Sequential);
    let mut workspace = small_plan.workspace(BatchRequest::values(2)).unwrap();
    let small_requirements = small_plan
        .workspace_requirements(BatchRequest::values(2))
        .unwrap();
    assert_eq!(
        workspace.check(&small_requirements).unwrap(),
        Capacity::Sufficient
    );
    let wide_requirements = wide_plan
        .workspace_requirements(BatchRequest::values(4).with_reports())
        .unwrap();
    assert_eq!(
        (wide_requirements.width(), wide_requirements.workers()),
        (3, 1)
    );
    assert_eq!(
        wide_requirements.staging_bytes().unwrap(),
        12 * 8 + 4 * core::mem::size_of::<ExtrapolationReport<f64, 1>>()
    );
    // The worker workspace is too small for the wider model.
    assert_eq!(
        workspace.check(&wide_requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&wide_requirements).unwrap();
    assert_eq!(
        workspace.check(&wide_requirements).unwrap(),
        Capacity::Sufficient
    );
    assert_eq!(
        workspace.check(&small_requirements).unwrap(),
        Capacity::Sufficient
    );
    // Grid families share their workspace type across plans.
    let grid = BilinearInterpolator::new(vec![0., 1.], vec![0., 1.], vec![0., 1., 2., 3.]).unwrap();
    let surface =
        BicubicInterpolator::new(vec![0., 1.], vec![0., 1.], vec![0., 1., 2., 3.]).unwrap();
    let grid_plan = BatchEvaluator::new(&grid, ExecutionPolicy::Sequential);
    let surface_plan = BatchEvaluator::new(&surface, ExecutionPolicy::Sequential);
    let mut shared = grid_plan.workspace(BatchRequest::values(1)).unwrap();
    let hessians = surface_plan
        .workspace_requirements(BatchRequest::hessians(1))
        .unwrap();
    assert_eq!(shared.check(&hessians).unwrap(), Capacity::NeedsGrowth);
    shared.try_reserve(&hessians).unwrap();
    let queries = NdView::new(&[0.5, 0.5], layout(&[1, 2], &[2, 1])).unwrap();
    let mut storage = [0.; 4];
    let mut out = NdViewMut::new(&mut storage, layout(&[1, 2, 2, 1], &[4, 2, 1, 1])).unwrap();
    surface_plan
        .hessian_into(
            &queries,
            REJECT,
            &mut out,
            None,
            &mut shared,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
}

#[test]
fn execution_policy_bounds_reserved_workers() {
    let model = ramp(1);
    let three = NonZeroUsize::new(3).unwrap();
    let workers = |policy, queries| {
        BatchEvaluator::new(&model, policy)
            .workspace_requirements(BatchRequest::values(queries))
            .unwrap()
            .workers()
    };
    assert_eq!(workers(ExecutionPolicy::Sequential, 100), 1);
    assert_eq!(workers(ExecutionPolicy::Parallel(three), 2), 2);
    assert_eq!(workers(ExecutionPolicy::Parallel(three), 100), 3);
    assert_eq!(workers(ExecutionPolicy::Auto, 100), 64);
    assert_eq!(workers(ExecutionPolicy::Auto, 0), 0);
}

#[test]
fn empty_zero_width_and_broadcast_batches() {
    let model = ramp(1);
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let mut workspace = plan
        .workspace(BatchRequest::values(3).with_reports())
        .unwrap();
    let cx = ExecutionContext::sequential_native();
    // Broadcast queries evaluate one coordinate repeatedly.
    let queries = NdView::new(
        &[0.5],
        StridedLayout::new_broadcast(&[3, 1], &[0, 0]).unwrap(),
    )
    .unwrap();
    let mut storage = [0.; 3];
    let mut out = NdViewMut::new(&mut storage, layout(&[3, 1], &[1, 1])).unwrap();
    plan.evaluate_into(&queries, REJECT, &mut out, None, &mut workspace, cx)
        .unwrap();
    assert_eq!(storage, [0.5; 3]);
    // Empty batches validate shapes and succeed.
    let empty = NdView::new(&[], layout(&[0, 1], &[1, 1])).unwrap();
    let mut nothing: [f64; 0] = [];
    let mut out = NdViewMut::new(&mut nothing, layout(&[0, 1], &[1, 1])).unwrap();
    plan.evaluate_into(&empty, REJECT, &mut out, Some(&mut []), &mut workspace, cx)
        .unwrap();
    // Zero-width values still validate and report each coordinate.
    let flat = ramp(0);
    let flat_plan = BatchEvaluator::new(&flat, ExecutionPolicy::Sequential);
    let mut flat_workspace = flat_plan
        .workspace(BatchRequest::values(3).with_reports())
        .unwrap();
    let points = [0.5, 1.5, 12.];
    let queries = NdView::new(&points, layout(&[3, 1], &[1, 1])).unwrap();
    let mut out = NdViewMut::new(&mut nothing, layout(&[3, 0], &[0, 1])).unwrap();
    let mut reports = [sentinel(); 3];
    assert!(
        flat_plan
            .evaluate_into(
                &queries,
                REJECT,
                &mut out,
                Some(&mut reports),
                &mut flat_workspace,
                cx
            )
            .is_err()
    );
    assert_eq!(reports, [sentinel(); 3]);
    flat_plan
        .evaluate_into(
            &queries,
            ExtrapolationPolicy::Clamp,
            &mut out,
            Some(&mut reports),
            &mut flat_workspace,
            cx,
        )
        .unwrap();
    assert_eq!(reports[2].effective(), [10.]);
}

struct FaultyExecutor {
    mode: usize,
}

impl scioxide_core::Executor for FaultyExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(3).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        use scioxide_core::SequentialExecutor;
        match self.mode {
            1 => Err(SciOxideError::execution_failure(
                None,
                scioxide_core::ExecutionFailureReason::ProtocolViolation,
                "rejected",
            )),
            // Skip every partition.
            2 => SequentialExecutor.execute(
                context,
                context.plan(
                    ExecutionPolicy::Sequential,
                    0,
                    scioxide_core::PlanLimits {
                        work_per_item: NonZeroUsize::MIN,
                        auto_threshold: NonZeroUsize::MIN,
                        worker_limit: NonZeroUsize::MIN,
                    },
                ),
                operation,
            ),
            // Run every partition twice.
            3 => {
                SequentialExecutor.execute(context, plan, operation)?;
                SequentialExecutor.execute(context, plan, operation)
            }
            _ => SequentialExecutor.execute(context, plan, operation),
        }
    }
}

#[test]
fn executor_failures_and_cancellation_are_atomic() {
    let model = ramp(1);
    let policy = ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap());
    let plan = BatchEvaluator::new(&model, policy);
    let points = [0.25, 0.5, 1.5];
    let queries = NdView::new(&points, layout(&[3, 1], &[1, 1])).unwrap();
    for mode in 0..5 {
        let executor = FaultyExecutor { mode };
        let token = scioxide_core::CancellationToken::new();
        if mode == 4 {
            token.cancel();
        }
        let context = ExecutionContext::new(&executor).with_cancellation(&token);
        let mut workspace = plan
            .workspace(BatchRequest::values(3).with_reports())
            .unwrap();
        let mut storage = [99.; 3];
        let mut reports = [sentinel(); 3];
        let mut out = NdViewMut::new(&mut storage, layout(&[3, 1], &[1, 1])).unwrap();
        let result = plan.evaluate_into(
            &queries,
            REJECT,
            &mut out,
            Some(&mut reports),
            &mut workspace,
            context,
        );
        if mode == 0 {
            result.unwrap();
            assert_eq!(storage, points);
        } else {
            assert!(result.is_err(), "mode {mode}");
            assert_eq!(storage, [99.; 3]);
            assert_eq!(reports, [sentinel(); 3]);
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn parallel_policy_matches_sequential_bits() {
    let workers = NonZeroUsize::new(3).unwrap();
    let pool = scioxide_core::ThreadExecutor::new(workers);
    let model = quadratic_surface();
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Parallel(workers));
    let points: Vec<f64> = (0..40).map(|i| -1.0 + 0.07 * f64::from(i)).collect();
    let queries = NdView::new(&points, layout(&[20, 2], &[2, 1])).unwrap();
    let mut results = [[0.; 80]; 2];
    for (storage, context) in results.iter_mut().zip([
        ExecutionContext::sequential_native(),
        ExecutionContext::new(&pool),
    ]) {
        let mut workspace = plan.workspace(BatchRequest::gradients(20)).unwrap();
        let mut out = NdViewMut::new(storage, layout(&[20, 2, 2], &[4, 2, 1])).unwrap();
        plan.gradient_into(
            &queries,
            ExtrapolationPolicy::MethodDefined,
            &mut out,
            None,
            &mut workspace,
            context,
        )
        .unwrap();
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn zero_component_batches_execute_values_gradients_and_hessians() {
    let model =
        BicubicInterpolator::new(vec![0., 1.], vec![0., 1.], vec![[0.0_f64; 0]; 4]).unwrap();
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let cx = ExecutionContext::sequential_native();
    for operation in 0..3 {
        let request = [
            BatchRequest::values(2),
            BatchRequest::gradients(2),
            BatchRequest::hessians(2),
        ][operation];
        let mut workspace = plan.workspace(request.with_reports()).unwrap();
        let shape: &[usize] = match operation {
            0 => &[2, 0],
            1 => &[2, 2, 0],
            _ => &[2, 2, 2, 0],
        };
        let strides = vec![0; shape.len()];
        let sentinel = sentinel::<2>();
        let mut reports = [sentinel; 2];
        let mut storage = [];
        let mut out = NdViewMut::new(&mut storage, layout(shape, &strides)).unwrap();
        for reported in [false, true] {
            for (point, policy) in [
                ([f64::NAN, 0.5], REJECT),
                ([f64::INFINITY, 0.5], ExtrapolationPolicy::Clamp),
                ([1.5, 0.5], REJECT),
                (
                    [0.5, 0.5],
                    ExtrapolationPolicy::BoundedContinue { max_distance: -1.0 },
                ),
                (
                    [0.5, 0.5],
                    ExtrapolationPolicy::BoundedContinue {
                        max_distance: f64::NAN,
                    },
                ),
                (
                    [0.5, 0.5],
                    ExtrapolationPolicy::BoundedContinue {
                        max_distance: f64::INFINITY,
                    },
                ),
                (
                    [1.5, 0.5],
                    ExtrapolationPolicy::BoundedContinue { max_distance: 0.25 },
                ),
                (
                    [1.5, 0.5],
                    ExtrapolationPolicy::BoundedContinue { max_distance: 0.5 },
                ),
            ] {
                let coordinates = [0.25, 0.25, point[0], point[1]];
                let queries = NdView::new(&coordinates, layout(&[2, 2], &[2, 1])).unwrap();
                let expected = match operation {
                    0 => model.evaluate(point, policy).map(|_| ()),
                    1 => model.gradient(point, policy).map(|_| ()),
                    _ => model.hessian(point, policy).map(|_| ()),
                };
                reports.fill(sentinel);
                {
                    let reports = if reported {
                        Some(&mut reports[..])
                    } else {
                        None
                    };
                    let result = match operation {
                        0 => plan.evaluate_into(
                            &queries,
                            policy,
                            &mut out,
                            reports,
                            &mut workspace,
                            cx,
                        ),
                        1 => plan.gradient_into(
                            &queries,
                            policy,
                            &mut out,
                            reports,
                            &mut workspace,
                            cx,
                        ),
                        _ => plan.hessian_into(
                            &queries,
                            policy,
                            &mut out,
                            reports,
                            &mut workspace,
                            cx,
                        ),
                    };
                    assert_eq!(
                        result.as_ref().err().map(|e| e.kind()),
                        expected.as_ref().err().map(|e| e.kind())
                    );
                };
                if expected.is_err() {
                    assert_eq!(reports, [sentinel; 2]);
                } else if reported {
                    assert_eq!(
                        reports[1],
                        model
                            .report(point, policy, &mut model.workspace().unwrap())
                            .unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn zero_component_reports_check_cancellation_before_publication() {
    struct CancelAfterWork<'a>(&'a scioxide_core::CancellationToken);
    impl scioxide_core::Executor for CancelAfterWork<'_> {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::MIN
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: scioxide_core::ExecutionPlan,
            operation: &scioxide_core::PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            scioxide_core::SequentialExecutor.execute(context, plan, operation)?;
            self.0.cancel();
            Ok(())
        }
    }
    let token = scioxide_core::CancellationToken::new();
    let executor = CancelAfterWork(&token);
    let context = ExecutionContext::new(&executor).with_cancellation(&token);
    let plan = BatchEvaluator::new(ramp(0), ExecutionPolicy::Sequential);
    let mut workspace = plan
        .workspace(BatchRequest::values(1).with_reports())
        .unwrap();
    let queries = NdView::new(&[0.5], layout(&[1, 1], &[1, 1])).unwrap();
    let mut storage = [];
    let mut out = NdViewMut::new(&mut storage, layout(&[1, 0], &[0, 1])).unwrap();
    let mut reports = [sentinel()];
    assert!(
        plan.evaluate_into(
            &queries,
            REJECT,
            &mut out,
            Some(&mut reports),
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(reports, [sentinel()]);
}
