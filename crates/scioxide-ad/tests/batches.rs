//! Batches: modes, determinism, item counts, errors, sources and allocation.
#![allow(clippy::unwrap_used, clippy::panic, clippy::needless_range_loop)]

use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use scioxide_ad::{
    BaseReal, BatchData, BatchDataFn, BatchEstimate, BatchFunction, BatchMode, BatchOptions,
    BatchOutputs, BatchPlan, BatchReduction, BatchWorkspace, DerivativeMode, Dual, KinkPolicy,
    NonFinitePolicy, Real, Tape, TapeOptions, batch_mean, batch_mean_and_grad,
};

use scioxide_core::{
    CancellationToken, ExecutionContext, NonFiniteOrigin, RealScalar, SciOxideError,
    SciOxideErrorKind, SciOxideErrorView,
};

/// `θ₀·a₀² + θ₁·√(1 + a₁²)·d₀ − θ₀θ₁ + max(a₀ − d₀, 0)`: arithmetic only, so lanes
/// equal scalar evaluation bit for bit.
struct Model;

struct Shared<R> {
    scale: R,
    weight: R,
    offset: R,
}

impl BatchFunction for Model {
    type Shared<R: Real> = Shared<R>;
    fn prepare<R: Real>(&self, p: &[R]) -> Result<Shared<R>, SciOxideError> {
        Ok(Shared {
            scale: p[0],
            weight: p[1],
            offset: p[0] * p[1],
        })
    }
    fn item<R: Real>(&self, s: &Shared<R>, a: &[R], d: &[R::Base]) -> Result<R, SciOxideError> {
        let d0 = R::passive(d[0]);
        let radial = (a[1] * a[1] + 1.0).sqrt();
        Ok(s.scale * a[0] * a[0] + s.weight * radial * d0 - s.offset + (a[0] - d0).max(R::zero()))
    }
}

/// `θ₀·a₀`, failing on every item from `from` on with an error that names the first
/// failing item; `d₀` holds the item index.
struct Failing {
    from: usize,
}

impl Failing {
    /// The error of a batch that contains item `from`.
    fn error(&self) -> SciOxideError {
        SciOxideError::out_of_bounds(scioxide_core::quantity!("item"), self.from, self.from)
    }
}

impl BatchFunction for Failing {
    type Shared<R: Real> = R;
    fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> {
        Ok(p[0])
    }
    fn item<R: Real>(&self, s: &R, a: &[R], d: &[R::Base]) -> Result<R, SciOxideError> {
        let failing = (0..)
            .map_while(|lane| d[0].lane(lane))
            .find(|&index| index >= self.from as f64);
        match failing {
            Some(index) => Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("item"),
                index as usize,
                self.from,
            )),
            None => Ok(*s * a[0]),
        }
    }
}

fn inputs(items: usize) -> Vec<f64> {
    (0..items)
        .flat_map(|i| [0.1 * i as f64 - 0.7, (0.37 * i as f64).sin()])
        .collect()
}

fn data(items: usize) -> Vec<f64> {
    (0..items).map(|i| 0.25 * (i % 7) as f64 - 0.5).collect()
}

/// The item indices, the data of [`Failing`].
fn indices(items: usize) -> Vec<f64> {
    (0..items).map(|i| i as f64).collect()
}

const THETA: [f64; 2] = [1.3, -0.6];

fn plan<const W: usize>(
    items: usize,
    reduction: BatchReduction,
    options: BatchOptions,
) -> BatchPlan<W> {
    BatchPlan::<W>::new(items, 2, 2, 1, reduction, options).unwrap()
}

/// Value-only evaluation (`None`: no gradient requested), forward and reverse mode.
const MODES: [Option<DerivativeMode>; 3] = [
    None,
    Some(DerivativeMode::Forward),
    Some(DerivativeMode::Reverse),
];

/// Blocks of `block` items, in the derivative mode of `mode`.
fn options(block: usize, mode: Option<DerivativeMode>) -> BatchOptions {
    BatchOptions::new()
        .with_block(NonZeroUsize::new(block).unwrap())
        .with_derivative_mode(mode.unwrap_or(DerivativeMode::Auto))
}

/// The initial value of every output buffer.
const UNSET: f64 = -7.25;

/// The values, parameter gradient, gradient error and input gradient of a plan.
struct Buffers {
    values: Vec<f64>,
    gradient: Vec<f64>,
    error: Vec<f64>,
    input_gradient: Vec<f64>,
}

impl Buffers {
    /// The buffers of `plan`, filled with [`UNSET`].
    fn unset<const W: usize>(plan: &BatchPlan<W>) -> Self {
        let (n, np) = (plan.items(), plan.parameters());
        Self {
            values: vec![UNSET; n],
            gradient: vec![UNSET; np],
            error: vec![UNSET; np],
            input_gradient: vec![UNSET; n * plan.inputs()],
        }
    }

    /// The bits of every entry, buffer by buffer.
    fn bits(&self) -> Vec<u64> {
        [
            &self.values,
            &self.gradient,
            &self.error,
            &self.input_gradient,
        ]
        .into_iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect()
    }
}

/// Evaluates `function` at [`THETA`], [`inputs`] and `data`, requesting the values
/// and, with `gradient`, every gradient output (the gradient error from two blocks).
fn evaluate<const W: usize>(
    plan: &BatchPlan<W>,
    function: &impl BatchFunction,
    data: &[f64],
    gradient: bool,
    workspace: &mut BatchWorkspace<f64, W>,
    context: ExecutionContext<'_>,
) -> (Result<BatchEstimate<f64>, SciOxideError>, Buffers) {
    let a = inputs(plan.items());
    evaluate_sources(plan, function, &a[..], data, gradient, workspace, context)
}

/// [`evaluate`] with the inputs `a` and the data `d` from any source.
fn evaluate_sources<const W: usize, A, D>(
    plan: &BatchPlan<W>,
    function: &impl BatchFunction,
    a: &A,
    d: &D,
    gradient: bool,
    workspace: &mut BatchWorkspace<f64, W>,
    context: ExecutionContext<'_>,
) -> (Result<BatchEstimate<f64>, SciOxideError>, Buffers)
where
    A: BatchData + ?Sized,
    D: BatchData + ?Sized,
{
    let mut buffers = Buffers::unset(plan);
    let result = evaluate_buffers(
        plan,
        function,
        a,
        d,
        gradient,
        &mut buffers,
        workspace,
        context,
    );
    (result, buffers)
}

/// [`evaluate_sources`] into `buffers` of the lengths of [`Buffers::unset`]; allocates
/// nothing itself.
#[allow(clippy::too_many_arguments)]
fn evaluate_buffers<const W: usize, A, D>(
    plan: &BatchPlan<W>,
    function: &impl BatchFunction,
    a: &A,
    d: &D,
    gradient: bool,
    buffers: &mut Buffers,
    workspace: &mut BatchWorkspace<f64, W>,
    context: ExecutionContext<'_>,
) -> Result<BatchEstimate<f64>, SciOxideError>
where
    A: BatchData + ?Sized,
    D: BatchData + ?Sized,
{
    let Buffers {
        values,
        gradient: parameter_gradient,
        error,
        input_gradient,
    } = buffers;
    let mut outputs = BatchOutputs::new().with_values(values);
    if gradient {
        outputs = outputs
            .with_parameter_gradient(parameter_gradient)
            .with_input_gradient(input_gradient);
        if plan.blocks() >= 2 {
            outputs = outputs.with_parameter_gradient_error(error);
        }
    }
    plan.evaluate_into(function, &THETA, a, d, outputs, workspace, context)
}

/// The bits of an estimate's value, standard error and derivative, with its item and
/// block counts, mode and kinks.
type EstimateBits = (
    u64,
    Option<u64>,
    Option<u64>,
    usize,
    usize,
    BatchMode,
    Option<u64>,
);

fn estimate_bits(estimate: &BatchEstimate<f64>) -> EstimateBits {
    (
        estimate.value().to_bits(),
        estimate.standard_error().map(f64::to_bits),
        estimate.derivative().map(f64::to_bits),
        estimate.items(),
        estimate.blocks(),
        estimate.mode(),
        estimate.kinks(),
    )
}

/// One item on plain numbers.
fn direct(item: usize) -> (f64, [f64; 2], [f64; 2]) {
    let a = inputs(item + 1);
    let d = data(item + 1);
    let row = [a[2 * item], a[2 * item + 1]];
    let shared = Model.prepare(&THETA).unwrap();
    let value: f64 = Model.item(&shared, &row, &d[item..=item]).unwrap();
    let point = [THETA[0], THETA[1], row[0], row[1]];
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut g = [0.0; 4];
    tape.gradient_into(
        |x| {
            let s = Model.prepare(&x[..2])?;
            Model.item(&s, &x[2..], &d[item..=item])
        },
        &point,
        &mut g,
        ExecutionContext::sequential(),
    )
    .unwrap();
    (value, [g[0], g[1]], [g[2], g[3]])
}

fn counts<const W: usize>() {
    let block = 16;
    let largest = 3 * block + 5;
    let expected: Vec<_> = (0..largest).map(direct).collect();
    let cx = ExecutionContext::sequential();
    // One workspace serves every evaluation, the largest batch first, so each one
    // runs on staging that a larger batch left behind.
    let mut workspace = plan::<W>(largest, BatchReduction::Sum, options(block, None))
        .workspace(1)
        .unwrap();
    let counts = [0, 1, W - 1, W, W + 1, block - 1, block, block + 1, largest];
    for mode in MODES {
        let gradient = mode.is_some();
        for items in counts.into_iter().rev() {
            for reduction in [BatchReduction::Sum, BatchReduction::Mean] {
                let options = options(block, mode);
                if items == 0 && reduction == BatchReduction::Mean {
                    assert!(BatchPlan::<W>::new(0, 2, 2, 1, reduction, options).is_err());
                    continue;
                }
                let case = format!("W={W} {mode:?} items={items} {reduction:?}");
                let plan = plan::<W>(items, reduction, options);
                let (estimate, buffers) =
                    evaluate(&plan, &Model, &data(items), gradient, &mut workspace, cx);
                let estimate = estimate.unwrap();
                assert_eq!(estimate.items(), items, "{case}");
                let mut sum = -0.0;
                let mut parameter = [0.0; 2];
                for (i, &(value, gp, ga)) in expected[..items].iter().enumerate() {
                    assert_eq!(
                        buffers.values[i].to_bits(),
                        value.to_bits(),
                        "{case} item {i}"
                    );
                    for j in 0..2 {
                        parameter[j] += gp[j];
                        if gradient {
                            let batch = buffers.input_gradient[2 * i + j];
                            assert!(
                                (batch - ga[j]).abs() <= 1e-14 * ga[j].abs().max(1.0),
                                "{case} item {i}"
                            );
                        }
                    }
                    sum += value;
                }
                let scale = if reduction == BatchReduction::Mean {
                    items as f64
                } else {
                    1.0
                };
                assert!(
                    (estimate.value() - sum / scale).abs() <= 1e-13 * sum.abs().max(1.0),
                    "{case}"
                );
                if reduction == BatchReduction::Mean && items >= 2 {
                    let mean = sum / scale;
                    let squares: f64 = expected[..items]
                        .iter()
                        .map(|&(value, ..)| (value - mean) * (value - mean))
                        .sum();
                    let se = (squares / (items - 1) as f64 / scale).sqrt();
                    let error = estimate.standard_error().unwrap();
                    assert!((error - se).abs() <= 1e-12 * se, "{case}");
                } else {
                    assert_eq!(estimate.standard_error(), None, "{case}");
                }
                if gradient {
                    for j in 0..2 {
                        let expected = parameter[j] / scale;
                        let batch = buffers.gradient[j];
                        assert!(
                            (batch - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                            "{case}: {batch} against {expected}"
                        );
                    }
                } else {
                    let untouched = buffers.gradient.iter().chain(&buffers.error);
                    let mut untouched = untouched.chain(&buffers.input_gradient);
                    assert!(untouched.all(|v| v.to_bits() == UNSET.to_bits()), "{case}");
                }
                if items == 0 {
                    let zero = 0.0_f64.to_bits();
                    assert_eq!(estimate.value().to_bits(), zero, "{case}");
                    if gradient {
                        let zeros = buffers.gradient.iter().all(|g| g.to_bits() == zero);
                        assert!(zeros, "{case}");
                    }
                }
            }
        }
    }
}

#[test]
fn item_counts_and_padded_lanes() {
    counts::<1>();
    counts::<4>();
    counts::<8>();
}

#[test]
fn the_value_has_the_same_bits_in_every_mode() {
    let n = 101;
    let d = data(n);
    let cx = ExecutionContext::sequential();
    let mut workspace = plan::<8>(n, BatchReduction::Mean, options(32, None))
        .workspace(1)
        .unwrap();
    let runs: Vec<_> = MODES
        .into_iter()
        .map(|mode| {
            let plan = plan::<8>(n, BatchReduction::Mean, options(32, mode));
            let (estimate, buffers) =
                evaluate(&plan, &Model, &d, mode.is_some(), &mut workspace, cx);
            let estimate = estimate.unwrap();
            let values: Vec<u64> = buffers.values.iter().map(|v| v.to_bits()).collect();
            let value = estimate.value().to_bits();
            let error = estimate.standard_error().map(f64::to_bits);
            (estimate.mode(), value, error, values)
        })
        .collect();
    assert_eq!(runs[0].0, BatchMode::Value);
    assert_eq!(runs[1].0, BatchMode::Forward { tangents: 4 });
    assert_eq!(runs[2].0, BatchMode::Reverse);
    assert!(runs[0].2.is_some());
    for run in &runs[1..] {
        assert_eq!(run.1, runs[0].1, "{:?}", run.0);
        assert_eq!(run.2, runs[0].2, "{:?}", run.0);
        assert_eq!(run.3, runs[0].3, "{:?}", run.0);
    }
}

#[cfg(feature = "std")]
#[test]
fn results_are_identical_for_every_execution_policy() {
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    let (n, block) = (203, 16);
    let (d, items) = (data(n), indices(n));
    // Items from 150 on (block 9 of 13) fail, so several partitions fail.
    let failing = Failing { from: 150 };
    let token = CancellationToken::new();
    token.cancel();
    let cx = ExecutionContext::sequential();
    for mode in MODES {
        let gradient = mode.is_some();
        let sequential = plan::<4>(n, BatchReduction::Mean, options(block, mode));
        let unset = Buffers::unset(&sequential).bits();
        let mut workspace = sequential.workspace(1).unwrap();
        let (reference, outputs) = evaluate(&sequential, &Model, &d, gradient, &mut workspace, cx);
        let reference = reference.unwrap();
        let (failure, _) = evaluate(&sequential, &failing, &items, gradient, &mut workspace, cx);
        assert_eq!(failure, Err(failing.error()), "{mode:?}");
        for workers in [1, 2, 3, 8] {
            let case = format!("{mode:?} with {workers} workers");
            let k = NonZeroUsize::new(workers).unwrap();
            let executor = ThreadExecutor::new(k);
            let context = ExecutionContext::new(&executor);
            let parallel = options(block, mode).with_execution_policy(ExecutionPolicy::Parallel(k));
            let plan = plan::<4>(n, BatchReduction::Mean, parallel);
            let mut workspace = plan.workspace(8).unwrap();
            let (estimate, buffers) =
                evaluate(&plan, &Model, &d, gradient, &mut workspace, context);
            let estimate = estimate.unwrap();
            assert_eq!(estimate.mode(), reference.mode(), "{case}");
            assert_eq!(estimate.kinks(), reference.kinks(), "{case}");
            assert_eq!(
                estimate.value().to_bits(),
                reference.value().to_bits(),
                "{case}"
            );
            assert_eq!(
                estimate.standard_error().map(f64::to_bits),
                reference.standard_error().map(f64::to_bits),
                "{case}"
            );
            assert_eq!(buffers.bits(), outputs.bits(), "{case}");
            let (error, buffers) =
                evaluate(&plan, &failing, &items, gradient, &mut workspace, context);
            assert_eq!(error, failure, "{case}");
            assert_eq!(buffers.bits(), unset, "{case}");
            let stopped = context.with_cancellation(&token);
            let (error, buffers) = evaluate(&plan, &Model, &d, gradient, &mut workspace, stopped);
            assert!(
                matches!(
                    error.unwrap_err().view(),
                    SciOxideErrorView::ExecutionStopped { .. }
                ),
                "{case}"
            );
            assert_eq!(buffers.bits(), unset, "{case}");
        }
    }
}

#[test]
fn standard_errors_match_a_direct_computation() {
    let block = 16;
    let n = 5 * block + 3;
    let large = plan::<8>(n, BatchReduction::Mean, options(block, None));
    let mut workspace = large.workspace(1).unwrap();
    let cx = ExecutionContext::sequential();
    let (estimate, buffers) = evaluate(&large, &Model, &data(n), true, &mut workspace, cx);
    let estimate = estimate.unwrap();
    let Buffers {
        values,
        gradient,
        error,
        ..
    } = buffers;
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
    let expected = (variance / n as f64).sqrt();
    assert!((estimate.standard_error().unwrap() - expected).abs() <= 1e-12 * expected);
    let blocks = n.div_ceil(block);
    for j in 0..2 {
        let per_item: Vec<f64> = (0..n).map(|i| direct(i).1[j]).collect();
        let overall = per_item.iter().sum::<f64>() / n as f64;
        assert!((gradient[j] - overall).abs() <= 1e-12 * overall.abs().max(1.0));
        let mut sum = 0.0;
        for b in 0..blocks {
            let range = b * block..((b + 1) * block).min(n);
            let count = range.len() as f64;
            let block_mean = per_item[range].iter().sum::<f64>() / count;
            sum += count * (block_mean - overall).powi(2);
        }
        let se = (sum / ((blocks - 1) * n) as f64).sqrt();
        assert!(
            (error[j] - se).abs() <= 1e-10 * se.max(1e-300),
            "{} {se}",
            error[j]
        );
    }
    // Fewer than two blocks cannot give batch-means errors.
    let small = plan::<8>(block, BatchReduction::Mean, options(block, None));
    let mut workspace = small.workspace(1).unwrap();
    let mut error = [7.0; 2];
    let (a, d) = (inputs(block), data(block));
    let failure = small
        .evaluate_into(
            &Model,
            &THETA,
            &a[..],
            &d[..],
            BatchOutputs::new().with_parameter_gradient_error(&mut error),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        failure.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    assert_eq!(error, [7.0; 2]);
}

/// The values of `inputs` and `data`, generated on demand.
struct Generated {
    inputs: bool,
}

impl BatchData for Generated {
    fn fill(
        &self,
        first: usize,
        width: usize,
        column: usize,
        lanes: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let items = first + lanes.len();
        let source = if self.inputs {
            inputs(items)
        } else {
            data(items)
        };
        for (item, lane) in (first..).zip(lanes) {
            *lane = source[item * width + column];
        }
        Ok(())
    }
}

#[test]
fn a_data_source_gives_the_same_bits_as_a_slice() {
    let n = 45;
    let plan = plan::<4>(n, BatchReduction::Sum, BatchOptions::new());
    let (a, d) = (inputs(n), data(n));
    let mut workspace = plan.workspace(1).unwrap();
    let cx = ExecutionContext::sequential();
    let mut g1 = [0.0; 2];
    let from_slices = plan
        .evaluate_into(
            &Model,
            &THETA,
            &a[..],
            &d[..],
            BatchOutputs::new().with_parameter_gradient(&mut g1),
            &mut workspace,
            cx,
        )
        .unwrap();
    let mut g2 = [0.0; 2];
    let generated = plan
        .evaluate_into(
            &Model,
            &THETA,
            &Generated { inputs: true },
            &Generated { inputs: false },
            BatchOutputs::new().with_parameter_gradient(&mut g2),
            &mut workspace,
            cx,
        )
        .unwrap();
    assert_eq!(from_slices.value().to_bits(), generated.value().to_bits());
    assert_eq!(g1, g2);
}

#[test]
fn cancellation_and_failing_items_leave_outputs_unchanged() {
    let n = 63;
    let (d, items) = (data(n), indices(n));
    let token = CancellationToken::new();
    token.cancel();
    let cx = ExecutionContext::sequential();
    for mode in MODES {
        let gradient = mode.is_some();
        let plan = plan::<4>(n, BatchReduction::Sum, options(8, mode));
        let unset = Buffers::unset(&plan).bits();
        let mut workspace = plan.workspace(1).unwrap();
        // The first item, an item after five staged blocks and the last item, next
        // to a padded lane.
        for from in [0, 45, n - 1] {
            let failing = Failing { from };
            let (error, buffers) = evaluate(&plan, &failing, &items, gradient, &mut workspace, cx);
            assert_eq!(error, Err(failing.error()), "{mode:?} from {from}");
            assert_eq!(buffers.bits(), unset, "{mode:?} from {from}");
        }
        let stopped = cx.with_cancellation(&token);
        let (error, buffers) = evaluate(&plan, &Model, &d, gradient, &mut workspace, stopped);
        assert!(
            matches!(
                error.unwrap_err().view(),
                SciOxideErrorView::ExecutionStopped { .. }
            ),
            "{mode:?}"
        );
        assert_eq!(buffers.bits(), unset, "{mode:?}");
        // The workspace stays usable and gives the bits of a fresh one.
        let (estimate, buffers) = evaluate(&plan, &Model, &d, gradient, &mut workspace, cx);
        let mut fresh = plan.workspace(1).unwrap();
        let (expected, expected_buffers) = evaluate(&plan, &Model, &d, gradient, &mut fresh, cx);
        assert_eq!(
            estimate.unwrap().value().to_bits(),
            expected.unwrap().value().to_bits(),
            "{mode:?}"
        );
        assert_eq!(buffers.bits(), expected_buffers.bits(), "{mode:?}");
    }
}

#[test]
fn warm_sequential_evaluations_execute() {
    let n = 70;
    let (a, d) = (inputs(n), data(n));
    let cx = ExecutionContext::sequential();
    for mode in [DerivativeMode::Auto, DerivativeMode::Reverse] {
        let plan = plan::<8>(
            n,
            BatchReduction::Mean,
            BatchOptions::new().with_derivative_mode(mode),
        );
        let mut workspace = plan.workspace(1).unwrap();
        let mut gradient = [0.0; 2];
        let mut values = vec![0.0; n];
        let mut input_gradient = vec![0.0; 2 * n];
        let mut run = |workspace: &mut scioxide_ad::BatchWorkspace<f64, 8>| {
            plan.evaluate_into(
                &Model,
                &THETA,
                &a[..],
                &d[..],
                BatchOutputs::new()
                    .with_values(&mut values)
                    .with_parameter_gradient(&mut gradient)
                    .with_input_gradient(&mut input_gradient),
                workspace,
                cx,
            )
            .unwrap()
        };
        run(&mut workspace);
        run(&mut workspace);
        plan.evaluate_into(
            &Model,
            &THETA,
            &a[..],
            &d[..],
            BatchOutputs::new(),
            &mut workspace,
            cx,
        )
        .unwrap();
    }
}

#[test]
fn directional_products_agree_with_item_by_item_duals() {
    let n = 21;
    let (a, d) = (inputs(n), data(n));
    let u = [0.3, -0.8];
    let w: Vec<f64> = (0..2 * n).map(|k| 0.05 * k as f64 - 0.4).collect();
    let plan = plan::<4>(n, BatchReduction::Sum, BatchOptions::new());
    let mut workspace = plan.workspace(1).unwrap();
    let cx = ExecutionContext::sequential();
    let mut derivatives = vec![0.0; n];
    let estimate = plan
        .jvp_into(
            &Model,
            &THETA,
            &a[..],
            &d[..],
            &u,
            &w[..],
            &mut derivatives,
            &mut workspace,
            cx,
        )
        .unwrap();
    let mut total = 0.0;
    for i in 0..n {
        let p = [Dual::new(THETA[0], u[0]), Dual::new(THETA[1], u[1])];
        let x = [
            Dual::new(a[2 * i], w[2 * i]),
            Dual::new(a[2 * i + 1], w[2 * i + 1]),
        ];
        let shared = Model.prepare(&p).unwrap();
        let y = Model.item(&shared, &x, &d[i..=i]).unwrap();
        assert_eq!(
            derivatives[i].to_bits(),
            y.derivative().to_bits(),
            "item {i}"
        );
        total += y.derivative();
    }
    assert!((estimate.derivative().unwrap() - total).abs() <= 1e-13 * total.abs().max(1.0));

    let mut parameter_product = [0.0; 2];
    let mut input_product = vec![0.0; 2 * n];
    plan.hvp_into(
        &Model,
        &THETA,
        &a[..],
        &d[..],
        &u,
        &w[..],
        &mut parameter_product,
        &mut input_product,
        &mut workspace,
        cx,
    )
    .unwrap();
    let mut expected = [0.0; 2];
    for i in 0..n {
        let point = [THETA[0], THETA[1], a[2 * i], a[2 * i + 1]];
        let direction = [u[0], u[1], w[2 * i], w[2 * i + 1]];
        let mut tape = Tape::<Dual<f64>>::new(TapeOptions::new());
        let mut product = [0.0; 4];
        tape.hvp_into(
            |x| {
                let s = Model.prepare(&x[..2])?;
                Model.item(&s, &x[2..], &d[i..=i])
            },
            &point,
            &direction,
            &mut product,
            cx,
        )
        .unwrap();
        for j in 0..2 {
            expected[j] += product[j];
            let batch = input_product[2 * i + j];
            assert!((batch - product[2 + j]).abs() <= 1e-13 * product[2 + j].abs().max(1.0));
        }
    }
    for j in 0..2 {
        assert!((parameter_product[j] - expected[j]).abs() <= 1e-12 * expected[j].abs().max(1.0));
    }
}

#[test]
fn forward_mode_agrees_with_reverse_mode() {
    let n = 77;
    let (a, d) = (inputs(n), data(n));
    let cx = ExecutionContext::sequential();
    let options = BatchOptions::new().with_block(NonZeroUsize::new(16).unwrap());
    let mut results = Vec::new();
    for mode in [
        DerivativeMode::Auto,
        DerivativeMode::Forward,
        DerivativeMode::Reverse,
    ] {
        let plan = plan::<8>(n, BatchReduction::Mean, options.with_derivative_mode(mode));
        let mut workspace = plan.workspace(1).unwrap();
        let mut gradient = [0.0; 2];
        let mut input_gradient = vec![0.0; 2 * n];
        let estimate = plan
            .evaluate_into(
                &Model,
                &THETA,
                &a[..],
                &d[..],
                BatchOutputs::new()
                    .with_parameter_gradient(&mut gradient)
                    .with_input_gradient(&mut input_gradient),
                &mut workspace,
                cx,
            )
            .unwrap();
        results.push((estimate, gradient, input_gradient));
    }
    assert_eq!(results[0].0.mode(), BatchMode::Forward { tangents: 4 });
    assert_eq!(results[1].0.mode(), BatchMode::Forward { tangents: 4 });
    assert_eq!(results[2].0.mode(), BatchMode::Reverse);
    for (estimate, gradient, input_gradient) in &results[..2] {
        assert_eq!(estimate.value().to_bits(), results[2].0.value().to_bits());
        for (f, r) in gradient.iter().zip(&results[2].1) {
            assert!((f - r).abs() <= 1e-13 * r.abs().max(1.0));
        }
        for (f, r) in input_gradient.iter().zip(&results[2].2) {
            assert!((f - r).abs() <= 1e-13 * r.abs().max(1.0));
        }
    }
    // Hessian-vector products: forward over forward and forward over reverse.
    let u = [0.3, -0.8];
    let w: Vec<f64> = (0..2 * n).map(|k| 0.05 * k as f64 - 0.4).collect();
    let mut products = Vec::new();
    for mode in [DerivativeMode::Forward, DerivativeMode::Reverse] {
        let plan = plan::<8>(n, BatchReduction::Mean, options.with_derivative_mode(mode));
        let mut workspace = plan.workspace(1).unwrap();
        let mut parameter = [0.0; 2];
        let mut input = vec![0.0; 2 * n];
        plan.hvp_into(
            &Model,
            &THETA,
            &a[..],
            &d[..],
            &u,
            &w[..],
            &mut parameter,
            &mut input,
            &mut workspace,
            cx,
        )
        .unwrap();
        products.push((parameter, input));
    }
    for (f, r) in products[0].0.iter().zip(&products[1].0) {
        assert!((f - r).abs() <= 1e-13 * r.abs().max(1.0));
    }
    for (f, r) in products[0].1.iter().zip(&products[1].1) {
        assert!((f - r).abs() <= 1e-13 * r.abs().max(1.0));
    }
    // More than four active values cannot run forward.
    let wide = BatchPlan::<8>::new(
        n,
        3,
        2,
        1,
        BatchReduction::Sum,
        options.with_derivative_mode(DerivativeMode::Forward),
    )
    .unwrap();
    let mut workspace = wide.workspace(1).unwrap();
    let mut gradient = [0.0; 3];
    let error = wide
        .evaluate_into(
            &Model,
            &[1.0, 2.0, 3.0],
            &a[..],
            &d[..],
            BatchOutputs::new().with_parameter_gradient(&mut gradient),
            &mut workspace,
            cx,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
}

/// `√θ` of the single parameter for every item.
struct RootParameter;

impl BatchFunction for RootParameter {
    type Shared<R: Real> = R;
    fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> {
        Ok(p[0].sqrt())
    }
    fn item<R: Real>(&self, root: &R, _: &[R], _: &[R::Base]) -> Result<R, SciOxideError> {
        Ok(*root)
    }
}

#[test]
fn the_error_policy_checks_only_the_outputs_of_the_evaluation() {
    let options = BatchOptions::new().with_non_finite_policy(scioxide_ad::NonFinitePolicy::Error);
    let plan = BatchPlan::<4>::new(6, 1, 0, 0, BatchReduction::Mean, options).unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let cx = ExecutionContext::sequential();
    let none: &[f64] = &[];
    // d√θ/dθ is infinite at θ = 0.
    let mut gradient = [5.0];
    let outputs = BatchOutputs::new().with_parameter_gradient(&mut gradient);
    let error = plan
        .evaluate_into(
            &RootParameter,
            &[0.0],
            none,
            none,
            outputs,
            &mut workspace,
            cx,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Output,
            ..
        }
    ));
    assert_eq!(gradient, [5.0]);
    // A value-only evaluation on the same workspace ignores the stale gradients.
    let estimate = plan
        .evaluate_into(
            &RootParameter,
            &[4.0],
            none,
            none,
            BatchOutputs::new(),
            &mut workspace,
            cx,
        )
        .unwrap();
    assert_eq!(estimate.value(), 2.0);
}

/// `θ₀·a₀·d₀ + θ₁·a₁·(d₁ − d₂·d₃) + a₀·a₁·d₄ − d₅`: arithmetic only over six data
/// values per item.
struct Wide;

impl BatchFunction for Wide {
    type Shared<R: Real> = [R; 2];
    fn prepare<R: Real>(&self, p: &[R]) -> Result<[R; 2], SciOxideError> {
        Ok([p[0], p[1]])
    }
    fn item<R: Real>(&self, p: &[R; 2], a: &[R], d: &[R::Base]) -> Result<R, SciOxideError> {
        let d = |k: usize| R::passive(d[k]);
        Ok(p[0] * a[0] * d(0) + p[1] * a[1] * (d(1) - d(2) * d(3)) + a[0] * a[1] * d(4) - d(5))
    }
}

/// A row-major table of `items × width` values.
fn table(items: usize, width: usize) -> Vec<f64> {
    (0..items * width)
        .map(|j| 0.3 * (0.7 * (j / width) as f64 + (j % width) as f64).sin() + 0.1)
        .collect()
}

/// A table held value by value: `columns[k][i]` is value `k` of item `i`.
struct Columns(Vec<Vec<f64>>);

impl Columns {
    /// The columns of a row-major table of `width` values per item.
    fn of(table: &[f64], width: usize) -> Self {
        let column = |k| table.iter().skip(k).step_by(width).copied().collect();
        Self((0..width).map(column).collect())
    }
}

impl BatchData for Columns {
    fn fill(
        &self,
        first: usize,
        width: usize,
        column: usize,
        lanes: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(width, self.0.len());
        lanes.copy_from_slice(&self.0[column][first..first + lanes.len()]);
        Ok(())
    }
}

fn sources_agree<const W: usize>() {
    // Three full blocks and a partial last group.
    let n = 3 * 16 + 5;
    let (a, d) = (table(n, 2), table(n, 6));
    let (a_columns, d_columns) = (Columns::of(&a, 2), Columns::of(&d, 6));
    let cx = ExecutionContext::sequential();
    for mode in MODES {
        let gradient = mode.is_some();
        let options = options(16, mode);
        let plan = BatchPlan::<W>::new(n, 2, 2, 6, BatchReduction::Mean, options).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let (expected, expected_buffers) =
            evaluate_sources(&plan, &Wide, &a[..], &d[..], gradient, &mut workspace, cx);
        let (estimate, buffers) = evaluate_sources(
            &plan,
            &Wide,
            &a_columns,
            &d_columns,
            gradient,
            &mut workspace,
            cx,
        );
        assert_eq!(estimate.unwrap(), expected.unwrap(), "W={W} {mode:?}");
        assert_eq!(buffers.bits(), expected_buffers.bits(), "W={W} {mode:?}");
        // The values are those of one item on plain numbers.
        let shared = Wide.prepare(&THETA).unwrap();
        for (i, value) in buffers.values.iter().enumerate() {
            let direct: f64 = Wide
                .item(&shared, &a[2 * i..2 * i + 2], &d[6 * i..6 * i + 6])
                .unwrap();
            assert_eq!(value.to_bits(), direct.to_bits(), "W={W} {mode:?} item {i}");
        }
    }
}

#[test]
fn a_source_held_value_by_value_gives_the_bits_of_its_table() {
    sources_agree::<1>();
    sources_agree::<4>();
    sources_agree::<8>();
}

#[test]
fn a_table_fills_lanes_from_its_rows() {
    let (rows, columns) = (
        scioxide_core::quantity!("batch_rows"),
        scioxide_core::quantity!("batch_columns"),
    );
    for width in 1..=6 {
        let count = 11;
        let values = table(count, width);
        for (first, lanes) in [(0, 8), (3, 8), (10, 1), (4, 5), (0, 0)] {
            for column in 0..width {
                let mut filled = vec![UNSET; lanes];
                BatchData::fill(&values[..], first, width, column, &mut filled).unwrap();
                let expected: Vec<_> = (first..first + lanes)
                    .map(|item| values[item * width + column])
                    .collect();
                assert_eq!(
                    filled, expected,
                    "width {width} first {first} column {column}"
                );
            }
        }
        // A column outside the row, then rows beyond the table.
        let mut lanes = [UNSET; 4];
        let error = BatchData::fill(&values[..], 0, width, width, &mut lanes).unwrap_err();
        assert_eq!(error, SciOxideError::out_of_bounds(columns, width, width));
        let error = BatchData::fill(&values[..], 9, width, 0, &mut lanes).unwrap_err();
        assert_eq!(error, SciOxideError::out_of_bounds(rows, count, count));
        let error = BatchData::fill(&values[..], 14, width, 0, &mut lanes).unwrap_err();
        assert_eq!(error, SciOxideError::out_of_bounds(rows, 14, count));
        assert_eq!(lanes, [UNSET; 4]);
    }
}

/// The values of [`data`], failing for every group that reaches item `from`.
struct Truncated {
    from: usize,
}

impl Truncated {
    /// The error of a batch that contains item `from`.
    fn error(&self) -> SciOxideError {
        SciOxideError::out_of_bounds(scioxide_core::quantity!("source"), self.from, self.from)
    }
}

impl BatchData for Truncated {
    fn fill(
        &self,
        first: usize,
        _: usize,
        _: usize,
        lanes: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let items = first + lanes.len();
        if items > self.from {
            return Err(self.error());
        }
        lanes.copy_from_slice(&data(items)[first..]);
        Ok(())
    }
}

#[test]
fn failing_sources_and_short_tables_leave_outputs_unchanged() {
    let n = 21;
    let (a, d) = (inputs(n), data(n));
    let cx = ExecutionContext::sequential();
    let rows = scioxide_core::quantity!("batch_rows");
    for mode in MODES {
        let gradient = mode.is_some();
        let plan = plan::<4>(n, BatchReduction::Sum, options(8, mode));
        let unset = Buffers::unset(&plan).bits();
        let mut workspace = plan.workspace(1).unwrap();
        // A source that fails in the first group, after two blocks and in the
        // partial last group.
        for from in [2, 17, n - 1] {
            let source = Truncated { from };
            let (error, buffers) =
                evaluate_sources(&plan, &Model, &a[..], &source, gradient, &mut workspace, cx);
            assert_eq!(error, Err(source.error()), "{mode:?} from {from}");
            assert_eq!(buffers.bits(), unset, "{mode:?} from {from}");
        }
        // Tables that end early: the data one row short, the inputs a row and a half.
        let short = [
            (&a[..], &d[..n - 1], n - 1),
            (&a[..2 * n - 3], &d[..], n - 2),
        ];
        for (a, d, missing) in short {
            let (error, buffers) =
                evaluate_sources(&plan, &Model, a, d, gradient, &mut workspace, cx);
            let expected = SciOxideError::out_of_bounds(rows, missing, missing);
            assert_eq!(error, Err(expected), "{mode:?} missing {missing}");
            assert_eq!(buffers.bits(), unset, "{mode:?} missing {missing}");
        }
        // The workspace stays usable and gives the bits of a fresh one.
        let (estimate, buffers) = evaluate(&plan, &Model, &d, gradient, &mut workspace, cx);
        let mut fresh = plan.workspace(1).unwrap();
        let (expected, expected_buffers) = evaluate(&plan, &Model, &d, gradient, &mut fresh, cx);
        assert_eq!(estimate.unwrap(), expected.unwrap(), "{mode:?}");
        assert_eq!(buffers.bits(), expected_buffers.bits(), "{mode:?}");
    }
}

#[test]
fn non_finite_inputs_are_rejected_under_the_error_policy() {
    let n = 21;
    let d = data(n);
    let cx = ExecutionContext::sequential();
    for mode in MODES {
        let gradient = mode.is_some();
        let options = options(8, mode).with_non_finite_policy(scioxide_ad::NonFinitePolicy::Error);
        let plan = plan::<4>(n, BatchReduction::Sum, options);
        let unset = Buffers::unset(&plan).bits();
        let mut workspace = plan.workspace(1).unwrap();
        // The first value of a full group and the last value of the partial group.
        for (position, value) in [(8, f64::NAN), (2 * n - 1, f64::INFINITY)] {
            let mut a = inputs(n);
            a[position] = value;
            let (error, buffers) =
                evaluate_sources(&plan, &Model, &a[..], &d[..], gradient, &mut workspace, cx);
            assert!(
                matches!(
                    error.unwrap_err().view(),
                    SciOxideErrorView::NonFinite {
                        origin: scioxide_core::NonFiniteOrigin::Input,
                        ..
                    }
                ),
                "{mode:?} position {position}"
            );
            assert_eq!(buffers.bits(), unset, "{mode:?} position {position}");
        }
    }
}

/// `θ·a₀·a₁ + d₀·a₁²`: one parameter and two inputs, three active values.
struct Three;

impl BatchFunction for Three {
    type Shared<R: Real> = R;
    fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> {
        Ok(p[0])
    }
    fn item<R: Real>(&self, theta: &R, a: &[R], d: &[R::Base]) -> Result<R, SciOxideError> {
        Ok(*theta * a[0] * a[1] + R::passive(d[0]) * a[1] * a[1])
    }
}

#[test]
fn three_active_values_carry_three_tangents() {
    let n = 37;
    let (a, d) = (inputs(n), data(n));
    let w: Vec<f64> = (0..2 * n).map(|k| 0.05 * k as f64 - 0.4).collect();
    let cx = ExecutionContext::sequential();
    // The mode, value and gradient of an evaluation, then the mode and products of
    // a Hessian-vector product; the parameter entry comes first.
    let run = |mode: DerivativeMode| {
        let options = BatchOptions::new().with_derivative_mode(mode);
        let plan = BatchPlan::<8>::new(n, 1, 2, 1, BatchReduction::Mean, options).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut gradient = vec![0.0; 1 + 2 * n];
        let (parameter, input) = gradient.split_at_mut(1);
        let outputs = BatchOutputs::new()
            .with_parameter_gradient(parameter)
            .with_input_gradient(input);
        let first = plan
            .evaluate_into(&Three, &[0.7], &a[..], &d[..], outputs, &mut workspace, cx)
            .unwrap();
        let mut product = vec![0.0; 1 + 2 * n];
        let (parameter, input) = product.split_at_mut(1);
        let second = plan
            .hvp_into(
                &Three,
                &[0.7],
                &a[..],
                &d[..],
                &[-0.3],
                &w[..],
                parameter,
                input,
                &mut workspace,
                cx,
            )
            .unwrap();
        (first, gradient, second.mode(), product)
    };
    let forward = run(DerivativeMode::Auto);
    let reverse = run(DerivativeMode::Reverse);
    assert_eq!(forward.0.mode(), BatchMode::Forward { tangents: 3 });
    assert_eq!(forward.2, BatchMode::Forward { tangents: 6 });
    assert_eq!(reverse.0.mode(), BatchMode::Reverse);
    assert_eq!(reverse.2, BatchMode::Reverse);
    assert_eq!(forward.0.value().to_bits(), reverse.0.value().to_bits());
    for (forward, reverse) in [(&forward.1, &reverse.1), (&forward.3, &reverse.3)] {
        for (f, r) in forward.iter().zip(reverse) {
            assert!((f - r).abs() <= 1e-13 * r.abs().max(1.0), "{f} against {r}");
        }
    }
    // The gradient in θ is the mean of a₀·a₁.
    let mean = a.chunks_exact(2).map(|row| row[0] * row[1]).sum::<f64>() / n as f64;
    assert!((forward.1[0] - mean).abs() <= 1e-13 * mean.abs().max(1.0));
}

/// `|θ₀ − aᵢ| + θ₁·dᵢ`: a kink where an item's input equals `θ₀`.
struct Distance;

impl BatchFunction for Distance {
    type Shared<R: Real> = [R; 2];
    fn prepare<R: Real>(&self, p: &[R]) -> Result<[R; 2], SciOxideError> {
        Ok([p[0], p[1]])
    }
    fn item<R: Real>(&self, p: &[R; 2], a: &[R], d: &[R::Base]) -> Result<R, SciOxideError> {
        Ok((p[0] - a[0]).abs() + p[1] * R::passive(d[0]))
    }
}

#[cfg(feature = "std")]
#[test]
fn the_kink_error_policy_reaches_every_worker_tape() {
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    let n = 21;
    let a: Vec<f64> = (0..n).map(|i| 0.25 * i as f64).collect();
    let d = indices(n);
    // a₁₇ = θ₀ in the fifth of six blocks: 17 items below θ₀ give +1, 3 above −1.
    let theta = [4.25, 0.5];
    for workers in [1, 3] {
        let k = NonZeroUsize::new(workers).unwrap();
        let executor = ThreadExecutor::new(k);
        let cx = ExecutionContext::new(&executor);
        for policy in [KinkPolicy::Midpoint, KinkPolicy::Error] {
            let options = BatchOptions::new()
                .with_block(NonZeroUsize::new(4).unwrap())
                .with_kink_policy(policy)
                .with_derivative_mode(DerivativeMode::Reverse)
                .with_execution_policy(ExecutionPolicy::Parallel(k));
            let plan = BatchPlan::<4>::new(n, 2, 1, 1, BatchReduction::Sum, options).unwrap();
            let mut workspace = plan.workspace(workers).unwrap();
            let (mut gradient, mut values) = ([UNSET; 2], vec![UNSET; n]);
            let outputs = BatchOutputs::new()
                .with_values(&mut values)
                .with_parameter_gradient(&mut gradient);
            let result = plan.evaluate_into(
                &Distance,
                &theta,
                &a[..],
                &d[..],
                outputs,
                &mut workspace,
                cx,
            );
            let (mut product, mut input_product) = ([UNSET; 2], vec![UNSET; n]);
            let hvp = plan.hvp_into(
                &Distance,
                &theta,
                &a[..],
                &d[..],
                &[1.0, 0.0],
                &vec![0.0; n][..],
                &mut product,
                &mut input_product,
                &mut workspace,
                cx,
            );
            if policy == KinkPolicy::Error {
                for error in [result.unwrap_err(), hvp.unwrap_err()] {
                    let domain = matches!(error.view(), SciOxideErrorView::DomainError { .. });
                    assert!(domain, "{workers} workers: {error}");
                }
                let mut outputs = values
                    .iter()
                    .chain(&gradient)
                    .chain(&product)
                    .chain(&input_product);
                assert!(outputs.all(|v| v.to_bits() == UNSET.to_bits()));
            } else {
                let estimate = result.unwrap();
                assert_eq!(estimate.kinks(), Some(1), "{workers} workers");
                assert_eq!(gradient, [17.0 - 3.0, d.iter().sum::<f64>()]);
                hvp.unwrap();
            }
        }
    }
}

#[test]
fn containers_and_closures_are_sources_with_the_bits_of_a_slice() {
    let n = 37;
    let (a, d) = (inputs(n), data(n));
    let plan = plan::<4>(n, BatchReduction::Mean, BatchOptions::new());
    let mut workspace = plan.workspace(1).unwrap();
    let cx = ExecutionContext::sequential();
    let (expected, buffers) =
        evaluate_sources(&plan, &Model, &a[..], &d[..], true, &mut workspace, cx);
    let expected = expected.unwrap().value().to_bits();
    let (from_vec, vec_buffers) = evaluate_sources(&plan, &Model, &a, &d, true, &mut workspace, cx);
    assert_eq!(from_vec.unwrap().value().to_bits(), expected);
    assert_eq!(vec_buffers.bits(), buffers.bits());
    let closure = BatchDataFn::new(|first, width, column, lanes: &mut [f64]| {
        for (k, lane) in lanes.iter_mut().enumerate() {
            *lane = a[(first + k) * width + column];
        }
        Ok(())
    });
    let (from_closure, closure_buffers) =
        evaluate_sources(&plan, &Model, &closure, &d, true, &mut workspace, cx);
    assert_eq!(from_closure.unwrap().value().to_bits(), expected);
    assert_eq!(closure_buffers.bits(), buffers.bits());
    // An array is a table, and a closure error aborts the batch unchanged.
    let mut lanes = [0.0; 2];
    [1.0, 2.0, 3.0, 4.0].fill(0, 2, 1, &mut lanes).unwrap();
    assert_eq!(lanes, [2.0, 4.0]);
    let error = SciOxideError::out_of_bounds(scioxide_core::quantity!("draws"), 9, 3);
    let failing = BatchDataFn::new(|_, _, _, _: &mut [f64]| Err(error.clone()));
    let (result, failed) =
        evaluate_sources(&plan, &Model, &a[..], &failing, true, &mut workspace, cx);
    assert_eq!(result.unwrap_err(), error);
    assert_eq!(failed.bits(), Buffers::unset(&plan).bits());
}

/// `θ₀·zᵢ + θ₁·zᵢ²` over the rows `zᵢ` of a table with one value.
struct Quadratic;

impl<S: RealScalar> BatchFunction<S> for Quadratic {
    type Shared<R: Real<S>> = [R; 2];
    fn prepare<R: Real<S>>(&self, p: &[R]) -> Result<[R; 2], SciOxideError> {
        Ok([p[0], p[1]])
    }
    fn item<R: Real<S>>(&self, p: &[R; 2], _: &[R], z: &[R::Base]) -> Result<R, SciOxideError> {
        let z = R::passive(z[0]);
        Ok(p[0] * z + p[1] * z * z)
    }
}

#[test]
fn one_shot_means_equal_the_plan() {
    let z: Vec<f64> = (0..29).map(|i| 0.1 * i as f64 - 1.0).collect();
    let theta = [0.7, -1.3];
    let plan =
        BatchPlan::<8>::new(z.len(), 2, 0, 1, BatchReduction::Mean, BatchOptions::new()).unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let none: &[f64] = &[];
    let mut expected = [0.0; 2];
    let reference = plan
        .evaluate_into(
            &Quadratic,
            &theta,
            none,
            &z[..],
            BatchOutputs::new().with_parameter_gradient(&mut expected),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let value = batch_mean(&Quadratic, &theta, &z, z.len()).unwrap();
    let (estimate, gradient) = batch_mean_and_grad(&Quadratic, &theta, &z, z.len()).unwrap();
    assert_eq!(value.value().to_bits(), reference.value().to_bits());
    assert_eq!(value.mode(), BatchMode::Value);
    assert_eq!(estimate.value().to_bits(), reference.value().to_bits());
    assert_eq!(estimate.mode(), reference.mode());
    assert_eq!(gradient.as_slice(), expected);
    // ∂/∂θ is the mean of (zᵢ, zᵢ²).
    let n = z.len() as f64;
    let mean = [
        z.iter().sum::<f64>() / n,
        z.iter().map(|z| z * z).sum::<f64>() / n,
    ];
    for (g, m) in gradient.as_slice().iter().zip(mean) {
        assert!((g - m).abs() < 1e-15);
    }
    // Two values per item: (zᵢ, unused).
    let pairs: Vec<f64> = z.iter().flat_map(|&z| [z, 9.0]).collect();
    let paired = batch_mean(&Quadratic, &theta, &pairs, z.len()).unwrap();
    assert_eq!(paired.value().to_bits(), reference.value().to_bits());
    // No items, and a table that does not split into rows.
    let empty = batch_mean(&Quadratic, &theta, &z, 0).unwrap_err();
    assert!(matches!(
        empty.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    let ragged = batch_mean_and_grad(&Quadratic, &theta, &z[..28], 3).unwrap_err();
    assert!(matches!(
        ragged.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    // No parameters: an empty gradient and value mode.
    struct Mean;
    impl BatchFunction for Mean {
        type Shared<R: Real> = ();
        fn prepare<R: Real>(&self, _: &[R]) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn item<R: Real>(&self, _: &(), _: &[R], z: &[R::Base]) -> Result<R, SciOxideError> {
            Ok(R::passive(z[0]))
        }
    }
    let (estimate, gradient) = batch_mean_and_grad(&Mean, &[], &z, z.len()).unwrap();
    assert_eq!(gradient.len(), 0);
    assert_eq!(estimate.mode(), BatchMode::Value);
    assert!((estimate.value() - z.iter().sum::<f64>() / n).abs() < 1e-15);
}

/// `θ₀θ₁·a₀`, panicking on the item whose `d₀` is 2 while `armed`.
struct Panicking {
    armed: AtomicBool,
}

impl BatchFunction for Panicking {
    type Shared<R: Real> = R;
    fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> {
        Ok(p[0] * p[1])
    }
    fn item<R: Real>(&self, s: &R, a: &[R], d: &[R::Base]) -> Result<R, SciOxideError> {
        let mut indices = (0..).map_while(|lane| d[0].lane(lane));
        if self.armed.load(Ordering::Relaxed) && indices.any(|index| index == 2.0) {
            panic!("item 2");
        }
        Ok(*s * a[0])
    }
}

/// The bits of the value and the parameter gradient of [`Panicking`] at [`THETA`]
/// over `aᵢ = 1 + i`.
fn panicking_run(
    plan: &BatchPlan<1>,
    function: &Panicking,
    workspace: &mut BatchWorkspace<f64, 1>,
    context: ExecutionContext<'_>,
) -> (Result<u64, SciOxideError>, [u64; 2]) {
    let n = plan.items();
    let a: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    let mut gradient = [UNSET; 2];
    let outputs = BatchOutputs::new().with_parameter_gradient(&mut gradient);
    let result = plan.evaluate_into(
        function,
        &THETA,
        &a[..],
        &indices(n)[..],
        outputs,
        workspace,
        context,
    );
    (
        result.map(|e| e.value().to_bits()),
        gradient.map(f64::to_bits),
    )
}

#[test]
fn a_panicking_item_leaves_the_workspace_usable() {
    let n = 16;
    let function = Panicking {
        armed: AtomicBool::new(false),
    };
    let reverse = options(4, Some(DerivativeMode::Reverse));
    let cx = ExecutionContext::sequential();
    let plan = BatchPlan::<1>::new(n, 2, 1, 1, BatchReduction::Sum, reverse).unwrap();
    let mut fresh = plan.workspace(1).unwrap();
    let expected = panicking_run(&plan, &function, &mut fresh, cx);
    // Σ aᵢ = 136: the value is θ₀θ₁·136 and the gradient (θ₁, θ₀)·136.
    let value = f64::from_bits(*expected.0.as_ref().unwrap());
    let gradient = expected.1.map(f64::from_bits);
    let exact = [THETA[0] * THETA[1], THETA[1], THETA[0]].map(|c| 136.0 * c);
    for (computed, exact) in [value, gradient[0], gradient[1]].into_iter().zip(exact) {
        assert!(
            (computed - exact).abs() <= 1e-13 * exact.abs(),
            "{computed} {exact}"
        );
    }
    // Sequential: the panic of item 2, after two groups were swept, reaches the caller.
    let mut workspace = plan.workspace(1).unwrap();
    assert_eq!(
        panicking_run(&plan, &function, &mut workspace, cx),
        expected
    );
    function.armed.store(true, Ordering::Relaxed);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        panicking_run(&plan, &function, &mut workspace, cx)
    }));
    assert!(panicked.is_err());
    function.armed.store(false, Ordering::Relaxed);
    assert_eq!(
        panicking_run(&plan, &function, &mut workspace, cx),
        expected
    );
    assert_eq!(
        panicking_run(&plan, &function, &mut workspace, cx),
        expected
    );
    // Two workers: the executor reports the panic.
    #[cfg(feature = "std")]
    {
        use scioxide_core::{ExecutionFailureReason, ExecutionPolicy, ThreadExecutor};
        let two = NonZeroUsize::new(2).unwrap();
        let executor = ThreadExecutor::new(two);
        let cx = ExecutionContext::new(&executor);
        let parallel = reverse.with_execution_policy(ExecutionPolicy::Parallel(two));
        let plan = BatchPlan::<1>::new(n, 2, 1, 1, BatchReduction::Sum, parallel).unwrap();
        let mut fresh = plan.workspace(2).unwrap();
        assert_eq!(panicking_run(&plan, &function, &mut fresh, cx), expected);
        let mut workspace = plan.workspace(2).unwrap();
        assert_eq!(
            panicking_run(&plan, &function, &mut workspace, cx),
            expected
        );
        function.armed.store(true, Ordering::Relaxed);
        let (error, gradient) = panicking_run(&plan, &function, &mut workspace, cx);
        function.armed.store(false, Ordering::Relaxed);
        let error = error.unwrap_err();
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::ExecutionFailure {
                    reason: ExecutionFailureReason::CallbackPanicked,
                    ..
                }
            ),
            "{error}"
        );
        assert_eq!(gradient, [UNSET.to_bits(); 2]);
        assert_eq!(
            panicking_run(&plan, &function, &mut workspace, cx),
            expected
        );
        assert_eq!(
            panicking_run(&plan, &function, &mut workspace, cx),
            expected
        );
    }
}

#[test]
fn the_evaluating_plan_sets_the_byte_limit_and_the_kink_policy() {
    let n = 64;
    let d = data(n);
    let cx = ExecutionContext::sequential();
    let reverse = options(16, Some(DerivativeMode::Reverse));
    let unlimited = plan::<4>(n, BatchReduction::Sum, reverse);
    let limited = plan::<4>(n, BatchReduction::Sum, reverse.with_byte_limit(Some(256)));
    let unset = Buffers::unset(&limited).bits();
    let mut fresh = unlimited.workspace(1).unwrap();
    let (expected, expected_buffers) = evaluate(&unlimited, &Model, &d, true, &mut fresh, cx);
    let (expected, expected_buffers) = (estimate_bits(&expected.unwrap()), expected_buffers.bits());
    let exceeds = |workspace: &mut BatchWorkspace<f64, 4>, case: &str| {
        let (result, buffers) = evaluate(&limited, &Model, &d, true, workspace, cx);
        let error = result.unwrap_err();
        assert_eq!(
            error.kind(),
            SciOxideErrorKind::ResourceLimit,
            "{case}: {error}"
        );
        assert_eq!(buffers.bits(), unset, "{case}");
    };
    let mut own = limited.workspace(1).unwrap();
    exceeds(&mut own, "own workspace");
    let mut foreign = unlimited.workspace(1).unwrap();
    exceeds(&mut foreign, "workspace of the unlimited plan");
    // The unlimited plan grows the limited plan's workspaces, fresh and after a failure.
    for (mut workspace, case) in [(limited.workspace(1).unwrap(), "fresh"), (own, "failed")] {
        let (estimate, buffers) = evaluate(&unlimited, &Model, &d, true, &mut workspace, cx);
        assert_eq!(estimate_bits(&estimate.unwrap()), expected, "{case}");
        assert_eq!(buffers.bits(), expected_buffers, "{case}");
    }

    // a₁₇ = θ₀: one kink, 17 items below θ₀ and 3 above.
    let n = 21;
    let a: Vec<f64> = (0..n).map(|i| 0.25 * i as f64).collect();
    let (d, w) = (indices(n), vec![0.0; n]);
    let theta = [4.25, 0.5];
    let kinked = |policy| {
        let options = reverse.with_kink_policy(policy);
        BatchPlan::<4>::new(n, 2, 1, 1, BatchReduction::Sum, options).unwrap()
    };
    let (midpoint, error) = (kinked(KinkPolicy::Midpoint), kinked(KinkPolicy::Error));
    let run = |plan: &BatchPlan<4>, workspace: &mut BatchWorkspace<f64, 4>| {
        let mut gradient = [UNSET; 2];
        let outputs = BatchOutputs::new().with_parameter_gradient(&mut gradient);
        let estimate =
            plan.evaluate_into(&Distance, &theta, &a[..], &d[..], outputs, workspace, cx);
        let (mut product, mut input_product) = ([UNSET; 2], vec![UNSET; n]);
        let hvp = plan.hvp_into(
            &Distance,
            &theta,
            &a[..],
            &d[..],
            &[1.0, 0.0],
            &w[..],
            &mut product,
            &mut input_product,
            workspace,
            cx,
        );
        let mut unchanged = gradient.iter().chain(&product).chain(&input_product);
        let unchanged = unchanged.all(|v| v.to_bits() == UNSET.to_bits());
        (estimate, hvp, gradient, unchanged)
    };
    let domain = |plan: &BatchPlan<4>, workspace: &mut BatchWorkspace<f64, 4>, case: &str| {
        let (estimate, hvp, _, unchanged) = run(plan, workspace);
        for error in [estimate.unwrap_err(), hvp.unwrap_err()] {
            assert_eq!(
                error.kind(),
                SciOxideErrorKind::DomainError,
                "{case}: {error}"
            );
        }
        assert!(unchanged, "{case}");
    };
    let midpoints = |workspace: &mut BatchWorkspace<f64, 4>, case: &str| {
        let (estimate, hvp, gradient, _) = run(&midpoint, workspace);
        assert_eq!(estimate.unwrap().kinks(), Some(1), "{case}");
        assert_eq!(hvp.unwrap().kinks(), Some(1), "{case}");
        assert_eq!(gradient, [17.0 - 3.0, d.iter().sum::<f64>()], "{case}");
    };
    // The error plan with a midpoint plan's workspace, cold and warm, and back.
    let mut workspace = midpoint.workspace(1).unwrap();
    domain(&error, &mut workspace, "cold midpoint workspace");
    midpoints(&mut workspace, "after the error");
    domain(&error, &mut workspace, "warm midpoint workspace");
    let mut workspace = error.workspace(1).unwrap();
    midpoints(&mut workspace, "cold error workspace");
    domain(&error, &mut workspace, "warm error workspace");
}

/// `θ₀·a₀`; `θ₁` and the data are unused.
struct Linear;

impl BatchFunction for Linear {
    type Shared<R: Real> = R;
    fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> {
        Ok(p[0])
    }
    fn item<R: Real>(&self, theta: &R, a: &[R], _: &[R::Base]) -> Result<R, SciOxideError> {
        Ok(*theta * a[0])
    }
}

fn is_non_finite_output(error: &SciOxideError) -> bool {
    matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Output,
            ..
        }
    )
}

#[test]
fn the_error_policy_covers_every_output() {
    let cx = ExecutionContext::sequential();
    let none: &[f64] = &[];
    for policy in [NonFinitePolicy::Propagate, NonFinitePolicy::Error] {
        let checked = policy == NonFinitePolicy::Error;
        let options = |mode| options(2, mode).with_non_finite_policy(policy);
        // Two derivatives u₀·aᵢ = 10³⁰⁸ whose sum overflows.
        let plan = BatchPlan::<2>::new(2, 2, 1, 0, BatchReduction::Sum, options(None)).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let (a, w) = ([1.0, 1.0], [0.0, 0.0]);
        let mut derivatives = [UNSET; 2];
        let result = plan.jvp_into(
            &Linear,
            &[1.0, 0.0],
            &a[..],
            none,
            &[1e308, 0.0],
            &w[..],
            &mut derivatives,
            &mut workspace,
            cx,
        );
        if checked {
            assert!(is_non_finite_output(&result.unwrap_err()));
            assert_eq!(derivatives, [UNSET; 2]);
        } else {
            assert_eq!(result.unwrap().derivative(), Some(f64::INFINITY));
            assert_eq!(derivatives, [1e308; 2]);
        }
        // Values ±1.3·10²⁰⁰ with mean 0 whose standard error overflows, then a sum
        // whose gradient error overflows; every other output is finite.
        let cases = [
            (BatchReduction::Mean, [1e200, -1e200, 1e200, -1e200]),
            (BatchReduction::Sum, [1e200, 1e200, -1e200, -1e200]),
        ];
        for (reduction, a) in cases {
            for mode in MODES {
                let gradient = mode.is_some();
                if reduction == BatchReduction::Sum && !gradient {
                    continue;
                }
                let case = format!("{policy:?} {reduction:?} {mode:?}");
                let plan = BatchPlan::<2>::new(4, 2, 1, 0, reduction, options(mode)).unwrap();
                assert_eq!(plan.blocks(), 2);
                let mut workspace = plan.workspace(1).unwrap();
                let (result, buffers) =
                    evaluate_sources(&plan, &Linear, &a[..], none, gradient, &mut workspace, cx);
                if checked {
                    assert!(is_non_finite_output(&result.unwrap_err()), "{case}");
                    assert_eq!(buffers.bits(), Buffers::unset(&plan).bits(), "{case}");
                    continue;
                }
                let estimate = result.unwrap();
                assert_eq!(estimate.value(), 0.0, "{case}");
                if reduction == BatchReduction::Mean {
                    let error = estimate.standard_error().unwrap();
                    assert!(!error.is_finite(), "{case}");
                } else {
                    assert!(!buffers.error[0].is_finite(), "{case}");
                    assert!(buffers.gradient.iter().all(|g| g.is_finite()), "{case}");
                }
            }
        }
    }
}

#[test]
fn sizes_that_overflow_are_errors() {
    let unit = BatchOptions::new().with_block(NonZeroUsize::new(1).unwrap());
    // The dual seeds `parameters + inputs`, the block gradients `blocks·parameters`
    // and their sum with the reduced gradient, and the bytes of the block results.
    let shapes = [
        (1, usize::MAX, 1),
        (4, usize::MAX / 4, 0),
        (8, usize::MAX / 4, 0),
        (usize::MAX / 8, 0, 0),
    ];
    for (items, parameters, inputs) in shapes {
        let case = format!("{items} items, {parameters} parameters, {inputs} inputs");
        let plan = BatchPlan::<1>::new(items, parameters, inputs, 0, BatchReduction::Sum, unit);
        let plan = plan.unwrap();
        let error = plan.workspace_requirements(1).unwrap_err();
        assert_eq!(
            error.kind(),
            SciOxideErrorKind::SizeOverflow,
            "{case}: {error}"
        );
        let error = plan.workspace::<f64>(1).unwrap_err();
        assert_eq!(
            error.kind(),
            SciOxideErrorKind::SizeOverflow,
            "{case}: {error}"
        );
    }
    // Items × inputs overflows; the generated inputs need no memory.
    let defaults = BatchOptions::new();
    let plan = BatchPlan::<8>::new(usize::MAX / 2, 1, 4, 0, BatchReduction::Sum, defaults);
    let plan = plan.unwrap();
    let small = BatchPlan::<8>::new(1, 1, 4, 0, BatchReduction::Sum, defaults).unwrap();
    let mut workspace = small.workspace(1).unwrap();
    let generated = BatchDataFn::new(|first, _, _, lanes: &mut [f64]| {
        for (k, lane) in lanes.iter_mut().enumerate() {
            *lane = (first + k) as f64;
        }
        Ok(())
    });
    let none: &[f64] = &[];
    let cx = ExecutionContext::sequential();
    let (mut gradient, mut input_gradient) = ([UNSET], [UNSET; 4]);
    for request in 0..3 {
        let outputs = match request {
            0 => BatchOutputs::new(),
            1 => BatchOutputs::new().with_parameter_gradient(&mut gradient),
            _ => BatchOutputs::new().with_input_gradient(&mut input_gradient),
        };
        let result = plan.evaluate_into(
            &Linear,
            &[1.0],
            &generated,
            none,
            outputs,
            &mut workspace,
            cx,
        );
        let error = result.unwrap_err();
        assert_eq!(
            error.kind(),
            SciOxideErrorKind::SizeOverflow,
            "request {request}: {error}"
        );
    }
    let error = plan
        .hvp_into(
            &Linear,
            &[1.0],
            &generated,
            none,
            &[1.0],
            &generated,
            &mut gradient,
            &mut input_gradient,
            &mut workspace,
            cx,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::SizeOverflow, "{error}");
    let mut derivatives = [UNSET; 4];
    let error = plan
        .jvp_into(
            &Linear,
            &[1.0],
            &generated,
            none,
            &[1.0],
            &generated,
            &mut derivatives,
            &mut workspace,
            cx,
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        SciOxideErrorKind::DimensionMismatch,
        "{error}"
    );
    let unchanged = gradient.iter().chain(&input_gradient).chain(&derivatives);
    assert!(unchanged.copied().all(|v| v.to_bits() == UNSET.to_bits()));
    // The workspace still serves its own plan.
    let mut value = [UNSET];
    let outputs = BatchOutputs::new().with_values(&mut value);
    let estimate = small.evaluate_into(
        &Linear,
        &[2.0],
        &generated,
        none,
        outputs,
        &mut workspace,
        cx,
    );
    assert_eq!(estimate.unwrap().value(), 0.0);
    assert_eq!(value, [0.0]);
}

/// `θ·a₀`, counting the calls of `item`.
struct Counted {
    calls: AtomicUsize,
}

impl BatchFunction for Counted {
    type Shared<R: Real> = R;
    fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> {
        Ok(p[0])
    }
    fn item<R: Real>(&self, theta: &R, a: &[R], _: &[R::Base]) -> Result<R, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(*theta * a[0])
    }
}

#[test]
fn gradient_errors_over_one_block_fail_before_any_item() {
    let n = 256;
    let a = vec![1.0; n];
    let none: &[f64] = &[];
    let cx = ExecutionContext::sequential();
    for mode in [DerivativeMode::Auto, DerivativeMode::Reverse] {
        let options = BatchOptions::new().with_derivative_mode(mode);
        let plan = BatchPlan::<8>::new(n, 1, 1, 0, BatchReduction::Mean, options).unwrap();
        assert_eq!(plan.blocks(), 1);
        let mut workspace = plan.workspace(1).unwrap();
        let function = Counted {
            calls: AtomicUsize::new(0),
        };
        let (mut gradient, mut error) = ([UNSET], [UNSET]);
        let outputs = BatchOutputs::new()
            .with_parameter_gradient(&mut gradient)
            .with_parameter_gradient_error(&mut error);
        let failure = plan
            .evaluate_into(&function, &[1.0], &a[..], none, outputs, &mut workspace, cx)
            .unwrap_err();
        assert!(
            matches!(failure.view(), SciOxideErrorView::InvalidInput { .. }),
            "{mode:?}: {failure}"
        );
        assert_eq!(function.calls.load(Ordering::Relaxed), 0, "{mode:?}");
        assert_eq!((gradient, error), ([UNSET], [UNSET]), "{mode:?}");
    }
    // Two blocks admit the errors and call `item` once per group of eight.
    let options = BatchOptions::new().with_block(NonZeroUsize::new(128).unwrap());
    let plan = BatchPlan::<8>::new(n, 1, 1, 0, BatchReduction::Mean, options).unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let function = Counted {
        calls: AtomicUsize::new(0),
    };
    let mut error = [UNSET];
    let outputs = BatchOutputs::new().with_parameter_gradient_error(&mut error);
    plan.evaluate_into(&function, &[1.0], &a[..], none, outputs, &mut workspace, cx)
        .unwrap();
    assert_eq!(function.calls.load(Ordering::Relaxed), n / 8);
    assert_eq!(error, [0.0]);
}

/// `θ₀·d₀`, failing in `prepare` or else in `item`.
struct Rejecting {
    in_prepare: bool,
}

impl Rejecting {
    fn error(&self) -> SciOxideError {
        let draws = scioxide_core::quantity!("draws");
        SciOxideError::out_of_bounds(draws, usize::from(self.in_prepare), 1)
    }
}

impl BatchFunction for Rejecting {
    type Shared<R: Real> = R;
    fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> {
        if self.in_prepare {
            return Err(self.error());
        }
        Ok(p[0])
    }
    fn item<R: Real>(&self, _: &R, _: &[R], _: &[R::Base]) -> Result<R, SciOxideError> {
        Err(self.error())
    }
}

#[test]
fn one_shots_have_the_bits_of_the_plan_for_every_item_count() {
    let theta = [0.7, -1.3];
    let none: &[f64] = &[];
    let cx = ExecutionContext::sequential_native();
    for n in [1, 7, 8, 9, 255, 256, 257] {
        let z: Vec<f64> = (0..n).map(|i| (0.37 * i as f64).sin()).collect();
        let plan = BatchPlan::<8>::new(n, 2, 0, 1, BatchReduction::Mean, BatchOptions::new());
        let plan = plan.unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let value = plan
            .evaluate_into(
                &Quadratic,
                &theta,
                none,
                &z[..],
                BatchOutputs::new(),
                &mut workspace,
                cx,
            )
            .unwrap();
        let mut expected = [UNSET; 2];
        let outputs = BatchOutputs::new().with_parameter_gradient(&mut expected);
        let reference = plan
            .evaluate_into(
                &Quadratic,
                &theta,
                none,
                &z[..],
                outputs,
                &mut workspace,
                cx,
            )
            .unwrap();
        let one_shot = batch_mean(&Quadratic, &theta, &z, n).unwrap();
        assert_eq!(estimate_bits(&one_shot), estimate_bits(&value), "{n} items");
        let (estimate, gradient) = batch_mean_and_grad(&Quadratic, &theta, &z, n).unwrap();
        assert_eq!(
            estimate_bits(&estimate),
            estimate_bits(&reference),
            "{n} items"
        );
        let gradient = gradient.as_slice().iter().map(|g| g.to_bits());
        assert!(gradient.eq(expected.map(f64::to_bits)), "{n} items");
        assert_eq!((one_shot.items(), one_shot.blocks()), (n, n.div_ceil(256)));
        assert_eq!(one_shot.mode(), BatchMode::Value);
        assert_eq!(estimate.mode(), BatchMode::Forward { tangents: 2 });
        assert_eq!(one_shot.standard_error().is_none(), n == 1, "{n} items");
    }
    // No data: width 0 for any item count; √θ has the gradient 1/(2√θ).
    let plan = BatchPlan::<8>::new(3, 1, 0, 0, BatchReduction::Mean, BatchOptions::new()).unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let mut expected = [UNSET];
    let outputs = BatchOutputs::new().with_parameter_gradient(&mut expected);
    let reference = plan
        .evaluate_into(
            &RootParameter,
            &[4.0],
            none,
            none,
            outputs,
            &mut workspace,
            cx,
        )
        .unwrap();
    let (estimate, gradient) = batch_mean_and_grad(&RootParameter, &[4.0], &[], 3).unwrap();
    assert_eq!(estimate_bits(&estimate), estimate_bits(&reference));
    assert_eq!(
        (estimate.value(), estimate.standard_error()),
        (2.0, Some(0.0))
    );
    assert_eq!((gradient.as_slice(), expected), (&[0.25][..], [0.25]));
    assert_eq!(
        batch_mean(&RootParameter, &[4.0], &[], 3).unwrap().value(),
        2.0
    );
    // Errors of `prepare` and `item` come back unchanged.
    let z: Vec<f64> = (0..9).map(f64::from).collect();
    for in_prepare in [true, false] {
        let function = Rejecting { in_prepare };
        let error = batch_mean(&function, &[1.0], &z, 9).unwrap_err();
        assert_eq!(error, function.error(), "in prepare: {in_prepare}");
        let error = batch_mean_and_grad(&function, &[1.0], &z, 9).unwrap_err();
        assert_eq!(error, function.error(), "in prepare: {in_prepare}");
    }
    // f32: the bits of the plan on f32, and the f64 mean to f32 accuracy.
    let n = 37;
    let z: Vec<f32> = (0..n).map(|i| (0.37 * i as f32).sin()).collect();
    let theta = [0.7_f32, -1.3];
    let plan = BatchPlan::<8>::new(n, 2, 0, 1, BatchReduction::Mean, BatchOptions::new()).unwrap();
    let mut workspace = plan.workspace::<f32>(1).unwrap();
    let mut expected = [0.0_f32; 2];
    let outputs = BatchOutputs::new().with_parameter_gradient(&mut expected);
    let no_inputs: &[f32] = &[];
    let reference = plan
        .evaluate_into(
            &Quadratic,
            &theta,
            no_inputs,
            &z[..],
            outputs,
            &mut workspace,
            cx,
        )
        .unwrap();
    let (estimate, gradient) = batch_mean_and_grad(&Quadratic, &theta, &z, n).unwrap();
    assert_eq!(estimate.value().to_bits(), reference.value().to_bits());
    assert_eq!(
        estimate.standard_error().map(f32::to_bits),
        reference.standard_error().map(f32::to_bits)
    );
    assert!(
        gradient
            .as_slice()
            .iter()
            .map(|g| g.to_bits())
            .eq(expected.map(f32::to_bits))
    );
    let value = batch_mean(&Quadratic, &theta, &z, n).unwrap();
    assert_eq!(value.value().to_bits(), reference.value().to_bits());
    let wide: Vec<f64> = z.iter().map(|&z| f64::from(z)).collect();
    let mean = |f: &dyn Fn(f64) -> f64| wide.iter().map(|&z| f(z)).sum::<f64>() / n as f64;
    let [t0, t1] = theta.map(f64::from);
    let exact = [
        mean(&|z| t0 * z + t1 * z * z),
        mean(&|z| z),
        mean(&|z| z * z),
    ];
    let computed = [
        estimate.value(),
        gradient.as_slice()[0],
        gradient.as_slice()[1],
    ];
    for (computed, exact) in computed.into_iter().zip(exact) {
        let computed = f64::from(computed);
        assert!(
            (computed - exact).abs() <= 1e-6 * exact.abs().max(1.0),
            "{computed} {exact}"
        );
    }
}

/// A row-major table read through a closure; one function gives a source and its
/// direction the same type.
#[allow(clippy::type_complexity)]
fn closure_table(
    table: &[f64],
) -> BatchDataFn<
    f64,
    impl Fn(usize, usize, usize, &mut [f64]) -> Result<(), SciOxideError> + Sync + '_,
> {
    BatchDataFn::new(move |first, width, column, lanes: &mut [f64]| {
        for (k, lane) in lanes.iter_mut().enumerate() {
            *lane = table[(first + k) * width + column];
        }
        Ok(())
    })
}

/// The estimates and the bits of every output of [`BatchPlan::jvp_into`] and
/// [`BatchPlan::hvp_into`] for [`Model`] at [`THETA`] in the direction `(u, w)`.
fn products<A, D>(
    plan: &BatchPlan<4>,
    a: &A,
    w: &A,
    d: &D,
    workspace: &mut BatchWorkspace<f64, 4>,
) -> (EstimateBits, EstimateBits, Vec<u64>)
where
    A: BatchData + ?Sized,
    D: BatchData + ?Sized,
{
    let n = plan.items();
    let u = [0.3, -0.8];
    let cx = ExecutionContext::sequential();
    let mut derivatives = vec![UNSET; n];
    let jvp = plan
        .jvp_into(&Model, &THETA, a, d, &u, w, &mut derivatives, workspace, cx)
        .unwrap();
    let (mut parameter, mut input) = ([UNSET; 2], vec![UNSET; 2 * n]);
    let hvp = plan
        .hvp_into(
            &Model,
            &THETA,
            a,
            d,
            &u,
            w,
            &mut parameter,
            &mut input,
            workspace,
            cx,
        )
        .unwrap();
    let outputs = derivatives.into_iter().chain(parameter).chain(input);
    (
        estimate_bits(&jvp),
        estimate_bits(&hvp),
        outputs.map(f64::to_bits).collect(),
    )
}

#[test]
fn every_source_gives_the_bits_of_a_slice_in_directional_products() {
    const N: usize = 21;
    let (a, d) = (inputs(N), data(N));
    let w: Vec<f64> = (0..2 * N).map(|k| 0.05 * k as f64 - 0.4).collect();
    let a_array: [f64; 2 * N] = a[..].try_into().unwrap();
    let w_array: [f64; 2 * N] = w[..].try_into().unwrap();
    let d_array: [f64; N] = d[..].try_into().unwrap();
    for mode in [DerivativeMode::Auto, DerivativeMode::Reverse] {
        let plan = plan::<4>(N, BatchReduction::Mean, options(8, Some(mode)));
        let mut workspace = plan.workspace(1).unwrap();
        let expected = products(&plan, &a[..], &w[..], &d[..], &mut workspace);
        let from_vec = products(&plan, &a, &w, &d, &mut workspace);
        assert_eq!(from_vec, expected, "{mode:?} Vec");
        let from_array = products(&plan, &a_array, &w_array, &d_array, &mut workspace);
        assert_eq!(from_array, expected, "{mode:?} array");
        let (a, w, d) = (closure_table(&a), closure_table(&w), closure_table(&d));
        let from_closure = products(&plan, &a, &w, &d, &mut workspace);
        assert_eq!(from_closure, expected, "{mode:?} closure");
    }
}

#[test]
fn an_empty_array_is_a_source_of_no_values() {
    let n = 6;
    let plan = BatchPlan::<4>::new(n, 1, 0, 0, BatchReduction::Mean, BatchOptions::new()).unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let cx = ExecutionContext::sequential();
    let none: &[f64] = &[];
    let mut expected = [UNSET];
    let outputs = BatchOutputs::new().with_parameter_gradient(&mut expected);
    let reference = plan
        .evaluate_into(
            &RootParameter,
            &[4.0],
            none,
            none,
            outputs,
            &mut workspace,
            cx,
        )
        .unwrap();
    let mut gradient = [UNSET];
    let outputs = BatchOutputs::new().with_parameter_gradient(&mut gradient);
    let estimate = plan
        .evaluate_into(
            &RootParameter,
            &[4.0],
            &[],
            &[],
            outputs,
            &mut workspace,
            cx,
        )
        .unwrap();
    assert_eq!(estimate_bits(&estimate), estimate_bits(&reference));
    assert_eq!((gradient, expected), ([0.25], [0.25]));
    let mut derivatives = [UNSET; 6];
    let jvp = plan
        .jvp_into(
            &RootParameter,
            &[4.0],
            &[],
            &[],
            &[1.0],
            &[],
            &mut derivatives,
            &mut workspace,
            cx,
        )
        .unwrap();
    assert_eq!(jvp.derivative(), Some(0.25));
    assert_eq!(derivatives, [0.25; 6]);
}

#[test]
fn a_closure_source_allocates_nothing_when_warm() {
    let n = 70;
    let (a, d) = (inputs(n), data(n));
    let (a, d) = (closure_table(&a), closure_table(&d));
    let cx = ExecutionContext::sequential();
    for mode in [DerivativeMode::Auto, DerivativeMode::Reverse] {
        let plan = plan::<8>(n, BatchReduction::Mean, options(16, Some(mode)));
        let mut workspace = plan.workspace(1).unwrap();
        let mut buffers = Buffers::unset(&plan);
        for gradient in [true, false] {
            let mut run = || {
                evaluate_buffers(
                    &plan,
                    &Model,
                    &a,
                    &d,
                    gradient,
                    &mut buffers,
                    &mut workspace,
                    cx,
                )
                .unwrap()
            };
            run();
            (run)();
        }
    }
}

#[test]
fn empty_batches_and_final_callbacks_observe_cancellation() {
    struct Stop<'a>(&'a CancellationToken);
    impl BatchFunction for Stop<'_> {
        type Shared<R: Real> = ();
        fn prepare<R: Real>(&self, _: &[R]) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn item<R: Real>(&self, _: &(), _: &[R], _: &[R::Base]) -> Result<R, SciOxideError> {
            self.0.cancel();
            Ok(R::one())
        }
    }
    for n in [0, 1] {
        let token = CancellationToken::new();
        if n == 0 {
            token.cancel();
        }
        let plan =
            BatchPlan::<1>::new(n, 0, 0, 0, BatchReduction::Sum, BatchOptions::new()).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut values = vec![19.0; n];
        let result = plan.evaluate_into(
            &Stop(&token),
            &[],
            &[],
            &[],
            BatchOutputs::new().with_values(&mut values),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token),
        );
        assert!(matches!(
            result.unwrap_err().view(),
            SciOxideErrorView::ExecutionStopped { .. }
        ));
        assert!(values.iter().all(|&v| v == 19.0));
    }
}

#[test]
fn overflowing_active_count_is_rejected_before_batch_dispatch() {
    let small = BatchPlan::<1>::new(0, 0, 0, 0, BatchReduction::Sum, BatchOptions::new()).unwrap();
    let mut workspace = small.workspace(1).unwrap();
    let plan = BatchPlan::<1>::new(
        0,
        1,
        usize::MAX,
        0,
        BatchReduction::Sum,
        BatchOptions::new(),
    )
    .unwrap();
    let mut gradient = [19.0];
    let error = plan
        .evaluate_into(
            &Model,
            &[0.0],
            &[],
            &[],
            BatchOutputs::new().with_parameter_gradient(&mut gradient),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::SizeOverflow { .. }
    ));
    assert_eq!(gradient, [19.0]);
}
