//! Core API inventory. Every import is compiled as an external consumer.
#![allow(unused_imports)]

use scioxide_core::backend::scaled::{
    factor_common_scale, scaled_magnitude_le_tolerance, scaled_tolerance_threshold,
};
use scioxide_core::backend::{
    ComplexSlice, ComplexSliceMut, KernelBackend, KernelDispatch, PartitionSlot,
    PartitionSolveBatch, PartitionSolvePermit, PartitionSolvePermits, PartitionedSolve,
    RealBackend, Scalar, ScalarKind, ScalarSlice, ScalarSliceMut, ScalarValue, ShortReductionRange,
    TWO_OVER_PI_LIMBS, absolute_difference_le, checked_add, checked_final_rounding, checked_mul,
    checked_next_power_of_two, checked_product, checked_strided_span, compensated_indexed,
    component_count, components, components_mut, four_lane_sums, gcd, matrix_layout_is_injective,
    opposite_signs, product_sum_scaled, reduce_radians, scalar_from_u128_rounded, try_copy_slice,
    try_filled, try_reserve, try_with_capacity, validate_layout_backing_len, validate_then_commit,
    with_partitioned_solves,
};
use scioxide_core::{
    AbsoluteTolerance, AcceptedStateMetadata, AccountingIntegrity, Accuracy, AdjointOperator,
    Admission, AffineValue, AffineValueMut, AlgebraicScalar, AlgorithmId, Argument, AttemptKind,
    AttemptOutcome, AttemptRecord, AxisLayout, BatchLayout, BlockOperator, BlockVectorView,
    BlockVectorViewMut, Bounds, Budget, BudgetKind, BudgetLimit, BudgetReport, Cancellation,
    CancellationToken, Capability, CapabilityError, Capacity, Clock, ClockBinding, ClosureOperator,
    ClosurePreconditioner, ColoringKind, ColoringSchedule, CombinationEnumerator, CompensatedSum,
    Complex, Complex32, Complex64, ConvergenceDiagnostics, ConvergenceReason, ConversionType,
    Deadline, DifferentiationFrame, DifferentiationLocation, DifferentiationSite, Dimensions,
    DomainReason, EntropySource, ErrorTolerance, EstimateReliability, EvaluationCounts,
    EvaluationRole, ExecutionContext, ExecutionFailureReason, ExecutionPlan, ExecutionPolicy,
    ExecutionStopReason, Executor, ExtendedInterval, FeasibilityTolerance, FloatDecomposition,
    GradientTolerance, IdentifierError, IntegralityTolerance, Interval, InvalidInputReason,
    Invocation, JacobianOperator, LinearOperator, LogValue, MonotonicInstant, MultisetEnumerator,
    NdView, NdViewMut, NestedLevel, NonFiniteOrigin, NormSelection, NumericScalar, NumericalIssue,
    PartialFailure, PartialResultTrust, PartitionFn, PartitionId, Permutation,
    PermutationEnumerator, PermutationView, PlanLimits, PortableProcessor, Preconditioner,
    ProblemToken, ProcessorCapabilities, ProcessorFeature, ProductSum, ProjectiveValue,
    PromotingCompensatedSum, PromotingProduct, Quantity, RangeAccumulator, RankThreshold,
    RealScalar, RelativeTolerance, ReliableEstimate, ResidualOperator, ResidualReport,
    ResidualScaling, ResidualScalingRule, ResidualTolerance, ScalarConversionReason, Scaled,
    ScaledAccumulator, ScaledNorm, ScaledValue, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView, SequentialExecutor, Shape, SolveBudgets, SolveCheckpointMetadata,
    SolveContext, SolveCounts, SolveEpoch, SolveIdentity, SolveScope, SolveState, StepTolerance,
    StridedLayout, StructuralPattern, StructuralPatternView, Tolerance, ValueAndDerivative,
    WorkPartition, WorkPartitions, affine_combination_into, algorithm_id, argument, budget,
    check_value_shape, checked_affine, checked_binomial, checked_difference,
    checked_difference_of_products, checked_double_factorial, checked_factorial,
    checked_float_product, checked_float_product_factors, checked_float_quotient, checked_midpoint,
    checked_mul_add, checked_multifactorial, checked_multiset_count, checked_next_after,
    checked_permutation_count, checked_product_quotient, checked_quadratic_roots,
    checked_rising_factorial, checked_scalbn, checked_scaled_norm, checked_stirling_first_kind,
    checked_stirling_second_kind, checked_sum, checked_sum_of_products, combination_rank,
    combination_unrank_into, cos_degrees, cos_pi, cosm1, exprel, log_sigmoid, log_softmax,
    log_softmax_axis, log_softmax_axis_into, log_softmax_into, logaddexp, logit, logsumexp,
    logsumexp_axis, logsumexp_axis_into, logsumexp_value, multiset_rank, multiset_unrank_into,
    permutation_rank, permutation_unrank_into, permutation_validation_scratch_bytes, powm1,
    quantity, sigmoid, sin_cos_degrees, sin_cos_pi, sin_degrees, sin_pi, sinc, sincpi, softmax,
    softmax_axis, softmax_axis_into, softmax_into, softplus, tan_degrees, tan_pi, transform_into,
    validate_value, weighted_logsumexp, weighted_logsumexp_axis, weighted_logsumexp_axis_into,
    write_value, xlog1py, xlogy,
};
#[cfg(feature = "std")]
use scioxide_core::{StdClock, StdProcessor, ThreadExecutor};

// Compile canonical conversions as an external consumer.
const _: fn(Vec<usize>) -> Dimensions = Dimensions::from;
const _: fn(Box<[usize]>) -> Dimensions = Dimensions::from;
const _: fn(usize) -> Result<Budget, SciOxideError> = Budget::try_from;
const _: fn(core::num::NonZeroUsize) -> Budget = Budget::from;
const _: fn(f32) -> Result<f64, SciOxideError> = <f32 as RealScalar>::try_to_f64_exact;
const _: fn(f64) -> Result<f64, SciOxideError> = <f64 as RealScalar>::try_to_f64_exact;
const _: fn(f64, f64) -> Result<Interval<f64>, SciOxideError> = Interval::new;
const _: fn(f64) -> Result<f64, SciOxideError> = softplus;
const _: fn(f64) -> Result<f64, SciOxideError> = scioxide_core::log1mexp;
const _: fn(f64, f64) -> (f64, f64) = scioxide_core::backend::two_sum;
const _: fn(f64, f64) -> (f64, f64) = scioxide_core::backend::two_product;
const _: fn([f64; 4]) -> [f64; 4] = scioxide_core::backend::lanes_exp::<4>;
const _: fn([f64; 4]) -> [f64; 4] = scioxide_core::backend::lanes_exp_m1::<4>;
const _: fn([f64; 4]) -> [f64; 4] = scioxide_core::backend::lanes_ln::<4>;
const _: fn([f64; 4]) -> [f64; 4] = scioxide_core::backend::lanes_ln_1p::<4>;
const _: fn([f64; 4]) -> [f64; 4] = scioxide_core::backend::lanes_sin::<4>;
const _: fn([f64; 4]) -> [f64; 4] = scioxide_core::backend::lanes_cos::<4>;
const _: fn([f64; 4]) -> [f64; 4] = scioxide_core::backend::lanes_tanh::<4>;
const _: fn([f64; 4], [f64; 4]) -> [f64; 4] = scioxide_core::backend::lanes_powf::<4>;
const _: fn(&mut CompensatedSum<f64>, CompensatedSum<f64>) -> Result<(), SciOxideError> =
    CompensatedSum::merge;
type CompensatedBlockResult = Result<[CompensatedSum<f64>; 1], SciOxideError>;
const _: fn([[f64; 1]; 3]) -> CompensatedBlockResult = scioxide_core::compensated_reduce;
const _: fn(&[f64]) -> Result<Vec<f64>, SciOxideError> = softmax;
const _: fn(f64, f64, f64) -> Result<[Complex64; 2], SciOxideError> = checked_quadratic_roots;
const _: fn(usize, usize) -> Result<isize, SciOxideError> = checked_stirling_first_kind;
const _: fn(usize, usize) -> Result<usize, SciOxideError> = checked_stirling_second_kind;
const _: fn(&'static dyn Executor) -> ExecutionContext<'static> = ExecutionContext::native;
const _: Budget = budget!(256);

// Row-major layouts and copyable views.
const fn assert_copy<T: Copy>() {}
const _: () = assert_copy::<NdView<'static, 'static, f64>>();
const _: () = assert_copy::<StridedLayout<'static>>();
const _: () = assert!(StridedLayout::ROW_MAJOR_MAX_RANK == 8);
const _: fn(&'static [usize]) -> Result<StridedLayout<'static>, SciOxideError> =
    StridedLayout::row_major;
const _: for<'s> fn(&'s StridedLayout<'static>) -> &'s [usize] = StridedLayout::strides;
type View = NdView<'static, 'static, f64>;
type ViewMut<T> = NdViewMut<'static, 'static, T>;
const _: fn(&'static [f64], &'static [usize]) -> Result<View, SciOxideError> = NdView::row_major;
type RowMajorMut = fn(&'static mut [f64], &'static [usize]) -> Result<ViewMut<f64>, SciOxideError>;
const _: RowMajorMut = NdViewMut::row_major;

// Axis reductions: owned forms and `_into` forms with a policy and a context.
type Log = LogValue<f64, f64>;
type Cx = ExecutionContext<'static>;
type AxisInto<T> =
    fn(View, usize, ExecutionPolicy, ViewMut<T>, &mut [T], Cx) -> Result<(), SciOxideError>;
type WeightedInto = fn(
    View,
    View,
    usize,
    ExecutionPolicy,
    ViewMut<Log>,
    &mut [Log],
    Cx,
) -> Result<(), SciOxideError>;
const _: fn(View, usize) -> Result<Vec<f64>, SciOxideError> = softmax_axis;
const _: fn(View, usize) -> Result<Vec<f64>, SciOxideError> = log_softmax_axis;
const _: fn(View, usize) -> Result<Vec<f64>, SciOxideError> = logsumexp_axis;
const _: fn(View, View, usize) -> Result<Vec<Log>, SciOxideError> = weighted_logsumexp_axis;
const _: AxisInto<f64> = softmax_axis_into;
const _: AxisInto<f64> = log_softmax_axis_into;
const _: AxisInto<f64> = logsumexp_axis_into;
const _: WeightedInto = weighted_logsumexp_axis_into;
const _: fn(KernelBackend, &dyn ProcessorCapabilities) -> Result<KernelDispatch, SciOxideError> =
    KernelDispatch::for_backend;

// Extended-range signatures require only numerical inputs.
mod extended_range {
    use super::*;
    type R = Scaled<f64>;
    type V = ScaledValue<Complex64>;
    type A = ScaledAccumulator<f64>;
    const _: fn(A, i64) -> bool = A::absorbs;
    const _: fn(&mut A, A) -> Result<(), SciOxideError> = A::merge;
    type L = LogValue<Complex64, f64>;
    type P = ProjectiveValue<Complex64>;

    const _: R = R::ZERO;
    const _: R = R::ONE;
    const _: R = R::HALF;
    const _: V = V::ZERO;
    const _: V = V::ONE;
    const _: V = V::HALF;
    const _: A = A::ZERO;
    const _: L = L::ZERO;
    const _: PromotingCompensatedSum<f64> = PromotingCompensatedSum::ZERO;
    const _: PromotingProduct<f64> = PromotingProduct::ONE;
    const _: fn(&mut PromotingCompensatedSum<f64>, R) -> Result<(), SciOxideError> =
        PromotingCompensatedSum::add_scaled;
    const _: fn(PromotingProduct<f64>, f64) -> Result<PromotingProduct<f64>, SciOxideError> =
        PromotingProduct::multiplied;

    const _: fn(f64) -> Result<R, SciOxideError> = R::from_value;
    const _: fn(f64, i64) -> Result<R, SciOxideError> = R::from_parts;
    const _: fn(f64, f64) -> Result<R, SciOxideError> = R::from_log;
    const _: fn(R, R) -> Result<R, SciOxideError> = R::add;
    const _: fn(R, R) -> Result<R, SciOxideError> = R::sub;
    const _: fn(R, R) -> Result<R, SciOxideError> = R::mul;
    const _: fn(R, R) -> Result<R, SciOxideError> = R::div;
    const _: fn(R) -> Result<R, SciOxideError> = R::square;
    const _: fn(R) -> Result<R, SciOxideError> = R::sqrt;
    const _: fn(R, R, R, R) -> Result<R, SciOxideError> = R::difference_of_products;
    const _: fn(R, i64) -> Result<R, SciOxideError> = R::shifted;
    const _: fn(R) -> Result<f64, SciOxideError> = R::ln_abs;
    const _: fn(R) -> Result<f64, SciOxideError> = R::exp;
    const _: fn(R) -> Result<f64, SciOxideError> = R::round;
    const _: fn(R) -> Result<f32, SciOxideError> = R::round_to::<f32>;
    const _: fn(R) -> Result<f64, SciOxideError> = R::try_value;
    const _: fn(R) -> Result<f32, SciOxideError> = R::try_value_to::<f32>;

    const _: fn(R, R) -> ScaledValue<f64> = ScaledValue::<f64>::from_components;
    const _: fn(R, R) -> V = V::from_components;
    const _: fn(Complex64) -> Result<V, SciOxideError> = V::from_value;
    const _: fn(Complex64, i64) -> Result<V, SciOxideError> = V::from_parts;
    const _: fn(Complex64, [i64; 2]) -> Result<V, SciOxideError> = V::from_component_parts;
    const _: fn(f64, f64) -> Result<ScaledValue<f64>, SciOxideError> = ScaledValue::<f64>::from_log;
    const _: fn(V, V) -> Result<V, SciOxideError> = V::add;
    const _: fn(V, V) -> Result<V, SciOxideError> = V::sub;
    const _: fn(V, V) -> Result<V, SciOxideError> = V::mul;
    const _: fn(V, V) -> Result<V, SciOxideError> = V::div;
    const _: fn(V, V, V, V) -> Result<V, SciOxideError> = V::difference_of_products;
    const _: fn(V) -> Result<V, SciOxideError> = V::sqrt;
    const _: fn(V) -> Result<R, SciOxideError> = V::magnitude;
    const _: fn(V, i64) -> Result<V, SciOxideError> = V::shifted;
    const _: fn(V, R) -> Result<V, SciOxideError> = V::mul_real;
    const _: fn(V, f64) -> Result<f64, SciOxideError> = V::scale_absolute_error;
    const _: fn(V) -> Result<Complex64, SciOxideError> = V::round;
    const _: fn(V) -> Result<Complex64, SciOxideError> = V::try_value;
    const _: fn(V) -> Result<P, SciOxideError> = V::try_projective;
    const _: fn(P) -> Result<V, SciOxideError> = P::into_scaled;

    const _: fn(f64) -> Result<A, SciOxideError> = A::from_value;
    const _: fn(&mut A, f64) -> Result<(), SciOxideError> = A::add_value;
    const _: fn(&mut A, R) -> Result<(), SciOxideError> = A::add_scaled;
    const _: fn(&mut A, R) -> Result<(), SciOxideError> = A::sub_scaled;
    const _: fn(&mut A, R, R) -> Result<(), SciOxideError> = A::add_product;
    const _: fn(A, i64) -> Result<A, SciOxideError> = A::shifted;
    const _: fn(A, R) -> Result<A, SciOxideError> = A::multiplied;
    const _: fn(A) -> Result<R, SciOxideError> = A::total;
    const _: fn(A) -> Result<f64, SciOxideError> = A::round;

    const _: fn(Complex64) -> Result<L, SciOxideError> = L::from_value;
    const _: fn(Complex64, f64) -> Result<L, SciOxideError> = L::from_log;
    const _: fn(V, f64) -> Result<L, SciOxideError> = L::from_scaled_log;
    const _: fn(L) -> Result<Complex64, SciOxideError> = L::round;
    const _: fn(L) -> Result<Complex64, SciOxideError> = L::try_value;
}

const _: fn(ExecutionContext<'static>, ExecutionPolicy, usize, PlanLimits) -> ExecutionPlan =
    ExecutionContext::plan;

use scioxide_core::{DiagonalKind, Triangle};
const _: Triangle = Triangle::Lower;
const _: DiagonalKind = DiagonalKind::Unit;

use scioxide_core::backend::standard::{
    dot, exponent, factor_quotient, finish_substitution, finite_candidate, fused_update, magnitude,
    ordinary_quotient, product, product_needs_retry, product_scales, restore, scale,
    substitute_row, sum_products, unit_then_normalized,
};

const _: fn(f64, f64, f64, f64) -> Option<f64> = scioxide_core::backend::certified_sum_two_products;
#[test]
fn supplied_fma_contraction_certificate() {
    assert_eq!(
        scioxide_core::backend::certified_sum_two_products_with(
            1.5_f64,
            2.5,
            3.5,
            4.5,
            f64::mul_add
        ),
        Some(19.5)
    );
}

const _: fn(&[(f64, f64)]) -> Option<f64> = scioxide_core::backend::certified_short_sum_products;
type ProductPairs = [(f64, f64); 4];
const _: fn(ProductPairs) -> Result<f64, SciOxideError> =
    scioxide_core::backend::checked_product_sum;

use scioxide_core::backend::{ProductFactor, RealProductSum};
const _: fn(f64) -> Result<ProductFactor<f64>, SciOxideError> = ProductFactor::new;
const _: fn(ProductFactor<f64>) -> ProductFactor<f64> = ProductFactor::normalized;
const _: fn() -> RealProductSum<f64> = RealProductSum::new;
const _: fn(&mut RealProductSum<f64>) = RealProductSum::clear;
const _: fn(&mut RealProductSum<f64>, f64, f64) -> Result<(), SciOxideError> =
    RealProductSum::add_product;
const _: fn(&mut RealProductSum<f64>) -> Result<f64, SciOxideError> = RealProductSum::finish;
const _: fn(&mut RealProductSum<f64>, f64, f64, f64) -> Result<(), SciOxideError> =
    RealProductSum::add_weighted_pair;
type WeightedPairOperation<T> = fn(
    &mut RealProductSum<f64>,
    ProductFactor<f64>,
    ProductFactor<f64>,
    ProductFactor<f64>,
) -> Result<T, SciOxideError>;
const _: WeightedPairOperation<()> = RealProductSum::add_weighted_pair_factors;
const _: WeightedPairOperation<f64> = RealProductSum::add_weighted_pair_prefix;

type TripleTerms = [(f64, f64, f64)];
const _: fn(&TripleTerms) -> Option<f64> = scioxide_core::backend::certified_short_sum_triples;

#[test]
fn shared_gcd_and_real_storage_projections() {
    assert_eq!(scioxide_core::backend::gcd(0, 0), 0);
    assert_eq!(scioxide_core::backend::gcd(0, usize::MAX), usize::MAX);
    assert_eq!(scioxide_core::backend::gcd(usize::MAX, usize::MAX - 1), 1);
    let mut values = [-0.0_f64, f64::INFINITY];
    let pointer = values.as_ptr();
    let result = f64::from_scalar_slice(f64::slice(&values));
    assert!(result.is_ok_and(|v| v.as_ptr() == pointer && v[0].to_bits() == (-0.0_f64).to_bits()));
    assert!(f32::from_scalar_slice(f64::slice(&values)).is_err());
    let projected = f64::from_scalar_slice_mut(f64::slice_mut(&mut values));
    assert!(projected.is_ok_and(|v| {
        v[1] = 3.;
        true
    }));
    assert_eq!(values[1], 3.);
}
