//! Reviewed crate-root inventory, compiled as an external Rust consumer.
#![expect(
    unused_imports,
    reason = "The external API inventory verifies crate-root exports by importing them."
)]
use scioxide_core::{Complex, ExecutionPolicy, SciOxideError, Tolerance};
use scioxide_interp::{
    AaaOptions, AdaptiveSplineOptions, AdaptiveSplineReport, AdaptiveSplineTermination,
    AkimaInterpolator, AkimaMethod, BSpline, BSplineBasis, BarycentricInterpolator,
    BarycentricRational, BatchEvaluator, BatchOperation, BatchRequest, BatchRequirements,
    BatchWorkspace, BicubicInterpolator, BilinearInterpolator, BulirschStoerInterpolator,
    CloughTocherInterpolator, CloughTocherOptions, CloughTocherReport, CloughTocherTermination,
    Covariance, CovarianceKernel, CubicSpline, CurveParameterization, Differentiable,
    ExtrapolationPolicy, ExtrapolationReport, FitConditionNorm, FitFactorization,
    FloaterHormannInterpolator, GaussianPrediction, GaussianProcessModel, HermiteInterpolator,
    Integrable, Interpolant, KnotSide, KrigingMean, LocalFitReport, LocalModelOptions,
    LocalModelPlan, LocalPolynomialModel, LocalPolynomialOptions, LocalPredictionWorkspace,
    LocalRbfModel, LocalRbfOptions, LocalSamples, NaturalNeighborInterpolator,
    NearestNeighborInterpolator, NevilleInterpolator, NewtonInterpolator, NodeSeparation, Nurbs,
    PadeApproximant, ParametricSplineFit, PchipInterpolator, PenalizedSplineFit,
    PenalizedSplinePlan, PenalizedSplineRequirements, PenalizedSplineWorkspace,
    PreparedPolynomialAntiderivative, PreparedSplineAntiderivative, QueryWorkspace,
    RationalCleanup, RationalDiagnostics, RationalDomain, RationalFit, RationalPole,
    RationalReport, RationalRequest, RationalRequirements, RationalTermination, RationalWorkspace,
    RbfConditioning, RbfInterpolator, RbfKernel, RbfOptions, RbfRegularization, SampleOptions,
    ScatteredOptions, ScatteredValues, ScatteredWorkspace, SimplexLinearInterpolator,
    SphericalSamples, SphericalSpline, SphericalSplineFit, SphericalSplineFitRequirements,
    SphericalSplineFitWorkspace, SphericalSplinePlan, SphericalSplineWorkspace,
    SplineBasisWorkspace, SplineBoundary, SplineFitDiagnostics, SplineFitLimits, SplineFitReport,
    SplineFitTarget, SplinePenalty, TensorSplineInterpolationPlan,
    TensorSplineInterpolationRequirements, TensorSplineInterpolationWorkspace, TwiceDifferentiable,
    Univariate, aaa, aaa_into, aaa_with_options, cardinal_bspline, difference_coefficients,
    fit_parametric_spline, fit_parametric_spline_scoped, fit_parametric_spline_with_options,
    fit_spherical_spline, fit_spherical_spline_scoped, fit_spherical_spline_with_options,
    interpolate_scattered,
};

// Owned constructors take their samples by value and validate once.
type Options = SampleOptions<f64>;
type Built<T> = Result<T, SciOxideError>;
type Samples<T> = fn(Vec<f64>, Vec<f64>) -> Built<T>;
type SamplesWith<T> = fn(Vec<f64>, Vec<f64>, Options) -> Built<T>;
type Structured<T> = fn(Vec<f64>, f64, f64) -> Built<T>;
type Slopes<T> = fn(Vec<f64>, Vec<f64>, Vec<f64>) -> Built<T>;
type Boundary<T> = fn(Vec<f64>, Vec<f64>, SplineBoundary<f64>) -> Built<T>;
type BoundaryWith<T> = fn(Vec<f64>, Vec<f64>, SplineBoundary<f64>, Options) -> Built<T>;
type Grid<T> = fn(Vec<f64>, Vec<f64>, Vec<Complex<f64>>) -> Built<T>;
type Tensor<T> = fn([BSplineBasis<f64>; 2], Vec<f64>) -> Built<T>;
type Rbf<T> = fn(Vec<[f64; 2]>, Vec<f64>, RbfKernel<f64>) -> Built<T>;
type Penalized<T> =
    fn(BSplineBasis<f64>, Vec<f64>, Option<Vec<f64>>, SplinePenalty<f64>) -> Built<T>;
type Batch<T> = fn(BilinearInterpolator<f64>, ExecutionPolicy) -> T;
type Method<T> = fn(Vec<f64>, Vec<f64>, AkimaMethod) -> Built<T>;
type Knots = fn(&[f64], usize) -> Built<BSplineBasis<f64>>;
type GridFit<T> = fn([Vec<f64>; 2], &[f64], usize) -> Built<T>;
type Scattered = fn(&[[f64; 2]], Vec<f64>, &[[f64; 2]]) -> Built<Vec<f64>>;
type CurveFit<T> = fn(&[[f64; 3]], f64, Tolerance<f64>) -> Built<T>;
type SphereFit<T> = fn(&[[f64; 2]], &[f64], f64, Tolerance<f64>) -> Built<T>;
const _: Samples<BarycentricInterpolator<f64>> = BarycentricInterpolator::new;
const _: SamplesWith<BarycentricInterpolator<f64>> = BarycentricInterpolator::with_options;
const _: Structured<BarycentricInterpolator<f64>> = BarycentricInterpolator::from_chebyshev_lobatto;
const _: Samples<NewtonInterpolator<f64>> = NewtonInterpolator::new;
const _: Samples<BSpline<f64>> = BSpline::linear;
const _: Slopes<HermiteInterpolator<f64>> = HermiteInterpolator::new;
const _: Samples<CubicSpline<f64>> = CubicSpline::new;
const _: Boundary<CubicSpline<f64>> = CubicSpline::with_boundary;
const _: BoundaryWith<CubicSpline<f64>> = CubicSpline::with_options;
const _: Grid<BicubicInterpolator<f64, Complex<f64>>> = BicubicInterpolator::new;
const _: Grid<BicubicInterpolator<f64, Complex<f64>>> = BicubicInterpolator::monotone;
const _: Tensor<BSpline<f64, f64, 2>> = BSpline::new;
const _: Rbf<RbfInterpolator<f64>> = RbfInterpolator::new;
const _: Penalized<PenalizedSplinePlan<f64>> = PenalizedSplinePlan::new;
const _: fn(&[f64], &[f64]) -> Built<BarycentricRational<f64>> = aaa::<f64>;
const _: Method<AkimaInterpolator<f64>> = AkimaInterpolator::with_method;
const _: Knots = BSplineBasis::clamped;
const _: Knots = BSplineBasis::not_a_knot;
const _: GridFit<BSpline<f64, f64, 2>> = BSpline::<f64, f64, 2>::interpolate;
const _: Scattered = interpolate_scattered::<f64, f64, 2>;
const _: CurveFit<BSpline<f64, [f64; 3], 1>> = fit_parametric_spline::<f64, [f64; 3]>;
const _: SphereFit<SphericalSpline<f64>> = fit_spherical_spline::<f64, f64>;
const _: Batch<BatchEvaluator<f64, BilinearInterpolator<f64>, 2>> = BatchEvaluator::new;

// Reports expose one termination authority through accessors.
const _: fn(&RationalReport<f64>) -> RationalTermination = RationalReport::termination;
const _: fn(&AdaptiveSplineReport<f64>) -> AdaptiveSplineTermination =
    AdaptiveSplineReport::termination;
const _: fn(&CloughTocherReport<f64>) -> CloughTocherTermination = CloughTocherReport::termination;
const _: fn(&ExtrapolationReport<f64, 2>) -> [f64; 2] = ExtrapolationReport::effective;

#[test]
fn standard_plans_interpolants_and_options_are_clone_send_sync() {
    fn check<T: Clone + Send + Sync>() {}
    // Shared configuration and batching.
    check::<BatchEvaluator<f64, BicubicInterpolator<f64>, 2>>();
    check::<ExtrapolationPolicy<f64>>();
    check::<NodeSeparation<f64>>();
    check::<SampleOptions<f64>>();
    // Grids and global polynomials.
    check::<BilinearInterpolator<f64, Vec<f64>>>();
    check::<BicubicInterpolator<f64, Complex<f64>>>();
    check::<BarycentricInterpolator<f64, Vec<f64>>>();
    check::<NewtonInterpolator<f64, Complex<f64>>>();
    check::<NevilleInterpolator<f64>>();
    check::<HermiteInterpolator<f64>>();
    check::<BulirschStoerInterpolator<f64>>();
    check::<PreparedPolynomialAntiderivative<f64, Vec<f64>>>();
    // Rational approximation.
    check::<AaaOptions<f64>>();
    check::<RationalRequest<f64>>();
    check::<RationalFit<Complex<f64>>>();
    check::<BarycentricRational<f64>>();
    check::<BarycentricRational<Complex<f64>>>();
    check::<FloaterHormannInterpolator<f64, Complex<f64>>>();
    check::<PadeApproximant<f64>>();
    // Piecewise cubics and splines.
    check::<AkimaInterpolator<f64, [f64; 3]>>();
    check::<PchipInterpolator<f64>>();
    check::<CubicSpline<f64, Complex<f64>>>();
    check::<BSpline<f64>>();
    check::<BSpline<f64, Vec<f64>, 2>>();
    check::<BSplineBasis<f64>>();
    check::<Nurbs<f64, [f64; 3], 2>>();
    check::<PreparedSplineAntiderivative<f64>>();
    check::<SplinePenalty<f64>>();
    check::<PenalizedSplinePlan<f64>>();
    check::<TensorSplineInterpolationPlan<f64, 2>>();
    // Scattered data and local models.
    check::<RbfOptions<f64>>();
    check::<RbfInterpolator<f64, Complex<f64>>>();
    check::<ScatteredOptions>();
    check::<ScatteredValues<f64>>();
    check::<CloughTocherOptions<f64>>();
    check::<NearestNeighborInterpolator<'static, f64, 2>>();
    check::<SimplexLinearInterpolator<'static, f64, 3>>();
    check::<CloughTocherInterpolator<'static, f64>>();
    check::<NaturalNeighborInterpolator<'static, f64>>();
    check::<LocalModelOptions<f64>>();
    check::<LocalRbfOptions<f64>>();
    check::<LocalPolynomialOptions<f64>>();
    check::<LocalModelPlan<'static, 'static, f64, 2>>();
    check::<LocalRbfModel<f64, 2>>();
    check::<LocalPolynomialModel<f64, 2>>();
    check::<Covariance<f64, 2>>();
    check::<GaussianProcessModel<f64, 2>>();
    // Spherical, parametric and adaptive fits.
    check::<SplineFitLimits>();
    check::<AdaptiveSplineOptions<f64>>();
    check::<SphericalSplinePlan<f64>>();
    check::<SphericalSpline<f64, Complex<f64>>>();
}

#[test]
fn functions_of_one_variable_have_the_scalar_forms() {
    fn univariate<I: Univariate<f64>>() {}
    univariate::<CubicSpline<f64>>();
    univariate::<PchipInterpolator<f64, Complex<f64>>>();
    univariate::<AkimaInterpolator<f64, [f64; 2]>>();
    univariate::<BarycentricInterpolator<f64>>();
    univariate::<NewtonInterpolator<f64>>();
    univariate::<HermiteInterpolator<f64>>();
    univariate::<NevilleInterpolator<f64>>();
    univariate::<BulirschStoerInterpolator<f64>>();
    univariate::<FloaterHormannInterpolator<f64>>();
    univariate::<BarycentricRational<f64>>();
    univariate::<BarycentricRational<Complex<f64>>>();
    univariate::<PadeApproximant<f64>>();
    univariate::<PadeApproximant<Complex<f64>>>();
    univariate::<BSpline<f64>>();
    univariate::<Nurbs<f64, f64, 1>>();
    univariate::<NearestNeighborInterpolator<'static, f64, 1>>();
    univariate::<SimplexLinearInterpolator<'static, f64, 1>>();
    univariate::<LocalRbfModel<f64, 1>>();
    univariate::<LocalPolynomialModel<f64, 1>>();
    univariate::<GaussianPrediction<'static, f64, 1>>();
    univariate::<&CubicSpline<f64>>();
}

#[test]
fn interpolants_share_one_protocol() {
    fn interpolant<I: Interpolant<f64, D>, const D: usize>() {}
    fn gradient<I: Differentiable<f64, D>, const D: usize>() {}
    fn hessian<I: TwiceDifferentiable<f64, D>, const D: usize>() {}
    fn integral<I: Integrable<f64, D>, const D: usize>() {}
    interpolant::<NevilleInterpolator<f64>, 1>();
    interpolant::<BulirschStoerInterpolator<f64>, 1>();
    interpolant::<FloaterHormannInterpolator<f64>, 1>();
    interpolant::<BarycentricRational<f64>, 1>();
    interpolant::<BarycentricRational<Complex<f64>>, 2>();
    interpolant::<PadeApproximant<f64>, 1>();
    interpolant::<BilinearInterpolator<f64>, 2>();
    interpolant::<RbfInterpolator<f64>, 2>();
    interpolant::<NearestNeighborInterpolator<'static, f64, 3>, 3>();
    interpolant::<NaturalNeighborInterpolator<'static, f64>, 2>();
    interpolant::<LocalRbfModel<f64, 2>, 2>();
    interpolant::<LocalPolynomialModel<f64, 2>, 2>();
    interpolant::<GaussianPrediction<'static, f64, 2>, 2>();
    gradient::<BarycentricInterpolator<f64>, 1>();
    gradient::<NewtonInterpolator<f64>, 1>();
    gradient::<HermiteInterpolator<f64>, 1>();
    gradient::<AkimaInterpolator<f64>, 1>();
    gradient::<SimplexLinearInterpolator<'static, f64, 2>, 2>();
    gradient::<CloughTocherInterpolator<'static, f64>, 2>();
    hessian::<BicubicInterpolator<f64>, 2>();
    hessian::<BSpline<f64, f64, 3>, 3>();
    hessian::<Nurbs<f64, f64, 2>, 2>();
    hessian::<SphericalSpline<f64>, 2>();
    integral::<PchipInterpolator<f64>, 1>();
    integral::<CubicSpline<f64>, 1>();
    integral::<BSpline<f64, f64, 2>, 2>();
    integral::<SphericalSpline<f64>, 2>();
}
