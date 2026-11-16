//! Interpolation and approximation of functions from samples.
//!
//! The crate builds a function from values at given points and evaluates it,
//! its derivatives and its integrals. It covers global polynomial and rational
//! interpolation, piecewise cubic interpolation, B-splines and NURBS, smoothing
//! splines, interpolation on rectilinear grids and of scattered data, local
//! regression and kriging, splines on the sphere and parametric curves.
//!
//! # Quick start
//!
//! ```
//! use scioxide_interp::{CubicSpline, Univariate};
//! // A not-a-knot cubic spline reproduces the cubic f(x) = x³.
//! let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
//! let y: Vec<f64> = x.iter().map(|&t| t * t * t).collect();
//! let spline = CubicSpline::new(x, y)?;
//! assert!((spline.value_at(2.5)? - 15.625).abs() < 1e-10);
//! assert!((spline.derivative_at(2.5)? - 18.75).abs() < 1e-10); // f′(2.5) = 3 · 2.5²
//! assert!((spline.integrate(0.0, 4.0)? - 64.0).abs() < 1e-10);
//! let values = spline.values_at(&[0.5, 1.5])?;
//! assert!((values[0] - 0.125).abs() < 1e-10 && (values[1] - 3.375).abs() < 1e-10);
//! assert!(spline.value_at(5.0).is_err()); // outside [0, 4]
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Data and goal | Entry points | Method |
//! |---|---|---|
//! | Samples on a line, smooth curve | [`CubicSpline`] | cubic spline with natural, clamped, not-a-knot or periodic ends |
//! | Samples on a line, straight segments | [`BSpline::linear`] | piecewise linear interpolation |
//! | Monotone samples on a line, no overshoot | [`PchipInterpolator`] | piecewise cubic Hermite with Fritsch–Butland slopes |
//! | Samples on a line, fewer wiggles than a spline | [`AkimaInterpolator`] | piecewise cubic Hermite with Akima or modified Akima slopes |
//! | Values and first derivatives | [`HermiteInterpolator`] | Hermite interpolation by divided differences |
//! | One polynomial through all points | [`BarycentricInterpolator`], [`NewtonInterpolator`], [`NevilleInterpolator`] | barycentric formula of the second kind; Newton divided differences, with point insertion; Neville's algorithm, with an error estimate |
//! | One rational function through all points | [`FloaterHormannInterpolator`], [`BulirschStoerInterpolator`] | Floater–Hormann blending; Bulirsch–Stoer diagonal rational interpolation |
//! | Rational approximation of a real or complex function | [`aaa`], [`BarycentricRational`] | AAA algorithm |
//! | Rational approximation from Taylor coefficients | [`PadeApproximant`] | Padé approximation |
//! | Given knots and coefficients | [`BSpline`], [`BSplineBasis`], [`Nurbs`], [`cardinal_bspline`] | B-spline and NURBS evaluation in any number of variables |
//! | Interpolation on a tensor grid of any dimension | [`BSpline::interpolate`], [`TensorSplineInterpolationPlan`] | tensor-product B-spline collocation, not-a-knot or with given bases |
//! | Smoothing noisy samples | [`PenalizedSplinePlan`] | penalized least squares B-splines (P-splines) |
//! | Values on a rectilinear grid in the plane | [`BilinearInterpolator`], [`BicubicInterpolator`] | bilinear; bicubic Hermite with finite-difference slopes, or a monotone variant |
//! | Scattered points in the plane | [`RbfInterpolator`], [`CloughTocherInterpolator`], [`NaturalNeighborInterpolator`] | radial basis functions; Clough–Tocher C¹ cubic triangles; Sibson natural neighbors |
//! | Scattered points in `D` dimensions | [`interpolate_scattered`], [`SimplexLinearInterpolator`], [`NearestNeighborInterpolator`] | linear on Delaunay simplices; nearest site |
//! | Many scattered points, local fits | [`LocalModelPlan`], [`LocalRbfModel`], [`LocalPolynomialModel`] | radial basis interpolation or polynomial least squares on nearest neighbors |
//! | Kriging with a known covariance | [`GaussianProcessModel`] | simple or ordinary kriging with squared exponential or Matérn kernels |
//! | Samples on the sphere | [`SphericalSplinePlan`], [`fit_spherical_spline`] | C¹ spherical cubic splines, fixed or adaptive knots |
//! | Curves through points in space | [`fit_parametric_spline`] | adaptive spline fit with chord-length or supplied parameters |
//!
//! The triangulation-based interpolators take a [`Delaunay`](scioxide_spatial::Delaunay)
//! triangulation from `scioxide-spatial` and borrow it; [`interpolate_scattered`]
//! builds one for a single set of queries.
//!
//! # Construction
//!
//! A model is built from its samples by `new(data)`, which fixes every setting
//! to the default its rustdoc states, such as the not-a-knot ends of
//! [`CubicSpline::new`]. `with_options(data, options)` is the configured form,
//! and `build(data, options, scope)`, where it exists, runs inside a caller's
//! solve (see [solve accounting](scioxide_core#solve-accounting)). Options types
//! are passed by value and implement `Default`; a value that defines the result
//! and has no neutral default, such as the kernel of an [`RbfInterpolator`] or
//! the blending degree of a [`FloaterHormannInterpolator`], is part of the data
//! instead. The local models are the exception: [`LocalPolynomialOptions::new`]
//! takes the degree and neighbor counts, and [`LocalRbfOptions::new`] the
//! neighbor count and kernel, that define the model, so these two options types
//! have no `Default`. Named constructors select a different kind of preparation, such as
//! [`CubicSpline::with_boundary`], [`BicubicInterpolator::monotone`] or
//! [`BarycentricInterpolator::from_chebyshev_lobatto`].
//!
//! The iterative fits [`aaa`], [`fit_parametric_spline`] and
//! [`fit_spherical_spline`], and [`CloughTocherInterpolator::new`], return the
//! model only when the iteration succeeds, and `ConvergenceFailure` otherwise.
//! Their `with_options` forms return the model together with its report.
//!
//! # Values and coordinates
//!
//! Coordinates are [`RealScalar`](scioxide_core::RealScalar)s. A value can be any
//! [`AffineValue`](scioxide_core::AffineValue) whose samples have the same shape:
//! a real or complex scalar, an array or a vector. Each real component goes
//! through the same scalar computation, so the result equals component-wise
//! scalar interpolation bit for bit. [`aaa`], [`BarycentricRational`] and
//! [`PadeApproximant`] need division of values and take a
//! [`NumericScalar`](scioxide_core::NumericScalar) instead.
//!
//! ```
//! use scioxide_core::Complex;
//! use scioxide_interp::{CubicSpline, ExtrapolationPolicy, Interpolant, Univariate};
//! // f(t) = 2t + i(1 − t), a line in the complex plane.
//! let x = vec![0.0_f64, 1.0, 2.0, 3.0];
//! let y: Vec<Complex<f64>> = x.iter().map(|&t| Complex::new(2.0 * t, 1.0 - t)).collect();
//! let spline = CubicSpline::new(x, y)?;
//! let value = spline.value_at(1.5)?;
//! assert!((value.re - 3.0).abs() < 1e-14 && (value.im + 0.5).abs() < 1e-14);
//! assert!((spline.derivative_at(1.5)?.re - 2.0).abs() < 1e-14);
//! // Clamping extends the spline by its end value beyond x = 3.
//! let clamped = spline.evaluate([4.0], ExtrapolationPolicy::Clamp)?;
//! assert_eq!(clamped, spline.value_at(3.0)?);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! One-dimensional nodes must be strictly increasing. Passing
//! [`NodeSeparation::Tolerant`] in [`SampleOptions`] to a `with_options` form
//! also rejects nodes closer than a tolerance.
//!
//! # Queries
//!
//! Every interpolant implements [`Interpolant`], and where defined
//! [`Differentiable`], [`TwiceDifferentiable`] and [`Integrable`]. An
//! interpolant of one variable also has the scalar forms of [`Univariate`].
//!
//! | Points | On the domain | With a policy | Into caller storage |
//! |---|---|---|---|
//! | One | [`value_at`](Univariate::value_at), [`derivative_at`](Univariate::derivative_at), [`integrate`](Univariate::integrate) | [`evaluate`](Interpolant::evaluate), [`gradient`](Differentiable::gradient), [`hessian`](TwiceDifferentiable::hessian), [`integral`](Integrable::integral) | [`evaluate_into`](Interpolant::evaluate_into), [`gradient_into`](Differentiable::gradient_into), [`hessian_into`](TwiceDifferentiable::hessian_into), [`integral_into`](Integrable::integral_into) |
//! | Many | [`values_at`](Univariate::values_at) | [`evaluate_many`](Interpolant::evaluate_many) | [`BatchEvaluator`] |
//!
//! The owned forms return the result and allocate a workspace for each call;
//! the many-point forms allocate one for all points. The `*_into` forms write
//! into a value you provide and use a workspace from
//! [`Interpolant::workspace`]. [`BatchEvaluator`] writes values, gradients or
//! Hessians at many points into array views, optionally in parallel. Some
//! families add higher derivatives, such as [`BSpline::derivative_into`], and
//! antiderivatives prepared once for repeated integrals, such as
//! [`CubicSpline::prepare_antiderivative`].
//!
//! # Extrapolation
//!
//! The scalar forms of [`Univariate`] are defined on the closed domain and
//! return `DomainError` outside it. Every other query takes an
//! [`ExtrapolationPolicy`]; none is applied implicitly. The domain is the box
//! spanned by the nodes, or the convex hull of the sites for
//! triangulation-based interpolators.
//!
//! - `Reject` returns `DomainError` outside the domain.
//! - `Clamp` evaluates at the nearest point of the domain. Along an axis that is
//!   clamped, derivatives are zero, and integrals add the boundary value times
//!   the length outside. On a convex hull, derivatives are those of the
//!   function composed with the nearest-point projection; where the limits from
//!   neighboring hull faces disagree, the query returns `DomainError`.
//! - `BoundedContinue { max_distance }` continues the interpolant's formula up
//!   to and including `max_distance` outside the domain on each axis.
//! - `MethodDefined` continues the formula without a limit: the global
//!   polynomial or rational function, or the outermost piece.
//!
//! Each interpolator states what its formula is outside the domain.
//!
//! [`Interpolant::report`] and [`Interpolant::evaluate_into_with_report`]
//! return an [`ExtrapolationReport`] with the distance outside the domain.
//!
//! # Errors
//!
//! The rules shared by all SciOxide crates are in the
//! [`scioxide_core` crate docs](scioxide_core#errors). In particular, a failed query
//! leaves its output unchanged ([output on error](scioxide_core#output-on-error)).
//! Constructors check their input as follows.
//!
//! | Condition | Error |
//! |---|---|
//! | A required input is empty | `InvalidInput` (`Empty`) |
//! | Fewer samples than the method needs | `InvalidInput` (`OutOfRange`) |
//! | A length that does not match its partner, or values of different shapes | `DimensionMismatch` |
//! | NaN or infinite coordinate, value or weight | `NonFinite` (`Input`) |
//! | Decreasing or coincident nodes, knots or scattered points | `InvalidInput` (`Inconsistent`) |
//! | Negative weight, or nonpositive where a positive one is needed | `InvalidInput` (`OutOfRange`) |
//! | No weight is positive | `InvalidInput` (`Inconsistent`) |
//! | Estimated memory above a configured limit | `ResourceLimit` |
//!
//! Queries return `NonFinite` for a NaN or infinite coordinate, `DomainError`
//! outside the domain (see [Extrapolation](#extrapolation)), and
//! `DimensionMismatch` for an output of the wrong size or a workspace that is
//! too small. A NaN or infinity that arises during a computation is
//! `NonFinite` (`Intermediate`). Each type lists its own errors as well, for
//! example `NumericalInstability` (`Breakdown`) at a pole of a rational
//! interpolant, or `Singular` for a singular kernel matrix.
//!
//! Iterative fits (AAA, the adaptive spherical and parametric fits, and the
//! Clough–Tocher gradient estimation) report a termination status, residual and
//! iteration counts, and count their work as described in
//! [solve accounting](scioxide_core#solve-accounting). Their one-shot forms
//! return `ConvergenceFailure` for an unsuccessful status (see
//! [Construction](#construction)).
//!
//! # Accuracy
//!
//! The error of an interpolant is the error of the method plus rounding error.
//! For samples of a smooth function at spacing `h`, piecewise linear and
//! bilinear interpolation have error `O(h²)`. A cubic spline with clamped or
//! not-a-knot ends has error `O(h⁴)`; natural ends reduce this to `O(h²)` near
//! the ends unless the function's second derivative vanishes there.
//!
//! A polynomial through many equispaced points can be far from the function
//! (the Runge phenomenon), because the Lebesgue constant grows exponentially
//! with the number of points `n`. On Chebyshev points it grows like `log n`. Use
//! [`BarycentricInterpolator::from_chebyshev_lobatto`] or
//! [`BarycentricInterpolator::from_chebyshev_roots`] when you can choose the
//! points, and a spline or a rational method otherwise.
//!
//! Fits report what they achieved: [`RationalReport`], [`SplineFitReport`],
//! [`AdaptiveSplineReport`], [`LocalFitReport`] and [`CloughTocherReport`] give
//! residuals or condition estimates.
//!
//! # Reusing plans and workspaces
//!
//! Construction does the expensive work once, and the result is immutable and
//! can be shared between threads. For repeated queries, create one workspace
//! per thread with [`Interpolant::workspace`] and call the `*_into` forms; they
//! then do not allocate. Four plans precompute what repeated fits share:
//!
//! - [`TensorSplineInterpolationPlan`]: the banded LU factor of each axis
//!   collocation matrix, for many value sets on one grid.
//! - [`PenalizedSplinePlan`]: the penalized normal matrix and its banded
//!   Cholesky factor, for many value sets at one set of sample sites.
//! - [`LocalModelPlan`]: a k-d tree over the sites, shared by every local fit.
//! - [`SphericalSplinePlan`]: the design matrix and its Householder factor.
//!
//! The spline plans report their storage with `workspace_requirements` and fit
//! into caller storage with `fit_into`. AAA has nothing to precompute:
//! [`RationalWorkspace::workspace`] sizes a workspace for a [`RationalRequest`],
//! and [`aaa_into`] refits into a reserved [`RationalFit`] without allocating.
//! See [plans and workspaces](scioxide_core#plans-and-workspaces) and
//! [allocation](scioxide_core#allocation).
//!
//! # Feature flags
//!
//! - `std` (default): enables `std` in the SciOxide dependencies. Without it the
//!   crate is `no_std` and needs `alloc`; the API is the same.
//!
//! # References
//!
//! - C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001.
//! - J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation",
//!   *SIAM Review* 46(3), 2004.
//! - L. N. Trefethen, *Approximation Theory and Approximation Practice*,
//!   extended ed., SIAM, 2019.
//! - H. Wendland, *Scattered Data Approximation*, Cambridge University Press,
//!   2005.

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)]
#[macro_use]
extern crate alloc;
#[cfg(not(test))]
extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod test_support;

mod adaptive;
mod batch;
mod checked;
mod error;
mod factor;
pub use factor::{FitConditionNorm, FitFactorization};
mod geometry;
mod grid;
mod hermite_basis;
mod local;
mod parametric;
mod piecewise;
mod polynomial;
mod protocol;
mod radial;
mod rational;
mod rbf;
mod spherical;
mod spline;
mod tessellation;
mod validate;
mod values;

// Shared protocol, value storage and batch execution.
pub use batch::{BatchEvaluator, BatchOperation, BatchRequest, BatchRequirements, BatchWorkspace};
pub use protocol::{
    Differentiable, ExtrapolationPolicy, ExtrapolationReport, Integrable, Interpolant,
    NodeSeparation, SampleOptions, TwiceDifferentiable, Univariate,
};
pub use values::QueryWorkspace;

// Rectilinear grids.
pub use grid::{BicubicInterpolator, BilinearInterpolator};

// Global polynomials and rational approximation.
pub use polynomial::{
    BarycentricInterpolator, BulirschStoerInterpolator, HermiteInterpolator, NevilleInterpolator,
    NewtonInterpolator, PreparedPolynomialAntiderivative,
};
pub use rational::{
    AaaOptions, BarycentricRational, FloaterHormannInterpolator, PadeApproximant, RationalCleanup,
    RationalDiagnostics, RationalDomain, RationalFit, RationalPole, RationalReport,
    RationalRequest, RationalRequirements, RationalTermination, RationalWorkspace, aaa, aaa_into,
    aaa_with_options,
};

// Piecewise cubics, B-splines and spline fits.
pub use piecewise::{
    AkimaInterpolator, AkimaMethod, CubicSpline, PchipInterpolator, SplineBoundary,
};
pub use spline::{
    BSpline, BSplineBasis, KnotSide, Nurbs, PenalizedSplineFit, PenalizedSplinePlan,
    PenalizedSplineRequirements, PenalizedSplineWorkspace, PreparedSplineAntiderivative,
    SplineBasisWorkspace, SplineFitDiagnostics, SplineFitReport, SplinePenalty,
    TensorSplineInterpolationPlan, TensorSplineInterpolationRequirements,
    TensorSplineInterpolationWorkspace, cardinal_bspline, difference_coefficients,
};

// Scattered data, RBF and local models.
pub use local::{
    Covariance, CovarianceKernel, GaussianPrediction, GaussianProcessModel, KrigingMean,
    LocalFitReport, LocalModelOptions, LocalModelPlan, LocalPolynomialModel,
    LocalPolynomialOptions, LocalPredictionWorkspace, LocalRbfModel, LocalRbfOptions, LocalSamples,
};
pub use rbf::{RbfConditioning, RbfInterpolator, RbfKernel, RbfOptions, RbfRegularization};
pub use tessellation::{
    CloughTocherInterpolator, CloughTocherOptions, CloughTocherReport, CloughTocherTermination,
    NaturalNeighborInterpolator, NearestNeighborInterpolator, ScatteredOptions, ScatteredValues,
    ScatteredWorkspace, SimplexLinearInterpolator, interpolate_scattered,
};

// Spherical, parametric and adaptive spline fits.
pub use adaptive::{
    AdaptiveSplineOptions, AdaptiveSplineReport, AdaptiveSplineTermination, SplineFitLimits,
    SplineFitTarget,
};
pub use parametric::{
    CurveParameterization, ParametricSplineFit, fit_parametric_spline,
    fit_parametric_spline_scoped, fit_parametric_spline_with_options,
};
pub use spherical::{
    SphericalSamples, SphericalSpline, SphericalSplineFit, SphericalSplineFitRequirements,
    SphericalSplineFitWorkspace, SphericalSplinePlan, SphericalSplineWorkspace,
    fit_spherical_spline, fit_spherical_spline_scoped, fit_spherical_spline_with_options,
};
