//! Numerical integration: quadrature, cubature, Monte Carlo and infinite series.
//!
//! The crate computes definite integrals of real, complex and vector-valued
//! functions in one or more dimensions. It covers fixed and adaptive quadrature
//! on finite and infinite intervals, integrals with singular or oscillatory
//! weights, integrals of sampled data, cubature on boxes, simplices and the
//! sphere, sparse grids and adaptive Monte Carlo methods. It also sums infinite
//! series with convergence acceleration and computes derivatives of integrals
//! with respect to parameters. Every method supports `f32` and `f64`.
//!
//! # Quick start
//!
//! ```
//! let integral = scioxide_integrate::integrate(|x: f64| Ok(x * x), 0.0, 1.0)?;
//! assert!((integral.value() - 1.0 / 3.0).abs() < 1e-12);
//! assert!(integral.error_estimate() < 1e-12);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Problem | One call | Configured entry points | Method |
//! |---|---|---|---|
//! | Smooth integrand or endpoint singularity on `[a, b]` | [`integrate`] | [`AdaptiveIntegrator`] | globally adaptive Gauss–Kronrod (G7K15) with optional epsilon-algorithm extrapolation, or adaptive Simpson |
//! | Kinks or jumps at known points | [`integrate_piecewise`] | [`WeightedQuadraturePlan::breakpoints`] | adaptive Gauss–Kronrod on the pieces |
//! | Fixed number of evaluations | [`GaussianRulePlan::gauss_legendre`] | [`FixedRule`] with [`GaussKronrod`], [`GaussianRulePlan`], [`ClenshawCurtis`], [`NewtonCotes`] | Gauss, Radau, Lobatto, Gauss–Kronrod, Clenshaw–Curtis and composite Newton–Cotes rules |
//! | Classical weight functions (Legendre, Laguerre, Hermite families) | [`GaussianRulePlan::new`] | [`GaussianRulePlan`] | Gaussian rules from the Jacobi matrix of the weight |
//! | Very smooth integrand, nested refinement | — | [`Romberg`] | Richardson extrapolation of the trapezoidal rule |
//! | Endpoint singularities, half-line, real line | [`integrate`], [`integrate_unbounded`] | [`DoubleExponentialIntegrator`] | tanh-sinh, exp-sinh and sinh-sinh quadrature |
//! | Algebraic–logarithmic endpoint weights, Cauchy principal values, `cos(ωx)` and `sin(ωx)` weights | [`integrate_fourier`] | [`WeightedQuadraturePlan`] | adaptive quadrature with modified Chebyshev moments |
//! | Fourier integrals on a half-line or the real line | — | [`FourierTailPlan`] | integration cycle by cycle with extrapolation of the partial sums |
//! | Oscillatory integrands `f(x) cos(ω g(x))` with a known phase `g` | — | [`OscillatoryQuadraturePlan`] | Levin collocation or Filon-type quadrature |
//! | Contour integrals in the complex plane | — | [`ContourPlan`] | adaptive Gauss–Kronrod in the curve parameter |
//! | Tabulated samples, totals and cumulative integrals | [`trapezoid`], [`simpson`], [`cumulative_trapezoid`] | [`SampledTrapezoidPlan`], [`SampledSimpsonPlan`], [`SampledRombergPlan`] | composite trapezoidal, Simpson and Romberg rules |
//! | Smooth integrand on a box, low dimension | [`integrate_box`] | [`CubaturePlan`], [`ProductRule`] | adaptive Genz–Malik cubature; tensor products of a fixed rule |
//! | Limits that depend on outer variables | [`integrate_iterated`] | [`IteratedDomainPlan`] | iterated adaptive Gauss–Kronrod |
//! | Smooth integrand on a box, moderate dimension | — | [`SparseGridPlan`] | dimension-adaptive Smolyak sparse grids of Clenshaw–Curtis rules |
//! | Simplices and the sphere | — | [`SimplexRule`], [`LebedevRule`], [`SphereProductRule`] | Grundmann–Möller and positive interior simplex rules; Lebedev–Laikov and product rules on the sphere |
//! | High dimension or nonsmooth integrand | [`integrate_monte_carlo`] | [`MonteCarloIntegrator`], [`VegasTrainer`], [`MiserTrainer`] | plain, importance and stratified Monte Carlo; VEGAS and MISER training |
//! | Infinite series | [`sum_series`] | [`SeriesSummation`] | Levin `u` transform, Wynn's epsilon algorithm or Euler–Maclaurin tail |
//! | Derivatives of integrals with respect to parameters | — | [`QuadratureSensitivityPlan`] | Leibniz rule, forward (JVP) and reverse (VJP) mode |
//!
//! `integrate` enables Wynn’s epsilon extrapolation. `sum_series` uses Levin’s
//! u transform; other summation methods use `SeriesSummation`.
//!
//! # Integrands
//!
//! Closures are passed by value; pass `&f` to keep using a closure.
//! An integrand is a closure that returns `Result<V, SciOxideError>`, where `V` is
//! a real scalar, a [`Complex`](scioxide_core::Complex) number or a fixed-size
//! array of either. One-dimensional integrands take `x: S` and multidimensional
//! integrands take `x: &[S]`. The prepared forms (`integrate_into`) also accept
//! slices and vectors as output, for vector values whose length is known only at
//! run time. A complex or vector value is integrated componentwise over its real
//! components, and tolerances apply to all components together through the norm
//! chosen in [`ErrorTolerance`](scioxide_core::ErrorTolerance).
//!
//! An error returned by the integrand stops the integration and is returned
//! unchanged. A NaN or infinite value from the integrand is a `NonFinite` error
//! with origin `Callback`. Integrands must be `Sync`, because an execution policy
//! other than `Sequential` evaluates independent points on several workers. The
//! result does not depend on the number of workers.
//!
//! # Termination
//!
//! The one-shot quadrature and cubature functions ([`integrate`],
//! [`integrate_unbounded`], [`integrate_piecewise`], [`integrate_fourier`],
//! [`integrate_box`] and [`integrate_iterated`]) return an [`Integral`] with the
//! value and estimated absolute error, and [`sum_series`] returns a
//! [`SeriesResult`]. They return an error on any unsuccessful termination.
//! [`trapezoid`], [`simpson`] and [`cumulative_trapezoid`] return the value alone,
//! without an error estimate, and [`integrate_monte_carlo`] returns
//! [`MonteCarloStatistics`].
//! Configured methods return an [`IntegrationReport`]. Its
//! [`IntegrationTermination`] says why the method stopped, and
//! [`IntegrationReport::is_success`] says whether the method finished as
//! configured. [`SeriesSummation`] and [`QuadratureSensitivityPlan`] have their own
//! reports with the same structure.
//!
//! When an evaluation or subdivision limit, cancellation or a deadline stops a
//! method that already has an estimate of the whole integral, the method returns
//! that estimate with an unsuccessful termination such as `BudgetLimit`,
//! `RefinementLimit` or `Stagnation`. When it stops before such an estimate
//! exists, it returns an error instead:
//!
//! - `ExecutionStopped` for cancellation or a deadline;
//! - `ConvergenceFailure` for an evaluation, iteration or refinement limit,
//!   stagnation or divergence;
//! - `NumericalInstability` (`Breakdown`) when no representable evaluation point
//!   exists.
//!
//! One-shot errors preserve stop reasons. Evaluation and iteration limits become
//! `EvaluationLimit` and `IterationLimit`; other budgets become
//! `ResourceLimit(kind)`, refinement limits become `TruncationLimit`, and
//! stagnation and probable divergence become `Stagnation` and `Divergence`.
//! Geometry failures are `NumericalInstability` (`Breakdown`).
//!
//! Series heuristic tolerance completion also counts as success; agreement of
//! transformed estimates does not prove convergence or bound the infinite tail.
//!
//! Iterative workspaces keep a record of their last call, with `last_attempt`.
//!
//! # Error estimates
//!
//! Error estimates are not bounds. For a smooth integrand that the method
//! resolves, they are usually pessimistic; for an integrand with features that
//! the method never samples, such as a narrow peak between nodes, they can be
//! arbitrarily wrong. Gauss–Kronrod, Simpson, Romberg and double-exponential
//! estimates include a sampled rounding-error floor, `50 ε Σ|wᵢ f(xᵢ)|`, where
//! `ε` is the machine epsilon and `wᵢ`, `xᵢ` are the weights and nodes.
//! Each plan documents its discretization and tail model. Sparse-grid surplus
//! estimates, in particular, do not include a separate rounding-error floor.
//! [`IntegrationErrorComponents`] lists the parts of the estimate: discretization,
//! rounding, inner integrals, moments, extrapolation and truncated tails.
//! [`ErrorReliability::Heuristic`] marks parts that rest on observed decay
//! rather than on a rule difference, for example the tail of a
//! double-exponential sum or a series remainder.
//!
//! Monte Carlo methods report a sample standard error and a confidence interval
//! in [`MonteCarloStatistics`]. These are statistical statements and assume that
//! the integrand has finite variance.
//!
//! For an enclosure that is guaranteed to contain the integral, use the verified
//! quadrature in `scioxide-validated`.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: invalid options, such as a tolerance or level count out of
//!   range, unsorted or out-of-range breakpoints, or workspace requirements from
//!   a different plan type.
//! - `NonFinite`: a bound, breakpoint or parameter is NaN or infinite where a
//!   finite value is required (origin `Input`), or the integrand returned one
//!   (origin `Callback`).
//! - `DimensionMismatch`: the output, a vector tolerance or the workspace does
//!   not match the number of value components.
//! - `ConvergenceFailure`, `ExecutionStopped`, `NumericalInstability`: a method
//!   stopped before it had an estimate of the whole integral, as described in
//!   [Termination](#termination). `NumericalInstability` also reports overflow
//!   of a result although the integrand values are finite.
//! - `AllocationFailure`, `SizeOverflow`: a workspace or plan cannot be
//!   allocated.
//!
//! A workspace that is too small for a call is a `DimensionMismatch` for some
//! plans and an `InvalidInput` for others. Each plan lists its own conditions,
//! including this one. On error, outputs are unchanged as
//! described in [Output on error](scioxide_core#output-on-error).
//!
//! # Reusing plans and workspaces
//!
//! Start with a one-shot function, which uses default options and returns an
//! answer after successful termination. Configure an integrator or plan to
//! choose another method or options; its owned `integrate` returns the value
//! and report. For repeated calls, prepare a workspace with `workspace` and
//! use `integrate_into` with caller output and an execution context or scope.
//! With sufficient storage, numerical execution does not allocate. Sampled plans
//! provide scalar, cumulative and axis workspaces from their prepared grid.
//!
//! ```
//! use scioxide_core::{ExecutionContext, Interval};
//! use scioxide_integrate::{AdaptiveIntegrator, AdaptiveRule, SubdivisionOptions};
//!
//! let options = SubdivisionOptions::<f64>::default();
//! let plan = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options)?;
//! let mut workspace = plan.workspace(1)?;
//! for n in 1..4 {
//!     // ∫₀¹ xⁿ dx = 1/(n + 1)
//!     let mut value = 0.0;
//!     let f = |x: f64| Ok(x.powi(n));
//!     let cx = ExecutionContext::sequential_native();
//!     plan.integrate_into(&f, Interval::new(0.0, 1.0)?, &mut value, &mut workspace, cx)?;
//!     assert!((value - 1.0 / f64::from(n + 1)).abs() < 1e-12);
//! }
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Feature flags
//!
//! - `std` (default): enables `std` in the SciOxide crates this crate depends on,
//!   which provides the thread executor and clock of `scioxide-core`.
//!
//! Without default features the crate is `no_std` and needs only `alloc`. The
//! API is the same.
//!
//! # References
//!
//! - P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed.,
//!   Academic Press, 1984.
//! - R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner,
//!   *QUADPACK: A Subroutine Package for Automatic Integration*, Springer, 1983.
//! - G. H. Golub and J. H. Welsch, "Calculation of Gauss quadrature rules",
//!   *Math. Comp.* 23(106), 221–230, 1969.
//! - H. Takahasi and M. Mori, "Double exponential formulas for numerical
//!   integration", *Publ. RIMS Kyoto Univ.* 9(3), 721–741, 1974.
//! - A. C. Genz and A. A. Malik, "An adaptive algorithm for numerical integration
//!   over an N-dimensional rectangular region", *J. Comput. Appl. Math.* 6(4),
//!   295–302, 1980.
//! - T. Gerstner and M. Griebel, "Dimension-adaptive tensor-product quadrature",
//!   *Computing* 71(1), 65–87, 2003.
//! - G. P. Lepage, "A new algorithm for adaptive multidimensional integration",
//!   *J. Comput. Phys.* 27(2), 192–203, 1978.
//! - W. H. Press and G. R. Farrar, "Recursive stratified sampling for
//!   multidimensional Monte Carlo integration", *Computers in Physics* 4(2),
//!   190–195, 1990.
//! - D. Levin, "Development of non-linear transformations for improving
//!   convergence of sequences", *Int. J. Comput. Math.* 3, 371–388, 1973.
//! - P. Wynn, "On a device for computing the eₘ(Sₙ) transformation",
//!   *Math. Tables Aids Comput.* 10(54), 91–96, 1956.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod adaptive;
mod double_exponential;
mod evaluation;
mod execution;
mod fixed;
mod gauss;
mod geometry;
mod kernel;
mod level;
mod multidim;
mod oscillatory;
mod report;
mod romberg;
mod sampled;
mod sensitivity;
mod series;
mod weighted;
mod workspace;

pub(crate) use execution::RegionalProgress;

pub use adaptive::{
    AdaptiveIntegrator, AdaptiveRule, AdaptiveWorkspace, SubdivisionOptions, integrate,
};

pub use double_exponential::{
    DoubleExponentialDomain, DoubleExponentialIntegrator, DoubleExponentialPoint,
    DoubleExponentialWorkspace, integrate_unbounded,
};

pub use fixed::{
    ClenshawCurtis, FixedRule, FixedRuleNode, FixedRuleWorkspace, NewtonCotes, NewtonCotesEndpoints,
};

pub use gauss::{
    GaussKronrod, GaussKronrodRule, GaussianDomain, GaussianRuleOptions, GaussianRulePlan,
    GaussianRuleVariant, GaussianSupport,
};

pub use multidim::{
    CoordinateMap, CubaturePlan, CubatureWorkspace, GeometricOptions, GeometricWorkspace,
    IdentityMap, IteratedDomainPlan, IteratedOptions, IteratedWorkspace, LebedevRule,
    ProductOptions, ProductRule, ProductWorkspace, SimplexMeasure, SimplexRule, SparseGridOptions,
    SparseGridPlan, SparseGridWorkspace, SphereProductRule, integrate_box, integrate_iterated,
    integrate_monte_carlo,
};

pub use multidim::{
    MiserOptions, MiserStrata, MiserTrainer, MiserWorkspace, MonteCarloChild, MonteCarloIntegrator,
    MonteCarloOptions, MonteCarloProposal, MonteCarloStatistics, MonteCarloWorkspace,
    ProposalUniforms, RandomStreams, SAMPLE_BLOCK, TrainingReport, UniformProposal, UniformSource,
    VarianceBasis, VegasMap, VegasOptions, VegasTrainer, VegasWorkspace,
};

pub use oscillatory::{
    OscillatoryGeometry, OscillatoryMethod, OscillatoryOptions, OscillatoryPhase,
    OscillatoryPhaseInverse, OscillatoryQuadraturePlan, OscillatoryRegion, OscillatoryWorkspace,
};

pub use report::{
    ErrorEstimate, ErrorReliability, Integral, IntegrationErrorComponents, IntegrationReport,
    IntegrationTermination, TailDiagnostics, TailStatus,
};

pub use level::LevelOptions;
pub use romberg::{Romberg, RombergWorkspace};

pub use sampled::{
    SampledAxisWorkspace, SampledRombergPlan, SampledSimpsonPlan, SampledTrapezoidPlan,
    SampledWorkspace, cumulative_trapezoid, simpson, trapezoid,
};

pub use sensitivity::{
    JvpQuery, QuadratureSensitivityPlan, QuadratureSensitivityWorkspace, SensitivityDiscretization,
    SensitivityErrorComponents, SensitivityIntegrand, SensitivityMode, SensitivityOptions,
    SensitivityPartition, SensitivityReport, SensitivityRequirements, SensitivitySample, VjpQuery,
};

pub use series::{
    EulerMaclaurinProvider, SeriesError, SeriesMethod, SeriesOptions, SeriesProblem, SeriesReport,
    SeriesRequirements, SeriesResult, SeriesState, SeriesSummation, SeriesTermination, SeriesTerms,
    SeriesWorkspace, sum_series,
};

pub use weighted::{
    ContourPlan, EndpointWeight, FourierDomain, FourierTailOptions, FourierTailPlan,
    FourierTailWorkspace, FourierWeight, WeightedQuadratureOptions, WeightedQuadraturePlan,
    WeightedQuadratureWorkspace, integrate_fourier, integrate_piecewise,
};

pub use workspace::IntegrationRequirements;
