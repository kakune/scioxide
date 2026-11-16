//! Numerical derivatives by finite differences, extrapolation and complex step.
//!
//! The crate approximates first and second derivatives, gradients, Jacobians,
//! Hessians and Jacobian-vector products of a function that you can only
//! evaluate. It offers fixed-step forward and central differences, adaptive
//! differences with Ridders' or Richardson extrapolation and an error estimate,
//! sparse Jacobians from a column coloring, finite-difference weights on nodes
//! you choose, and the complex-step method. It also provides
//! [`DerivativeStrategy`], which solvers use to choose between an analytic
//! derivative, automatic differentiation and finite differences.
//!
//! # Quick start
//!
//! ```
//! use scioxide_differentiate::{AdaptiveDifference, derivative, gradient};
//!
//! // f(x) = x³, so f′(2) = 12.
//! let slope = derivative(|x: f64| Ok(x * x * x), 2.0)?;
//! assert!((slope - 12.0).abs() < 1e-8);
//!
//! // f(x) = x₀² + x₁², so ∇f(1, 2) = (2, 4).
//! let g = gradient(|p: &[f64]| Ok(p[0] * p[0] + p[1] * p[1]), &[1.0, 2.0])?;
//! assert!((g[0] - 2.0).abs() < 1e-8 && (g[1] - 4.0).abs() < 1e-8);
//!
//! // The adaptive method also estimates the error.
//! let result = AdaptiveDifference::default().derivative(|x: f64| Ok(x * x * x), 2.0)?;
//! let (value, error) = (result.values().unwrap()[0], result.errors().unwrap()[0]);
//! assert!((value - 12.0).abs() < 1e-8 && error < 1e-6);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Task | Entry points | Method, cost and sampling radius |
//! |---|---|---|
//! | Derivative, gradient, Jacobian, Hessian or `J v` for general use | [`derivative`], [`second_derivative`], [`gradient`], [`jacobian`], [`hessian`], [`jvp`] | central differences with the default step; `2n + 1` evaluations for a gradient or Jacobian, samples within `ε^(1/3)·max(1, |xᵢ|)` |
//! | Bounds, another scheme or step, a known `f(x)`, workspaces | [`FiniteDifference`] | forward or central differences with a fixed step; `n + 1` or `2n + 1` evaluations |
//! | Derivative with an error estimate | [`AdaptiveDifference`] | Ridders' or Richardson extrapolation over up to `L = 16` shrinking steps from `0.5·max(1, |xᵢ|)`; about `2L` evaluations per entry |
//! | Sparse Jacobian with a known pattern | [`sparse_jacobian`], [`ColoredJacobianPlan`] | two perturbations per column color and level, then extrapolation; at most `1 + 2cL` evaluations for `c` colors, including with bounds |
//! | Derivative on nodes you choose | [`physical_difference_derivative`], [`PhysicalDifferencePlan`], [`RelativeDifference`] | Fornberg weights on arbitrary nodes; one evaluation per node |
//! | Finite-difference weights only | [`FiniteDifferenceStencil`] | Fornberg weights |
//! | Function with an analytic extension | [`complex_step_derivative`], [`ComplexStep`], [`ComplexStepPlan`], [`RelativeComplexStep`] | complex step `Im f(x + ih)/h` with `h = ε`; `n + 1` evaluations for a gradient |
//! | Derivative source in a solver | [`DerivativeStrategy`] | analytic, automatic or finite differences, in a fixed order |
//!
//! The one-shot functions take the function and the point and return the
//! derivative. Fixed-step differences are the cheapest but return no error
//! estimate. Adaptive differences cost more, report how accurate each entry
//! is, and start with a large step: near the edge of the domain of `f`, such as
//! `ln x` near zero, pass bounds or a smaller step. The complex step has no
//! subtractive cancellation and gives first derivatives accurate to rounding
//! error, but it needs a callback that accepts complex arguments. For
//! derivatives exact up to rounding of functions written as code, use
//! automatic differentiation in `scioxide-ad` instead.
//!
//! # Conventions
//!
//! The method values [`FiniteDifference`], [`AdaptiveDifference`] and
//! [`ComplexStep`] hold settings only and name the operation by their methods;
//! the number of inputs `n` comes from the point `x`, and the number of
//! callback outputs `m` is an argument of the Jacobian and Jacobian-vector
//! product. Results are row-major:
//!
//! | Operation | Output |
//! |---|---|
//! | `derivative`, `second_derivative` | `f′(x)` or `f″(x)` of `f: ℝ → ℝ` |
//! | `gradient` | `∇f(x)` of `f: ℝⁿ → ℝ`, length `n` |
//! | `jacobian` | `m × n`; entry `i·n + j` is `∂fᵢ/∂xⱼ` |
//! | `hessian` | symmetric `n × n`; entry `i·n + j` is `∂²f/∂xᵢ∂xⱼ` |
//! | `jvp` | `J(x) v`, length `m` |
//!
//! The node plans take a [`DerivativeOperation`] instead. One-shot callbacks
//! have the following forms; none accepts [`Contextual`] or [`Parallel`].
//!
//! | One-shot functions | Callback |
//! |---|---|
//! | `derivative`, `second_derivative`, `physical_difference_derivative`, `complex_step_derivative` | scalar point → `Result` containing a scalar |
//! | `gradient`, `hessian` | point slice → `Result` containing a scalar |
//! | `jacobian`, `jvp`, `sparse_jacobian` | point slice and mutable output slice → `Result<()>` |
//!
//! Scalar methods likewise return a scalar from the callback. Vector method
//! callbacks write all `m` values into the output slice.
//! Every value must be finite. Coordinates are real (`f32` or `f64`); adaptive
//! and node methods also accept complex callback values and then
//! differentiate the real and imaginary parts with respect to the real
//! coordinates. A closure runs sequentially. [`Contextual`] passes the solve
//! context to the callback, and [`Parallel`] evaluates independent samples on
//! the executor of the [execution context](scioxide_core#execution-contexts).
//!
//! The default fixed step for coordinate `xᵢ` is `h = c · max(1, |xᵢ|)`, where
//! `c` balances truncation against rounding error: `√ε` for forward first
//! derivatives, `ε^(1/3)` for central first and forward second derivatives,
//! and `ε^(1/4)` for central second derivatives (Nocedal and Wright, §8.1).
//! Adaptive evaluations start at `h = 0.5 · max(1, |xᵢ|)`. [`DifferenceStep`]
//! replaces either default. Weights are computed from the displacements that
//! survive rounding, not from the requested ones.
//!
//! Optional bounds ([`Bounds`](scioxide_core::Bounds)) restrict every sample
//! point. Near a bound a forward stencil reverses direction, and a central
//! stencil becomes a one-sided stencil of the same order that points into the
//! domain.
//!
//! # Results and termination
//!
//! The one-shot functions return the derivative, or an error for any
//! termination that is not a success, as
//! [`require_success`](DifferentiationReport::require_success) maps it. Every
//! other evaluation returns a [`DifferentiationReport`]. Its
//! [`termination`](DifferentiationReport::termination) is a
//! [`DifferentiationTermination`], and it counts callback evaluations and
//! iterations. Results are written only when
//! [`output_available`](DifferentiationReport::output_available) is true:
//!
//! - Fixed-step, node and complex-step evaluations end with `Completed`. This
//!   says that the stencil was evaluated, not that the result is accurate.
//! - Adaptive evaluations end with `Converged` when every estimated error meets
//!   the tolerance. When extrapolation stops early (`RoundoffLimit`,
//!   `RefinementLimit` or `StepSizeLimit`), the best estimates and their errors
//!   are still written, but [`is_success`](DifferentiationReport::is_success)
//!   is false.
//! - An evaluation or iteration limit, cancellation or deadline returns `Ok`
//!   with that termination and writes nothing.
//!
//! Owned forms return a [`DifferentiationResult`], whose values are `None` when
//! no output is available;
//! [`into_values`](DifferentiationResult::into_values) returns the values or
//! the error of `require_success`.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors), and an error leaves
//! the output unchanged ([Output on error](scioxide_core#output-on-error)). In
//! this crate:
//!
//! - `InvalidInput` (`self`) for method settings outside their range, such as
//!   adaptive levels outside 3 to 64 (`OutOfRange`), a complex step that is not
//!   positive (`UnsupportedConfiguration`), or a policy that the operation does
//!   not use (`Inconsistent`). Every operation checks the settings it uses.
//! - `InvalidInput` (`bounds` or `x`, `UnsupportedConfiguration`) when no fixed
//!   stencil fits at a coordinate, and (`x`, `OutOfRange`) when `x` lies outside
//!   the bounds.
//! - `DimensionMismatch` when the lengths of the bounds, direction, known base
//!   values, outputs or per-coordinate steps do not match the point.
//! - `NonFinite` with origin `Input` for a NaN or infinite point, direction or
//!   complex step. Nonfinite `DifferenceStep` magnitudes instead return
//!   `InvalidInput` (`OutOfRange`) at construction.
//! - `InvalidInput` (`function`, `UnsupportedConfiguration`) from the complex
//!   step if `f(x)` has a nonzero imaginary part at the real point `x`.
//! - `DifferentiationFailure` for a failure while sampling the callback. It
//!   records the sample location and the evaluation count, and its `source()`
//!   is the error returned by the callback, `NonFinite` with origin `Callback`
//!   for a NaN or infinite callback value, or `NumericalInstability` for a
//!   sample point or stencil sum outside the finite range.
//!
//! # Accuracy
//!
//! With the default steps and a smooth function of moderate size, the error of
//! a fixed-step derivative is of order `√ε` for forward and `ε^(2/3)` for
//! central first derivatives: `1.5·10⁻⁸` and `3.7·10⁻¹¹` for `f64`,
//! `3.5·10⁻⁴` and `2.4·10⁻⁵` for `f32`. Central second derivatives have error
//! of order `√ε`, and forward second derivatives of order `ε^(1/3)`.
//!
//! Adaptive evaluations report an absolute error estimate for each entry,
//! taken from the extrapolation table plus a bound on the propagated rounding
//! error. It is an estimate, not a bound. An entry converges when its estimate
//! is at most `atol + rtol·|D|` on two consecutive levels and the best estimate
//! also meets its tolerance, where `D` is the entry. At least three levels are
//! needed; the default is `atol = rtol = √ε`.
//!
//! The complex step evaluates `Im f(x + ih) / h`, which has no subtractive
//! cancellation. Its truncation error is `O(h²)`, so with the default `h = ε`
//! the error of a first derivative is the rounding error of evaluating `f`
//! itself. Second derivatives combine complex-step gradients with a real
//! stencil and have the usual finite-difference error.
//!
//! # Reusing plans and workspaces
//!
//! Plans and workspaces follow the
//! [shared rules](scioxide_core#plans-and-workspaces). A method value has no
//! plan: size its workspace with the method and the shape of the evaluation,
//! for example
//! `FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(&method, DerivativeShape::jacobian(n, m)))`,
//! and likewise for [`AdaptiveDifferenceWorkspace`] and
//! [`ComplexStepWorkspace`]. The plans and the node-offset method values fix
//! their sizes, so their workspaces come from `workspace(())`. Then call the
//! `_into` forms at each point with an
//! [`ExecutionContext`](scioxide_core::ExecutionContext) or a child
//! [`SolveScope`](scioxide_core::SolveScope). With a sufficient workspace, `_into`
//! forms do not allocate, including when an adaptive evaluation stops at
//! `StepSizeLimit`. One workspace serves every evaluation of the same family
//! whose requirements it meets. The `max_workspace_bytes` setting limits the
//! capacity the workspace holds, including capacity left from earlier, larger
//! evaluations. Every workspace exposes `retained_bytes()` to inspect this
//! capacity; if it exceeds the limit, use a new workspace. A
//! failed `try_reserve` can leave some buffers grown, so call `check` again
//! before evaluating. Owned forms run with
//! [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
//!
//! Each callback call counts as one function evaluation and each stencil level
//! or derivative entry as one iteration toward the
//! [solve budgets](scioxide_core#solve-accounting) set with `with_budgets`.
//!
//! # Feature flags
//!
//! - `std` (default): enables `std` in `scioxide-core`.
//!
//! Without default features the crate is `no_std` and needs only `alloc`; all
//! methods are available.
//!
//! # References
//!
//! - B. Fornberg, "Generation of finite difference formulas on arbitrarily
//!   spaced grids", *Math. Comp.* 51(184), 1988, 699–706.
//! - C. J. F. Ridders, "Accurate computation of F′(x) and F′(x)F″(x)",
//!   *Adv. Eng. Softw.* 4(2), 1982, 75–76.
//! - W. Squire and G. Trapp, "Using complex variables to estimate derivatives
//!   of real functions", *SIAM Rev.* 40(1), 1998, 110–112.
//! - J. R. R. A. Martins, P. Sturdza and J. J. Alonso, "The complex-step
//!   derivative approximation", *ACM Trans. Math. Softw.* 29(3), 2003,
//!   245–262.
//! - A. R. Curtis, M. J. D. Powell and J. K. Reid, "On the estimation of
//!   sparse Jacobian matrices", *J. Inst. Math. Appl.* 13, 1974, 117–119.
//! - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
//!   2006, §8.1.
//! - W. H. Press et al., *Numerical Recipes*, 3rd ed., Cambridge, 2007, §5.7.
#![forbid(unsafe_code)]
#![deny(unreachable_pub)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
#[cfg(test)]
extern crate self as scioxide_differentiate;
#[cfg(all(test, not(feature = "std")))]
extern crate std;
mod adaptive;
pub use adaptive::{
    AdaptiveDifference, AdaptiveDifferenceMethod, AdaptiveDifferenceRequirements,
    AdaptiveDifferenceWorkspace, ColoredJacobianPlan, ColoredJacobianRequirements,
    ColoredJacobianWorkspace,
};
mod complex;
pub use complex::{
    ComplexStep, ComplexStepPlan, ComplexStepRequirements, ComplexStepWorkspace,
    RelativeComplexStep,
};
mod convenience;
pub use convenience::{
    complex_step_derivative, derivative, gradient, hessian, jacobian, jvp,
    physical_difference_derivative, second_derivative, sparse_jacobian,
};
mod fixed;
pub use fixed::{
    FiniteDifference, FiniteDifferenceRequirements, FiniteDifferenceWorkspace,
    FixedCoordinatePolicy,
};
mod nodes;
pub use nodes::{
    FiniteDifferenceStencil, NodeDifferenceOptions, PhysicalDifferencePlan,
    PhysicalDifferenceRequirements, PhysicalDifferenceWorkspace, RelativeDifference,
    RelativeDifferenceRequirements, RelativeDifferenceWorkspace,
};
mod options;
pub use options::{
    DerivativeOperation, DerivativeRequest, DerivativeShape, DifferenceScheme, DifferenceStep,
    ZeroDirectionPolicy,
};
mod outcome;
pub use outcome::{
    DerivativeSource, DifferentiationReport, DifferentiationResult, DifferentiationTermination,
};
mod samples;
mod sampling;
pub use sampling::{Contextual, Parallel, SampleFunction};
mod selection;
pub use selection::{DerivativeCandidate, DerivativeStrategy};
mod stencil;
mod validation;
mod workspace;
