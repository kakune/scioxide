//! Automatic differentiation for `f32` and `f64` by number types.
//!
//! Write the function once against [`Real`], then choose the number. The same
//! generic code prices on plain floats, runs on SIMD [`Lanes`], carries tangents in
//! forward mode and records a [`Tape`] in reverse mode:
//!
//! | Number | Use |
//! |---|---|
//! | `f32`, `f64` | plain evaluation, debugging, unit tests |
//! | [`Lanes<S, W>`](Lanes) | `W` independent values per operation (SIMD) |
//! | [`Dual<V>`](Dual) | one directional derivative |
//! | [`MultiDual<V, N>`](MultiDual) | `N` directional derivatives at once |
//! | [`Jet<V, N>`](Jet) | normalized Taylor coefficients `cₖ = (1/k!)·dᵏ/dtᵏ` |
//! | [`Var<'t, V>`](Var) | reverse mode on a [`Tape`] |
//!
//! Every wrapper takes any inner number: `Dual<Lanes<f64, 8>>` is one tangent of
//! eight lanes, `Dual<Dual<f64>>` a second derivative, `Var<Dual<f64>>` forward
//! over reverse for Hessian-vector products.
//!
//! ```
//! use scioxide_ad::{Dual, Jet, Lanes, Real, Tape, Var};
//! use scioxide_core::ExecutionContext;
//!
//! fn model<R: Real>(x: R) -> R {
//!     (x * x).sin() + x.exp() * 0.5
//! }
//! let derivative = |x: f64| 2.0 * x * (x * x).cos() + x.exp() * 0.5;
//! assert_eq!(model(Lanes::<f64, 8>::splat(0.3)).lane(7), Some(model(0.3)));
//! assert!((model(Dual::variable(0.3)).derivative() - derivative(0.3)).abs() < 1e-15);
//! assert!((model(Jet::<f64, 3>::variable(0.3)).coefficients()[1] - derivative(0.3)).abs() < 1e-15);
//! let cx = ExecutionContext::sequential();
//! let mut g = [0.0];
//! Tape::<f64>::default().gradient_into(|x| Ok(model(x[0])), &[0.3], &mut g, cx)?;
//! let mut lanes = [Lanes::<f64, 8>::splat(0.0)];
//! Tape::default().gradient_into(|x| Ok(model(x[0])), &[Lanes::splat(0.3)], &mut lanes, cx)?;
//! let mut second = [Dual::new(0.0, 0.0)];
//! Tape::default().gradient_into(|x: &[Var<'_, Dual<f64>>]| Ok(model(x[0])), &[Dual::variable(0.3)], &mut second, cx)?;
//! assert_eq!((lanes[0].lane(0), second[0].value()), (Some(g[0]), g[0]));
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Entry points
//!
//! - One-shot transforms: [`grad`], [`value_and_grad`], [`vjp`], [`jacobian`],
//!   [`hvp`], [`hessian`], [`sparse_jacobian`], [`sparse_hessian`] (reverse mode),
//!   [`jvp`] and [`jet`] (forward mode). They use default options and allocate
//!   their results.
//! - Reusable reverse mode: [`Tape::gradient_into`], [`Tape::linearize`] and the
//!   products of a [`Linearization`] (`vjp_into`, `jvp_into`, `jacobian_into`,
//!   structural patterns, sparse Jacobians, a [`JacobianOperator`]);
//!   `Tape<Dual<S>>` adds [`Tape::hvp_into`], sparse Hessians and a
//!   [`HessianOperator`].
//! - Batches: [`batch_mean`] and [`batch_mean_and_grad`] average a
//!   [`BatchFunction`] over the rows of a table in one call; [`BatchPlan`] sums or
//!   averages it over many independent items (Monte Carlo paths, per-point
//!   gradients, partially separable sums) with per-item inputs, `W` items per lane
//!   pack, with and without derivatives, sequentially or in parallel with identical
//!   results. Items come from any [`BatchData`] source: a slice, `Vec` or array
//!   table, or a closure through [`BatchDataFn`].
//! - Structured operations, valid for every number type: [`dot`], [`gemv_into`],
//!   [`gemm_into`], [`solve_into`], [`cholesky_into`], [`qr_into`],
//!   [`least_squares_into`], [`logsumexp`], [`softmax_into`], [`log_softmax_into`],
//!   [`implicit_solution_into`] and [`checkpointed_loop`]; caller rules through
//!   [`ExternalRule`], [`Real::map_scalar`] and [`Real::combine`]; complex values
//!   through [`ComplexElementary`].
//!
//! # Rules and conventions
//!
//! Values are IEEE 754 in a fixed operation order: no operation checks its
//! arguments, NaN and infinity propagate, and every wrapper takes its value from its
//! [`Primal`](Real::Primal), so a value never depends on the derivative mode.
//! `Dual`, `MultiDual` and `Var` apply the chain rule with the partials of each
//! function evaluated in `Primal` arithmetic, so nesting differentiates the
//! evaluated form. Where a form depends on the regime (`atan`, `tanh`, `asinh`,
//! `acosh`, `hypot`, `atan2`), both forms are evaluated and chosen with `select` on
//! base comparisons. The fourteen stable transforms (`softplus` … `tan_degrees`)
//! take their value from the `scioxide_core` kernels (NaN on a kernel error) and
//! their first and second derivatives from formulas evaluated in `f64`. The binary
//! `logaddexp`, `xlogy`, `xlog1py` and `powm1` take their value from kernels too,
//! so a kernel error is NaN where IEEE arithmetic would give a limit:
//! `powm1(0, y < 0)`, `powm1(∞, 0)` and `logaddexp(∞, ∞)` are NaN. `Sum` of a
//! number type starts from `−0`, as `f64`'s does, so an empty `Iterator::sum` is
//! `−0`; the reductions of this crate ([`dot`], [`gemm_into`]) return `+0` for
//! empty input. `Jet`
//! computes coefficients by Taylor recurrences in `V` arithmetic; `atan2` and
//! `combine` retain coefficients through `c₁`, and `exprel`, `sinc`, `sincpi` and
//! `map_scalar` through `c₂`; coefficients above those orders are NaN; [`implicit_solution_into`] builds its
//! result with `combine` too.
//!
//! **Products.** [`Real::azmul`] is the absolute-zero product: `a·b`, except that
//! an exactly zero operand gives `+0`. Tape sweeps use it for every update, and
//! `Dual` and `MultiDual` use it to apply every partial except those of `×`, scalar
//! factors (`x·c`, `x/c`, `to_degrees`, `to_radians`), `fract`, `exp`, `exp2`,
//! `exp_m1`, `sin`, `cos`, `sin_cos`, `sinh`, `cosh`, `tanh`, `atan` and `asinh`,
//! which use IEEE products. `Jet` uses IEEE products in its Taylor recurrences and
//! `azmul` in the kink rules of `abs`, `min`, `max` and `%`. Consequently
//! `sqrt(max(v, 0))` at `v < 0` has derivative `+0` in `Dual`, `MultiDual` and
//! reverse mode; its first `Jet` coefficient is NaN. In addition,
//! in the rules with IEEE products an infinite partial times an exactly zero tangent
//! is NaN in forward mode and `0` in a sweep. Products are a separate multiply and
//! add in chain-rule accumulation and Taylor recurrences. The scalar `sinc`
//! derivative kernels use explicitly fused products to recover rounding residuals;
//! the fused operation has the same semantics with and without hardware FMA.
//!
//! ## Kinks
//!
//! Every number type uses the midpoint of the one-sided derivatives at a kink, and
//! kink rules select components instead of multiplying:
//!
//! | Function | Derivative | Kink |
//! |---|---|---|
//! | `abs` | `sign x`, 0 at 0 | `x = 0` |
//! | `min`, `max` | 1 for the returned operand, ½ each at a tie; 0 for a NaN operand | `a = b` |
//! | `floor`, `ceil`, `round`, `trunc`, `signum` | 0 | integers, half-integers (`round`), non-zero integers (`trunc`), 0 (`signum`) |
//! | `fract`, `%` | 1; `(1, −trunc(a/b))`, `(1, −(q − ½·sign q))` at `a/b = q` | `x` or `a/b` a non-zero integer |
//! | `hypot`, `atan2` | `(0, 0)` at the origin | the origin; for `atan2` also `a = 0`, `b < 0` |
//!
//! `select` and comparisons are not kinks. A tape counts one hit per operation and
//! lane in which an active argument sits on a kink; [`KinkPolicy::Error`] turns a non-zero count
//! into `DomainError` (argument `point`, reason `Undefined`) at the transform
//! boundary. Forward types apply the convention only, and so does the body of a
//! [`checkpointed_loop`], which runs on values while the loop is recorded.
//!
//! **Non-finite values.** Number types follow IEEE 754. Transforms take a
//! [`NonFinitePolicy`]: `Propagate` (the default) writes what was computed; `Error`
//! rejects a non-finite point before the function runs (`NonFinite`, origin
//! `Input`) and a non-finite value or derivative before anything is written (origin
//! `Output`).
//!
//! # Tapes
//!
//! A [`Tape`] holds plain vectors: `u64` statement words (one per statement of at
//! most two arguments, more for `dot`, `combine`, lane `gather` and call
//! statements), full and scalar partial columns, an adjoint per word and a call
//! table. A transform builds
//! a recorder on its stack whose columns are cell views of those vectors, so
//! operators on [`Var`] append statements through shared references; there is no
//! thread-local or global state. An operation whose operands are all passive
//! records nothing; `x ± c` keeps the slot of `x`; `−x`, `x·c` and `x/c` record a
//! unary statement with a constant partial; `x·y` records both operands as partials;
//! elementary functions record their partial, except `floor`, `ceil`, `round`,
//! `trunc` and `signum`, which return passive values and record nothing (their
//! kinks are still counted); `dot` and `combine` record one statement. `select` and
//! `gather` return the chosen `Var` on scalar tapes and record nothing; on lane
//! tapes `select` records one statement when the lanes choose different sides and
//! `gather` one statement when an entry is active. `solve_into`,
//! `checkpointed_loop` and [`Var::external`] record one call statement. A reverse
//! sweep is one loop over the words from the end; a tangent sweep runs forward;
//! both call `context.check()` before and after a sweep and every 1,024 words,
//! and transforms and the products of
//! a [`Linearization`] call it on entry and before every sweep.
//!
//! A recording that does not fit is still counted; the transform grows the tape to
//! the counted size and runs the function again, so a function runs at most twice
//! per transform call and must be deterministic. Warm transforms on a tape that has
//! recorded the same function allocate nothing, except in structured operations
//! that allocate scratch ([`solve_into`], the factorizations, implicit solutions and
//! checkpointed loops). Outputs are unchanged when a transform fails.
//! `Tape::default()` is `Tape::new(TapeOptions::new())`: the default policies and
//! no byte limit; [`TapeOptions`] sets the others.
//!
//! # Derivatives in solver callbacks
//!
//! A solver hook that takes `&self` or `Fn` keeps its tape in a `RefCell` and
//! calls [`Tape::gradient_into`], [`Tape::hvp_into`] on a `Tape<Dual<f64>>`, or
//! [`Tape::linearize`] followed by [`Linearization::jacobian_into`]; after the
//! first call the hook allocates nothing. A one-shot such as [`grad`] inside a hook
//! allocates a tape and runs the function twice per call. Forward mode ([`Dual`],
//! [`MultiDual`]) needs no tape. A solver calls the hook only if its derivative
//! options select that source.
//!
//! ```
//! use std::cell::RefCell;
//! use scioxide_ad::{Real, Tape};
//! use scioxide_core::{ExecutionContext, SciOxideError};
//!
//! fn rosenbrock<R: Real>(x: &[R]) -> R { (-x[0] + 1.0).powi(2) + (x[1] - x[0] * x[0]).powi(2) * 100.0 }
//! /// Stands in for a solver that asks a hook for gradients.
//! fn descend(hook: impl Fn(&[f64], &mut [f64]) -> Result<(), SciOxideError>) -> Result<[f64; 2], SciOxideError> {
//!     let (mut x, mut g) = ([0.5, 0.5], [0.0; 2]);
//!     for _ in 0..3 { hook(&x, &mut g)?; x = [x[0] - 1e-3 * g[0], x[1] - 1e-3 * g[1]]; }
//!     Ok(x)
//! }
//! let (tape, cx) = (RefCell::new(Tape::<f64>::default()), ExecutionContext::sequential_native());
//! let x = descend(|x, g| tape.borrow_mut().gradient_into(|x| Ok(rosenbrock(x)), x, g, cx).map(drop))?;
//! assert!(rosenbrock(&x) < rosenbrock(&[0.5, 0.5]));
//! # Ok::<(), SciOxideError>(())
//! ```
//!
//! # Accuracy
//!
//! - Values keep the envelopes of the `scioxide_core` elementary functions; the eight
//!   lane kernels (`exp`, `exp_m1`, `ln`, `ln_1p`, `sin`, `cos`, `tanh`, `powf`) are
//!   within 4 ulps and can differ from the scalar function in the last ulps; every
//!   other lane operation equals the scalar function bit for bit, and results are
//!   identical for every lane width.
//! - First partials of the unary functions are within 64 ulps of the
//!   high-precision references (6 ulps for `log10` and `tanh` near their boundaries,
//!   2 ulps for the constant-scaled `log2`, `log10`, `exp2`); first and second
//!   derivatives of the stable transforms within 64 ulps.
//! - First partials of binary functions are within 64 ulps and second partials by
//!   nesting within 256 ulps, on points where the arguments, the value and the
//!   partials are zero or have magnitude in `[√MIN_POSITIVE, √MAX]`; outside that
//!   range plain IEEE evaluation can overflow or underflow.
//! - Taylor coefficients `cₖ` are within `64·ε·(k + 1)·g` plus the smallest
//!   subnormal of the reference, where `g` bounds the growth of the coefficients, on
//!   the same restricted points.
//!
//! # Costs of the design
//!
//! - Lanes vectorize for the target features of the build, which Rust fixes at
//!   compile time: AVX2 needs `-C target-cpu=x86-64-v3` (or `native`), and a default
//!   x86-64 build uses SSE2, about 1.2–1.7× slower on lane-heavy code.
//! - A scalar tape on large element-wise code costs tens of nanoseconds per element;
//!   batch such code over lanes or use forward mode.
//! - A Hessian records the function once per direction (or per color for a sparse
//!   Hessian).
//! - Every number type evaluates in plain IEEE arithmetic without range scaling, so
//!   outside the ranges of the accuracy bounds above intermediate values can
//!   overflow or underflow where a range-safe formulation would not.
//!
//! # Features
//!
//! - `std` (default): enables the `std` features of the SciOxide dependencies. Without
//!   it the crate is `no_std` and needs `alloc`.
//!
//! # References
//!
//! - A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008.
//! - B. M. Bell, CppAD: absolute-zero multiplication (`azmul`).
//! - M. Giles and P. Glasserman, "Smoking adjoints: fast Monte Carlo Greeks",
//!   *Risk* 19(1), 2006.
#![cfg_attr(
    not(test),
    deny(
        clippy::disallowed_types,
        clippy::disallowed_macros,
        clippy::disallowed_methods
    )
)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![deny(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod arrays;
mod batch;
mod checkpoint;
mod complex;
mod forward;
mod lanes;
mod options;
mod real;
mod reverse;
mod rules;
mod scalar;
mod stable;
mod structured;
mod support;

pub use batch::{
    BatchData, BatchDataFn, BatchEstimate, BatchFunction, BatchMode, BatchOptions, BatchOutputs,
    BatchPlan, BatchReduction, BatchRequirements, BatchWorkspace, batch_mean, batch_mean_and_grad,
};
pub use checkpoint::{CheckpointAction, CheckpointSchedule};
pub use complex::ComplexElementary;
pub use forward::{Dual, Jet, MultiDual, jet, jvp};
pub use lanes::{LaneMask, Lanes};
pub use options::{DerivativeMode, JacobianMode, KinkPolicy, NonFinitePolicy};
pub use real::{BaseReal, Mask, Real};
pub use reverse::{
    ExternalRule, HessianOperator, JacobianOperator, Linearization, Tape, TapeOptions, TapeReport,
    TapeRequirements, TapeValue, Var, grad, hessian, hvp, jacobian, sparse_hessian,
    sparse_jacobian, value_and_grad, vjp,
};
pub use structured::{
    LoopBody, Residual, checkpointed_loop, cholesky_into, dot, gemm_into, gemv_into,
    implicit_solution_into, least_squares_into, log_softmax_into, logsumexp, qr_into, softmax_into,
    solve_into,
};
