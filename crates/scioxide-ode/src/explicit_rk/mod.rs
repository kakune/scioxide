//! Explicit Runge–Kutta methods for nonstiff systems.
//!
//! Fixed-step methods without error estimate: [`ForwardEuler`] (order 1),
//! [`ExplicitMidpoint`] and [`Heun`] (order 2) and [`ClassicalRk4`] (order 4).
//! Adaptive embedded pairs: [`Rkf45`] 5(4), [`DormandPrince5`] 5(4),
//! [`Verner65`] 6(5) and [`DormandPrince853`] 8(5,3).
//! Generates a public method type; its coefficients and optional extensions are method data.
macro_rules! explicit_solver {
    ($(#[$meta:meta])* $name:ident, $stages:expr, $order:expr, $adaptive:literal) => {
        explicit_solver!($(#[$meta])* $name, $stages, $order, $adaptive, $stages);
    };
    ($(#[$meta:meta])* $name:ident, $stages:expr, $order:expr, $adaptive:literal, $main:expr) => {
        explicit_solver!($(#[$meta])* $name, $stages, $order, $adaptive, $main, false);
    };
    ($(#[$meta:meta])* $name:ident, $stages:expr, $order:expr, $adaptive:literal, $main:expr, $corrected:literal) => {
        $(#[$meta])*
        #[doc = concat!(
            "\n# Examples\n\n```\nuse scioxide_core::SciOxideError;\n",
            "use scioxide_linalg::Vector;\n",
            "use scioxide_ode::{", stringify!($name), ", OdeFn, OdeOptions, OdeMethod, OdeProblem, OdeState};\n",
            "\n// y′ = −y, y(0) = 1, integrated to t = 0.5.\n",
            "let system = OdeFn::new(1, |_: f64, y: &[f64], f: &mut [f64]| {\n",
            "    f[0] = -y[0];\n",
            "    Ok(())\n",
            "})?;\n",
            "let plan = (", stringify!($name), "::<f64>::new(), OdeOptions::new());\n",
            "let initial = OdeState::new(0.0, Vector::from_vec(vec![1.0]));\n",
            "let solution = plan.0.solve(&OdeProblem::new(&system, 0.5), &plan.1, initial)?;\n",
            "assert!(solution.report().is_success());\n",
            "# Ok::<(), SciOxideError>(())\n```"
        )]
        #[derive(Debug, Clone)]
        pub struct $name<
            S: scioxide_core::RealScalar,
            T: scioxide_core::NumericScalar<Real = S> = S,
        > {
            tableau: super::ButcherTableau<S, $stages, $main, $adaptive, $corrected>,
            marker: core::marker::PhantomData<T>,
        }
        impl<S: scioxide_core::RealScalar, T: scioxide_core::NumericScalar<Real = S>> $name<S, T> {
/// Creates the method.
            #[must_use]
            pub fn new() -> Self {
                Self {
                    tableau: tableau(),
                    marker: core::marker::PhantomData,
                }
            }
        }
        impl<S: scioxide_core::RealScalar, T: scioxide_core::NumericScalar<Real = S>> Default
            for $name<S, T>
        {
            fn default() -> Self {
                Self::new()
            }
        }
        impl<
            S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
            T: scioxide_core::NumericScalar<Real = S>,
        > crate::stepper::Stepper<S, T> for $name<S, T>
        {
            const ADAPTIVE: Option<bool> = Some($adaptive);
            const BORROWS_BASE_STATE: bool = true;
            const CHECKED_RESULT: bool = true;

            type Workspace = crate::workspace::ExplicitOdeWorkspace<S, T>;
            fn info(&self) -> crate::stepper::MethodInfo {
                let dense = self.tableau.dense();
                crate::stepper::MethodInfo {
                    name: stringify!($name),
                    algorithm: scioxide_core::algorithm_id!("ode", stringify!($name), 1),
                    order: $order,
                    initial_step_order: $order,
                    controller: if $adaptive {
                        crate::stepper::DefaultController::Integral
                    } else {
                        crate::stepper::DefaultController::ProportionalIntegral
                    },
                    dense_output: dense,
                    events: dense,
                    adaptive: $adaptive,
                    mass_matrices: false,
                    injectable: false,
                }
            }
            fn stability(&self) -> crate::OdeLinearStability<S> {
                crate::OdeLinearStability::bounded(STABILITY.0, STABILITY.1)
            }
            fn accepts_initial_rhs(&self) -> bool {
                true
            }
            fn fits(&self, dimension: usize, _provided: bool, workspace: &Self::Workspace) -> bool {
                self.tableau.fits(dimension, workspace)
            }
            fn reserve(
                &self,
                dimension: usize,
                _provided: bool,
                workspace: &mut Self::Workspace,
            ) -> Result<(), scioxide_core::SciOxideError> {
                self.tableau.prepare_workspace(dimension, workspace)
            }
            fn begin(
                &self,
                _dimension: usize,
                _provided: bool,
                workspace: &mut Self::Workspace,
            ) -> Result<(), scioxide_core::SciOxideError> {
                workspace.reset_epoch();
                Ok(())
            }
            fn accept(
                &self,
                workspace: &mut Self::Workspace,
            ) -> Result<(), scioxide_core::SciOxideError> {
                workspace.accept_step();
                Ok(())
            }
            fn discard(
                &self,
                workspace: &mut Self::Workspace,
            ) -> Result<(), scioxide_core::SciOxideError> {
                if $adaptive {
                    workspace.discard_step();
                } else {
                    workspace.reset_epoch();
                }
                Ok(())
            }
            fn step<'w, System: crate::OdeSystem<S, T> + ?Sized>(
                &self,
                system: &System,
                t: S,
                y: &mut scioxide_linalg::Vector<T>,
                h: S,
                request: crate::stepper::StepRequest<'_, S, T>,
                workspace: &'w mut Self::Workspace,
                cx: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<crate::result::StepResult<'w, S, T>, scioxide_core::SciOxideError> {
                super::perform_explicit_step(
                    system,
                    t,
                    y,
                    h,
                    &self.tableau,
                    request,
                    workspace,
                    request.initial_rhs,
                    cx,
                )
            }
            fn complete_dense<'w, System: crate::OdeSystem<S, T> + ?Sized>(
                &self,
                system: &System,
                t: S,
                h: S,
                request: crate::stepper::StepRequest<'_, S, T>,
                workspace: &'w mut Self::Workspace,
                cx: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<
                &'w crate::dense_output::DenseOutputStorage<S, T>,
                scioxide_core::SciOxideError,
            > {
                super::complete_extension(system, t, h, &self.tableau, request, workspace, cx)
            }
        }
    };
}

pub(crate) mod dormand_prince5;
pub(crate) mod dormand_prince853;
pub(crate) mod euler;
pub(crate) mod heun;
pub(crate) mod midpoint;
pub(crate) mod rk4;
pub(crate) mod rkf45;
pub(crate) mod verner65;

pub use dormand_prince5::DormandPrince5;
pub use dormand_prince853::DormandPrince853;
pub use euler::ForwardEuler;
pub use heun::Heun;
pub use midpoint::ExplicitMidpoint;
pub use rk4::ClassicalRk4;
pub use rkf45::Rkf45;
pub use verner65::Verner65;

use crate::dense_output::DenseExtension;
use crate::result::StepResult;
use crate::safety::prepared_stage_time;
use crate::stepper::StepRequest;
use crate::traits::{OdeSystem, evaluate_rhs_at_finite_state};
use crate::workspace::ExplicitOdeWorkspace;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

/// Source of the endpoint derivative `f(tₙ₊₁, yₙ₊₁)` of a step.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Endpoint {
    /// Evaluated only when an accepted step's interpolant is read; it then serves as
    /// the next base derivative.
    Deferred,
    /// The main stage of this index (first same as last).
    Stage(usize),
    /// Evaluated with every attempt.
    Evaluate,
}

/// Immutable scalar coefficients shared by fixed, embedded and extended RK methods.
/// Only the strictly lower triangular prefix of each stage row is active.
/// MAIN_STAGES <= STAGES; remaining rows hold an endpoint and optional extension.
/// Both counts are static so sharing the kernel preserves loop specialization.
#[derive(Debug, Clone)]
pub(crate) struct ButcherTableau<
    S: RealScalar,
    const STAGES: usize = 7,
    const MAIN_STAGES: usize = STAGES,
    const ADAPTIVE: bool = false,
    const CORRECTED: bool = false,
> {
    pub c: [S; STAGES],
    pub a: [[S; STAGES]; STAGES],
    pub b: [S; STAGES],
    pub(crate) error: [S; STAGES],
    correction: [S; STAGES],
    error_order: Option<usize>,
    endpoint: Endpoint,
    extension: Option<DenseExtension>,
    pub dense: [[S; STAGES]; 4],
}

/// Expand constant lower-triangular rows at compile time, never during stepping.
pub(crate) const fn square<const N: usize>(rows: [&[f64]; N]) -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];
    let mut i = 0;
    while i < N {
        assert!(rows[i].len() == i);
        let mut j = 0;
        while j < i {
            result[i][j] = rows[i][j];
            j += 1;
        }
        i += 1;
    }
    result
}

impl<S: RealScalar, const N: usize, const MAIN: usize> ButcherTableau<S, N, MAIN> {
    /// All internal constants are finite and representable in both supported widths.
    /// IEEE conversion keeps the rounding from `f64` to the scalar type.
    fn new(c: [f64; N], a: [[f64; N]; N], b: [f64; N]) -> Self {
        Self {
            c: c.map(S::from_f64_ieee),
            a: a.map(|row| row.map(S::from_f64_ieee)),
            b: b.map(S::from_f64_ieee),
            error: [S::ZERO; N],
            correction: [S::ZERO; N],
            error_order: None,
            endpoint: Endpoint::Deferred,
            extension: None,
            dense: [[S::ZERO; N]; 4],
        }
    }
    fn with_embedded(self, lower: [f64; N], order: usize) -> ButcherTableau<S, N, MAIN, true> {
        let weights = core::array::from_fn(|i| self.b[i] - S::from_f64_ieee(lower[i]));
        self.with_error(weights, order)
    }
    fn with_error(self, weights: [S; N], order: usize) -> ButcherTableau<S, N, MAIN, true> {
        ButcherTableau {
            c: self.c,
            a: self.a,
            b: self.b,
            error: weights,
            correction: self.correction,
            error_order: Some(order),
            endpoint: self.endpoint,
            extension: self.extension,
            dense: self.dense,
        }
    }
}

impl<S: RealScalar, const N: usize, const MAIN: usize> ButcherTableau<S, N, MAIN, true> {
    fn with_correction(self, weights: [S; N]) -> ButcherTableau<S, N, MAIN, true, true> {
        ButcherTableau {
            c: self.c,
            a: self.a,
            b: self.b,
            error: self.error,
            correction: weights,
            error_order: self.error_order,
            endpoint: self.endpoint,
            extension: self.extension,
            dense: self.dense,
        }
    }
}

impl<S: RealScalar, const N: usize, const MAIN: usize, const ADAPTIVE: bool, const CORRECTED: bool>
    ButcherTableau<S, N, MAIN, ADAPTIVE, CORRECTED>
{
    fn with_dense(mut self, extension: DenseExtension, weights: [[f64; N]; 4]) -> Self {
        self.extension = Some(extension);
        self.dense = weights.map(|row| row.map(S::from_f64_ieee));
        self
    }
    fn with_endpoint(mut self, endpoint: Endpoint) -> Self {
        self.endpoint = endpoint;
        self
    }
    /// Whether the method has a continuous extension (dense output and events).
    const fn dense(&self) -> bool {
        self.extension.is_some()
    }
    /// Whether dense output waits for an accepted step whose interpolant is read: the
    /// extension uses stages beyond the main ones or a deferred endpoint derivative.
    const fn defers_dense(&self) -> bool {
        self.extension.is_some() && (MAIN < N || matches!(self.endpoint, Endpoint::Deferred))
    }
    /// Whether `workspace` already serves `dimension` with every stage of this tableau.
    fn fits<T: NumericScalar<Real = S>>(
        &self,
        dimension: usize,
        workspace: &ExplicitOdeWorkspace<S, T>,
    ) -> bool {
        workspace.fits(
            dimension,
            N,
            self.extension.map_or(0, DenseExtension::degree),
        )
    }
    fn prepare_workspace<T: NumericScalar<Real = S>>(
        &self,
        dimension: usize,
        workspace: &mut ExplicitOdeWorkspace<S, T>,
    ) -> Result<(), SciOxideError> {
        if CORRECTED {
            scioxide_core::backend::checked_mul(
                dimension,
                36,
                scioxide_core::quantity!("dop853_complete_workspace"),
            )?;
        }
        workspace.try_resize(
            dimension,
            N,
            self.extension.map_or(0, DenseExtension::degree),
        )
    }
}

/// Evaluate a stage using the same arithmetic for main and continuous-extension rows.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_stage<
    S: RealScalar,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
    const N: usize,
    const MAIN: usize,
    const ADAPTIVE: bool,
    const CORRECTED: bool,
>(
    sys: &System,
    t: S,
    y: &Vector<T>,
    dt: S,
    tableau: &ButcherTableau<S, N, MAIN, ADAPTIVE, CORRECTED>,
    stage: usize,
    workspace: &mut ExplicitOdeWorkspace<S, T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    crate::safety::combine_stages(
        Some(y.as_slice()),
        dt,
        &tableau.a[stage][..stage],
        &workspace.stages[..stage],
        workspace.work.as_mut_slice(),
    )?;
    evaluate_rhs_at_finite_state(
        sys,
        prepared_stage_time(t, dt, tableau.c[stage]),
        &workspace.work,
        &mut workspace.stages[stage],
        &mut workspace.rhs_candidate,
        request,
        cx,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn perform_explicit_step<
    'a,
    S: RealScalar,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
    const N: usize,
    const MAIN: usize,
    const ADAPTIVE: bool,
    const CORRECTED: bool,
>(
    sys: &System,
    t: S,
    output: &mut Vector<T>,
    dt: S,
    tableau: &ButcherTableau<S, N, MAIN, ADAPTIVE, CORRECTED>,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut ExplicitOdeWorkspace<S, T>,
    initial_rhs: Option<&[T]>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError> {
    if request.requires_dense_output() {
        explicit_step::<S, T, System, N, MAIN, ADAPTIVE, CORRECTED, true>(
            sys,
            t,
            output,
            dt,
            tableau,
            request,
            workspace,
            initial_rhs,
            cx,
        )
    } else {
        explicit_step::<S, T, System, N, MAIN, ADAPTIVE, CORRECTED, false>(
            sys,
            t,
            output,
            dt,
            tableau,
            request,
            workspace,
            initial_rhs,
            cx,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn explicit_step<
    'a,
    S: RealScalar,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
    const N: usize,
    const MAIN: usize,
    const ADAPTIVE: bool,
    const CORRECTED: bool,
    const DENSE: bool,
>(
    sys: &System,
    t: S,
    output: &mut Vector<T>,
    dt: S,
    tableau: &ButcherTableau<S, N, MAIN, ADAPTIVE, CORRECTED>,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut ExplicitOdeWorkspace<S, T>,
    initial_rhs: Option<&[T]>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError> {
    let y = request.base.map_or(&*output, |(_, base)| base);
    let endpoint_time = request.endpoint(t, dt)?;
    workspace.pending_endpoint = false;
    if let Some(rhs) = initial_rhs {
        workspace.reset_epoch();
        workspace.stages[0].as_mut_slice().copy_from_slice(rhs);
        workspace.cache.hold(request.accepted_step);
    } else if !ADAPTIVE {
        workspace.reset_epoch();
        crate::traits::evaluate_rhs_at_base(
            sys,
            prepared_stage_time(t, dt, tableau.c[0]),
            y,
            &mut workspace.stages[0],
            &mut workspace.rhs_candidate,
            request,
            cx,
        )?;
    } else {
        workspace.cache.evaluate_into(
            sys,
            t,
            y,
            request,
            &mut workspace.stages[0],
            &mut workspace.rhs_candidate,
            cx,
        )?;
    }
    for stage in 1..MAIN {
        evaluate_stage(sys, t, y, dt, tableau, stage, workspace, request, cx)?;
    }
    crate::safety::combine_stages(
        Some(y.as_slice()),
        dt,
        &tableau.b[..MAIN],
        &workspace.stages[..MAIN],
        workspace.trial.as_mut_slice(),
    )?;
    // Fixed methods do not enter error, endpoint-cache or dense-extension machinery.
    // The adaptive type flag is established by coefficient preparation.
    if !ADAPTIVE && !DENSE {
        output
            .as_mut_slice()
            .copy_from_slice(workspace.trial.as_slice());
        return Ok(StepResult::from_checked_parts(None, None));
    }
    if ADAPTIVE {
        let weights = &tableau.error;
        if CORRECTED {
            let third = &tableau.correction;
            crate::safety::combine_stage_pair(
                [weights, third],
                &workspace.stages[..MAIN],
                workspace.error.as_mut_slice(),
                |e5, e3| dormand_prince853::combined_error(dt, e5, e3),
            )?;
        } else {
            crate::safety::combine_stages(
                None,
                dt,
                weights,
                &workspace.stages[..N],
                workspace.error.as_mut_slice(),
            )?;
        }
    }
    let retain_endpoint = match tableau.endpoint {
        Endpoint::Stage(stage) => {
            workspace
                .endpoint_rhs
                .as_mut_slice()
                .copy_from_slice(workspace.stages[stage].as_slice());
            true
        }
        Endpoint::Evaluate => {
            evaluate_rhs_at_finite_state(
                sys,
                endpoint_time,
                &workspace.trial,
                &mut workspace.endpoint_rhs,
                &mut workspace.rhs_candidate,
                request,
                cx,
            )?;
            true
        }
        Endpoint::Deferred => false,
    };
    // Extension stages and a deferred endpoint derivative wait for acceptance.
    let deferred = DENSE && tableau.defers_dense();
    let prepared = if let Some(extension) = tableau.extension.filter(|_| DENSE && !deferred) {
        prepare_extension(t, dt, y, extension, tableau, workspace)?;
        true
    } else {
        false
    };
    if retain_endpoint {
        workspace.pending_endpoint = true;
    }
    output
        .as_mut_slice()
        .copy_from_slice(workspace.trial.as_slice());
    let mut result = StepResult::from_checked_parts(
        ADAPTIVE.then_some(workspace.error.as_slice()),
        prepared.then_some(&workspace.dense_output),
    );
    result.dense_deferred = deferred;
    result.stages = Some(&workspace.stages[..MAIN]);

    result.endpoint_rhs = workspace
        .pending_endpoint
        .then_some(workspace.endpoint_rhs.as_slice());
    result.error_order = tableau.error_order;
    Ok(result)
}

/// Evaluates the endpoint derivative and extension stages an accepted step deferred and
/// builds its dense output from the stored stages, endpoint derivative and trial
/// state. A deferred endpoint derivative becomes the next base derivative.
pub(crate) fn complete_extension<
    'a,
    S: RealScalar,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
    const N: usize,
    const MAIN: usize,
    const ADAPTIVE: bool,
    const CORRECTED: bool,
>(
    sys: &System,
    t: S,
    dt: S,
    tableau: &ButcherTableau<S, N, MAIN, ADAPTIVE, CORRECTED>,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut ExplicitOdeWorkspace<S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<&'a crate::dense_output::DenseOutputStorage<S, T>, SciOxideError> {
    let (Some(extension), Some((_, y)), true) =
        (tableau.extension, request.base, tableau.defers_dense())
    else {
        return Err(crate::safety::invalid(
            scioxide_core::argument!("method"),
            "the step deferred no dense extension",
        ));
    };
    if matches!(tableau.endpoint, Endpoint::Deferred) {
        evaluate_rhs_at_finite_state(
            sys,
            request.endpoint(t, dt)?,
            &workspace.trial,
            &mut workspace.endpoint_rhs,
            &mut workspace.rhs_candidate,
            request,
            cx,
        )?;
        workspace.pending_endpoint = true;
    }
    if MAIN < N {
        workspace.stages[MAIN]
            .as_mut_slice()
            .copy_from_slice(workspace.endpoint_rhs.as_slice());
        for stage in MAIN + 1..N {
            evaluate_stage(sys, t, y, dt, tableau, stage, workspace, request, cx)?;
        }
    }
    prepare_extension(t, dt, y, extension, tableau, workspace)?;
    Ok(&workspace.dense_output)
}

/// Prepare the continuous extension once every stage it uses is available.
fn prepare_extension<
    S: RealScalar,
    T: NumericScalar<Real = S>,
    const N: usize,
    const MAIN: usize,
    const ADAPTIVE: bool,
    const CORRECTED: bool,
>(
    t: S,
    dt: S,
    y: &Vector<T>,
    extension: DenseExtension,
    tableau: &ButcherTableau<S, N, MAIN, ADAPTIVE, CORRECTED>,
    workspace: &mut ExplicitOdeWorkspace<S, T>,
) -> Result<(), SciOxideError> {
    let rows = match extension {
        DenseExtension::DormandPrince5 => 1,
        DenseExtension::DormandPrince853 => 4,
        _ => 0,
    };
    workspace.dense_output.prepare_rk(
        extension,
        dt,
        y,
        &workspace.trial,
        &workspace.stages[0],
        &workspace.endpoint_rhs,
        &workspace.stages[..N],
        &tableau.dense[..rows],
    )?;
    workspace
        .dense_output
        .finish(extension, t, dt, y, &workspace.trial)?;
    Ok(())
}

#[cfg(test)]
mod stability_tests {
    use alloc::vec::Vec;

    // Stability polynomial R(z) = 1 + Σ zᵏ bᵀAᵏ⁻¹1 over the advancing stages.
    fn polynomial<const N: usize, const M: usize, const A: bool, const C: bool>(
        tableau: &super::ButcherTableau<f64, N, M, A, C>,
    ) -> Vec<f64> {
        let mut powers = [1.0; M];
        let mut coefficients = alloc::vec![1.0];
        for _ in 0..M {
            coefficients.push((0..M).map(|i| tableau.b[i] * powers[i]).sum());
            powers = core::array::from_fn(|i| (0..M).map(|j| tableau.a[i][j] * powers[j]).sum());
        }
        coefficients
    }

    fn magnitude(coefficients: &[f64], re: f64, im: f64) -> f64 {
        let (r, i) = coefficients.iter().rev().fold((0.0, 0.0), |(r, i), &a| {
            (r * re - i * im + a, r * im + i * re)
        });
        (r * r + i * i).sqrt()
    }

    fn check(coefficients: &[f64], (real, imaginary): (f64, f64)) {
        let stable = |re: f64, im: f64| magnitude(coefficients, re, im) <= 1.0 + 1e-12;
        for k in 1..=1000 {
            let s = f64::from(k) / 1000.0;
            assert!(stable(-real * s, 0.0), "real axis at {}", real * s);
            assert!(
                stable(0.0, imaginary * s),
                "imaginary axis at {}",
                imaginary * s
            );
        }
        // A zero bound only needs some nearby instability; growth may be of high order.
        let beyond = |bound: f64| if bound == 0.0 { 0.05 } else { bound + 2e-3 };
        assert!(!stable(-beyond(real), 0.0), "real bound is not sharp");
        assert!(
            !stable(0.0, beyond(imaginary)),
            "imaginary bound is not sharp"
        );
    }

    #[test]
    fn declared_segments_match_the_stability_polynomials() {
        check(
            &polynomial(&super::euler::tableau::<f64>()),
            super::euler::STABILITY,
        );
        check(
            &polynomial(&super::heun::tableau::<f64>()),
            super::heun::STABILITY,
        );
        check(
            &polynomial(&super::midpoint::tableau::<f64>()),
            super::midpoint::STABILITY,
        );
        check(
            &polynomial(&super::rk4::tableau::<f64>()),
            super::rk4::STABILITY,
        );
        check(
            &polynomial(&super::rkf45::tableau::<f64>()),
            super::rkf45::STABILITY,
        );
        check(
            &polynomial(&super::dormand_prince5::tableau::<f64>()),
            super::dormand_prince5::STABILITY,
        );
        check(
            &polynomial(&super::dormand_prince853::tableau::<f64>()),
            super::dormand_prince853::STABILITY,
        );
        check(
            &polynomial(&super::verner65::tableau::<f64>()),
            super::verner65::STABILITY,
        );
    }
}

/// Order conditions of Butcher tableaus through their elementary weights.
///
/// A rooted tree `t` of order `|t|` has density `γ(t) = |t| Π γ(tₖ)` over its root's
/// subtrees `tₖ`, and stage vector `Φ(•) = 1`, `Φ(t)ᵢ = Πₖ (A Φ(tₖ))ᵢ`. Weights `w`
/// have order `p` when `wᵀ Φ(t) = 1/γ(t)` for every tree with `|t| ≤ p`; dense weights
/// `w(θ)` have order `q` when `w(θ)ᵀ Φ(t) = θ^|t|/γ(t)` for `|t| ≤ q` (Hairer, Nørsett
/// and Wanner, *Solving Ordinary Differential Equations I*, 2nd ed., §II.2 and §II.6).
#[cfg(test)]
pub(crate) mod order_conditions {
    use alloc::vec::Vec;

    /// Rounding allowance relative to the magnitude of the summed terms.
    const TOLERANCE: f64 = 1e-14;

    struct Tree {
        order: usize,
        density: f64,
        children: Vec<usize>,
    }

    /// Every rooted tree of order at most `order`, in nondecreasing order.
    fn trees(order: usize) -> Vec<Tree> {
        fn forests(
            trees: &[Tree],
            remaining: usize,
            start: usize,
            current: &mut Vec<usize>,
            out: &mut Vec<Vec<usize>>,
        ) {
            if remaining == 0 {
                out.push(current.clone());
                return;
            }
            for (index, tree) in trees.iter().enumerate().skip(start) {
                if tree.order <= remaining {
                    current.push(index);
                    forests(trees, remaining - tree.order, index, current, out);
                    current.pop();
                }
            }
        }
        let mut trees = Vec::new();
        for n in 1..=order {
            let mut children = Vec::new();
            forests(&trees, n - 1, 0, &mut Vec::new(), &mut children);
            for children in children {
                let density = children
                    .iter()
                    .fold(n as f64, |density, &child| density * trees[child].density);
                trees.push(Tree {
                    order: n,
                    density,
                    children,
                });
            }
        }
        trees
    }

    /// The stage vectors `Φ(t)` of every tree, with the terms' magnitudes `|A||Φ|`.
    fn elementary<const N: usize>(a: &[[f64; N]; N], trees: &[Tree]) -> Vec<([f64; N], [f64; N])> {
        let mut phi: Vec<([f64; N], [f64; N])> = Vec::new();
        for tree in trees {
            let mut value = [1.0; N];
            let mut magnitude = [1.0; N];
            for &child in &tree.children {
                let (inner, size) = phi[child];
                for i in 0..N {
                    value[i] *= (0..N).map(|j| a[i][j] * inner[j]).sum::<f64>();
                    magnitude[i] *= (0..N).map(|j| (a[i][j] * size[j]).abs()).sum::<f64>();
                }
            }
            phi.push((value, magnitude));
        }
        phi
    }

    /// Assert `cᵢ = Σⱼ aᵢⱼ` for every stage.
    pub(crate) fn assert_row_sums<const N: usize>(label: &str, c: &[f64; N], a: &[[f64; N]; N]) {
        for i in 0..N {
            let sum: f64 = a[i].iter().sum();
            let scale = a[i].iter().map(|value| value.abs()).sum::<f64>().max(1.0);
            assert!(
                (sum - c[i]).abs() <= TOLERANCE * scale,
                "{label}: row {i} sums to {sum}, node {}",
                c[i]
            );
        }
    }

    /// Assert that `weights` have exactly order `order`: every condition of order at
    /// most `order` holds and some condition of order `order + 1` fails.
    pub(crate) fn assert_weights<const N: usize>(
        label: &str,
        a: &[[f64; N]; N],
        order: usize,
        weights: &[f64; N],
    ) {
        assert_order(label, a, order, &[1.0], true, |_| *weights);
    }

    /// Assert that dense weights `weights(θ)` satisfy every condition of order at most
    /// `order` at each `θ`.
    pub(crate) fn assert_dense<const N: usize>(
        label: &str,
        a: &[[f64; N]; N],
        order: usize,
        weights: impl Fn(f64) -> [f64; N],
    ) {
        assert_order(label, a, order, &[0.2, 0.5, 0.8], false, weights);
    }

    fn assert_order<const N: usize>(
        label: &str,
        a: &[[f64; N]; N],
        order: usize,
        thetas: &[f64],
        sharp: bool,
        weights: impl Fn(f64) -> [f64; N],
    ) {
        let trees = trees(order + 1);
        let phi = elementary(a, &trees);
        for &theta in thetas {
            let w = weights(theta);
            let mut violated = false;
            for (tree, (value, magnitude)) in trees.iter().zip(&phi) {
                let sum: f64 = (0..N).map(|i| w[i] * value[i]).sum();
                let scale = (0..N).map(|i| (w[i] * magnitude[i]).abs()).sum::<f64>();
                let exact = theta.powi(tree.order as i32) / tree.density;
                let error = (sum - exact).abs();
                if tree.order <= order {
                    assert!(
                        error <= TOLERANCE * scale.max(1.0),
                        "{label}: θ = {theta}, order {} tree error {error:e}",
                        tree.order
                    );
                } else {
                    violated |= error > 1e-8;
                }
            }
            assert!(!sharp || violated, "{label}: order exceeds {order}");
        }
    }

    /// The weights `w(θ)` of the alternating-basis dense polynomial
    /// `θ(v₀ + (1 − θ)(v₁ + θ(v₂ + (1 − θ)(v₃ + …))))` with `v₀ = b`, `v₁ = e₀ − b`,
    /// `v₂ = 2b − e₀ − e_last` and `v₃₊ₖ = dₖ`, as built from the stages.
    pub(crate) fn alternating<const N: usize>(
        b: &[f64; N],
        last: usize,
        rows: &[[f64; N]],
        theta: f64,
    ) -> [f64; N] {
        let mut basis = alloc::vec![*b, [0.0; N], [0.0; N]];
        for i in 0..N {
            basis[1][i] = -b[i];
            basis[2][i] = 2.0 * b[i];
        }
        basis[1][0] += 1.0;
        basis[2][0] -= 1.0;
        basis[2][last] -= 1.0;
        basis.extend_from_slice(rows);
        let mut weights = [0.0; N];
        for (j, row) in basis.iter().enumerate().rev() {
            let factor = if j % 2 == 0 { theta } else { 1.0 - theta };
            for i in 0..N {
                weights[i] = (weights[i] + row[i]) * factor;
            }
        }
        weights
    }

    #[test]
    fn tree_counts_and_densities_match_the_enumeration() {
        let trees = trees(8);
        let counts: Vec<usize> = (1..=8)
            .map(|n| trees.iter().filter(|tree| tree.order == n).count())
            .collect();
        assert_eq!(counts, [1, 1, 2, 4, 9, 20, 48, 115]);
        // The tall tree has density n! and the bushy tree density n.
        assert!(
            trees
                .iter()
                .any(|tree| tree.order == 5 && tree.density == 120.0)
        );
        assert!(
            trees
                .iter()
                .any(|tree| tree.order == 5 && tree.density == 5.0)
        );
    }
}
