//! Shared contracts of the adaptive and refinement families: globally adaptive
//! Gauss–Kronrod and Simpson, Romberg and double-exponential quadrature.
#![allow(clippy::unwrap_used)]

use core::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use scioxide_core::{
    AffineValue, AffineValueMut, AttemptOutcome, AttemptRecord, Budget, BudgetKind, Capacity,
    Complex, ConvergenceReason, ErrorTolerance, ExecutionContext, ExecutionPolicy, Interval,
    NormSelection, ProblemToken, RealScalar, SciOxideError, SciOxideErrorView, SolveBudgets,
    SolveIdentity, SolveScope, SolveState,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, AdaptiveWorkspace, DoubleExponentialDomain,
    DoubleExponentialIntegrator, DoubleExponentialWorkspace, IntegrationErrorComponents,
    IntegrationReport, IntegrationRequirements, IntegrationTermination, LevelOptions, Romberg,
    RombergWorkspace, SubdivisionOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Kronrod,
    Simpson,
    Romberg,
    DoubleExponential,
}

const KINDS: [Kind; 4] = [
    Kind::Kronrod,
    Kind::Simpson,
    Kind::Romberg,
    Kind::DoubleExponential,
];

/// The configuration every family shares: tolerance, evaluation budget, policy.
#[derive(Clone)]
struct Config<S: RealScalar> {
    tolerance: ErrorTolerance<S>,
    max_evaluations: Option<Budget>,
    max_subdivisions: Budget,
    policy: ExecutionPolicy,
}

impl<S: RealScalar> Config<S> {
    fn new(tolerance: ErrorTolerance<S>) -> Self {
        Self {
            tolerance,
            max_evaluations: None,
            max_subdivisions: SubdivisionOptions::<S>::default().max_subdivisions(),
            policy: ExecutionPolicy::Sequential,
        }
    }

    fn scalar(absolute: f64, relative: f64) -> Self {
        Self::new(
            ErrorTolerance::scalar(s(absolute), s(relative), NormSelection::Infinity).unwrap(),
        )
    }

    fn evaluations(mut self, budget: usize) -> Self {
        self.max_evaluations = Some(Budget::try_from(budget).unwrap());
        self
    }
}

enum Plan<S: RealScalar> {
    Kronrod(AdaptiveIntegrator<S>),
    Simpson(AdaptiveIntegrator<S>),
    Romberg(Romberg<S>),
    DoubleExponential([DoubleExponentialIntegrator<S>; 2]),
}

#[derive(Debug)]
enum Workspace<S: RealScalar> {
    Adaptive(AdaptiveWorkspace<S>),
    Romberg(RombergWorkspace<S>),
    DoubleExponential(DoubleExponentialWorkspace<S>),
}

impl<S: RealScalar> Workspace<S> {
    fn last_attempt(&self) -> AttemptRecord<IntegrationReport<S>> {
        match self {
            Self::Adaptive(ws) => ws.last_attempt(),
            Self::Romberg(ws) => ws.last_attempt(),
            Self::DoubleExponential(ws) => ws.last_attempt(),
        }
        .unwrap()
    }

    fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        match self {
            Self::Adaptive(ws) => ws.component_errors(),
            Self::Romberg(ws) => ws.component_errors(),
            Self::DoubleExponential(ws) => ws.component_errors(),
        }
    }

    fn check(&self, requirements: &IntegrationRequirements) -> Capacity {
        match self {
            Self::Adaptive(ws) => ws.check(requirements),
            Self::Romberg(ws) => ws.check(requirements),
            Self::DoubleExponential(ws) => ws.check(requirements),
        }
        .unwrap()
    }
}

/// Unit interval, split at `0.5` for double-exponential runs with two segments.
const SINGULAR: [f64; 1] = [0.5];

impl<S: RealScalar> Plan<S> {
    fn new(kind: Kind, config: Config<S>) -> Self {
        match kind {
            Kind::Kronrod | Kind::Simpson => {
                let options = SubdivisionOptions::default()
                    .with_tolerance(config.tolerance)
                    .with_max_evaluations(config.max_evaluations)
                    .with_max_subdivisions(config.max_subdivisions)
                    .with_execution_policy(config.policy);
                if kind == Kind::Kronrod {
                    Self::Kronrod(
                        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options).unwrap(),
                    )
                } else {
                    Self::Simpson(AdaptiveIntegrator::new(AdaptiveRule::Simpson, options).unwrap())
                }
            }
            Kind::Romberg => Self::Romberg(
                Romberg::new(
                    LevelOptions::default()
                        .with_tolerance(config.tolerance)
                        .with_max_evaluations(config.max_evaluations)
                        .with_execution_policy(config.policy),
                )
                .unwrap(),
            ),
            Kind::DoubleExponential => {
                let options = LevelOptions::default()
                    .with_tolerance(config.tolerance)
                    .with_max_evaluations(config.max_evaluations)
                    .with_execution_policy(config.policy);
                let unit = Interval::new(S::ZERO, S::ONE).unwrap();
                let singular = SINGULAR.map(s::<S>);
                Self::DoubleExponential([
                    DoubleExponentialIntegrator::new(
                        DoubleExponentialDomain::Finite {
                            interval: unit,
                            singular_points: &[],
                        },
                        options.clone(),
                    )
                    .unwrap(),
                    DoubleExponentialIntegrator::new(
                        DoubleExponentialDomain::Finite {
                            interval: unit,
                            singular_points: &singular,
                        },
                        options,
                    )
                    .unwrap(),
                ])
            }
        }
    }

    fn requirements(&self, components: usize) -> IntegrationRequirements {
        match self {
            Self::Kronrod(plan) => plan.workspace_requirements(components),
            Self::Simpson(plan) => plan.workspace_requirements(components),
            Self::Romberg(plan) => plan.workspace_requirements(components),
            Self::DoubleExponential(plan) => plan[1].workspace_requirements(components),
        }
        .unwrap()
    }

    fn workspace(&self, components: usize) -> Workspace<S> {
        match self {
            Self::Kronrod(plan) => Workspace::Adaptive(plan.workspace(components).unwrap()),
            Self::Simpson(plan) => Workspace::Adaptive(plan.workspace(components).unwrap()),
            Self::Romberg(plan) => Workspace::Romberg(plan.workspace(components).unwrap()),
            Self::DoubleExponential(plan) => {
                Workspace::DoubleExponential(plan[1].workspace(components).unwrap())
            }
        }
    }

    /// Integrate over the unit interval; `split` places a double-exponential
    /// breakpoint at `0.5`, which every other family ignores.
    fn run<'scope, 'capability, F, V, O>(
        &self,
        f: &F,
        out: &mut O,
        workspace: &mut Workspace<S>,
        split: bool,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let unit = Interval::new(S::ZERO, S::ONE).unwrap();
        match (self, workspace) {
            (Self::Kronrod(plan), Workspace::Adaptive(ws)) => {
                plan.integrate_into(f, unit, out, ws, scope)
            }
            (Self::Simpson(plan), Workspace::Adaptive(ws)) => {
                plan.integrate_into(f, unit, out, ws, scope)
            }
            (Self::Romberg(plan), Workspace::Romberg(ws)) => {
                plan.integrate_into(f, unit, out, ws, scope)
            }
            (Self::DoubleExponential(plan), Workspace::DoubleExponential(ws)) => {
                let plan = &plan[usize::from(split)];
                plan.integrate_into(
                    |point: scioxide_integrate::DoubleExponentialPoint<_>| f(point.coordinate()),
                    out,
                    ws,
                    scope,
                )
            }
            _ => unreachable!("workspace belongs to its family"),
        }
    }
}

fn s<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

fn native<'scope>() -> SolveScope<'scope, 'static> {
    ExecutionContext::sequential_native().into()
}

fn close<S: RealScalar + Debug>(actual: S, expected: f64, tolerance: f64) {
    assert!(
        (actual.into_f64() - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual:?} != {expected}"
    );
}

const E_MINUS_ONE: f64 = core::f64::consts::E - 1.0;

/// Real, complex and vector values in one precision, plus scalar/vector identity.
fn values<S: RealScalar + Debug>(kind: Kind, tolerance: f64) {
    let plan = Plan::<S>::new(
        kind,
        Config::new(
            ErrorTolerance::scalar(s(tolerance), s(tolerance), NormSelection::Infinity).unwrap(),
        ),
    );
    let requirements = plan.requirements(1);
    assert_eq!(requirements.components(), 1);
    let mut ws1 = plan.workspace(1);
    assert_eq!(ws1.check(&requirements), Capacity::Sufficient);
    assert_eq!(ws1.check(&plan.requirements(2)), Capacity::NeedsGrowth);

    let mut scalar = S::ZERO;
    let report = plan
        .run(&|x: S| Ok(x.exp()), &mut scalar, &mut ws1, false, native())
        .unwrap();
    assert!(report.is_success(), "{kind:?}: {report:?}");
    close(scalar, E_MINUS_ONE, 10.0 * tolerance);
    assert!(report.evaluations() > 0);
    assert!(report.error_estimate().is_some());
    assert_eq!(ws1.last_attempt().outcome(), AttemptOutcome::Succeeded);
    assert_eq!(ws1.last_attempt().progress(), Some(&report));
    let scalar_errors = ws1.component_errors().unwrap()[0];

    let mut ws2 = plan.workspace(2);
    let mut complex = Complex::new(S::ZERO, S::ZERO);
    let report = plan
        .run(
            &|x: S| Ok(Complex::new(x.exp(), x * x)),
            &mut complex,
            &mut ws2,
            false,
            native(),
        )
        .unwrap();
    assert!(report.is_success(), "{kind:?}: {report:?}");
    close(complex.re, E_MINUS_ONE, 10.0 * tolerance);
    close(complex.im, 1.0 / 3.0, 10.0 * tolerance);

    // Identical components decide refinement exactly as the scalar path does.
    let mut ws3 = plan.workspace(3);
    let mut array = [S::ZERO; 3];
    let vector = plan
        .run(
            &|x: S| Ok([x.exp(); 3]),
            &mut array,
            &mut ws3,
            false,
            native(),
        )
        .unwrap();
    let scalar_report = plan
        .run(&|x: S| Ok(x.exp()), &mut scalar, &mut ws1, false, native())
        .unwrap();
    assert_eq!(array, [scalar; 3]);
    assert_eq!(vector.evaluations(), scalar_report.evaluations());
    assert_eq!(vector.subdivisions(), scalar_report.subdivisions());
    assert_eq!(vector.termination(), scalar_report.termination());
    assert_eq!(vector.error_components(), scalar_report.error_components());
    assert_eq!(ws3.component_errors().unwrap(), &[scalar_errors; 3]);

    // Dynamically sized values and outputs use the same kernel.
    let mut dynamic = vec![S::ZERO; 3];
    let report = plan
        .run(
            &|x: S| Ok(vec![x.exp(); 3]),
            dynamic.as_mut_slice(),
            &mut ws3,
            false,
            native(),
        )
        .unwrap();
    assert_eq!(dynamic, array);
    assert_eq!(report, vector);
}

#[test]
fn real_complex_and_vector_values_in_both_precisions() {
    for kind in KINDS {
        values::<f64>(kind, 1e-10);
        values::<f32>(kind, 2e-4);
    }
}

#[test]
fn owned_forms_match_the_atomic_forms() {
    let interval = Interval::new(0.0, 1.0).unwrap();
    let f = |x: f64| Ok(x.exp());
    let (kronrod, _) =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
            .unwrap()
            .integrate(&f, interval)
            .unwrap();
    let (simpson, _) =
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default())
            .unwrap()
            .integrate(&f, interval)
            .unwrap();
    let (romberg, _) = Romberg::new(LevelOptions::default())
        .unwrap()
        .integrate(&f, interval)
        .unwrap();
    let (de, _): (Complex<f64>, _) = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval,
            singular_points: &SINGULAR,
        },
        LevelOptions::default(),
    )
    .unwrap()
    .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f64>| {
        let x = point.coordinate();
        Ok(Complex::new(x.exp(), 0.0))
    })
    .unwrap();
    for value in [kronrod, simpson, romberg, de.re] {
        close(value, E_MINUS_ONE, 1e-7);
    }
}

fn assert_kind(error: &SciOxideError, expected: &str) {
    let actual = match error.view() {
        SciOxideErrorView::DimensionMismatch { .. } => "dimension",
        SciOxideErrorView::ConvergenceFailure { .. } => "convergence",
        SciOxideErrorView::InvalidInput { .. } => "input",
        SciOxideErrorView::ExecutionStopped { .. } => "stopped",
        _ => "other",
    };
    assert_eq!(actual, expected, "{error:?}");
}

#[test]
fn preflight_rejections_leave_output_unchanged() {
    let calls = AtomicUsize::new(0);
    let f = |x: f64| {
        calls.fetch_add(1, Relaxed);
        Ok([x, x])
    };
    for kind in KINDS {
        let plan = Plan::new(kind, Config::scalar(1e-10, 1e-10));
        let mut ws = plan.workspace(2);
        let mut wrong = [7.0; 3];
        let error = plan
            .run(&f, &mut wrong[..], &mut ws, false, native())
            .unwrap_err();
        assert_kind(&error, "dimension");
        assert_eq!(wrong, [7.0; 3]);
        assert_eq!(
            ws.last_attempt().outcome(),
            AttemptOutcome::PreflightRejected
        );
        assert!(ws.last_attempt().epoch().is_none());
        assert!(ws.component_errors().is_none());

        let vector =
            ErrorTolerance::try_vector(&[1e-10; 3], 1e-10, NormSelection::Infinity).unwrap();
        let plan = Plan::new(kind, Config::new(vector));
        let mut out = [7.0; 2];
        let error = plan
            .run(&f, &mut out, &mut ws, false, native())
            .unwrap_err();
        assert_kind(&error, "dimension");
        assert_eq!(out, [7.0; 2]);
        assert_eq!(
            ws.last_attempt().outcome(),
            AttemptOutcome::PreflightRejected
        );
    }
    assert_eq!(calls.load(Relaxed), 0);

    // Family-specific preflight: insufficient Simpson stencils, unreserved or
    // out-of-range singular points.
    let simpson =
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default()).unwrap();
    let mut kronrod_ws =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
            .unwrap()
            .workspace(1)
            .unwrap();
    let mut out = 7.0;
    let unit = Interval::new(0.0, 1.0).unwrap();
    let error = simpson
        .integrate_into(|x: f64| Ok(x), unit, &mut out, &mut kronrod_ws, native())
        .unwrap_err();
    assert_kind(&error, "dimension");
    kronrod_ws
        .try_reserve(&simpson.workspace_requirements(1).unwrap())
        .unwrap();
    let report = simpson
        .integrate_into(|x: f64| Ok(x), unit, &mut out, &mut kronrod_ws, native())
        .unwrap();
    assert!(report.is_success());
    close(out, 0.5, 1e-14);

    let de = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: unit,
            singular_points: &[],
        },
        LevelOptions::default(),
    )
    .unwrap();
    let mut ws = de.workspace(1).unwrap();
    let mut out = 7.0;
    let larger = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: unit,
            singular_points: &[0.25, 0.75],
        },
        LevelOptions::default(),
    )
    .unwrap();
    let error = larger
        .integrate_into(
            |point: scioxide_integrate::DoubleExponentialPoint<_>| Ok(point.coordinate()),
            &mut out,
            &mut ws,
            native(),
        )
        .unwrap_err();
    assert_kind(&error, "dimension");
    assert_eq!(out, 7.0);
    assert_eq!(
        ws.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let invalid = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: unit,
            singular_points: &[1.5],
        },
        LevelOptions::default(),
    )
    .unwrap_err();
    assert_kind(&invalid, "input");
    // Normalization owns just the one interior cut, even after its input is gone.
    let repeated = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: unit,
            singular_points: &[0.0, 0.5, 0.5, 1.0],
        },
        LevelOptions::default(),
    )
    .unwrap();
    let error = repeated
        .integrate_into(
            |point: scioxide_integrate::DoubleExponentialPoint<_>| Ok(point.coordinate()),
            &mut out,
            &mut ws,
            native(),
        )
        .unwrap_err();
    assert_kind(&error, "dimension");
    ws.try_reserve(&repeated.workspace_requirements(1).unwrap())
        .unwrap();
    let report = repeated
        .integrate_into(
            |point: scioxide_integrate::DoubleExponentialPoint<_>| Ok(point.coordinate()),
            &mut out,
            &mut ws,
            native(),
        )
        .unwrap();
    assert!(report.is_success());
    close(out, 0.5, 1e-10);
}

/// Callbacks of the first complete-domain estimate: G7K15, nine Simpson nodes,
/// both Romberg endpoints, and both double-exponential segment centres.
fn first_estimate(kind: Kind) -> usize {
    match kind {
        Kind::Kronrod => 15,
        Kind::Simpson => 9,
        Kind::Romberg => 2,
        Kind::DoubleExponential => 2,
    }
}

#[test]
fn budget_stops_publish_complete_estimates_or_nothing() {
    let calls = AtomicUsize::new(0);
    let f = |x: f64| {
        calls.fetch_add(1, Relaxed);
        Ok(1.0 / (1e-3 + (x - 0.3) * (x - 0.3)))
    };
    for kind in KINDS {
        let first = first_estimate(kind);
        let exact = Config::scalar(0.0, 0.0);

        let plan = Plan::new(kind, exact.clone().evaluations(first + 1));
        let mut ws = plan.workspace(1);
        let mut out = 7.0;
        calls.store(0, Relaxed);
        let report = plan.run(&f, &mut out, &mut ws, true, native()).unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation),
            "{kind:?}"
        );
        assert!(report.evaluations() >= first && report.evaluations() <= first + 1);
        assert_eq!(report.evaluations(), calls.load(Relaxed));
        assert!(out.is_finite() && out != 7.0);
        assert_eq!(ws.last_attempt().outcome(), AttemptOutcome::Stopped);
        assert!(ws.component_errors().is_some());

        let plan = Plan::new(kind, exact.evaluations(first - 1));
        let mut out = 7.0;
        calls.store(0, Relaxed);
        let error = plan.run(&f, &mut out, &mut ws, true, native()).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::EvaluationLimit,
                ..
            }
        ));
        assert_eq!(out, 7.0);
        assert_eq!(calls.load(Relaxed), 0);
        assert_eq!(ws.last_attempt().outcome(), AttemptOutcome::Failed);
        assert!(ws.component_errors().is_none());
    }

    // The subdivision budget bounds both adaptive families.
    for kind in [Kind::Kronrod, Kind::Simpson] {
        let mut config = Config::scalar(0.0, 0.0);
        config.max_subdivisions = Budget::try_from(15).unwrap();
        let plan = Plan::new(kind, config);
        let mut ws = plan.workspace(1);
        let mut out = 0.0;
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Subdivision, Budget::try_from(2).unwrap());
        let identity = SolveIdentity::new(
            ProblemToken::new(2167),
            scioxide_core::algorithm_id!("integrate", "adaptive-family-test", 1),
            0,
            0,
        );
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential_native(),
                identity,
                &mut budgets,
            )
            .unwrap();
        let report = plan.run(&f, &mut out, &mut ws, false, &mut parent).unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Subdivision)
        );
        assert_eq!(report.subdivisions(), 2);
    }

    // An inherited ancestor budget stops the child epoch like a configured one.
    for kind in KINDS {
        let plan = Plan::new(kind, Config::scalar(0.0, 0.0));
        let mut ws = plan.workspace(1);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(
            BudgetKind::Evaluation,
            Budget::try_from(first_estimate(kind) + 1).unwrap(),
        );
        let identity = SolveIdentity::new(
            ProblemToken::new(2038),
            scioxide_core::algorithm_id!("integrate", "adaptive-family-test", 1),
            0,
            0,
        );
        let mut parent = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let mut out = 0.0;
        let report = plan.run(&f, &mut out, &mut ws, true, &mut parent).unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation),
            "{kind:?}"
        );
        assert_eq!(
            parent
                .counts()
                .total()
                .unwrap()
                .provider(scioxide_core::EvaluationRole::Function),
            report.evaluations()
        );
    }
}

#[test]
fn configured_tolerance_governs_vector_integration() {
    let f = |x: f64| Ok([x.exp(), 1.0 / (1.0 + 25.0 * x * x)]);
    let reference = [E_MINUS_ONE, 5.0_f64.atan() / 5.0];
    for kind in KINDS {
        let run = |absolute: f64| {
            let tolerance =
                ErrorTolerance::try_vector(&[absolute; 2], 0.0, NormSelection::Infinity).unwrap();
            let plan = Plan::new(kind, Config::new(tolerance));
            let mut ws = plan.workspace(2);
            let mut out = [0.0; 2];
            let report = plan.run(&f, &mut out, &mut ws, false, native()).unwrap();
            assert!(report.is_success(), "{kind:?} {absolute}: {report:?}");
            for (value, expected) in out.iter().zip(reference) {
                close(*value, expected, 100.0 * absolute.max(1e-14));
            }
            (report.evaluations(), report.subdivisions())
        };
        let loose = run(1e-3);
        let tight = run(1e-11);
        assert!(tight.0 > loose.0, "{kind:?}: {loose:?} vs {tight:?}");
        if matches!(kind, Kind::Kronrod | Kind::Simpson) {
            assert!(tight.1 > loose.1, "{kind:?}: {loose:?} vs {tight:?}");
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn executor_sampling_is_bit_identical() {
    use core::num::NonZeroUsize;
    let pool = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(3).unwrap());
    let f = |x: f64| Ok([x.exp(), (10.0 * x).sin(), 1.0 / (0.01 + x * x)]);
    for kind in KINDS {
        let mut config = Config::scalar(1e-10, 1e-10);
        let sequential = Plan::new(kind, config.clone());
        config.policy = ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap());
        let parallel = Plan::new(kind, config);
        let mut ws = sequential.workspace(3);
        let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
        let first = sequential
            .run(&f, &mut a, &mut ws, true, ExecutionContext::sequential())
            .unwrap();
        let first_errors = ws.component_errors().unwrap().to_vec();
        let second = parallel
            .run(&f, &mut b, &mut ws, true, ExecutionContext::new(&pool))
            .unwrap();
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits), "{kind:?}");
        assert_eq!(first, second, "{kind:?}");
        assert_eq!(ws.component_errors().unwrap(), first_errors.as_slice());
    }
}

#[test]
fn degenerate_domains_and_empty_values_are_exact() {
    let point = Interval::new(0.25, 0.25).unwrap();
    let calls = AtomicUsize::new(0);
    let f = |x: f64| {
        calls.fetch_add(1, Relaxed);
        Ok(x)
    };
    let kronrod =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default()).unwrap();
    let (value, report) = kronrod.integrate(&f, point).unwrap();
    assert_eq!(
        (value, report.termination()),
        (0.0, IntegrationTermination::DegenerateDomain)
    );
    assert!(report.error_components().is_none());
    let romberg = Romberg::new(LevelOptions::default()).unwrap();
    let (value, report) = romberg.integrate(&f, point).unwrap();
    assert_eq!(
        (value, report.termination()),
        (0.0, IntegrationTermination::DegenerateDomain)
    );
    let de = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: point,
            singular_points: &[0.25],
        },
        LevelOptions::default(),
    )
    .unwrap();
    let (value, report) = de
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<_>| f(point.coordinate()))
        .unwrap();
    assert_eq!(
        (value, report.termination()),
        (0.0, IntegrationTermination::DegenerateDomain)
    );
    let empty: [f64; 0] = [];
    let (_, report) = kronrod
        .integrate(|_: f64| Ok(empty), Interval::new(0.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::FixedRuleCompleted
    );
    assert_eq!(calls.load(Relaxed), 0);
    // Levels outside the supported range are rejected at construction.
    assert!(
        Romberg::<f64>::new(LevelOptions::default().with_max_levels(Budget::try_from(31).unwrap()))
            .is_err()
    );
    assert!(
        DoubleExponentialIntegrator::<f64>::new(
            DoubleExponentialDomain::RealLine,
            LevelOptions::default().with_max_levels(Budget::try_from(2).unwrap())
        )
        .is_err()
    );
}

#[test]
fn plans_and_options_are_shareable() {
    fn shareable<T: Send + Sync + Clone>() {}
    shareable::<SubdivisionOptions<f64>>();
    shareable::<AdaptiveIntegrator<f64>>();
    shareable::<AdaptiveIntegrator<f32>>();
    shareable::<LevelOptions<f64>>();
    shareable::<Romberg<f32>>();
    shareable::<DoubleExponentialIntegrator<f64>>();
    shareable::<DoubleExponentialDomain<'static, f64>>();
    fn sendable<T: Send + Sync>() {}
    sendable::<AdaptiveWorkspace<f64>>();
    sendable::<RombergWorkspace<f64>>();
    sendable::<DoubleExponentialWorkspace<f64>>();
}
