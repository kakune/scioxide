//! Independent canonical oscillator and mechanical contract checks.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Complex, EvaluationRole, ExecutionContext,
    ExecutionStopReason, LinearOperator, NumericScalar, RealScalar, ResidualTolerance,
    SciOxideError, SciOxideErrorView, SolveContext,
};
use scioxide_linalg::{Matrix, MatrixViewMut};
use scioxide_ode::{
    HolonomicConstraints, MechanicalOptions, MechanicalReport, MechanicalTermination, PhaseSpace,
    Rattle, RattleOptions, RattleProblem, RattleWorkspace, SeparableHamiltonian, StormerVerlet,
    StormerVerletWorkspace,
};

struct Oscillator(usize);
impl<T: NumericScalar> SeparableHamiltonian<T> for Oscillator {
    fn dimension(&self) -> usize {
        self.0
    }
    fn velocity(
        &self,
        p: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(p);
        Ok(())
    }
    fn gradient(
        &self,
        q: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(q);
        Ok(())
    }
}
fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn phase<T: NumericScalar>(q: &[T], p: &[T]) -> PhaseSpace<T> {
    PhaseSpace::new(q.to_vec(), p.to_vec()).unwrap()
}
/// Advance `state` with a fresh plan of its dimension and a reused workspace.
fn verlet<T: NumericScalar, H: SeparableHamiltonian<T> + ?Sized>(
    problem: &H,
    state: &mut PhaseSpace<T>,
    options: (T::Real, usize, MechanicalOptions),
    workspace: &mut StormerVerletWorkspace<T>,
) -> Result<MechanicalReport<T::Real>, SciOxideError> {
    {
        let (step, steps, options) = options;
        StormerVerlet.advance(
            problem,
            step,
            steps,
            &options,
            state,
            &mut (),
            workspace,
            ExecutionContext::sequential(),
        )
    }
}

fn oscillator<S: RealScalar + NumericScalar<Real = S>>() {
    let mut workspace = StormerVerletWorkspace::<S>::new();
    let mut errors = vec![];
    for n in [20, 40, 80] {
        let h = s::<S>(1.0 / n as f64);
        let mut state = phase(&[s::<S>(1.)], &[s::<S>(0.)]);
        let report = verlet(
            &Oscillator(1),
            &mut state,
            (h, n, MechanicalOptions::new()),
            &mut workspace,
        )
        .unwrap();
        assert!(report.is_success());
        assert_eq!(report.accepted_steps(), n);
        assert_eq!(report.counts().provider(EvaluationRole::Gradient), n + 1);
        let error = NumericScalar::abs(state.position()[0] - s::<S>(1.).cos())
            + NumericScalar::abs(state.momentum()[0] + s::<S>(1.).sin());
        errors.push(error);
        assert!(
            verlet(
                &Oscillator(1),
                &mut state,
                (-h, n, MechanicalOptions::new()),
                &mut workspace
            )
            .unwrap()
            .is_success()
        );
        assert!(NumericScalar::abs(state.position()[0] - s::<S>(1.)) < s::<S>(200.) * S::EPSILON);
        assert!(NumericScalar::abs(state.momentum()[0]) < s::<S>(200.) * S::EPSILON);
    }
    assert!(errors[0] / errors[1] > s::<S>(3.5));
    assert!(errors[1] / errors[2] > s::<S>(3.));
    let mut state = phase(&[s::<S>(1.)], &[s::<S>(0.)]);
    let plan = (s(0.05), 100, MechanicalOptions::new());
    let mut maximum = s::<S>(0.);
    for _ in 0..100 {
        assert!(
            StormerVerlet
                .advance(
                    &Oscillator(1),
                    plan.0,
                    plan.1,
                    &plan.2,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .unwrap()
                .is_success()
        );
        let energy = (state.position()[0] * state.position()[0]
            + state.momentum()[0] * state.momentum()[0])
            / s::<S>(2.);
        maximum = maximum.max(NumericScalar::abs(energy - s::<S>(0.5)));
    }
    assert!(maximum < s::<S>(0.0004));
}
#[test]
fn oscillator_f64() {
    oscillator::<f64>();
}
#[test]
fn oscillator_f32() {
    oscillator::<f32>();
}
fn complex_equivalence<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let complex = phase(
        &[Complex::new(s::<S>(1.), s::<S>(2.))],
        &[Complex::new(s::<S>(0.3), s::<S>(-0.2))],
    );
    let real = phase(&[s::<S>(1.), s::<S>(2.)], &[s::<S>(0.3), s::<S>(-0.2)]);
    let options = steps(s::<S>(0.01), 100);
    let (complex, complex_report) = {
        let (step, steps, options) = options;
        StormerVerlet.solve(&Oscillator(1), step, steps, &options, complex)
    }
    .unwrap()
    .into_parts();
    let (real, real_report) = {
        let (step, steps, options) = options;
        StormerVerlet.solve(&Oscillator(2), step, steps, &options, real)
    }
    .unwrap()
    .into_parts();
    assert!(complex_report.is_success() && real_report.is_success());
    for (c, r) in [
        (complex.position(), real.position()),
        (complex.momentum(), real.momentum()),
    ] {
        assert!(NumericScalar::abs(c[0].real_component() - r[0]) < s::<S>(20.) * S::EPSILON);
        assert!(NumericScalar::abs(c[0].imaginary_component() - r[1]) < s::<S>(20.) * S::EPSILON);
    }
}
#[test]
fn complex_f32() {
    complex_equivalence::<f32>();
}
#[test]
fn complex_f64() {
    complex_equivalence::<f64>();
}
#[test]
fn empty_and_invalid() {
    let mut state = phase::<f64>(&[], &[]);
    assert!(
        verlet(
            &Oscillator(0),
            &mut state,
            (0.1, 1, MechanicalOptions::new()),
            &mut StormerVerletWorkspace::new()
        )
        .unwrap()
        .is_success()
    );
    assert!(
        {
            let (step, steps, options) = (0., 0, MechanicalOptions::new());
            StormerVerlet.solve(
                &Oscillator(0),
                step,
                steps,
                &options,
                phase::<f64>(&[0.0; 0], &[0.0; 0]),
            )
        }
        .is_err()
    );
    assert!(PhaseSpace::new(vec![1.], vec![]).is_err());
}

struct Circle {
    gravity: f64,
}
impl<S: RealScalar + NumericScalar<Real = S>> HolonomicConstraints<S> for Circle {
    fn dimension(&self) -> usize {
        2
    }
    fn constraints(&self) -> usize {
        1
    }
    fn gradient(
        &self,
        _: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = s(0.);
        out[1] = s(self.gravity);
        Ok(())
    }
    fn values(
        &self,
        q: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = (q[0] * q[0] + q[1] * q[1] - s::<S>(1.)) / s::<S>(2.);
        Ok(())
    }
    fn jacobian(
        &self,
        q: &[S],
        out: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.set(0, 0, q[0])?;
        out.set(0, 1, q[1])
    }
}
fn projection<S: RealScalar>(
    h: f64,
    steps: usize,
    tol: f64,
    iters: usize,
) -> (S, usize, RattleOptions<S>) {
    let tolerance = ResidualTolerance::new(s(tol), s(0.)).unwrap();
    (
        s(h),
        steps,
        RattleOptions::new()
            .with_position_tolerance(tolerance, s(1.))
            .with_momentum_tolerance(tolerance, s(1.))
            .with_newton_iterations(iters),
    )
}
/// Advance `state` with a fresh plan for `constraints` and a reused workspace.
fn rattle<S: RealScalar + NumericScalar<Real = S>, P: HolonomicConstraints<S> + ?Sized>(
    constraints: &P,
    inverse_mass: &dyn LinearOperator<S>,
    state: &mut PhaseSpace<S>,
    options: (S, usize, RattleOptions<S>),
    workspace: &mut RattleWorkspace<S>,
) -> Result<MechanicalReport<S>, SciOxideError> {
    {
        let (step, steps, options) = options;
        Rattle.advance(
            &RattleProblem::new(constraints, inverse_mass),
            step,
            steps,
            &options,
            state,
            &mut (),
            workspace,
            ExecutionContext::sequential(),
        )
    }
}
fn circle<S: RealScalar + NumericScalar<Real = S>>(tol: f64, roundoff: f64) {
    let problem = Circle { gravity: 0. };
    let mass = Matrix::<S>::identity(2).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    let mut workspace = RattleWorkspace::new();
    let mut state = phase(&[s::<S>(1.), s(0.)], &[s::<S>(0.), s(1.)]);
    let report = rattle(
        &problem,
        &inverse,
        &mut state,
        projection(0.01, 100, tol, 10),
        &mut workspace,
    )
    .unwrap();
    assert!(report.is_success(), "{report:?}");
    assert!(report.position_residual().unwrap() <= s::<S>(tol));
    assert!(report.momentum_residual().unwrap() <= s::<S>(tol));
    // Exact discrete solution on the unit circle has angle N asin(h).
    let angle = 100.0 * 0.01_f64.asin();
    assert!(NumericScalar::abs(state.position()[0] - s::<S>(angle.cos())) < s::<S>(roundoff));
    assert!(NumericScalar::abs(state.position()[1] - s::<S>(angle.sin())) < s::<S>(roundoff));
    assert!(
        rattle(
            &problem,
            &inverse,
            &mut state,
            projection(-0.01, 100, tol, 10),
            &mut workspace,
        )
        .unwrap()
        .is_success()
    );
    assert!(NumericScalar::abs(state.position()[0] - s::<S>(1.)) < s::<S>(roundoff));
    assert!(NumericScalar::abs(state.position()[1]) < s::<S>(roundoff));
    assert!(NumericScalar::abs(state.momentum()[0]) < s::<S>(roundoff));
    assert!(NumericScalar::abs(state.momentum()[1] - s::<S>(1.)) < s::<S>(roundoff));
}
#[test]
fn rattle_circle_f64() {
    circle::<f64>(1e-13, 2e-9);
}
#[test]
fn rattle_circle_f32() {
    circle::<f32>(2e-7, 5e-4);
}
#[test]
fn rattle_pendulum_order_energy_and_reversal() {
    let problem = Circle { gravity: 1. };
    let mass = Matrix::identity(2).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    let mut workspace = RattleWorkspace::<f64>::new();
    // Independent angular RK4 oracle for theta''=-sin(theta), q=(sin theta,-cos theta).
    let mut theta = 0.7_f64;
    let mut omega = 0.2_f64;
    let dt = 0.0001;
    for _ in 0..10000 {
        let a = -theta.sin();
        let b = -(theta + dt * omega / 2.).sin();
        let c = -(theta + dt * (omega + dt * a / 2.) / 2.).sin();
        let d = -(theta + dt * (omega + dt * b / 2.)).sin();
        theta += dt
            * (omega + 2. * (omega + dt * a / 2.) + 2. * (omega + dt * b / 2.) + (omega + dt * c))
            / 6.;
        omega += dt * (a + 2. * b + 2. * c + d) / 6.;
    }
    let initial = phase(
        &[0.7_f64.sin(), -0.7_f64.cos()],
        &[0.2 * 0.7_f64.cos(), 0.2 * 0.7_f64.sin()],
    );
    let mut errors = vec![];
    for steps in [50, 100, 200] {
        let mut state = initial.clone();
        let h = 1. / steps as f64;
        let r = rattle(
            &problem,
            &inverse,
            &mut state,
            projection(h, steps, 1e-13, 10),
            &mut workspace,
        )
        .unwrap();
        assert!(r.is_success());
        assert_eq!(r.counts().provider(EvaluationRole::Gradient), steps + 1);
        errors.push(
            (state.position()[0] - theta.sin()).abs() + (state.position()[1] + theta.cos()).abs(),
        );
        let r = rattle(
            &problem,
            &inverse,
            &mut state,
            projection(-h, steps, 1e-13, 10),
            &mut workspace,
        )
        .unwrap();
        assert!(r.is_success());
        for (a, b) in state
            .position()
            .iter()
            .chain(state.momentum())
            .zip(initial.position().iter().chain(initial.momentum()))
        {
            assert!((a - b).abs() < 2e-9);
        }
    }
    assert!(
        errors[0] / errors[1] > 3.8 && errors[1] / errors[2] > 3.8,
        "{errors:?}"
    );
    let e0 = 0.02 - 0.7_f64.cos();
    let mut state = initial;
    for _ in 0..100 {
        assert!(
            rattle(
                &problem,
                &inverse,
                &mut state,
                projection(0.02, 100, 1e-13, 10),
                &mut workspace
            )
            .unwrap()
            .is_success()
        );
        let e =
            (state.momentum()[0].powi(2) + state.momentum()[1].powi(2)) / 2. + state.position()[1];
        assert!((e - e0).abs() < 5e-5);
    }
}
#[test]
fn projection_failure_preserves_state() {
    let mut workspace = RattleWorkspace::<f64>::new();
    let mass = Matrix::identity(2).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    let mut state = phase(&[1., 0.], &[0., 1.]);
    let r = rattle(
        &Circle { gravity: 0. },
        &inverse,
        &mut state,
        projection(0.1, 1, 1e-13, 0),
        &mut workspace,
    )
    .unwrap();
    assert_eq!(
        r.termination(),
        Some(MechanicalTermination::ProjectionLimit)
    );
    assert_eq!(r.accepted_steps(), 0);
    assert_eq!(state.position(), [1., 0.]);
    assert_eq!(state.momentum(), [0., 1.]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    let mut state = phase(&[2., 0.], &[0., 1.]);
    assert!(matches!(
        rattle(
            &Circle { gravity: 0. },
            &inverse,
            &mut state,
            projection(0.1, 1, 1e-13, 10),
            &mut workspace
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.progress().unwrap().accepted_steps(), 0);
    assert_eq!(state.position(), [2., 0.]);
}
#[test]
fn exact_verlet_budgets_and_reuse() {
    let mut workspace = StormerVerletWorkspace::new();
    for limit in 1..7 {
        let mut state = phase(&[1.], &[0.]);
        let options = (
            0.1,
            2,
            MechanicalOptions::new().with_max_evaluations(Budget::try_from(limit).unwrap()),
        );
        let r = verlet(&Oscillator(1), &mut state, options, &mut workspace).unwrap();
        assert_eq!(r.accepted_steps(), ((limit - 1) / 2).min(2));
        if limit < 5 {
            assert_eq!(
                r.termination(),
                Some(MechanicalTermination::BudgetLimit(BudgetKind::Evaluation))
            );
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Stopped
            );
        }
        assert_eq!(r.counts().total_providers().unwrap(), limit.min(5));
        if limit < 3 {
            assert_eq!(state.position(), [1.]);
            assert_eq!(state.momentum(), [0.]);
        }
    }
}
#[test]
fn verlet_symplectic_map() {
    // The exact oscillator step is a linear canonical map; columns are computed independently.
    let mut a = phase(&[1.], &[0.]);
    let mut b = phase(&[0.], &[1.]);
    let mut workspace = StormerVerletWorkspace::new();
    let options = (0.2, 1, MechanicalOptions::new());
    assert!(
        verlet(&Oscillator(1), &mut a, options, &mut workspace)
            .unwrap()
            .is_success()
    );
    assert!(
        verlet(&Oscillator(1), &mut b, options, &mut workspace)
            .unwrap()
            .is_success()
    );
    let det: f64 = a.position()[0] * b.momentum()[0] - b.position()[0] * a.momentum()[0];
    assert!((det - 1.).abs() < 4. * f64::EPSILON);
}
#[test]
fn rattle_reduced_symplectic_map() {
    let mass = Matrix::identity(2).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    let mut workspace = RattleWorkspace::new();
    let mut map = |theta: f64, l: f64| {
        let mut state = phase(
            &[theta.sin(), -theta.cos()],
            &[l * theta.cos(), l * theta.sin()],
        );
        assert!(
            rattle(
                &Circle { gravity: 1. },
                &inverse,
                &mut state,
                projection(0.1, 1, 1e-14, 12),
                &mut workspace
            )
            .unwrap()
            .is_success()
        );
        let (q, p) = (state.position(), state.momentum());
        (q[0].atan2(-q[1]), q[0] * p[1] - q[1] * p[0])
    };
    let e = 1e-5;
    let a = map(0.7 + e, 0.2);
    let b = map(0.7 - e, 0.2);
    let c = map(0.7, 0.2 + e);
    let d = map(0.7, 0.2 - e);
    let determinant = ((a.0 - b.0) * (c.1 - d.1) - (c.0 - d.0) * (a.1 - b.1)) / (4. * e * e);
    assert!((determinant - 1.).abs() < 2e-8, "{determinant}");
}

struct Kepler;
impl<S: RealScalar + NumericScalar<Real = S>> SeparableHamiltonian<S> for Kepler {
    fn dimension(&self) -> usize {
        2
    }
    fn velocity(
        &self,
        p: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(p);
        Ok(())
    }
    fn gradient(
        &self,
        q: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let cube = r * r * r;
        out.iter_mut().zip(q).for_each(|(out, &q)| *out = q / cube);
        Ok(())
    }
}
fn kepler<S: RealScalar + NumericScalar<Real = S>>(roundoff: f64) {
    // a=1,e=0.3, mu=1, start at periapsis. Kepler's equation supplies an
    // independent reference, without using another time integration method.
    let eccentricity = 0.3_f64;
    let time = 3.0_f64;
    let mut anomaly = time;
    for _ in 0..12 {
        anomaly -=
            (anomaly - eccentricity * anomaly.sin() - time) / (1. - eccentricity * anomaly.cos());
    }
    let expected = [
        anomaly.cos() - eccentricity,
        (1. - eccentricity * eccentricity).sqrt() * anomaly.sin(),
    ];
    let initial = phase(
        &[s::<S>(0.7), s(0.)],
        &[s::<S>(0.), s((1.3_f64 / 0.7).sqrt())],
    );
    let l0 = initial.position()[0] * initial.momentum()[1];
    let mut workspace = StormerVerletWorkspace::new();
    let mut errors = vec![];
    for n in [300, 600, 1200] {
        let mut state = initial.clone();
        assert!(
            verlet(
                &Kepler,
                &mut state,
                steps(s::<S>(time / n as f64), n),
                &mut workspace
            )
            .unwrap()
            .is_success()
        );
        let (q, p) = (state.position(), state.momentum());
        errors.push(
            NumericScalar::abs(q[0] - s::<S>(expected[0]))
                + NumericScalar::abs(q[1] - s::<S>(expected[1])),
        );
        let angular = q[0] * p[1] - q[1] * p[0];
        assert!(NumericScalar::abs(angular - l0) < s::<S>(roundoff));
    }
    // Accumulated scalar rounding is O(n eps); every adjacent pair whose
    // finer error exceeds that floor must retain second order.
    let mut pairs = 0;
    for (i, pair) in errors.windows(2).enumerate() {
        let floor = s::<S>((600 << i) as f64) * S::EPSILON;
        if pair[1] <= floor {
            break;
        }
        assert!(pair[0] / pair[1] > s::<S>(3.5), "{errors:?}");
        pairs += 1;
    }
    assert!(pairs > 0);
    assert!(errors[2] < s::<S>(6e-5 + roundoff), "{errors:?}");
    let mut state = initial;
    for _ in 0..200 {
        assert!(
            verlet(
                &Kepler,
                &mut state,
                steps(s::<S>(0.01), 100),
                &mut workspace
            )
            .unwrap()
            .is_success()
        );
        let (q, p) = (state.position(), state.momentum());
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let energy = (p[0] * p[0] + p[1] * p[1]) / s::<S>(2.) - s::<S>(1.) / r;
        assert!(NumericScalar::abs(energy + s::<S>(0.5)) < s::<S>(1e-4 + roundoff));
    }
}
#[test]
fn kepler_f64() {
    kepler::<f64>(1e-12);
}
#[test]
fn kepler_f32() {
    kepler::<f32>(2e-5);
}

struct Broken;
impl SeparableHamiltonian<f64> for Broken {
    fn dimension(&self) -> usize {
        1
    }
    fn gradient(
        &self,
        _: &[f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn velocity(
        &self,
        _: &[f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
}
#[test]
fn malformed_callback_and_arithmetic_do_not_commit() {
    let mut state = phase(&[1.], &[0.]);
    let mut workspace = StormerVerletWorkspace::new();
    assert!(matches!(
        verlet(
            &Broken,
            &mut state,
            (0.1, 1, MechanicalOptions::new()),
            &mut workspace
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(state.position(), [1.]);
    assert_eq!(state.momentum(), [0.]);
    let mut state = phase(&[f64::MAX], &[-f64::MAX]);
    assert!(
        verlet(
            &Oscillator(1),
            &mut state,
            (4., 1, MechanicalOptions::new()),
            &mut workspace
        )
        .is_err()
    );
    assert_eq!(state.position(), [f64::MAX]);
    assert_eq!(state.momentum(), [-f64::MAX]);
}
#[test]
fn rattle_evaluation_boundaries_are_transactional() {
    let mass = Matrix::identity(2).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    let mut workspace = RattleWorkspace::new();
    let mut full = phase(&[1., 0.], &[0., 1.]);
    let options = projection(0.1, 1, 1e-13, 10);
    let full_report = rattle(
        &Circle { gravity: 0. },
        &inverse,
        &mut full,
        options,
        &mut workspace,
    )
    .unwrap();
    let count = full_report.counts().total_providers().unwrap();
    for limit in 1..=count {
        let budget = Budget::try_from(limit).unwrap();
        let options = (
            options.0,
            options.1,
            options
                .2
                .with_integration(options.2.integration().with_max_evaluations(budget)),
        );
        let mut state = phase(&[1., 0.], &[0., 1.]);
        let r = rattle(
            &Circle { gravity: 0. },
            &inverse,
            &mut state,
            options,
            &mut workspace,
        )
        .unwrap();
        assert_eq!(r.counts().total_providers().unwrap(), limit);
        if limit < count {
            assert_eq!(
                r.termination(),
                Some(MechanicalTermination::BudgetLimit(BudgetKind::Evaluation))
            );
            assert_eq!(state.position(), [1., 0.]);
            assert_eq!(state.momentum(), [0., 1.]);
        } else {
            assert!(r.is_success());
            assert_eq!(state.position(), full.position());
        }
    }
}
#[test]
fn iteration_and_proposal_limits() {
    for (kind, options) in [
        (
            BudgetKind::Iteration,
            (
                0.1,
                2,
                MechanicalOptions::new().with_max_iterations(Budget::ONE),
            ),
        ),
        (
            BudgetKind::Proposal,
            (0.1, 2, MechanicalOptions::new().with_max_steps(Budget::ONE)),
        ),
    ] {
        let mut state = phase(&[1.], &[0.]);
        let report = verlet(
            &Oscillator(1),
            &mut state,
            options,
            &mut StormerVerletWorkspace::new(),
        )
        .unwrap();
        assert_eq!(
            report.termination(),
            Some(MechanicalTermination::BudgetLimit(kind))
        );
        assert_eq!(report.accepted_steps(), 1);
    }
}

use scioxide_core::{Cancellation, SequentialExecutor};
struct Identity;
impl LinearOperator<f64> for Identity {
    fn rows(&self) -> usize {
        2
    }
    fn cols(&self) -> usize {
        2
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert!(scratch.is_empty());
        output.copy_from_slice(input);
        Ok(())
    }
}
/// The identity with a scratch requirement, to exercise retained excess scratch.
struct ScratchIdentity;
impl LinearOperator<f64> for ScratchIdentity {
    fn rows(&self) -> usize {
        2
    }
    fn cols(&self) -> usize {
        2
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(5)
    }
    fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), 5);
        output.copy_from_slice(input);
        Ok(())
    }
}
#[test]
fn dense_sparse_operator_and_borrowed_execution_agree() {
    let matrix = Matrix::identity(2).unwrap();
    let sparse =
        scioxide_sparse::CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![1., 1.])
            .unwrap();
    let view = matrix.view();
    let dense = matrix.view().try_finite().unwrap();
    let view = view.try_finite().unwrap();
    let plan = projection(0.01, 100, 1e-13, 10);
    let executor = SequentialExecutor;
    // One workspace serves every operator; the first reserves excess scratch for the rest.
    let mut workspace = Rattle
        .workspace(&RattleProblem::new(
            &Circle { gravity: 0. },
            &ScratchIdentity,
        ))
        .unwrap();
    let mut expected: Option<PhaseSpace<f64>> = None;
    for mass in [
        &ScratchIdentity as &dyn LinearOperator<f64>,
        &dense,
        &sparse,
        &view,
        &Identity,
    ] {
        let problem = RattleProblem::new(&Circle { gravity: 0. }, mass);
        let mut state = phase(&[1., 0.], &[0., 1.]);
        let r = Rattle
            .advance(
                &problem,
                plan.0,
                plan.1,
                &plan.2,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::new(&executor),
            )
            .unwrap();
        assert!(r.is_success());
        if let Some(old) = &expected {
            assert_eq!(&state, old);
        } else {
            expected = Some(state);
        }
    }
}
struct Cancelled;
impl Cancellation for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}
#[test]
fn cancellation_precedes_callbacks() {
    let plan = (0.1, 1, MechanicalOptions::new());
    let mut workspace = StormerVerlet.workspace(&Oscillator(1)).unwrap();
    let mut state = phase(&[1.], &[0.]);
    let r = StormerVerlet
        .advance(
            &Broken,
            plan.0,
            plan.1,
            &plan.2,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&Cancelled),
        )
        .unwrap();
    assert_eq!(
        r.termination(),
        Some(MechanicalTermination::ExecutionStopped(
            ExecutionStopReason::Cancelled
        ))
    );
    assert_eq!(r.counts().total_providers().unwrap(), 0);
    assert_eq!(state.position(), [1.]);
    assert_eq!(state.momentum(), [0.]);
}
struct Dependent;
impl HolonomicConstraints<f64> for Dependent {
    fn dimension(&self) -> usize {
        2
    }
    fn constraints(&self) -> usize {
        2
    }
    fn gradient(
        &self,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.fill(0.);
        Ok(())
    }
    fn values(
        &self,
        q: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.fill(q[0] - 1.);
        Ok(())
    }
    fn jacobian(
        &self,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.set(0, 0, 1.)?;
        out.set(0, 1, 0.)?;
        out.set(1, 0, 1.)?;
        out.set(1, 1, 0.)
    }
}
#[test]
fn rank_deficiency_is_an_error() {
    let mut state = phase(&[1., 0.], &[0., 1.]);
    assert!(matches!(
        rattle(
            &Dependent,
            &Identity,
            &mut state,
            projection(0.1, 1, 1e-13, 10),
            &mut RattleWorkspace::new()
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::Singular { .. }
    ));
    assert_eq!(state.position(), [1., 0.]);
    assert_eq!(state.momentum(), [0., 1.]);
}
#[test]
fn gauss_legendre_fixed_step_structure() {
    use scioxide_core::{ErrorTolerance, NormSelection};
    use scioxide_linalg::Vector;
    use scioxide_ode::{GaussRk, OdeFn, OdeMethod, OdeOptions, OdeProblem, OdeState};
    let system = OdeFn::new(2, |_: f64, y: &[f64], out: &mut [f64]| {
        out[0] = y[1];
        out[1] = -y[0];
        Ok(())
    })
    .unwrap();
    let tolerance = ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap();
    let options = OdeOptions::new()
        .with_error_tolerance(tolerance)
        .with_initial_step(0.1);
    let plan = (GaussRk::new(), options);
    let advance = |end: f64, state: &mut OdeState<f64>| {
        let problem = OdeProblem::new(&system, end);
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
    };
    let mut columns = vec![];
    for initial in [vec![1., 0.], vec![0., 1.]] {
        let mut state = OdeState::new(0., Vector::from_vec(initial));
        advance(0.1, &mut state);
        columns.push(state);
    }
    let a = columns[0].values();
    let b = columns[1].values();
    assert!((a[0] * b[1] - a[1] * b[0] - 1.).abs() < 1e-11);
    let mut state = OdeState::new(0., Vector::from_vec(vec![1., 0.]));
    advance(100., &mut state);
    let y = state.values();
    assert!((y[0] * y[0] + y[1] * y[1] - 1.).abs() < 1e-9);
    advance(0., &mut state);
    let y = state.values();
    assert!((y[0] - 1.).abs() + y[1].abs() < 1e-9);
}

struct Sphere;
impl<S: RealScalar + NumericScalar<Real = S>> HolonomicConstraints<S> for Sphere {
    fn dimension(&self) -> usize {
        3
    }
    fn constraints(&self) -> usize {
        1
    }
    fn gradient(
        &self,
        _: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.fill(s(0.));
        Ok(())
    }
    fn values(
        &self,
        q: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = (q.iter().fold(s::<S>(0.), |sum, &v| sum + v * v) - s::<S>(1.)) / s::<S>(2.);
        Ok(())
    }
    fn jacobian(
        &self,
        q: &[S],
        out: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for (i, &v) in q.iter().enumerate() {
            out.set(0, i, v)?;
        }
        Ok(())
    }
}
fn sphere<S: RealScalar + NumericScalar<Real = S>>(tolerance: f64, threshold: f64) {
    let q0 = [1. / 3.0_f64.sqrt(); 3];
    let p0 = [1. / 2.0_f64.sqrt(), -1. / 2.0_f64.sqrt(), 0.];
    let mut state = phase(&q0.map(s::<S>), &p0.map(s::<S>));
    let mass = Matrix::<S>::identity(3).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    let r = rattle(
        &Sphere,
        &inverse,
        &mut state,
        projection(0.01, 100, tolerance, 10),
        &mut RattleWorkspace::new(),
    )
    .unwrap();
    assert!(r.is_success());
    let angle = 100. * 0.01_f64.asin();
    for i in 0..3 {
        let exact = q0[i] * angle.cos() + p0[i] * angle.sin();
        assert!(NumericScalar::abs(state.position()[i] - s::<S>(exact)) < s::<S>(threshold));
    }
}
#[test]
fn sphere_f32() {
    sphere::<f32>(2e-7, 5e-4);
}
#[test]
fn sphere_f64() {
    sphere::<f64>(1e-13, 2e-9);
}
#[test]
fn borrowed_context_respects_already_consumed_budget() {
    use scioxide_core::{ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let mut epoch = SolveState::new();
    let mut budgets = SolveBudgets::default().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = epoch
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(661),
                scioxide_core::algorithm_id!("test", "mechanical", 1),
                1,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    assert!(
        context
            .record_provider(EvaluationRole::Function)
            .unwrap()
            .is_granted()
    );
    let plan = (0.1, 1, MechanicalOptions::new());
    let mut workspace = StormerVerlet.workspace(&Oscillator(1)).unwrap();
    let mut state = phase(&[1.], &[0.]);
    // A child epoch shares the ancestor's exhausted evaluation budget.
    let report = StormerVerlet
        .advance(
            &Broken,
            plan.0,
            plan.1,
            &plan.2,
            &mut state,
            &mut (),
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(MechanicalTermination::BudgetLimit(BudgetKind::Evaluation))
    );
    assert_eq!(report.counts().total_providers().unwrap(), 0);
    assert_eq!(
        context.counts().total().unwrap().total_providers().unwrap(),
        1
    );
    assert_eq!(state.position(), [1.]);
    assert_eq!(state.momentum(), [0.]);
}

struct PartialComplex;
impl SeparableHamiltonian<Complex<f64>> for PartialComplex {
    fn dimension(&self) -> usize {
        1
    }
    fn gradient(
        &self,
        _: &[Complex<f64>],
        out: &mut [Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0].re = 0.;
        Ok(())
    }
    fn velocity(
        &self,
        _: &[Complex<f64>],
        out: &mut [Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0].re = 0.;
        Ok(())
    }
}
#[test]
fn incomplete_complex_callback_is_rejected() {
    let mut state = phase(&[Complex::new(1., 2.)], &[Complex::new(0., 0.)]);
    let result = verlet(
        &PartialComplex,
        &mut state,
        (0.1, 1, MechanicalOptions::new()),
        &mut StormerVerletWorkspace::new(),
    );
    assert!(matches!(
        result.unwrap_err().view(),
        SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(state.position(), [Complex::new(1., 2.)]);
}

#[test]
fn malformed_steps_and_shapes_have_specific_errors() {
    assert!(matches!(
        {
            let (step, steps, options) = steps(f64::NAN, 1);
            StormerVerlet.solve(
                &Oscillator(1),
                step,
                steps,
                &options,
                phase::<f64>(&[0.0; 1], &[0.0; 1]),
            )
        }
        .unwrap_err()
        .error()
        .view(),
        SciOxideErrorView::NonFinite { .. }
    ));
    assert!(matches!(
        {
            let (step, steps, options) = steps(f64::INFINITY, 1);
            StormerVerlet.solve(
                &Oscillator(1),
                step,
                steps,
                &options,
                phase::<f64>(&[0.0; 1], &[0.0; 1]),
            )
        }
        .unwrap_err()
        .error()
        .view(),
        SciOxideErrorView::NonFinite { .. }
    ));
    let mut state = phase(&[1., 0.], &[0., 1.]);
    let mut workspace = StormerVerletWorkspace::new();
    assert!(matches!(
        verlet(
            &Broken,
            &mut state,
            (0.1, 1, MechanicalOptions::new()),
            &mut workspace
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.progress().is_none());
    let mass = Matrix::identity(1).unwrap();
    let mut workspace = RattleWorkspace::new();
    assert!(matches!(
        rattle(
            &Circle { gravity: 0. },
            &mass.view().try_finite().unwrap(),
            &mut state,
            projection(0.1, 1, 1e-13, 10),
            &mut workspace
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(state.position(), [1., 0.]);
}

struct SpherePlane;
impl HolonomicConstraints<f64> for SpherePlane {
    fn dimension(&self) -> usize {
        3
    }
    fn constraints(&self) -> usize {
        2
    }
    fn gradient(
        &self,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.fill(0.);
        Ok(())
    }
    fn values(
        &self,
        q: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = (q.iter().map(|v| v * v).sum::<f64>() - 1.) / 2.;
        out[1] = q[2] - 0.3;
        Ok(())
    }
    fn jacobian(
        &self,
        q: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for (col, &value) in q.iter().enumerate() {
            out.set(0, col, value)?;
            out.set(1, col, if col == 2 { 1. } else { 0. })?;
        }
        Ok(())
    }
}
#[test]
fn multiple_constraints_with_coupled_mass_are_reversible() {
    // Strictly diagonally dominant symmetric inverse mass is SPD. Its
    // off-diagonal couplings make the position Newton matrix nonsymmetric.
    let mass = Matrix::from_vec(3, 3, vec![2., 0., 0.5, 0., 1., 0.25, 0.5, 0.25, 1.]).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    // Exactly in rational arithmetic: inverse_mass * p = (0,1,0).
    let initial = phase(
        &[0.91_f64.sqrt(), 0., 0.3],
        &[1. / 13., 14. / 13., -4. / 13.],
    );
    let mut state = initial.clone();
    let mut workspace = RattleWorkspace::new();
    let report = rattle(
        &SpherePlane,
        &inverse,
        &mut state,
        projection(0.01, 100, 1e-13, 10),
        &mut workspace,
    )
    .unwrap();
    assert!(report.is_success());
    assert!(report.position_residual().unwrap() <= 1e-13);
    assert!(report.momentum_residual().unwrap() <= 1e-13);
    let mut velocity = [0.; 3];
    inverse
        .apply_into(state.momentum(), &mut velocity, &mut [0.; 3])
        .unwrap();
    let energy = state
        .momentum()
        .iter()
        .zip(velocity)
        .map(|(&p, v)| p * v)
        .sum::<f64>()
        / 2.;
    assert!((energy - 7. / 13.).abs() < 1e-4);
    assert!(
        rattle(
            &SpherePlane,
            &inverse,
            &mut state,
            projection(-0.01, 100, 1e-13, 10),
            &mut workspace
        )
        .unwrap()
        .is_success()
    );
    for (a, b) in state
        .position()
        .iter()
        .chain(state.momentum())
        .zip(initial.position().iter().chain(initial.momentum()))
    {
        assert!((a - b).abs() < 2e-9);
    }
}

struct PackedKepler;
impl<S: RealScalar + NumericScalar<Real = S>> SeparableHamiltonian<Complex<S>> for PackedKepler
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn dimension(&self) -> usize {
        1
    }
    fn velocity(
        &self,
        p: &[Complex<S>],
        out: &mut [Complex<S>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(p);
        Ok(())
    }
    fn gradient(
        &self,
        q: &[Complex<S>],
        out: &mut [Complex<S>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let radius = q[0].abs();
        out[0] = q[0] / Complex::new(radius * radius * radius, S::ZERO);
        Ok(())
    }
}
fn packed_kepler<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut packed = phase(
        &[Complex::new(s::<S>(1.), s(0.))],
        &[Complex::new(s(0.), s(1.))],
    );
    let initial = packed.clone();
    let mut real = phase(&[s::<S>(1.), s(0.)], &[s::<S>(0.), s(1.)]);
    let mut complex_workspace = StormerVerletWorkspace::new();
    let mut real_workspace = StormerVerletWorkspace::new();
    let mut maximum = S::ZERO;
    for _ in 0..20 {
        let options = steps(s::<S>(0.005), 100);
        assert!(
            verlet(&PackedKepler, &mut packed, options, &mut complex_workspace)
                .unwrap()
                .is_success()
        );
        assert!(
            verlet(&Kepler, &mut real, options, &mut real_workspace)
                .unwrap()
                .is_success()
        );
        let (q, p) = (packed.position()[0], packed.momentum()[0]);
        let energy = p.norm_sqr() / s::<S>(2.) - s::<S>(1.) / q.abs();
        maximum = maximum.max((energy + s::<S>(0.5)).abs());
        assert!((q - Complex::new(real.position()[0], real.position()[1])).abs() < s::<S>(0.001));
    }
    assert!(maximum < s::<S>(0.0001));
    let exact = Complex::new(s::<S>(10.).cos(), s::<S>(10.).sin());
    assert!((packed.position()[0] - exact).abs() < s::<S>(0.0005));
    assert!(
        verlet(
            &PackedKepler,
            &mut packed,
            steps(s::<S>(-0.005), 2000),
            &mut complex_workspace,
        )
        .unwrap()
        .is_success()
    );
    assert!((packed.position()[0] - initial.position()[0]).abs() < s::<S>(0.0001));
    assert!((packed.momentum()[0] - initial.momentum()[0]).abs() < s::<S>(0.0001));
}
#[test]
fn packed_kepler_f32() {
    packed_kepler::<f32>();
}
#[test]
fn packed_kepler_f64() {
    packed_kepler::<f64>();
}

#[test]
fn force_caches_restart_for_each_invocation_and_zero_steps_evaluate_no_force() {
    let mass = Matrix::identity(2).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    for n in [0, 1, 8, 0, 1] {
        let plan = (0.01, n, MechanicalOptions::new());
        let mut workspace = StormerVerlet.workspace(&Oscillator(1)).unwrap();
        let mut state = phase(&[1.0], &[0.0]);
        for _ in 0..2 {
            let report = StormerVerlet
                .advance(
                    &Oscillator(1),
                    plan.0,
                    plan.1,
                    &plan.2,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(
                report.counts().provider(EvaluationRole::Gradient),
                if n == 0 { 0 } else { n + 1 }
            );
        }
        let mut workspace = RattleWorkspace::new();
        let mut state = phase(&[1.0, 0.0], &[0.0, 1.0]);
        for _ in 0..2 {
            let report = rattle(
                &Circle { gravity: 0.0 },
                &inverse,
                &mut state,
                projection(0.01, n, 1e-12, 10),
                &mut workspace,
            )
            .unwrap();
            assert!(report.is_success());
            assert_eq!(
                report.counts().provider(EvaluationRole::Gradient),
                if n == 0 { 0 } else { n + 1 }
            );
        }
    }
}

/// The oscillator outside `q >= 0.99`, where the potential is undefined.
struct Bounded;
impl SeparableHamiltonian<f64> for Bounded {
    fn dimension(&self) -> usize {
        1
    }
    fn velocity(
        &self,
        p: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(p);
        Ok(())
    }
    fn gradient(
        &self,
        q: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if q[0] < 0.99 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("position"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "outside the potential's domain",
            ));
        }
        out.copy_from_slice(q);
        Ok(())
    }
}
#[test]
fn verlet_failure_keeps_the_last_accepted_step_and_progress() {
    // q1 = 1 - h^2/2 = 0.995 is accepted; the second step's force at q2 = 0.98 fails.
    let plan = (0.1, 5, MechanicalOptions::new());
    let mut reference = phase(&[1.], &[0.]);
    assert!(
        verlet(
            &Oscillator(1),
            &mut reference,
            (0.1, 1, MechanicalOptions::new()),
            &mut StormerVerletWorkspace::new(),
        )
        .unwrap()
        .is_success()
    );
    let mut workspace = StormerVerlet.workspace(&Oscillator(1)).unwrap();
    let mut state = phase(&[1.], &[0.]);
    let mut emitted = vec![];
    let mut sink = |step: usize, q: &[f64], _: &[f64]| {
        emitted.push((step, q[0]));
        Ok(())
    };
    let error = StormerVerlet
        .advance(
            &Bounded,
            plan.0,
            plan.1,
            &plan.2,
            &mut state,
            &mut sink,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    assert_eq!(state, reference);
    assert_eq!(emitted, [(1, reference.position()[0])]);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert_eq!(progress.accepted_steps(), 1);
    assert_eq!(progress.termination(), None);
    assert!(!progress.is_success());
    assert_eq!(progress.counts().provider(EvaluationRole::Gradient), 3);
    // The owned convenience returns the same progress with the error.
    let failure = StormerVerlet
        .solve(&Bounded, plan.0, plan.1, &plan.2, phase(&[1.], &[0.]))
        .unwrap_err();
    assert!(matches!(
        failure.error().view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    assert_eq!(failure.progress().unwrap().accepted_steps(), 1);
    // A preflight rejection of the owned form has no progress.
    let failure = StormerVerlet
        .solve(
            &Bounded,
            plan.0,
            plan.1,
            &plan.2,
            phase(&[1., 0.], &[0., 0.]),
        )
        .unwrap_err();
    assert!(failure.progress().is_none());
}
#[test]
fn sink_errors_fail_after_the_emitted_step() {
    let mass = Matrix::identity(2).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    let problem = RattleProblem::new(&Circle { gravity: 1. }, &inverse);
    let plan = projection(0.05, 4, 1e-13, 10);
    let mut workspace = Rattle.workspace(&problem).unwrap();
    let mut state = phase(&[0.7_f64.sin(), -0.7_f64.cos()], &[0., 0.]);
    let mut last = None;
    let mut sink = |step: usize, q: &[f64], p: &[f64]| {
        last = Some(phase(q, p));
        if step == 2 {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("sink"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "stop recording",
            ))
        } else {
            Ok(())
        }
    };
    assert!(
        Rattle
            .advance(
                &problem,
                plan.0,
                plan.1,
                &plan.2,
                &mut state,
                &mut sink,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    let emitted = last.as_ref().unwrap();
    assert_eq!(state.position(), emitted.position());
    assert_eq!(state.momentum(), emitted.momentum());
    assert_eq!(state.time(), 0.1);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert_eq!(progress.accepted_steps(), 2);
    assert_eq!(progress.final_time(), 0.1);
    assert_eq!(progress.termination(), None);
    assert!(progress.position_residual().unwrap() <= 1e-13);
    // The state continues: two further steps match a four-step reference.
    let mut reference = phase(&[0.7_f64.sin(), -0.7_f64.cos()], &[0., 0.]);
    assert!(
        Rattle
            .advance(
                &problem,
                plan.0,
                plan.1,
                &plan.2,
                &mut reference,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap()
            .is_success()
    );
    let two = projection(0.05, 2, 1e-13, 10);
    assert!(
        Rattle
            .advance(
                &problem,
                two.0,
                two.1,
                &two.2,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap()
            .is_success()
    );
    assert_eq!(state, reference);
    let solution = Rattle
        .solve(&problem, plan.0, plan.1, &plan.2, state)
        .unwrap();
    assert_eq!(solution.report().accepted_steps(), 4);
}
#[test]
fn plans_are_send_sync_clone() {
    fn shared<T: Send + Sync + Clone>() {}
    shared::<StormerVerlet>();
    shared::<StormerVerlet>();
    shared::<Rattle>();
    shared::<MechanicalOptions>();
    shared::<RattleOptions<f64>>();
}
#[test]
fn shared_termination_names_exist() {
    for termination in [
        MechanicalTermination::ReachedEnd,
        MechanicalTermination::TerminalEvent,
        MechanicalTermination::BudgetLimit(BudgetKind::Evaluation),
        MechanicalTermination::StepSizeLimit,
        MechanicalTermination::ExecutionStopped(ExecutionStopReason::Cancelled),
        MechanicalTermination::ProjectionLimit,
    ] {
        assert_eq!(
            termination.is_success(),
            matches!(
                termination,
                MechanicalTermination::ReachedEnd | MechanicalTermination::TerminalEvent
            )
        );
    }
}

#[test]
fn report_time_tracks_successive_calls_and_early_stops() {
    let mut state = phase(&[1_f64], &[0.]);
    let mut workspace = StormerVerletWorkspace::new();
    let report = verlet(
        &Oscillator(1),
        &mut state,
        (
            0.125,
            3,
            MechanicalOptions::new().with_max_steps(Budget::ONE),
        ),
        &mut workspace,
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        Some(MechanicalTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert_eq!(report.accepted_steps(), 1);
    assert_eq!(report.rejected_steps(), 0);
    assert_eq!(report.final_time(), 0.125);
    assert_eq!(state.time(), 0.125);
    assert_eq!(
        report.trust(),
        scioxide_core::PartialResultTrust::Admissible
    );
    let report = verlet(
        &Oscillator(1),
        &mut state,
        (-0.125, 2, MechanicalOptions::new()),
        &mut workspace,
    )
    .unwrap();
    assert!(report.is_success());
    assert_eq!(report.final_time(), -0.125);
    assert_eq!(state.time(), -0.125);
}

fn steps<S: RealScalar>(step: S, count: usize) -> (S, usize, MechanicalOptions) {
    (step, count, MechanicalOptions::new())
}

#[test]
fn mechanical_steps_require_representable_time_progress() {
    for (initial_step, next_step) in [
        (-2.0_f64.powi(54), -1.0),
        (2.0_f64.powi(54), 1.0),
        (-f64::MAX, -f64::MAX),
        (f64::MAX, f64::MAX),
    ] {
        let mut state = phase(&[0.0], &[0.0]);
        let mut workspace = StormerVerletWorkspace::new();
        assert!(
            verlet(
                &Oscillator(1),
                &mut state,
                steps(initial_step, 1),
                &mut workspace
            )
            .unwrap()
            .is_success()
        );
        let before = state.clone();
        let report = verlet(
            &Oscillator(1),
            &mut state,
            steps(next_step, 1),
            &mut workspace,
        )
        .unwrap();
        assert_eq!(
            report.termination(),
            Some(MechanicalTermination::StepSizeLimit)
        );
        assert_eq!(report.accepted_steps(), 0);
        assert_eq!(state.time(), before.time());
        assert_eq!(state.position(), before.position());
        assert_eq!(state.momentum(), before.momentum());

        let problem = Circle { gravity: 0.0 };
        let mass = Matrix::<f64>::identity(2).unwrap();
        let inverse = mass.view().try_finite().unwrap();
        let mut state = phase(&[1.0, 0.0], &[0.0, 0.0]);
        let mut workspace = RattleWorkspace::new();
        assert!(
            rattle(
                &problem,
                &inverse,
                &mut state,
                projection(initial_step, 1, 1e-12, 10),
                &mut workspace
            )
            .unwrap()
            .is_success()
        );
        let before = state.clone();
        let report = rattle(
            &problem,
            &inverse,
            &mut state,
            projection(next_step, 1, 1e-12, 10),
            &mut workspace,
        )
        .unwrap();
        assert_eq!(
            report.termination(),
            Some(MechanicalTermination::StepSizeLimit)
        );
        assert_eq!(report.accepted_steps(), 0);
        assert_eq!(state.time(), before.time());
        assert_eq!(state.position(), before.position());
        assert_eq!(state.momentum(), before.momentum());
    }
}

#[test]
fn verlet_drift_preserves_finite_cancellation() {
    struct FreeParticle;
    impl SeparableHamiltonian<f64> for FreeParticle {
        fn dimension(&self) -> usize {
            1
        }
        fn velocity(
            &self,
            p: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(p);
            Ok(())
        }
        fn gradient(
            &self,
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.fill(0.0);
            Ok(())
        }
    }
    let mut state = phase(&[-f64::MAX], &[f64::MAX]);
    let report = verlet(
        &FreeParticle,
        &mut state,
        steps(2.0, 1),
        &mut StormerVerletWorkspace::new(),
    )
    .unwrap();
    assert!(report.is_success());
    assert_eq!(state.position(), &[f64::MAX]);
    assert_eq!(state.momentum(), &[f64::MAX]);
}

#[test]
fn rattle_drift_preserves_finite_cancellation() {
    struct FixedFirstCoordinate;
    impl HolonomicConstraints<f64> for FixedFirstCoordinate {
        fn dimension(&self) -> usize {
            2
        }
        fn constraints(&self) -> usize {
            1
        }
        fn gradient(
            &self,
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.fill(0.0);
            Ok(())
        }
        fn values(
            &self,
            q: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out[0] = q[0] - 1.0;
            Ok(())
        }
        fn jacobian(
            &self,
            _: &[f64],
            out: &mut MatrixViewMut<'_, f64>,
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.set(0, 0, 1.0)?;
            out.set(0, 1, 0.0)
        }
    }
    let mass = Matrix::<f64>::identity(2).unwrap();
    let inverse = mass.view().try_finite().unwrap();
    let mut state = phase(&[1.0, -f64::MAX], &[0.0, f64::MAX]);
    let report = rattle(
        &FixedFirstCoordinate,
        &inverse,
        &mut state,
        projection(2.0, 1, 1e-12, 10),
        &mut RattleWorkspace::new(),
    )
    .unwrap();
    assert!(report.is_success());
    assert_eq!(state.position(), &[1.0, f64::MAX]);
    assert_eq!(state.momentum(), &[0.0, f64::MAX]);
}
