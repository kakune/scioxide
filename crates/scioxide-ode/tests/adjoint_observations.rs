//! Independent affine trajectories qualify sampled discrete objectives.
#![allow(clippy::unwrap_used, clippy::too_many_arguments)]
use scioxide_ad::CheckpointSchedule;
use scioxide_core::*;
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::*;
fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
struct Affine<S> {
    p: [S; 1],
}
fn alpha<T: NumericScalar>() -> T {
    T::from_components(T::Real::ONE, r(0.25))
}
impl<T: NumericScalar> OdeSystem<T::Real, T> for Affine<T::Real> {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: T::Real,
        _: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = alpha::<T>() * T::from_real(self.p[0]);
        Ok(())
    }
    fn jacobian(
        &self,
        _: T::Real,
        _: &[T],
        out: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, T::ZERO)?;
        Ok(Some(()))
    }
}
impl<T: NumericScalar> OdeAdjointSystem<T> for Affine<T::Real> {
    fn parameters(&self) -> &[T::Real] {
        &self.p
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: T::Real,
        _: &[T],
        seed: &[T],
        y: &mut [T],
        p: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        y.fill(T::ZERO);
        p[0] = (seed[0].conj() * alpha::<T>()).real_component();
        Ok(())
    }
}
struct Samples<S> {
    times: Vec<S>,
}
impl<T: NumericScalar> AdjointObjective<T> for Samples<T::Real> {
    fn observation_times(&self) -> &[T::Real] {
        &self.times
    }
    fn observation(
        &self,
        i: usize,
        y: &[T],
        p: &[T::Real],
        gy: &mut [T],
        gp: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        let weight = r::<T::Real>((i + 1) as f64);
        gy[0] = T::from_real(weight) * y[0];
        gp[0] = p[0];
        Ok((weight * y[0].norm_sqr() + p[0] * p[0]) * T::Real::HALF)
    }
    fn evaluate(
        &self,
        _: AdjointObjectivePart,
        _: T::Real,
        _: &[T],
        _: &[T::Real],
        gy: &mut [T],
        gp: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<T::Real>, SciOxideError> {
        gy.fill(T::ZERO);
        gp.fill(T::Real::ZERO);
        Ok(AdjointObjectiveValue {
            value: T::Real::ZERO,
            time_derivative: T::Real::ZERO,
        })
    }
}
fn options<S: RealScalar>(mode: AdjointMode, step: S) -> AdjointOptions<S> {
    AdjointOptions::new(mode, step)
        .unwrap()
        .with_tape_capacity(64)
        .with_max_replay_steps(10000)
        .with_max_reverse_steps(1000)
        .with_max_workspace_bytes(2 * 1024 * 1024)
}
fn check<T: NumericScalar>() {
    let model = Affine {
        p: [r::<T::Real>(0.4)],
    };
    let initial = T::from_components(r(0.7), r(-0.3));
    for (start, end, times) in [
        (0., 1., vec![0., 0.2, 0.2, 0.7, 1.]),
        (1., 0., vec![0., 0.2, 0.2, 0.7, 1.]),
        (0.2, 0.2, vec![0.2, 0.2]),
    ] {
        let samples = Samples {
            times: times.into_iter().map(r).collect(),
        };
        let mut gy = T::ZERO;
        let mut gp = T::Real::ZERO;
        let mut value = T::Real::ZERO;
        for (i, &time) in samples.times.iter().enumerate() {
            let dt = time - r(start);
            let y = initial + alpha::<T>() * T::from_real(model.p[0] * dt);
            let w = r::<T::Real>((i + 1) as f64);
            gy = gy + T::from_real(w) * y;
            gp = gp + w * (y.conj() * alpha::<T>()).real_component() * dt + model.p[0];
            value += (w * y.norm_sqr() + model.p[0] * model.p[0]) * T::Real::HALF;
        }
        for policy in [
            CheckpointPolicy::StoreAll,
            CheckpointPolicy::Explicit(CheckpointSchedule::binomial(64, 2).unwrap()),
        ] {
            let options = options(AdjointMode::Discrete, r(0.3))
                .with_fixed_step(true)
                .with_checkpoint_policy(policy);
            let ode = OdeAdjointSolver::<T>::new(1, 1, options.clone()).unwrap();
            let explicit = ode
                .solve(&scioxide_ode::OdeAdjointProblem::new(
                    &model,
                    &samples,
                    &[initial],
                    (r(start), r(end)),
                ))
                .unwrap();
            let residual = OdeAdjointResidual::new(&model).unwrap();
            let dae = DaeAdjointSolver::<T>::backward_euler(
                residual.dimension(),
                residual.parameter_dimension(),
                OdeStageBackend::Dense,
                scioxide_ode::DaeAdjointOptions::new(
                    options,
                    (DaeOptions::new()).numerics().clone(),
                )
                .unwrap(),
            )
            .unwrap();
            let implicit = dae
                .solve(&scioxide_ode::DaeAdjointProblem::new(
                    &residual,
                    &samples,
                    &[initial],
                    &[alpha::<T>() * T::from_real(model.p[0])],
                    (r(start), r(end)),
                ))
                .unwrap();
            for result in [explicit, implicit] {
                assert!(result.report().is_success());
                let tolerance = r::<T::Real>(512.) * T::Real::EPSILON;
                assert!(
                    (result.initial_gradient()[0] - gy).abs()
                        <= tolerance * (T::Real::ONE + gy.abs())
                );
                assert!(
                    (result.parameter_gradient()[0] - gp).abs()
                        <= tolerance * (T::Real::ONE + gp.abs())
                );
                assert!(
                    (result.report().objective().unwrap() - value).abs()
                        <= tolerance * (T::Real::ONE + value.abs())
                );
            }
        }
    }
}
#[test]
fn scalar_domains() {
    check::<f32>();
    check::<f64>();
    check::<Complex32>();
    check::<Complex64>();
}
#[test]
fn schedule_validation_and_storage() {
    let model = Affine { p: [0.4] };
    let options = options(AdjointMode::Discrete, 0.3);
    for times in [vec![0.5, 0.2], vec![f64::NAN]] {
        let plan = OdeAdjointSolver::<f64>::new(1, 1, options.clone()).unwrap();
        assert!(
            plan.workspace_requirements(&OdeAdjointProblem::new(
                &model,
                &Samples { times },
                &[1.],
                (0., 1.)
            ))
            .is_err()
        );
    }
    let mut samples = Samples { times: vec![0.2] };
    let plan = OdeAdjointSolver::<f64>::new(1, 1, options.clone()).unwrap();
    let empty = OdeAdjointSolver::<f64>::new(1, 1, options.clone()).unwrap();
    let bytes = plan
        .workspace_requirements(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &samples,
            &[1.],
            (0., 1.),
        ))
        .unwrap()
        .required_bytes();
    assert!(
        bytes
            > empty
                .workspace_requirements(&scioxide_ode::OdeAdjointProblem::new(
                    &model,
                    &Samples { times: vec![] },
                    &[],
                    (
                        <_ as scioxide_core::AlgebraicScalar>::ZERO,
                        <_ as scioxide_core::AlgebraicScalar>::ZERO
                    )
                ))
                .unwrap()
                .required_bytes()
    );
    assert!(
        OdeAdjointSolver::<f64>::new(1, 1, options.clone().with_max_workspace_bytes(bytes - 1))
            .unwrap()
            .workspace_requirements(&OdeAdjointProblem::new(&model, &samples, &[1.], (0., 1.)))
            .is_err()
    );
    samples.times[0] = 0.4;
    assert!(
        plan.solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &samples,
            &[1.],
            (0., 1.)
        ))
        .is_ok()
    );
    let outside = OdeAdjointSolver::<f64>::new(1, 1, options).unwrap();
    assert!(
        outside
            .solve(&scioxide_ode::OdeAdjointProblem::new(
                &model,
                &samples,
                &[1.],
                (0., 0.3)
            ))
            .is_err()
    );
    assert!(
        OdeAdjointSolver::<f64>::new(1, 1, self::options(AdjointMode::Continuous, 0.3))
            .unwrap()
            .workspace_requirements(&OdeAdjointProblem::new(&model, &samples, &[1.], (0., 1.)))
            .is_err()
    );
}

struct ChangedSample {
    calls: core::cell::Cell<usize>,
    samples: Samples<f64>,
}
impl AdjointObjective<f64> for ChangedSample {
    fn observation_times(&self) -> &[f64] {
        &self.samples.times
    }
    fn observation(
        &self,
        i: usize,
        y: &[f64],
        p: &[f64],
        gy: &mut [f64],
        gp: &mut [f64],
        w: &mut [f64],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<f64, SciOxideError> {
        let value = self.samples.observation(i, y, p, gy, gp, w, ctx)?;
        self.calls.set(self.calls.get() + 1);
        gy[0] += self.calls.get() as f64;
        Ok(value)
    }
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        t: f64,
        y: &[f64],
        p: &[f64],
        gy: &mut [f64],
        gp: &mut [f64],
        w: &mut [f64],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        self.samples.evaluate(part, t, y, p, gy, gp, w, ctx)
    }
}
#[test]
fn changed_observation_cotangent_is_detected_during_replay() {
    let model = Affine { p: [0.4] };
    let objective = ChangedSample {
        calls: core::cell::Cell::new(0),
        samples: Samples {
            times: vec![0.25, 0.5],
        },
    };
    let options = options(AdjointMode::Discrete, 0.25)
        .with_fixed_step(true)
        .with_checkpoint_policy(CheckpointPolicy::Explicit(
            CheckpointSchedule::binomial(64, 1).unwrap(),
        ));
    let plan = OdeAdjointSolver::new(1, 1, options).unwrap();
    let failure = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[1.],
            (0., 0.5),
        ))
        .unwrap_err();
    assert_eq!(
        failure.progress().unwrap().termination(),
        Some(AdjointTermination::ReplayMismatch)
    );
    assert!(matches!(
        failure.error().view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::ReplayMismatch,
            ..
        }
    ));
}
