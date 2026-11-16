use super::*;
use crate::{MethodOfLinesPlan, MethodOfLinesStagePreconditioner};
use core::cell::{Cell, RefCell};
use scioxide_core::{Budget, ErrorTolerance, ExecutionContext, NormSelection};
use scioxide_ode::{
    OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeStageBackend, OdeState,
    RosenbrockMethod, RosenbrockW,
};

const TEST_ARGUMENT: scioxide_core::Argument = scioxide_core::argument!("fixture");

struct StageInverse<T: NumericScalar> {
    inverse: RefCell<Vec<T>>,
    calls: Cell<usize>,
}
impl<T: NumericScalar> Preconditioner<T> for StageInverse<T> {
    fn dimension(&self) -> usize {
        self.inverse.borrow().len()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn precondition_into(&self, x: &[T], out: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        for ((out, &x), &d) in out.iter_mut().zip(x).zip(self.inverse.borrow().iter()) {
            *out = x * d;
        }
        Ok(())
    }
}
struct StageProvider<'a, T: NumericScalar> {
    provider: &'a ReactionDiffusion<'a, T>,
    scales: &'a [T::Real],
    matrix: &'a CsrMatrix<T>,
    blocks: [&'a StageInverse<T>; 2],
    inverse: &'a dyn Preconditioner<T>,
    prepares: Cell<usize>,
}
impl<T: NumericScalar> MethodOfLinesStagePreconditioner<T::Real, T> for StageProvider<'_, T> {
    fn preconditioner(
        &self,
        _: T::Real,
        z: &[T],
        shift: T::Real,
    ) -> Result<Option<&dyn Preconditioner<T>>, SciOxideError> {
        self.prepares.set(self.prepares.get() + 1);
        for i in 0..3 {
            let j = self.provider.derivative(
                z[i] * T::from_real(self.scales[i]),
                z[3 + i] * T::from_real(self.scales[3 + i]),
            );
            for (field, block) in self.blocks.iter().enumerate() {
                let diagonal = -ReactionDiffusion::<T>::diffusivity(field)
                    * self.matrix.get(i, i)?
                    / T::from_real(self.provider.mass[i])
                    + j[field][field];
                block.inverse.borrow_mut()[i] = T::ONE / (T::ONE - T::from_real(shift) * diagonal);
            }
        }
        Ok(Some(self.inverse))
    }
}
fn evolve<T: NumericScalar>() {
    let (a, csr, mass, target, laplace, pattern) = fixture::<T>(3);
    let mut provider = ReactionDiffusion::new(&a, &mass, &target, &laplace).unwrap();
    provider.assembled = Some(&csr);
    provider.decay = real(0.8);
    let scales = vec![
        real(0.2),
        real(0.3),
        real(0.5),
        real(4.),
        real(6.),
        real(8.),
    ];
    let all_mass: Vec<_> = mass.iter().chain(&mass).copied().collect();
    let plan = MethodOfLinesPlan::new(6, Some(&all_mass), Some(&scales)).unwrap();
    let fields = PdeFieldLayout::new(&[3, 3]).unwrap();
    let first = StageInverse {
        inverse: RefCell::new(vec![T::ONE; 3]),
        calls: Cell::new(0),
    };
    let second = StageInverse {
        inverse: RefCell::new(vec![T::ONE; 3]),
        calls: Cell::new(0),
    };
    let blocks: [&dyn Preconditioner<T>; 2] = [&first, &second];
    let pre = PdeFieldPreconditioner::new(&fields, &blocks).unwrap();
    let stage = StageProvider {
        provider: &provider,
        scales: &scales,
        matrix: &csr,
        blocks: [&first, &second],
        inverse: &pre,
        prepares: Cell::new(0),
    };
    let loose = T::Real::EPSILON > real(1e-10);
    let tolerance = ErrorTolerance::try_vector(
        &[real(if loose { 2e-5 } else { 1e-10 }); 6],
        real(if loose { 2e-5 } else { 1e-10 }),
        NormSelection::Infinity,
    )
    .unwrap();
    for backend in [
        OdeStageBackend::Dense,
        OdeStageBackend::Sparse {
            pattern,
            limits: scioxide_sparse::SparseLuLimits::new(72, 65536),
        },
        OdeStageBackend::Krylov {
            iterations: Budget::try_from(40).unwrap(),
            restart: Budget::try_from(6).unwrap(),
            operator_scratch: 0,
            preconditioner_scratch: pre.precondition_workspace_len().unwrap(),
        },
    ] {
        let is_krylov = matches!(backend, OdeStageBackend::Krylov { .. });
        let mut w = plan.workspace().unwrap();
        let system = plan
            .bind(&provider, &mut w)
            .unwrap()
            .with_stage_preconditioner(&stage);
        let mut initial = vec![T::ZERO; 6];
        plan.encode_into(&target, &mut initial).unwrap();
        let solver = (
            RosenbrockW::new(RosenbrockMethod::Ros34Pw2)
                .unwrap()
                .with_backend(backend),
            OdeOptions::new()
                .with_error_tolerance(plan.error_tolerance(&tolerance).unwrap())
                .with_initial_step(real(0.01)),
        );
        let problem = OdeProblem::new(&system, real(0.2));
        let mut state = OdeState::new(real(0.), scioxide_linalg::Vector::from_vec(initial));
        let reports = Cell::new(0);
        let report = solver
            .0
            .advance(
                &problem,
                &solver.1,
                &mut state,
                &mut |_, _: &[T]| {
                    reports.set(reports.get() + 1);
                    Ok(())
                },
                &mut solver.0.workspace(&problem, &solver.1).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success(), "{:?}", report.termination());
        let mut physical = vec![T::ZERO; 6];
        plan.decode_into(state.values(), &mut physical).unwrap();
        let mut expected = physical.clone();
        provider.exact_into(real(0.2), &mut expected);
        for (&x, &y) in physical.iter().zip(&expected) {
            assert!(
                (x - y).abs() < real(if loose { 2e-4 } else { 2e-8 }),
                "error {}",
                (x - y).abs().into_f64()
            );
        }
        assert!(reports.get() > 0);
        if is_krylov {
            assert!(stage.prepares.get() > 0);
            assert!(first.calls.get() > 0);
            assert_eq!(first.calls.get(), second.calls.get());
        }
    }
}
#[test]
fn stiff_transients_preserve_scaling_across_four_domains_and_backends() {
    evolve::<f32>();
    evolve::<f64>();
    evolve::<Complex<f32>>();
    evolve::<Complex<f64>>();
}

#[test]
fn normalization_requires_explicit_stage_provider_and_rejects_invalid_requests() {
    let fields = PdeFieldLayout::new(&[1]).unwrap();
    let p = Inverse(vec![2.]);
    let blocks: [&dyn Preconditioner<f64>; 1] = [&p];
    let pre = PdeFieldPreconditioner::new(&fields, &blocks).unwrap();
    let plan = MethodOfLinesPlan::new(1, Some(&[2.]), Some(&[4.])).unwrap();
    let mut w = plan.workspace().unwrap();
    let system = plan.bind(&Scalar, &mut w).unwrap();
    assert!(
        context(|cx| system
            .stage_preconditioner(0., &[1.], 0.1, cx)
            .map(|p| p.is_none()))
        .unwrap()
    );
    struct Stage<'a>(&'a dyn Preconditioner<f64>, Cell<usize>);
    impl MethodOfLinesStagePreconditioner<f64> for Stage<'_> {
        fn preconditioner(
            &self,
            time: f64,
            z: &[f64],
            shift: f64,
        ) -> Result<Option<&dyn Preconditioner<f64>>, SciOxideError> {
            self.1.set(self.1.get() + 1);
            assert_eq!((time, z[0], shift), (0.3, 0.5, 0.1));
            Ok(Some(self.0))
        }
    }
    let stage = Stage(&pre, Cell::new(0));
    let system = system.with_stage_preconditioner(&stage);
    let prepare = |time: f64, z: &[f64]| {
        context(|cx| {
            system
                .stage_preconditioner(time, z, 0.1, cx)
                .map(|p| p.is_some())
        })
    };
    assert!(prepare(f64::NAN, &[0.5]).is_err());
    assert!(prepare(0.3, &[f64::NAN]).is_err());
    assert!(prepare(0.3, &[]).is_err());
    assert_eq!(stage.1.get(), 0);
    assert!(prepare(0.3, &[0.5]).unwrap());
    assert_eq!(stage.1.get(), 1);
}

#[test]
fn conservative_exchange_balance_and_time_derivative() {
    let (a, _, mass, target, laplace, _) = fixture::<f64>(3);
    let mut provider = ReactionDiffusion::new(&a, &mass, &target, &laplace).unwrap();
    provider.loss = 0.;
    provider.decay = 0.7;
    for (u, v) in [(0.2, 0.3), (2., -0.5)] {
        let r = provider.reaction(u, v);
        assert_eq!(r[0] + r[1], 0.);
    }
    // Summing the integrated equations cancels interior diffusion fluxes and
    // cubic exchange. Only the two exterior fluxes and manufactured sources remain.
    let state: Vec<_> = target
        .iter()
        .enumerate()
        .map(|(i, x)| x * (1. + 0.2 * i as f64))
        .collect();
    let mut balance = [0.; 6];
    context(|cx| provider.rhs(0.2, &state, &mut balance, cx)).unwrap();
    let e = (-0.7_f64 * 0.2).exp();
    let boundary = -0.05 * (state[0] + state[2]) / mass[0] - 0.1 * (state[3] + state[5]) / mass[0];
    let source: f64 = (0..3)
        .map(|i| {
            mass[i]
                * e
                * (-0.7 * (target[i] + target[3 + i]) + 0.05 * laplace[i] + 0.1 * laplace[3 + i])
        })
        .sum();
    assert!((balance.iter().sum::<f64>() - boundary - source).abs() < 1e-13);
    let mut f = [0.; 6];
    let mut fp = f;
    let mut fm = f;
    context(|cx| {
        provider.time_derivative(0.2, &target, &mut f, cx)?;
        provider.rhs(0.20001, &target, &mut fp, cx)?;
        provider.rhs(0.19999, &target, &mut fm, cx)
    })
    .unwrap();
    for i in 0..6 {
        assert!((f[i] - (fp[i] - fm[i]) / 0.00002).abs() < 1e-10);
    }
}

#[test]
fn failed_stage_and_failed_stream_preserve_accepted_time_and_state() {
    let (a, _, mass, target, laplace, _) = fixture::<f64>(3);
    let mut provider = ReactionDiffusion::new(&a, &mass, &target, &laplace).unwrap();
    provider.decay = 0.8;
    let all_mass: Vec<_> = mass.iter().chain(&mass).copied().collect();
    let plan = MethodOfLinesPlan::new(6, Some(&all_mass), None).unwrap();
    let mut work = plan.workspace().unwrap();
    struct FailingStage;
    impl MethodOfLinesStagePreconditioner<f64> for FailingStage {
        fn preconditioner(
            &self,
            _: f64,
            _: &[f64],
            _: f64,
        ) -> Result<Option<&dyn Preconditioner<f64>>, SciOxideError> {
            Err(invalid(TEST_ARGUMENT, "injected stage preparation failure"))
        }
    }
    let backend = || OdeStageBackend::Krylov {
        iterations: Budget::try_from(30).unwrap(),
        restart: Budget::try_from(6).unwrap(),
        operator_scratch: 0,
        preconditioner_scratch: 0,
    };
    let solver = (
        RosenbrockW::new(RosenbrockMethod::Ros34Pw2)
            .unwrap()
            .with_backend(backend()),
        OdeOptions::new()
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps)
            .with_error_tolerance(
                ErrorTolerance::try_vector(&[1e-9; 6], 1e-9, NormSelection::Infinity).unwrap(),
            )
            .with_initial_step(0.01)
            .with_maximum_step(0.01),
    );
    let system = plan
        .bind(&provider, &mut work)
        .unwrap()
        .with_stage_preconditioner(&FailingStage);
    let problem = OdeProblem::new(&system, 0.2);
    let mut state = OdeState::new(0., scioxide_linalg::Vector::from_vec(target.clone()));
    assert!(
        solver
            .0
            .advance(
                &problem,
                &solver.1,
                &mut state,
                &mut (),
                &mut solver.0.workspace(&problem, &solver.1).unwrap(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(state.time(), 0.);
    assert_eq!(state.values(), target);
    let system = plan.bind(&provider, &mut work).unwrap();
    let problem = OdeProblem::new(&system, 0.2);
    let mut state = OdeState::new(0., scioxide_linalg::Vector::from_vec(target.clone()));
    let mut emitted = 0;
    let mut previous = (0., target.clone());
    assert!(
        solver
            .0
            .advance(
                &problem,
                &solver.1,
                &mut state,
                &mut |t, x: &[f64]| {
                    emitted += 1;
                    if emitted == 3 {
                        return Err(invalid(TEST_ARGUMENT, "injected output failure"));
                    }
                    previous = (t, x.to_vec());
                    Ok(())
                },
                &mut solver.0.workspace(&problem, &solver.1).unwrap(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(state.time(), previous.0);
    assert_eq!(state.values(), previous.1);
    let mut expected = vec![0.; 6];
    provider.exact_into(state.time(), &mut expected);
    assert!(
        state
            .values()
            .iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
}

#[test]
fn nonlinear_backward_euler_stages_have_first_order_temporal_convergence() {
    let (a, csr, mass, target, laplace, pattern) = fixture::<f64>(3);
    let mut provider = ReactionDiffusion::new(&a, &mass, &target, &laplace).unwrap();
    provider.assembled = Some(&csr);
    provider.decay = 0.8;
    let all_mass: Vec<_> = mass.iter().chain(&mass).copied().collect();
    let plan = MethodOfLinesPlan::new(6, Some(&all_mass), None).unwrap();
    let mut expected = target.clone();
    provider.exact_into(0.2, &mut expected);
    for backend in [
        OdeStageBackend::Dense,
        OdeStageBackend::Sparse {
            pattern,
            limits: scioxide_sparse::SparseLuLimits::new(72, 65536),
        },
        OdeStageBackend::Krylov {
            iterations: Budget::try_from(40).unwrap(),
            restart: Budget::try_from(6).unwrap(),
            operator_scratch: 0,
            preconditioner_scratch: 0,
        },
    ] {
        let mut errors = Vec::new();
        for step in [0.05, 0.025] {
            let mut work = plan.workspace().unwrap();
            let system = plan.bind(&provider, &mut work).unwrap();
            let solve = (
                scioxide_ode::BackwardEuler::new()
                    .with_newton_mode(scioxide_ode::NewtonMode::Full)
                    .with_backend(backend.clone()),
                OdeOptions::new()
                    .with_initial_step(step)
                    .with_maximum_step(step)
                    .with_error_tolerance(
                        ErrorTolerance::try_vector(&[1e-10; 6], 1e-10, NormSelection::Infinity)
                            .unwrap(),
                    ),
            );
            let solution = solve
                .0
                .solve(
                    &OdeProblem::new(&system, 0.2),
                    &solve.1,
                    OdeState::new(0., scioxide_linalg::Vector::from_vec(target.clone())),
                )
                .unwrap();
            assert!(solution.report().is_success());
            errors.push(
                solution
                    .state()
                    .values()
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max),
            );
        }
        assert!(errors[0] / errors[1] > 1.8, "{errors:?}");
        assert!(errors[1] < 0.002, "{errors:?}");
    }
}
