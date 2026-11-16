//! Default termination certificates are invariant to arbitrary function units.
use scioxide_core::{
    Complex, Interval, JacobianOperator, RealScalar, ResidualOperator, ResidualTolerance,
    SciOxideError,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::*;

macro_rules! precision_cases {
    ($module:ident, $r:ty, $tiny:expr, $huge:expr) => {
        mod $module {
            use super::*;
            struct Equation {
                scale: $r,
                kind: usize,
            }
            impl ScalarFunction<$r> for Equation {
                fn eval(&self, x: $r) -> Result<$r, SciOxideError> {
                    Ok(self.scale
                        * match self.kind {
                            0 => x - 1.0,
                            1 => x * x - 2.0,
                            _ => x.cos() - x,
                        })
                }
                fn derivative(&self, x: $r) -> Result<Option<$r>, SciOxideError> {
                    Ok(Some(
                        self.scale
                            * match self.kind {
                                0 => 1.0,
                                1 => 2.0 * x,
                                _ => -x.sin() - 1.0,
                            },
                    ))
                }
                fn second_derivative(&self, x: $r) -> Result<Option<$r>, SciOxideError> {
                    Ok(Some(
                        self.scale
                            * match self.kind {
                                0 => 0.0,
                                1 => 2.0,
                                _ => -x.cos(),
                            },
                    ))
                }
            }
            #[test]
            fn scalar_defaults_certify_coordinates_across_function_scales() {
                for scale in [$tiny, 1e-9, 1.0, $huge] {
                    for kind in 0..3 {
                        let f = Equation { scale, kind };
                        let interval = Interval::new(0.0, 3.0).unwrap();
                        let safeguarded = SafeguardedInitialGuess::new(interval, 0.5).unwrap();
                        let options = ScalarRootOptions::default();
                        let results = [
                            Bisection.solve(&f, interval, &options),
                            RegulaFalsi.solve(&f, interval, &options),
                            Illinois.solve(&f, interval, &options),
                            Ridder.solve(&f, interval, &options),
                            Brent.solve(&f, interval, &options),
                            Brenth.solve(&f, interval, &options),
                            Toms748.solve(&f, interval, &options),
                            Chandrupatla.solve(&f, interval, &options),
                            Newton::new(DerivativeStrategy::Analytic).solve(
                                &f,
                                if kind == 0 { 0.0 } else { 0.5 },
                                &options,
                            ),
                            Halley.solve(&f, 0.5, &options),
                            SafeguardedNewton::new(DerivativeStrategy::Analytic).solve(
                                &f,
                                safeguarded,
                                &options,
                            ),
                            SafeguardedHalley.solve(&f, safeguarded, &options),
                            Secant.solve(&f, (0.0, 0.5), &options),
                            Muller.solve(&f, (0.0, 0.25, 0.5), &options),
                        ];
                        let root = match kind {
                            0 => 1.0,
                            1 => (2.0 as $r).sqrt(),
                            _ => 0.7390851332151607_f64 as $r,
                        };
                        for (method, outcome) in results.into_iter().enumerate() {
                            let result = outcome.unwrap();
                            if method >= 8 {
                                assert!(
                                    result.is_success(),
                                    "scale={scale} kind={kind} method={method}: {result:?}"
                                );
                            }
                            if result.is_success() {
                                assert!(
                                    (result.point() - root).abs() <= 32.0 * <$r>::EPSILON,
                                    "scale={scale} kind={kind}: {result:?}"
                                );
                                match result.termination() {
                                    RootTermination::ResidualTolerance => {
                                        assert_eq!(result.residual(), 0.0)
                                    }
                                    RootTermination::BracketTolerance => {
                                        let b = result.bracket().unwrap();
                                        assert!(
                                            options
                                                .step_tolerance()
                                                .tolerance()
                                                .is_converged(b.lower(), b.upper())
                                                .unwrap()
                                        );
                                    }
                                    RootTermination::StepTolerance => assert!(
                                        (f.eval(result.point()).unwrap() / scale).abs()
                                            <= 4.0 * <$r>::EPSILON
                                    ),
                                    other => assert_eq!(other, RootTermination::StepTolerance),
                                }
                            } else {
                                assert_ne!(
                                    result.termination(),
                                    RootTermination::ResidualTolerance
                                );
                            }
                        }
                    }
                }
            }
            #[test]
            fn initial_and_endpoint_exact_zeros_need_no_iteration() {
                for scale in [$tiny, 1.0, $huge] {
                    let f = Equation { scale, kind: 0 };
                    let opts = ScalarRootOptions::default();
                    let interval = Interval::new(1.0, 3.0).unwrap();
                    for result in [
                        Bisection.solve(&f, interval, &opts),
                        Brent.solve(&f, interval, &opts),
                        Toms748.solve(&f, interval, &opts),
                        Illinois.solve(&f, interval, &opts),
                        Ridder.solve(&f, interval, &opts),
                        Newton::new(DerivativeStrategy::Analytic).solve(&f, 1.0, &opts),
                        Halley.solve(&f, 1.0, &opts),
                        Secant.solve(&f, (1.0, 2.0), &opts),
                        Muller.solve(&f, (1.0, 2.0, 3.0), &opts),
                    ] {
                        let result = result.unwrap();
                        assert!(result.is_success());
                        assert_eq!(result.point(), 1.0);
                        assert_eq!(result.counts().iterations(), 0);
                    }
                }
            }
            #[test]
            fn complex_defaults_do_not_accept_small_function_units() {
                let target = Complex::new(1.0 as $r, -0.5);
                let initial = Complex::new(0.0, 0.0);
                for scale in [$tiny, 1e-9, 1.0, $huge] {
                    let f = ComplexScalarFn::new(|z: Complex<$r>| Ok((z - target) * scale))
                        .with_derivative(|_| Ok(Complex::new(scale, 0.0)))
                        .with_second_derivative(|_| Ok(Complex::new(0.0, 0.0)));
                    let options = ScalarRootOptions::default();
                    for result in [
                        ComplexNewton.solve(&f, initial, &options),
                        ComplexHalley.solve(&f, initial, &options),
                        ComplexSecant.solve(&f, (initial, Complex::new(0.5, 0.0)), &options),
                        ComplexMuller.solve(
                            &f,
                            (initial, Complex::new(0.25, 0.0), Complex::new(0.5, 0.0)),
                            &options,
                        ),
                    ] {
                        let result = result.unwrap();
                        assert!(result.is_success(), "{result:?}");
                        assert!(
                            (result.point() - target).norm() <= 16.0 * <$r>::EPSILON,
                            "{result:?}"
                        );
                    }
                }
            }
            struct Linear($r);
            impl ResidualOperator<$r> for Linear {
                fn input_dimension(&self) -> usize {
                    2
                }
                fn output_dimension(&self) -> usize {
                    2
                }
                fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
                    Ok(0)
                }
                fn residual_into(
                    &self,
                    x: &[$r],
                    f: &mut [$r],
                    _: &mut [$r],
                ) -> Result<(), SciOxideError> {
                    f[0] = self.0 * (x[0] - 1.0);
                    f[1] = self.0 * (x[1] + 2.0);
                    Ok(())
                }
            }
            impl JacobianOperator<$r> for Linear {
                fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
                    Ok(0)
                }
                fn jvp_into(
                    &self,
                    _: &[$r],
                    v: &[$r],
                    out: &mut [$r],
                    _: &mut [$r],
                ) -> Result<(), SciOxideError> {
                    out[0] = self.0 * v[0];
                    out[1] = self.0 * v[1];
                    Ok(())
                }
            }
            impl DenseJacobian<$r> for Linear {
                fn jacobian_into(
                    &self,
                    _: &[$r],
                    j: &mut [$r],
                    _: &mut [$r],
                ) -> Result<bool, SciOxideError> {
                    j.copy_from_slice(&[self.0, 0.0, 0.0, self.0]);
                    Ok(true)
                }
            }
            #[test]
            fn system_defaults_reduce_the_initial_residual() {
                for scale in [$tiny, 1e-9, 1.0, $huge] {
                    // One residual type serves the dense, fixed-point and
                    // Newton–Krylov solvers.
                    let linear = Linear(scale);
                    let f = &linear;

                    for start in [[0.0, 0.0], [1.0, -2.0]] {
                        for result in [
                            NewtonSystem::new(
                                (scioxide_rootfind::NewtonSystemOptions::default())
                                    .with_derivative_strategy(DerivativeStrategy::Analytic),
                            )
                            .unwrap()
                            .solve(f, &start),
                            Broyden::new(
                                BroydenUpdate::First,
                                (scioxide_rootfind::BroydenOptions::default())
                                    .with_derivative_strategy(DerivativeStrategy::Analytic),
                            )
                            .unwrap()
                            .solve(f, &start),
                            Broyden::new(
                                BroydenUpdate::Second,
                                (scioxide_rootfind::BroydenOptions::default())
                                    .with_derivative_strategy(DerivativeStrategy::Analytic),
                            )
                            .unwrap()
                            .solve(f, &start),
                            TrustRegionSystem::new(
                                (scioxide_rootfind::TrustRegionSystemOptions::default())
                                    .with_derivative_strategy(DerivativeStrategy::Analytic),
                            )
                            .unwrap()
                            .solve(f, &start),
                        ] {
                            let result = result.unwrap();
                            assert!(result.is_success(), "{result:?}");
                            assert!((result.point()[0] - 1.0).abs() <= 8.0 * <$r>::SQRT_EPSILON);
                            assert!((result.point()[1] + 2.0).abs() <= 8.0 * <$r>::SQRT_EPSILON);
                            assert_eq!(result.iterations() == 0, start[0] == 1.0);
                        }
                        let problem = NewtonKrylovProblem::with_jacobian(&linear);
                        let options = NewtonKrylovOptions::default();
                        let result = NewtonKrylovSolver::new((options).clone())
                            .and_then(|root_plan| root_plan.solve(&problem, &start))
                            .unwrap();
                        assert!(result.is_success(), "{result:?}");
                        assert!((result.point()[0] - 1.0).abs() <= 8.0 * <$r>::SQRT_EPSILON);
                        assert!((result.point()[1] + 2.0).abs() <= 8.0 * <$r>::SQRT_EPSILON);
                        assert_eq!(result.iterations() == 0, start[0] == 1.0);
                    }
                }
            }
            #[test]
            fn mixing_defaults_never_accept_an_arbitrarily_small_nonzero_start() {
                for method in [
                    FixedPointMethod::Picard,
                    FixedPointMethod::Steffensen,
                    FixedPointMethod::Anderson,
                    FixedPointMethod::BroydenFirst,
                    FixedPointMethod::BroydenSecond,
                    FixedPointMethod::DiagonalBroyden,
                    FixedPointMethod::ExcitingMixing,
                ] {
                    for scale in [$tiny, 1e-9, 1.0, $huge] {
                        let linear = Linear(scale);
                        let opts = FixedPointOptions::default();
                        let solver = FixedPointSolver::new(
                            method,
                            FixedPointInput::Residual,
                            (opts).clone(),
                        )
                        .unwrap();
                        if let Ok(result) = solver.solve(&linear, &[0.0, 0.0]) {
                            if result.is_success() {
                                assert!(
                                    (result.point()[0] - 1.0).abs() <= 8.0 * <$r>::SQRT_EPSILON,
                                    "{method:?}: {result:?}"
                                );
                                assert!(
                                    (result.point()[1] + 2.0).abs() <= 8.0 * <$r>::SQRT_EPSILON
                                );
                                assert!(result.counts().iterations() > 0);
                            }
                        }
                        let result = solver.solve(&linear, &[1.0, -2.0]).unwrap();
                        assert!(result.is_success());
                        assert_eq!(result.counts().iterations(), 0);
                    }
                }
            }
        }
    };
}
precision_cases!(single, f32, 1e-30, 1e30);
precision_cases!(double, f64, 1e-300, 1e300);

#[test]
fn default_brent_and_toms_accuracy() {
    let f = ScalarFn::new(|x: f64| Ok(x * x - 2.0));
    for result in [
        Brent.solve(
            &f,
            Interval::new(1.0, 2.0).unwrap(),
            &ScalarRootOptions::default(),
        ),
        Toms748.solve(
            &f,
            Interval::new(1.0, 2.0).unwrap(),
            &ScalarRootOptions::default(),
        ),
    ] {
        let result = result.unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!(
            (result.point() - 2.0_f64.sqrt()).abs()
                <= 2e-12 + 4.0 * f64::EPSILON * result.point().abs()
        );
    }
}

#[test]
fn explicit_absolute_tolerance_remains_an_opt_in() {
    let f = ScalarFn::new(|x: f64| Ok(1e-9 * (x - 1.0)));
    let opts = ScalarRootOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(1e-8, 0.0).unwrap());
    let result = Brent
        .solve(&f, Interval::new(0.0, 3.0).unwrap(), &opts)
        .unwrap();
    assert!(result.is_success());
    assert_eq!(result.point(), 0.0);
}

#[test]
fn broyden_tridiagonal_defaults_keep_improving_small_steps() {
    let n = 16;
    let f = ResidualFn::new(n, |x: &[f64], f: &mut [f64]| {
        for (i, out) in f.iter_mut().enumerate() {
            *out = (3.0 - 2.0 * x[i]) * x[i] + 1.0
                - if i > 0 { x[i - 1] } else { 0.0 }
                - if i + 1 < n { 2.0 * x[i + 1] } else { 0.0 };
        }
        Ok(())
    })
    .with_jacobian(|x: &[f64], j: &mut [f64]| {
        j.fill(0.0);
        for (i, row) in j.chunks_exact_mut(n).enumerate() {
            row[i] = 3.0 - 4.0 * x[i];
            if i > 0 {
                row[i - 1] = -1.0;
            }
            if i + 1 < n {
                row[i + 1] = -2.0;
            }
        }
        Ok(())
    });
    for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Forward] {
        let result = Broyden::new(
            BroydenUpdate::First,
            (scioxide_rootfind::BroydenOptions::default()).with_derivative_strategy(strategy),
        )
        .unwrap()
        .solve(&f, &[-1.0; 16])
        .unwrap();
        assert!(result.is_success(), "{strategy:?}: {result:?}");
        let mut residual = vec![0.0; n];
        f.residual_into(result.point(), &mut residual, &mut [])
            .unwrap();
        assert!(residual.iter().all(|r| r.abs() <= 1e-7));
    }
}

#[test]
fn open_success_has_a_small_independent_newton_correction() {
    let f = ScalarFn::new(|x: f64| Ok(x * x - 2.0)).with_derivative(|x| Ok(2.0 * x));
    let opts = ScalarRootOptions::default();
    let result = Newton::new(DerivativeStrategy::Analytic)
        .solve(&f, 1.0, &opts)
        .unwrap();
    assert_eq!(result.termination(), RootTermination::StepTolerance);
    let x = result.point();
    let correction = (x * x - 2.0) / (2.0 * x);
    assert!(correction.abs() <= opts.step_tolerance().threshold(x.abs()).unwrap());
    assert!((x * x - 2.0).abs() <= f64::EPSILON.sqrt());
}

#[test]
fn small_nonimproving_open_steps_are_unsuccessful_by_default() {
    let f = ScalarFn::new(|_: f64| Ok(1e-300)).with_derivative(|_| Ok(1.0));
    let result = Newton::new(DerivativeStrategy::Analytic)
        .solve(&f, 1.0, &ScalarRootOptions::default())
        .unwrap();
    assert_eq!(result.termination(), RootTermination::Stagnation);
    let callback = ResidualFn::new(1, |_: &[f64], out: &mut [f64]| {
        out[0] = 1e-300;
        Ok(())
    });
    for method in [
        FixedPointMethod::Picard,
        FixedPointMethod::Anderson,
        FixedPointMethod::BroydenFirst,
    ] {
        let result = FixedPointSolver::new(
            method,
            FixedPointInput::Residual,
            (FixedPointOptions::default()).clone(),
        )
        .and_then(|solver| solver.solve(&callback, &[1.0]))
        .unwrap();
        assert_eq!(result.termination(), RootTermination::Stagnation);
    }
}

#[test]
fn system_defaults_reject_stalled_small_nonzero_residuals() {
    let f = ResidualFn::new(1, |_: &[f64], out: &mut [f64]| {
        out[0] = 1e-300;
        Ok(())
    })
    .with_jacobian(|_: &[f64], j: &mut [f64]| {
        j[0] = 1.0;
        Ok(())
    });

    for outcome in [
        NewtonSystem::new(
            (scioxide_rootfind::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap()
        .solve(&f, &[1.0]),
        Broyden::new(
            BroydenUpdate::First,
            (scioxide_rootfind::BroydenOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap()
        .solve(&f, &[1.0]),
        TrustRegionSystem::new(
            (scioxide_rootfind::TrustRegionSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap()
        .solve(&f, &[1.0]),
    ] {
        let result = outcome.unwrap();
        assert_eq!(result.termination(), RootTermination::Stagnation);
        assert_eq!(result.point(), &[1.0]);
    }
    struct Stalled;
    impl ResidualOperator<f64> for Stalled {
        fn input_dimension(&self) -> usize {
            1
        }
        fn output_dimension(&self) -> usize {
            1
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            _: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = 1e-300;
            Ok(())
        }
    }
    impl JacobianOperator<f64> for Stalled {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn jvp_into(
            &self,
            _: &[f64],
            v: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(v);
            Ok(())
        }
    }
    let result = NewtonKrylovSolver::new(NewtonKrylovOptions::default())
        .and_then(|root_plan| {
            root_plan.solve(&NewtonKrylovProblem::with_jacobian(&Stalled), &[1.0])
        })
        .unwrap();
    assert_eq!(result.termination(), RootTermination::Stagnation);
}

#[test]
fn unrepresentable_component_scaled_residual_cannot_become_an_exact_zero() {
    let callback = ResidualFn::new(1, |_: &[f64], out: &mut [f64]| {
        out[0] = 1e-300;
        Ok(())
    });
    let options = FixedPointOptions::default().with_component_scales(vec![1e300]);
    let error = FixedPointSolver::new(
        FixedPointMethod::Anderson,
        FixedPointInput::Residual,
        options,
    )
    .and_then(|solver| solver.solve(&callback, &[0.0]))
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
}
