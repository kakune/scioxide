//! Exact affine stage work and accepted endpoint derivative reuse.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use core::cell::Cell;
use scioxide_core::{
    ErrorTolerance, EvaluationRole, NormSelection, SciOxideError, SolveContext, StructuralPattern,
    StructuralPatternView,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::*;
use scioxide_sparse::SparseLuLimits;

struct Heat {
    n: usize,
    constant: bool,
    calls: Cell<usize>,
}
impl Heat {
    fn new(n: usize, constant: bool) -> Self {
        Self {
            n,
            constant,
            calls: Cell::new(0),
        }
    }
    fn scale(&self) -> f64 {
        ((self.n + 1) as f64).powi(2)
    }
}
impl OdeSystem<f64> for Heat {
    fn dimension(&self) -> usize {
        self.n
    }
    fn jacobian_is_constant(&self) -> bool {
        self.constant
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        for (i, out) in out.iter_mut().enumerate() {
            let left = i.checked_sub(1).map_or(0., |j| y[j]);
            let right = y.get(i + 1).copied().unwrap_or(0.);
            *out = self.scale() * (left - 2. * y[i] + right);
        }
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..self.n {
            for j in 0..self.n {
                out.set(
                    i,
                    j,
                    self.scale()
                        * if i == j {
                            -2.
                        } else if i.abs_diff(j) == 1 {
                            1.
                        } else {
                            0.
                        },
                )?;
            }
        }
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: f64,
        _: &[f64],
        pattern: StructuralPatternView<'_>,
        values: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (row, bounds) in pattern.row_ptr().windows(2).enumerate() {
            for (value, &column) in values[bounds[0]..bounds[1]]
                .iter_mut()
                .zip(&pattern.col_indices()[bounds[0]..bounds[1]])
            {
                *value = self.scale() * if column == row { -2. } else { 1. };
            }
        }
        Ok(Some(()))
    }
}
fn sparse(n: usize) -> OdeStageBackend {
    let mut rows = vec![0];
    let mut columns = Vec::new();
    for row in 0..n {
        columns.extend(row.saturating_sub(1)..(row + 2).min(n));
        rows.push(columns.len());
    }
    OdeStageBackend::Sparse {
        pattern: StructuralPattern::new(n, n, rows, columns).unwrap(),
        limits: SparseLuLimits::new(n * n * 16, 64 * 1024 * 1024),
    }
}
fn options(h: f64) -> OdeOptions<f64> {
    OdeOptions::new()
        .with_initial_step(h)
        .with_error_tolerance(ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::Infinity).unwrap())
}
fn heat(method: BuiltinOdeMethod<f64>, n: usize, rhs: usize, corrections: usize) {
    let initial: Vec<_> = (1..=n)
        .map(|i| (core::f64::consts::PI * i as f64 / (n + 1) as f64).sin())
        .collect();
    let plan = (method, options(1e-3));
    let system = Heat::new(n, true);
    let result = plan
        .0
        .solve(
            &OdeProblem::new(&system, 0.2),
            &plan.1,
            OdeState::from_slice(0., &initial).unwrap(),
        )
        .unwrap();
    let report = result.report();
    assert!(report.is_success());
    assert_eq!(report.accepted_steps(), 200);
    assert_eq!(report.rejected_steps(), 0);
    assert!(
        system.calls.get() <= 1 + rhs * 200,
        "rhs={}",
        system.calls.get()
    );
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        system.calls.get()
    );
    assert_eq!(
        report.counts().provider(EvaluationRole::LinearSolve),
        corrections * 200
    );
    assert_eq!(report.accounting().nested().iterations(), corrections * 200);
    // A conservative declaration exercises the ordinary Newton convergence path.
    let reference = plan
        .0
        .solve(
            &OdeProblem::new(&Heat::new(n, false), 0.2),
            &plan.1,
            OdeState::from_slice(0., &initial).unwrap(),
        )
        .unwrap();
    assert!(reference.report().is_success());
    for (&actual, &expected) in result
        .state()
        .values()
        .iter()
        .zip(reference.state().values())
    {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }
}
#[test]
fn dense_and_sparse_affine_stages_need_one_correction() {
    for n in [16, 64] {
        for backend in [OdeStageBackend::Dense, sparse(n)] {
            heat(
                BackwardEuler::new().with_backend(backend.clone()).into(),
                n,
                2,
                1,
            );
            heat(
                ImplicitTrapezoid::new()
                    .with_backend(backend.clone())
                    .into(),
                n,
                2,
                1,
            );
            heat(GaussRk::new().with_backend(backend.clone()).into(), n, 4, 1);
            heat(Sdirk::new().with_backend(backend.clone()).into(), n, 6, 3);
            heat(Bdf::new(2).unwrap().with_backend(backend).into(), n, 2, 1);
        }
    }
}

struct Forced {
    calls: Cell<usize>,
}
impl OdeSystem<f64> for Forced {
    fn dimension(&self) -> usize {
        1
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        // Exact solution y=t², with an arbitrary initial transient.
        out[0] = -2. * y[0] + 2. * t * t + 2. * t;
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, -2.)?;
        Ok(Some(()))
    }
    fn automatic_jacobian(
        &self,
        t: f64,
        y: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.jacobian(t, y, out, cx)
    }
}
#[test]
fn gauss_affine_transport_handles_time_dependent_forcing_and_full_newton() {
    for mode in [NewtonMode::Simplified, NewtonMode::Full] {
        for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Automatic] {
            let system = Forced {
                calls: Cell::new(0),
            };
            let method = GaussRk::new()
                .with_newton_mode(mode)
                .with_derivative_strategy(strategy);
            let plan = (method, options(1. / 16.));
            let result = plan
                .0
                .solve(
                    &OdeProblem::new(&system, 1.),
                    &plan.1,
                    OdeState::from_slice(0., &[1.]).unwrap(),
                )
                .unwrap();
            assert!(result.report().is_success());
            assert_eq!(
                result
                    .report()
                    .counts()
                    .provider(EvaluationRole::LinearSolve),
                16
            );
            assert!(system.calls.get() <= 1 + 4 * 16);
            assert!((result.state().values()[0] - (1. + (-2_f64).exp())).abs() < 1e-6);
        }
    }
}
#[test]
fn finite_difference_factors_keep_newton_convergence_tests() {
    let method = GaussRk::new().with_derivative_strategy(DerivativeStrategy::Forward);
    let system = Forced {
        calls: Cell::new(0),
    };
    let plan = (method, options(1. / 16.));
    let result = plan
        .0
        .solve(
            &OdeProblem::new(&system, 1. / 16.),
            &plan.1,
            OdeState::from_slice(0., &[1.]).unwrap(),
        )
        .unwrap();
    assert!(result.report().is_success());
    assert!(
        result
            .report()
            .counts()
            .provider(EvaluationRole::LinearSolve)
            > 1
    );
}

struct AffineState<T> {
    constant: bool,
    marker: core::marker::PhantomData<T>,
}
impl<
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
    T: scioxide_core::NumericScalar<Real = S>,
> OdeSystem<S, T> for AffineState<T>
{
    fn dimension(&self) -> usize {
        1
    }
    fn jacobian_is_constant(&self) -> bool {
        self.constant
    }
    fn rhs(
        &self,
        time: S,
        state: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let slope = T::from_components(-S::TWO, if T::IS_REAL { S::ZERO } else { S::ONE });
        out[0] = slope * state[0] + T::from_real(time * time);
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[T],
        out: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(
            0,
            0,
            T::from_components(-S::TWO, if T::IS_REAL { S::ZERO } else { S::ONE }),
        )?;
        Ok(Some(()))
    }
}
fn scalar_domains<
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
    T: scioxide_core::NumericScalar<Real = S>,
>() {
    let h = S::try_from_f64_lossy(0.125).unwrap();
    let initial = [T::from_components(
        S::ONE,
        if T::IS_REAL { S::ZERO } else { S::ONE },
    )];
    let methods: [BuiltinOdeMethod<S, T>; 4] = [
        BackwardEuler::new().into(),
        ImplicitTrapezoid::new().into(),
        GaussRk::new().into(),
        Sdirk::new().into(),
    ];
    for method in methods {
        let plan = (method, OdeOptions::new().with_initial_step(h));
        let solve = |constant| {
            let system = AffineState {
                constant,
                marker: core::marker::PhantomData,
            };
            plan.0
                .solve(
                    &OdeProblem::new(&system, S::ONE),
                    &plan.1,
                    OdeState::from_slice(S::ZERO, &initial).unwrap(),
                )
                .unwrap()
        };
        let affine = solve(true);
        let reference = solve(false);
        assert!(affine.report().is_success());
        assert!(reference.report().is_success());
        let tolerance = S::EPSILON * S::try_from_f64_lossy(128.).unwrap();
        assert!((affine.state().values()[0] - reference.state().values()[0]).abs() < tolerance);
    }
}
#[test]
fn affine_stages_preserve_real_and_complex_scalar_accuracy() {
    scalar_domains::<f32, f32>();
    scalar_domains::<f64, f64>();
    scalar_domains::<f32, scioxide_core::Complex<f32>>();
    scalar_domains::<f64, scioxide_core::Complex<f64>>();
}

#[test]
fn changed_affine_coefficients_rebuild_exact_factors() {
    let methods: [BuiltinOdeMethod<f64>; 4] = [
        BackwardEuler::new().into(),
        ImplicitTrapezoid::new().into(),
        GaussRk::new().into(),
        Sdirk::new().into(),
    ];
    for method in methods {
        let stages = if matches!(method, BuiltinOdeMethod::Sdirk(_)) {
            3
        } else {
            1
        };
        let plan = (method, options(0.125));
        let result = plan
            .0
            .solve(
                &OdeProblem::new(
                    &Forced {
                        calls: Cell::new(0),
                    },
                    0.2375,
                ),
                &plan.1,
                OdeState::from_slice(0., &[1.]).unwrap(),
            )
            .unwrap();
        assert!(result.report().is_success());
        assert_eq!(result.report().accepted_steps(), 2);
        assert_eq!(
            result
                .report()
                .counts()
                .provider(EvaluationRole::LinearSolve),
            stages * 2
        );
        assert_eq!(
            result
                .report()
                .counts()
                .provider(EvaluationRole::Factorization),
            2
        );
    }
}
