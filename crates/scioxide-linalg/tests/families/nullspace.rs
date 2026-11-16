#![allow(clippy::unwrap_used)]
use scioxide_core::{
    BlockVectorView, Budget, Complex, LinearOperator, NumericScalar, RealScalar, ResidualTolerance,
    Tolerance,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BorderedSystem, BorderedSystemWorkspaceRequest, CompatibilityPolicy, GmresSolver,
    LinalgWorkspace, Matrix,
};

fn qualify<S: NumericScalar>() {
    let r = |x: f64| S::from_real(S::Real::try_from_f64_lossy(x).unwrap());
    // A has distinct left and right nullspaces; the left nullspace is e2.
    let a = Matrix::from_vec(2, 2, vec![r(1.0), r(-1.0), r(0.0), r(0.0)]).unwrap();
    let left = [r(0.0), r(1.0)];
    let right = [r(0.5).real_component().sqrt(); 2].map(S::from_real);
    let tolerance = Tolerance::new(
        S::Real::try_from_f64_lossy(2e-5).unwrap(),
        S::Real::try_from_f64_lossy(2e-5).unwrap(),
    )
    .unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let plan = BorderedSystem::new(
        &a_finite,
        BlockVectorView::new(&left, 2, 1).unwrap(),
        BlockVectorView::new(&right, 2, 1).unwrap(),
        S::Real::try_from_f64_lossy(2.0).unwrap(),
        &scioxide_linalg::BorderedSystemOptions::new()
            .with_validation_tolerance(tolerance)
            .with_compatibility_tolerance(tolerance),
    )
    .unwrap();
    let rhs = [r(2.0), r(3.0)];
    assert!(
        plan.prepare_rhs(
            scioxide_linalg::VectorView::from_slice(&rhs),
            scioxide_linalg::VectorView::from_slice(&[r(0.0)]),
            CompatibilityPolicy::RequireCompatible
        )
        .is_err()
    );
    let load = plan
        .prepare_rhs(
            scioxide_linalg::VectorView::from_slice(&rhs),
            scioxide_linalg::VectorView::from_slice(&[r(0.0)]),
            CompatibilityPolicy::Project,
        )
        .unwrap();
    let solver = GmresSolver::new(
        ResidualTolerance::from_tolerance(tolerance),
        Budget::try_from(20).unwrap(),
    )
    .with_restart(Budget::try_from(3).unwrap());
    let augmented = plan.augmented();
    let result = solver
        .solve_with_options(&augmented, &load, Preconditioning::None, None)
        .unwrap();
    assert!(result.report().termination().is_success());
    assert!(
        (result.solution().as_slice()[0] - r(1.0)).abs()
            < S::Real::try_from_f64_lossy(2e-4).unwrap()
    );
    assert!(
        (result.solution().as_slice()[1] + r(1.0)).abs()
            < S::Real::try_from_f64_lossy(2e-4).unwrap()
    );
    let report = plan
        .diagnose_scoped(
            scioxide_linalg::VectorView::from_slice(&rhs),
            scioxide_linalg::VectorView::from_slice(load.as_slice()),
            scioxide_linalg::VectorView::from_slice(&result.solution().as_slice()[..2]),
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(
        (report.original_residual_norm() - S::Real::try_from_f64_lossy(3.0).unwrap()).abs()
            < S::Real::try_from_f64_lossy(2e-4).unwrap()
    );
    assert!(report.projected_residual_norm() < S::Real::try_from_f64_lossy(2e-4).unwrap());
    assert!(report.gauge_defect() < S::Real::try_from_f64_lossy(2e-4).unwrap());
    let mut output = [r(9.0); 3];
    assert!(
        augmented
            .apply_into(&[r(0.0); 3], &mut output, &mut [])
            .is_err()
    );
    assert!(output.iter().all(|&v| v == r(9.0)));
}
#[test]
fn all_scalar_domains() {
    qualify::<f32>();
    qualify::<f64>();
    qualify::<Complex<f32>>();
    qualify::<Complex<f64>>();
}

#[test]
fn complex_nonconstant_basis_and_multiple_modes() {
    let z = Complex::new(0.0, 0.0);
    let o = Complex::new(1.0, 0.0);
    let i = Complex::new(0.0, 1.0);
    let a = Matrix::from_vec(3, 3, vec![o, i, z, -i, o, z, z, z, z]).unwrap();
    let q = 0.5_f64.sqrt();
    let modes = [-i * q, o * q, z, z, z, i];
    let b = BlockVectorView::new(&modes, 3, 2).unwrap();
    let t = Tolerance::new(1e-12, 1e-12).unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let plan = BorderedSystem::new(
        &a_finite,
        b,
        b,
        2.0,
        &scioxide_linalg::BorderedSystemOptions::new()
            .with_validation_tolerance(t)
            .with_compatibility_tolerance(t),
    )
    .unwrap();
    let load = plan
        .prepare_rhs(
            scioxide_linalg::VectorView::from_slice(&[o, -i, z]),
            scioxide_linalg::VectorView::from_slice(&[z, o]),
            CompatibilityPolicy::RequireCompatible,
        )
        .unwrap();
    let result = GmresSolver::new(
        ResidualTolerance::from_tolerance(t),
        Budget::try_from(20).unwrap(),
    )
    .with_restart(Budget::try_from(5).unwrap())
    .solve_with_options(&plan.augmented(), &load, Preconditioning::None, None)
    .unwrap();
    assert!(result.report().termination().is_success());
    assert!((result.solution().as_slice()[2] - i).abs() < 1e-10);
    let bad = [o, z, z, o, z, z];
    let b = BlockVectorView::new(&bad, 3, 2).unwrap();
    assert!(
        BorderedSystem::new(
            &a.view().try_finite().unwrap(),
            b,
            b,
            2.0,
            &scioxide_linalg::BorderedSystemOptions::new()
                .with_validation_tolerance(t)
                .with_compatibility_tolerance(t)
        )
        .is_err()
    );
}

#[test]
fn augmented_solver_respects_its_iteration_limit() {
    use scioxide_core::{
        BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    use scioxide_linalg::{IterativeWorkspace, LinearSolveTermination, Vector};
    let a = Matrix::from_vec(2, 2, vec![1.0, -1.0, -1.0, 1.0]).unwrap();
    let modes = [0.5_f64.sqrt(); 2];
    let basis = BlockVectorView::new(&modes, 2, 1).unwrap();
    let tolerance = Tolerance::new(1e-12, 1e-12).unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let plan = BorderedSystem::new(
        &a_finite,
        basis,
        basis,
        2.0,
        &scioxide_linalg::BorderedSystemOptions::new()
            .with_validation_tolerance(tolerance)
            .with_compatibility_tolerance(tolerance),
    )
    .unwrap();
    let augmented = plan.augmented();
    let b = plan
        .prepare_rhs(
            scioxide_linalg::VectorView::from_slice(&[1.0, -1.0]),
            scioxide_linalg::VectorView::from_slice(&[1.0]),
            CompatibilityPolicy::RequireCompatible,
        )
        .unwrap();
    let solver = GmresSolver::new(ResidualTolerance::from_tolerance(tolerance), Budget::ONE)
        .with_restart(Budget::try_from(3).unwrap());
    let mut workspace = IterativeWorkspace::new();
    let mut x = Vector::zeros(3).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(629),
                scioxide_core::algorithm_id!("tests", "nullspace_parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let report = solver
        .solve_into(
            &augmented,
            &b,
            Preconditioning::None,
            x.view_mut(),
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_eq!(report.termination(), LinearSolveTermination::IterationLimit);
    assert_eq!(report.iterations(), 1);
    // Iterations are local to the child epoch; the parent records them as nested work.
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 1);
}

fn extreme_cancellation<S: NumericScalar>(imaginary: bool) {
    let r = |v| S::Real::try_from_f64_lossy(v).unwrap();
    let zero = S::ZERO;
    let matrix = Matrix::<S>::zeros(4, 4).unwrap();
    let signs = [
        1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
    ];
    let modes = signs.map(|v| {
        if imaginary {
            S::from_components(r(0.0), r(v * 0.5))
        } else {
            S::from_real(r(v * 0.5))
        }
    });
    let basis = BlockVectorView::new(&modes, 4, 4).unwrap();
    let matrix_finite = matrix.view().try_finite().unwrap();
    let plan = BorderedSystem::new(
        &matrix_finite,
        basis,
        basis,
        r(0.0),
        &scioxide_linalg::BorderedSystemOptions::new()
            .with_validation_tolerance(Tolerance::from_machine_epsilon())
            .with_compatibility_tolerance(Tolerance::from_machine_epsilon()),
    )
    .unwrap();
    let operator = plan.augmented();
    let large = S::Real::MAX * r(0.75);
    let mut input = [zero; 8];
    input[4..].copy_from_slice(&[
        S::from_real(large),
        S::from_real(large),
        S::from_real(large),
        S::from_real(-large),
    ]);
    let mut output = [zero; 8];
    let mut scratch = vec![zero; operator.apply_workspace_len().unwrap()];
    operator
        .apply_into(&input, &mut output, &mut scratch)
        .unwrap();
    let expected = if imaginary {
        S::from_components(r(0.0), large)
    } else {
        S::from_real(large)
    };
    assert_eq!(&output[..4], &[expected, expected, expected, -expected]);
    // The first row's first three positive contributions exceed MAX, but the
    // complete Hadamard contraction is exactly representable in every row.
}
#[test]
fn complete_mode_sums_rescue_intermediate_overflow() {
    extreme_cancellation::<f32>(false);
    extreme_cancellation::<f64>(false);
    extreme_cancellation::<Complex<f32>>(true);
    extreme_cancellation::<Complex<f64>>(true);
}

/// Zero operator whose declared apply scratch can change after preparation.
struct Zero {
    order: usize,
    scratch: core::sync::atomic::AtomicUsize,
}
impl Zero {
    fn apply(&self, input: &[f64], output: &mut [f64], workspace: &mut [f64]) {
        assert_eq!(input.len(), self.order);
        assert_eq!(
            workspace.len(),
            self.scratch.load(core::sync::atomic::Ordering::Relaxed)
        );
        output.fill(0.0);
    }
}
impl LinearOperator<f64> for Zero {
    fn rows(&self) -> usize {
        self.order
    }
    fn cols(&self) -> usize {
        self.order
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(self.scratch.load(core::sync::atomic::Ordering::Relaxed))
    }
    fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        workspace: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        self.apply(input, output, workspace);
        Ok(())
    }
}
impl scioxide_core::AdjointOperator<f64> for Zero {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(self.scratch.load(core::sync::atomic::Ordering::Relaxed))
    }
    fn apply_adjoint_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        workspace: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        self.apply(input, output, workspace);
        Ok(())
    }
}

#[test]
fn bordered_construction_and_diagnostics_forward_scopes_and_stops() {
    use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    use scioxide_core::{
        AdjointOperator, CancellationToken, EvaluationRole, ExecutionContext, ProblemToken,
        SciOxideError, SolveBudgets, SolveIdentity, SolveState,
    };
    use scioxide_linalg::{BorderedSystemOptions, VectorViewMut};
    // Deliberately not Debug: formatting the borrowed plan imposes no provider bound.
    struct Scoped<'a> {
        calls: AtomicUsize,
        stop: AtomicU8,
        token: &'a CancellationToken,
    }
    impl LinearOperator<f64> for Scoped<'_> {
        fn rows(&self) -> usize {
            2
        }
        fn cols(&self) -> usize {
            2
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, _: &[f64], _: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
            Err(SciOxideError::singular(None))
        }
        fn apply_into_scoped(
            &self,
            x: &[f64],
            y: &mut [f64],
            _: &mut [f64],
            context: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
        ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
            context
                .record_provider(EvaluationRole::Function)?
                .require()?;
            self.calls.fetch_add(1, Ordering::Relaxed);
            y[0] = 2.0 * x[0];
            y[1] = 0.0;
            match self.stop.load(Ordering::Relaxed) {
                1 => self.token.cancel(),
                2 => return Err(SciOxideError::singular(None)),
                _ => (),
            }
            Ok(scioxide_core::Invocation::Evaluated(()))
        }
    }
    impl AdjointOperator<f64> for Scoped<'_> {
        fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_adjoint_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            w: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.apply_into(x, y, w)
        }
        fn apply_adjoint_into_scoped(
            &self,
            x: &[f64],
            y: &mut [f64],
            w: &mut [f64],
            context: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
        ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
            self.apply_into_scoped(x, y, w, context)
        }
    }
    let mode = [0.0, 1.0];
    let basis = BlockVectorView::new(&mode, 2, 1).unwrap();
    for stop in [0, 1, 2] {
        let token = CancellationToken::new();
        let op = Scoped {
            calls: AtomicUsize::new(0),
            stop: AtomicU8::new(0),
            token: &token,
        };
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let id = SolveIdentity::new(
            ProblemToken::new(1101),
            scioxide_core::algorithm_id!("test", "bordered_scope", 1),
            0,
            0,
        );
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&token),
                id,
                &mut budgets,
            )
            .unwrap();
        let plan = BorderedSystem::build(
            &op,
            basis,
            basis,
            2.0,
            &BorderedSystemOptions::new(),
            &mut parent,
        )
        .unwrap();
        assert!(format!("{:?}", plan.clone()).contains("BorderedSystem"));
        assert_eq!(op.calls.load(Ordering::Relaxed), 2);
        let mut load = [91.0; 3];
        let mut workspace = LinalgWorkspace::new();
        plan.prepare_rhs_into(
            &[4.0, 0.0][..],
            &[3.0][..],
            CompatibilityPolicy::RequireCompatible,
            VectorViewMut::from_slice(&mut load),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
        op.stop.store(stop, Ordering::Relaxed);
        let result = plan.diagnose_scoped(
            &[4.0, 0.0][..],
            load.as_slice(),
            &[2.0, 3.0][..],
            &mut workspace,
            &mut parent,
        );
        if stop == 0 {
            let report = result.unwrap();
            assert_eq!(report.original_residual_norm(), 0.0);
            assert_eq!(report.gauge_defect(), 0.0);
            assert_eq!(parent.counts().own().iterations(), 0);
            assert_eq!(parent.counts().nested().iterations(), 4);
            assert_eq!(
                parent.counts().nested().provider(EvaluationRole::Function),
                3
            );
        } else {
            assert!(result.is_err());
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                if stop == 1 {
                    scioxide_core::AttemptOutcome::Stopped
                } else {
                    scioxide_core::AttemptOutcome::Failed
                }
            );
        }
        assert_eq!(load, [4.0, 0.0, 3.0]);
    }
    let token = CancellationToken::new();
    token.cancel();
    let op = Scoped {
        calls: AtomicUsize::new(0),
        stop: AtomicU8::new(0),
        token: &token,
    };
    assert!(
        BorderedSystem::build(
            &op,
            basis,
            basis,
            2.0,
            &BorderedSystemOptions::new(),
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(op.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn composed_byte_limits_bound_validation_and_call_scratch() {
    use scioxide_core::SciOxideErrorKind;
    let operator = Zero {
        order: 2,
        scratch: core::sync::atomic::AtomicUsize::new(3),
    };
    let modes = [1.0, 0.0, 0.0, 1.0];
    let basis = BlockVectorView::new(&modes, 2, 2).unwrap();
    let limited = |bytes| {
        scioxide_linalg::BorderedSystemOptions::new().with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(bytes),
        )
    };
    // Validation stages one vector plus provider scratch: 2 + 3 scalars.
    let validation = 5 * core::mem::size_of::<f64>();
    assert_eq!(
        BorderedSystem::new(&operator, basis, basis, 0.0, &limited(validation - 1))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let plan = BorderedSystem::new(&operator, basis, basis, 0.0, &limited(validation)).unwrap();
    for request in [
        BorderedSystemWorkspaceRequest::PrepareRhs,
        BorderedSystemWorkspaceRequest::Diagnose,
    ] {
        assert_eq!(
            plan.workspace_requirements(request).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    }
}

#[test]
fn diagnostics_preserve_the_composed_product_accuracy() {
    use scioxide_core::Accuracy;
    use scioxide_linalg::{BorderedSystemOptions, LinalgExecutionOptions, VectorView};
    let operator = Zero {
        order: 3,
        scratch: core::sync::atomic::AtomicUsize::new(0),
    };
    let mode = [1.0 / 3.0_f64.sqrt(); 3];
    let basis = BlockVectorView::new(&mode, 3, 1).unwrap();
    // The compatibility dot cancels catastrophically: only exact accumulation
    // retains the unit middle component.
    let rhs = [1e16, 1.0, -1e16];
    let tolerance = Tolerance::new(1e-12, 1e-12).unwrap();
    let defect = |accuracy| {
        let execution = LinalgExecutionOptions::new().with_accuracy(accuracy);
        let plan = BorderedSystem::new(
            &operator,
            basis,
            basis,
            0.0,
            &BorderedSystemOptions::new()
                .with_validation_tolerance(tolerance)
                .with_compatibility_tolerance(tolerance)
                .with_execution_options(execution),
        )
        .unwrap();
        let diagnostics = plan
            .diagnose(
                VectorView::from_slice(&rhs),
                VectorView::from_slice(&[0.0; 4]),
                VectorView::from_slice(&[0.0; 3]),
            )
            .unwrap();
        let expected = VectorView::from_slice(&mode)
            .dotc_left_scoped(
                VectorView::from_slice(&rhs),
                &execution,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap()
            .abs();
        assert_eq!(diagnostics.compatibility_defect(), expected, "{accuracy:?}");
        expected
    };
    let exact = defect(Accuracy::Exact);
    assert!((exact - mode[0]).abs() <= f64::EPSILON);
    assert_ne!(defect(Accuracy::Standard), exact);
}

/// Zero operator that returns NaN once poisoned.
struct Poisoned(core::sync::atomic::AtomicBool);
impl LinearOperator<f64> for Poisoned {
    fn rows(&self) -> usize {
        2
    }
    fn cols(&self) -> usize {
        2
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_into(
        &self,
        _: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        let poisoned = self.0.load(core::sync::atomic::Ordering::Relaxed);
        y.fill(if poisoned { f64::NAN } else { 0.0 });
        Ok(())
    }
}
impl scioxide_core::AdjointOperator<f64> for Poisoned {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        self.apply_into(x, y, w)
    }
}

#[test]
fn nonfinite_operator_images_have_callback_origin() {
    use scioxide_linalg::{BorderedSystemOptions, VectorView};
    let callback = |error: scioxide_core::SciOxideError| {
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Callback,
                    ..
                }
            ),
            "{error:?}"
        );
    };
    let operator = Poisoned(core::sync::atomic::AtomicBool::new(true));
    let mode = [1.0, 0.0];
    let basis = BlockVectorView::new(&mode, 2, 1).unwrap();
    let options = BorderedSystemOptions::new();
    callback(BorderedSystem::new(&operator, basis, basis, 0.0, &options).unwrap_err());
    operator
        .0
        .store(false, core::sync::atomic::Ordering::Relaxed);
    let plan = BorderedSystem::new(&operator, basis, basis, 0.0, &options).unwrap();
    operator
        .0
        .store(true, core::sync::atomic::Ordering::Relaxed);
    callback(
        plan.diagnose(
            VectorView::from_slice(&[0.0; 2]),
            VectorView::from_slice(&[0.0; 3]),
            VectorView::from_slice(&[0.0; 2]),
        )
        .unwrap_err(),
    );
    // The augmented operator checks the wrapped provider in both directions.
    let augmented = plan.augmented();
    let mut output = [0.0; 3];
    let mut scratch = vec![0.0; augmented.apply_workspace_len().unwrap()];
    callback(
        augmented
            .apply_into(&[1.0; 3], &mut output, &mut scratch)
            .unwrap_err(),
    );
    let mut scratch =
        vec![0.0; scioxide_core::AdjointOperator::apply_adjoint_workspace_len(&augmented).unwrap()];
    callback(
        scioxide_core::AdjointOperator::apply_adjoint_into(
            &augmented,
            &[1.0; 3],
            &mut output,
            &mut scratch,
        )
        .unwrap_err(),
    );
}
