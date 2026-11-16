//! Operators preserve resolved processor dispatch and caller-owned transactions.
use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::backend::KernelDispatch;
use scioxide_core::{
    AdjointOperator, Budget, CancellationToken, Complex, DenseOperatorInput, DenseOperatorKernel,
    ExecutionContext, NumericScalar, OperatorContext, Preconditioner, ProcessorCapabilities,
    ProcessorFeature, ResidualTolerance, SciOxideError, SolveBudgets, SolveContext, SolveIdentity,
    SolveState, Triangle,
};
use scioxide_linalg::*;

fn scoped<T>(
    execution: ExecutionContext<'_>,
    call: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "operator_dispatch", 1),
        0,
        0,
    );
    call(&mut state.begin_solve(execution, identity, &mut budgets)?)
}

struct CountedProcessor<'a> {
    base: &'a dyn ProcessorCapabilities,
    queries: AtomicUsize,
}
impl ProcessorCapabilities for CountedProcessor<'_> {
    fn supports(&self, feature: ProcessorFeature) -> bool {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.base.supports(feature)
    }
}
struct Resolved {
    dispatch: KernelDispatch,
    calls: std::cell::Cell<usize>,
}
impl<S: NumericScalar> DenseOperatorKernel<S> for Resolved {
    fn dispatch(&self) -> KernelDispatch {
        self.calls.set(self.calls.get() + 1);
        self.dispatch
    }
    fn apply_dense_into(
        &self,
        _: DenseOperatorInput<'_, S>,
        _: &[S],
        _: S::Real,
        _: &mut [S],
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("kernel"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "factor and compact operators must use their own kernels",
        ))
    }
}
fn bits<S: NumericScalar<Real = f64>>(actual: &[S], expected: &[S]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.real_component().to_bits(), b.real_component().to_bits());
        assert_eq!(
            a.imaginary_component().to_bits(),
            b.imaginary_component().to_bits()
        );
    }
}
fn matrix(n: usize) -> Result<Matrix<f64>, SciOxideError> {
    Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                if r == c {
                    2.0 + r as f64 / n as f64
                } else {
                    (((r.min(c) * 17 + r.max(c) * 31) % 103) as f64 / 103.0 - 0.5) / n as f64
                }
            })
            .collect(),
    )
}

fn check_inverse(
    op: &dyn AdjointOperator<f64>,
    p: &dyn Preconditioner<f64>,
    rhs: &[f64],
    expected: &[f64],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let processor = CountedProcessor {
        base: execution.processor(),
        queries: AtomicUsize::new(0),
    };
    let execution = execution.with_processor(&processor);
    let kernels = Resolved {
        dispatch: KernelDispatch::detect(&processor),
        calls: std::cell::Cell::new(0),
    };
    let queries = processor.queries.load(Ordering::Relaxed);
    let n = rhs.len();
    let mut out = vec![0.; n];
    let mut scratch = vec![
        0.;
        op.apply_workspace_len()?
            .max(op.apply_adjoint_workspace_len()?)
    ];
    scoped(execution, |solve| {
        let c = &mut OperatorContext::with_kernels(solve, &kernels);
        for finite in [false, true] {
            (if finite {
                op.apply_finite_into_scoped(rhs, 1., &mut out, &mut scratch, c)
                    .and_then(scioxide_core::Invocation::require)
            } else {
                op.apply_into_scoped(rhs, &mut out, &mut scratch, c)
                    .and_then(scioxide_core::Invocation::require)
            })?;
            bits(&out, expected);
        }
        p.precondition_into_scoped(rhs, &mut out, &mut scratch, c)?;
        bits(&out, expected);
        assert_eq!(processor.queries.load(Ordering::Relaxed), queries);
        assert!(kernels.calls.get() > 0);
        // A bare scoped caller must also retain its processor policy.
        op.apply_into_scoped(rhs, &mut out, &mut scratch, &mut OperatorContext::new(c))
            .and_then(scioxide_core::Invocation::require)?;
        bits(&out, expected);
        Ok(())
    })?;
    // The bare context above is allowed one resolution. Injected repeated calls
    // must not query processor policy; check those independently below.
    let resolved_queries = processor.queries.load(Ordering::Relaxed);
    scoped(execution, |solve| {
        let c = &mut OperatorContext::with_kernels(solve, &kernels);
        op.apply_into_scoped(rhs, &mut out, &mut scratch, c)
            .and_then(scioxide_core::Invocation::require)?;
        p.precondition_into_scoped(rhs, &mut out, &mut scratch, c)?;
        op.apply_adjoint_into_scoped(rhs, &mut out, &mut scratch, c)
            .and_then(scioxide_core::Invocation::require)?;
        let adjoint = out.clone();
        op.apply_adjoint_finite_into_scoped(rhs, 1., &mut out, &mut scratch, &mut c.reborrow())
            .and_then(scioxide_core::Invocation::require)?;
        bits(&out, &adjoint);
        Ok(())
    })?;
    assert_eq!(processor.queries.load(Ordering::Relaxed), resolved_queries);
    Ok(())
}

#[test]
fn inverse_forms_match_native_and_portable_solves_at_256() -> Result<(), SciOxideError> {
    for n in [0, 1, 256] {
        let a = matrix(n)?;
        let options = LinalgExecutionOptions::new();
        let lu = LuDecomposition::factor(&a)?;
        let ch = CholeskyDecomposition::factor(&a, HermitianInput::Exact, CholeskyPivoting::None)?;
        let ld = LdlDecomposition::factor(&a, HermitianInput::Exact)?;
        let rhs: Vec<_> = (0..n).map(|i| (i % 37) as f64 / 37.0).collect();
        let inverse = [
            &lu.inverse_operator() as &dyn AdjointOperator<f64>,
            &ch.inverse_operator(),
            &ld.inverse_operator(),
        ];
        let preconditioner = [
            &lu.inverse_operator() as &dyn Preconditioner<f64>,
            &ch.inverse_operator(),
            &ld.inverse_operator(),
        ];
        for execution in [
            ExecutionContext::sequential_native(),
            ExecutionContext::sequential(),
        ] {
            let mut expected = vec![vec![0.; n]; 3];
            let solve = DirectSolveOptions::new(RankPolicy::Exact);
            let report = lu.solve_into(
                rhs.as_slice(),
                &solve,
                VectorViewMut::from_slice(&mut expected[0]),
                &mut LinalgWorkspace::new(),
                execution,
            )?;
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            let report = ch.solve_into(
                rhs.as_slice(),
                &solve,
                VectorViewMut::from_slice(&mut expected[1]),
                &mut LinalgWorkspace::new(),
                execution,
            )?;
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            let report = ld.solve_into(
                rhs.as_slice(),
                &solve,
                VectorViewMut::from_slice(&mut expected[2]),
                &mut LinalgWorkspace::new(),
                execution,
            )?;
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            for ((op, p), reference) in inverse.iter().zip(preconditioner).zip(&expected) {
                check_inverse(*op, p, &rhs, reference, execution)?;
            }
        }
        let native = ExecutionContext::sequential_native();
        for (op, p) in inverse.iter().zip(preconditioner) {
            let mut out = vec![0.; n];
            let mut expected = vec![0.; n];
            let mut w = vec![0.; op.apply_workspace_len()?];
            op.apply_into(&rhs, &mut expected, &mut w)?;
            p.precondition_into(&rhs, &mut out, &mut w)?;
            bits(&out, &expected);
            scoped(native, |solve| {
                op.apply_into_scoped(&rhs, &mut out, &mut w, &mut OperatorContext::new(solve))
                    .and_then(scioxide_core::Invocation::require)
            })?;
            bits(&out, &expected);
            op.apply_adjoint_into(&rhs, &mut expected, &mut w)?;
            scoped(native, |solve| {
                op.apply_adjoint_into_scoped(
                    &rhs,
                    &mut out,
                    &mut w,
                    &mut OperatorContext::new(solve),
                )
                .and_then(scioxide_core::Invocation::require)
            })?;
            bits(&out, &expected);
            let mut product = vec![0.; n];
            a.view().adjoint().mul_vec_into(
                out.as_slice(),
                &options,
                VectorViewMut::from_slice(&mut product),
                &mut LinalgWorkspace::new(),
                native,
            )?;
            for (actual, expected) in product.iter().zip(&rhs) {
                assert!((actual - expected).abs() <= 256. * f64::EPSILON);
            }
        }
    }
    Ok(())
}

fn check_product<S: NumericScalar<Real = f64>>(
    op: &dyn AdjointOperator<S>,
    direct: impl Fn(bool, &[S], &mut [S], ExecutionContext<'_>) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    for execution in [
        ExecutionContext::sequential_native(),
        ExecutionContext::sequential(),
    ] {
        let processor = CountedProcessor {
            base: execution.processor(),
            queries: AtomicUsize::new(0),
        };
        let kernels = Resolved {
            dispatch: KernelDispatch::detect(&processor),
            calls: std::cell::Cell::new(0),
        };
        let queries = processor.queries.load(Ordering::Relaxed);
        for adjoint in [false, true] {
            let (rows, cols) = if adjoint {
                (op.cols(), op.rows())
            } else {
                (op.rows(), op.cols())
            };
            let input: Vec<_> = (0..cols)
                .map(|i| S::from_components((i % 37) as f64 / 37., (i % 11) as f64 / 11.))
                .collect();
            let mut expected = vec![S::ZERO; rows];
            direct(adjoint, &input, &mut expected, execution)?;
            let mut out = vec![S::ZERO; rows];
            let mut w = vec![S::ZERO; rows];
            scoped(execution.with_processor(&processor), |solve| {
                let c = &mut OperatorContext::with_kernels(solve, &kernels);
                for finite in [false, true] {
                    match (adjoint, finite) {
                        (false, false) => op
                            .apply_into_scoped(&input, &mut out, &mut w, c)
                            .and_then(scioxide_core::Invocation::require),
                        (false, true) => op
                            .apply_finite_into_scoped(&input, 1., &mut out, &mut w, c)
                            .and_then(scioxide_core::Invocation::require),
                        (true, false) => op
                            .apply_adjoint_into_scoped(&input, &mut out, &mut w, c)
                            .and_then(scioxide_core::Invocation::require),
                        (true, true) => op
                            .apply_adjoint_finite_into_scoped(&input, 1., &mut out, &mut w, c)
                            .and_then(scioxide_core::Invocation::require),
                    }?;
                    bits(&out, &expected);
                }
                Ok(())
            })?;
        }
        assert_eq!(processor.queries.load(Ordering::Relaxed), queries);
        assert!(kernels.calls.get() > 0);
    }
    Ok(())
}

#[test]
fn compact_complex_views_match_products_at_256() -> Result<(), SciOxideError> {
    type C = Complex<f64>;
    let n = 256;
    let options = LinalgExecutionOptions::new();
    let operation = |adjoint| {
        if adjoint {
            StructuredOperation::Adjoint
        } else {
            StructuredOperation::Normal
        }
    };
    let band = BandedMatrix::new(
        n,
        n + 1,
        1,
        2,
        5,
        (0..(n + 1) * 5)
            .map(|i| {
                let (j, k) = (i / 5, i % 5);
                if k < 4 && j + k >= 2 && j + k - 2 < n {
                    C::new((i % 17) as f64 / 17., (i % 7) as f64 / 7.)
                } else {
                    C::new(f64::NAN, f64::NAN)
                }
            })
            .collect(),
    )?;
    check_product(&band.view(), |adj, x, y, c| {
        band.view().mul_vec_in_place(
            operation(adj),
            C::new(1., 0.),
            x,
            C::new(0., 0.),
            &options,
            VectorViewMut::from_slice(y),
            &mut LinalgWorkspace::new(),
            c,
        )
    })?;
    for triangle in [Triangle::Lower, Triangle::Upper] {
        let values = (0..n * 3)
            .map(|i| {
                let (j, k) = (i / 3, i % 3);
                let diagonal = match triangle {
                    Triangle::Lower => k == 0,
                    Triangle::Upper => k == 2,
                };
                let valid = match triangle {
                    Triangle::Lower => j + k < n,
                    Triangle::Upper => j + k >= 2,
                };
                if !valid {
                    C::new(f64::NAN, f64::NAN)
                } else {
                    C::new(
                        (i % 17) as f64 / 17.,
                        if diagonal { 0. } else { (i % 7) as f64 / 7. },
                    )
                }
            })
            .collect();
        let band = HermitianBandMatrix::new(n, 2, triangle, values)?;
        check_product(&band.view(), |_, x, y, c| {
            band.view().mul_vec_in_place(
                C::new(1., 0.),
                x,
                C::new(0., 0.),
                &options,
                VectorViewMut::from_slice(y),
                &mut LinalgWorkspace::new(),
                c,
            )
        })?;
        for kind in [PackedKind::Triangular, PackedKind::Hermitian] {
            let mut packed =
                PackedMatrix::new(n, triangle, kind, vec![C::new(0., 0.); n * (n + 1) / 2])?;
            for j in 0..n {
                let rows = match triangle {
                    Triangle::Lower => j..n,
                    Triangle::Upper => 0..j + 1,
                };
                for i in rows {
                    packed.view_mut().set(
                        i,
                        j,
                        C::new(
                            ((i * 13 + j * 19) % 101) as f64 / 101.,
                            if i == j { 0. } else { 0.125 },
                        ),
                    )?;
                }
            }
            check_product(&packed.view(), |adj, x, y, c| {
                packed.view().mul_vec_in_place(
                    operation(adj),
                    C::new(1., 0.),
                    x,
                    C::new(0., 0.),
                    &options,
                    VectorViewMut::from_slice(y),
                    &mut LinalgWorkspace::new(),
                    c,
                )
            })?;
        }
    }
    Ok(())
}

#[test]
fn cg_and_gmres_preconditioners_match_owned_native_results() -> Result<(), SciOxideError> {
    let a = matrix(256)?;
    let ch = CholeskyDecomposition::factor(&a, HermitianInput::Exact, CholeskyPivoting::None)?;
    let lu = LuDecomposition::factor(&a)?;
    let tolerance = ResidualTolerance::new(1e-12, 0.)?;
    let cg = CgSolver::new(tolerance, Budget::try_from(10)?);
    let gmres = GmresSolver::new(tolerance, Budget::try_from(10)?);
    let operator = a.view().try_finite()?;
    let rhs = vec![1.; 256];
    let solvers: [&dyn IterativeLinearSolver<f64>; 2] = [&cg, &gmres];
    let preconditioners: [&dyn Preconditioner<f64>; 2] =
        [&ch.inverse_operator(), &lu.inverse_operator()];
    for (solver, inverse) in solvers.into_iter().zip(preconditioners) {
        struct Counted<'a> {
            inverse: &'a dyn Preconditioner<f64>,
            calls: std::cell::Cell<usize>,
        }
        impl Preconditioner<f64> for Counted<'_> {
            fn dimension(&self) -> usize {
                self.inverse.dimension()
            }
            fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
                self.inverse.precondition_workspace_len()
            }
            fn precondition_into(
                &self,
                _: &[f64],
                _: &mut [f64],
                _: &mut [f64],
            ) -> Result<(), SciOxideError> {
                Err(SciOxideError::singular(None))
            }
            fn precondition_into_scoped(
                &self,
                x: &[f64],
                y: &mut [f64],
                w: &mut [f64],
                c: &mut OperatorContext<'_, '_, '_, f64>,
            ) -> Result<(), SciOxideError> {
                assert_eq!(
                    c.kernels().map(DenseOperatorKernel::dispatch),
                    Some(KernelDispatch::detect(
                        ExecutionContext::sequential_native().processor()
                    ))
                );
                self.calls.set(self.calls.get() + 1);
                self.inverse.precondition_into_scoped(x, y, w, c)
            }
        }
        let counted = Counted {
            inverse,
            calls: std::cell::Cell::new(0),
        };
        let p = &counted;
        let owned = solver.solve_with_options(
            &operator,
            VectorView::from_slice(&rhs),
            Preconditioning::Inverse(p),
            None,
        )?;
        let owned_calls = counted.calls.replace(0);
        let request = IterativeWorkspaceRequest::operator(&operator)?
            .with_preconditioner(Preconditioning::Inverse(p))?;
        let mut workspace = solver.workspace(&request)?;
        let mut out = vec![0.; 256];
        let report = solver.solve_into(
            &operator,
            VectorView::from_slice(&rhs),
            Preconditioning::Inverse(p),
            VectorViewMut::from_slice(&mut out),
            &mut workspace,
            ExecutionContext::sequential_native().into(),
        )?;
        assert!(report.is_success());
        assert!(owned_calls > 0);
        assert_eq!(counted.calls.get(), owned_calls);
        assert_eq!(
            workspace
                .last_attempt()
                .ok_or_else(|| SciOxideError::singular(None))?
                .counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Preconditioner),
            owned_calls
        );
        bits(&out, owned.solution().as_slice());
        assert_eq!(report.iterations(), owned.report().iterations());
        assert_eq!(report.true_residual(), owned.report().true_residual());
    }
    Ok(())
}

#[test]
fn inverse_scoped_errors_preserve_output() -> Result<(), SciOxideError> {
    let a = matrix(2)?;
    let lu = LuDecomposition::factor(&a)?;
    let ch = CholeskyDecomposition::factor(&a, HermitianInput::Exact, CholeskyPivoting::None)?;
    let ld = LdlDecomposition::factor(&a, HermitianInput::Exact)?;
    let ops: [&dyn AdjointOperator<f64>; 3] = [
        &lu.inverse_operator(),
        &ch.inverse_operator(),
        &ld.inverse_operator(),
    ];
    for op in ops {
        let mut out = [13., 17.];
        let mut scratch = vec![0.; op.apply_workspace_len()?];
        scoped(ExecutionContext::sequential_native(), |solve| {
            let c = &mut OperatorContext::new(solve);
            assert!(
                op.apply_into_scoped(&[1.], &mut out, &mut scratch, c)
                    .and_then(scioxide_core::Invocation::require)
                    .is_err()
            );
            assert!(
                op.apply_into_scoped(&[1., 2.], &mut out, &mut [], c)
                    .and_then(scioxide_core::Invocation::require)
                    .is_err()
            );
            assert!(
                op.apply_into_scoped(&[f64::NAN, 2.], &mut out, &mut scratch, c)
                    .and_then(scioxide_core::Invocation::require)
                    .is_err()
            );
            for bound in [f64::NAN, f64::INFINITY, -1.] {
                assert!(
                    op.apply_finite_into_scoped(&[1., 2.], bound, &mut out, &mut scratch, c)
                        .and_then(scioxide_core::Invocation::require)
                        .is_err()
                );
                assert!(
                    op.apply_adjoint_finite_into_scoped(
                        &[1., 2.],
                        bound,
                        &mut out,
                        &mut scratch,
                        c
                    )
                    .and_then(scioxide_core::Invocation::require)
                    .is_err()
                );
            }
            Ok(())
        })?;
        let stop = CancellationToken::new();
        stop.cancel();
        scoped(
            ExecutionContext::sequential_native().with_cancellation(&stop),
            |solve| {
                let c = &mut OperatorContext::new(solve);
                assert!(
                    op.apply_into_scoped(&[1., 2.], &mut out, &mut scratch, c)
                        .and_then(scioxide_core::Invocation::require)
                        .is_err()
                );
                assert!(
                    op.apply_adjoint_into_scoped(&[1., 2.], &mut out, &mut scratch, c)
                        .and_then(scioxide_core::Invocation::require)
                        .is_err()
                );
                Ok(())
            },
        )?;
        assert_eq!(out, [13., 17.]);
    }
    let singular = Matrix::from_vec(2, 2, vec![1., 0., 0., 0.])?;
    let ch = CholeskyDecomposition::factor(
        &singular,
        HermitianInput::Exact,
        CholeskyPivoting::Diagonal(RankPolicy::Exact),
    )?;
    let ld = LdlDecomposition::factor(&singular, HermitianInput::Exact)?;
    for op in [
        &ch.inverse_operator() as &dyn AdjointOperator<f64>,
        &ld.inverse_operator(),
    ] {
        let mut out = [13., 17.];
        let mut scratch = vec![0.; op.apply_workspace_len()?];
        scoped(ExecutionContext::sequential_native(), |solve| {
            let c = &mut OperatorContext::new(solve);
            assert!(
                op.apply_into_scoped(&[1., 2.], &mut out, &mut scratch, c)
                    .and_then(scioxide_core::Invocation::require)
                    .is_err()
            );
            assert!(
                op.apply_adjoint_into_scoped(&[1., 2.], &mut out, &mut scratch, c)
                    .and_then(scioxide_core::Invocation::require)
                    .is_err()
            );
            Ok(())
        })?;
        assert_eq!(out, [13., 17.]);
    }
    Ok(())
}

struct ScopedOnly<'a, S: NumericScalar> {
    operator: &'a (dyn scioxide_core::LinearOperator<S> + Sync),
    dispatch: KernelDispatch,
    calls: AtomicUsize,
}
impl<S: NumericScalar> scioxide_core::LinearOperator<S> for ScopedOnly<'_, S> {
    fn rows(&self) -> usize {
        self.operator.rows()
    }
    fn cols(&self) -> usize {
        self.operator.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.operator.apply_workspace_len()
    }
    fn apply_into(&self, _: &[S], _: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("context"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "scoped application required",
        ))
    }
    fn apply_into_scoped(
        &self,
        x: &[S],
        y: &mut [S],
        w: &mut [S],
        c: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        assert_eq!(
            c.kernels().map(DenseOperatorKernel::dispatch),
            Some(self.dispatch)
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.operator.apply_into_scoped(x, y, w, c)
    }
}
impl<S: NumericScalar> Preconditioner<S> for ScopedOnly<'_, S> {
    fn dimension(&self) -> usize {
        self.operator.rows()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.operator.apply_workspace_len()
    }
    fn precondition_into(&self, x: &[S], y: &mut [S], w: &mut [S]) -> Result<(), SciOxideError> {
        scioxide_core::LinearOperator::apply_into(self, x, y, w)
    }
    fn precondition_into_scoped(
        &self,
        x: &[S],
        y: &mut [S],
        w: &mut [S],
        c: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::LinearOperator::apply_into_scoped(self, x, y, w, c)
            .and_then(scioxide_core::Invocation::require)
    }
}

#[test]
fn lobpcg_and_refinement_forward_resolved_contexts() -> Result<(), SciOxideError> {
    for execution in [
        ExecutionContext::sequential(),
        ExecutionContext::sequential_native(),
    ] {
        let dispatch = KernelDispatch::detect(execution.processor());
        let a = Matrix::from_vec(
            12,
            12,
            (0..144)
                .map(|i| {
                    if i / 12 == i % 12 {
                        1. + (i / 12) as f64
                    } else {
                        0.
                    }
                })
                .collect(),
        )?;
        let lu = LuDecomposition::factor(&a)?;
        let inverse = lu.inverse_operator();
        let preconditioner = ScopedOnly {
            operator: &inverse,
            dispatch,
            calls: AtomicUsize::new(0),
        };
        let operator = a.view().try_finite()?;
        let plan = LobpcgSolver::new(1);
        let problem =
            scioxide_linalg::LobpcgOperator::new(&operator).with_preconditioner(&preconditioner);
        let mut workspace = plan.workspace(problem, 0)?;
        let mut output = LobpcgBuffer::new(12, 1)?;
        let report = plan.solve_into(
            problem,
            scioxide_core::BlockVectorView::new(&[1.; 12], 12, 1)?,
            scioxide_core::BlockVectorView::new(&[], 12, 0)?,
            &mut output,
            &mut workspace,
            execution,
        )?;
        assert_eq!(report.termination(), SpectralTermination::Converged);
        assert!(preconditioner.calls.load(Ordering::Relaxed) > 0);

        let low_operator =
            scioxide_core::ClosureOperator::new(12, 12, |_: &[f32], y: &mut [f32]| {
                y.fill(0.);
                Ok(())
            });
        let low = ScopedOnly {
            operator: &low_operator,
            dispatch,
            calls: AtomicUsize::new(0),
        };
        let system = ScopedOnly {
            operator: &operator,
            dispatch,
            calls: AtomicUsize::new(0),
        };
        let problem = RefinementProblem::new(
            RefinementSystem::Operator {
                operator: &system,
                norm_bound: 12.,
            },
            &low,
            None,
            Some(&preconditioner),
        )?;
        let plan = IterativeRefinement::new(
            RefinementOptions::new(ResidualTolerance::new(1e-12, 0.)?)
                .with_criterion(RefinementCriterion::Normwise)
                .with_fallback(RefinementFallback::OnFailure),
        )?;
        let mut workspace =
            plan.workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))?;
        let mut solution = [0.; 12];
        preconditioner.calls.store(0, Ordering::Relaxed);
        let reports = plan.refine_into(
            &problem,
            VectorView::from_slice(&[1.; 12]),
            VectorViewMut::from_slice(&mut solution),
            &mut workspace,
            execution,
        )?;
        assert!(reports[0].is_success());
        assert!(low.calls.load(Ordering::Relaxed) > 0);
        assert!(preconditioner.calls.load(Ordering::Relaxed) > 0);
        assert!(system.calls.load(Ordering::Relaxed) > 0);
    }
    Ok(())
}

#[test]
fn complex_factor_adjoints_and_native_dense_conveniences() -> Result<(), SciOxideError> {
    use scioxide_core::LinearOperator;
    let c = Complex::new;
    let a = Matrix::from_vec(
        3,
        3,
        vec![
            c(4., 0.),
            c(1., 0.5),
            c(0.25, -0.5),
            c(1., -0.5),
            c(3., 0.),
            c(0.5, 0.25),
            c(0.25, 0.5),
            c(0.5, -0.25),
            c(2., 0.),
        ],
    )?;
    let options = LinalgExecutionOptions::new();
    let lu = LuDecomposition::factor(&a)?;
    let ch = CholeskyDecomposition::factor(&a, HermitianInput::Exact, CholeskyPivoting::None)?;
    let ld = LdlDecomposition::factor(&a, HermitianInput::Exact)?;
    let rhs = [c(1., 0.5), c(-2., 1.), c(0.25, -0.75)];
    for op in [
        &lu.inverse_operator() as &dyn AdjointOperator<Complex<f64>>,
        &ch.inverse_operator(),
        &ld.inverse_operator(),
    ] {
        let mut out = [c(0., 0.); 3];
        let mut expected = out;
        let mut w = vec![c(0., 0.); op.apply_adjoint_workspace_len()?];
        op.apply_adjoint_into(&rhs, &mut expected, &mut w)?;
        scoped(ExecutionContext::sequential_native(), |solve| {
            op.apply_adjoint_finite_into_scoped(
                &rhs,
                2.,
                &mut out,
                &mut w,
                &mut OperatorContext::new(solve),
            )
            .and_then(scioxide_core::Invocation::require)
        })?;
        bits(&out, &expected);
        a.view().adjoint().mul_vec_into(
            out.as_slice(),
            &options,
            VectorViewMut::from_slice(&mut expected),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        for (actual, expected) in expected.iter().zip(rhs) {
            assert!((*actual - expected).norm() <= 32. * f64::EPSILON);
        }
    }
    let a = matrix(256)?;
    let rhs = vec![1.; 256];
    let mut out = vec![0.; 256];
    let mut expected = out.clone();
    a.view().mul_vec_into(
        rhs.as_slice(),
        &options,
        VectorViewMut::from_slice(&mut expected),
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    a.view()
        .try_finite()?
        .apply_into(&rhs, &mut out, &mut [0.; 256])?;
    bits(&out, &expected);
    Ok(())
}
