//! Shared Krylov failure and iteration-limit fixtures.
use crate::Preconditioning;
use crate::{IterativeLinearSolver, IterativeSolveResult, Matrix, Vector};
use alloc::vec::Vec;
use scioxide_core::{Preconditioner, SciOxideError};
pub(super) struct FailingPreconditioner;

impl Preconditioner<f64> for FailingPreconditioner {
    fn dimension(&self) -> usize {
        2
    }

    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }

    fn precondition_into(
        &self,
        _input: &[f64],
        _output: &mut [f64],
        _workspace: &mut [f64],
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("injected_preconditioner"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "injected preconditioner failure",
        ))
    }
}

pub(super) fn max_iter_behavior(
    solver: &impl IterativeLinearSolver<f64>,
) -> IterativeSolveResult<f64> {
    let a = Matrix::from_vec(2, 2, vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let b = Vector::from_vec(vec![1.0, 2.0]);
    let result = solver
        .solve_with_options(
            &a.view().try_finite().unwrap(),
            b.view(),
            Preconditioning::None,
            None,
        )
        .unwrap();
    assert!(!result.is_success());
    result
}

pub(super) fn deterministic_complex_system(
    hermitian: bool,
) -> (
    crate::dense::Matrix<scioxide_core::Complex<f64>>,
    crate::dense::Vector<scioxide_core::Complex<f64>>,
) {
    use scioxide_core::Complex;

    const ORDER: usize = 4;
    fn sample(index: usize) -> f64 {
        let mixed = (index as u64 + 0x9e37_79b9_7f4a_7c15)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9)
            .rotate_left(23)
            .wrapping_mul(0x94d0_49bb_1331_11eb);
        (mixed >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64)) - 0.5
    }

    let mut values = vec![Complex::new(0.0, 0.0); ORDER * ORDER];
    if hermitian {
        (0..ORDER).for_each(|row| {
            values[row * ORDER + row] = Complex::new(6.0 + row as f64, 0.0);
            (row + 1..ORDER).for_each(|col| {
                let value = Complex::new(
                    0.2 * sample(row * ORDER + col),
                    0.2 * sample(ORDER * ORDER + row * ORDER + col),
                );
                values[row * ORDER + col] = value;
                values[col * ORDER + row] = value.conj();
            });
        });
    } else {
        values.iter_mut().enumerate().for_each(|(index, value)| {
            *value = Complex::new(0.2 * sample(index), 0.2 * sample(ORDER * ORDER + index));
        });
        (0..ORDER).for_each(|index| values[index * ORDER + index].re += 6.0 + index as f64);
    }
    let matrix = crate::dense::Matrix::from_vec(ORDER, ORDER, values).unwrap();
    let solution = crate::dense::Vector::from_vec(
        (0..ORDER)
            .map(|index| {
                Complex::new(
                    sample(2 * ORDER * ORDER + index),
                    sample(3 * ORDER * ORDER + index),
                )
            })
            .collect(),
    );
    let right_hand_side = matrix.view().mul_vec(&solution).unwrap();
    (matrix, right_hand_side)
}

/// Test-only verification trace. Preparation reserves every sample; recording
/// neither allocates nor changes production provider/iteration accounting.
#[derive(Debug)]
pub(super) struct CheckpointTrace {
    side: usize,
    rhs: Vec<f64>,
    pub(super) samples: Vec<(usize, f64)>,
}
std::thread_local! {
    static TRACE: core::cell::RefCell<Option<CheckpointTrace>> = const { core::cell::RefCell::new(None) };
}
pub(super) struct TraceGuard;
impl TraceGuard {
    pub(super) fn start(side: usize, rhs: Vec<f64>) -> Self {
        TRACE.with(|trace| {
            *trace.borrow_mut() = Some(CheckpointTrace {
                side,
                rhs,
                samples: Vec::with_capacity(20_000),
            })
        });
        Self
    }
    pub(super) fn finish(self) -> CheckpointTrace {
        TRACE.with(|trace| trace.borrow_mut().take().unwrap())
    }
}
impl Drop for TraceGuard {
    fn drop(&mut self) {
        TRACE.with(|trace| {
            trace.borrow_mut().take();
        });
    }
}
pub(super) fn checkpoint<S: scioxide_core::NumericScalar>(
    x: &[S],
    residual: S::Real,
    iteration: usize,
) {
    use scioxide_core::RealScalar;
    TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        let Some(trace) = trace.as_mut() else {
            return;
        };
        let side = trace.side;
        let n = side * side;
        assert_eq!(x.len(), trace.rhs.len());
        let mut maximum = 0_f64;
        for (column, rhs) in x.chunks_exact(n).zip(trace.rhs.chunks_exact(n)) {
            let get = |j: usize| column[j].real_component().into_f64();
            let norm = (0..n).fold(0_f64, |norm, i| {
                let mut ax = 4. * get(i);
                if i % side > 0 {
                    ax -= get(i - 1);
                }
                if i % side + 1 < side {
                    ax -= get(i + 1);
                }
                if i >= side {
                    ax -= get(i - side);
                }
                if i + side < n {
                    ax -= get(i + side);
                }
                norm.hypot(rhs[i] - ax)
            });
            maximum = maximum.max(norm);
        }
        let residual = residual.into_f64();
        assert!(
            (maximum - residual).abs() <= 1e-10 * maximum.max(1.),
            "{maximum:e} != {residual:e}"
        );
        assert!(trace.samples.len() < trace.samples.capacity());
        trace.samples.push((iteration, residual));
    });
}

#[test]
fn cg_and_bicgstab_provider_counts_include_only_real_work() {
    use crate::{
        BicgstabSolver, CgSolver, IterativeWorkspace, IterativeWorkspaceRequest,
        LinearSolveTermination,
    };
    use core::cell::Cell;
    use scioxide_core::{
        Budget, EvaluationRole, ExecutionContext, LinearOperator, ResidualTolerance,
    };

    struct Diagonal {
        n: usize,
        calls: Cell<usize>,
    }
    impl LinearOperator<f64> for Diagonal {
        fn rows(&self) -> usize {
            self.n
        }
        fn cols(&self) -> usize {
            self.n
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
            self.calls.set(self.calls.get() + 1);
            for (i, (y, x)) in y.iter_mut().zip(x).enumerate() {
                *y = (i + 1) as f64 * x;
            }
            Ok(())
        }
    }
    struct Identity {
        n: usize,
        calls: Cell<usize>,
    }
    impl Preconditioner<f64> for Identity {
        fn dimension(&self) -> usize {
            self.n
        }
        fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn precondition_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.calls.set(self.calls.get() + 1);
            y.copy_from_slice(x);
            Ok(())
        }
    }
    for execution in [
        ExecutionContext::sequential(),
        ExecutionContext::sequential_native(),
    ] {
        // The one-dimensional case ends at BiCGSTAB's first half-step.
        // The 65-step case crosses the periodic checkpoint at step 50.
        for (n, steps, tolerance) in [(1, 1, 1e-12), (180, 65, 0.)] {
            let tolerance = ResidualTolerance::new(0., tolerance).unwrap();
            let cg = CgSolver::new(tolerance, Budget::try_from(steps).unwrap());
            let bicgstab = BicgstabSolver::new(tolerance, Budget::try_from(steps).unwrap());
            let solvers: [(&dyn IterativeLinearSolver<f64>, usize, usize); 2] = [
                (&cg, if n == 1 { 3 } else { 68 }, steps),
                (
                    &bicgstab,
                    if n == 1 { 3 } else { 133 },
                    if n == 1 { 1 } else { 2 * steps },
                ),
            ];
            for (solver, operators, inverses) in solvers {
                for preconditioned in [false, true] {
                    let a = Diagonal {
                        n,
                        calls: Cell::new(0),
                    };
                    let inverse = Identity {
                        n,
                        calls: Cell::new(0),
                    };
                    let p = if preconditioned {
                        Preconditioning::Inverse(&inverse)
                    } else {
                        Preconditioning::None
                    };
                    let mut workspace = IterativeWorkspace::new();
                    workspace
                        .try_reserve(
                            &solver
                                .workspace_requirements(
                                    &IterativeWorkspaceRequest::operator(&a)
                                        .unwrap()
                                        .with_preconditioner(p)
                                        .unwrap(),
                                )
                                .unwrap(),
                        )
                        .unwrap();
                    let mut x = Vector::zeros(n).unwrap();
                    let b = vec![1.; n];
                    let report = solver
                        .solve_into(
                            &a,
                            b.as_slice().into(),
                            p,
                            x.view_mut(),
                            &mut workspace,
                            execution.into(),
                        )
                        .unwrap();
                    assert_eq!(report.iterations(), steps);
                    assert_eq!(
                        report.termination(),
                        if n == 1 {
                            LinearSolveTermination::Converged
                        } else {
                            LinearSolveTermination::IterationLimit
                        }
                    );
                    assert_eq!(a.calls.get(), operators);
                    assert_eq!(
                        inverse.calls.get(),
                        if preconditioned { inverses } else { 0 }
                    );
                    let counts = workspace.last_attempt().unwrap().counts().own();
                    assert_eq!(
                        counts.provider(EvaluationRole::LinearOperator),
                        a.calls.get()
                    );
                    assert_eq!(
                        counts.provider(EvaluationRole::Preconditioner),
                        inverse.calls.get()
                    );
                    let residual = x
                        .as_slice()
                        .iter()
                        .enumerate()
                        .fold(0_f64, |sum, (i, x)| sum.hypot(1. - (i + 1) as f64 * x));
                    assert!((report.true_residual().unwrap().raw_norm() - residual).abs() <= 1e-12);
                }
            }
        }
    }
}
