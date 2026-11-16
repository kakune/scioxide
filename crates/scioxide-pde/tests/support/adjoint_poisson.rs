//! Independent quadratic manufactured equilibrium of a hand-built residual and
//! prepared shared CG solves, for the generic caller-derivative steady path.
//! `geometry` selects whether the domain-length parameter is differentiated.
#![allow(dead_code, clippy::too_many_arguments)]
use scioxide_core::*;
use scioxide_linalg::*;
use scioxide_ode::*;
use scioxide_sparse::CsrMatrix;
pub fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap_or(S::ZERO)
}
pub fn bad() -> SciOxideError {
    SciOxideError::external("invalid shape or failed solve")
}
pub fn phase<T: NumericScalar>() -> T {
    T::from_components(T::Real::ONE, r(0.3))
}
pub struct Poisson<T: NumericScalar> {
    pub p: [T::Real; 4],
    pub n: usize,
    pub geometry: bool,
}
impl<T: NumericScalar> Poisson<T> {
    pub fn new(n: usize, geometry: bool) -> Self {
        Self {
            p: [r(1.3), r(0.2), r(0.8), r(1.7)],
            n,
            geometry,
        }
    }
    pub fn h(&self) -> T::Real {
        self.p[3] / r((self.n + 1) as f64)
    }
    pub fn coefficient(&self) -> T::Real {
        self.p[0] / (self.h() * self.h())
    }
    pub fn solution(&self) -> Vec<T> {
        (0..self.n)
            .map(|i| {
                let q = r::<T::Real>((i + 1) as f64) / r((self.n + 1) as f64);
                phase::<T>()
                    * T::from_real(
                        self.p[1]
                            + self.p[2] / self.p[0]
                                * self.p[3]
                                * self.p[3]
                                * q
                                * (T::Real::ONE - q)
                                * T::Real::HALF,
                    )
            })
            .collect()
    }
    pub fn parameter_matrix(&self) -> Result<Matrix<T>, SciOxideError> {
        let u = self.solution();
        let mut m = Matrix::zeros(self.n, 4)?;
        for (i, &y) in u.iter().enumerate() {
            let neighbors = if i == 0 {
                phase::<T>() * T::from_real(self.p[1])
            } else {
                u[i - 1]
            } + if i + 1 == self.n {
                phase::<T>() * T::from_real(self.p[1])
            } else {
                u[i + 1]
            };
            let lap = (y + y - neighbors) * T::from_real(T::Real::ONE / (self.h() * self.h()));
            m.view_mut().set(i, 0, lap)?;
            m.view_mut().set(
                i,
                1,
                if i == 0 || i + 1 == self.n {
                    -phase::<T>()
                        * T::from_real(self.coefficient() * r(if self.n == 1 { 2. } else { 1. }))
                } else {
                    T::ZERO
                },
            )?;
            m.view_mut().set(i, 2, -phase::<T>())?;
            m.view_mut().set(
                i,
                3,
                if self.geometry {
                    -lap * T::from_real(r::<T::Real>(2.) * self.p[0] / self.p[3])
                } else {
                    T::ZERO
                },
            )?;
        }
        Ok(m)
    }
    pub fn matrix(&self) -> Result<Matrix<T>, SciOxideError> {
        let mut m = Matrix::zeros(self.n, self.n)?;
        for i in 0..self.n {
            m.view_mut()
                .set(i, i, T::from_real(r::<T::Real>(2.) * self.coefficient()))?;
            if i > 0 {
                m.view_mut()
                    .set(i, i - 1, -T::from_real(self.coefficient()))?;
            }
            if i + 1 < self.n {
                m.view_mut()
                    .set(i, i + 1, -T::from_real(self.coefficient()))?;
            }
        }
        Ok(m)
    }
    pub fn sparse(&self) -> Result<CsrMatrix<T>, SciOxideError> {
        let mut rows = vec![0];
        let mut cols = vec![];
        let mut vals = vec![];
        for i in 0..self.n {
            for j in i.saturating_sub(1)..=(i + 1).min(self.n - 1) {
                cols.push(j);
                vals.push(T::from_real(
                    self.coefficient() * r(if i == j { 2. } else { -1. }),
                ));
            }
            rows.push(cols.len());
        }
        CsrMatrix::from_raw(self.n, self.n, rows, cols, vals)
    }
    pub fn expected(&self) -> [T::Real; 4] {
        let mut out = [T::Real::ZERO; 4];
        for (i, y) in self.solution().into_iter().enumerate() {
            let q = r::<T::Real>((i + 1) as f64) / r((self.n + 1) as f64);
            let a = self.p[3] * self.p[3] * q * (T::Real::ONE - q) * T::Real::HALF;
            let dy = [
                -self.p[2] * a / (self.p[0] * self.p[0]),
                T::Real::ONE,
                a / self.p[0],
                if self.geometry {
                    r::<T::Real>(2.) * self.p[2] * a / (self.p[0] * self.p[3])
                } else {
                    T::Real::ZERO
                },
            ];
            for (j, v) in out.iter_mut().enumerate() {
                *v += self.h() * (y.conj() * phase::<T>() * T::from_real(dy[j])).real_component();
            }
            if self.geometry {
                out[3] += self.h() * y.norm_sqr() * T::Real::HALF / self.p[3];
            }
        }
        out
    }
}
impl<T: NumericScalar> LinearOperator<T> for Poisson<T> {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[T], out: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        if x.len() != self.n || out.len() != self.n {
            return Err(bad());
        }
        for (i, v) in out.iter_mut().enumerate() {
            *v = T::from_real(self.coefficient())
                * (x[i] + x[i]
                    - if i > 0 { x[i - 1] } else { T::ZERO }
                    - if i + 1 < self.n { x[i + 1] } else { T::ZERO });
        }
        Ok(())
    }
}
impl<T: NumericScalar> AdjointOperator<T> for Poisson<T> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[T],
        out: &mut [T],
        work: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, out, work)
    }
}
impl<T: NumericScalar> ResidualOperator<T> for Poisson<T> {
    fn input_dimension(&self) -> usize {
        self.n
    }
    fn output_dimension(&self) -> usize {
        self.n
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[T], out: &mut [T], work: &mut [T]) -> Result<(), SciOxideError> {
        self.apply_into(x, out, work)?;
        for (i, v) in out.iter_mut().enumerate() {
            *v = *v - phase::<T>() * T::from_real(self.p[2]);
            if i == 0 {
                *v = *v - phase::<T>() * T::from_real(self.coefficient() * self.p[1]);
            }
            if i + 1 == self.n {
                *v = *v - phase::<T>() * T::from_real(self.coefficient() * self.p[1]);
            }
        }
        Ok(())
    }
}
impl<T: NumericScalar> AdjointObjective<T> for Poisson<T> {
    fn workspace_len(&self) -> usize {
        self.n
    }
    fn evaluate(
        &self,
        _: AdjointObjectivePart,
        _: T::Real,
        y: &[T],
        _: &[T::Real],
        gy: &mut [T],
        gp: &mut [T::Real],
        work: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<T::Real>, SciOxideError> {
        assert_eq!(work.len(), self.n);
        let mut value = T::Real::ZERO;
        for (out, &v) in gy.iter_mut().zip(y) {
            *out = T::from_real(self.h()) * v;
            value += self.h() * v.norm_sqr() * T::Real::HALF;
        }
        gp.fill(T::Real::ZERO);
        if self.geometry {
            gp[3] = value / self.p[3];
        }
        Ok(AdjointObjectiveValue {
            value,
            time_derivative: T::Real::ZERO,
        })
    }
}
pub struct PreparedCg<T: NumericScalar> {
    solver: CgSolver<T>,
    work: IterativeWorkspace<T>,
    rhs: Vector<T>,
    x: Vector<T>,
}
impl<T: NumericScalar> PreparedCg<T> {
    pub fn new(operator: &dyn LinearOperator<T>) -> Result<Self, SciOxideError> {
        let tol = T::Real::EPSILON * r(8.);
        let solver = CgSolver::new(ResidualTolerance::new(tol, tol)?, Budget::try_from(5000)?);
        let work = solver.workspace(&IterativeWorkspaceRequest::operator(operator)?)?;
        Ok(Self {
            solver,
            work,
            rhs: Vector::zeros(operator.rows())?,
            x: Vector::zeros(operator.rows())?,
        })
    }
    pub fn solve(
        &mut self,
        a: &dyn LinearOperator<T>,
        b: &[T],
        out: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.rhs.as_mut_slice().copy_from_slice(b);
        self.x.as_mut_slice().fill(T::ZERO);
        // CG runs in a child epoch of the adjoint solve's context.
        let report = self.solver.solve_into(
            a,
            &self.rhs,
            Preconditioning::None,
            self.x.view_mut(),
            &mut self.work,
            &mut *context,
        )?;
        if !report.is_success() {
            return Err(bad());
        }
        out.copy_from_slice(self.x.as_slice());
        Ok(())
    }
}
pub fn context<R>(
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<R, SciOxideError>,
) -> Result<R, SciOxideError> {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("pde", "adjoint-example", 1),
        0,
        0,
    );
    let mut c = state.begin_solve(ExecutionContext::sequential(), id, &mut budgets)?;
    run(&mut c)
}
