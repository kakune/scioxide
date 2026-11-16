//! Example-owned manufactured reaction model; no model catalogue in the PDE API.
#![allow(dead_code, clippy::unwrap_used)]
use scioxide_core::AlgebraicScalar;
extern crate alloc;
use alloc::vec::Vec;
use core::cell::RefCell;
use scioxide_core::{
    AdjointOperator, ConvergenceDiagnostics, ConvergenceReason, EvaluationRole, LinearOperator,
    NumericScalar, RealScalar, SciOxideError, SolveContext, StructuralPatternView,
};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::OdeSystem;
use scioxide_sparse::CsrMatrix;

pub fn real<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}

/// Two fields with unequal diffusivities and a conservative cubic exchange.
/// `M u' = -D A u + M (R(u) + s(t))`,
/// `R = (-loss*u - g, -loss*v + g)`, `g = u²v/4`.
/// The independent forcing makes `exp(-decay*t)*target` the continuum solution.
pub struct ReactionDiffusion<'a, T: NumericScalar> {
    pub spatial: &'a dyn LinearOperator<T>,
    pub assembled: Option<&'a CsrMatrix<T>>,
    pub spatial_adjoint: Option<&'a dyn AdjointOperator<T>>,
    pub mass: &'a [T::Real],
    pub target: &'a [T],
    /// Independently derived `A*target/mass`, before each field's diffusivity.
    pub laplace: &'a [T],
    pub decay: T::Real,
    pub loss: T::Real,
    scratch: RefCell<Vec<T>>,
}
impl<'a, T: NumericScalar> ReactionDiffusion<'a, T> {
    pub fn new(
        spatial: &'a dyn LinearOperator<T>,
        mass: &'a [T::Real],
        target: &'a [T],
        laplace: &'a [T],
    ) -> Result<Self, SciOxideError> {
        if spatial.rows() != mass.len()
            || spatial.cols() != mass.len()
            || target.len() != 2 * mass.len()
            || laplace.len() != target.len()
        {
            return Err(SciOxideError::external("model shapes disagree"));
        }
        Ok(Self {
            spatial,
            assembled: None,
            spatial_adjoint: None,
            mass,
            target,
            laplace,
            decay: T::Real::ZERO,
            loss: T::Real::ONE,
            scratch: RefCell::new(vec![T::ZERO; spatial.apply_workspace_len()?]),
        })
    }
    pub fn diffusivity(field: usize) -> T {
        T::from_real(real(if field == 0 { 0.05 } else { 0.1 }))
    }
    pub fn reaction(&self, u: T, v: T) -> [T; 2] {
        let g = T::from_real(real(0.25)) * u * u * v;
        let loss = T::from_real(self.loss);
        [-loss * u - g, -loss * v + g]
    }
    pub fn derivative(&self, u: T, v: T) -> [[T; 2]; 2] {
        let a = T::from_real(real(0.5)) * u * v;
        let b = T::from_real(real(0.25)) * u * u;
        let loss = T::from_real(self.loss);
        [[-loss - a, -b], [a, -loss + b]]
    }
    pub fn exact_into(&self, time: T::Real, out: &mut [T]) {
        let factor = T::from_real((-self.decay * time).exp());
        out.iter_mut()
            .zip(self.target)
            .for_each(|(out, &x)| *out = factor * x);
    }
    fn spatial_apply(
        &self,
        input: &[T],
        out: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.mass.len();
        let mut scratch = self
            .scratch
            .try_borrow_mut()
            .map_err(|_| SciOxideError::external("reentrant spatial workspace"))?;
        for field in 0..2 {
            let range = field * n..(field + 1) * n;
            if let scioxide_core::Admission::Denied(report) =
                context.record_provider(EvaluationRole::LinearOperator)?
            {
                return Err(SciOxideError::convergence_failure(
                    ConvergenceReason::EvaluationLimit,
                    ConvergenceDiagnostics::new()
                        .with_counts(context.counts())
                        .with_budget(report),
                    "spatial product budget exhausted",
                ));
            }
            self.spatial
                .apply_into_scoped(
                    &input[range.clone()],
                    &mut out[range],
                    &mut scratch,
                    &mut scioxide_core::OperatorContext::new(context),
                )
                .and_then(scioxide_core::Invocation::require)?;
        }
        Ok(())
    }
    fn evaluate(
        &self,
        time: T::Real,
        x: &[T],
        out: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.spatial_apply(x, out, context)?;
        let n = self.mass.len();
        let e = T::from_real((-self.decay * time).exp());
        for i in 0..n {
            let r = self.reaction(x[i], x[n + i]);
            let q = [e * self.target[i], e * self.target[n + i]];
            let rq = self.reaction(q[0], q[1]);
            for field in 0..2 {
                let k = field * n + i;
                let d = Self::diffusivity(field);
                let source =
                    -T::from_real(self.decay) * q[field] + d * e * self.laplace[k] - rq[field];
                out[k] = -d * out[k] + T::from_real(self.mass[i]) * (r[field] + source);
            }
        }
        Ok(())
    }
    fn product(
        &self,
        x: &[T],
        v: &[T],
        out: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.spatial_apply(v, out, context)?;
        let n = self.mass.len();
        for i in 0..n {
            let j = self.derivative(x[i], x[n + i]);
            for (field, row) in j.iter().enumerate() {
                let k = field * n + i;
                out[k] = -Self::diffusivity(field) * out[k]
                    + T::from_real(self.mass[i]) * (row[0] * v[i] + row[1] * v[n + i]);
            }
        }
        Ok(())
    }
    pub fn adjoint(
        &self,
        _: T::Real,
        x: &[T],
        v: &[T],
        out: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        let a = self
            .spatial_adjoint
            .ok_or_else(|| SciOxideError::external("adjoint must be supplied"))?;
        let n = self.mass.len();
        for field in 0..2 {
            let range = field * n..(field + 1) * n;
            a.apply_adjoint_into(&v[range.clone()], &mut out[range], scratch)?;
        }
        for i in 0..n {
            let j = self.derivative(x[i], x[n + i]);
            for (field, (&a, &b)) in j[0].iter().zip(&j[1]).enumerate() {
                let k = field * n + i;
                out[k] = -Self::diffusivity(field) * out[k]
                    + T::from_real(self.mass[i]) * (a.conj() * v[i] + b.conj() * v[n + i]);
            }
        }
        Ok(())
    }
    fn entry(&self, x: &[T], row: usize, col: usize) -> Result<T, SciOxideError> {
        let n = self.mass.len();
        let (a, i) = (row / n, row % n);
        let (b, k) = (col / n, col % n);
        let matrix = self
            .assembled
            .ok_or_else(|| SciOxideError::external("assembled derivative was not prepared"))?;
        let mut value = if a == b {
            -Self::diffusivity(a) * matrix.get(i, k)?
        } else {
            T::ZERO
        };
        if i == k {
            value = value + T::from_real(self.mass[i]) * self.derivative(x[i], x[n + i])[a][b];
        }
        Ok(value)
    }
}
impl<T: NumericScalar> OdeSystem<T::Real, T> for ReactionDiffusion<'_, T> {
    fn dimension(&self) -> usize {
        self.target.len()
    }
    fn rhs(
        &self,
        time: T::Real,
        x: &[T],
        out: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(time, x, out, context)
    }
    fn jacobian_product(
        &self,
        _: T::Real,
        x: &[T],
        v: &[T],
        out: &mut [T],
        _: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.product(x, v, out, context).map(Some)
    }
    fn jacobian(
        &self,
        _: T::Real,
        x: &[T],
        out: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..self.dimension() {
            for col in 0..self.dimension() {
                out.set(row, col, self.entry(x, row, col)?)?;
            }
        }
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: T::Real,
        x: &[T],
        p: StructuralPatternView<'_>,
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..self.dimension() {
            let range = p.row_ptr()[row]..p.row_ptr()[row + 1];
            for (value, &column) in out[range.clone()].iter_mut().zip(&p.col_indices()[range]) {
                *value = self.entry(x, row, column)?;
            }
        }
        Ok(Some(()))
    }
    fn time_derivative(
        &self,
        time: T::Real,
        _: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let e = T::from_real((-self.decay * time).exp());
        let rate = T::from_real(self.decay);
        let n = self.mass.len();
        for i in 0..n {
            let q = [e * self.target[i], e * self.target[n + i]];
            let j = self.derivative(q[0], q[1]);
            for field in 0..2 {
                out[field * n + i] = T::from_real(self.mass[i])
                    * rate
                    * (rate * q[field]
                        - Self::diffusivity(field) * e * self.laplace[field * n + i]
                        + j[field][0] * q[0]
                        + j[field][1] * q[1]);
            }
        }
        Ok(Some(()))
    }
}
