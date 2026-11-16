//! Conjugate-transposed mixed residual equations, using existing stage backends.
use super::*;
use crate::dae::linear::EffectiveJacobian;
use crate::termination::Fault;
use crate::{DaePartition, OdeStageBackend};

use core::cell::RefCell;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{LinearOperator, Preconditioner, StructuralPatternView};
use scioxide_linalg::Matrix;

pub(super) fn prepare_backend(
    backend: &OdeStageBackend,
    n: usize,
    provider_scratch: usize,
) -> Result<(OdeStageBackend, Vec<usize>), SciOxideError> {
    Ok(match backend {
        OdeStageBackend::Dense => (OdeStageBackend::Dense, Vec::new()),
        OdeStageBackend::Krylov {
            iterations,
            restart,
            preconditioner_scratch,
            ..
        } => (
            OdeStageBackend::Krylov {
                iterations: *iterations,
                restart: *restart,
                operator_scratch: checked_add(
                    checked_mul(n, 2, scioxide_core::quantity!("direction_scratch"))?,
                    provider_scratch,
                    scioxide_core::quantity!("operator_scratch"),
                )?,
                preconditioner_scratch: *preconditioner_scratch,
            },
            Vec::new(),
        ),
        OdeStageBackend::Sparse { pattern, limits } => {
            if pattern.view().rows() != n || pattern.view().cols() != n {
                return Err(invalid(
                    scioxide_core::argument!("backend"),
                    "sparse adjoint pattern dimension",
                ));
            }
            let (pattern, permutation) =
                scioxide_sparse::transpose_pattern(pattern.view())?.into_parts();
            (
                OdeStageBackend::Sparse {
                    pattern,
                    limits: *limits,
                },
                permutation,
            )
        }
    })
}

/// Conservative payload envelope for forward and transpose storage together.
/// Dense is quadratic, sparse respects the caller factor cap, and Krylov is O(n*r+r*r).
pub(super) fn storage_bound<T: NumericScalar>(
    backend: &OdeStageBackend,
    n: usize,
    p: usize,
    scratch: usize,
) -> Result<usize, SciOxideError> {
    let scalar = core::mem::size_of::<T>();
    let vectors = checked_add(
        checked_mul(
            checked_add(n, p, scioxide_core::quantity!("workspace_bytes"))?,
            256,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        scratch,
        scioxide_core::quantity!("workspace_bytes"),
    )?;
    let linear = match backend {
        OdeStageBackend::Dense => checked_mul(
            checked_mul(n, n, scioxide_core::quantity!("workspace_bytes"))?,
            64 * scalar,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        OdeStageBackend::Sparse { pattern, limits } => checked_add(
            checked_mul(
                scioxide_sparse::SparseLuSymbolic::analyze(
                    pattern.view(),
                    scioxide_sparse::SparseOrdering::FillReducing,
                    *limits,
                )?
                .workspace_requirements::<T>(())?
                .total_bytes(),
                4,
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            checked_mul(
                pattern.view().col_indices().len(),
                256 * scalar,
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        OdeStageBackend::Krylov {
            restart,
            operator_scratch,
            preconditioner_scratch,
            ..
        } => {
            let r = checked_add(
                restart.get(),
                1,
                scioxide_core::quantity!("workspace_bytes"),
            )?;
            let elements = checked_add(
                checked_mul(n, r, scioxide_core::quantity!("workspace_bytes"))?,
                checked_mul(r, r, scioxide_core::quantity!("workspace_bytes"))?,
                scioxide_core::quantity!("workspace_bytes"),
            )?;
            checked_add(
                checked_mul(
                    elements,
                    128 * scalar,
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                checked_mul(
                    checked_add(
                        *operator_scratch,
                        *preconditioner_scratch,
                        scioxide_core::quantity!("workspace_bytes"),
                    )?,
                    4 * scalar,
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                scioxide_core::quantity!("workspace_bytes"),
            )?
        }
    };
    checked_add(
        32768,
        checked_add(
            linear,
            checked_mul(vectors, scalar, scioxide_core::quantity!("workspace_bytes"))?,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        scioxide_core::quantity!("workspace_bytes"),
    )
}

pub(super) struct Transpose<'a, T: NumericScalar, P: DaeAdjointSystem<T>> {
    pub problem: &'a P,
    pub options: &'a crate::DaeNumericalOptions<T::Real>,
    pub partition: &'a DaePartition,
    pub time: T::Real,
    pub state: &'a [T],
    pub derivative: &'a [T],
    pub state_weights: &'a [T::Real],
    pub derivative_weights: &'a [T::Real],
    pub original_pattern: Option<StructuralPatternView<'a>>,
    pub permutation: &'a [usize],
    pub parameter_scratch: RefCell<&'a mut [T::Real]>,
    pub provider_scratch: usize,
    pub preconditioner: Option<&'a dyn Preconditioner<T>>,
}
impl<T: NumericScalar, P: DaeAdjointSystem<T>> Transpose<'_, T, P> {
    fn point(&self) -> crate::dae::derivatives::Point<'_, T, P> {
        crate::dae::derivatives::Point {
            problem: self.problem,
            partition: self.partition,
            options: self.options,
            time: self.time,
            state: self.state,
            derivative: self.derivative,
            base: None,
        }
    }
}
impl<T: NumericScalar, P: DaeAdjointSystem<T>> LinearOperator<T> for Transpose<'_, T, P> {
    fn rows(&self) -> usize {
        self.state.len()
    }
    fn cols(&self) -> usize {
        self.state.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        checked_add(
            checked_mul(
                self.rows(),
                2,
                scioxide_core::quantity!("direction_scratch"),
            )?,
            self.provider_scratch,
            scioxide_core::quantity!("operator_scratch"),
        )
    }
    fn apply_into(&self, _: &[T], _: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        Err(invalid(
            scioxide_core::argument!("context"),
            "adjoint residual operators require a shared solve context",
        ))
    }
    fn apply_into_scoped(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        let n = self.rows();
        if input.len() != n || output.len() != n || scratch.len() < self.apply_workspace_len()? {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "adjoint residual operator shape/scratch",
            ));
        }
        let (y, rest) = scratch.split_at_mut(n);
        let (v, provider) = rest.split_at_mut(n);
        let mut parameters = self.parameter_scratch.borrow_mut();
        y.fill(T::from_real(T::Real::NAN));
        v.fill(T::from_real(T::Real::NAN));
        parameters.fill(T::Real::NAN);
        self.problem.pullback(
            self.time,
            self.state,
            self.derivative,
            input,
            y,
            v,
            &mut parameters,
            provider,
            context,
        )?;
        finite(y, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(v, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(&parameters, scioxide_core::NonFiniteOrigin::Callback)?;
        for (i, out) in output.iter_mut().enumerate() {
            *out = T::from_real(self.state_weights[i]) * y[i]
                + T::from_real(self.derivative_weights[i]) * v[i];
        }
        finite(output, scioxide_core::NonFiniteOrigin::Callback)
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<T: NumericScalar, P: DaeAdjointSystem<T>> EffectiveJacobian<T> for Transpose<'_, T, P> {
    fn dense(
        &mut self,
        output: &mut Matrix<T>,
        other: &mut Matrix<T>,
        differences: &mut crate::dae::derivatives::Blocks<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        differences.acquire(
            self.point(),
            crate::dae::derivatives::Output::Dense(output, other),
            context,
        )
    }
    fn assemble_dense(
        &self,
        output: &mut Matrix<T>,
        state: &Matrix<T>,
        derivative: &Matrix<T>,
    ) -> Result<(), SciOxideError> {
        let n = self.rows();
        for (index, out) in output.as_mut_slice().iter_mut().enumerate() {
            let row = index / n;
            let col = index % n;
            let source = col * n + row;
            *out = crate::dae::mixed_entry(
                self.partition,
                col,
                row,
                state.as_slice()[source],
                derivative.as_slice()[source],
                (self.state_weights[row], self.derivative_weights[row]),
            )?
            .conj();
        }
        finite(output.as_slice(), scioxide_core::NonFiniteOrigin::Callback)
    }
    fn sparse(
        &mut self,
        _: StructuralPatternView<'_>,
        output: &mut [T],
        other: &mut [T],
        differences: &mut crate::dae::derivatives::Blocks<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        let pattern = self.original_pattern.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("backend"),
                "missing original sparse adjoint pattern",
            )
        })?;
        differences.acquire(
            self.point(),
            crate::dae::derivatives::Output::Sparse(pattern, output, other),
            context,
        )
    }
    fn assemble_sparse(
        &self,
        pattern: StructuralPatternView<'_>,
        output: &mut [T],
        state: &[T],
        derivative: &[T],
    ) -> Result<(), SciOxideError> {
        for row in 0..pattern.rows() {
            let range = pattern.row_ptr()[row]..pattern.row_ptr()[row + 1];
            for ((out, &col), &source) in output[range.clone()]
                .iter_mut()
                .zip(&pattern.col_indices()[range.clone()])
                .zip(&self.permutation[range])
            {
                *out = crate::dae::mixed_entry(
                    self.partition,
                    col,
                    row,
                    state[source],
                    derivative[source],
                    (self.state_weights[row], self.derivative_weights[row]),
                )?
                .conj();
            }
        }
        finite(output, scioxide_core::NonFiniteOrigin::Callback)
    }
    fn preconditioner(&self) -> Option<&dyn Preconditioner<T>> {
        self.preconditioner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_storage_uses_factor_requirements_instead_of_the_byte_ceiling()
    -> Result<(), SciOxideError> {
        let matrix = scioxide_sparse::CsrMatrix::from_raw(
            2,
            2,
            alloc::vec![0, 1, 2],
            alloc::vec![0, 1],
            alloc::vec![1.0_f64, 1.0],
        )?;
        let backend = |bytes| -> Result<_, SciOxideError> {
            Ok(OdeStageBackend::Sparse {
                pattern: matrix.structural_pattern().to_owned()?,
                limits: scioxide_sparse::SparseLuLimits::new(2, bytes),
            })
        };
        let unlimited = storage_bound::<f64>(&backend(usize::MAX)?, 2, 1, 0)?;
        assert_eq!(
            unlimited,
            storage_bound::<f64>(&backend(1 << 20)?, 2, 1, 0)?
        );
        assert!(unlimited > 0 && unlimited < 1 << 20);
        Ok(())
    }
}
