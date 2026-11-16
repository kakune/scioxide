//! Sparse stage factors selected by the Jacobian pattern.
//!
//! A pattern confined to the main and first off-diagonals uses a tridiagonal LU when its
//! limits allow one. Collocation uses compact bordered block elimination; other
//! patterns use the general sparse LU.
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{ExecutionContext, NumericScalar, SciOxideError};
use scioxide_linalg::backend::bordered_bidiagonal::BorderedBidiagonalFactor;
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, StructuredOperation, StructuredWorkspaceRequest,
    TridiagonalLuDecomposition, TridiagonalMatrixView, VectorViewMut,
};
use scioxide_sparse::{SparseLu, SparseLuSymbolic, SparseLuWorkspace, SparseOperation, SparseView};

// The tridiagonal workspace keeps its solve-scope metadata inline so construction of
// its scopes never allocates; one factor exists per stage solver.
#[allow(clippy::large_enum_variant)]
pub(crate) enum SparseStageFactor<T: NumericScalar> {
    Bordered {
        pattern: scioxide_core::StructuralPattern,
        factor: BorderedBidiagonalFactor<T>,
        index_bytes: usize,
        limits: scioxide_sparse::SparseLuLimits,
    },
    General {
        factor: Option<SparseLu<T>>,
        work: SparseLuWorkspace<T>,
    },
    Tridiagonal {
        symbolic: SparseLuSymbolic,
        lower: Vec<T>,
        main: Vec<T>,
        upper: Vec<T>,
        factor: TridiagonalLuDecomposition<T>,
        work: LinalgWorkspace<'static, T>,
    },
}

impl<T: NumericScalar> SparseStageFactor<T> {
    /// Select and reserve a tridiagonal or general factor.
    pub(crate) fn new(symbolic: SparseLuSymbolic) -> Result<Self, SciOxideError> {
        let n = symbolic.dimension();
        let p = symbolic.pattern();
        let limits = symbolic.limits();
        let tridiagonal = (0..n).all(|i| {
            p.col_indices()[p.row_ptr()[i]..p.row_ptr()[i + 1]]
                .iter()
                .all(|j| i.abs_diff(*j) <= 1)
        });
        // Conservative payload bound: symbolic arrays, the published pivoted
        // factor and its workspace candidate, three input diagonals, the
        // staged solve vector, and minimum Vec capacities.
        let bytes = checked_add(
            symbolic.workspace_requirements::<T>(())?.symbolic_bytes(),
            checked_add(
                checked_mul(
                    n.max(4),
                    20 * core::mem::size_of::<T>(),
                    scioxide_core::quantity!("sparse_stage.value_bytes"),
                )?,
                checked_mul(
                    n.max(4),
                    8 * core::mem::size_of::<usize>(),
                    scioxide_core::quantity!("sparse_stage.index_bytes"),
                )?,
                scioxide_core::quantity!("sparse_stage.numeric_bytes"),
            )?,
            scioxide_core::quantity!("sparse_stage.workspace_bytes"),
        )?;
        if tridiagonal
            && limits.max_factor_entries()
                >= checked_mul(n, 4, scioxide_core::quantity!("sparse_stage.band_entries"))?
            && bytes <= limits.max_workspace_bytes()
        {
            let options = LinalgExecutionOptions::new();
            let requirements = TridiagonalLuDecomposition::<T>::workspace_requirements(
                &StructuredWorkspaceRequest::factor(n, 1, 1, &options),
            )?
            .max(&TridiagonalLuDecomposition::<T>::workspace_requirements(
                &StructuredWorkspaceRequest::solve(n, &options),
            )?);
            let mut work = LinalgWorkspace::new();
            work.try_reserve(&requirements)?;
            let lower = try_filled(
                n.saturating_sub(1),
                T::ZERO,
                scioxide_core::quantity!("sparse_stage.lower_diagonal"),
            )?;
            let upper = try_filled(
                n.saturating_sub(1),
                T::ZERO,
                scioxide_core::quantity!("sparse_stage.upper_diagonal"),
            )?;
            // Factoring the identity grows the factor storage once here, so
            // later refactors and solves of this order make no allocation.
            let identity = try_filled(
                n,
                T::ONE,
                scioxide_core::quantity!("sparse_stage.main_diagonal"),
            )?;
            let mut factor = TridiagonalLuDecomposition::empty();
            TridiagonalLuDecomposition::factor_into(
                TridiagonalMatrixView::new(&lower, &identity, &upper)?,
                &options,
                &mut factor,
                &mut work,
                ExecutionContext::sequential(),
            )?;
            let mut main = identity;
            main.fill(T::ZERO);
            Ok(Self::Tridiagonal {
                symbolic,
                lower,
                main,
                upper,
                factor,
                work,
            })
        } else {
            let work = SparseLuWorkspace::prepare(symbolic)?;
            Ok(Self::General { factor: None, work })
        }
    }
    /// Select compact frontal elimination when the block structure and its
    /// conservative storage fit; otherwise keep the general sparse factor.
    pub(crate) fn bordered(
        pattern: scioxide_core::StructuralPatternView<'_>,
        n: usize,
        k: usize,
        nodes: usize,
        limits: scioxide_sparse::SparseLuLimits,
    ) -> Result<Self, SciOxideError> {
        let q = scioxide_core::quantity!("bordered_factor_bytes");
        let bytes = checked_add(
            checked_mul(
                BorderedBidiagonalFactor::<T>::required_bytes(n, k, nodes)?,
                2,
                q,
            )?,
            checked_mul(
                pattern.row_ptr().len() + pattern.col_indices().len(),
                2 * core::mem::size_of::<usize>(),
                q,
            )?,
            q,
        )?;
        if BorderedBidiagonalFactor::<T>::required_entries(n, k, nodes)?
            <= limits.max_factor_entries()
            && bytes <= limits.max_workspace_bytes()
        {
            Ok(Self::Bordered {
                pattern: pattern.to_owned()?,
                factor: BorderedBidiagonalFactor::new(n, k, nodes)?,
                index_bytes: (pattern.row_ptr().len() + pattern.col_indices().len())
                    * 2
                    * core::mem::size_of::<usize>(),
                limits,
            })
        } else {
            Self::new(SparseLuSymbolic::analyze(
                pattern,
                scioxide_sparse::SparseOrdering::Natural,
                limits,
            )?)
        }
    }
    /// Reprepare natural-order storage without discarding a general factor's capacity.
    pub(crate) fn reprepare_natural(
        &mut self,
        pattern: scioxide_core::StructuralPatternView<'_>,
        limits: scioxide_sparse::SparseLuLimits,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Bordered {
                pattern: stored,
                factor,
                index_bytes,
                limits: bounds,
            } => {
                let (n, k) = factor.block_shape();
                let nodes = (pattern.rows() - k) / n;
                if BorderedBidiagonalFactor::<T>::required_entries(n, k, nodes)?
                    > limits.max_factor_entries()
                {
                    *self = Self::new(SparseLuSymbolic::analyze(
                        pattern,
                        scioxide_sparse::SparseOrdering::Natural,
                        limits,
                    )?)?;
                } else {
                    let indices = (*index_bytes).max(
                        (pattern.row_ptr().len() + pattern.col_indices().len())
                            * 2
                            * core::mem::size_of::<usize>(),
                    );
                    let bytes = checked_add(
                        checked_mul(
                            factor
                                .retained_bytes()
                                .max(BorderedBidiagonalFactor::<T>::required_bytes(n, k, nodes)?),
                            2,
                            scioxide_core::quantity!("bordered_factor_bytes"),
                        )?,
                        indices,
                        scioxide_core::quantity!("bordered_factor_bytes"),
                    )?;
                    if bytes > limits.max_workspace_bytes() {
                        return Err(SciOxideError::resource_limit(
                            scioxide_core::quantity!("bordered_factor_bytes"),
                            bytes,
                            limits.max_workspace_bytes(),
                        ));
                    }
                    stored.try_assign(pattern)?;
                    factor.resize(nodes)?;
                    *bounds = limits;
                    *index_bytes = indices;
                }
                Ok(())
            }
            Self::General { factor, work } => {
                work.reprepare_natural(factor.take(), pattern, limits)
            }
            Self::Tridiagonal { .. } => {
                *self = Self::new(SparseLuSymbolic::analyze(
                    pattern,
                    scioxide_sparse::SparseOrdering::Natural,
                    limits,
                )?)?;
                Ok(())
            }
        }
    }
    pub(crate) fn pattern(
        &self,
    ) -> Result<scioxide_core::StructuralPatternView<'_>, SciOxideError> {
        match self {
            Self::Bordered { pattern, .. } => Ok(pattern.view()),
            Self::General { factor, work } => factor
                .as_ref()
                .map(SparseLu::symbolic)
                .or_else(|| work.initial_symbolic())
                .map(SparseLuSymbolic::pattern)
                .ok_or_else(|| SciOxideError::singular(None)),
            Self::Tridiagonal { symbolic, .. } => Ok(symbolic.pattern()),
        }
    }
    pub(crate) fn refactor(
        &mut self,
        matrix: SparseView<'_, T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Bordered { factor, .. } => factor.factor(matrix.values(), execution),
            Self::General { factor, work } => {
                if let Some(factor) = factor {
                    factor.refactor(
                        matrix,
                        scioxide_sparse::SparseSolveOptions::new(),
                        work,
                        execution,
                    )
                } else {
                    *factor = Some(work.factor_scoped(
                        matrix,
                        scioxide_sparse::SparseSolveOptions::new(),
                        execution,
                    )?);
                    Ok(())
                }
            }
            Self::Tridiagonal {
                symbolic,
                lower,
                main,
                upper,
                factor,
                work,
                ..
            } => {
                lower.fill(T::ZERO);
                main.fill(T::ZERO);
                upper.fill(T::ZERO);
                let p = symbolic.pattern();
                for i in 0..p.rows() {
                    for k in p.row_ptr()[i]..p.row_ptr()[i + 1] {
                        let j = p.col_indices()[k];
                        let v = matrix.values()[k];
                        if i == j {
                            main[i] = v;
                        } else if j < i {
                            lower[j] = v;
                        } else {
                            upper[i] = v;
                        }
                    }
                }
                TridiagonalLuDecomposition::factor_into(
                    TridiagonalMatrixView::new(lower, main, upper)?,
                    &LinalgExecutionOptions::new(),
                    factor,
                    work,
                    execution,
                )
            }
        }
    }
    pub(crate) fn solve_into(
        &mut self,
        input: &[T],
        out: &mut [T],
        scratch: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Bordered { factor, .. } => {
                scratch[..input.len()].copy_from_slice(input);
                factor.solve_in_place(&mut scratch[..input.len()], execution)?;
                out.copy_from_slice(&scratch[..input.len()]);
                Ok(())
            }
            Self::General { factor, .. } => factor
                .as_ref()
                .ok_or_else(|| SciOxideError::singular(None))?
                .solve_into(
                    input,
                    SparseOperation::Normal,
                    scioxide_sparse::SparseSolveOptions::new(),
                    out,
                    &mut scioxide_sparse::SparseSolveWorkspace::borrowed(scratch),
                    execution,
                ),
            Self::Tridiagonal { factor, work, .. } => {
                execution.check()?;
                factor
                    .solve_into(
                        StructuredOperation::Normal,
                        input,
                        &scioxide_linalg::DirectSolveOptions::new(
                            scioxide_linalg::RankPolicy::Exact,
                        ),
                        VectorViewMut::from_slice(out),
                        work,
                        execution,
                    )
                    .and_then(|report| report.termination().require_accepted())
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::AlgebraicScalar;
    use scioxide_core::RealScalar;
    use scioxide_sparse::{CsrMatrix, SparseLuLimits};

    fn qualify<T: NumericScalar>() {
        let v = |x| T::from_components(T::Real::try_from_f64_lossy(x).unwrap(), T::Real::ZERO);
        let matrix = CsrMatrix::from_raw(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![v(0.), v(1.), v(1.), v(2.)],
        )
        .unwrap();
        let pattern = matrix.structural_pattern();
        for capacity in [4, 8] {
            let symbolic = SparseLuSymbolic::analyze(
                pattern,
                scioxide_sparse::SparseOrdering::default(),
                SparseLuLimits::new(capacity, 16_384),
            )
            .unwrap();
            let mut factor = SparseStageFactor::new(symbolic).unwrap();
            assert_eq!(
                matches!(factor, SparseStageFactor::Tridiagonal { .. }),
                capacity == 8
            );
            factor
                .refactor(matrix.view(), ExecutionContext::sequential())
                .unwrap();
            let mut out = [v(0.); 2];
            factor
                .solve_into(
                    &[v(2.), v(5.)],
                    &mut out,
                    &mut [v(0.); 4],
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!((out[0] - v(1.)).abs() < T::Real::try_from_f64_lossy(1e-5).unwrap());
            assert!((out[1] - v(2.)).abs() < T::Real::try_from_f64_lossy(1e-5).unwrap());
        }
    }
    #[test]
    fn tridiagonal_and_general_budget_choices_preserve_pivots() {
        qualify::<f32>();
        qualify::<f64>();
        qualify::<scioxide_core::Complex32>();
        qualify::<scioxide_core::Complex64>();
    }
}
