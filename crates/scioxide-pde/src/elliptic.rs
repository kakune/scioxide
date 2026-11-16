//! Direct elliptic kernels: compact (cyclic) tridiagonal bands for one-dimensional
//! plans and fill-reducing sparse LU for every other prepared operator.
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{
    CyclicTridiagonalPlan, DirectSolveOptions, LinalgExecutionOptions, LinalgWorkspace, RankPolicy,
    StructuredOperation, TridiagonalMatrixView, Vector, VectorViewMut,
};
use scioxide_sparse::CsrMatrix;

/// Retained tridiagonal factor of the last compact solve, refactored in place.
pub(crate) type FactorCache<S> =
    Option<scioxide_linalg::backend::tridiagonal::TridiagonalFactor<S>>;

/// Numerical storage reused across elliptic solves of one workspace.
pub(crate) struct EllipticStorage<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) factor: FactorCache<S>,
    pub(crate) iterative: scioxide_linalg::IterativeWorkspace<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for EllipticStorage<S> {
    fn default() -> Self {
        Self {
            factor: None,
            iterative: scioxide_linalg::IterativeWorkspace::new(),
        }
    }
}

/// Canonical bands; periodic corner contributions are folded for orders one and two.
pub(crate) struct EllipticMatrix<S> {
    pub(crate) main: Vec<S>,
    pub(crate) lower: Vec<S>,
    pub(crate) upper: Vec<S>,
    pub(crate) corner_upper: S,
    pub(crate) corner_lower: S,
}
impl<S: RealScalar + NumericScalar<Real = S>> EllipticMatrix<S> {
    /// Extract (cyclic) tridiagonal bands, or `None` when the pattern is wider.
    pub(crate) fn from_csr(csr: &CsrMatrix<S>) -> Result<Option<Self>, SciOxideError> {
        let n = csr.rows();
        let mut matrix = Self::new(n)?;
        for (row, col, value) in csr.iter() {
            let banded = row.abs_diff(col) <= 1
                || n > 2 && (row == 0 && col + 1 == n || col == 0 && row + 1 == n);
            if !banded {
                return Ok(None);
            }
            matrix.push(row, col, value)?;
        }
        Ok(Some(matrix))
    }
    pub(crate) fn new(n: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            main: try_filled(n, S::ZERO, scioxide_core::quantity!("elliptic_diagonal"))?,
            lower: try_filled(
                n.saturating_sub(1),
                S::ZERO,
                scioxide_core::quantity!("elliptic_lower"),
            )?,
            upper: try_filled(
                n.saturating_sub(1),
                S::ZERO,
                scioxide_core::quantity!("elliptic_upper"),
            )?,
            corner_upper: S::ZERO,
            corner_lower: S::ZERO,
        })
    }
    pub(crate) fn rows(&self) -> usize {
        self.main.len()
    }
    #[inline(always)]
    pub(crate) fn push(&mut self, row: usize, col: usize, value: S) -> Result<(), SciOxideError> {
        let n = self.rows();
        if row >= n || col >= n {
            return Err(crate::validation::invalid(
                scioxide_core::argument!("matrix"),
                "band index outside the matrix",
            ));
        }
        let target = if row == col {
            &mut self.main[row]
        } else if row == col + 1 {
            &mut self.lower[col]
        } else if col == row + 1 {
            &mut self.upper[row]
        } else if row == 0 && col + 1 == n {
            &mut self.corner_upper
        } else if col == 0 && row + 1 == n {
            &mut self.corner_lower
        } else {
            return Err(crate::validation::invalid(
                scioxide_core::argument!("matrix"),
                "entry outside cyclic tridiagonal bands",
            ));
        };
        let sum = *target + value;
        *target = if sum.is_finite() {
            if sum == S::ZERO { S::ZERO } else { sum }
        } else {
            scioxide_core::checked_sum(&[*target, value])?
        };
        Ok(())
    }
    /// Independently evaluate original equations with Standard arithmetic and range recovery.
    pub(crate) fn residual(
        &self,
        rhs: &[S],
        x: &[S],
    ) -> Result<scioxide_core::ResidualReport<S>, SciOxideError> {
        let mut raw = S::ZERO;
        let mut anorm = S::ZERO;
        let n = self.rows();
        for (row, &b) in rhs.iter().enumerate() {
            let diagonal = self.main[row] * x[row];
            let mut dot = diagonal;
            let mut row_norm = <S as NumericScalar>::abs(self.main[row]);
            let mut retry = scioxide_core::backend::standard::product_needs_retry(
                self.main[row],
                x[row],
                diagonal,
            );
            if row > 0 {
                let a = self.lower[row - 1];
                let product = a * x[row - 1];
                dot += product;
                row_norm += <S as NumericScalar>::abs(a);
                retry |=
                    scioxide_core::backend::standard::product_needs_retry(a, x[row - 1], product);
            }
            if row + 1 < n {
                let a = self.upper[row];
                let product = a * x[row + 1];
                dot += product;
                row_norm += <S as NumericScalar>::abs(a);
                retry |=
                    scioxide_core::backend::standard::product_needs_retry(a, x[row + 1], product);
            }
            if row == 0 && n > 2 {
                let product = self.corner_upper * x[n - 1];
                dot += product;
                row_norm += <S as NumericScalar>::abs(self.corner_upper);
                retry |= scioxide_core::backend::standard::product_needs_retry(
                    self.corner_upper,
                    x[n - 1],
                    product,
                );
            }
            if row + 1 == n && n > 2 {
                let product = self.corner_lower * x[0];
                dot += product;
                row_norm += <S as NumericScalar>::abs(self.corner_lower);
                retry |= scioxide_core::backend::standard::product_needs_retry(
                    self.corner_lower,
                    x[0],
                    product,
                );
            }
            let mut residual = b - dot;
            if retry || !residual.is_finite() || !row_norm.is_finite() {
                let entries = [
                    (row, self.main[row]),
                    (
                        row.saturating_sub(1),
                        if row > 0 {
                            self.lower[row - 1]
                        } else {
                            S::ZERO
                        },
                    ),
                    (
                        (row + 1).min(n - 1),
                        if row + 1 < n {
                            self.upper[row]
                        } else {
                            S::ZERO
                        },
                    ),
                    (
                        0,
                        if row + 1 == n && n > 2 {
                            self.corner_lower
                        } else {
                            S::ZERO
                        },
                    ),
                    (
                        n - 1,
                        if row == 0 && n > 2 {
                            self.corner_upper
                        } else {
                            S::ZERO
                        },
                    ),
                ];
                residual = scioxide_core::backend::standard::sum_products(
                    core::iter::once((b, S::ONE))
                        .chain(entries.into_iter().map(|(j, a)| (-a, x[j]))),
                    scioxide_core::Accuracy::Exact,
                )?;
                row_norm = scioxide_core::backend::standard::sum_products(
                    entries
                        .into_iter()
                        .map(|(_, a)| (<S as NumericScalar>::abs(a), S::ONE)),
                    scioxide_core::Accuracy::Exact,
                )?;
            }
            raw = raw.max(<S as NumericScalar>::abs(residual));
            anorm = anorm.max(row_norm);
        }
        let xnorm = x
            .iter()
            .fold(S::ZERO, |s, &v| s.max(<S as NumericScalar>::abs(v)));
        let bnorm = rhs
            .iter()
            .fold(S::ZERO, |s, &v| s.max(<S as NumericScalar>::abs(v)));
        let scale = scioxide_core::Scaled::from_value(anorm)?
            .mul(scioxide_core::Scaled::from_value(xnorm)?)?
            .add(scioxide_core::Scaled::from_value(bnorm)?)?
            .round()?;
        let scaling = if scale == S::ZERO {
            scioxide_core::ResidualScaling::unscaled()
        } else {
            scioxide_core::ResidualScaling::new(
                scioxide_core::ResidualScalingRule::BackwardError,
                scale,
            )?
        };
        scioxide_core::ResidualReport::new(raw, scioxide_core::NormSelection::Infinity, scaling)
    }
}

/// Solve the complete bands, or with `components` constant modes pin the first
/// unknown of every component block, solve each remaining block and remove each
/// block mean, as [`sparse_direct`] does.
pub(crate) fn direct<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &EllipticMatrix<S>,
    rhs: &[S],
    pinned_components: Option<usize>,
    workspace: &mut FactorCache<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<Vector<S>, SciOxideError> {
    let n = matrix.rows();
    let capacity =
        scioxide_core::backend::checked_add(n, 2, scioxide_core::quantity!("elliptic_solution"))?;
    let mut values = try_with_capacity(capacity, scioxide_core::quantity!("elliptic_solution"))?;
    values.extend_from_slice(rhs);
    let mut x = Vector::from_vec(values);
    let Some(components) = pinned_components else {
        if n > 2 && (matrix.corner_upper != S::ZERO || matrix.corner_lower != S::ZERO) {
            let view = TridiagonalMatrixView::new(&matrix.lower, &matrix.main, &matrix.upper)?;
            let plan = CyclicTridiagonalPlan::build(
                view,
                matrix.corner_upper,
                matrix.corner_lower,
                &LinalgExecutionOptions::new(),
                &mut *context,
            )?;
            let report = plan.solve_into(
                StructuredOperation::Normal,
                rhs,
                &DirectSolveOptions::new(RankPolicy::Exact),
                VectorViewMut::from_slice(x.as_mut_slice()),
                &mut LinalgWorkspace::new(),
                &mut *context,
            )?;
            if !report.termination().is_success() {
                return Err(SciOxideError::singular(None));
            }
        } else {
            solve_band(matrix, 0..n, x.as_mut_slice(), workspace, context)?;
        }
        return Ok(x);
    };
    if components == 0 || !n.is_multiple_of(components) {
        return Err(crate::validation::invalid(
            scioxide_core::argument!("matrix"),
            "component blocks do not partition the unknowns",
        ));
    }
    let block = n / components;
    // Pinning the first unknown of each block removes every band entry that
    // couples neighbouring blocks, so the remaining blocks are solved separately.
    for values in x.as_mut_slice().chunks_exact_mut(block) {
        values[0] = S::ZERO;
    }
    for start in (0..n).step_by(block) {
        let range = start + 1..start + block;
        if !range.is_empty() {
            let output = &mut x.as_mut_slice()[range.clone()];
            solve_band(matrix, range, output, workspace, context)?;
        }
    }
    for values in x.as_mut_slice().chunks_exact_mut(block) {
        let mean = scioxide_core::checked_sum(values)? / S::try_from_usize_exact(block)?;
        for v in values {
            *v = scioxide_core::checked_difference(*v, mean)?;
        }
    }
    Ok(x)
}

/// Factor the non-cyclic bands of the unknowns in `range` into the cached factor
/// and solve in place.
fn solve_band<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &EllipticMatrix<S>,
    range: core::ops::Range<usize>,
    values: &mut [S],
    workspace: &mut FactorCache<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let couplings = range.start..range.end.saturating_sub(1).max(range.start);
    let lower = &matrix.lower[couplings.clone()];
    let main = &matrix.main[range];
    let upper = &matrix.upper[couplings];
    if let Some(factor) = workspace.as_mut() {
        factor.refactor(lower, main, upper, context.execution())?;
    } else {
        let copy = scioxide_core::backend::try_copy_slice;
        *workspace = Some(
            scioxide_linalg::backend::tridiagonal::TridiagonalFactor::factor(
                copy(lower, scioxide_core::quantity!("elliptic_lower_factor"))?,
                copy(main, scioxide_core::quantity!("elliptic_main_factor"))?,
                copy(upper, scioxide_core::quantity!("elliptic_upper_factor"))?,
                context.execution(),
            )?,
        );
    }
    let factor = workspace.as_ref().ok_or_else(|| {
        crate::validation::invalid(
            scioxide_core::argument!("workspace"),
            "factor preparation omitted its output",
        )
    })?;
    factor.solve_in_place(values, context.execution())
}

/// Solve with fill-reducing sparse LU. With `components` constant modes, the
/// first unknown of every component block is pinned and each block mean removed.
pub(crate) fn sparse_direct<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &CsrMatrix<S>,
    rhs: &[S],
    pinned_components: Option<usize>,
    limits: Option<scioxide_sparse::SparseLuLimits>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<Vec<S>, SciOxideError> {
    use scioxide_sparse::{
        DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder, SparseLu, SparseLuLimits,
        SparseLuSymbolic, SparseOperation, SparseOrdering, SparseSolveOptions,
    };
    let n = matrix.rows();
    let block = pinned_components.map_or(n, |components| n / components);
    let pinned = |i: usize| pinned_components.is_some() && i.is_multiple_of(block);
    let mut reduced = try_filled(n, None, scioxide_core::quantity!("pinned_unknowns"))?;
    let mut free = 0;
    for (i, slot) in reduced.iter_mut().enumerate() {
        if !pinned(i) {
            *slot = Some(free);
            free += 1;
        }
    }
    let mut builder = SparseBuilder::with_capacity(free, free, matrix.nnz())?;
    for (row, col, value) in matrix.iter() {
        if let (Some(r), Some(c)) = (reduced[row], reduced[col]) {
            builder.push(r, c, value)?;
        }
    }
    let system = builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)?;
    let mut load = try_with_capacity(free, scioxide_core::quantity!("elliptic_load"))?;
    load.extend((0..n).filter(|&i| !pinned(i)).map(|i| rhs[i]));
    let entries = system
        .nnz()
        .saturating_add(free)
        .saturating_mul(64)
        .max(1 << 12);
    let limits = limits.unwrap_or_else(|| SparseLuLimits::new(entries, entries.saturating_mul(64)));
    let symbolic = SparseLuSymbolic::analyze(
        system.structural_pattern(),
        SparseOrdering::FillReducing,
        limits,
    )?;
    let factor = SparseLu::factor_scoped(
        symbolic,
        system.view(),
        SparseSolveOptions::new(),
        &mut *context,
    )?;
    let mut solution = try_filled(free, S::ZERO, scioxide_core::quantity!("elliptic_solution"))?;
    factor.solve_into(
        &load,
        SparseOperation::Normal,
        SparseSolveOptions::new(),
        &mut solution,
        &mut factor.workspace(1)?,
        &mut *context,
    )?;
    let mut x = try_filled(n, S::ZERO, scioxide_core::quantity!("elliptic_solution"))?;
    for (i, slot) in reduced.iter().enumerate() {
        if let Some(r) = slot {
            x[i] = solution[*r];
        }
    }
    if pinned_components.is_some() {
        for values in x.chunks_exact_mut(block) {
            let mean = scioxide_core::checked_sum(values)? / S::try_from_usize_exact(block)?;
            for v in values {
                *v = scioxide_core::checked_difference(*v, mean)?;
            }
        }
    }
    Ok(x)
}

/// Infinity-norm backward-error report of `A x = b` for a sparse operator.
pub(crate) fn sparse_residual<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &CsrMatrix<S>,
    rhs: &[S],
    x: &[S],
) -> Result<scioxide_core::ResidualReport<S>, SciOxideError> {
    let (pointers, columns, values) = (matrix.row_ptr(), matrix.col_indices(), matrix.values());
    let mut raw = S::ZERO;
    let mut anorm = S::ZERO;
    for (row, &b) in rhs.iter().enumerate() {
        let entries = (pointers[row]..pointers[row + 1]).map(|k| (values[k], x[columns[k]]));
        let residual = scioxide_core::backend::standard::sum_products(
            core::iter::once((b, S::ONE)).chain(entries.clone().map(|(a, v)| (-a, v))),
            scioxide_core::Accuracy::Exact,
        )?;
        let row_norm = scioxide_core::backend::standard::sum_products(
            entries.map(|(a, _)| (<S as NumericScalar>::abs(a), S::ONE)),
            scioxide_core::Accuracy::Exact,
        )?;
        raw = raw.max(<S as NumericScalar>::abs(residual));
        anorm = anorm.max(row_norm);
    }
    let xnorm = x
        .iter()
        .fold(S::ZERO, |s, &v| s.max(<S as NumericScalar>::abs(v)));
    let bnorm = rhs
        .iter()
        .fold(S::ZERO, |s, &v| s.max(<S as NumericScalar>::abs(v)));
    let scale = scioxide_core::Scaled::from_value(anorm)?
        .mul(scioxide_core::Scaled::from_value(xnorm)?)?
        .add(scioxide_core::Scaled::from_value(bnorm)?)?
        .round()?;
    let scaling = if scale == S::ZERO {
        scioxide_core::ResidualScaling::unscaled()
    } else {
        scioxide_core::ResidualScaling::new(
            scioxide_core::ResidualScalingRule::BackwardError,
            scale,
        )?
    };
    scioxide_core::ResidualReport::new(raw, scioxide_core::NormSelection::Infinity, scaling)
}
