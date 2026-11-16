//! Cuppen splitting, rank-one secular-equation merges and Löwner-formula weights.
use super::*;
use crate::FiniteMatrixView;
use crate::dense::{MatrixView, MatrixViewMut};
use crate::kernel::{GemmConfig, WorkspaceGemmProblem, gemm_workspace_with_context};
use crate::secular::breakdown;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Budget, NumericalIssue};

pub(super) const LEAF: usize = 24;

/// Secular-merge scratch lent by the job's direct arenas.
pub(super) struct Workspace<'a, S: NumericScalar> {
    original_diagonal: &'a mut [S::Real],
    original_off: &'a mut [S::Real],
    poles: &'a mut [S::Real],
    weights: &'a mut [S::Real],
    origins: &'a mut [S::Real],
    deltas: &'a mut [S::Real],
    secular_poles: &'a mut [S::Real],
    secular_weights: &'a mut [S::Real],
    active: &'a mut [usize],
    order: &'a mut [usize],
    permutation: &'a mut [usize],
    /// Child support classes: 0 = left, 1 = mixed, 2 = right.
    support: &'a mut [usize],
    basis: ScratchMatrix<'a, S>,
    coefficients: ScratchMatrix<'a, S>,
    leaf: ScratchMatrix<'a, S>,
}

impl<'a, S: NumericScalar> Workspace<'a, S> {
    /// Scalar, real and index elements of an order-`n` merge tree.
    pub(super) fn lengths(n: usize) -> Result<[usize; 3], SciOxideError> {
        let overflow =
            || SciOxideError::size_overflow(scioxide_core::quantity!("hermitian_secular_scratch"));
        let square = scioxide_core::backend::checked_product(
            &[n, n],
            scioxide_core::quantity!("hermitian_secular_scratch"),
        )?;
        let scalars = square
            .checked_mul(2)
            .and_then(|total| total.checked_add(LEAF * LEAF))
            .ok_or_else(overflow)?;
        let reals = n.checked_mul(8).ok_or_else(overflow)?;
        let indices = n.checked_mul(4).ok_or_else(overflow)?;
        Ok([scalars, reals, indices])
    }

    /// Carve an order-`n` merge tree from the front of each arena.
    pub(super) fn new(
        scalars: &'a mut [S],
        reals: &'a mut [S::Real],
        indices: &'a mut [usize],
        n: usize,
    ) -> Result<Self, SciOxideError> {
        let (basis, scalars) = crate::workspace::split_mut(scalars, n * n)?;
        let (coefficients, scalars) = crate::workspace::split_mut(scalars, n * n)?;
        let leaf = &mut scalars[..LEAF * LEAF];
        let (original_diagonal, reals) = crate::workspace::split_mut(reals, n)?;
        let (original_off, reals) = crate::workspace::split_mut(reals, n)?;
        let (poles, reals) = crate::workspace::split_mut(reals, n)?;
        let (weights, reals) = crate::workspace::split_mut(reals, n)?;
        let (origins, reals) = crate::workspace::split_mut(reals, n)?;
        let (deltas, reals) = crate::workspace::split_mut(reals, n)?;
        let (secular_poles, reals) = crate::workspace::split_mut(reals, n)?;
        let secular_weights = &mut reals[..n];
        let (active, indices) = crate::workspace::split_mut(indices, n)?;
        let (order, indices) = crate::workspace::split_mut(indices, n)?;
        let (permutation, indices) = crate::workspace::split_mut(indices, n)?;
        let support = &mut indices[..n];
        Ok(Self {
            original_diagonal,
            original_off,
            poles,
            weights,
            origins,
            deltas,
            secular_poles,
            secular_weights,
            active,
            order,
            permutation,
            support,
            basis: ScratchMatrix::new(basis, n, n)?,
            coefficients: ScratchMatrix::new(coefficients, n, n)?,
            leaf: ScratchMatrix::new(leaf, LEAF, LEAF)?,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub(super) fn solve(
        &mut self,
        diagonal: &mut [S::Real],
        off: &mut [S::Real],
        vectors: &mut impl MatrixStorage<S>,
        options: &HermitianEigenOptions<S::Real>,
        context: KernelContext<'_>,
        solve: &mut SolveContext<'_, '_>,
        gemm: &mut LinalgWorkspace<'_, S>,
    ) -> Result<(), SciOxideError> {
        let budget = options.resolved_iteration_budget(diagonal.len(), true)?;
        let options = &options.with_execution_options(
            options
                .execution_options()
                .with_iteration_budget(Budget::try_from(budget)?),
        );
        let mut used = 0;
        self.original_diagonal[..diagonal.len()].copy_from_slice(diagonal);
        self.original_off[..off.len()].copy_from_slice(off);
        let result = self.range(
            TridiagonalRange {
                diagonal,
                off,
                vectors,
                offset: 0,
            },
            RangePolicy {
                options,
                execution: context,
            },
            solve,
            &mut used,
            gemm,
        );
        if result.as_ref().is_err_and(|error| {
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::LossOfSignificance,
                    ..
                }
            )
        }) {
            diagonal.copy_from_slice(&self.original_diagonal[..diagonal.len()]);
            off.copy_from_slice(&self.original_off[..off.len()]);
            fill_identity(vectors);
            let remaining = budget.saturating_sub(used);
            if remaining == 0 {
                return Err(iteration_limit(used));
            }
            let fallback = (*options).with_execution_options(
                options
                    .execution_options()
                    .with_iteration_budget(Budget::ONE.saturating_add(remaining - 1)),
            );
            tridiagonal_ql(
                diagonal,
                off,
                Some(QlVectors {
                    matrix: vectors,
                    cosines: self.origins,
                    sines: self.deltas,
                }),
                &fallback,
                context.execution(),
                solve,
            )
            .map_err(|error| shared_limit(error, used, remaining))?;
            return Ok(());
        }
        result
    }

    fn range<M: MatrixStorage<S>>(
        &mut self,
        problem: TridiagonalRange<'_, S, M>,
        policy: RangePolicy<'_, S::Real>,
        solve: &mut SolveContext<'_, '_>,
        used: &mut usize,
        gemm: &mut LinalgWorkspace<'_, S>,
    ) -> Result<(), SciOxideError> {
        let TridiagonalRange {
            diagonal,
            off,
            vectors,
            offset,
        } = problem;
        let RangePolicy {
            options,
            execution: context,
        } = policy;

        context.execution().check()?;
        let n = diagonal.len();
        let budget = options.resolved_iteration_budget(n, true)?;
        let stride = vectors.cols();
        if n <= LEAF {
            self.leaf.try_resize_flat(n, n)?;
            fill_identity(&mut self.leaf);
            let remaining = budget.saturating_sub(*used);
            if remaining == 0 {
                return Err(iteration_limit(*used));
            }
            let local_options = (*options).with_execution_options(
                options
                    .execution_options()
                    .with_iteration_budget(Budget::ONE.saturating_add(remaining - 1)),
            );
            *used += tridiagonal_ql(
                diagonal,
                off,
                Some(QlVectors {
                    matrix: &mut self.leaf,
                    cosines: self.origins,
                    sines: self.deltas,
                }),
                &local_options,
                context.execution(),
                solve,
            )
            .map_err(|error| shared_limit(error, *used, remaining))?;
            for row in 0..n {
                for col in 0..n {
                    vectors.as_mut_slice()[(offset + row) * stride + offset + col] =
                        self.leaf.get_element(row, col);
                }
            }
            return Ok(());
        }
        let middle = n / 2;
        let coupling = off[middle - 1];
        let rho = coupling.abs();
        diagonal[middle - 1] -= rho;
        diagonal[middle] -= rho;
        off[middle - 1] = S::Real::ZERO;
        let (left, right) = crate::workspace::split_mut(diagonal, middle)?;
        let (left_off, right_off) = crate::workspace::split_mut(off, middle)?;
        self.range(
            TridiagonalRange {
                diagonal: left,
                off: left_off,
                vectors,
                offset,
            },
            RangePolicy {
                options,
                execution: context,
            },
            solve,
            used,
            gemm,
        )?;
        self.range(
            TridiagonalRange {
                diagonal: right,
                off: right_off,
                vectors,
                offset: offset + middle,
            },
            RangePolicy {
                options,
                execution: context,
            },
            solve,
            used,
            gemm,
        )?;
        if rho == S::Real::ZERO {
            return Ok(());
        }
        let geometry = MergeGeometry {
            offset,
            middle,
            coupling,
        };
        let count = self.prepare_merge(diagonal, vectors, geometry)?;
        for (target, &index) in self.active[..count].iter().enumerate() {
            self.secular_poles[target] = self.poles[index];
            self.secular_weights[target] = self.weights[index];
        }
        for root in 0..count {
            let (origin, delta) = secular_root(
                &self.secular_poles[..count],
                &self.secular_weights[..count],
                root,
                rho,
                budget,
                used,
                context,
                solve,
            )?;
            self.origins[root] = origin;
            self.deltas[root] = delta;
        }
        self.reconstruct_merge(
            diagonal,
            vectors,
            geometry,
            count,
            options.execution_options().execution_policy(),
            context,
            gemm,
        )
    }

    /// Sort the children's eigenpairs by pole and deflate them.
    ///
    /// Deflation rotations act in place on the local rows of `vectors`; the
    /// returned count of active poles is followed by their sorted indices in
    /// `active`, with `permutation` mapping sorted indices to local rows.
    fn prepare_merge(
        &mut self,
        diagonal: &[S::Real],
        vectors: &mut impl MatrixStorage<S>,
        geometry: MergeGeometry<S::Real>,
    ) -> Result<usize, SciOxideError> {
        let MergeGeometry {
            offset,
            middle,
            coupling,
        } = geometry;
        let n = diagonal.len();
        let stride = vectors.cols();
        let rho = coupling.abs();
        let permutation = &mut self.permutation[..n];
        permutation
            .iter_mut()
            .enumerate()
            .for_each(|(index, row)| *row = index);
        // The index tie-break makes the in-place sort stable, which fixes the
        // deflation and secular-root order.
        permutation.sort_unstable_by(|&a, &b| {
            diagonal[a]
                .partial_cmp(&diagonal[b])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });
        let sign = if coupling < S::Real::ZERO {
            -S::Real::ONE
        } else {
            S::Real::ONE
        };
        for (sorted, &row) in permutation.iter().enumerate() {
            self.poles[sorted] = diagonal[row];
            self.support[sorted] = if row < middle { 0 } else { 2 };
            self.weights[sorted] = if row < middle {
                vectors
                    .get_element(offset + row, offset + middle - 1)
                    .real_component()
            } else {
                vectors
                    .get_element(offset + row, offset + middle)
                    .real_component()
                    * sign
            };
        }
        let norm = self.poles[..n]
            .iter()
            .fold(rho, |scale, value| scale.max(value.abs()));
        let threshold = norm * S::Real::EPSILON * S::Real::TWO.scalbn(2);
        let permutation = &self.permutation[..n];
        crate::secular::deflate(
            &self.poles[..n],
            &mut self.weights[..n],
            &mut self.active[..n],
            rho,
            threshold,
            |previous, index, c, s| {
                if self.support[previous] != self.support[index] {
                    self.support[previous] = 1;
                    self.support[index] = 1;
                }
                let first = offset + permutation[previous];
                let second = offset + permutation[index];
                // The sorted pole permutation need not preserve row order.
                let rotation = crate::rotation::Rotation {
                    c,
                    s: S::from_real(if first < second { s } else { -s }),
                };
                rotation.rotate_rows(
                    vectors.as_mut_slice(),
                    stride,
                    first.min(second),
                    first.max(second),
                    offset..offset + n,
                )?;
                Ok(())
            },
        )
    }

    /// Reconstruct Löwner-formula weights and multiply only the active blocks.
    ///
    /// Active rows are packed as `[left, dense, right]` by child support
    /// maintained through deflation rotations, so two products of the secular
    /// eigenvectors with the nonzero halves form the merged vectors. Deflated
    /// rows are copied unchanged after the active ones.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    fn reconstruct_merge(
        &mut self,
        diagonal: &mut [S::Real],
        vectors: &mut impl MatrixStorage<S>,
        geometry: MergeGeometry<S::Real>,
        count: usize,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
        gemm: &mut LinalgWorkspace<'_, S>,
    ) -> Result<(), SciOxideError> {
        let MergeGeometry {
            offset,
            middle,
            coupling,
        } = geometry;
        let n = diagonal.len();
        let stride = vectors.cols();
        let rho = coupling.abs();
        // Distances to the poles come from the accurate root shifts, never
        // from differences of rounded eigenvalues.
        for &index in &self.active[..count] {
            let pole = self.poles[index];
            let magnitude = crate::secular::weight(
                (0..count).map(|root| (self.origins[root] - pole) + self.deltas[root]),
                core::iter::once(rho).chain(
                    self.active[..count]
                        .iter()
                        .filter(|&&other| other != index)
                        .map(|&other| self.poles[other] - pole),
                ),
            )?;
            self.weights[index] = magnitude.copysign(self.weights[index]);
        }
        let row = |index: usize| {
            let start = (offset + self.permutation[index]) * stride + offset;
            &vectors.as_slice()[start..start + n]
        };
        let support = |index: usize| self.support[index];
        let mut cursors = self.active[..count]
            .iter()
            .fold([0; 3], |mut counts, &index| {
                counts[support(index)] += 1;
                counts
            });
        let [left, dense, _] = cursors;
        cursors = [0, left, left + dense];
        for &index in &self.active[..count] {
            let group = support(index);
            self.order[cursors[group]] = index;
            cursors[group] += 1;
        }

        self.basis.try_resize_flat(n, n)?;
        let mut deflated = count;
        let mut active = self.active[..count].iter().peekable();
        for sorted in 0..n {
            if active.next_if_eq(&&sorted).is_none() {
                diagonal[deflated] = self.poles[sorted];
                self.basis.as_mut_slice()[deflated * n..(deflated + 1) * n]
                    .copy_from_slice(row(sorted));
                deflated += 1;
            }
        }
        for (target, &index) in self.order[..count].iter().enumerate() {
            self.basis.as_mut_slice()[target * n..(target + 1) * n].copy_from_slice(row(index));
        }

        self.coefficients.try_resize_flat(count, count)?;
        for (root, eigenvalue) in diagonal.iter_mut().enumerate().take(count) {
            let coefficients = &mut self.coefficients.as_mut_slice()[root * count..][..count];
            for (slot, &index) in coefficients.iter_mut().zip(&self.order[..count]) {
                let distance = (self.poles[index] - self.origins[root]) - self.deltas[root];
                let component = self.weights[index] / distance;
                if !component.is_finite() {
                    return Err(breakdown());
                }
                *slot = S::from_real(component);
            }
            let scale = coefficients
                .iter()
                .fold(S::Real::ZERO, |scale, value| scale.max(value.abs()));
            if scale == S::Real::ZERO {
                return Err(breakdown());
            }
            let norm = coefficients
                .iter()
                .fold(S::Real::ZERO, |sum, value| {
                    let value = value.real_component() / scale;
                    sum + value * value
                })
                .sqrt();
            coefficients.iter_mut().for_each(|value| {
                *value = S::from_real(value.real_component() / scale / norm);
            });
            *eigenvalue = self.origins[root] + self.deltas[root];
        }

        let local = &mut vectors.as_mut_slice()[offset * stride + offset..];
        for target in count..n {
            local[target * stride..target * stride + n]
                .copy_from_slice(&self.basis.as_slice()[target * n..(target + 1) * n]);
        }
        if count == 0 {
            return Ok(());
        }
        let coefficients = self.coefficients.as_slice();
        let basis = self.basis.as_slice();
        // Left columns see left and dense rows; right columns dense and right rows.
        for (columns, rows) in [(0..middle, 0..left + dense), (middle..n, left..count)] {
            if rows.is_empty() {
                for target in 0..count {
                    local[target * stride..][columns.clone()].fill(S::ZERO);
                }
                continue;
            }
            let output = MatrixViewMut::from_slice_with_strides(
                count,
                columns.len(),
                &mut local[columns.start..],
                stride,
                1,
            )?;
            gemm_workspace_with_context(
                WorkspaceGemmProblem {
                    left: FiniteMatrixView::from_workspace(MatrixView::from_slice_with_strides(
                        count,
                        rows.len(),
                        &coefficients[rows.start..],
                        count,
                        1,
                    )?),
                    right: FiniteMatrixView::from_workspace(MatrixView::from_slice_with_strides(
                        rows.len(),
                        columns.len(),
                        &basis[rows.start * n + columns.start..],
                        n,
                        1,
                    )?),
                    output,
                    config: GemmConfig::ordinary(),
                },
                policy,
                gemm,
                context,
            )?;
        }
        Ok(())
    }
}

fn iteration_limit(used: usize) -> SciOxideError {
    SciOxideError::convergence_failure(
        ConvergenceReason::IterationLimit,
        ConvergenceDiagnostics::new()
            .with_counts((scioxide_core::EvaluationCounts::from_iterations(used)).into()),
        "Hermitian divide-and-conquer exhausted its shared iteration budget",
    )
}

/// Reports a QL budget failure, which spent the `remaining` iterations left
/// after `used`, with the count of the whole solve.
fn shared_limit(error: SciOxideError, used: usize, remaining: usize) -> SciOxideError {
    if matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::IterationLimit,
            ..
        }
    ) {
        iteration_limit(used + remaining)
    } else {
        error
    }
}
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn secular_root<R: RealScalar>(
    poles: &[R],
    weights: &[R],
    root: usize,
    rho: R,
    budget: usize,
    used: &mut usize,
    context: KernelContext<'_>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(R, R), SciOxideError> {
    crate::secular::rank_one_root(poles, weights, root, rho, || {
        if *used >= budget {
            return Err(iteration_limit(*used));
        }
        context.execution().check()?;
        crate::solve_support::admit_iteration(
            Some(solve.execution()),
            Some(&mut *solve),
            "solve iteration budget exhausted",
        )?;
        *used += 1;
        Ok(())
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Matrix;
    use alloc::vec::Vec;

    /// Tridiagonal fixtures: glued Wilkinson blocks, repeated decoupled blocks
    /// (cross-child deflation), clusters, graded entries, a mirror-symmetric
    /// Wilkinson matrix and a generic pseudo-random matrix.
    fn fixture<R: RealScalar>(n: usize, kind: usize) -> (Vec<R>, Vec<R>) {
        let real = |x: f64| R::try_from_f64_lossy(x).unwrap();
        let mut state = 0x2545_f491_4f6c_dd1d_u64 + kind as u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        };
        let diagonal: Vec<R> = (0..n)
            .map(|i| match kind {
                0 => real(((i % 21) as f64 - 10.0).abs()),
                1 => R::ONE,
                2 => R::ONE + real((i % 3) as f64 * 32.0) * R::EPSILON,
                3 => R::ONE.scalbn(-(i as i32)),
                4 => real((i as f64 - (n / 2) as f64).abs()),
                _ => real(random()),
            })
            .collect();
        let off = (0..n)
            .map(|i| match kind {
                _ if i + 1 == n => R::ZERO,
                0 if (i + 1) % 21 == 0 => real(1.0e-14),
                1 if (i + 1) % 7 == 0 => R::ZERO,
                2 => real(64.0) * R::EPSILON,
                3 => diagonal[i] * real(-0.125),
                0 | 1 | 4 => R::ONE,
                _ => real(random()),
            })
            .collect();
        (diagonal, off)
    }

    /// Residual, orthogonality and sorted eigenvalues of transposed vectors.
    fn invariants<R: RealScalar>(
        diagonal: &[R],
        off: &[R],
        values: &[R],
        vectors: &Matrix<R>,
    ) -> (R, R, Vec<R>) {
        let n = diagonal.len();
        let (mut residual, mut orthogonality) = (R::ZERO, R::ZERO);
        for (root, &value) in values.iter().enumerate() {
            let v = |i: usize| vectors.get_element(root, i);
            for i in 0..n {
                let below = if i > 0 {
                    off[i - 1] * v(i - 1)
                } else {
                    R::ZERO
                };
                let above = if i + 1 < n {
                    off[i] * v(i + 1)
                } else {
                    R::ZERO
                };
                residual = residual.max((diagonal[i] * v(i) + below + above - value * v(i)).abs());
            }
            for other in 0..n {
                let dot = (0..n).fold(R::ZERO, |sum, i| sum + v(i) * vectors.get_element(other, i));
                let identity = if root == other { R::ONE } else { R::ZERO };
                orthogonality = orthogonality.max((dot - identity).abs());
            }
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (residual, orthogonality, sorted)
    }

    fn compare<R: RealScalar + NumericScalar<Real = R>>(n: usize, kind: usize) {
        let (diagonal, off) = fixture::<R>(n, kind);
        let options = HermitianEigenOptions::default();
        let norm = diagonal
            .iter()
            .chain(&off)
            .fold(R::ZERO, |scale, value| scale.max(value.abs()));
        let bound = R::try_from_f64_lossy(100.0 * n as f64).unwrap() * R::EPSILON;

        let (mut ql, mut ql_off) = (diagonal.clone(), off.clone());
        let mut ql_vectors = Matrix::<R>::identity(n).unwrap();
        super::super::tests::with_solve(|solve| {
            tridiagonal_ql(
                &mut ql,
                &mut ql_off,
                Some(QlVectors::new(&mut ql_vectors, &mut alloc::vec![R::ZERO; 2 * n], n).unwrap()),
                &options,
                ExecutionContext::sequential(),
                solve,
            )
        })
        .unwrap();

        let (mut dc, mut dc_off) = (diagonal.clone(), off.clone());
        let mut dc_vectors = Matrix::<R>::identity(n).unwrap();
        let [scalars, reals, indices] = Workspace::<R>::lengths(n).unwrap();
        let mut scalars = alloc::vec![R::ZERO; scalars];
        let mut reals = alloc::vec![R::ZERO; reals];
        let mut indices = alloc::vec![0; indices];
        let mut scratch = Workspace::new(&mut scalars, &mut reals, &mut indices, n).unwrap();
        super::super::tests::with_solve(|solve| {
            scratch.solve(
                &mut dc,
                &mut dc_off,
                &mut dc_vectors,
                &options,
                KernelContext::new(crate::kernel::tests::test_context()),
                solve,
                &mut LinalgWorkspace::workspace(&crate::DenseWorkspaceRequest::product(
                    n,
                    n,
                    n,
                    &crate::LinalgExecutionOptions::new(),
                ))
                .unwrap(),
            )
        })
        .unwrap();

        let (ql_residual, ql_orthogonality, ql_sorted) =
            invariants(&diagonal, &off, &ql, &ql_vectors);
        let (residual, orthogonality, sorted) = invariants(&diagonal, &off, &dc, &dc_vectors);
        let spectrum = ql_sorted
            .iter()
            .zip(&sorted)
            .fold(R::ZERO, |worst, (a, b)| worst.max((*a - *b).abs()));
        let case = alloc::format!("n={n} kind={kind}");
        assert!(ql_residual <= bound * norm, "{case} QL residual");
        assert!(ql_orthogonality <= bound, "{case} QL orthogonality");
        assert!(residual <= bound * norm, "{case} residual {residual:?}");
        assert!(
            orthogonality <= bound,
            "{case} orthogonality {orthogonality:?}"
        );
        assert!(spectrum <= bound * norm, "{case} spectrum {spectrum:?}");
    }

    #[test]
    fn divide_conquer_agrees_with_ql_on_deflated_clustered_and_graded_tridiagonals() {
        for n in [65, 129] {
            for kind in 0..6 {
                compare::<f64>(n, kind);
                compare::<f32>(n, kind);
            }
        }
        compare::<f64>(257, 4);
    }

    #[test]
    fn clustered_and_graded_spectra_cross_the_vector_crossover() {
        for n in [VECTOR_ORDER - 1, VECTOR_ORDER, VECTOR_ORDER + 1] {
            for kind in [2, 3] {
                compare::<f32>(n, kind);
                compare::<f64>(n, kind);
                let (diagonal, off) = fixture::<f64>(n, kind);
                let matrix = Matrix::from_vec(
                    n,
                    n,
                    (0..n * n)
                        .map(|i| {
                            let (row, col) = (i / n, i % n);
                            match row.abs_diff(col) {
                                0 => scioxide_core::Complex::new(diagonal[row], 0.0),
                                1 => scioxide_core::Complex::new(
                                    0.0,
                                    if row < col { off[row] } else { -off[col] },
                                ),
                                _ => scioxide_core::Complex::new(0.0, 0.0),
                            }
                        })
                        .collect(),
                )
                .unwrap();
                for ordering in [EigenvalueOrder::Ascending, EigenvalueOrder::Descending] {
                    let options = HermitianEigenOptions::default().with_ordering(ordering);
                    let real = HermitianEigenDecomposition::tridiagonal_with_options(
                        &diagonal,
                        &off[..n - 1],
                        &options,
                    )
                    .unwrap();
                    let complex = HermitianEigenDecomposition::decompose_with_options(
                        &matrix,
                        crate::HermitianInput::Exact,
                        &options,
                    )
                    .unwrap();
                    let bound = 100.0 * n as f64 * f64::EPSILON;
                    for (a, b) in real.eigenvalues().iter().zip(complex.eigenvalues()) {
                        assert!((a - b).abs() <= bound);
                    }
                    assert!(real.eigenvalues().windows(2).all(|pair| match ordering {
                        EigenvalueOrder::Ascending => pair[0] <= pair[1],
                        EigenvalueOrder::Descending => pair[0] >= pair[1],
                    }));
                    let q = complex.eigenvectors();
                    for col in 0..n {
                        for row in 0..n {
                            let aq = (row.saturating_sub(1)..=(row + 1).min(n - 1))
                                .map(|k| matrix.get_element(row, k) * q.get_element(k, col))
                                .sum::<scioxide_core::Complex<f64>>();
                            assert!(
                                (aq - q.get_element(row, col) * complex.eigenvalues()[col]).norm()
                                    <= bound
                            );
                            let dot = (0..n)
                                .map(|k| q.get_element(k, row).conj() * q.get_element(k, col))
                                .sum::<scioxide_core::Complex<f64>>();
                            assert!(
                                (dot - scioxide_core::Complex::new(f64::from(row == col), 0.0))
                                    .norm()
                                    <= bound
                            );
                        }
                    }
                }
            }
        }
    }
}
#[derive(Clone, Copy)]
struct MergeGeometry<R> {
    offset: usize,
    middle: usize,
    coupling: R,
}
struct TridiagonalRange<'a, S: NumericScalar, M> {
    diagonal: &'a mut [S::Real],
    off: &'a mut [S::Real],
    vectors: &'a mut M,
    offset: usize,
}
#[derive(Clone, Copy)]
struct RangePolicy<'a, R: RealScalar> {
    options: &'a HermitianEigenOptions<R>,
    execution: KernelContext<'a>,
}
