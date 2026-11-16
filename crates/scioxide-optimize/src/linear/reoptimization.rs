//! Explicit basis snapshots and appended-row remapping.
use super::*;

use scioxide_core::backend::{try_copy_slice as copy, try_with_capacity};

impl<S: RealScalar + NumericScalar<Real = S>> SimplexWarmStart<S> {
    /// Returns whether the attempt that recorded this snapshot started from a stored basis.
    pub const fn reused_basis(&self) -> bool {
        self.reused
    }

    /// Returns the bytes held by the snapshot, excluding allocator overhead.
    pub fn storage_bytes(&self) -> usize {
        (self.indices.capacity() + self.rows.capacity()) * core::mem::size_of::<usize>()
            + self.upper.capacity() * core::mem::size_of::<bool>()
            + self.structure.as_ref().map_or(0, |s| {
                (s.values.capacity() + s.signs.capacity()) * core::mem::size_of::<S>()
                    + (s.columns.capacity() + s.offsets.capacity()) * core::mem::size_of::<usize>()
                    + (s.row_exponents.capacity() + s.column_exponents.capacity())
                        * core::mem::size_of::<i32>()
                    + s.bounds
                        .as_ref()
                        .map_or(0, |b| core::mem::size_of_val(b.as_slice()))
            })
    }

    /// Copies the snapshot, including its reuse report.
    ///
    /// A snapshot stores the basis, not its numerical factors.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the copy cannot be allocated.
    pub fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            structure: self
                .structure
                .as_ref()
                .map(|s| {
                    Ok::<_, SciOxideError>(WarmStructure {
                        values: copy(&s.values, scioxide_core::quantity!("warm_values"))?,
                        columns: copy(&s.columns, scioxide_core::quantity!("warm_columns"))?,
                        offsets: copy(&s.offsets, scioxide_core::quantity!("warm_offsets"))?,
                        signs: copy(&s.signs, scioxide_core::quantity!("warm_signs"))?,
                        bounds: s
                            .bounds
                            .as_ref()
                            .map(|b| {
                                scioxide_core::Bounds::new(copy(
                                    b.as_slice(),
                                    scioxide_core::quantity!("warm_bounds"),
                                )?)
                            })
                            .transpose()?,
                        dimension: s.dimension,
                        input_rows: s.input_rows,
                        transformed_variables: s.transformed_variables,
                        row_exponents: copy(
                            &s.row_exponents,
                            scioxide_core::quantity!("warm_row_scales"),
                        )?,
                        column_exponents: copy(
                            &s.column_exponents,
                            scioxide_core::quantity!("warm_column_scales"),
                        )?,
                    })
                })
                .transpose()?,
            record: self.record,
            reused: self.reused,
            indices: copy(&self.indices, scioxide_core::quantity!("warm_basis"))?,
            rows: copy(&self.rows, scioxide_core::quantity!("warm_rows"))?,
            upper: copy(&self.upper, scioxide_core::quantity!("warm_bound_states"))?,
        })
    }

    // Native bounds keep existing column and row labels stable when rows are appended.
    // Compare coefficients in their original, unscaled frame.
    pub(super) fn extend(
        &mut self,
        p: &LpProblem<S>,
        f: &StandardForm<S>,
    ) -> Result<bool, SciOxideError> {
        let Some(s) = &self.structure else {
            return Ok(false);
        };
        let new_rows = p.constraints().map_or(0, |c| c.len());
        if s.dimension != p.dimension()
            || new_rows < s.input_rows
            || s.transformed_variables != f.transformed_variables
            || s.bounds.as_ref().map(|b| b.as_slice()) != p.bounds().map(|b| b.as_slice())
            || f.columns < s.column_exponents.len()
        {
            self.clear();
            return Ok(false);
        }
        let row_shift = new_rows - s.input_rows;
        let col_shift = f.columns - s.column_exponents.len();
        if row_shift != col_shift {
            self.clear();
            return Ok(false);
        }
        let old_columns = s.column_exponents.len();
        for old_row in 0..s.signs.len() {
            let range = s.offsets[old_row]..s.offsets[old_row + 1];
            if range.len() != f.row_entries(old_row).count() {
                self.clear();
                return Ok(false);
            }
            for k in range {
                let old_col = s.columns[k];
                let old = s.values[k]
                    .scalbn(-s.row_exponents[old_row] - s.column_exponents[old_col])
                    * s.signs[old_row];
                let new = f
                    .coefficient(old_row, old_col)
                    .scalbn(-f.row_exponents[old_row] - f.column_exponents[old_col])
                    * f.row_signs[old_row];
                if !old.is_finite()
                    || old == <S as scioxide_core::AlgebraicScalar>::ZERO
                    || old != new
                {
                    self.clear();
                    return Ok(false);
                }
            }
        }
        let mut indices = try_with_capacity(
            self.indices.len() + row_shift,
            scioxide_core::quantity!("extended_basis"),
        )?;
        indices.extend_from_slice(&self.indices);
        let mut rows = try_with_capacity(
            self.rows.len() + row_shift,
            scioxide_core::quantity!("extended_basis_rows"),
        )?;
        rows.extend_from_slice(&self.rows);
        for row in s.input_rows..new_rows {
            let Some(col) = f
                .row_entries(row)
                .map(|(c, _)| c)
                .find(|&c| c >= old_columns && c < f.columns)
            else {
                self.clear();
                return Ok(false);
            };
            indices.push(col);
            rows.push(row);
        }
        self.indices = indices;
        self.rows = rows;
        crate::storage::resize(&mut self.upper, f.columns, false)?;
        self.structure = None;
        self.reused = true;
        Ok(true)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{LinearConstraintSense as Sense, LinearConstraints};
    use scioxide_core::Bounds;
    use scioxide_linalg::Matrix;

    fn problem(rhs: &[f64], senses: Vec<Sense>) -> LpProblem<f64> {
        LpProblem::new(vec![-1.0])
            .unwrap()
            .with_bounds(Bounds::new(vec![(Some(0.0), Some(10.0))]).unwrap())
            .unwrap()
            .with_constraints(
                LinearConstraints::new(
                    Matrix::from_vec(rhs.len(), 1, vec![1.0; rhs.len()]).unwrap(),
                    rhs.to_vec(),
                    senses,
                )
                .unwrap(),
            )
            .unwrap()
    }

    /// Resume from `w`, then replace `w` with the attempt's recorded basis.
    fn reopt(
        p: &LpProblem<f64>,
        w: &mut SimplexWarmStart<f64>,
        limit: Budget,
    ) -> ProgrammingResult<f64> {
        let solver = SimplexSolver::default().with_max_iterations(limit);
        let mut workspace = solver.workspace(p).unwrap();
        let mut output = ProgrammingResult::empty();
        solver
            .resume_into(
                p,
                w,
                &mut output,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        *w = workspace.warm_start().unwrap();
        output
    }

    /// Cold solve recording the final basis.
    fn cold(p: &LpProblem<f64>) -> SimplexWarmStart<f64> {
        let solver = SimplexSolver::default();
        let mut workspace = solver.workspace(p).unwrap();
        let mut output = ProgrammingResult::empty();
        solver
            .solve_into(
                p,
                &mut output,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        workspace.warm_start().unwrap()
    }

    #[test]
    fn appended_branch_reuses_basis_and_matches_cold() {
        let root = problem(&[2.5], vec![Sense::LessEqual]);
        let child = problem(&[2.5, 2.0], vec![Sense::LessEqual; 2]);
        let mut warm = cold(&root);
        let result = reopt(&child, &mut warm, Budget::ONE.saturating_add(99));
        let cold = SimplexSolver::default().solve(&child).unwrap();
        assert!(warm.reused_basis());
        assert!(result.is_success(), "{}", result.message());
        assert_eq!(result.x(), cold.x());
        assert_eq!(result.fun(), -2.0);
        assert!(result.verification_report().unwrap().is_accepted());
        let copied = warm.try_clone().unwrap();
        assert_eq!(copied.basis_indices(), warm.basis_indices());
        assert!(copied.storage_bytes() > 0);
    }

    #[test]
    fn dual_infeasibility_requires_verified_witness() {
        let root = problem(&[2.5], vec![Sense::LessEqual]);
        let child = problem(&[2.5, 3.0], vec![Sense::LessEqual, Sense::GreaterEqual]);
        let mut warm = cold(&root);
        let result = reopt(&child, &mut warm, Budget::ONE.saturating_add(99));
        assert!(warm.reused_basis());
        assert_eq!(result.termination(), OptimizeTermination::Infeasible);
        assert!(result.verification_report().unwrap().is_accepted());
    }

    #[test]
    fn rhs_reuse_and_incompatible_matrix_restart_are_explicit() {
        let root = problem(&[2.5], vec![Sense::LessEqual]);
        let mut warm = cold(&root);
        let changed = problem(&[1.5], vec![Sense::LessEqual]);
        let r = reopt(&changed, &mut warm, Budget::ONE.saturating_add(99));
        assert!(r.is_success());
        assert!(warm.reused_basis());
        assert_eq!(r.x(), &[1.5]);
        let changed = LpProblem::new(vec![-1.0])
            .unwrap()
            .with_bounds(Bounds::new(vec![(Some(0.0), Some(10.0))]).unwrap())
            .unwrap()
            .with_constraints(
                LinearConstraints::new(
                    Matrix::from_vec(1, 1, vec![2.0]).unwrap(),
                    vec![1.5],
                    vec![Sense::LessEqual],
                )
                .unwrap(),
            )
            .unwrap();
        let r = reopt(&changed, &mut warm, Budget::ONE.saturating_add(99));
        assert!(r.is_success());
        assert!(!warm.reused_basis());
        assert_eq!(r.x(), &[0.75]);
    }
}
