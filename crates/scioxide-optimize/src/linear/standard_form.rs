//! Shared checked conversion of validated LPs to nonnegative standard form.

#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, try_filled, try_reserve, try_with_capacity};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

use crate::problem::{LinearConstraintSense, LpProblem};

#[derive(Debug, Clone, Copy)]
enum VariableMap<S> {
    Affine {
        transformed: usize,
        offset: S,
        scale: S,
    },
    Free {
        positive: usize,
        negative: usize,
    },
    Fixed(S),
}

#[derive(Debug)]
pub(crate) struct StandardForm<S> {
    pub(crate) objective: Vec<S>,
    pub(crate) coefficients: Vec<S>,
    pub(crate) column_indices: Vec<usize>,
    pub(crate) row_offsets: Vec<usize>,
    pub(crate) rhs: Vec<S>,
    pub(crate) upper_bounds: Vec<Option<S>>,
    column_offsets: Vec<usize>,
    column_rows: Vec<usize>,
    column_positions: Vec<usize>,
    pub(crate) row_signs: Vec<S>,
    pub(crate) row_exponents: Vec<i32>,
    pub(crate) column_exponents: Vec<i32>,
    pub(crate) cost_exponent: i32,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) transformed_variables: usize,
    variable_map: Vec<VariableMap<S>>,
    upper_rows: Vec<(usize, S)>,
    original_row: Vec<S>,
}

impl<S> StandardForm<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    // Stable original coordinates: bounds are kept by the retained solver, so
    // fixing a variable never removes its column. Only inequality slacks are added.
    pub(crate) fn native(problem: &LpProblem<S>) -> Result<Self, SciOxideError> {
        let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        let n = problem.dimension();
        let m = problem.constraints().map_or(0, |c| c.len());
        let inequalities = problem.constraints().map_or(0, |c| {
            c.senses()
                .iter()
                .filter(|s| **s != LinearConstraintSense::Equal)
                .count()
        });
        let mut form = Self::new();
        form.rows = m;
        form.columns = checked_add(
            n,
            inequalities,
            scioxide_core::quantity!("native_lp_columns"),
        )?;
        form.transformed_variables = n;
        form.objective = try_filled(form.columns, z, scioxide_core::quantity!("native_lp_cost"))?;
        form.objective[..n].copy_from_slice(problem.objective());
        form.upper_bounds = try_filled(
            form.columns,
            None,
            scioxide_core::quantity!("native_lp_bounds"),
        )?;
        form.rhs = try_filled(m, z, scioxide_core::quantity!("native_lp_rhs"))?;
        form.row_signs = try_filled(m, one, scioxide_core::quantity!("native_lp_signs"))?;
        form.row_exponents = try_filled(m, 0, scioxide_core::quantity!("native_lp_row_scales"))?;
        form.column_exponents = try_filled(
            form.columns,
            0,
            scioxide_core::quantity!("native_lp_col_scales"),
        )?;
        try_reserve(
            &mut form.variable_map,
            n,
            scioxide_core::quantity!("native_lp_map"),
        )?;
        form.variable_map
            .extend((0..n).map(|transformed| VariableMap::Affine {
                transformed,
                offset: z,
                scale: one,
            }));
        let stored = problem.constraints().map_or(0, |c| match c.coefficients() {
            crate::CoefficientMatrix::Dense(a) => a.as_slice().len(),
            crate::CoefficientMatrix::Sparse(a) => a.nnz(),
        });
        let entries = checked_add(
            stored,
            inequalities,
            scioxide_core::quantity!("native_lp_entries"),
        )?;
        try_reserve(
            &mut form.coefficients,
            entries,
            scioxide_core::quantity!("native_lp_values"),
        )?;
        try_reserve(
            &mut form.column_indices,
            entries,
            scioxide_core::quantity!("native_lp_indices"),
        )?;
        try_reserve(
            &mut form.row_offsets,
            checked_add(m, 1, scioxide_core::quantity!("native_lp_offsets"))?,
            scioxide_core::quantity!("native_lp_offsets"),
        )?;
        form.row_offsets.push(0);
        let mut slack = n;
        if let Some(c) = problem.constraints() {
            for row in 0..m {
                // Keep >= rows negated, matching basis_dual's conventions.
                let sign = if c.senses()[row] == LinearConstraintSense::GreaterEqual {
                    -one
                } else {
                    one
                };
                for (column, value) in c.coefficients().row_entries(row)? {
                    if value != z {
                        form.coefficients.push(value * sign);
                        form.column_indices.push(column);
                    }
                }
                if c.senses()[row] != LinearConstraintSense::Equal {
                    form.coefficients.push(one);
                    form.column_indices.push(slack);
                    slack += 1;
                }
                form.row_offsets.push(form.coefficients.len());
                form.rhs[row] = c.rhs()[row] * sign;
            }
        }
        form.index_columns()?;
        form.equilibrate()?;
        Ok(form)
    }

    pub(crate) fn original_column(&self, transformed: usize) -> Option<usize> {
        if transformed >= self.transformed_variables {
            return Some(self.variable_map.len() + transformed - self.transformed_variables);
        }
        self.variable_map.iter().position(|map| match *map {
            VariableMap::Affine {
                transformed: column,
                ..
            } => column == transformed,
            VariableMap::Free { positive, negative } => {
                positive == transformed || negative == transformed
            }
            VariableMap::Fixed(_) => false,
        })
    }
    pub(crate) const fn new() -> Self {
        Self {
            objective: Vec::new(),
            coefficients: Vec::new(),
            column_indices: Vec::new(),
            row_offsets: Vec::new(),
            rhs: Vec::new(),
            upper_bounds: Vec::new(),
            column_offsets: Vec::new(),
            column_rows: Vec::new(),
            column_positions: Vec::new(),
            row_signs: Vec::new(),
            row_exponents: Vec::new(),
            column_exponents: Vec::new(),
            cost_exponent: 0,
            rows: 0,
            columns: 0,
            transformed_variables: 0,
            variable_map: Vec::new(),
            upper_rows: Vec::new(),
            original_row: Vec::new(),
        }
    }

    pub(crate) fn coefficient(&self, row: usize, column: usize) -> S {
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        self.column_indices[range.clone()]
            .binary_search(&column)
            .map_or(<S as scioxide_core::AlgebraicScalar>::ZERO, |i| {
                self.coefficients[range.start + i]
            })
    }
    pub(crate) fn column_entries(&self, column: usize) -> impl Iterator<Item = (usize, S)> + '_ {
        let range = self.column_offsets[column]..self.column_offsets[column + 1];
        self.column_rows[range.clone()].iter().copied().zip(
            self.column_positions[range]
                .iter()
                .map(|&k| self.coefficients[k]),
        )
    }

    fn index_columns(&mut self) -> Result<(), SciOxideError> {
        resize(
            &mut self.column_offsets,
            checked_add(
                self.columns,
                1,
                scioxide_core::quantity!("lp_column_offsets"),
            )?,
            0,
        )?;
        self.column_offsets.fill(0);
        for &column in &self.column_indices {
            self.column_offsets[column + 1] += 1;
        }
        for column in 0..self.columns {
            self.column_offsets[column + 1] += self.column_offsets[column];
        }
        resize(&mut self.column_rows, self.coefficients.len(), 0)?;
        resize(&mut self.column_positions, self.coefficients.len(), 0)?;
        let mut next = scioxide_core::backend::try_copy_slice(
            &self.column_offsets,
            scioxide_core::quantity!("lp_column_cursors"),
        )?;
        for row in 0..self.rows {
            for k in self.row_offsets[row]..self.row_offsets[row + 1] {
                let column = self.column_indices[k];
                let target = next[column];
                self.column_rows[target] = row;
                self.column_positions[target] = k;
                next[column] += 1;
            }
        }
        Ok(())
    }

    pub(crate) fn row_entries(&self, row: usize) -> impl Iterator<Item = (usize, S)> + '_ {
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        self.column_indices[range.clone()]
            .iter()
            .copied()
            .zip(self.coefficients[range].iter().copied())
    }
    pub(crate) fn row_dot(&self, row: usize, x: &[S]) -> Result<S, SciOxideError> {
        let mut sum = scioxide_core::ProductSum::new();
        for (column, value) in self.row_entries(row) {
            sum.add_product(value, x[column])?;
        }
        sum.finish()
    }

    pub(crate) fn equilibrate(&mut self) -> Result<(), SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let exponent = |v: S| {
            let scioxide_core::FloatDecomposition::Finite {
                significand: bits,
                exponent: e,
                ..
            } = v.decompose()
            else {
                unreachable!("linear model coefficients were checked finite");
            };
            if bits == 0 {
                0
            } else {
                i32::from(e) + 63 - i32::try_from(bits.leading_zeros()).unwrap_or(0)
            }
        };
        // Scale original structural data before implicit slack magnitudes.
        // Keep exponents separate so a large row scale and its compensating
        // slack-column scale never create an overflowing intermediate.
        for row in 0..self.rows {
            let scale = self
                .row_entries(row)
                .filter(|(col, _)| *col < self.transformed_variables)
                .fold(NumericScalar::abs(self.rhs[row]), |a, (_, v)| {
                    a.max(NumericScalar::abs(v))
                });
            let e = if scale == zero { 0 } else { -exponent(scale) };
            self.row_exponents[row] = e;
            let rhs = self.rhs[row].scalbn(e);
            if self.rhs[row] != zero && rhs == zero {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "row scaling lost a nonzero right-hand side",
                ));
            }
            self.rhs[row] = rhs;
        }
        let mut norms = try_filled(
            self.columns,
            i32::MIN,
            scioxide_core::quantity!("lp_column_exponents"),
        )?;
        for row in 0..self.rows {
            for (col, value) in self.row_entries(row) {
                norms[col] =
                    norms[col].max(exponent(NumericScalar::abs(value)) + self.row_exponents[row]);
            }
        }
        for (e, &maximum) in self.column_exponents.iter_mut().zip(&norms) {
            *e = if maximum == i32::MIN { 0 } else { -maximum };
        }
        for row in 0..self.rows {
            for k in self.row_offsets[row]..self.row_offsets[row + 1] {
                let e = self.row_exponents[row] + self.column_exponents[self.column_indices[k]];
                let value = self.coefficients[k].scalbn(e);
                if value == zero || !value.is_finite() {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Underflow,
                        "combined scaling cannot preserve a coefficient",
                    ));
                }
                self.coefficients[k] = value;
            }
        }
        for (bound, &exponent) in self.upper_bounds.iter_mut().zip(&self.column_exponents) {
            if let Some(width) = bound {
                let scaled = width.scalbn(-exponent);
                if !scaled.is_finite() || (*width != zero && scaled == zero) {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::LossOfSignificance,
                        "column scaling cannot preserve a finite bound",
                    ));
                }
                *width = scaled;
            }
        }
        let cost = self
            .objective
            .iter()
            .zip(&self.column_exponents)
            .filter(|(v, _)| **v != zero)
            .map(|(&v, &e)| exponent(NumericScalar::abs(v)) + e)
            .max()
            .unwrap_or(0);
        self.cost_exponent = cost;
        for (value, &e) in self.objective.iter_mut().zip(&self.column_exponents) {
            let scaled = value.scalbn(e - cost);
            if *value != zero && scaled == zero {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "cost scaling lost a nonzero coefficient",
                ));
            }
            *value = scaled;
        }
        Ok(())
    }
    fn original_component(
        &self,
        mapping: VariableMap<S>,
        standard: &[S],
        point: bool,
    ) -> Result<S, SciOxideError> {
        use scioxide_core::Scaled;
        let value = |i: usize| {
            Scaled::from_value(standard[i])?.shifted(i64::from(self.column_exponents[i]))
        };
        match mapping {
            VariableMap::Affine {
                transformed,
                offset,
                scale,
            } => value(transformed)?
                .mul(Scaled::from_value(scale)?)?
                .add(Scaled::from_value(if point {
                    offset
                } else {
                    <S as scioxide_core::AlgebraicScalar>::ZERO
                })?)?
                .round(),
            VariableMap::Free { positive, negative } => {
                value(positive)?.sub(value(negative)?)?.round()
            }
            VariableMap::Fixed(v) => Ok(if point {
                v
            } else {
                <S as scioxide_core::AlgebraicScalar>::ZERO
            }),
        }
    }

    pub(crate) fn reconstruct_direction(&self, standard: &[S]) -> Result<Vec<S>, SciOxideError> {
        let mut direction = try_with_capacity(
            self.variable_map.len(),
            scioxide_core::quantity!("lp_recession_direction"),
        )?;
        for mapping in &self.variable_map {
            direction.push(self.original_component(*mapping, standard, false)?);
        }
        Ok(direction)
    }

    /// Encode an original-coordinate direction, completing implicit slack columns.
    pub(super) fn encode_direction(&self, direction: &[S]) -> Result<Vec<S>, SciOxideError> {
        if direction.len() != self.variable_map.len() {
            return Err(SciOxideError::dimension_mismatch(
                [self.variable_map.len()],
                [direction.len()],
            ));
        }
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let mut standard = try_filled(
            self.columns,
            zero,
            scioxide_core::quantity!("lp_encoded_ray"),
        )?;
        for (mapping, &value) in self.variable_map.iter().zip(direction) {
            match *mapping {
                VariableMap::Affine {
                    transformed, scale, ..
                } => {
                    standard[transformed] = scioxide_core::checked_float_quotient(value, scale)?
                        .scalbn(-self.column_exponents[transformed]);
                }
                VariableMap::Free { positive, negative } => {
                    standard[positive] = value.max(zero).scalbn(-self.column_exponents[positive]);
                    standard[negative] =
                        (-value).max(zero).scalbn(-self.column_exponents[negative]);
                }
                VariableMap::Fixed(_) => {}
            }
        }
        for column in self.transformed_variables..self.columns {
            if let Some((row, coefficient)) = self.column_entries(column).next() {
                standard[column] = scioxide_core::checked_float_quotient(
                    -self.row_dot(row, &standard)?,
                    coefficient,
                )?;
            }
        }
        crate::utils::validate_finite(&standard)?;
        Ok(standard)
    }

    pub(crate) fn reconstruct_solution(&self, standard: &[S]) -> Result<Vec<S>, SciOxideError> {
        let mut output = try_with_capacity(
            self.variable_map.len(),
            scioxide_core::quantity!("lp_original_solution"),
        )?;
        self.reconstruct_solution_into(standard, &mut output)?;
        Ok(output)
    }

    pub(crate) fn reconstruct_solution_into(
        &self,
        standard: &[S],
        output: &mut Vec<S>,
    ) -> Result<(), SciOxideError> {
        if standard.len() < self.transformed_variables {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.transformed_variables),
                scioxide_core::Dimensions::one(standard.len()),
            ));
        }
        resize(
            output,
            self.variable_map.len(),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        output
            .iter_mut()
            .zip(&self.variable_map)
            .try_for_each(|(target, mapping)| {
                *target = self.original_component(*mapping, standard, true)?;
                Ok(())
            })
    }
}

pub(crate) fn standardize<S>(problem: &LpProblem<S>) -> Result<StandardForm<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let mut output = StandardForm::new();
    standardize_into(problem, &mut output)?;
    Ok(output)
}

use crate::storage::resize;

pub(crate) fn standardize_into<S>(
    problem: &LpProblem<S>,
    output: &mut StandardForm<S>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    standardize_with_bounds(problem, output, false)
}

pub(crate) fn standardize_bounded<S>(
    problem: &LpProblem<S>,
) -> Result<StandardForm<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let mut output = StandardForm::new();
    standardize_with_bounds(problem, &mut output, true)?;
    Ok(output)
}

fn standardize_with_bounds<S>(
    problem: &LpProblem<S>,
    output: &mut StandardForm<S>,
    native_bounds: bool,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let original_dimension = problem.dimension();
    let mut transformed_variables = 0usize;
    let variable_map = &mut output.variable_map;
    variable_map.clear();
    try_reserve(
        variable_map,
        original_dimension,
        scioxide_core::quantity!("lp_variable_transformation_map"),
    )?;
    let upper_rows = &mut output.upper_rows;
    upper_rows.clear();
    try_reserve(
        upper_rows,
        original_dimension,
        scioxide_core::quantity!("lp_transformed_upper_bounds"),
    )?;

    for variable in 0..original_dimension {
        let (lower, upper) = problem
            .bounds()
            .map_or((Some(zero), None), |bounds| bounds.as_slice()[variable]);
        let mapping = match (lower, upper) {
            (Some(lower), Some(upper)) if lower == upper => VariableMap::Fixed(lower),
            (Some(lower), upper) => {
                let transformed = transformed_variables;
                transformed_variables = checked_add(
                    transformed_variables,
                    1,
                    scioxide_core::quantity!("lp_transformed_variable_count"),
                )?;
                if let Some(upper) = upper {
                    let width = upper - lower;
                    ensure_finite(width)?;
                    upper_rows.push((transformed, width));
                }
                VariableMap::Affine {
                    transformed,
                    offset: lower,
                    scale: one,
                }
            }
            (None, Some(upper)) => {
                let transformed = transformed_variables;
                transformed_variables = checked_add(
                    transformed_variables,
                    1,
                    scioxide_core::quantity!("lp_transformed_variable_count"),
                )?;
                VariableMap::Affine {
                    transformed,
                    offset: upper,
                    scale: -one,
                }
            }
            (None, None) => {
                let positive = transformed_variables;
                let negative = checked_add(
                    positive,
                    1,
                    scioxide_core::quantity!("lp_free_variable_split"),
                )?;
                transformed_variables = checked_add(
                    negative,
                    1,
                    scioxide_core::quantity!("lp_transformed_variable_count"),
                )?;
                VariableMap::Free { positive, negative }
            }
        };
        variable_map.push(mapping);
    }

    let problem_rows = problem
        .constraints()
        .map_or(0, |constraints| constraints.len());
    let rows = checked_add(
        problem_rows,
        if native_bounds { 0 } else { upper_rows.len() },
        scioxide_core::quantity!("lp_constraint_count"),
    )?;
    let problem_inequalities = problem.constraints().map_or(0, |constraints| {
        constraints
            .senses()
            .iter()
            .filter(|sense| !matches!(sense, LinearConstraintSense::Equal))
            .count()
    });
    let inequalities = checked_add(
        problem_inequalities,
        if native_bounds { 0 } else { upper_rows.len() },
        scioxide_core::quantity!("lp_inequality_count"),
    )?;
    let columns = checked_add(
        transformed_variables,
        inequalities,
        scioxide_core::quantity!("lp_standard_variable_count"),
    )?;

    resize(&mut output.upper_bounds, columns, None)?;
    output.upper_bounds.fill(None);
    if native_bounds {
        for &(column, width) in upper_rows.iter() {
            output.upper_bounds[column] = Some(width);
        }
    }

    let objective = &mut output.objective;
    resize(objective, columns, zero)?;
    objective.fill(zero);
    for (&coefficient, mapping) in problem.objective().iter().zip(variable_map.iter()) {
        match *mapping {
            VariableMap::Affine {
                transformed, scale, ..
            } => {
                objective[transformed] = coefficient * scale;
                ensure_finite(objective[transformed])?;
            }
            VariableMap::Free { positive, negative } => {
                objective[positive] = coefficient;
                objective[negative] = -coefficient;
            }
            VariableMap::Fixed(_) => {}
        }
    }

    output.coefficients.clear();
    output.column_indices.clear();
    output.row_offsets.clear();
    try_reserve(
        &mut output.row_offsets,
        checked_add(rows, 1, scioxide_core::quantity!("lp_row_offsets"))?,
        scioxide_core::quantity!("lp_row_offsets"),
    )?;
    output.row_offsets.push(0);
    resize(&mut output.rhs, rows, zero)?;
    resize(&mut output.row_signs, rows, one)?;
    resize(&mut output.original_row, original_dimension, zero)?;
    let mut transformed_row = try_filled(
        columns,
        zero,
        scioxide_core::quantity!("lp_transformed_row"),
    )?;
    let mut output_inequality = 0;
    for row in 0..rows {
        transformed_row.fill(zero);
        if row < problem_rows {
            let constraints = problem.constraints().ok_or_else(|| {
                SciOxideError::invalid_input(
                    scioxide_core::argument!("problem"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing constraints",
                )
            })?;
            constraints
                .coefficients()
                .write_row(row, &mut output.original_row)?;
            let rhs = constraints.rhs()[row];
            let sense = constraints.senses()[row];
            write_transformed_row(
                &output.original_row,
                rhs,
                sense,
                &output.variable_map,
                transformed_variables,
                columns,
                0,
                &mut output_inequality,
                &mut transformed_row,
                &mut output.rhs[row..row + 1],
            )?;
        } else {
            let (variable, width) = output.upper_rows[row - problem_rows];
            transformed_row[variable] = one;
            transformed_row[transformed_variables + output_inequality] = one;
            output_inequality += 1;
            output.rhs[row] = width;
        }
        let sign = if output.rhs[row] < zero { -one } else { one };
        output.row_signs[row] = sign;
        output.rhs[row] *= sign;
        let count = transformed_row.iter().filter(|&&v| v != zero).count();
        try_reserve(
            &mut output.coefficients,
            count,
            scioxide_core::quantity!("lp_sparse_coefficients"),
        )?;
        try_reserve(
            &mut output.column_indices,
            count,
            scioxide_core::quantity!("lp_sparse_columns"),
        )?;
        for (column, &value) in transformed_row.iter().enumerate() {
            if value != zero {
                output.column_indices.push(column);
                output.coefficients.push(value * sign);
            }
        }
        output.row_offsets.push(output.coefficients.len());
    }

    resize(&mut output.row_exponents, rows, 0)?;
    output.row_exponents.fill(0);
    resize(&mut output.column_exponents, columns, 0)?;
    output.column_exponents.fill(0);
    output.cost_exponent = 0;
    output.rows = rows;
    output.columns = columns;
    output.transformed_variables = transformed_variables;
    output.index_columns()
}

#[allow(clippy::too_many_arguments)]
fn write_transformed_row<S>(
    original: &[S],
    original_rhs: S,
    sense: LinearConstraintSense,
    variable_map: &[VariableMap<S>],
    transformed_variables: usize,
    columns: usize,
    output_row: usize,
    output_inequality: &mut usize,
    coefficients: &mut [S],
    rhs: &mut [S],
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let row_start = output_row * columns;
    let row = &mut coefficients[row_start..row_start + columns];
    let mut shifted_rhs = scioxide_core::RangeAccumulator::new();
    shifted_rhs.add_product(original_rhs, one)?;
    for (&coefficient, mapping) in original.iter().zip(variable_map) {
        match *mapping {
            VariableMap::Affine {
                transformed,
                offset,
                scale,
            } => {
                row[transformed] += coefficient * scale;
                shifted_rhs.add_product(-coefficient, offset)?;
            }
            VariableMap::Free { positive, negative } => {
                row[positive] += coefficient;
                row[negative] -= coefficient;
            }
            VariableMap::Fixed(value) => {
                shifted_rhs.add_product(-coefficient, value)?;
            }
        }
    }
    let mut transformed_rhs = shifted_rhs.finish()?;
    if row[..transformed_variables]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "constraint transformation produced a non-finite coefficient",
        ));
    }

    match sense {
        LinearConstraintSense::Equal => {}
        LinearConstraintSense::LessEqual => {
            row[transformed_variables + *output_inequality] = one;
            *output_inequality += 1;
        }
        LinearConstraintSense::GreaterEqual => {
            for value in &mut row[..transformed_variables] {
                *value = -*value;
            }
            transformed_rhs = -transformed_rhs;
            row[transformed_variables + *output_inequality] = one;
            *output_inequality += 1;
        }
    }
    rhs[output_row] = transformed_rhs;
    Ok(())
}

fn ensure_finite<S>(value: S) -> Result<(), SciOxideError>
where
    S: RealScalar,
{
    if !value.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "finite inputs produced a non-finite transformed value",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Bounds;

    #[test]
    fn constant_shifts_cancel_before_rounding_in_both_precisions() {
        macro_rules! check {
            ($scalar:ty) => {{
                let bounds =
                    Bounds::new(vec![(Some(<$scalar>::MAX), Some(<$scalar>::MAX)); 2]).unwrap();
                let matrix = scioxide_linalg::Matrix::from_vec(1, 2, vec![2.0, -2.0]).unwrap();
                let constraints = crate::LinearConstraints::new(
                    matrix,
                    vec![1.0],
                    vec![LinearConstraintSense::Equal],
                )
                .unwrap();
                let problem = LpProblem::new(vec![0.0 as $scalar; 2])
                    .unwrap()
                    .with_bounds(bounds)
                    .unwrap()
                    .with_constraints(constraints)
                    .unwrap();
                let mut form = StandardForm::new();
                standardize_into(&problem, &mut form).unwrap();
                assert_eq!(form.columns, 0);
                assert_eq!(form.rhs, [1.0]);
                standardize_into(&problem, &mut form).unwrap();
                assert_eq!(form.rhs, [1.0]);
                assert_eq!(form.reconstruct_solution(&[]).unwrap(), [<$scalar>::MAX; 2]);
            }};
        }
        check!(f32);
        check!(f64);
    }

    #[test]
    fn transforms_free_shifted_upper_only_and_fixed_variables() {
        let bounds = Bounds::new(vec![
            (None, None),
            (Some(-2.0_f64), Some(3.0)),
            (None, Some(4.0)),
            (Some(5.0), Some(5.0)),
        ])
        .unwrap();
        let problem = LpProblem::new(vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .with_bounds(bounds)
            .unwrap();
        let form = standardize(&problem).unwrap();
        assert_eq!(form.transformed_variables, 4);
        assert_eq!(form.rows, 1);
        assert_eq!(form.columns, 5);

        let original = form
            .reconstruct_solution(&[3.0, 1.0, 2.0, 1.5, 0.0])
            .unwrap();
        assert_eq!(original, vec![2.0, 0.0, 2.5, 5.0]);
        // Direction encoding carries no affine offsets, including for free,
        // upper-only and fixed variables. Slack completion is homogeneous.
        let direction = [2.0, 0.0, -1.0, 0.0];
        let encoded = form.encode_direction(&direction).unwrap();
        assert_eq!(form.reconstruct_direction(&encoded).unwrap(), direction);
        assert!(form.row_dot(0, &encoded).unwrap().abs() < 1e-14);
        assert!(form.encode_direction(&direction[..3]).is_err());
    }
}
