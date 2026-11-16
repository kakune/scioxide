//! Sparse rows, exact presolve, and reconstruction in original coordinates.
use super::*;
use crate::{CoefficientMatrix, LinearConstraintSense as Sense, LinearConstraints};
use scioxide_core::Bounds;

use scioxide_linalg::Matrix;
use scioxide_sparse::CsrMatrix;

#[derive(Debug, PartialEq)]
pub(super) struct Row<S> {
    pub entries: Vec<(usize, S)>,
    pub rhs: S,
    pub sense: Sense,
    pub provenance: Option<(MipCutKind, usize)>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Row<S> {
    pub(super) fn value(&self, x: &[S]) -> Result<S, SciOxideError> {
        let mut sum = scioxide_core::ProductSum::new();
        for &(j, a) in &self.entries {
            sum.add_product(a, x[j])?;
        }
        sum.finish()
    }
    pub(super) fn bytes(&self) -> usize {
        core::mem::size_of::<Self>() + self.entries.capacity() * core::mem::size_of::<(usize, S)>()
    }
}

/// Maps the presolved model of [`BranchAndBound`](crate::BranchAndBound) back to the original problem.
#[derive(Debug)]
pub struct MipPresolveMap<S> {
    pub(super) active: Vec<usize>,
    pub(super) fixed: Vec<Option<S>>,
    pub(super) offset: S,
    pub(super) rows: Vec<usize>,
}
impl<S: RealScalar> MipPresolveMap<S> {
    /// Returns the original indices of the variables kept by presolve, in increasing order.
    pub fn active_variables(&self) -> &[usize] {
        &self.active
    }
    /// Returns the original index of each row kept by presolve.
    pub fn original_rows(&self) -> &[usize] {
        &self.rows
    }
    /// Returns the constant objective contribution of the removed variables.
    pub const fn objective_offset(&self) -> S {
        self.offset
    }
    /// Returns the original point for a point `reduced` of the presolved model.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `reduced` does not have one entry per kept
    /// variable, or `AllocationFailure` if the point cannot be allocated.
    pub fn reconstruct(&self, reduced: &[S]) -> Result<Vec<S>, SciOxideError> {
        if reduced.len() != self.active.len() {
            return Err(invalid(
                scioxide_core::argument!("reduced"),
                "presolve point dimension",
            ));
        }
        let mut x = Vec::new();
        self.reconstruct_into(reduced, &mut x)?;
        Ok(x)
    }
    pub(super) fn reconstruct_into(
        &self,
        reduced: &[S],
        x: &mut Vec<S>,
    ) -> Result<(), SciOxideError> {
        if reduced.len() != self.active.len() {
            return Err(invalid(
                scioxide_core::argument!("reduced"),
                "presolve point dimension",
            ));
        }
        crate::storage::resize(x, self.fixed.len(), S::ZERO)?;
        for (v, fixed) in x.iter_mut().zip(&self.fixed) {
            if let Some(f) = fixed {
                *v = *f;
            }
        }
        for (&j, &value) in self.active.iter().zip(reduced) {
            x[j] = value;
        }
        Ok(())
    }
}

pub(super) struct Model<S: RealScalar + NumericScalar<Real = S>> {
    pub objective: Vec<S>,
    pub bounds: Vec<(Option<S>, Option<S>)>,
    pub rows: Vec<Row<S>>,
    pub integer: Vec<usize>,
    pub sparse: bool,
    pub map: MipPresolveMap<S>,
    pub infeasible: bool,
}

// Accept only exactly representable substitution operations. Comparing decoded
// integer products also detects residuals too tiny for FMA to represent. TwoDiff
// detects subtraction loss between represented scalars.
fn exact_sub_product<S: RealScalar>(b: S, a: S, x: S) -> Option<S> {
    let p = a * x;
    if !p.is_finite()
        || (p != S::ZERO && !p.is_normal())
        || (p == S::ZERO && a != S::ZERO && x != S::ZERO)
    {
        return None;
    }
    let scioxide_core::FloatDecomposition::Finite {
        significand: am,
        exponent: ae,
        ..
    } = a.decompose()
    else {
        unreachable!("caller established a finite scalar before decomposition");
    };
    let scioxide_core::FloatDecomposition::Finite {
        significand: xm,
        exponent: xe,
        ..
    } = x.decompose()
    else {
        unreachable!("caller established a finite scalar before decomposition");
    };
    let scioxide_core::FloatDecomposition::Finite {
        significand: pm,
        exponent: pe,
        ..
    } = p.decompose()
    else {
        unreachable!("caller established a finite scalar before decomposition");
    };
    let product = u128::from(am) * u128::from(xm);
    if product != 0 {
        if pm == 0 {
            return None;
        }
        let shift = product.trailing_zeros();
        let rounded_shift = pm.trailing_zeros();
        if product >> shift != u128::from(pm >> rounded_shift)
            || i32::from(ae) + i32::from(xe) + shift as i32 != i32::from(pe) + rounded_shift as i32
        {
            return None;
        }
    }
    let d = b - p;
    let v = b - d;
    let error = (b - (d + v)) + (v - p);
    (d.is_finite() && error == S::ZERO).then_some(d)
}

/// Fill missing column bounds implied by rows whose other columns have finite
/// activity bounds.
///
/// Floating-point dual-bound estimates need a finite box wherever a reduced cost
/// may have the wrong sign. Root support is prepared before branching: a
/// rounding-level residual on a column without one would leave a node LP with
/// no verified bound. Each implied side is rounded outward exactly, so it is
/// redundant for the relaxation and holds at every feasible point. Only missing
/// sides are filled, each at most once, so the Gauss–Seidel passes terminate.
/// Sides whose quotient is not representable stay missing.
fn complete_bounds<S: RealScalar + NumericScalar<Real = S>>(
    rows: &[Row<S>],
    bounds: &mut [(Option<S>, Option<S>)],
) -> Result<(), SciOxideError> {
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    loop {
        let mut changed = false;
        for row in rows {
            // `at_most` treats the row as `a x <= b`, using minimum activity;
            // otherwise as `a x >= b`, using maximum activity.
            for at_most in [true, false] {
                let applies = match row.sense {
                    Sense::LessEqual => at_most,
                    Sense::GreaterEqual => !at_most,
                    Sense::Equal => true,
                };
                if !applies {
                    continue;
                }
                // Activity side of `x_k`; the implied side is the other one.
                let side = |a: S, (lower, upper): (Option<S>, Option<S>)| {
                    if (a > z) == at_most { lower } else { upper }
                };
                // Exact `b - sum_k a_k bound_k` over the finite activity terms.
                let mut base = scioxide_core::RangeAccumulator::<S>::new();
                base.add(row.rhs)?;
                let mut missing = None;
                let mut count = 0_usize;
                for &(k, a) in &row.entries {
                    match side(a, bounds[k]) {
                        Some(bound) => base.subtract_product(a, bound)?,
                        None => {
                            count += 1;
                            missing = Some(k);
                        }
                    }
                }
                if count > 1 {
                    continue;
                }
                for &(j, a) in &row.entries {
                    if count == 1 && missing != Some(j) {
                        continue;
                    }
                    let upper = (a > z) == at_most;
                    if (if upper { bounds[j].1 } else { bounds[j].0 }).is_some() {
                        continue;
                    }
                    let mut numerator = base.clone();
                    if let Some(own) = side(a, bounds[j]) {
                        numerator.add_product(a, own)?;
                    }
                    let Some(bound) = outward_quotient(&numerator, a, upper)? else {
                        continue;
                    };
                    if upper {
                        bounds[j].1 = Some(bound);
                    } else {
                        bounds[j].0 = Some(bound);
                    }
                    changed = true;
                }
            }
        }
        if !changed {
            return Ok(());
        }
    }
}

/// `numerator / denominator` rounded toward `+inf` when `upward`, else toward
/// `-inf`; `None` when that bound is not finite.
fn outward_quotient<S: RealScalar + NumericScalar<Real = S>>(
    numerator: &scioxide_core::RangeAccumulator<S>,
    denominator: S,
    upward: bool,
) -> Result<Option<S>, SciOxideError> {
    use core::cmp::Ordering;
    let mut quotient = match numerator.finish_quotient(denominator) {
        Ok(quotient) => quotient,
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let (target, direction) = if upward {
        (Ordering::Greater, S::INFINITY)
    } else {
        (Ordering::Less, S::NEG_INFINITY)
    };
    // sign(exact - quotient) = sign(numerator - denominator * quotient) * sign(denominator).
    let ahead = |quotient: S| -> Result<bool, SciOxideError> {
        let mut residual = numerator.clone();
        residual.subtract_product(denominator, quotient)?;
        let sign = residual.sign();
        Ok(
            if denominator > <S as scioxide_core::AlgebraicScalar>::ZERO {
                sign == target
            } else {
                sign == target.reverse()
            },
        )
    };
    while ahead(quotient)? {
        quotient = quotient.next_after(direction);
        if !quotient.is_finite() {
            return Ok(None);
        }
    }
    Ok(Some(quotient))
}

impl<S: RealScalar + NumericScalar<Real = S>> Model<S> {
    pub(super) fn prepare(problem: &MipProblem<S>, enabled: bool) -> Result<Self, SciOxideError> {
        let p = problem.relaxation();
        let n = p.dimension();
        let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let mut bounds = match p.bounds() {
            Some(b) => copy(b.as_slice(), scioxide_core::quantity!("mip_bounds"))?,
            None => filled(n, (Some(z), None), scioxide_core::quantity!("mip_bounds"))?,
        };
        let mut rows = capacity(
            p.constraints().map_or(0, |c| c.len()),
            scioxide_core::quantity!("mip_rows"),
        )?;
        if let Some(c) = p.constraints() {
            for i in 0..c.len() {
                let mut entries = capacity(
                    match c.coefficients() {
                        CoefficientMatrix::Dense(_) => n,
                        CoefficientMatrix::Sparse(a) => a.row_ptr()[i + 1] - a.row_ptr()[i],
                    },
                    scioxide_core::quantity!("mip_coefficients"),
                )?;
                match c.coefficients() {
                    CoefficientMatrix::Dense(a) => entries.extend(
                        a.as_slice()[i * n..(i + 1) * n]
                            .iter()
                            .copied()
                            .enumerate()
                            .filter(|(_, a)| *a != z),
                    ),
                    CoefficientMatrix::Sparse(a) => entries.extend(
                        (a.row_ptr()[i]..a.row_ptr()[i + 1])
                            .map(|k| (a.col_indices()[k], a.values()[k]))
                            .filter(|(_, a)| *a != z),
                    ),
                }
                rows.push(Row {
                    entries,
                    rhs: c.rhs()[i],
                    sense: c.senses()[i],
                    provenance: None,
                });
            }
        }
        if enabled {
            // Singleton divisions are used only when the product reconstructs
            // the original RHS exactly; arbitrary rounded quotients are skipped.
            for row in &rows {
                if let [(j, a)] = row.entries.as_slice() {
                    let q = row.rhs / *a;
                    if q.is_finite() && exact_sub_product(row.rhs, *a, q) == Some(z) {
                        let sense = if *a < z {
                            match row.sense {
                                Sense::LessEqual => Sense::GreaterEqual,
                                Sense::GreaterEqual => Sense::LessEqual,
                                Sense::Equal => Sense::Equal,
                            }
                        } else {
                            row.sense
                        };
                        let (lo, hi) = &mut bounds[*j];
                        if sense != Sense::LessEqual {
                            *lo = Some(lo.map_or(q, |v| v.max(q)));
                        }
                        if sense != Sense::GreaterEqual {
                            *hi = Some(hi.map_or(q, |v| v.min(q)));
                        }
                    }
                }
            }
        }
        complete_bounds(&rows, &mut bounds)?;
        if enabled {
            for &j in problem.integer_variables() {
                bounds[j] = (
                    bounds[j].0.map(RealScalar::ceil),
                    bounds[j].1.map(RealScalar::floor),
                );
            }
        }
        // Crossing bounds prove infeasibility; `Bounds` would reject them.
        let mut infeasible = bounds
            .iter()
            .any(|&(l, u)| matches!((l, u), (Some(l), Some(u)) if l > u));
        let mut fixed = filled(n, None, scioxide_core::quantity!("mip_fixed_map"))?;
        let mut offset = z;
        if enabled && !infeasible {
            let mut trial = capacity(rows.len(), scioxide_core::quantity!("presolve_trial_rhs"))?;
            for j in 0..n {
                let value = match bounds[j] {
                    (Some(lo), Some(hi)) if lo == hi => Some(lo),
                    (lo, hi)
                        if rows
                            .iter()
                            .all(|row| row.entries.iter().all(|(k, _)| *k != j)) =>
                    {
                        let cost = p.objective()[j];
                        if cost > z {
                            lo
                        } else if cost < z {
                            hi
                        } else {
                            Some(lo.unwrap_or(z).max(z).min(hi.unwrap_or(S::MAX)))
                        }
                    }
                    _ => None,
                };
                let Some(lo) = value else {
                    continue;
                };
                let Some(new_offset) = exact_sub_product(offset, -p.objective()[j], lo) else {
                    continue;
                };
                trial.clear();
                for row in &rows {
                    let a = row
                        .entries
                        .iter()
                        .find(|(k, _)| *k == j)
                        .map_or(z, |(_, a)| *a);
                    let Some(rhs) = exact_sub_product(row.rhs, a, lo) else {
                        break;
                    };
                    trial.push(rhs);
                }
                if trial.len() != rows.len() {
                    continue;
                }
                for (row, &rhs) in rows.iter_mut().zip(&trial) {
                    row.rhs = rhs;
                    row.entries.retain(|(k, _)| *k != j);
                }
                fixed[j] = Some(lo);
                offset = new_offset;
            }
        }
        let mut active = capacity(n, scioxide_core::quantity!("mip_active_map"))?;
        active.extend((0..n).filter(|&j| fixed[j].is_none()));
        let mut inverse = filled(n, 0, scioxide_core::quantity!("mip_inverse_map"))?;
        for (k, &j) in active.iter().enumerate() {
            inverse[j] = k;
        }
        let mut original_rows = capacity(rows.len(), scioxide_core::quantity!("mip_row_map"))?;
        let mut retained = capacity(rows.len(), scioxide_core::quantity!("mip_retained_rows"))?;
        for (i, mut row) in rows.into_iter().enumerate() {
            if row.entries.is_empty() {
                infeasible |= match row.sense {
                    Sense::LessEqual => row.rhs < z,
                    Sense::GreaterEqual => row.rhs > z,
                    Sense::Equal => row.rhs != z,
                };
                continue;
            }
            for (j, _) in &mut row.entries {
                *j = inverse[*j];
            }
            original_rows.push(i);
            retained.push(row);
        }
        let mut objective = capacity(active.len(), scioxide_core::quantity!("mip_objective"))?;
        objective.extend(active.iter().map(|&j| p.objective()[j]));
        let mut reduced_bounds =
            capacity(active.len(), scioxide_core::quantity!("mip_reduced_bounds"))?;
        reduced_bounds.extend(active.iter().map(|&j| bounds[j]));
        let mut integer = capacity(
            problem.integer_variables().len(),
            scioxide_core::quantity!("mip_integers"),
        )?;
        integer.extend(
            problem
                .integer_variables()
                .iter()
                .copied()
                .filter(|&j| fixed[j].is_none())
                .map(|j| inverse[j]),
        );
        integer.sort_unstable();
        Ok(Self {
            objective,
            bounds: reduced_bounds,
            rows: retained,
            integer,
            sparse: p
                .constraints()
                .is_some_and(|c| c.coefficients().is_sparse()),
            map: MipPresolveMap {
                active,
                fixed,
                offset,
                rows: original_rows,
            },
            infeasible,
        })
    }

    pub(super) fn lp<'a>(
        &'a self,
        extra: impl Iterator<Item = &'a Row<S>> + Clone,
    ) -> Result<LpProblem<S>, SciOxideError> {
        let n = self.objective.len();
        let m = checked_add(
            self.rows.len(),
            extra.clone().count(),
            scioxide_core::quantity!("mip_lp_rows"),
        )?;
        let mut p = LpProblem::new(copy(
            &self.objective,
            scioxide_core::quantity!("mip_lp_objective"),
        )?)?
        .with_bounds(Bounds::new(copy(
            &self.bounds,
            scioxide_core::quantity!("mip_lp_bounds"),
        )?)?)?;
        if m == 0 {
            return Ok(p);
        }
        let nnz = self
            .rows
            .iter()
            .chain(extra.clone())
            .try_fold(0usize, |v, r| {
                checked_add(
                    v,
                    r.entries.len(),
                    scioxide_core::quantity!("mip_lp_nonzeros"),
                )
            })?;
        let mut values = capacity(nnz, scioxide_core::quantity!("mip_lp_values"))?;
        let mut columns = capacity(nnz, scioxide_core::quantity!("mip_lp_columns"))?;
        let mut offsets = capacity(
            checked_add(m, 1, scioxide_core::quantity!("mip_offsets"))?,
            scioxide_core::quantity!("mip_lp_offsets"),
        )?;
        let mut rhs = capacity(m, scioxide_core::quantity!("mip_lp_rhs"))?;
        let mut senses = capacity(m, scioxide_core::quantity!("mip_lp_senses"))?;
        offsets.push(0);
        for row in self.rows.iter().chain(extra.clone()) {
            values.extend(row.entries.iter().map(|(_, a)| *a));
            columns.extend(row.entries.iter().map(|(j, _)| *j));
            offsets.push(values.len());
            rhs.push(row.rhs);
            senses.push(row.sense);
        }
        let coefficients = if self.sparse {
            CoefficientMatrix::Sparse(CsrMatrix::from_raw(m, n, offsets, columns, values)?)
        } else {
            let mut matrix = Matrix::zeros(m, n)?;
            for (i, row) in self.rows.iter().chain(extra).enumerate() {
                for &(j, a) in &row.entries {
                    matrix.view_mut().set(i, j, a)?;
                }
            }
            CoefficientMatrix::Dense(matrix)
        };
        p = p.with_constraints(LinearConstraints::new(coefficients, rhs, senses)?)?;
        Ok(p)
    }

    pub(super) fn bytes(&self) -> usize {
        self.objective.capacity() * core::mem::size_of::<S>()
            + self.bounds.capacity() * core::mem::size_of::<(Option<S>, Option<S>)>()
            + self.rows.iter().map(Row::bytes).sum::<usize>()
            + self.rows.capacity() * core::mem::size_of::<Row<S>>()
            + (self.integer.capacity() + self.map.active.capacity() + self.map.rows.capacity())
                * core::mem::size_of::<usize>()
            + self.map.fixed.capacity() * core::mem::size_of::<Option<S>>()
    }
}
