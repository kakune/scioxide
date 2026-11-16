//! Fixed-work certification of rounded halfspace vertices.
use crate::tessellation::linear::LinearFactor;
use crate::tessellation::numeric::convert;
use crate::tessellation::storage::Storage;
use crate::tessellation::work::Work;
use crate::tessellation::{Result, unstable};
use alloc::vec::Vec;
use scioxide_core::RangeAccumulator;
use scioxide_core::RealScalar;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_linalg::{Matrix, Vector};

/// Scratch for one defining system, reused across all dual facets.
pub(super) struct VertexWorkspace {
    original: Vec<f64>,
    scales: Vec<f64>,
    matrix: Matrix<f64>,
    inverse: Matrix<f64>,
    factor: LinearFactor,
    rhs: Vector<f64>,
    solution: Vector<f64>,
    defect: Vec<f64>,
    correction: Vec<f64>,
    radii: Vec<f64>,
}

impl VertexWorkspace {
    pub(super) fn new(d: usize, storage: &Storage) -> Result<Self> {
        let quantity = scioxide_core::quantity!("halfspace_vertex_workspace");
        storage.charge(checked_mul(
            checked_add(checked_mul(2, d * d, quantity)?, 2 * d, quantity)?,
            size_of::<f64>(),
            quantity,
        )?)?;
        Ok(Self {
            original: storage.filled(d * (d + 1), 0., quantity)?,
            scales: storage.filled(d, 0., quantity)?,
            matrix: Matrix::zeros(d, d)?,
            inverse: Matrix::zeros(d, d)?,
            factor: LinearFactor::new(d, storage)?,
            rhs: Vector::zeros(d)?,
            solution: Vector::zeros(d)?,
            defect: storage.filled(d * d, 0., quantity)?,
            correction: storage.filled(d, 0., quantity)?,
            radii: storage.filled(d, 0., quantity)?,
        })
    }

    pub(super) fn solve<S: RealScalar>(
        &mut self,
        homogeneous: &[f64],
        planes: &[usize],
        output: &mut Vec<S>,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<()> {
        let d = self.scales.len();

        for (i, &plane) in planes.iter().enumerate() {
            let row = &homogeneous[plane * (d + 1)..(plane + 1) * (d + 1)];
            self.original[i * (d + 1)..(i + 1) * (d + 1)].copy_from_slice(row);
            let scale = row.iter().fold(0_f64, |m, x| m.max(x.abs()));
            self.scales[i] = scale;
            for (entry, a) in self.matrix.as_mut_slice()[i * d..(i + 1) * d]
                .iter_mut()
                .zip(row)
            {
                *entry = a / scale;
            }
            self.rhs.as_mut_slice()[i] = row[d] / scale;
        }
        self.factor.factor(&self.matrix, work.storage)?;
        self.factor
            .solve(&self.matrix, &self.rhs, &mut self.solution, work.storage)?;
        let relative = 32. * S::EPSILON.try_to_f64_exact()? * (d + 1) as f64;
        // The initial solve and at most two corrections have fixed work per
        // vertex. Every candidate is narrowed before verification.
        for attempt in 0..=2 {
            work.check()?;
            for value in self.solution.as_mut_slice() {
                *value = convert::<S>(*value)?.try_to_f64_exact()?;
            }
            self.restore_exact_zeros(relative, work)?;
            if self.exact_solution()? {
                return self.append(output);
            }
            if attempt == 0 {
                self.prepare_inverse(work.storage)?;
            }
            if self.certify(relative, work)? {
                return self.append(output);
            }
            if attempt < 2 {
                for (value, correction) in self
                    .solution
                    .as_mut_slice()
                    .iter_mut()
                    .zip(&self.correction)
                {
                    *value += correction;
                }
            }
        }
        Err(unstable(
            "intersection vertex rounding could not be certified",
        ))
    }

    // A bordered determinant is det(A) * (value - x*_axis). Its sign
    // brackets the exact coordinate without the inverse's coupling error.
    fn coordinate_sign(
        &self,
        axis: usize,
        value: f64,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<i8> {
        let d = self.scales.len();
        work.workspace.rows[..d * (d + 1)].copy_from_slice(&self.original);
        work.workspace.rows[d * (d + 1)..(d + 1) * (d + 1)].fill(0.);
        work.workspace.rows[d * (d + 1) + axis] = 1.;
        work.workspace.rows[(d + 1) * (d + 1) - 1] = value;
        work.determinant(d + 1, d + 1, false, true)
    }

    fn restore_exact_zeros(
        &mut self,
        relative: f64,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<()> {
        let scale = self
            .solution
            .as_slice()
            .iter()
            .fold(0_f64, |m, x| m.max(x.abs()));
        for axis in 0..self.scales.len() {
            let value = self.solution.as_slice()[axis];
            if value != 0.
                && value.abs() <= relative * scale
                && self.coordinate_sign(axis, 0., work)? == 0
            {
                self.solution.as_mut_slice()[axis] = 0.;
            }
        }
        Ok(())
    }

    fn certify_coordinate(&self, axis: usize, work: &mut Work<'_, '_, '_, '_>) -> Result<bool> {
        let value = self.solution.as_slice()[axis];
        let radius = self.radii[axis];
        if radius == 0. {
            return Ok(self.coordinate_sign(axis, value, work)? == 0);
        }
        // Round inward so the interval remains within the allowed radius.
        let lower = (value - radius).next_up();
        let upper = (value + radius).next_down();
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Ok(false);
        }
        let lo = self.coordinate_sign(axis, lower, work)?;
        let hi = self.coordinate_sign(axis, upper, work)?;
        Ok(lo == 0 || hi == 0 || lo != hi)
    }

    fn append<S: RealScalar>(&self, output: &mut Vec<S>) -> Result<()> {
        for &value in self.solution.as_slice() {
            output.push(convert(value)?);
        }
        Ok(())
    }

    fn exact_solution(&mut self) -> Result<bool> {
        let d = self.scales.len();
        for row in self.original.chunks_exact(d + 1) {
            #[cfg(test)]
            {
                crate::tessellation::tests::record_residual_check();
            }
            let mut residual = RangeAccumulator::<f64>::new();
            residual.add(row[d])?;
            for (&a, &x) in row[..d].iter().zip(self.solution.as_slice()) {
                residual.subtract_product(a, x)?;
            }
            if !residual.sign().is_eq() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn prepare_inverse(&mut self, storage: &Storage) -> Result<()> {
        let d = self.scales.len();
        self.factor
            .inverse(&self.matrix, &mut self.inverse, storage)?;
        // R approximates the inverse of the ORIGINAL A. The proof below
        // includes any errors from equilibration, factorization and inversion.
        for row in self.inverse.as_mut_slice().chunks_exact_mut(d) {
            for (value, scale) in row.iter_mut().zip(&self.scales) {
                *value /= scale;
                if !value.is_finite() {
                    return Err(unstable("unrepresentable intersection inverse"));
                }
            }
        }
        for (i, inverse) in self.inverse.as_slice().chunks_exact(d).enumerate() {
            let mut norm = 0.;
            for j in 0..d {
                #[cfg(test)]
                {
                    crate::tessellation::tests::record_residual_check();
                }
                let bound = defect_upper(inverse, &self.original, j, i == j)?;
                self.defect[i * d + j] = bound;
                norm = add_up(norm, bound);
            }
            if norm >= 1. {
                return Err(unstable("intersection inverse is ill-conditioned"));
            }
        }
        Ok(())
    }

    // Let x* be the exact vertex, e=x*-q, E=I-RA and z=R(c-Aq).
    // Then e=z+Ee. With ||E||inf<1, |z|+|E|t<=t proves |e|<=t
    // by the convergent Neumann series. Coordinates proved directly by exact
    // determinants can instead be treated as bounded forcing terms: the
    // remaining principal submatrix of |E| still has norm less than one.
    // Since x* satisfies EVERY inequality,
    // a.q+b <= sum |a_i|t_i <= relative * sum |a_i q_i|, even for
    // nearly tight non-active planes. No incidence-neighborhood assumption is
    // involved. This is the preconditioned residual certificate of Rump,
    // Verification Methods (Acta Numerica 2010), section 10.3, componentwise.
    fn certify(&mut self, relative: f64, work: &mut Work<'_, '_, '_, '_>) -> Result<bool> {
        let d = self.scales.len();

        for (radius, &value) in self.radii.iter_mut().zip(self.solution.as_slice()) {
            let product = relative * value.abs();
            if !product.is_finite() || relative >= 1. {
                return Err(unstable("unrepresentable intersection error radius"));
            }
            *radius = product.next_down().max(0.);
        }
        let mut admitted = true;
        for (i, inverse) in self.inverse.as_slice().chunks_exact(d).enumerate() {
            work.check()?;
            #[cfg(test)]
            {
                crate::tessellation::tests::record_residual_check();
            }
            // Expand R(c-Aq) into exact pairs/triples. Rounding a residual
            // before multiplying by R would lose cancellation in thin facets.
            let mut residual = RangeAccumulator::<f64>::new();
            for (&r, row) in inverse.iter().zip(self.original.chunks_exact(d + 1)) {
                residual.add_product(r, row[d])?;
                for (&a, &q) in row[..d].iter().zip(self.solution.as_slice()) {
                    residual.subtract_triple_product(r, a, q)?;
                }
            }
            self.correction[i] = residual.finish()?;
            let mut bound = if residual.sign().is_eq() {
                0.
            } else {
                self.correction[i].abs().next_up()
            };
            for (&entry, &radius) in self.defect[i * d..(i + 1) * d].iter().zip(&self.radii) {
                let product = if entry == 0. || radius == 0. {
                    0.
                } else {
                    (entry * radius).next_up()
                };
                bound = add_up(bound, product);
            }
            admitted &= bound <= self.radii[i] || self.certify_coordinate(i, work)?;
        }
        Ok(admitted)
    }
}

// Only nonnegative bounds are added here. Preserve exact zeros; next_up also
// encloses positive sums/products that rounded to a subnormal or to zero.
fn add_up(a: f64, b: f64) -> f64 {
    if a == 0. {
        b
    } else if b == 0. {
        a
    } else {
        (a + b).next_up()
    }
}

// Enclose one entry of I-RA with directed endpoints. This needs no precision
// or conditioning estimate. If intermediate products/sums overflow, the exact
// accumulator recovers cancellation before rounding the final bound.
fn defect_upper(inverse: &[f64], original: &[f64], column: usize, diagonal: bool) -> Result<f64> {
    let mut lower = f64::from(diagonal);
    let mut upper = lower;
    for (&r, row) in inverse.iter().zip(original.chunks_exact(inverse.len() + 1)) {
        let a = row[column];
        if r == 0. || a == 0. {
            continue;
        }
        let product = r * a;
        lower = (lower - product.next_up()).next_down();
        upper = (upper - product.next_down()).next_up();
    }
    if lower.is_finite() && upper.is_finite() {
        return Ok(lower.abs().max(upper.abs()));
    }
    let mut exact = RangeAccumulator::<f64>::new();
    exact.add(f64::from(diagonal))?;
    for (&r, row) in inverse.iter().zip(original.chunks_exact(inverse.len() + 1)) {
        exact.subtract_product(r, row[column])?;
    }
    absolute_upper(&exact)
}

// An exact zero stays zero. Otherwise nearest rounding followed by next_up
// encloses the absolute exact sum, including gradual underflow. An infinite
// bound cannot certify a finite rounding allowance.
fn absolute_upper(sum: &RangeAccumulator<f64>) -> Result<f64> {
    if sum.sign().is_eq() {
        return Ok(0.);
    }
    let upper = sum.finish()?.abs().next_up();
    if !upper.is_finite() {
        return Err(unstable("unrepresentable intersection certificate"));
    }
    Ok(upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tessellation::tests::{test_requirements, test_workspace};
    use crate::tessellation::{GeometryLimits, GeometryOptions};

    #[test]
    fn certificate_rejects_error_hidden_by_active_residuals() {
        let storage = Storage::new(usize::MAX, 0).unwrap();

        let options = GeometryOptions::new().with_limits(
            GeometryLimits::new()
                .with_max_dimension(2)
                .with_max_points(4)
                .with_max_facets(8)
                .with_max_predicate_terms(100_000)
                .with_max_storage_bytes(1_000_000),
        );
        let mut geometry = test_workspace(test_requirements(2, options).unwrap()).unwrap();
        let mut work = Work::new(
            &mut geometry,
            2,
            options,
            None,
            scioxide_core::argument!("interior"),
            &storage,
        )
        .unwrap();
        let delta = 2_f64.powi(-30);
        let rows = [1., 1., 2., 1., 1. + delta, 2. + delta];
        let mut workspace = VertexWorkspace::new(2, &storage).unwrap();
        let mut output = Vec::<f64>::new();
        workspace
            .solve(&rows, &[0, 1], &mut output, &mut work)
            .unwrap();
        assert_eq!(output, [1., 1.]);
        workspace.prepare_inverse(&storage).unwrap();
        // Exact endpoint proofs accept a containing interval and reject a
        // displaced one, even when its endpoints round close to the root.
        workspace.radii.fill(128. * f64::EPSILON);
        assert!(workspace.certify_coordinate(0, &mut work).unwrap());
        workspace.solution.as_mut_slice()[0] = 1_f64.next_up();
        assert!(workspace.certify_coordinate(0, &mut work).unwrap());
        workspace.radii[0] = 0.;
        assert!(!workspace.certify_coordinate(0, &mut work).unwrap());
        workspace.solution.as_mut_slice()[0] = 1.;
        assert!(workspace.certify_coordinate(0, &mut work).unwrap());
        let error = 2_f64.powi(-12);
        workspace
            .solution
            .as_mut_slice()
            .copy_from_slice(&[1. + error, 1. - error]);
        let epsilon = 128. * f64::EPSILON * 3.;
        for row in rows.chunks_exact(3) {
            let x = workspace.solution.as_slice();
            let residual = row[0] * x[0] + row[1] * x[1] - row[2];
            let magnitude = (row[0] * x[0]).abs() + (row[1] * x[1]).abs() + row[2].abs();
            assert!(residual.abs() <= epsilon * magnitude);
        }
        // The non-active plane x <= 1 + 2^-50 is violated far beyond allowance.
        assert!(workspace.solution.as_slice()[0] - (1. + 2_f64.powi(-50)) > epsilon * 3.);
        assert!(!workspace.certify(epsilon / 4., &mut work).unwrap());
    }

    #[test]
    fn exact_accumulation_bounds_cover_underflow_and_cancellation() {
        let tiny = f64::from_bits(1);
        let mut sum = RangeAccumulator::<f64>::new();
        sum.add_product(tiny, tiny).unwrap();
        assert_eq!(absolute_upper(&sum).unwrap(), tiny);
        sum.subtract_product(tiny, tiny).unwrap();
        assert_eq!(absolute_upper(&sum).unwrap(), 0.);
        sum.add_product(f64::MAX, f64::MAX).unwrap();
        sum.subtract_product(f64::MAX, f64::MAX).unwrap();
        sum.add(1.).unwrap();
        assert_eq!(absolute_upper(&sum).unwrap(), 1_f64.next_up());
    }

    #[test]
    fn inverse_defect_enclosures_cover_exact_dots() {
        let mut state = 1117_u64;
        for case in 0..128 {
            let mut rows = [0.; 12];
            let mut inverse = [0.; 3];
            for value in rows.iter_mut().chain(&mut inverse) {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *value = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            }
            if case % 4 == 0 {
                rows[0] = f64::from_bits(1);
                inverse[0] = f64::from_bits(1);
            }
            for column in 0..3 {
                let diagonal = column == 0;
                let bound = defect_upper(&inverse, &rows, column, diagonal).unwrap();
                for sign in [-1., 1.] {
                    let mut difference = RangeAccumulator::<f64>::new();
                    difference.add(sign * f64::from(diagonal)).unwrap();
                    for (&r, row) in inverse.iter().zip(rows.chunks_exact(4)) {
                        difference.subtract_product(sign * r, row[column]).unwrap();
                    }
                    difference.add(-bound).unwrap();
                    assert!(!difference.sign().is_gt());
                }
            }
        }
        let rows = [f64::MAX, 0., 0., f64::MAX, 0., 0.];
        let bound = defect_upper(&[f64::MAX, -f64::MAX], &rows, 0, true).unwrap();
        assert_eq!(bound, 1_f64.next_up());
    }
}
