//! Direct contour summation: exactly reduced and cached coefficients, the
//! compensated working-precision tier with its error-bound test, and the
//! extended-exponent fallback.
use super::contour::radial;
use super::{CztContour, CztPlan, CztPlanSpec};
use crate::chirp::phase;
use crate::numeric::compensated_term;
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{Complex, ExecutionContext, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator, ScaledValue};

#[derive(Debug, Clone)]
pub(super) struct Direct<S: RealScalar> {
    columns: Vec<Column<S>>,
    rows: Vec<Row>,
    exponents: Vec<[i64; 2]>,
    coefficients: Vec<Complex<S>>,
    coefficient_minimum: S,
    ar: f64,
    dr: f64,
    accuracy: Option<DirectAccuracy<S>>,
}

/// Per-column factors keep the fixed radial-log evaluation order.
#[derive(Debug, Clone, Copy)]
struct Column<S> {
    start: Complex<S>,
    radial_start: f64,
    radial_step: f64,
}

#[derive(Debug, Clone, Copy)]
enum Row {
    Normalized(i64),
    Extended,
    Built(Option<i64>),
}

#[derive(Debug, Clone, Copy)]
struct DirectAccuracy<S: RealScalar> {
    error_factor: S,
    result_factor: S,
    underflow: S,
    relative: S,
}
impl<S: RealScalar> DirectAccuracy<S> {
    fn new(n: usize) -> Result<Option<Self>, SciOxideError> {
        let u = S::EPSILON.into_f64() / 2.0;
        let q = 16.0 * (n as f64 + 1.0);
        if q * u > 1.0 / 128.0 {
            return Ok(None);
        }
        let gamma = |r: f64| r * u / (1.0 - r * u);
        let product = gamma(2.0);
        let reduction = 2.0 * gamma(2.0 * n as f64 + 8.0).powi(2) * (1.0 + product);
        // Sum2: |fl(sum)-sum| <= u*|sum| + gamma(n)^2*sum|terms|.
        // The unrounded four-lane sums/corrections are merged with Sum2 too.
        // Ordinary complex products cost gamma(2)*mass; scaled cross-products
        // round once each, costing u*mass, and reduce 2n terms per component. Eliminate the unknown exact sum using
        // |sum| <= (|total| + reduction*mass)/(1-u).
        let inflation = 1.0 / (1.0 - q * u);
        Ok(Some(Self {
            error_factor: S::try_from_f64_lossy(
                inflation * ((1.0 + u) * (product + u) + 2.0 * reduction / (1.0 - u))
                    / ((1.0 - u) * (1.0 - gamma(2.0 * n as f64 + 8.0))),
            )?,
            result_factor: S::try_from_f64_lossy(inflation * 2.0 * u / (1.0 - u))?,
            underflow: S::MIN_SUBNORMAL * S::try_from_f64_lossy(8.0 * q)?,
            relative: S::try_from_f64_lossy(64.0 * S::EPSILON.into_f64() * (n as f64 + 2.0))?,
        }))
    }

    fn admits(self, total: S, mass: S) -> Option<S> {
        if mass.is_zero() {
            return Some(S::ZERO);
        }
        let error = (mass * self.error_factor + total.abs() * self.result_factor + self.underflow)
            .next_after(S::INFINITY);
        let allowance = (total.abs() * self.relative).next_after(S::ZERO);
        // Uncertain subnormal certificates use the extended row. Directed
        // rounding makes this comparison conservative on the ordinary grid.
        (error.is_normal() && allowance.is_normal() && error <= allowance).then_some(error)
    }
}

impl<S: RealScalar> Direct<S> {
    /// Prepare column factors and row frames, caching small representable rows
    /// in one pass without temporary coefficient or exponent storage.
    pub(super) fn prepare(
        spec: &CztPlanSpec<S>,
        small: bool,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let c = spec.contour;
        let n = spec.n;
        let ar = c.log_radius().try_to_f64_exact()?;
        let dr = c.log_radius_step().try_to_f64_exact()?;
        let cached = if small { spec.n * spec.m } else { 0 };
        let admit = |n, m, cached, extended| {
            crate::memory::admit_bytes(
                preparation_bytes::<S>(n, m, cached, extended)?,
                spec.limits.preparation_bytes,
                scioxide_core::quantity!("czt_preparation_bytes"),
            )
        };
        admit(n, spec.m, cached, 0)?;
        let mut columns = try_filled(
            n,
            Column {
                start: Complex::new(S::ZERO, S::ZERO),
                radial_start: 0.0,
                radial_step: 0.0,
            },
            scioxide_core::quantity!("czt_column_factors"),
        )?;
        let mut rows = try_filled(
            spec.m,
            Row::Built(None),
            scioxide_core::quantity!("czt_rows"),
        )?;
        let mut coefficients = try_filled(
            cached,
            Complex::new(S::ZERO, S::ZERO),
            scioxide_core::quantity!("czt_coefficients"),
        )?;
        let mut exponents: Vec<[i64; 2]> = Vec::new();
        admit(
            columns.capacity(),
            rows.capacity(),
            coefficients.capacity(),
            0,
        )?;
        for (j, column) in columns.iter_mut().enumerate() {
            context.check()?;
            *column = Column {
                start: phase::<S>(-c.start_turns(), j as u128)?,
                radial_start: -ar * j as f64,
                radial_step: dr * j as f64,
            };
        }
        let row_capacity = rows.capacity();
        for (k, row) in rows.iter_mut().enumerate() {
            context.check()?;
            let end = columns[n - 1];
            let log = end.radial_start - end.radial_step * k as f64;
            // Checking both extremes also avoids allocating extended storage
            // for an unrepresentable row. Such rows retain late errors and
            // zero-input success, and need no coefficient-row scratch.
            let frame = radial::<S>(log).ok().map(|r| r.exponent().max(0) + 2);
            if !small {
                *row = Row::Built(frame);
                continue;
            }
            let Some(frame) = frame else { continue };
            *row = Row::Normalized(frame);
            for (j, &column) in columns.iter().enumerate() {
                if j % 256 == 0 {
                    context.check()?;
                }
                let Ok(value) = coefficient(c, column, j, k) else {
                    *row = Row::Built(None);
                    break;
                };
                let index = k * n + j;
                if matches!(row, Row::Normalized(_)) {
                    if let (Some(re), Some(im)) = (
                        normal_coefficient(value.real(), frame),
                        normal_coefficient(value.imaginary(), frame),
                    ) {
                        coefficients[index] = Complex::new(re, im);
                        continue;
                    }
                    if exponents.is_empty() {
                        // Admission precedes the lazy n*m*16-byte allocation;
                        // re-admit actual capacity, never silently drop caching.
                        admit(
                            columns.capacity(),
                            row_capacity,
                            coefficients.capacity(),
                            cached,
                        )?;
                        exponents =
                            try_filled(cached, [0; 2], scioxide_core::quantity!("czt_exponents"))?;
                        admit(
                            columns.capacity(),
                            row_capacity,
                            coefficients.capacity(),
                            exponents.capacity(),
                        )?;
                    }
                    for index in k * n..index {
                        let previous = ScaledValue::from_parts(coefficients[index], frame)?;
                        store_extended(previous, &mut coefficients[index], &mut exponents[index]);
                    }
                    *row = Row::Extended;
                }
                store_extended(value, &mut coefficients[index], &mut exponents[index]);
            }
        }
        Ok(Self {
            coefficient_minimum: coefficients
                .iter()
                .flat_map(|z| [z.re.abs(), z.im.abs()])
                .filter(|x| !x.is_zero())
                .fold(S::INFINITY, |a, b| a.min(b)),
            columns,
            rows,
            exponents,
            coefficients,
            ar,
            dr,
            accuracy: DirectAccuracy::new(spec.n)?,
        })
    }

    /// Complex direct scratch beyond the candidate output: the normalized
    /// input, plus one coefficient row for uncached plans.
    pub(super) fn work_len(&self, n: usize) -> Result<usize, SciOxideError> {
        checked_mul(
            n,
            if self.coefficients.is_empty() { 2 } else { 1 },
            scioxide_core::quantity!("czt_direct_scratch"),
        )
    }

    /// Add the stored direct-summation capacities to `bytes`.
    pub(super) fn retained_bytes(&self, bytes: usize) -> Result<usize, SciOxideError> {
        [
            (self.columns.capacity(), size_of::<Column<S>>()),
            (self.coefficients.capacity(), size_of::<Complex<S>>()),
            (self.rows.capacity(), size_of::<Row>()),
            (self.exponents.capacity(), size_of::<[i64; 2]>()),
        ]
        .into_iter()
        .try_fold(bytes, |bytes, (capacity, width)| {
            checked_add(
                bytes,
                checked_mul(
                    capacity,
                    width,
                    scioxide_core::quantity!("czt_vector_bytes"),
                )?,
                scioxide_core::quantity!("czt_plan_bytes"),
            )
        })
    }
}

impl<S: RealScalar> CztPlan<S> {
    /// Evaluate nonzero validated input of maximum component `magnitude` into
    /// `candidate`, row by row, using the working-precision tier where its
    /// error bound holds.
    pub(super) fn evaluate_direct(
        &self,
        direct: &Direct<S>,
        input: &impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        magnitude: S,
        candidate: &mut [Complex<S>],
        work: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (bounded, coefficients) = work.split_at_mut(self.spec.n);
        let input_exponent = crate::scaling::range_exponent(magnitude);
        let input_scale = crate::scaling::BinaryScale::new(-input_exponent);
        let mut input_minimum = S::INFINITY;
        let mut exact_input = direct.accuracy.is_some();
        if exact_input {
            for (j, out) in bounded.iter_mut().enumerate() {
                let x = input(j)?;
                *out = Complex::new(input_scale.apply(x.re), input_scale.apply(x.im));
                if (!x.re.is_zero() && !out.re.is_normal())
                    || (!x.im.is_zero() && !out.im.is_normal())
                {
                    exact_input = false;
                    break;
                }
                for x in [out.re, out.im] {
                    if !x.is_zero() {
                        input_minimum = input_minimum.min(x.abs());
                    }
                }
            }
        }
        for (k, out) in candidate.iter_mut().enumerate() {
            context.check()?;
            let (ordinary, built) = if exact_input {
                self.ordinary_row(
                    direct,
                    bounded,
                    coefficients,
                    k,
                    (input_exponent, input_minimum),
                    context,
                )?
            } else {
                (None, None)
            };
            *out = match ordinary {
                Some(value) => value,
                None => self.scaled_row_impl::<true>(
                    direct,
                    input,
                    k,
                    built.map(|e| (&*coefficients, e)),
                    context,
                )?,
            };
        }
        Ok(())
    }

    #[cfg(test)]
    fn direct_coefficient(
        &self,
        direct: &Direct<S>,
        j: usize,
        k: usize,
    ) -> Result<ScaledValue<Complex<S>>, SciOxideError> {
        match direct.rows[k] {
            Row::Normalized(exponent) => {
                ScaledValue::from_parts(direct.coefficients[k * self.spec.n + j], exponent)
            }
            Row::Extended => ScaledValue::from_component_parts(
                direct.coefficients[k * self.spec.n + j],
                direct.exponents[k * self.spec.n + j],
            ),
            Row::Built(_) => coefficient(self.spec.contour, direct.columns[j], j, k),
        }
    }

    #[cfg(test)]
    fn scaled_row(
        &self,
        direct: &Direct<S>,
        input: &impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        k: usize,
        context: ExecutionContext<'_>,
    ) -> Result<Complex<S>, SciOxideError> {
        self.scaled_row_impl::<true>(direct, input, k, None, context)
    }

    fn scaled_row_impl<const SKIP: bool>(
        &self,
        direct: &Direct<S>,
        input: &impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        k: usize,
        built: Option<(&[Complex<S>], i64)>,
        context: ExecutionContext<'_>,
    ) -> Result<Complex<S>, SciOxideError> {
        let mut re = ScaledAccumulator::ZERO;
        let mut im = ScaledAccumulator::ZERO;
        let decreasing = direct.ar + direct.dr * k as f64 >= 0.0;
        for t in 0..self.spec.n {
            if t % 256 == 0 {
                context.check()?;
            }
            let j = if decreasing { t } else { self.spec.n - 1 - t };
            let x = input(j)?;
            if x.re.is_zero() && x.im.is_zero() {
                continue;
            }
            let input_exponent =
                i64::from(crate::scaling::range_exponent(x.re.abs().max(x.im.abs()))) - 1;
            let column = direct.columns[j];
            let normalized = match direct.rows[k] {
                Row::Normalized(e) => Some((direct.coefficients[k * self.spec.n + j], e)),
                _ => built.map(|(row, e)| (row[j], e)),
            };
            // Bound cached terms directly from their stored exponents before
            // reconstructing components. Absorbed terms need no scaled values.
            let (cached, radius, coefficient_bound) = if let Some((value, exponent)) = normalized {
                let e = i64::from(crate::scaling::range_exponent(
                    value.re.abs().max(value.im.abs()),
                ));
                (Some((value, [exponent; 2])), Scaled::ZERO, exponent + e)
            } else if matches!(direct.rows[k], Row::Extended) {
                let index = k * self.spec.n + j;
                let value = direct.coefficients[index];
                let exponents = direct.exponents[index];
                let e = if value.re.is_zero() {
                    exponents[1]
                } else if value.im.is_zero() {
                    exponents[0]
                } else {
                    exponents[0].max(exponents[1])
                };
                (Some((value, exponents)), Scaled::ZERO, e + 1)
            } else {
                let radius = radial::<S>(column.radial_start - column.radial_step * k as f64)?;
                // Built angles multiply two unit phases: allow two orders over
                // the normalized radius exponent, not radial's log floor.
                (None, radius, radius.exponent() + 2)
            };
            let bound = input_exponent + coefficient_bound;
            let skip_re = SKIP && re.absorbs(bound);
            let skip_im = SKIP && im.absorbs(bound);
            if skip_re && skip_im {
                continue;
            }
            let c = match cached {
                Some((value, exponents)) => ScaledValue::from_component_parts(value, exponents)?,
                None => coefficient_angle(self.spec.contour, column.start, j, k, radius)?,
            };
            let x = ScaledValue::from_value(x)?;
            let (xr, xi, cr, ci) = (x.real(), x.imaginary(), c.real(), c.imaginary());
            let add = |sum: &mut ScaledAccumulator<S>, a: Scaled<S>, b: Scaled<S>| {
                if a.is_zero() || b.is_zero() {
                    Ok(())
                } else {
                    sum.add_product(a, b)
                }
            };
            // Each component can absorb independently: a permanently zero
            // imaginary sum must not prevent exact skips in the real sum.
            if !skip_re {
                add(&mut re, xr, cr)?;
                add(&mut re, core::ops::Neg::neg(xi), ci)?;
            }
            if !skip_im {
                add(&mut im, xr, ci)?;
                add(&mut im, xi, cr)?;
            }
        }
        let scale = Scaled::from_value(self.scale)?;
        Ok(Complex::new(
            re.total()?.mul(scale)?.round()?,
            im.total()?.mul(scale)?.round()?,
        ))
    }

    fn ordinary_row(
        &self,
        direct: &Direct<S>,
        input: &[Complex<S>],
        coefficients: &mut [Complex<S>],
        k: usize,
        (input_exponent, input_minimum): (i32, S),
        context: ExecutionContext<'_>,
    ) -> Result<(Option<Complex<S>>, Option<i64>), SciOxideError> {
        let (Row::Normalized(row_exponent) | Row::Built(Some(row_exponent))) = direct.rows[k]
        else {
            return Ok((None, None));
        };
        let Some(accuracy) = direct.accuracy else {
            return Ok((None, None));
        };
        let coefficients = if direct.coefficients.is_empty() {
            for (j, (x, coefficient)) in input.iter().zip(coefficients.iter_mut()).enumerate() {
                if j % 256 == 0 {
                    context.check()?;
                }
                if x.re.is_zero() && x.im.is_zero() {
                    *coefficient = Complex::new(S::ZERO, S::ZERO);
                    continue;
                }
                let c = self::coefficient(self.spec.contour, direct.columns[j], j, k)?;
                let (Some(re), Some(im)) = (
                    normal_coefficient(c.real(), row_exponent),
                    normal_coefficient(c.imaginary(), row_exponent),
                ) else {
                    return Ok((None, None));
                };
                *coefficient = Complex::new(re, im);
            }
            &*coefficients
        } else {
            &direct.coefficients[k * self.spec.n..(k + 1) * self.spec.n]
        };
        let built = direct.coefficients.is_empty().then_some(row_exponent);
        let normal_products = !direct.coefficients.is_empty()
            && (input_minimum * direct.coefficient_minimum).is_normal();
        let result = if normal_products {
            ordinary_dot::<S, false>(input, coefficients, context)?
        } else {
            ordinary_dot::<S, true>(input, coefficients, context)?
        };
        let Some([total, mass]) = result else {
            return Ok((None, built));
        };
        let (Some(re_error), Some(im_error)) = (
            accuracy.admits(total.re, mass.re),
            accuracy.admits(total.im, mass.im),
        ) else {
            return Ok((None, built));
        };
        let exponent = row_exponent + i64::from(input_exponent);
        let binary_restore = i32::try_from(exponent)
            .ok()
            .map(crate::scaling::BinaryScale::new);
        let restore = |value: S, error: S| -> Result<Option<S>, SciOxideError> {
            if value.is_zero() && error.is_zero() {
                return Ok(Some(value));
            }
            if let Some(restore) = binary_restore {
                let center = value * self.scale;
                let bound = ((value.abs() + error).next_after(S::INFINITY) * self.scale)
                    .next_after(S::INFINITY);
                if (value.is_zero() || center.is_normal())
                    && bound.is_normal()
                    && restore.apply(bound).is_finite()
                {
                    return Ok(Some(restore.apply(center)));
                }
            }
            let error = Scaled::from_value(error)?;
            let scale = Scaled::from_value(self.scale)?;
            let center = Scaled::from_value(value)?;
            // If either uncertainty endpoint can overflow, the extended path
            // decides representability. A speculative row never raises a
            // spurious range error or publishes a partially restored result.
            for endpoint in [center.sub(error)?, center.add(error)?] {
                if endpoint.shifted(exponent)?.mul(scale)?.round().is_err() {
                    return Ok(None);
                }
            }
            Ok(Some(center.shifted(exponent)?.mul(scale)?.round()?))
        };
        let (Some(re), Some(im)) = (restore(total.re, re_error)?, restore(total.im, im_error)?)
        else {
            return Ok((None, built));
        };
        Ok((Some(Complex::new(re, im)), built))
    }
}

// Keep the admitted kernel free of per-product classification. The checked
// specialization handles rows whose minimum-magnitude proof is inconclusive.
fn ordinary_dot<S: RealScalar, const CHECK_PRODUCTS: bool>(
    input: &[Complex<S>],
    coefficients: &[Complex<S>],
    context: ExecutionContext<'_>,
) -> Result<Option<[Complex<S>; 2]>, SciOxideError> {
    let zero = Complex::new(S::ZERO, S::ZERO);
    let mut sums = [zero; 4];
    let mut corrections = [zero; 4];
    let mut masses = [zero; 4];
    // The fixed-size block must stay in the loop: an out-of-line closure
    // spills every partial sum and correction for each four coefficients.
    #[inline(always)]
    fn accumulate<S: RealScalar, const CHECK_PRODUCTS: bool>(
        sums: &mut [Complex<S>; 4],
        corrections: &mut [Complex<S>; 4],
        masses: &mut [Complex<S>; 4],
        xs: &[Complex<S>; 4],
        cs: &[Complex<S>; 4],
    ) -> Option<()> {
        for ((((sum, correction), mass), x), c) in sums
            .iter_mut()
            .zip(corrections.iter_mut())
            .zip(masses.iter_mut())
            .zip(xs)
            .zip(cs)
        {
            let products = [x.re * c.re, x.im * c.im, x.re * c.im, x.im * c.re];
            let operands = [(x.re, c.re), (x.im, c.im), (x.re, c.im), (x.im, c.re)];
            if CHECK_PRODUCTS
                && products
                    .iter()
                    .zip(operands)
                    .any(|(p, (a, b))| !a.is_zero() && !b.is_zero() && !p.is_normal())
            {
                return None;
            }
            mass.re = mass.re + products[0].abs() + products[1].abs();
            mass.im = mass.im + products[2].abs() + products[3].abs();
            compensated_term(&mut sum.re, &mut correction.re, products[0] - products[1]);
            compensated_term(&mut sum.im, &mut correction.im, products[2] + products[3]);
        }
        Some(())
    }
    let (xs, input_tail) = input.as_chunks::<4>();
    let (cs, coefficient_tail) = coefficients.as_chunks::<4>();
    for (chunk, (xs, cs)) in xs.iter().zip(cs).enumerate() {
        if chunk % 64 == 0 {
            context.check()?;
        }
        if accumulate::<S, CHECK_PRODUCTS>(&mut sums, &mut corrections, &mut masses, xs, cs)
            .is_none()
        {
            return Ok(None);
        }
    }
    if !input_tail.is_empty() {
        let (mut xs, mut cs) = ([zero; 4], [zero; 4]);
        xs[..input_tail.len()].copy_from_slice(input_tail);
        cs[..coefficient_tail.len()].copy_from_slice(coefficient_tail);
        if accumulate::<S, CHECK_PRODUCTS>(&mut sums, &mut corrections, &mut masses, &xs, &cs)
            .is_none()
        {
            return Ok(None);
        }
    }
    // Merge both unrounded parts, never round a lane's total separately.
    let (mut sum, mut correction, mut mass) = (zero, zero, zero);
    for ((s, c), a) in sums.into_iter().zip(corrections).zip(masses) {
        for z in [s, c] {
            compensated_term(&mut sum.re, &mut correction.re, z.re);
            compensated_term(&mut sum.im, &mut correction.im, z.im);
        }
        mass += a;
    }
    let total = sum + correction;
    Ok(Some([total, mass]))
}

fn coefficient<S: RealScalar>(
    contour: CztContour<S>,
    column: Column<S>,
    j: usize,
    k: usize,
) -> Result<ScaledValue<Complex<S>>, SciOxideError> {
    coefficient_angle(
        contour,
        column.start,
        j,
        k,
        radial::<S>(column.radial_start - column.radial_step * k as f64)?,
    )
}

fn coefficient_angle<S: RealScalar>(
    contour: CztContour<S>,
    start: Complex<S>,
    j: usize,
    k: usize,
    radius: Scaled<S>,
) -> Result<ScaledValue<Complex<S>>, SciOxideError> {
    let angle = start
        * contour.step_phase(
            (j as u128) * (k as u128),
            false,
            true,
            scioxide_core::argument!("self"),
        )?;
    Ok(ScaledValue::from_components(
        radius.mul(Scaled::from_value(angle.re)?)?,
        radius.mul(Scaled::from_value(angle.im)?)?,
    ))
}

fn normal_coefficient<S: RealScalar>(value: Scaled<S>, exponent: i64) -> Option<S> {
    if value.is_zero() {
        return Some(value.significand());
    }
    let shift = value.exponent() - exponent;
    let minimum = i64::from(crate::scaling::range_exponent(S::MIN_POSITIVE) - 1);
    if shift < minimum || shift > -1 {
        return None;
    }
    Some(value.significand().scalbn(shift as i32))
}

fn store_extended<S: RealScalar>(
    value: ScaledValue<Complex<S>>,
    coefficient: &mut Complex<S>,
    exponents: &mut [i64; 2],
) {
    *coefficient = Complex::new(value.real().significand(), value.imaginary().significand());
    *exponents = [value.real().exponent(), value.imaginary().exponent()];
}

// Inline metadata and the exact retained capacities; preparation has no
// temporary vectors, even when a normalized row becomes extended.
fn preparation_bytes<S: RealScalar>(
    n: usize,
    m: usize,
    cached: usize,
    extended: usize,
) -> Result<usize, SciOxideError> {
    [
        (n, size_of::<Column<S>>()),
        (m, size_of::<Row>()),
        (cached, size_of::<Complex<S>>()),
        (extended, size_of::<[i64; 2]>()),
    ]
    .into_iter()
    .try_fold(size_of::<CztPlan<S>>(), |bytes, (capacity, width)| {
        checked_add(
            bytes,
            checked_mul(
                capacity,
                width,
                scioxide_core::quantity!("czt_vector_bytes"),
            )?,
            scioxide_core::quantity!("czt_preparation_bytes"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::super::{CztAlgorithm, Preparation};
    use super::*;
    use crate::Normalization;

    fn direct<S: RealScalar>(plan: &CztPlan<S>) -> &Direct<S> {
        let Preparation::Direct(direct) = &plan.preparation else {
            unreachable!("direct test plan")
        };
        direct
    }

    fn prepared<S: RealScalar>(n: usize, ar: f64, dr: f64) -> CztPlan<S> {
        CztPlan::new(CztPlanSpec::new(
            n,
            32,
            CztContour::new(
                S::try_from_f64_lossy(ar).unwrap(),
                S::try_from_f64_lossy(dr).unwrap(),
                S::ZERO,
                S::ZERO,
            )
            .unwrap(),
            Normalization::Unscaled,
        ))
        .unwrap()
    }

    fn ordinary<S: RealScalar>(
        plan: &CztPlan<S>,
        input: &[Complex<S>],
        k: usize,
    ) -> Option<Complex<S>> {
        let magnitude = crate::scaling::complex_max(input.iter().copied()).unwrap();
        let exponent = crate::scaling::range_exponent(magnitude);
        let bounded: Vec<_> = input
            .iter()
            .map(|x| Complex::new(x.re.scalbn(-exponent), x.im.scalbn(-exponent)))
            .collect();
        plan.ordinary_row(
            direct(plan),
            &bounded,
            &mut alloc::vec![Complex::new(S::ZERO, S::ZERO); input.len()],
            k,
            (
                exponent,
                bounded
                    .iter()
                    .flat_map(|z| [z.re.abs(), z.im.abs()])
                    .filter(|x| !x.is_zero())
                    .fold(S::INFINITY, |a, b| a.min(b)),
            ),
            ExecutionContext::sequential(),
        )
        .unwrap()
        .0
    }

    fn admitted_and_fallback<S: RealScalar>() {
        let input = alloc::vec![Complex::new(S::ONE, S::ONE / (S::ONE + S::ONE)); 17];
        for (ar, dr) in [(0.25, 0.0), (-0.25, 0.0), (0.0, -0.999_f64.ln())] {
            let plan = prepared::<S>(17, ar, dr);
            assert_eq!(plan.algorithm(), CztAlgorithm::ScaledDirect);
            for k in 0..4 {
                let actual = ordinary(&plan, &input, k).unwrap();
                let expected = plan
                    .scaled_row(
                        direct(&plan),
                        &|j| Ok(input[j]),
                        k,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                let accuracy = direct(&plan).accuracy.unwrap();
                for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
                    assert!(
                        (a - e).abs()
                            <= accuracy.relative * e.abs() + S::MIN_SUBNORMAL + S::MIN_SUBNORMAL
                    );
                }
            }
        }
        for ar in [-800.0, 800.0] {
            assert!(ordinary(&prepared::<S>(17, ar, 0.0), &input, 0).is_none());
        }
        // Opposing first and second terms cancel in the stored coefficient
        // system; only an insignificant low term remains.
        let plan = prepared::<S>(3, 1.0, 0.0);
        let d = direct(&plan);
        let c = plan
            .direct_coefficient(d, 1, 0)
            .unwrap()
            .real()
            .round()
            .unwrap();
        let cancelling = [
            Complex::new(c, S::ZERO),
            Complex::new(-S::ONE, S::ZERO),
            Complex::new(S::EPSILON * S::EPSILON, S::ZERO),
        ];
        assert!(ordinary(&plan, &cancelling, 0).is_none());
        assert!(
            plan.execute_owned((&cancelling).into()).unwrap()[0]
                .re
                .abs()
                > S::ZERO
        );
    }

    #[test]
    fn positive_bounded_rows_use_the_ordinary_tier() {
        let n = 64;
        let plan = CztPlan::new(CztPlanSpec::new(
            n,
            n,
            CztContour::new(0.75, 0.75 / (n * n) as f64, 0.0, 0.0).unwrap(),
            Normalization::Unscaled,
        ))
        .unwrap();
        let input = alloc::vec![Complex::new(0.75, 0.5); n];
        for k in 0..plan.spec.m {
            assert!(ordinary(&plan, &input, k).is_some());
        }
    }

    #[test]
    fn admitted_and_fallback_f64() {
        admitted_and_fallback::<f64>();
    }
    #[test]
    fn admitted_and_fallback_f32() {
        admitted_and_fallback::<f32>();
    }

    fn boundaries<S: RealScalar>() {
        let accuracy = DirectAccuracy::<S>::new(17).unwrap().unwrap();
        let mass = S::ONE;
        let threshold = accuracy.error_factor / (accuracy.relative - accuracy.result_factor);
        assert!(
            accuracy
                .admits(threshold * S::try_from_f64_lossy(1.001).unwrap(), mass)
                .is_some()
        );
        assert!(
            accuracy
                .admits(threshold * S::try_from_f64_lossy(0.999).unwrap(), mass)
                .is_none()
        );
        assert!(accuracy.admits(S::ZERO, mass).is_none());
        assert!(accuracy.admits(S::ZERO, S::ZERO).is_some());
        // Locate the transition on the scalar grid, then exercise its two
        // adjacent values rather than merely testing distant tolerances.
        let mut admitted = threshold;
        for _ in 0..8 {
            if accuracy.admits(admitted, mass).is_some() {
                break;
            }
            admitted = admitted.next_after(S::INFINITY);
        }
        assert!(accuracy.admits(admitted, mass).is_some());
        for _ in 0..8 {
            let previous = admitted.next_after(S::ZERO);
            if accuracy.admits(previous, mass).is_none() {
                break;
            }
            admitted = previous;
        }
        assert!(
            accuracy
                .admits(admitted.next_after(S::ZERO), mass)
                .is_none()
        );
        let minimum = i64::from(crate::scaling::range_exponent(S::MIN_POSITIVE) - 1);
        assert_eq!(
            normal_coefficient(Scaled::<S>::ONE, -minimum),
            Some(S::MIN_POSITIVE)
        );
        assert!(normal_coefficient(Scaled::<S>::ONE, 1 - minimum).is_none());
        // Both operands can be normal while their product enters the
        // subnormal grid; such a row must also retain scaled arithmetic.
        let plan = prepared::<S>(2, 1.0, 0.0);
        let bounded = [
            Complex::new(S::ONE / (S::ONE + S::ONE), S::ZERO),
            Complex::new(S::MIN_POSITIVE, S::ZERO),
        ];
        assert!(
            plan.ordinary_row(
                direct(&plan),
                &bounded,
                &mut [Complex::new(S::ZERO, S::ZERO); 2],
                0,
                (0, S::MIN_POSITIVE),
                ExecutionContext::sequential()
            )
            .unwrap()
            .0
            .is_none()
        );
        let maximum_n = (1.0 / (128.0 * 16.0 * (S::EPSILON.into_f64() / 2.0)) - 1.0) as usize;
        assert!(DirectAccuracy::<S>::new(maximum_n).unwrap().is_some());
        if maximum_n < usize::MAX {
            assert!(DirectAccuracy::<S>::new(maximum_n + 1).unwrap().is_none());
        }
    }
    #[test]
    fn admission_boundaries_f64() {
        boundaries::<f64>();
    }
    #[test]
    fn admission_boundaries_f32() {
        boundaries::<f32>();
    }

    fn exact_skips<S: RealScalar>() {
        let n = 64;
        let cx = ExecutionContext::sequential();
        for ar in [0.75, 40.0, 800.0, -0.75, -2.0] {
            for phase in [0.0, 0.03125] {
                let s = |v| S::try_from_f64_lossy(v).unwrap();
                let plan = CztPlan::new(CztPlanSpec::new(
                    n,
                    n,
                    CztContour::new(s(ar), s(0.75 / (n * n) as f64), s(phase), s(phase)).unwrap(),
                    Normalization::Unscaled,
                ))
                .unwrap();
                let d = direct(&plan);
                // Small inputs keep the growing rows representable in f32.
                for imaginary in [false, true] {
                    let input: Vec<_> = (0..n)
                        .map(|j| {
                            let value = s(0.5 + (j % 7) as f64 / 16.0).scalbn(if ar < -1.0 {
                                -110
                            } else {
                                0
                            });
                            Complex::new(value, if imaginary { value * s(0.75) } else { S::ZERO })
                        })
                        .collect();
                    for k in [0, 7, 63] {
                        let a = plan
                            .scaled_row_impl::<true>(d, &|j| Ok(input[j]), k, None, cx)
                            .unwrap();
                        let b = plan
                            .scaled_row_impl::<false>(d, &|j| Ok(input[j]), k, None, cx)
                            .unwrap();
                        for (a, b) in [(a.re, b.re), (a.im, b.im)] {
                            assert_eq!(a.decompose(), b.decompose());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn absorbed_terms_match_no_skip_bitwise() {
        exact_skips::<f32>();
        exact_skips::<f64>();
    }

    #[test]
    fn uncached_admission_failure_retains_a_reusable_row() {
        fn check<S: RealScalar>() {
            let cx = ExecutionContext::sequential();
            let plan = prepared::<S>(3, 1.0, 0.0);
            let d = Direct::prepare(&plan.spec, false, cx).unwrap();
            let c = plan
                .direct_coefficient(&d, 1, 0)
                .unwrap()
                .real()
                .round()
                .unwrap();
            let input = [
                Complex::new(c, S::ZERO),
                Complex::new(-S::ONE, S::ZERO),
                Complex::new(S::EPSILON * S::EPSILON, S::ZERO),
            ];
            let bounded = input.map(|v| Complex::new(v.re * S::HALF, v.im));
            let mut row = [Complex::new(S::ZERO, S::ZERO); 3];
            let (value, exponent) = plan
                .ordinary_row(&d, &bounded, &mut row, 0, (1, bounded[2].re), cx)
                .unwrap();
            assert!(value.is_none());
            let exponent = exponent.unwrap();
            let reused = plan
                .scaled_row_impl::<true>(&d, &|j| Ok(input[j]), 0, Some((&row, exponent)), cx)
                .unwrap();
            let rebuilt = plan
                .scaled_row_impl::<false>(&d, &|j| Ok(input[j]), 0, None, cx)
                .unwrap();
            assert_eq!(reused.re.decompose(), rebuilt.re.decompose());
            assert_eq!(reused.im.decompose(), rebuilt.im.decompose());
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn uncertainty_near_maximum_selects_scaled_range_decision() {
        let plan = prepared::<f64>(2, 1.0, 0.0);
        let input = [Complex::new(f64::MAX, 0.0), Complex::new(0.0, 0.0)];
        assert!(ordinary(&plan, &input, 0).is_none());
        assert_eq!(
            plan.execute_owned((&input).into()).unwrap(),
            alloc::vec![input[0]; 32]
        );
    }
}
