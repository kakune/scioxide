//! Degree-sized native continuous extensions with a construction-selected scaled fallback.

use crate::safety::checked_step_endpoint;
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::backend::{checked_mul, try_reserve};
use scioxide_core::{
    Dimensions, InvalidInputReason, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_linalg::Vector;

/// Continuous extension selected by the stepping method.
///
/// Order `p` means an exact-start interior value error O(h^(p+1)) for a
/// sufficiently smooth problem, before rounding dominates. It is not a bound
/// on accumulated trajectory error or an error certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseExtension {
    /// History interpolation in normalized powers; degree and exact-start order.
    Multistep {
        /// Polynomial degree and exact-start value order.
        order: usize,
    },
    /// Reduced-order endpoint Hermite interpolation used by Rosenbrock, RKF45, Verner65 and Adams startup.
    CubicHermite,
    /// Cubic collocation polynomial of the order-five Radau IIA method.
    RadauIia,
    /// Shampine's quartic extension of Dormand-Prince 5.
    DormandPrince5,
    /// Seventh-order extension of the Dormand–Prince 8(5,3) method (Hairer, Nørsett
    /// and Wanner I, §II.6).
    DormandPrince853,
}

impl DenseExtension {
    /// Polynomial degree of this extension.
    pub(crate) const fn degree(self) -> usize {
        match self {
            Self::Multistep { order } => order,
            Self::CubicHermite | Self::RadauIia => 3,
            Self::DormandPrince5 => 4,
            Self::DormandPrince853 => 7,
        }
    }
}

/// Native coefficients, or scaled significands with parallel exponents. Both
/// buffers are reserved for the method degree before stepping. Real states have
/// one plane per component, complex states two; a query visits contiguous lanes.
#[derive(Debug)]
pub(crate) struct DenseOutputStorage<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S> = S,
> {
    coefficients: Vec<S>,
    exponents: Vec<i64>,
    maximum_degree: usize,
    initial: Vector<T>,
    endpoint: Vector<T>,
    interval: Option<(S, S, S)>,
    extension: DenseExtension,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Clone
    for DenseOutputStorage<S, T>
{
    fn clone(&self) -> Self {
        // Vec::clone discards spare capacity. A cloned prepared workspace must
        // retain room for a later representation or active-order transition.
        let count = self.initial.len() * Self::parts() * self.maximum_degree;
        let mut coefficients = Vec::with_capacity(count);
        coefficients.extend_from_slice(&self.coefficients);
        let mut exponents = Vec::with_capacity(count);
        exponents.extend_from_slice(&self.exponents);
        Self {
            coefficients,
            exponents,
            maximum_degree: self.maximum_degree,
            initial: self.initial.clone(),
            endpoint: self.endpoint.clone(),
            interval: self.interval,
            extension: self.extension,
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> DenseOutputStorage<S, T> {
    pub(crate) const fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            exponents: Vec::new(),
            maximum_degree: 0,
            initial: Vector::empty(),
            endpoint: Vector::empty(),
            interval: None,
            extension: DenseExtension::CubicHermite,
        }
    }

    const fn parts() -> usize {
        if T::IS_REAL { 1 } else { 2 }
    }

    pub(crate) fn fits(&self, dimension: usize, degree: usize) -> bool {
        self.initial.len() == dimension && self.maximum_degree == degree
    }

    pub(crate) fn try_resize(
        &mut self,
        dimension: usize,
        degree: usize,
    ) -> Result<(), SciOxideError> {
        let count = checked_mul(
            checked_mul(
                dimension,
                Self::parts(),
                scioxide_core::quantity!("dense_output.storage"),
            )?,
            degree,
            scioxide_core::quantity!("dense_output.storage"),
        )?;
        if !self.fits(dimension, degree) {
            self.interval = None;
        }
        if count > self.coefficients.capacity() {
            let additional = count - self.coefficients.len();
            try_reserve(
                &mut self.coefficients,
                additional,
                scioxide_core::quantity!("dense_output.coefficients"),
            )?;
        }
        if count > self.exponents.capacity() {
            let additional = count - self.exponents.len();
            try_reserve(
                &mut self.exponents,
                additional,
                scioxide_core::quantity!("dense_output.exponents"),
            )?;
        }
        self.initial.try_resize(dimension, T::ZERO)?;
        self.endpoint.try_resize(dimension, T::ZERO)?;
        self.maximum_degree = degree;
        Ok(())
    }

    pub(crate) fn copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        if self.dimension() != source.dimension() || self.maximum_degree < source.extension.degree()
        {
            return Err(crate::safety::invalid(
                scioxide_core::argument!("self"),
                "dense segment storage does not match source",
            ));
        }
        self.coefficients.resize(source.coefficients.len(), S::ZERO);
        self.coefficients.copy_from_slice(&source.coefficients);
        self.exponents.resize(source.exponents.len(), 0);
        self.exponents.copy_from_slice(&source.exponents);
        self.initial
            .as_mut_slice()
            .copy_from_slice(source.initial.as_slice());
        self.endpoint
            .as_mut_slice()
            .copy_from_slice(source.endpoint.as_slice());
        self.interval = source.interval;
        self.extension = source.extension;
        Ok(())
    }

    /// Try an entire native segment, then replay original operands in scaled
    /// arithmetic if any component loses range. Neither closure writes state.
    #[inline]
    pub(crate) fn prepare_coefficients(
        &mut self,
        extension: DenseExtension,
        dt: S,
        initial: &Vector<T>,
        native: impl Fn(usize, bool) -> Option<[S; 7]>,
        scaled: impl Fn(usize, bool) -> Result<[Scaled<S>; 7], SciOxideError>,
    ) -> Result<(), SciOxideError> {
        self.interval = None;
        let degree = extension.degree();
        if degree > self.maximum_degree || initial.len() != self.dimension() {
            return Err(crate::safety::invalid(
                scioxide_core::argument!("self"),
                "dense segment storage is not prepared",
            ));
        }
        self.extension = extension;
        self.exponents.clear();
        self.coefficients
            .resize(initial.len() * Self::parts() * degree, S::ZERO);
        let mut ordinary = true;
        'components: for (i, &old) in initial.as_slice().iter().enumerate() {
            for plane in 0..Self::parts() {
                let imaginary = plane == 1;
                let Some(values) = native(i, imaginary) else {
                    ordinary = false;
                    break 'components;
                };
                let initial = component(old, imaginary);
                let admitted = match extension {
                    DenseExtension::DormandPrince5 => admit_native(&values[..4], initial, dt),
                    DenseExtension::DormandPrince853 => admit_native(&values, initial, dt),
                    DenseExtension::Multistep { order } => {
                        admit_native(&values[..order], initial, dt)
                    }
                    _ => admit_native(&values[..3], initial, dt),
                };
                if !admitted {
                    ordinary = false;
                    break 'components;
                }
                let offset = (i * Self::parts() + plane) * degree;
                self.coefficients[offset..offset + degree].copy_from_slice(&values[..degree]);
            }
        }
        if ordinary {
            Ok(())
        } else {
            self.prepare_scaled(scaled)
        }
    }

    #[cold]
    #[inline(never)]
    fn prepare_scaled(
        &mut self,
        scaled: impl Fn(usize, bool) -> Result<[Scaled<S>; 7], SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let degree = self.extension.degree();
        self.exponents.resize(self.coefficients.len(), 0);
        for i in 0..self.dimension() {
            for plane in 0..Self::parts() {
                let values = scaled(i, plane == 1)?;
                let offset = (i * Self::parts() + plane) * degree;
                for (j, value) in values[..degree].iter().enumerate() {
                    self.coefficients[offset + j] = value.significand();
                    self.exponents[offset + j] = value.exponent();
                }
            }
        }
        Ok(())
    }

    /// Hermite and explicit RK extensions share their first three coefficients.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn prepare_rk(
        &mut self,
        extension: DenseExtension,
        dt: S,
        initial: &Vector<T>,
        endpoint: &Vector<T>,
        first: &Vector<T>,
        last: &Vector<T>,
        stages: &[Vector<T>],
        weights: &[impl AsRef<[S]>],
    ) -> Result<(), SciOxideError> {
        self.prepare_coefficients(
            extension,
            dt,
            initial,
            |i, imaginary| {
                let old = component(initial.as_slice()[i], imaginary);
                let new = component(endpoint.as_slice()[i], imaginary);
                let left = native_product(dt, component(first.as_slice()[i], imaginary))?;
                let right = native_product(dt, component(last.as_slice()[i], imaginary))?;
                let delta = new - old;
                let mut values = [S::ZERO; 7];
                values[0] = delta;
                values[1] = left - delta;
                values[2] = (delta - left) + (delta - right);
                for (out, weights) in values[3..].iter_mut().zip(weights) {
                    let sum =
                        crate::safety::native_stage_sum(weights.as_ref().iter().zip(stages).map(
                            |(&weight, stage)| (weight, component(stage.as_slice()[i], imaginary)),
                        ))?;
                    *out = native_product(dt, sum)?;
                }
                Some(values)
            },
            |i, imaginary| {
                let h = Scaled::from_value(dt)?;
                let old = Scaled::from_value(component(initial.as_slice()[i], imaginary))?;
                let new = Scaled::from_value(component(endpoint.as_slice()[i], imaginary))?;
                let left = h.mul(Scaled::from_value(component(
                    first.as_slice()[i],
                    imaginary,
                ))?)?;
                let right = h.mul(Scaled::from_value(component(
                    last.as_slice()[i],
                    imaginary,
                ))?)?;
                let delta = new.sub(old)?;
                let mut values = [Scaled::ZERO; 7];
                values[0] = delta;
                values[1] = left.sub(delta)?;
                values[2] = delta.sub(left)?.add(delta.sub(right)?)?;
                for (out, weights) in values[3..].iter_mut().zip(weights) {
                    *out = crate::safety::stage_sum(weights.as_ref().iter().zip(stages).map(
                        |(&weight, stage)| (weight, component(stage.as_slice()[i], imaginary)),
                    ))?
                    .mul(h)?;
                }
                Ok(values)
            },
        )
    }

    #[inline]
    pub(crate) fn prepare_endpoints(
        &mut self,
        dt: S,
        initial: &Vector<T>,
        endpoint: &Vector<T>,
        first: &Vector<T>,
        last: &Vector<T>,
    ) -> Result<(), SciOxideError> {
        self.prepare_rk(
            DenseExtension::CubicHermite,
            dt,
            initial,
            endpoint,
            first,
            last,
            &[],
            &[] as &[&[S]],
        )
    }

    pub(crate) fn finish(
        &mut self,
        extension: DenseExtension,
        start: S,
        dt: S,
        initial: &Vector<T>,
        endpoint: &Vector<T>,
    ) -> Result<&Self, SciOxideError> {
        let end = checked_step_endpoint(start, dt)?;
        debug_assert_eq!(extension.degree(), self.extension.degree());
        self.initial
            .as_mut_slice()
            .copy_from_slice(initial.as_slice());
        self.endpoint
            .as_mut_slice()
            .copy_from_slice(endpoint.as_slice());
        self.extension = extension;
        self.interval = Some((start, end, dt));
        Ok(self)
    }

    /// Only accepted full endpoints may seed a continuation predictor.
    pub(crate) fn continues(&self, time: S, state: &[T]) -> bool {
        self.interval.is_some_and(|(_, end, _)| end == time) && self.endpoint.as_slice() == state
    }

    /// Evaluate outside the accepted interval for a method-owned predictor. Native
    /// coefficients extrapolate in one native pass; a non-finite component replays the
    /// whole query through the checked scaled path. On error `output` is unspecified.
    pub(crate) fn extrapolate_into(&self, t: S, output: &mut [T]) -> Result<(), SciOxideError> {
        let (start, _, dt) = self.interval.ok_or_else(|| {
            crate::safety::invalid(
                scioxide_core::argument!("self"),
                "missing predictor interval",
            )
        })?;
        if output.len() != self.dimension() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.dimension()),
                Dimensions::one(output.len()),
            ));
        }
        let theta = coordinate(t, start, dt)?;
        if self.exponents.is_empty() {
            self.publish_native_extension::<false>(theta, dt, output);
            if output.iter().all(|value| value.has_finite_components()) {
                return Ok(());
            }
        }
        self.publish_checked(theta, dt, output, false)
    }

    #[inline]
    fn coordinate(&self, t: S) -> Result<(S, S), SciOxideError> {
        let (start, end, dt) = self.interval.ok_or_else(|| {
            crate::safety::invalid(
                scioxide_core::argument!("self"),
                "segment is not initialized",
            )
        })?;
        if !RealScalar::is_finite(t) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "query time must be finite",
            ));
        }
        if t < start.min(end) || t > start.max(end) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("t"),
                InvalidInputReason::OutOfRange,
                "query is outside the completed step",
            ));
        }
        let theta = if t == start {
            S::ZERO
        } else if t == end {
            S::ONE
        } else {
            coordinate(t, start, dt)?
        };
        Ok((theta, dt))
    }

    #[inline]
    fn native_component<const DERIVATIVE: bool, const IMAGINARY: bool>(
        &self,
        index: usize,
        theta: S,
        dt: S,
    ) -> S {
        if !DERIVATIVE {
            if theta == S::ZERO {
                return component(self.initial.as_slice()[index], IMAGINARY);
            }
            if theta == S::ONE {
                return component(self.endpoint.as_slice()[index], IMAGINARY);
            }
        }
        let degree = self.extension.degree();
        let offset = (index * Self::parts() + usize::from(IMAGINARY)) * degree;
        let coefficients = &self.coefficients[offset..offset + degree];
        let value = match self.extension {
            DenseExtension::DormandPrince5 => {
                native_polynomial::<S, DERIVATIVE, false, 4>(coefficients, theta, dt)
            }
            DenseExtension::DormandPrince853 => {
                native_polynomial::<S, DERIVATIVE, false, 7>(coefficients, theta, dt)
            }
            DenseExtension::Multistep { .. } => {
                native_polynomial::<S, DERIVATIVE, true, 0>(coefficients, theta, dt)
            }
            _ => native_polynomial::<S, DERIVATIVE, false, 3>(coefficients, theta, dt),
        };
        if DERIVATIVE {
            value
        } else {
            value + component(self.initial.as_slice()[index], IMAGINARY)
        }
    }

    fn component(
        &self,
        index: usize,
        theta: S,
        dt: S,
        derivative: bool,
        imaginary: bool,
    ) -> Result<S, SciOxideError> {
        if !derivative {
            if theta == S::ZERO {
                return Ok(component(self.initial.as_slice()[index], imaginary));
            }
            if theta == S::ONE {
                return Ok(component(self.endpoint.as_slice()[index], imaginary));
            }
        }
        if self.exponents.is_empty() {
            let value = match (derivative, imaginary) {
                (false, false) => self.native_component::<false, false>(index, theta, dt),
                (false, true) => self.native_component::<false, true>(index, theta, dt),
                (true, false) => self.native_component::<true, false>(index, theta, dt),
                (true, true) => self.native_component::<true, true>(index, theta, dt),
            };
            if value.is_finite() {
                return Ok(value);
            }
        }
        // Extrapolation is outside the native interpolation admission bound.
        self.scaled_component(index, theta, dt, derivative, imaginary)
    }

    #[cold]
    fn scaled_component(
        &self,
        index: usize,
        theta: S,
        dt: S,
        derivative: bool,
        imaginary: bool,
    ) -> Result<S, SciOxideError> {
        let degree = self.extension.degree();
        let offset = (index * Self::parts() + usize::from(imaginary)) * degree;
        let mut value = Scaled::ZERO;
        let mut slope = Scaled::ZERO;
        for j in (0..degree).rev() {
            let positive = matches!(self.extension, DenseExtension::Multistep { .. }) || j % 2 == 0;
            let factor = Scaled::from_value(if positive { theta } else { S::ONE - theta })?;
            let coefficient = Scaled::from_parts(
                self.coefficients[offset + j],
                self.exponents.get(offset + j).copied().unwrap_or(0),
            )?;
            let sum = value.add(coefficient)?;
            if derivative {
                slope = slope.mul(factor)?.add(if positive { sum } else { -sum })?;
            }
            value = sum.mul(factor)?;
        }
        if derivative {
            slope.div(Scaled::from_value(dt)?)?.round()
        } else {
            value
                .add(Scaled::from_value(component(
                    self.initial.as_slice()[index],
                    imaginary,
                ))?)?
                .round()
        }
    }

    #[inline]
    fn publish(
        &self,
        theta: S,
        dt: S,
        output: &mut [T],
        derivative: bool,
    ) -> Result<(), SciOxideError> {
        if output.len() != self.dimension() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.dimension()),
                Dimensions::one(output.len()),
            ));
        }
        if self.exponents.is_empty() && theta >= S::ZERO && theta <= S::ONE {
            // Construction bounds leave a fourfold range margin. Interpolation
            // cannot fail numerically, so publication needs no speculative pass.
            if derivative {
                self.publish_native_extension::<true>(theta, dt, output);
            } else {
                self.publish_native_extension::<false>(theta, dt, output);
            }
            return Ok(());
        }
        self.publish_checked(theta, dt, output, derivative)
    }

    #[inline]
    fn publish_native_extension<const DERIVATIVE: bool>(&self, theta: S, dt: S, output: &mut [T]) {
        match self.extension {
            DenseExtension::DormandPrince5 => {
                self.publish_native::<DERIVATIVE, false, 4>(theta, dt, output)
            }
            DenseExtension::DormandPrince853 => {
                self.publish_native::<DERIVATIVE, false, 7>(theta, dt, output)
            }
            DenseExtension::Multistep { .. } => {
                self.publish_native::<DERIVATIVE, true, 0>(theta, dt, output)
            }
            _ => self.publish_native::<DERIVATIVE, false, 3>(theta, dt, output),
        }
    }

    /// Select the basis once per query; stream contiguous component coefficients
    /// without repeated method dispatch or imaginary work for real states.
    #[inline]
    fn publish_native<const DERIVATIVE: bool, const POWERS: bool, const N: usize>(
        &self,
        theta: S,
        dt: S,
        output: &mut [T],
    ) {
        if !DERIVATIVE {
            if theta == S::ZERO {
                output.copy_from_slice(self.initial.as_slice());
                return;
            }
            if theta == S::ONE {
                output.copy_from_slice(self.endpoint.as_slice());
                return;
            }
        }
        let degree = if N == 0 { self.extension.degree() } else { N };
        for ((out, &initial), coefficients) in output
            .iter_mut()
            .zip(self.initial.as_slice())
            .zip(self.coefficients.chunks_exact(degree * Self::parts()))
        {
            let real =
                native_polynomial::<S, DERIVATIVE, POWERS, N>(&coefficients[..degree], theta, dt);
            let real = if DERIVATIVE {
                real
            } else {
                real + initial.real_component()
            };
            *out = if T::IS_REAL {
                T::from_real(real)
            } else {
                let imaginary = native_polynomial::<S, DERIVATIVE, POWERS, N>(
                    &coefficients[degree..],
                    theta,
                    dt,
                );
                let imaginary = if DERIVATIVE {
                    imaginary
                } else {
                    imaginary + initial.imaginary_component()
                };
                T::from_components(real, imaginary)
            };
        }
    }

    #[cold]
    #[inline(never)]
    fn publish_checked(
        &self,
        theta: S,
        dt: S,
        output: &mut [T],
        derivative: bool,
    ) -> Result<(), SciOxideError> {
        // Two deterministic passes preserve caller output on numerical failure.
        for i in 0..output.len() {
            self.component(i, theta, dt, derivative, false)?;
            if !T::IS_REAL {
                self.component(i, theta, dt, derivative, true)?;
            }
        }
        for (i, value) in output.iter_mut().enumerate() {
            let real = self.component(i, theta, dt, derivative, false)?;
            *value = if T::IS_REAL {
                T::from_real(real)
            } else {
                T::from_components(real, self.component(i, theta, dt, derivative, true)?)
            };
        }
        Ok(())
    }

    #[inline]
    fn evaluate_into(&self, t: S, output: &mut [T], derivative: bool) -> Result<(), SciOxideError> {
        let (theta, dt) = self.coordinate(t)?;
        self.publish(theta, dt, output, derivative)
    }
}

pub(crate) fn component<T: NumericScalar>(value: T, imaginary: bool) -> T::Real {
    if imaginary {
        value.imaginary_component()
    } else {
        value.real_component()
    }
}

/// A nonzero subnormal product may later be amplified; replay original operands.
#[inline]
pub(crate) fn native_product<S: RealScalar>(left: S, right: S) -> Option<S> {
    let product = left * right;
    (product.is_finite() && (left == S::ZERO || right == S::ZERO || product.is_normal()))
        .then_some(product)
}

/// Fixed-degree kernels unroll the ordinary RK extensions. N=0 handles a
/// variable-order power polynomial without duplicating the recurrence.
#[inline]
fn native_polynomial<S: RealScalar, const DERIVATIVE: bool, const POWERS: bool, const N: usize>(
    coefficients: &[S],
    theta: S,
    dt: S,
) -> S {
    let degree = if N == 0 { coefficients.len() } else { N };
    let mut value = S::ZERO;
    let mut slope = S::ZERO;
    for j in (0..degree).rev() {
        let positive = POWERS || j % 2 == 0;
        let factor = if positive { theta } else { S::ONE - theta };
        // Normalize before the recurrence to avoid losing tiny physical slopes.
        let sum = value
            + if DERIVATIVE {
                coefficients[j] / dt
            } else {
                coefficients[j]
            };
        if DERIVATIVE {
            slope = scioxide_core::fast_mul_add(slope, factor, if positive { sum } else { -sum });
        }
        value = sum * factor;
    }
    if DERIVATIVE { slope } else { value }
}

/// Absolute bounds leave a fourfold overflow margin for interpolation, including
/// rounding of at most seven basis factors. Derivative coefficients must also
/// stay normal; compare numerator bounds to a step-scaled limit without division.
#[inline]
fn admit_native<S: RealScalar>(values: &[S], initial: S, dt: S) -> bool {
    let step = dt.abs();
    let minimum = if step > S::ONE {
        (S::MIN_POSITIVE * step) * S::TWO
    } else {
        S::MIN_POSITIVE
    };
    let mut value_bound = initial.abs();
    let mut slope_bound = S::ZERO;
    let mut weight = S::ONE;
    for &value in values {
        let magnitude = value.abs();
        if value != S::ZERO && magnitude < minimum {
            return false;
        }
        value_bound += magnitude;
        slope_bound = scioxide_core::fast_mul_add(weight, magnitude, slope_bound);
        weight += S::ONE;
    }
    let limit = S::MAX / (S::TWO * S::TWO);
    // Absolute sums reject nonfinite coefficients as well. Saturating the step
    // at four keeps this comparison finite without a quotient per component.
    value_bound <= limit && slope_bound <= limit * step.min(S::TWO * S::TWO)
}

#[inline]
fn coordinate<S: RealScalar>(time: S, start: S, step: S) -> Result<S, SciOxideError> {
    let difference = time - start;
    let native = difference / step;
    if difference.is_finite() && native.is_finite() && (difference == S::ZERO || native.is_normal())
    {
        Ok(native)
    } else {
        scaled_coordinate(time, start, step)
    }
}

#[cold]
#[inline(never)]
fn scaled_coordinate<S: RealScalar>(time: S, start: S, step: S) -> Result<S, SciOxideError> {
    Scaled::from_value(time)?
        .sub(Scaled::from_value(start)?)?
        .div(Scaled::from_value(step)?)?
        .round()
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> DenseOutputStorage<S, T> {
    /// Return the completed interval in integration order.
    pub(crate) fn interval(&self) -> (S, S) {
        self.interval.map_or(
            (
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
            ),
            |(a, b, _)| (a, b),
        )
    }
    /// Return the interpolated state dimension.
    pub(crate) fn dimension(&self) -> usize {
        self.initial.len()
    }
    /// Evaluate the polynomial into caller storage without allocation; errors leave it unchanged.
    /// # Errors
    /// Returns shape, domain, nonfinite-input or numerical-range errors.
    pub(crate) fn eval_into(&self, t: S, output: &mut [T]) -> Result<(), SciOxideError> {
        self.evaluate_into(t, output, false)
    }
    /// Evaluate the physical-time derivative of the polynomial without another RHS call.
    /// # Errors
    /// Returns the same errors as `eval_into`, leaving output unchanged.
    pub(crate) fn derivative_into(&self, t: S, output: &mut [T]) -> Result<(), SciOxideError> {
        self.evaluate_into(t, output, true)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use scioxide_core::Complex;

    #[test]
    fn extended_coefficients_and_atomic_evaluation() {
        let mut dense = DenseOutputStorage::<f64>::new();
        dense.try_resize(2, 3).unwrap();
        assert!(dense.eval_into(0.0, &mut [0.0; 2]).is_err());
        let old = Vector::from_vec(vec![0.0, -f64::MAX]);
        let new = Vector::from_vec(vec![1.0, f64::MAX]);
        let slope = Vector::from_vec(vec![0.5, f64::MAX]);
        // The second delta is 2*MAX, but the linear curve and slope are finite.
        dense
            .prepare_endpoints(2.0, &old, &new, &slope, &slope)
            .unwrap();
        dense
            .finish(DenseExtension::CubicHermite, 0.0, 2.0, &old, &new)
            .unwrap();
        let mut output = [7.0; 2];
        dense.eval_into(1.0, &mut output).unwrap();
        assert_eq!(output, [0.5, 0.0]);
        dense.derivative_into(1.0, &mut output).unwrap();
        assert_eq!(output, [0.5, f64::MAX]);
        // The second component now overflows: the first must not be published.
        let coefficient = Scaled::from_value(f64::MAX).unwrap().shifted(4).unwrap();
        dense.coefficients[3] = coefficient.significand();
        dense.exponents[3] = coefficient.exponent();
        output = [7.0; 2];
        assert!(dense.eval_into(1.0, &mut output).is_err());
        assert_eq!(output, [7.0; 2]);
        assert!(dense.derivative_into(1.0, &mut output).is_err());
        assert_eq!(output, [7.0; 2]);
    }

    #[test]
    fn native_extrapolation_matches_the_checked_path_and_replays_overflow() {
        let initial = Vector::from_vec(vec![Complex::new(1.0, -2.0), Complex::new(0.5, 0.25)]);
        let endpoint = Vector::from_vec(vec![Complex::new(2.0, -3.0), Complex::new(0.75, 0.5)]);
        let prepare = |coefficients: [f64; 3], dt: f64| {
            let mut segment = DenseOutputStorage::<f64, Complex<f64>>::new();
            segment.try_resize(2, 3).unwrap();
            let values = |imaginary: bool| {
                let mut values = [0.0; 7];
                let sign = if imaginary { -0.5 } else { 1.0 };
                values[..3].copy_from_slice(&coefficients.map(|value| sign * value));
                values
            };
            segment
                .prepare_coefficients(
                    DenseExtension::RadauIia,
                    dt,
                    &initial,
                    |_, imaginary| Some(values(imaginary)),
                    |_, imaginary| {
                        let mut scaled = [Scaled::ZERO; 7];
                        for (out, value) in scaled.iter_mut().zip(values(imaginary)) {
                            *out = Scaled::from_value(value)?;
                        }
                        Ok(scaled)
                    },
                )
                .unwrap();
            segment
                .finish(DenseExtension::RadauIia, 1.0, dt, &initial, &endpoint)
                .unwrap();
            segment
        };
        let segment = prepare([1.0, 0.5, -0.25], 0.5);
        assert!(segment.exponents.is_empty());
        for time in [1.6, 2.0, 5.5] {
            let theta = (time - 1.0) / 0.5;
            let mut native = [Complex::new(0.0, 0.0); 2];
            let mut checked = native;
            segment.extrapolate_into(time, &mut native).unwrap();
            segment
                .publish_checked(theta, 0.5, &mut checked, false)
                .unwrap();
            assert_eq!(native, checked);
        }
        // An admitted degree-three term of size MAX/5 overflows at θ = 9 natively and in
        // the checked replay.
        let huge = prepare([0.0, 0.0, f64::MAX / 5.0], 4.0);
        assert!(huge.exponents.is_empty());
        let mut output = [Complex::new(0.0, 0.0); 2];
        huge.extrapolate_into(9.0, &mut output).unwrap();
        assert!(output.iter().all(|value| value.re.is_finite()));
        let error = huge.extrapolate_into(37.0, &mut output).unwrap_err();
        assert_eq!(
            error,
            huge.publish_checked(9.0, 4.0, &mut output, false)
                .unwrap_err()
        );
        assert!(
            huge.extrapolate_into(5.0, &mut [Complex::new(0.0, 0.0)])
                .is_err()
        );
    }

    #[test]
    fn degrees_domains_and_native_scaled_queries_agree() {
        for extension in [
            DenseExtension::CubicHermite,
            DenseExtension::RadauIia,
            DenseExtension::DormandPrince5,
            DenseExtension::DormandPrince853,
            DenseExtension::Multistep { order: 1 },
            DenseExtension::Multistep { order: 5 },
        ] {
            let degree = extension.degree();
            let initial = Vector::from_vec(vec![Complex::new(1.0, -2.0); 2]);
            let endpoint = Vector::from_vec(vec![Complex::new(2.0, -3.0); 2]);
            let coefficients = |imaginary| {
                let sign = if imaginary { -1.0 } else { 1.0 };
                [1.0, 0.5, -0.25, 0.125, -0.0625, 0.03125, -0.015625].map(|v| sign * v)
            };
            let scaled = |_: usize, imaginary| {
                let mut result = [Scaled::ZERO; 7];
                for (out, value) in result.iter_mut().zip(coefficients(imaginary)) {
                    *out = Scaled::from_value(value)?;
                }
                Ok(result)
            };
            let mut native = DenseOutputStorage::<f64, Complex<f64>>::new();
            let mut extended = DenseOutputStorage::<f64, Complex<f64>>::new();
            for segment in [&mut native, &mut extended] {
                segment.try_resize(2, degree).unwrap();
            }
            native
                .prepare_coefficients(
                    extension,
                    -2.0,
                    &initial,
                    |_, imaginary| Some(coefficients(imaginary)),
                    scaled,
                )
                .unwrap();
            extended
                .prepare_coefficients(extension, -2.0, &initial, |_, _| None, scaled)
                .unwrap();
            native
                .finish(extension, 1.0, -2.0, &initial, &endpoint)
                .unwrap();
            extended
                .finish(extension, 1.0, -2.0, &initial, &endpoint)
                .unwrap();
            assert_eq!(native.coefficients.len(), 4 * degree);
            assert!(native.exponents.is_empty());
            assert_eq!(extended.exponents.len(), 4 * degree);
            for time in [1.0, 0.75, 0.0, -0.75, -1.0] {
                for derivative in [false, true] {
                    let mut ordinary = [Complex::new(0.0, 0.0); 2];
                    let mut wide = ordinary;
                    native
                        .evaluate_into(time, &mut ordinary, derivative)
                        .unwrap();
                    extended.evaluate_into(time, &mut wide, derivative).unwrap();
                    for (a, b) in ordinary.into_iter().zip(wide) {
                        assert!((a.re - b.re).abs() <= 32.0 * f64::EPSILON);
                        assert!((a.im - b.im).abs() <= 32.0 * f64::EPSILON);
                    }
                }
            }
        }
    }

    #[test]
    fn tiny_step_derivative_avoids_normalized_slope_underflow() {
        let initial = Vector::from_vec(vec![0.0]);
        let endpoint = Vector::from_vec(vec![1e-300]);
        let extension = DenseExtension::Multistep { order: 2 };
        let mut dense = DenseOutputStorage::<f64>::new();
        dense.try_resize(1, 2).unwrap();
        let mut coefficients = [0.0; 7];
        coefficients[1] = 1e-300;
        dense
            .prepare_coefficients(
                extension,
                1e-300,
                &initial,
                |_, _| Some(coefficients),
                |_, _| Ok([Scaled::ZERO; 7]),
            )
            .unwrap();
        dense
            .finish(extension, 0.0, 1e-300, &initial, &endpoint)
            .unwrap();
        assert!(dense.exponents.is_empty());
        let mut slope = [0.0];
        dense.derivative_into(1e-320, &mut slope).unwrap();
        let expected = 2.0 * (1e-320 / 1e-300);
        assert!((slope[0] / expected - 1.0).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn subnormal_coefficients_and_overflowing_bounds_select_fallback() {
        for coefficient in [f64::from_bits(1), f64::MAX] {
            let initial = Vector::from_vec(vec![0.0]);
            let endpoint = Vector::from_vec(vec![coefficient]);
            let extension = DenseExtension::CubicHermite;
            let mut dense = DenseOutputStorage::<f64>::new();
            dense.try_resize(1, 3).unwrap();
            let mut coefficients = [0.0; 7];
            coefficients[0] = coefficient;
            if coefficient == f64::MAX {
                coefficients[1] = -coefficient;
            }
            dense
                .prepare_coefficients(
                    extension,
                    0.5,
                    &initial,
                    |_, _| Some(coefficients),
                    |_, _| {
                        let mut result = [Scaled::ZERO; 7];
                        for (out, value) in result.iter_mut().zip(coefficients) {
                            *out = Scaled::from_value(value)?;
                        }
                        Ok(result)
                    },
                )
                .unwrap();
            dense
                .finish(extension, 0.0, 0.5, &initial, &endpoint)
                .unwrap();
            assert_eq!(dense.exponents.len(), 3);
            let mut output = [0.0];
            if coefficient == f64::MAX {
                assert!(dense.derivative_into(0.25, &mut output).is_err());
                assert_eq!(output, [0.0]);
                // At theta=1/2 this derivative is 2*MAX and must fail atomically.
                // Query values still recover a finite MAX/4 after cancellation.
                dense.eval_into(0.25, &mut output).unwrap();
                assert!((output[0] / f64::MAX - 0.25).abs() <= f64::EPSILON);
            } else {
                dense.derivative_into(0.25, &mut output).unwrap();
                assert_eq!(output[0], f64::from_bits(2));
            }
        }
    }

    #[test]
    fn prepared_representation_transitions_and_copies_execute() {
        let old = Vector::from_vec(vec![-f64::MAX]);
        let new = Vector::from_vec(vec![f64::MAX]);
        let slope = Vector::from_vec(vec![f64::MAX]);
        let zero = Vector::from_vec(vec![0.0]);
        let unit = Vector::from_vec(vec![1.0]);
        let mut source = DenseOutputStorage::<f64>::new();
        source.try_resize(1, 3).unwrap();
        let mut target = source.clone();
        for extended in [false, true, false] {
            let (old, new, slope, dt) = if extended {
                (&old, &new, &slope, 2.0)
            } else {
                (&zero, &unit, &unit, 1.0)
            };
            source
                .prepare_endpoints(dt, old, new, slope, slope)
                .unwrap();
            source
                .finish(DenseExtension::CubicHermite, 0.0, dt, old, new)
                .unwrap();
            assert_eq!(!source.exponents.is_empty(), extended);
            target.copy_from(&source).unwrap();
            let mut output = [7.0];
            target.eval_into(dt * 0.5, &mut output).unwrap();
            assert_eq!(output[0], if extended { 0.0 } else { 0.5 });
            target.derivative_into(dt * 0.5, &mut output).unwrap();
            assert_eq!(output[0], if extended { f64::MAX } else { 1.0 });
        }
        assert_eq!(source.coefficients.len(), 3);
        let mut output = [7.0];
        for time in [f64::NAN, f64::INFINITY, -1.0, 2.0] {
            assert!(target.eval_into(time, &mut output).is_err());
            assert_eq!(output, [7.0]);
        }
        assert!(target.eval_into(0.5, &mut []).is_err());
        assert!(target.try_resize(usize::MAX, 7).is_err());
    }
}
