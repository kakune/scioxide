//! Bluestein convolution: chirp and kernel preparation, radial input
//! weighting with a separate binary exponent, and execution.
use super::{CztPlan, CztPlanSpec};
use crate::FftPlanningContext;
use crate::chirp::phase;
use crate::complex::ComplexCore;
use alloc::vec::Vec;
use scioxide_core::backend::{KernelBackend, checked_add, checked_mul, try_filled};
use scioxide_core::{Argument, Complex, ExecutionContext, RealScalar, Scaled, SciOxideError};

#[derive(Debug, Clone)]
pub(super) struct Convolution<S: RealScalar> {
    pub(super) fft: ComplexCore<S>,
    pub(super) kernel: Vec<Complex<S>>,
    pre: Vec<Complex<S>>,
    post: Vec<Complex<S>>,
    radius: Vec<Scaled<S>>,
    scalar_radius: Vec<S>,
    headroom: i32,
    kernel_exponent: i32,
    inverse_scale: S,
}

impl<S: RealScalar> Convolution<S> {
    /// Prepare chirps, radial factors and the transformed kernel. `scale` is
    /// the forward normalization folded into the post phases.
    pub(super) fn prepare(
        spec: &CztPlanSpec<S>,
        backend: KernelBackend,
        context: FftPlanningContext<'_, S>,
        scale: S,
    ) -> Result<Self, SciOxideError> {
        let spec_argument = scioxide_core::argument!("spec");
        // Lags run from -(n-1) to m-1, so any cyclic length >= n + m - 1
        // avoids wrap-around; a smooth length keeps the FFT mixed-radix.
        // The stored inverse scale supplies the cyclic convolution factor 1/l.
        let l = crate::smooth_len::fast_len(
            checked_add(
                spec.n,
                spec.m - 1,
                scioxide_core::quantity!("czt_convolution_length"),
            )?,
            true,
            spec_argument,
        )?;
        let fft = ComplexCore::with_backend_and_context(
            l,
            spec.policy,
            backend,
            context,
            spec_argument,
            scioxide_core::argument!("context"),
        )?;
        let entries = checked_add(
            checked_add(
                spec.n,
                spec.m,
                scioxide_core::quantity!("czt_preparation_entries"),
            )?,
            checked_add(
                l,
                fft.scratch_len(),
                scioxide_core::quantity!("czt_preparation_entries"),
            )?,
            scioxide_core::quantity!("czt_preparation_entries"),
        )?;
        let (radius, scalar_radius) = Self::radial_factors(spec, entries, context)?;
        let (pre, post, mut kernel) = Self::chirps(spec, l, scale, spec_argument)?;
        let mut work = try_filled(
            fft.scratch_len(),
            <scioxide_core::Complex<_> as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("czt_kernel_scratch"),
        )?;
        let capacity = [
            pre.capacity(),
            post.capacity(),
            kernel.capacity(),
            work.capacity(),
        ]
        .into_iter()
        .try_fold(0, |sum, capacity| {
            checked_add(
                sum,
                capacity,
                scioxide_core::quantity!("czt_preparation_entries"),
            )
        })?;
        crate::memory::admit_bytes(
            owned_bytes::<S>(capacity, radius.capacity(), scalar_radius.capacity())?,
            spec.limits.preparation_bytes,
            scioxide_core::quantity!("czt_preparation_bytes"),
        )?;
        fft.transform(&mut kernel, &mut work, false, S::ONE, context.execution())?;
        let kernel_exponent =
            crate::scaling::range_exponent(crate::scaling::complex_max(kernel.iter().copied())?);
        kernel.iter_mut().for_each(|z| {
            z.re = z.re.scalbn(-kernel_exponent);
            z.im = z.im.scalbn(-kernel_exponent);
        });
        Ok(Self {
            fft,
            kernel,
            pre,
            post,
            radius,
            scalar_radius,
            // Radial chirps are limited to exp(0.5); complex rotations
            // and spectral multiplication each reserve another two bits.
            headroom: crate::scaling::growth_exponent(spec.n) + 4,
            kernel_exponent,
            inverse_scale: crate::numeric::scalar_from_usize_rounded::<S>(l)?.recip(),
        })
    }

    // Extended radial input factors exp(-j*log_radius), plus their scalar
    // roundings when every factor is normal. `entries` counts the complex
    // preparation entries admitted alongside them.
    fn radial_factors(
        spec: &CztPlanSpec<S>,
        entries: usize,
        context: FftPlanningContext<'_, S>,
    ) -> Result<(Vec<Scaled<S>>, Vec<S>), SciOxideError> {
        let ar = spec.contour.log_radius().try_to_f64_exact()?;
        let radial_entries = if ar == 0.0 { 0 } else { spec.n };
        crate::memory::admit_bytes(
            owned_bytes::<S>(entries, radial_entries, 0)?,
            spec.limits.preparation_bytes,
            scioxide_core::quantity!("czt_preparation_bytes"),
        )?;
        let mut radius = try_filled(
            radial_entries,
            Scaled::ONE,
            scioxide_core::quantity!("czt_radius"),
        )?;
        for (j, factor) in radius.iter_mut().enumerate() {
            context.execution().check()?;
            // Reduce the exactly widened log in binary64, then round only
            // the significand to S. Never materialize exp(-ar*j) in S.
            let log = -ar * j as f64;
            if !log.is_finite() {
                return Err(crate::validation::arithmetic_overflow(
                    "CZT radial log overflow",
                ));
            }
            let r = Scaled::<f64>::from_log(1.0, log)?;
            *factor = Scaled::from_parts(S::try_from_f64_lossy(r.significand())?, r.exponent())?;
        }
        // Normal factors round exactly: retain the extended originals for
        // inputs whose weighted components do not fit the ordinary frame.
        let scalar_entries = if radius.iter().all(|r| r.round().is_ok_and(S::is_normal)) {
            radius.len()
        } else {
            0
        };
        crate::memory::admit_bytes(
            owned_bytes::<S>(entries, radius.capacity(), scalar_entries)?,
            spec.limits.preparation_bytes,
            scioxide_core::quantity!("czt_preparation_bytes"),
        )?;
        let mut scalar_radius = try_filled(
            scalar_entries,
            S::ZERO,
            scioxide_core::quantity!("czt_scalar_radius"),
        )?;
        for (out, r) in scalar_radius.iter_mut().zip(&radius) {
            *out = r.round()?;
        }
        Ok((radius, scalar_radius))
    }

    // With z_k = A W^(-k), A = exp(ar + 2*pi*i*ap), W = exp(-(dr + 2*pi*i*step)),
    // the transform is X[k] = sum_j x[j] z_k^(-j) = sum_j x[j] A^-j W^(jk).
    // The Bluestein identity jk = (j^2 + k^2 - (k - j)^2) / 2 splits it
    // into pre[j] = A^-j W^(j^2/2), a kernel W^(-d^2/2) in the lag
    // d = k - j, and post[k] = W^(k^2/2), radial parts included. The radius
    // A^-j is applied separately as an input weight.
    #[expect(clippy::type_complexity, reason = "the three prepared chirp vectors")]
    fn chirps(
        spec: &CztPlanSpec<S>,
        l: usize,
        scale: S,
        argument: Argument,
    ) -> Result<(Vec<Complex<S>>, Vec<Complex<S>>, Vec<Complex<S>>), SciOxideError> {
        let c = spec.contour;
        let ap = c.start_turns();
        let dr = c.log_radius_step().try_to_f64_exact()?;
        let mut pre = try_filled(
            spec.n,
            <scioxide_core::Complex<_> as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("czt_pre_chirp"),
        )?;
        let mut post = try_filled(
            spec.m,
            <scioxide_core::Complex<_> as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("czt_post_chirp"),
        )?;
        let mut kernel = try_filled(
            l,
            <scioxide_core::Complex<_> as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("czt_kernel"),
        )?;
        for (j, value) in pre.iter_mut().enumerate() {
            let jf = j as f64;
            *value = phase::<S>(-ap, j as u128)?
                * c.step_phase((j as u128) * (j as u128), true, true, argument)?
                * S::try_from_f64_lossy((-dr * jf * jf / 2.0).exp())?;
        }
        for (k, value) in post.iter_mut().enumerate() {
            *value = c.step_phase((k as u128) * (k as u128), true, true, argument)?
                * S::try_from_f64_lossy((-dr * (k as f64) * (k as f64) / 2.0).exp())?;
        }
        // Fold normalization into prepared post phases. Execution then
        // performs only the complex phase product before binary restore.
        post.iter_mut().for_each(|z| *z *= scale);
        // The kernel is even in the lag: non-negative lags (< m) at the
        // front, negative lags (> -n) wrapped to the back. Only its
        // spectrum is kept, overwriting the time-domain buffer.
        for j in 0..spec.m.max(spec.n) {
            let value = c.step_phase((j as u128) * (j as u128), true, false, argument)?
                * S::try_from_f64_lossy((dr * (j as f64) * (j as f64) / 2.0).exp())?;
            if j < spec.m {
                kernel[j] = value;
            }
            if j > 0 && j < spec.n {
                kernel[l - j] = value;
            }
        }
        Ok((pre, post, kernel))
    }

    /// Complex convolution scratch beyond the candidate output.
    pub(super) fn work_len(&self) -> Result<usize, SciOxideError> {
        checked_add(
            self.kernel.len(),
            self.fft.bounded_scratch_len(),
            scioxide_core::quantity!("czt_convolution_scratch"),
        )
    }

    /// Add the stored convolution capacities and child FFT tables to `bytes`.
    pub(super) fn retained_bytes(&self, bytes: usize) -> Result<usize, SciOxideError> {
        // The FFT value itself is already included in the plan's inline enum.
        let mut bytes = checked_add(
            bytes,
            self.fft.plan_bytes() - core::mem::size_of::<ComplexCore<S>>(),
            scioxide_core::quantity!("czt_plan_bytes"),
        )?;
        for capacity in [
            self.kernel.capacity(),
            self.pre.capacity(),
            self.post.capacity(),
        ] {
            bytes = checked_add(
                bytes,
                checked_mul(
                    capacity,
                    core::mem::size_of::<Complex<S>>(),
                    scioxide_core::quantity!("czt_vector_bytes"),
                )?,
                scioxide_core::quantity!("czt_plan_bytes"),
            )?;
        }
        checked_add(
            bytes,
            checked_add(
                checked_mul(
                    self.radius.capacity(),
                    size_of::<Scaled<S>>(),
                    scioxide_core::quantity!("czt_radius_bytes"),
                )?,
                checked_mul(
                    self.scalar_radius.capacity(),
                    size_of::<S>(),
                    scioxide_core::quantity!("czt_radius_bytes"),
                )?,
                scioxide_core::quantity!("czt_radius_bytes"),
            )?,
            scioxide_core::quantity!("czt_plan_bytes"),
        )
    }

    /// Evaluate nonzero validated input of maximum component `magnitude` into
    /// `candidate`, using `work` for the convolution buffer and child scratch.
    pub(super) fn evaluate(
        &self,
        input: &impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        magnitude: S,
        candidate: &mut [Complex<S>],
        work: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (values, work) = work.split_at_mut(self.kernel.len());
        values.fill(<scioxide_core::Complex<_> as scioxide_core::AlgebraicScalar>::ZERO);
        let exponent = self.weight(input, magnitude, values, context)?;
        self.fft
            .transform_bounded(values, work, false, S::ONE, context)?;
        values
            .iter_mut()
            .zip(&self.kernel)
            .for_each(|(v, k)| *v *= *k);
        self.fft
            .transform_bounded(values, work, true, self.inverse_scale, context)?;
        for ((out, value), post) in candidate.iter_mut().zip(values.iter()).zip(&self.post) {
            *out = *value * *post;
        }
        let restore = exponent
            .checked_add(i64::from(self.kernel_exponent))
            .ok_or_else(|| {
                crate::validation::arithmetic_overflow("CZT restoration exponent overflow")
            })?;
        if let Ok(restore) = i32::try_from(restore) {
            crate::scaling::restore_complex(candidate, restore, S::INFINITY)?;
        } else {
            for z in candidate {
                z.re = Scaled::from_value(z.re)?.shifted(restore)?.round()?;
                z.im = Scaled::from_value(z.im)?.shifted(restore)?.round()?;
            }
        }
        Ok(())
    }

    // Write the radially weighted, pre-chirped input in a common binary frame
    // and return that frame's exponent.
    fn weight(
        &self,
        input: &impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        magnitude: S,
        values: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<i64, SciOxideError> {
        if self.radius.is_empty() {
            let exponent = crate::scaling::range_exponent(magnitude) + self.headroom;
            let scale = crate::scaling::BinaryScale::new(-exponent);
            for (j, (value, pre)) in values.iter_mut().zip(&self.pre).enumerate() {
                let z = input(j)?;
                *value = Complex::new(scale.apply(z.re), scale.apply(z.im)) * *pre;
            }
            return Ok(i64::from(exponent));
        }
        if let Some(exponent) = self.weight_scalar(input, values, context)? {
            return Ok(exponent);
        }
        // Measure weighted components, so zero/tiny input at a large
        // radius factor cannot erase a significant smaller factor.
        let mut maximum = i64::MIN;
        for (j, &radius) in self.radius.iter().enumerate() {
            context.check()?;
            let z = input(j)?;
            for x in [z.re, z.im] {
                if !x.is_zero() {
                    maximum = maximum.max(Scaled::from_value(x)?.mul(radius)?.exponent());
                }
            }
        }
        let exponent = maximum
            .checked_add(i64::from(self.headroom) + 1)
            .ok_or_else(|| {
                crate::validation::arithmetic_overflow("CZT weighted exponent overflow")
            })?;
        for (j, ((value, pre), &radius)) in values
            .iter_mut()
            .zip(&self.pre)
            .zip(&self.radius)
            .enumerate()
        {
            context.check()?;
            let z = input(j)?;
            let weighted = |x| {
                Scaled::from_value(x)?
                    .mul(radius)?
                    .shifted(-exponent)?
                    .round()
            };
            *value = Complex::new(weighted(z.re)?, weighted(z.im)?) * *pre;
        }
        Ok(exponent)
    }

    // Speculation touches only convolution scratch. Failed products restart
    // with the original input and extended factors, including underflow to zero.
    fn weight_scalar(
        &self,
        input: &impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        values: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<Option<i64>, SciOxideError> {
        if self.scalar_radius.is_empty() {
            return Ok(None);
        }
        let mut maximum = S::ZERO;
        for (j, (value, radius)) in values.iter_mut().zip(&self.scalar_radius).enumerate() {
            context.check()?;
            let x = input(j)?;
            *value = x * *radius;
            if (!x.re.is_zero() && !value.re.is_normal())
                || (!x.im.is_zero() && !value.im.is_normal())
            {
                return Ok(None);
            }
            maximum = maximum.max(value.re.abs()).max(value.im.abs());
        }
        let exponent = crate::scaling::range_exponent(maximum) + self.headroom;
        let scale = crate::scaling::BinaryScale::new(-exponent);
        for (value, pre) in values.iter_mut().zip(&self.pre) {
            *value = Complex::new(scale.apply(value.re), scale.apply(value.im)) * *pre;
        }
        Ok(Some(i64::from(exponent)))
    }
}

// Inline plan metadata plus CZT-owned complex, extended-radial and scalar
// preparation entries.
fn owned_bytes<S: RealScalar>(
    entries: usize,
    radial_entries: usize,
    scalar_entries: usize,
) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("czt_preparation_bytes");
    checked_add(
        checked_add(
            size_of::<CztPlan<S>>(),
            checked_mul(entries, size_of::<Complex<S>>(), q)?,
            q,
        )?,
        checked_add(
            checked_mul(radial_entries, size_of::<Scaled<S>>(), q)?,
            checked_mul(scalar_entries, size_of::<S>(), q)?,
            q,
        )?,
        q,
    )
}

#[cfg(test)]
mod tests {
    use super::super::{CztContour, Preparation};
    use super::*;
    use crate::Normalization;

    fn scalar_weighting<S: RealScalar>()
    where
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        for ar in [-0.02, 0.02] {
            let plan = CztPlan::new(CztPlanSpec::new(
                65,
                65,
                CztContour::new(
                    S::try_from_f64_lossy(ar).unwrap(),
                    S::ZERO,
                    S::ZERO,
                    S::ZERO,
                )
                .unwrap(),
                Normalization::Forward,
            ))
            .unwrap();
            let Preparation::Convolution(c) = &plan.preparation else {
                unreachable!()
            };
            assert_eq!(c.scalar_radius.len(), 65);
            let mut extended = plan.clone();
            let Preparation::Convolution(reference) = &mut extended.preparation else {
                unreachable!()
            };
            reference.scalar_radius.clear();
            for amplitude in [S::ONE, S::MIN_POSITIVE, S::MIN_SUBNORMAL, S::MAX] {
                let mut input = alloc::vec![Complex::new(S::ZERO, S::ZERO); 65];
                input[64] = Complex::new(amplitude, -amplitude);
                let weighted = amplitude * c.scalar_radius[64];
                let ordinary = c
                    .weight_scalar(
                        &|j| Ok(input[j]),
                        &mut alloc::vec![Complex::new(S::ZERO, S::ZERO); c.kernel.len()],
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert_eq!(ordinary.is_some(), weighted.is_normal());
                let actual = plan.execute_owned((&input).into()).unwrap();
                let expected = extended.execute_owned((&input).into()).unwrap();
                for (a, e) in actual.iter().zip(expected) {
                    for (a, e) in [(a.re, e.re), (a.im, e.im)] {
                        assert!(
                            (a - e).abs()
                                <= S::try_from_f64_lossy(512.0).unwrap() * S::EPSILON * e.abs()
                                    + S::TWO * S::MIN_SUBNORMAL
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn scalar_weighting_and_extended_restart_f64() {
        scalar_weighting::<f64>();
    }
    #[test]
    fn scalar_weighting_and_extended_restart_f32() {
        scalar_weighting::<f32>();
    }
}
