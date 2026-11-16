//! Hermitian input checks and the bound on imaginary residuals of real results.

use scioxide_core::{Complex, RealScalar, SciOxideError};

/// One-shot [`Projector::project`] with maximum `scale`, preparing the frame
/// only for a bin that is not exactly real.
pub(crate) fn project_pair<S: RealScalar>(
    value: Complex<S>,
    mirror: Option<Complex<S>>,
    scale: S,
) -> Result<Complex<S>, SciOxideError> {
    if let Some(real) = exact_endpoint(value, mirror) {
        return Ok(real);
    }
    Projector::new(scale).project_scaled(value, mirror)
}

/// Validation and projection of the endpoint pairs of one spectrum with
/// component maximum `scale`. `None` denotes a self-conjugate bin. All components and `scale` must
/// already have passed finite-input validation.
///
/// Let ε be scalar epsilon, η its smallest subnormal, and M the maximum
/// absolute component of this independent spectrum. The projection allowance
/// is τ = max(8εM, η), a sixteen-unit-roundoff input perturbation budget.
/// A self-conjugate bin requires |Im z| <= τ; a distinct pair requires
/// |Re z - Re w| and |Im z + Im w| <= 2τ. Thus exact Hermitian projection
/// changes each component by at most τ. Binary-frame comparison rounding and
/// correctly rounded pair midpoints add at most εM + η. An N-point Fourier
/// transform therefore changes by at most sqrt(2) N (τ + εM + η), times its
/// normalization scale.
/// This is a backward input-error bound, not a forward-error promise for an
/// arbitrary producer. The one-subnormal floor is the only absolute allowance.
/// Comparisons use a bounded binary frame to avoid overflow and underflow of
/// the relative allowance; exactly zero spectra project to zero.
///
/// The frame and allowance are prepared once per spectrum; the frame's binary
/// scale is applied with [`crate::scaling::BinaryScale`], whose rounding
/// equals `scalbn`'s.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Projector<S> {
    down: crate::scaling::BinaryScale<S>,
    tolerance: S,
}

impl<S: RealScalar> Projector<S> {
    pub(crate) fn new(scale: S) -> Self {
        let exponent = crate::scaling::range_exponent(scale);
        let down = crate::scaling::BinaryScale::new(-exponent);
        Self {
            down,
            tolerance: (S::EPSILON.scalbn(3) * down.apply(scale)).max(down.apply(S::MIN_SUBNORMAL)),
        }
    }

    /// Validate and project one endpoint pair.
    pub(crate) fn project(
        self,
        value: Complex<S>,
        mirror: Option<Complex<S>>,
    ) -> Result<Complex<S>, SciOxideError> {
        if let Some(real) = exact_endpoint(value, mirror) {
            return Ok(real);
        }
        self.project_scaled(value, mirror)
    }

    fn project_scaled(
        self,
        value: Complex<S>,
        mirror: Option<Complex<S>>,
    ) -> Result<Complex<S>, SciOxideError> {
        let down = self.down;
        let re = down.apply(value.re);
        let im = down.apply(value.im);
        let mismatch = if let Some(other) = mirror {
            ((re - down.apply(other.re)).abs() * S::HALF)
                .max((im + down.apply(other.im)).abs() * S::HALF)
        } else {
            im.abs()
        };
        if mismatch > self.tolerance {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("input"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "packed input violates Hermitian symmetry",
            ));
        }
        match mirror {
            Some(other) => Ok(Complex::new(
                scioxide_core::checked_midpoint(value.re, other.re)?,
                scioxide_core::checked_midpoint(value.im, -other.im)?,
            )),
            None => Ok(Complex::new(value.re, S::ZERO)),
        }
    }
}

/// Exactly real self-conjugate endpoint, returned without a binary frame.
fn exact_endpoint<S: RealScalar>(
    value: Complex<S>,
    mirror: Option<Complex<S>>,
) -> Option<Complex<S>> {
    (mirror.is_none() && value.im.is_zero()).then(|| Complex::new(value.re, S::ZERO))
}

#[cfg(test)]
fn project_pair_scaled<S: RealScalar>(
    value: Complex<S>,
    mirror: Option<Complex<S>>,
    scale: S,
) -> Result<Complex<S>, SciOxideError> {
    Projector::new(scale).project_scaled(value, mirror)
}

/// Separate breakdown envelope in the range-scaled arithmetic frame.
///
/// The component-L1 input mass A bounds every Fourier component before
/// normalization, including cancellation-heavy outputs. The existing real FFT
/// arithmetic envelope 128ε max(1, log2 n) is applied to A, not to an output
/// component or an absolute scale of one. The extra 2nη covers gradual
/// underflow of the component products/reduction. This diagnostic envelope
/// does not control which user spectra satisfy the input contract.
pub(crate) fn residual_bound<S: RealScalar>(n: usize, mass: S) -> Result<S, SciOxideError> {
    let length = crate::numeric::scalar_from_usize_rounded::<S>(n)?;
    Ok(
        S::EPSILON.scalbn(7) * length.log2().max(S::ONE) * mass
            + S::MIN_SUBNORMAL * length * S::TWO,
    )
}

/// Component-L1 mass of bounded candidate values, rounded upward so that the
/// diagnostic bound remains conservative without allocating an accumulator.
///
/// Each block of [`MASS_BLOCK`] values is summed in four independent chains
/// of plain additions, so the scan is throughput-bound. Recursive summation of
/// nonnegative terms with at most `m` additions per chain underestimates by a
/// factor of at least `(1 - ε/2)^m` (subnormal sums are exact), which
/// `1 + mε` covers while `mε <= 1`; every block sum is inflated by
/// `1 + MASS_BLOCK ε` (`m < MASS_BLOCK`), rounded up, and accumulated with
/// outward rounding.
pub(crate) fn component_mass<S: RealScalar>(values: impl IntoIterator<Item = Complex<S>>) -> S {
    let mut values = values.into_iter();
    // 1 + MASS_BLOCK ε is exact in either scalar.
    let inflation = S::ONE + S::EPSILON.scalbn(MASS_BLOCK.ilog2() as i32);
    let mut mass = S::ZERO;
    loop {
        let mut chains = [S::ZERO; 4];
        let mut count = 0;
        for value in values.by_ref().take(MASS_BLOCK) {
            let chain = 2 * (count & 1);
            chains[chain] += value.re.abs();
            chains[chain + 1] += value.im.abs();
            count += 1;
        }
        let block = (chains[0] + chains[1]) + (chains[2] + chains[3]);
        if block != S::ZERO {
            mass = add_mass(mass, (block * inflation).next_after(S::INFINITY));
        }
        if count < MASS_BLOCK {
            return mass;
        }
    }
}

/// Values summed per block by [`component_mass`]; a power of two.
const MASS_BLOCK: usize = 256;

/// Add a nonnegative bounded component with outward rounding.
pub(crate) fn add_mass<S: RealScalar>(mass: S, component: S) -> S {
    if component == S::ZERO {
        mass
    } else {
        (mass + component).next_after(S::INFINITY)
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    fn exact_endpoints<S: RealScalar>() {
        for real in [
            S::ZERO,
            S::NEG_ZERO,
            S::MIN_SUBNORMAL,
            -S::MIN_SUBNORMAL,
            S::ONE,
            -S::ONE,
            S::MAX,
            -S::MAX,
        ] {
            for imaginary in [S::ZERO, S::NEG_ZERO] {
                for scale in [real.abs(), S::MAX] {
                    let value = Complex::new(real, imaginary);
                    let fast = project_pair(value, None, scale).unwrap();
                    let slow = project_pair_scaled(value, None, scale).unwrap();
                    assert_eq!(fast.re.decompose(), slow.re.decompose());
                    assert_eq!(fast.im.decompose(), slow.im.decompose());
                }
            }
        }
    }

    #[test]
    fn exact_self_conjugate_fast_path_matches_scaled_projection() {
        exact_endpoints::<f32>();
        exact_endpoints::<f64>();
    }
}
