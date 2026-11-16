//! Normalization, restoration and all-or-nothing output writes.
use super::*;

impl<S: scioxide_core::RealScalar> RealFftPlan<S> {
    /// Write the normalized bins of a transformed staging area, or zeros for
    /// an all-zero signal (`exponent` is `None`). A prechecked finite bound
    /// lets contiguous even output fuse the split with restoration; otherwise
    /// the split runs in place and restoration is checked before any bin is
    /// written.
    pub(super) fn publish_bins(
        &self,
        scale: S,
        sign: S,
        scratch: &mut [Complex<S>],
        exponent: Option<i32>,
        bins: Bins<'_, S>,
    ) -> Result<(), SciOxideError> {
        let Some(exponent) = exponent else {
            match bins {
                Bins::Contiguous(bins) => bins.fill(Complex::new(S::ZERO, S::ZERO)),
                Bins::Strided(mut bins) => bins.fill(Complex::new(S::ZERO, S::ZERO)),
            }
            return Ok(());
        };
        let bound = self.output_bound(scale);
        let prechecked = output_factor(bound, exponent, scale);
        let twiddles = match (&self.algorithm, prechecked, &bins) {
            (RealAlgorithm::Even { twiddles, .. }, Some(_), Bins::Contiguous(_)) => {
                Some(twiddles.as_slice())
            }
            _ => None,
        };
        if twiddles.is_none()
            && let RealAlgorithm::Even { twiddles, .. } = &self.algorithm
        {
            postprocess_even_forward(scratch, twiddles, self.n / 2);
        }
        let len = if twiddles.is_some() {
            self.n / 2
        } else {
            self.spectrum_len
        };
        let values = &mut scratch[..len];
        let (factor, restore) = match prechecked {
            Some(prepared) => prepared,
            None => restore_checked(values, exponent, scale, bound)?,
        };
        match restore {
            BinaryScale::Identity => {
                publish_restored_bins(values, twiddles, bins, factor, sign, Identity)
            }
            BinaryScale::Multiply(binary) => {
                publish_restored_bins(values, twiddles, bins, factor, sign, Multiply(binary))
            }
            BinaryScale::Extended(exponent) => {
                publish_restored_bins(values, twiddles, bins, factor, sign, Extended(exponent))
            }
        }
        Ok(())
    }

    /// Write staged real samples in order with one prepared restoration.
    pub(super) fn publish_samples<'a>(
        &self,
        samples: &[Complex<S>],
        factor: S,
        restore: BinaryScale<S>,
        destinations: impl Iterator<Item = &'a mut S>,
    ) where
        S: 'a,
    {
        let packed = self.is_packed();
        match restore {
            BinaryScale::Identity => {
                publish_restored_samples(samples, destinations, packed, factor, Identity)
            }
            BinaryScale::Multiply(binary) => {
                publish_restored_samples(samples, destinations, packed, factor, Multiply(binary))
            }
            BinaryScale::Extended(exponent) => {
                publish_restored_samples(samples, destinations, packed, factor, Extended(exponent))
            }
        }
    }
}
/// Write scalar samples with normalization and restoration selected once.
#[inline(always)]
pub(super) fn publish_restored_samples<'a, S: scioxide_core::RealScalar + 'a>(
    samples: &[Complex<S>],
    destinations: impl Iterator<Item = &'a mut S>,
    packed: bool,
    factor: S,
    restore: impl Scale<S>,
) {
    #[inline(always)]
    fn each<'a, S: scioxide_core::RealScalar + 'a>(
        samples: &[Complex<S>],
        destinations: impl Iterator<Item = &'a mut S>,
        packed: bool,
        publish: &impl Scale<S>,
    ) {
        if packed {
            for (destination, value) in destinations.zip(samples.iter().flat_map(|v| [v.re, v.im]))
            {
                *destination = publish.scale(value);
            }
        } else {
            for (destination, value) in destinations.zip(samples) {
                *destination = publish.scale(value.re);
            }
        }
    }
    if factor == S::ONE {
        each(
            samples,
            destinations,
            packed,
            &Components::<_, _, false> {
                factor,
                restore: &restore,
            },
        );
    } else {
        each(
            samples,
            destinations,
            packed,
            &Components::<_, _, true> {
                factor,
                restore: &restore,
            },
        );
    }
}

/// Apply one prepared publication policy to a fused even split or complete bins.
/// The concrete scaling policies inline through both scalar and block accessors.
#[inline(always)]
pub(super) fn publish_restored_bins<S: scioxide_core::RealScalar>(
    values: &[Complex<S>],
    twiddles: Option<&[Complex<S>]>,
    bins: Bins<'_, S>,
    factor: S,
    sign: S,
    restore: impl Scale<S>,
) {
    #[inline(always)]
    fn each<S: scioxide_core::RealScalar>(
        values: &[Complex<S>],
        twiddles: Option<&[Complex<S>]>,
        bins: Bins<'_, S>,
        publish: &impl Scale<Block<S, 1>>,
    ) {
        match bins {
            Bins::Contiguous(bins) => {
                if let Some(twiddles) = twiddles {
                    publish_even_forward(values, twiddles, bins, publish);
                } else {
                    for (bin, &value) in bins.iter_mut().zip(values) {
                        *bin = publish.scale(Block::from(value)).lane(0);
                    }
                }
            }
            Bins::Strided(mut bins) => {
                debug_assert!(twiddles.is_none());
                for (bin, &value) in bins.iter_mut().zip(values) {
                    *bin = publish.scale(Block::from(value)).lane(0);
                }
            }
        }
    }
    #[inline(always)]
    fn signed<S: scioxide_core::RealScalar, R: Scale<S>, const NORMALIZE: bool>(
        values: &[Complex<S>],
        twiddles: Option<&[Complex<S>]>,
        bins: Bins<'_, S>,
        factor: S,
        sign: S,
        restore: &R,
    ) {
        let components = Components::<_, _, NORMALIZE> { factor, restore };
        if sign == S::ONE {
            each(
                values,
                twiddles,
                bins,
                &BinPublication::<_, _, false> { components, sign },
            );
        } else {
            each(
                values,
                twiddles,
                bins,
                &BinPublication::<_, _, true> { components, sign },
            );
        }
    }
    if factor == S::ONE {
        signed::<_, _, false>(values, twiddles, bins, factor, sign, &restore);
    } else {
        signed::<_, _, true>(values, twiddles, bins, factor, sign, &restore);
    }
}

pub(super) fn checked_restore<S: scioxide_core::RealScalar>(
    value: S,
    exponent: i32,
    scale: S,
) -> Result<S, SciOxideError> {
    let value = BinaryScale::new(exponent).restore(value * scale);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(crate::validation::real_transform_overflow())
    }
}

/// Publication factor and restoration when `bound` proves every normalized,
/// restored output finite; `None` requires [`restore_checked`].
pub(super) fn output_factor<S: scioxide_core::RealScalar>(
    bound: S,
    exponent: i32,
    scale: S,
) -> Option<(S, BinaryScale<S>)> {
    bound
        .scalbn(2)
        .scalbn(exponent)
        .is_finite()
        .then(|| (scale, BinaryScale::new(exponent)))
}

/// Normalize and restore `values` in place, checking each for overflow.
/// The values are then written unchanged.
pub(super) fn restore_checked<S: scioxide_core::RealScalar>(
    values: &mut [Complex<S>],
    exponent: i32,
    scale: S,
    bound: S,
) -> Result<(S, BinaryScale<S>), SciOxideError> {
    if scale != S::ONE {
        values.iter_mut().for_each(|v| *v *= scale);
    }
    restore_complex(values, exponent, bound.scalbn(2))?;
    Ok((S::ONE, BinaryScale::Identity))
}

pub(super) fn check_odd_inverse_residual<S: scioxide_core::RealScalar>(
    values: &[Complex<S>],
    n: usize,
    mass: S,
) -> Result<(), SciOxideError> {
    let tolerance = crate::hermitian::residual_bound(n, mass)?;
    if values
        .iter()
        .all(|value| value.is_finite() && value.im.abs() <= tolerance)
    {
        Ok(())
    } else {
        Err(crate::validation::numerical(
            scioxide_core::NumericalIssue::Breakdown,
            "odd-length inverse produced an excessive global imaginary residual",
        ))
    }
}

/// Read-only scalar spectrum and separately borrowed publication storage.
struct RawBins<'a, S>(&'a [Complex<S>]);
struct PublishedBins<'a, S>(&'a mut [Complex<S>]);
impl<S: scioxide_core::RealScalar> Load<Block<S, 1>> for RawBins<'_, S> {
    #[inline(always)]
    fn load(&self, i: usize) -> Block<S, 1> {
        Block::from(self.0[i])
    }
}
impl<S: scioxide_core::RealScalar> Store<Block<S, 1>> for PublishedBins<'_, S> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, 1>) {
        self.0[i] = value.lane(0);
    }
}

/// The three prepared restoration operations, with no per-component dispatch.
struct Identity;
struct Multiply<S>(S);
struct Extended(i32);
impl<S: scioxide_core::RealScalar> Scale<S> for Identity {
    #[inline(always)]
    fn scale(&self, value: S) -> S {
        value
    }
}
impl<S: scioxide_core::RealScalar> Scale<S> for Multiply<S> {
    #[inline(always)]
    fn scale(&self, value: S) -> S {
        value * self.0
    }
}
impl<S: scioxide_core::RealScalar> Scale<S> for Extended {
    #[inline(always)]
    fn scale(&self, value: S) -> S {
        value.scalbn(self.0)
    }
}

/// One normalization, exact-zero canonicalization and binary restoration policy.
/// A nonzero value that underflows during restoration keeps its sign.
struct Components<'a, S, R, const NORMALIZE: bool> {
    factor: S,
    restore: &'a R,
}
impl<S: scioxide_core::RealScalar, R: Scale<S>, const NORMALIZE: bool> Scale<S>
    for Components<'_, S, R, NORMALIZE>
{
    #[inline(always)]
    fn scale(&self, value: S) -> S {
        let value = if NORMALIZE {
            value * self.factor
        } else {
            value
        };
        if value.is_zero() {
            S::ZERO
        } else {
            self.restore.scale(value)
        }
    }
}

/// Component publication followed by optional conjugation of the imaginary part.
struct BinPublication<P, S, const SIGNED: bool> {
    components: P,
    sign: S,
}
impl<S: scioxide_core::RealScalar, P: Scale<S>, const SIGNED: bool> Scale<Block<S, 1>>
    for BinPublication<P, S, SIGNED>
{
    #[inline(always)]
    fn scale(&self, value: Block<S, 1>) -> Block<S, 1> {
        let value = value.lane(0);
        let re = self.components.scale(value.re);
        let im = self.components.scale(value.im);
        Block::from(Complex::new(re, if SIGNED { im * self.sign } else { im }))
    }
}

/// Split a half-length spectrum and fuse each bin's publication.
#[inline(always)]
pub(super) fn publish_even_forward<S: scioxide_core::RealScalar>(
    z: &[Complex<S>],
    twiddles: &[Complex<S>],
    output: &mut [Complex<S>],
    publish: &impl Scale<Block<S, 1>>,
) {
    even::split_even(
        z.len(),
        twiddles,
        &RawBins(z),
        publish,
        &mut PublishedBins(&mut output[..=z.len()]),
    );
}
