//! Transform selectors, normalization and plan diagnostics.
use crate::numeric::scalar_from_usize_rounded;
use scioxide_core::SciOxideError;
/// Selects how a forward and inverse transform pair is scaled.
///
/// For a Fourier transform of length `n`:
///
/// | Variant | Forward scale | Inverse scale |
/// |---|---|---|
/// | `Forward` | `1/n` | `1` |
/// | `Inverse` | `1` | `1/n` |
/// | `Orthonormal` | `1/√n` | `1/√n` |
/// | `Unscaled` | `1` | `1` |
///
/// N-D transforms use the product of the selected axis lengths for `n`. DCT
/// and DST use their round-trip factor in place of `n`, and orthonormal DCT
/// and DST also weight the endpoints; see [`RealTransformPlan`](crate::RealTransformPlan).
/// Other transforms state their scale on their plan. See also
/// [Normalization](crate#normalization).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum Normalization {
    /// Divides the forward transform by the round-trip factor.
    Forward,
    /// Divides the inverse transform by the round-trip factor.
    Inverse,
    /// Scales both directions so that the transform is orthonormal (unitary).
    Orthonormal,
    /// Scales neither direction; a round trip multiplies by the round-trip factor.
    Unscaled,
}

impl Normalization {
    pub(crate) fn fourier_scales<S: scioxide_core::RealScalar>(
        self,
        n: usize,
    ) -> Result<(S, S), SciOxideError> {
        let n_scalar = scalar_from_usize_rounded::<S>(n)?;
        let reciprocal = n_scalar.recip();
        Ok(match self {
            Self::Forward => (reciprocal, S::ONE),
            Self::Inverse => (S::ONE, reciprocal),
            Self::Orthonormal => {
                let scale = n_scalar.sqrt().recip();
                (scale, scale)
            }
            Self::Unscaled => (S::ONE, S::ONE),
        })
    }
}

/// Names the algorithm a Fourier plan uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum FftAlgorithmKind {
    /// The identity, for length one.
    Identity,
    /// Mixed-radix Cooley–Tukey stages of radix 2, 3, 4, 5 and 7.
    MixedRadix,
    /// Rader's algorithm: a prime length reduced to a cyclic convolution of
    /// length `n − 1`.
    Rader,
    /// Bluestein's algorithm: a chirp-weighted convolution of 2/3/5/7-smooth
    /// length at least `2n − 1`.
    Bluestein,
    /// A kernel from an injected [`FftProvider`](crate::FftProvider).
    Provider,
}

/// Reports the algorithm and stage count of a Fourier plan.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct FftPlanDiagnostics {
    pub(crate) algorithm: FftAlgorithmKind,
    pub(crate) stage_count: usize,
}

impl FftPlanDiagnostics {
    /// Creates diagnostics for a provider kernel with the given stage count.
    ///
    /// ```
    /// use scioxide_fft::{FftAlgorithmKind, FftPlanDiagnostics};
    /// let diagnostics = FftPlanDiagnostics::for_provider(3);
    /// assert_eq!(diagnostics.algorithm(), FftAlgorithmKind::Provider);
    /// assert_eq!(diagnostics.stage_count(), 3);
    /// ```
    pub const fn for_provider(stage_count: usize) -> Self {
        Self {
            algorithm: FftAlgorithmKind::Provider,
            stage_count,
        }
    }

    /// Returns the selected algorithm.
    pub const fn algorithm(self) -> FftAlgorithmKind {
        self.algorithm
    }

    /// Returns the largest number of FFT stages on any execution path.
    pub const fn stage_count(self) -> usize {
        self.stage_count
    }

    pub(crate) fn with_additional_stage(mut self) -> Self {
        self.stage_count = self.stage_count.saturating_add(1);
        self
    }
}

/// Selects the DCT type I, II, III or IV.
///
/// The defining sums are listed on [`RealTransformPlan`](crate::RealTransformPlan).
/// Inverse transforms take the type of the forward transform they invert:
/// types I and IV are their own inverses up to scale, and types II and III are
/// inverses of each other up to scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctType {
    /// Type I, with weighted endpoints; needs at least two samples.
    I,
    /// Type II, the most common DCT.
    II,
    /// Type III, the inverse of type II up to scale.
    III,
    /// Type IV, with half-sample shifts in input and output.
    IV,
}

/// Selects the DST type I, II, III or IV.
///
/// The defining sums are listed on [`RealTransformPlan`](crate::RealTransformPlan).
/// Inverse transforms take the type of the forward transform they invert:
/// types I and IV are their own inverses up to scale, and types II and III are
/// inverses of each other up to scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DstType {
    /// Type I, without endpoint weights; needs at least one sample.
    I,
    /// Type II.
    II,
    /// Type III, the inverse of type II up to scale.
    III,
    /// Type IV, with half-sample shifts in input and output.
    IV,
}
