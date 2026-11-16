//! Planned real-valued FFT and inverse FFT.
//!
//! [`RealFftPlan`] owns Hermitian packing for one fixed logical length. Even
//! lengths use a half-size complex transform. Odd multiples of 3, 5 or 7 use a
//! real-data decimation (see `decimation`) at about half the cost of a complex
//! FFT; other odd lengths, including the primes 3, 5 and 7, use the
//! exact-length complex plan directly. Packing never adds a normalization
//! factor. Even and decimated plans stage samples `2i` and `2i + 1` in slot `i`.
//!
//! Contiguous views and N-D lanes stage and write with tight slice loops;
//! strided ones take separate iterator paths with identical arithmetic. The
//! input maximum is measured in one read-only pass, then range scaling and
//! packing share one pass; the even split is fused with restoration whenever
//! the prepared bound proves the written bins finite.
//!
//! Bounded N-D lanes of even plans whose smooth half-length child has at most
//! 256 points run four at a time. Packing and c2r merging stage bounded values
//! directly in each native lane's digit order; shared native stages transform
//! complete lanes. Scalar/AVX2 keep block rows on the stack and mirror each
//! native stage's exact products; other backends keep one-lane leaves. Split/merge adapters
//! read and write blocks directly, with full-block addressing selected once
//! per run and repeated finite lanes for tails. Results equal individual lanes bit for
//! bit, including the selected backend's fused arithmetic.

mod decimation;
mod even;
mod execution;
mod lanes;
mod planning;
mod publication;
mod staging;
#[cfg(test)]
#[expect(clippy::cast_precision_loss)]
mod tests;

pub(crate) use even::{merge_even, split_even};
use even::{postprocess_even_forward, preprocess_even_inverse};
use publication::*;
use staging::stage_bins;

use alloc::vec::Vec;
use core::mem::size_of;

use scioxide_core::Complex;
use scioxide_core::backend::KernelBackend;
use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{Argument, ExecutionContext, ExecutionPolicy, SciOxideError};
use scioxide_linalg::{VectorView, VectorViewMut};

use decimation::Decimation;

use crate::complex::ComplexCore;
use crate::kernel::block::Run;
use crate::kernel::block::{Block, BlockLanes, Load, ROWS, Reals, Scale, Store, WIDTH};
use crate::lanes::{LaneMode, LanePairs};
use crate::numeric::reduced_root;
use crate::numeric::scalar_from_usize_rounded;
use crate::scaling::{BinaryScale, complex_max, range_exponent, real_max, restore_complex};
use crate::types::FftPlanDiagnostics;

/// Reusable real-to-complex FFT plan for one fixed logical length.
///
/// The forward result contains `floor(n / 2) + 1` non-negative-frequency
/// bins. Sequential execution with supplied output and scratch performs no
/// heap allocation. For bounded inputs, normalized round trips are expected
/// within `128 * epsilon(S) * max(1, log2(n))`, scaled by the largest input
/// magnitude when it exceeds one.
#[derive(Debug, Clone)]
#[must_use]
pub(crate) struct RealFftPlan<S: scioxide_core::RealScalar> {
    n: usize,
    spectrum_len: usize,
    /// Unscaled output bound; infinite when any stored child is a provider.
    growth: S,
    backend: KernelBackend,
    scratch_len: usize,
    plan_bytes: usize,
    diagnostics: FftPlanDiagnostics,
    algorithm: RealAlgorithm<S>,
    /// Native N-D lane blocks for smooth half-length children through 256 points.
    blocks: Option<scioxide_core::backend::KernelDispatch>,
    /// Natural slot to native digit-order row for smooth lane blocks.
    digit_rows: Vec<usize>,
}

#[derive(Debug, Clone)]
enum RealAlgorithm<S: scioxide_core::RealScalar> {
    Singleton,
    Even {
        complex: ComplexCore<S>,
        twiddles: Vec<Complex<S>>,
    },
    Decimated(Decimation<S>),
    Odd {
        complex: ComplexCore<S>,
    },
}

impl<S: scioxide_core::RealScalar> RealFftPlan<S> {
    /// Visit each pair child, then its residue, without allocating a stack.
    pub(crate) fn resolved_backends(&self) -> impl Iterator<Item = crate::FftBackend> + '_ {
        let mut next = Some(self);
        core::iter::from_fn(move || {
            let plan = next.take()?;
            Some(match &plan.algorithm {
                RealAlgorithm::Singleton => crate::FftBackend::Cpu(plan.backend),
                RealAlgorithm::Decimated(decimation) => {
                    let (pair, residue) = decimation.children();
                    next = Some(residue);
                    pair.resolved_backend()
                }
                RealAlgorithm::Even { complex, .. } | RealAlgorithm::Odd { complex } => {
                    complex.resolved_backend()
                }
            })
        })
    }

    pub(crate) fn workspace_byte_limit(&self) -> usize {
        let work = self.scratch_len - self.spectrum_len;
        let bytes = |len: usize| len * size_of::<Complex<S>>();
        let child = match &self.algorithm {
            RealAlgorithm::Singleton => None,
            RealAlgorithm::Decimated(decimation) => {
                Some((decimation.work_byte_limit(work), bytes(work)))
            }
            RealAlgorithm::Even { complex, .. } | RealAlgorithm::Odd { complex } => Some((
                complex.bounded_workspace_byte_limit(),
                bytes(complex.bounded_scratch_len()),
            )),
        };
        crate::workspace::nested_byte_limit(usize::MAX, bytes(self.scratch_len), child)
    }
    /// Return the exact logical real-signal length.
    #[must_use]
    pub(crate) const fn size(&self) -> usize {
        self.n
    }

    /// Return the packed non-negative-frequency output length.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn spectrum_len(&self) -> usize {
        self.spectrum_len
    }

    /// Return resident plan bytes, including child plans and twiddles.
    #[must_use]
    pub(crate) const fn plan_bytes(&self) -> usize {
        self.plan_bytes
    }

    pub(crate) const fn diagnostics(&self) -> FftPlanDiagnostics {
        self.diagnostics
    }

    /// Reusable natural-order lanes and native stage scratch for N-D blocks.
    pub(crate) fn lane_scratch_len(&self) -> usize {
        let transposed = self.blocks.is_some()
            && self
                .even_parts()
                .and_then(|(child, _)| child.smooth_stages())
                .is_some_and(|stages| !stages.native_rows_supported());
        self.scratch_len
            .max(if transposed { WIDTH * self.n } else { 0 })
    }

    /// Return the minimum complex scratch length for either direction.
    #[must_use]
    pub(crate) const fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Length of the complex-promoted leaf ending the odd decimation chain.
    fn leaf_len(&self) -> usize {
        match &self.algorithm {
            RealAlgorithm::Decimated(decimation) => decimation.leaf_len(),
            _ => self.n,
        }
    }

    /// Whether samples `2t` and `2t + 1` share staged complex slot `t`
    /// (even and decimated plans).
    pub(crate) const fn is_packed(&self) -> bool {
        matches!(
            self.algorithm,
            RealAlgorithm::Even { .. } | RealAlgorithm::Decimated(_)
        )
    }

    /// Half-length complex child and split twiddles for an even real FFT.
    pub(crate) fn even_parts(&self) -> Option<(&ComplexCore<S>, &[Complex<S>])> {
        match &self.algorithm {
            RealAlgorithm::Even { complex, twiddles } => Some((complex, twiddles)),
            _ => None,
        }
    }

    /// Bound on unscaled outputs after normalization; providers are checked
    /// individually.
    fn output_bound(&self, scale: S) -> S {
        if self.growth.is_infinite() {
            S::INFINITY
        } else {
            self.growth * scale
        }
    }
}

/// Destination bins of a forward transform.
enum Bins<'a, S: scioxide_core::RealScalar> {
    Contiguous(&'a mut [Complex<S>]),
    Strided(VectorViewMut<'a, Complex<S>>),
}
