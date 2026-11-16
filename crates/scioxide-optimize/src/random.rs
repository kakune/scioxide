//! The counted random source of global search and workflow variation.
//!
//! [`SearchRandom`] forwards every native and uniform word of an underlying
//! generator unchanged. Before each draw it charges one sample to the caller's
//! sample limit. A refused draw records the stop reason, so the caller can
//! return its best accepted state instead of failing.

use scioxide_core::{RealScalar, SciOxideError};
use scioxide_random::BitGenerator;

use crate::OptimizeTermination;
use crate::context::SolveSession;

/// Charge for one random draw against the caller's resources.
pub(crate) trait DrawAdmission {
    /// Charges one draw. `Ok(Some(stop))` refuses it and names the stop.
    fn admit(&mut self) -> Result<Option<OptimizeTermination>, SciOxideError>;
    /// Error returned to the sampling algorithm for a draw refused by `stop`.
    fn refusal(&self, stop: OptimizeTermination) -> SciOxideError;
}

/// Global searches charge one sample of their solve session per draw.
impl DrawAdmission for &SolveSession<'_, '_, '_> {
    fn admit(&mut self) -> Result<Option<OptimizeTermination>, SciOxideError> {
        match self.samples(1) {
            Ok(()) => Ok(None),
            Err(error) => self.stopped().map_or(Err(error), |stop| Ok(Some(stop))),
        }
    }
    fn refusal(&self, stop: OptimizeTermination) -> SciOxideError {
        self.stop_error(stop).unwrap_or_else(|error| error)
    }
}

/// Draws random words for start designs and basin proposals, counting each
/// draw toward the solve's sample limit.
///
/// Every native word, uniform word and eight-byte chunk counts as one sample
/// and is charged before the underlying generator advances. A draw refused by
/// the sample limit therefore consumes neither the limit nor the stream
/// position, and the search stops with `SampleLimit`. The words are the
/// generator's own, unchanged. [`Self::uniform`] decodes one uniform 64-bit
/// word for every scalar width. See
/// [Randomness and parallelism](crate#randomness-and-parallelism).
pub struct SearchRandom<'a> {
    generator: &'a mut dyn BitGenerator,
    admission: &'a mut dyn DrawAdmission,
    stopped: Option<OptimizeTermination>,
}

impl core::fmt::Debug for SearchRandom<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SearchRandom")
            .field("position", &self.generator.position())
            .field("stopped", &self.stopped)
            .finish()
    }
}

impl<'a> SearchRandom<'a> {
    /// Counts every draw of `generator` with the given draw counter.
    pub(crate) fn new(
        generator: &'a mut dyn BitGenerator,
        admission: &'a mut dyn DrawAdmission,
    ) -> Self {
        Self {
            generator,
            admission,
            stopped: None,
        }
    }

    /// Stop that refused a draw, if any.
    pub(crate) const fn stopped(&self) -> Option<OptimizeTermination> {
        self.stopped
    }

    fn admit(&mut self) -> Result<(), SciOxideError> {
        match self.admission.admit()? {
            None => Ok(()),
            Some(stop) => {
                self.stopped = Some(stop);
                Err(self.admission.refusal(stop))
            }
        }
    }

    /// Draws a scalar uniformly from `[0, 1)` by decoding one uniform 64-bit word.
    ///
    /// # Errors
    ///
    /// Returns the stop error when the sample limit, cancellation or a
    /// deadline refuses the draw, and propagates generator errors.
    pub fn uniform<S: RealScalar>(&mut self) -> Result<S, SciOxideError> {
        self.next_uniform_u64().map(scioxide_random::decode_u64)
    }
}

impl BitGenerator for SearchRandom<'_> {
    fn position(&self) -> Option<u128> {
        self.generator.position()
    }
    fn stream_identity(&self) -> Option<scioxide_random::StreamIdentity<'_>> {
        self.generator.stream_identity()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        self.admit()?;
        self.generator.next_u32()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.admit()?;
        self.generator.next_u64()
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.admit()?;
        self.generator.next_uniform_u32()
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.admit()?;
        self.generator.next_uniform_u64()
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
        for chunk in output.chunks_mut(8) {
            let bytes = self.next_u64()?.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        Ok(())
    }
}

impl<S: RealScalar> scioxide_random::Rng<S> for SearchRandom<'_> {
    fn next_float(&mut self) -> Result<S, SciOxideError> {
        self.uniform()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_random::{Ranlux, SeedableRng};

    struct Allowance {
        remaining: usize,
        admitted: usize,
    }
    impl DrawAdmission for Allowance {
        fn admit(&mut self) -> Result<Option<OptimizeTermination>, SciOxideError> {
            if self.remaining == 0 {
                return Ok(Some(OptimizeTermination::SampleLimit));
            }
            self.remaining -= 1;
            self.admitted += 1;
            Ok(None)
        }
        fn refusal(&self, _: OptimizeTermination) -> SciOxideError {
            SciOxideError::invalid_input(
                scioxide_core::argument!("allowance"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "sample allowance exhausted",
            )
        }
    }

    #[test]
    fn admitted_words_are_native_and_refusals_do_not_advance() {
        let mut generator = Ranlux::from_seed(42);
        let mut expected = generator.clone();
        let mut allowance = Allowance {
            remaining: 3,
            admitted: 0,
        };
        let mut random = SearchRandom::new(&mut generator, &mut allowance);
        assert_eq!(
            random.next_uniform_u32().unwrap(),
            expected.next_uniform_u32().unwrap()
        );
        assert_eq!(random.next_u32().unwrap(), expected.next_u32().unwrap());
        assert_eq!(
            random.uniform::<f64>().unwrap().to_bits(),
            scioxide_random::decode_u64::<f64>(expected.next_uniform_u64().unwrap()).to_bits()
        );
        assert!(random.stopped().is_none());
        assert!(random.next_uniform_u64().is_err());
        assert_eq!(random.stopped(), Some(OptimizeTermination::SampleLimit));
        assert_eq!(random.position(), expected.position());
        assert_eq!(allowance.admitted, 3);
    }

    #[test]
    fn session_admission_charges_samples_and_reports_the_stop() {
        use scioxide_core::{Budget, BudgetKind, ExecutionContext, SolveBudgets, SolveState};
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(BudgetKind::Sample, Budget::ONE.saturating_add(1));
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", "test", 1),
                    1,
                    &(),
                ),
                &mut budgets,
            )
            .unwrap();
        {
            let session = SolveSession::new(&mut context, scioxide_core::argument!("context"));
            let mut generator = Ranlux::from_seed(42);
            let mut expected = generator.clone();
            let mut admission = &session;
            let mut random = SearchRandom::new(&mut generator, &mut admission);
            assert_eq!(
                random.next_uniform_u32().unwrap(),
                expected.next_uniform_u32().unwrap()
            );
            assert_eq!(
                random.next_uniform_u64().unwrap(),
                expected.next_uniform_u64().unwrap()
            );
            let error = random.next_uniform_u64().unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            ));
            assert_eq!(random.stopped(), Some(OptimizeTermination::SampleLimit));
            assert_eq!(random.position(), expected.position());
            assert_eq!(session.stopped(), Some(OptimizeTermination::SampleLimit));
        }
        assert_eq!(context.budget_report(BudgetKind::Sample).consumed(), 2);
    }
}
