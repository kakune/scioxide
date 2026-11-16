//! Event types, event options and the crate-private step result.

use core::fmt;

use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError, StepTolerance};
use scioxide_linalg::Vector;

use crate::stepper::StepRequest;

/// Selects which sign changes of an event function count as events.
///
/// Directions are measured from the start to the end of a step, so they follow the
/// integration order in both forward and backward integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventDirection {
    /// Detect crossings from negative to non-negative.
    Rising,
    /// Detect crossings from positive to non-positive.
    Falling,
    /// Detect crossings in either direction.
    Either,
}

/// Describes how a detected event function behaved at the event.
///
/// [`EventDirection`] selects which crossings to detect; this enum describes one
/// detected event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventOccurrence {
    /// The event function has a local extremum where `|g|` is within the contact
    /// tolerance, without a sign change. This is an estimate, not a proof of contact.
    Tangency,
    /// The event function changed from negative to non-negative in integration order.
    Rising,
    /// The event function changed from positive to non-positive in integration order.
    Falling,
}

/// Result of a single solver step.
///
/// Returned by [`Stepper::step`](crate::stepper::Stepper::step) to communicate the
/// error estimate and optional dense output to the driver. Borrowed storage remains
/// valid until the method workspace is mutably borrowed again.
pub(crate) struct StepResult<
    'a,
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S> = S,
> {
    pub(crate) stages: Option<&'a [Vector<T>]>,
    pub(crate) endpoint_rhs: Option<&'a [T]>,
    /// Method change applied at the accepted base state of this proposal.
    pub(crate) method_switch: Option<crate::OdeMethodSwitch<S>>,
    /// Power of the step in the local error estimate, when method-dependent.
    pub(crate) error_order: Option<usize>,
    /// Method-specific upper bound on the next step magnitude ratio.
    pub(crate) maximum_step_growth: Option<S>,
    /// Signed local component errors in state units, if estimated.
    pub(crate) raw_error: Option<&'a [T]>,
    /// Norm under the request tolerance and max(old, new) component scales.
    /// Checked methods may reuse a norm already computed for order selection.
    pub(crate) normalized_error: Option<S>,
    pub(crate) dense_output: Option<&'a crate::dense_output::DenseOutputStorage<S, T>>,
    /// Requested dense output waits for acceptance, through `Stepper::complete_dense`.
    pub(crate) dense_deferred: bool,
    pub(crate) num_nonlinear_iters: usize,
    pub(crate) num_jacobian_cache_hits: usize,
    pub(crate) num_factor_cache_hits: usize,
}

impl<'a, T: NumericScalar<Real = S>, S: RealScalar + NumericScalar<Real = S>> StepResult<'a, S, T> {
    /// Create a proposal with optional signed local component errors in state units.
    /// Fixed-step methods without an estimator supply `None`.
    ///
    /// # Errors
    /// Returns a non-finite input error for a non-finite error component.
    #[cfg(test)]
    pub(crate) fn new(
        raw_error: Option<&'a [T]>,
        dense_output: Option<&'a crate::dense_output::DenseOutputStorage<S, T>>,
    ) -> Result<Self, SciOxideError> {
        if raw_error.is_some_and(|errors| errors.iter().any(|e| !e.has_finite_components())) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "local errors must be finite",
            ));
        }
        Ok(Self::from_checked_parts(raw_error, dense_output))
    }

    /// Internal kernels have already checked immutable error components.
    pub(crate) const fn from_checked_parts(
        raw_error: Option<&'a [T]>,
        dense_output: Option<&'a crate::dense_output::DenseOutputStorage<S, T>>,
    ) -> Self {
        Self {
            stages: None,
            endpoint_rhs: None,
            method_switch: None,
            error_order: None,
            maximum_step_growth: None,
            raw_error,
            normalized_error: None,
            dense_output,
            dense_deferred: false,
            num_nonlinear_iters: 0,
            num_jacobian_cache_hits: 0,
            num_factor_cache_hits: 0,
        }
    }

    pub(crate) fn validate_for_step(
        &self,
        request: StepRequest<'_, S, T>,
        t_old: S,
        t_new: S,
        dimension: usize,
    ) -> Result<(), SciOxideError> {
        // Checked built-in kernels establish finite immutable errors; only the
        // state-dependent shape needs validation.
        if let Some(errors) = self.raw_error
            && errors.len() != dimension
        {
            return Err(SciOxideError::dimension_mismatch(dimension, errors.len()));
        }
        match (request.requires_dense_output(), self.dense_output) {
            (true, Some(dense)) => {
                let (dense_old, dense_new) = dense.interval();
                if dense_old != t_old || dense_new != t_new || dense.dimension() != dimension {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("self"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "interpolant interval or dimension does not match the completed step",
                    ));
                }
            }
            (true, None) if self.dense_deferred => {}
            (true, None) => {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("self"),
                    scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                    "step did not provide requested dense output",
                ));
            }
            (false, Some(_)) => {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("self"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "step returned unrequested dense output",
                ));
            }
            (false, None) => {}
        }
        Ok(())
    }
}

impl<T: NumericScalar<Real = S>, S: RealScalar + NumericScalar<Real = S>> fmt::Debug
    for StepResult<'_, S, T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StepResult")
            .field("raw_error", &self.raw_error)
            .field("has_dense_output", &self.dense_output.is_some())
            .field("num_nonlinear_iters", &self.num_nonlinear_iters)
            .field("num_jacobian_cache_hits", &self.num_jacobian_cache_hits)
            .field("num_factor_cache_hits", &self.num_factor_cache_hits)
            .finish_non_exhaustive()
    }
}

/// Chooses where the detector samples each event function within an accepted step.
///
/// The detector finds a root only where two samples differ in sign, so it can miss
/// pairs of roots between samples. No policy guarantees that every root is found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EventSearchPolicy {
    /// Samples only the step endpoints (the default).
    #[default]
    Endpoints,
    /// Samples the endpoints of this many equal subintervals of each step.
    Subdivide(Budget),
    /// Samples like `Subdivide` and also locates stationary points of `g` from
    /// [`EventFunction::total_derivative`](crate::EventFunction::total_derivative), which
    /// finds pairs of roots and tangencies between samples.
    DerivativeAssisted(Budget),
}

impl EventSearchPolicy {
    /// Returns the number of subintervals searched per step and event: one for
    /// `Endpoints`.
    pub const fn subdivisions(self) -> usize {
        match self {
            Self::Endpoints => 1,
            Self::Subdivide(n) | Self::DerivativeAssisted(n) => n.get(),
        }
    }
}

/// Describes how one event was located on the continuous extension of a step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventEvidence<S: RealScalar> {
    /// The event function returned exactly zero at a sampled time.
    SampledZero,
    /// A bracket with opposite signs, or with a zero at one end, contained the root,
    /// which was then refined.
    Crossing {
        /// Final bracket of the root, with ends in increasing time.
        bracket: scioxide_core::Interval<S>,
        /// `|g|` at the reported event time.
        residual: S,
    },
    /// A stationary point of `g` passed the contact tolerance; this is an estimate.
    Contact {
        /// Final bracket of the stationary point, with ends in increasing time.
        bracket: scioxide_core::Interval<S>,
        /// `|g|` at the stationary point.
        residual: S,
    },
}

/// Summary of the event search of one integration, reported also when no event was
/// found.
///
/// The search samples finitely many points, so it cannot rule out missed roots or
/// stationary points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventSearchEvidence {
    pub(crate) policy: EventSearchPolicy,
    pub(crate) evaluations: usize,
}

impl EventSearchEvidence {
    /// Returns the search policy that was used.
    pub const fn policy(&self) -> EventSearchPolicy {
        self.policy
    }
    /// Returns the number of calls to event functions and their total derivatives,
    /// including root refinement.
    ///
    /// It equals the [`scioxide_core::EvaluationRole::Event`] count of the integration and
    /// its nested root solves.
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }
}

/// Options for event detection and root refinement, separate from the error control
/// of the integration.
///
/// After a sign change is bracketed, the detector refines the root with the TOMS 748
/// bracketing method of Alefeld, Potra and Shi on the continuous extension of the step.
///
/// # References
///
/// G. E. Alefeld, F. A. Potra and Y. Shi, "Algorithm 748: Enclosing zeros of continuous
/// functions", *ACM Trans. Math. Softw.* 21(3), 327–344, 1995.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OdeEventOptions<S: RealScalar> {
    search: EventSearchPolicy,
    contact: scioxide_core::ResidualTolerance<S>,
    contact_scale: S,
    localization: StepTolerance<S>,
    max_iterations: Budget,
}

impl<S: RealScalar> OdeEventOptions<S> {
    /// Creates the default options: endpoint sampling, root refinement to absolute and
    /// relative time tolerances of `4ε` with at most 64 iterations, and a contact scale
    /// of one, where `ε` is the machine epsilon of `S`.
    pub fn new() -> Self {
        Self {
            search: EventSearchPolicy::Endpoints,
            contact: scioxide_core::ResidualTolerance::default(),
            contact_scale: S::ONE,
            localization: scioxide_rootfind::ScalarRootOptions::<S>::default().step_tolerance(),
            max_iterations: Budget::ONE.saturating_add(63),
        }
    }

    /// Sets the search policy.
    ///
    /// [`EventSearchPolicy::DerivativeAssisted`] requires every event to implement
    /// [`EventFunction::total_derivative`](crate::EventFunction::total_derivative); plans
    /// otherwise return `InvalidInput`.
    pub const fn with_search(mut self, search: EventSearchPolicy) -> Self {
        self.search = search;
        self
    }

    /// Sets the contact tolerance and its reference scale, in units of the event value.
    ///
    /// A stationary point of `g` counts as a tangency when `|g| ≤ atol + rtol · scale`.
    /// Plans return `InvalidInput` (`OutOfRange`) for a scale that is zero, negative or
    /// not finite.
    pub const fn with_contact_tolerance(
        mut self,
        tolerance: scioxide_core::ResidualTolerance<S>,
        scale: S,
    ) -> Self {
        self.contact = tolerance;
        self.contact_scale = scale;
        self
    }

    /// Rejects a non-finite or nonpositive contact scale or an unusable contact tolerance.
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        if !self.contact_scale.is_finite() || self.contact_scale <= S::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "event contact scale must be finite and positive",
            ));
        }
        self.contact.threshold(self.contact_scale)?;
        Ok(())
    }

    /// Returns the search policy.
    pub const fn search(self) -> EventSearchPolicy {
        self.search
    }
    /// Returns the contact tolerance.
    pub const fn contact_tolerance(self) -> scioxide_core::ResidualTolerance<S> {
        self.contact
    }
    /// Returns the reference scale of the contact tolerance.
    pub const fn contact_scale(self) -> S {
        self.contact_scale
    }

    /// Returns the time tolerance of root refinement.
    pub const fn localization(self) -> StepTolerance<S> {
        self.localization
    }

    /// Returns the iteration limit of root refinement per bracketed event.
    pub const fn max_iterations(self) -> Budget {
        self.max_iterations
    }

    /// Sets the time tolerance and iteration limit of root refinement.
    ///
    /// Refinement ends when the bracket meets the tolerance or contains no
    /// representable time strictly inside, so a zero tolerance refines to adjacent
    /// floating-point times.
    pub const fn with_localization(
        mut self,
        tolerance: StepTolerance<S>,
        max_iterations: Budget,
    ) -> Self {
        self.localization = tolerance;
        self.max_iterations = max_iterations;
        self
    }
}

impl<S: RealScalar> Default for OdeEventOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
