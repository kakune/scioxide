//! CZT specification and byte limits.
use super::CztContour;
use crate::{FftBackendStrategy, Normalization};
use scioxide_core::{ExecutionPolicy, RealScalar};

/// Byte limits for preparing and executing a [`CztPlan`](crate::CztPlan).
///
/// Both limits default to `usize::MAX`. The preparation limit counts the
/// storage the CZT plan itself holds at once, but not the FFT it prepares or
/// allocator overhead. The workspace limit counts the whole execution
/// workspace, including staging for axis transforms. Exceeding either is a
/// `ResourceLimit` error.
#[derive(Debug, Clone, Copy)]
pub struct CztLimits {
    /// Maximum bytes held at once during preparation.
    pub preparation_bytes: usize,
    /// Maximum bytes of an execution workspace.
    pub workspace_bytes: usize,
}
impl Default for CztLimits {
    fn default() -> Self {
        Self {
            preparation_bytes: usize::MAX,
            workspace_bytes: usize::MAX,
        }
    }
}

/// Specifies a chirp z-transform: lengths, contour, normalization and
/// execution options.
///
/// The transform is defined on [`CztPlan`](crate::CztPlan). The spec defaults to
/// sequential execution, automatic backend selection and no byte limits.
#[derive(Debug, Clone, Copy)]
pub struct CztPlanSpec<S: RealScalar> {
    pub(super) n: usize,
    pub(super) m: usize,
    pub(super) contour: CztContour<S>,
    pub(super) normalization: Normalization,
    pub(super) policy: ExecutionPolicy,
    pub(super) backend: FftBackendStrategy,
    pub(super) limits: CztLimits,
}
impl<S: RealScalar> CztPlanSpec<S> {
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the backend strategy.
    pub const fn backend_strategy(&self) -> FftBackendStrategy {
        self.backend
    }
    /// Returns the byte limits.
    pub const fn limits(&self) -> CztLimits {
        self.limits
    }

    /// Creates a spec for `n` input samples and `m` contour points.
    ///
    /// Preparation checks that `n` and `m` are positive.
    pub fn new(n: usize, m: usize, contour: CztContour<S>, normalization: Normalization) -> Self {
        Self {
            n,
            m,
            contour,
            normalization,
            policy: ExecutionPolicy::Sequential,
            backend: FftBackendStrategy::Automatic,
            limits: CztLimits::default(),
        }
    }
    /// Sets the execution policy.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Sets the backend strategy for the FFT used by Bluestein's algorithm.
    pub const fn with_backend_strategy(mut self, backend: FftBackendStrategy) -> Self {
        self.backend = backend;
        self
    }
    /// Sets the byte limits, which preparation and
    /// `workspace_requirements` enforce.
    pub const fn with_limits(mut self, limits: CztLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Returns the number of input samples `n`.
    pub const fn input_len(&self) -> usize {
        self.n
    }
    /// Returns the number of contour points `m`.
    pub const fn output_len(&self) -> usize {
        self.m
    }
    /// Returns the contour.
    ///
    /// # Examples
    ///
    /// Read the contour points of a prepared plan with [`CztContour::point`]:
    ///
    /// ```
    /// use scioxide_fft::{CztContour, CztPlan, CztPlanSpec, Normalization};
    ///
    /// // The second point of the four-point DFT contour is exp(2πi/4) = i.
    /// let spec = CztPlanSpec::new(4, 4, CztContour::dft(4)?, Normalization::Unscaled);
    /// let plan = CztPlan::<f64>::new(spec)?;
    /// let point = plan.spec().contour().point(1)?;
    /// assert!(point.re.abs() < 1e-12 && (point.im - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn contour(&self) -> CztContour<S> {
        self.contour
    }
    /// Returns the normalization; its scale uses the input length `n`.
    pub const fn normalization(&self) -> Normalization {
        self.normalization
    }
}
