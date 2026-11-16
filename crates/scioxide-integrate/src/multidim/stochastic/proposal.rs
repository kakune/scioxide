//! Proposal densities on the unit cube for importance sampling.

use scioxide_core::{RealScalar, SciOxideError};

/// Uniform variates in `[0, 1)` passed to a [`MonteCarloProposal`].
///
/// The constructor checks every value once, so a proposal can use them without
/// checking again.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::ProposalUniforms;
/// let uniforms = ProposalUniforms::new(&[0.0_f64, 0.5])?;
/// assert_eq!(uniforms.as_slice(), &[0.0, 0.5]);
/// assert!(ProposalUniforms::new(&[1.0]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ProposalUniforms<'a, S: RealScalar>(&'a [S]);

impl<'a, S: RealScalar> ProposalUniforms<'a, S> {
    /// Wraps uniforms that the source has already checked to be finite and in `[0, 1)`.
    pub(super) const fn from_admitted(uniforms: &'a [S]) -> Self {
        Self(uniforms)
    }
    /// Checks that every value of `uniforms` is finite and in `[0, 1)`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` (origin `Callback`) if a value is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if a value lies outside `[0, 1)`.
    #[inline]
    pub fn new(uniforms: &'a [S]) -> Result<Self, SciOxideError> {
        for &uniform in uniforms {
            if crate::kernel::checked_callback_value(uniform)? < S::ZERO || uniform >= S::ONE {
                return Err(super::invalid(
                    scioxide_core::argument!("source"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "uniforms must lie in [0, 1)",
                ));
            }
        }
        Ok(Self(uniforms))
    }

    /// Returns the uniforms.
    pub const fn as_slice(self) -> &'a [S] {
        self.0
    }
}

/// Proposal density `q` on the unit cube `[0, 1)ᵈ` for importance sampling.
///
/// [`sample`](Self::sample) maps `m = uniform_count()` independent uniforms `u`
/// to a point `y` of the unit cube and returns its density `q(y)` with respect
/// to Lebesgue measure on the cube. [`MonteCarloIntegrator`](crate::MonteCarloIntegrator)
/// then maps `y` affinely to the integration box and records the observation
/// `V f(x) / q(y)`, where `x` is the point in the box and `V` the box volume.
///
/// The estimate is unbiased only if `q` integrates to one over the cube and is
/// positive wherever `f` is nonzero. The plan cannot check this from samples.
/// The importance weights `1/q` are not self-normalized. `sample` must be a
/// deterministic function of `u`. A trained [`VegasMap`](crate::VegasMap) is
/// such a proposal; rejection sampling and adaptive or Markov chain samplers are
/// not.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideError;
/// use scioxide_integrate::{MonteCarloProposal, ProposalUniforms};
///
/// /// Density `q(y) = (1 + 2y) / 2` on `[0, 1)`, sampled by inverting its CDF.
/// struct Linear;
/// impl MonteCarloProposal<f64> for Linear {
///     fn dimension(&self) -> usize {
///         1
///     }
///     fn uniform_count(&self) -> usize {
///         1
///     }
///     fn sample(&self, u: ProposalUniforms<'_, f64>, y: &mut [f64]) -> Result<f64, SciOxideError> {
///         let u = u.as_slice()[0];
///         y[0] = 4.0 * u / ((1.0 + 8.0 * u).sqrt() + 1.0);
///         Ok((1.0 + 2.0 * y[0]) / 2.0)
///     }
/// }
/// let mut y = [0.0];
/// // u = 0 maps to y = 0, where q(0) = 1/2.
/// assert_eq!(Linear.sample(ProposalUniforms::new(&[0.0])?, &mut y)?, 0.5);
/// # Ok::<(), SciOxideError>(())
/// ```
pub trait MonteCarloProposal<S: RealScalar> {
    /// Returns the dimension `d` of the points.
    fn dimension(&self) -> usize;
    /// Returns the number `m` of uniforms consumed per point.
    fn uniform_count(&self) -> usize;
    /// Writes a point of `[0, 1)ᵈ` to `point` and returns its density.
    ///
    /// `uniforms` has length `uniform_count()` and `point` has length
    /// `dimension()`.
    ///
    /// # Errors
    ///
    /// An implementation returns an error if it cannot map `uniforms`. The plan
    /// returns `InvalidInput` if the density is not finite and positive, or a
    /// coordinate lies outside `[0, 1)`, and `NonFinite` if a coordinate is NaN or
    /// infinite.
    fn sample(
        &self,
        uniforms: ProposalUniforms<'_, S>,
        point: &mut [S],
    ) -> Result<S, SciOxideError>;
}

/// Uniform proposal `q = 1` on the unit cube `[0, 1)ᵈ`; `d = 0` is allowed.
///
/// [`MonteCarloIntegrator::new`](crate::MonteCarloIntegrator::new) uses it. Its
/// [`sample`](MonteCarloProposal::sample) copies the uniforms to the point and
/// returns `DimensionMismatch` if either slice length differs from `d`.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{MonteCarloProposal, ProposalUniforms, UniformProposal};
/// let proposal = UniformProposal::new(2);
/// let mut point = [0.0; 2];
/// let uniforms = ProposalUniforms::new(&[0.25, 0.5])?;
/// assert_eq!(proposal.sample(uniforms, &mut point)?, 1.0);
/// assert_eq!(point, [0.25, 0.5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformProposal {
    dimension: usize,
}

impl UniformProposal {
    /// Creates the uniform proposal on the `dimension`-dimensional unit cube.
    #[must_use]
    pub const fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl<S: RealScalar> MonteCarloProposal<S> for UniformProposal {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn uniform_count(&self) -> usize {
        self.dimension
    }
    #[inline]
    fn sample(
        &self,
        uniforms: ProposalUniforms<'_, S>,
        point: &mut [S],
    ) -> Result<S, SciOxideError> {
        let uniforms = uniforms.as_slice();
        crate::kernel::shape(self.dimension, uniforms.len())?;
        crate::kernel::shape(self.dimension, point.len())?;
        point.copy_from_slice(uniforms);
        Ok(S::ONE)
    }
}
