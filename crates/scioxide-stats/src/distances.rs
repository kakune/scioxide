//! Wasserstein-1 and energy distances between one-dimensional discrete distributions.
use crate::KernelWeights;
use crate::arithmetic::Compensated;
use crate::error_helpers::{check_weights, invalid, mass};
use crate::ranking::finite_cmp;
use alloc::vec::Vec;

use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::{RealScalar, SciOxideError};

/// Distance between two one-dimensional distributions with CDFs `F` and `G`.
///
/// # References
///
/// - S. S. Vallender, "Calculation of the Wasserstein distance between
///   probability distributions on the line", *Theory Probab. Appl.*, 1973.
/// - G. J. Székely and M. L. Rizzo, "Energy statistics: A class of statistics
///   based on distances", *J. Stat. Plan. Inference*, 2013.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionDistance {
    /// Wasserstein-1 (earth mover's) distance with ground distance `|x − y|`,
    /// `W₁ = ∫ |F(x) − G(x)| dx`.
    Wasserstein,
    /// Energy distance `D = (2 ∫ (F(x) − G(x))² dx)^(1/2)`, the square root of
    /// the energy statistic `2E|X − Y| − E|X − X′| − E|Y − Y′|`.
    Energy,
}
#[derive(Clone, Copy)]
struct Atom<S: RealScalar> {
    value: S,
    mass: Scaled<f64>,
    left: bool,
}
/// Workspace for [`distribution_distance_with_options`]: scratch for sorting the support
/// points of both distributions together.
///
/// See [reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct DistanceWorkspace<S: RealScalar> {
    atoms: Vec<Atom<S>>,
}
/// Storage that [`DistanceWorkspace`] needs for given support sizes.
#[derive(Clone, Copy, Debug)]
pub struct DistanceRequirements<S: RealScalar> {
    scalar: core::marker::PhantomData<S>,
    samples: usize,
    max_bytes: usize,
}
impl<S: RealScalar> DistanceRequirements<S> {
    /// Returns the heap bytes the workspace needs, excluding allocator overhead.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn required_bytes(self) -> Result<usize, SciOxideError> {
        crate::estimation::storage_bytes(&[(self.samples, core::mem::size_of::<Atom<S>>())])
    }
}
impl<S: RealScalar> DistanceWorkspace<S> {
    /// Returns the storage needed for distributions with `left` and `right` support points.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if it exceeds the byte limit in `options`.
    pub fn workspace_requirements(
        left: usize,
        right: usize,
        options: crate::StatisticsOptions,
    ) -> Result<DistanceRequirements<S>, SciOxideError> {
        let samples = left.checked_add(right).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("distance_support"))
        })?;
        let req = DistanceRequirements {
            scalar: core::marker::PhantomData,
            samples,
            max_bytes: options.max_bytes(),
        };
        crate::error_helpers::limit_bytes(req.required_bytes()?, req.max_bytes)?;
        Ok(req)
    }
    /// Returns whether the workspace is large enough for `req`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace already holds more than the byte limit of `req`.
    pub fn check(
        &self,
        req: &DistanceRequirements<S>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        crate::error_helpers::limit_bytes(
            crate::estimation::storage_bytes(&[(
                self.atoms.capacity(),
                core::mem::size_of::<Atom<S>>(),
            )])?,
            req.max_bytes,
        )?;
        Ok(if self.atoms.len() >= req.samples {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace to fit `req`, keeping storage that is large enough.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of `req`.
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn try_reserve(&mut self, req: &DistanceRequirements<S>) -> Result<(), SciOxideError> {
        if self.check(req)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        crate::error_helpers::limit_bytes(req.required_bytes()?, req.max_bytes)?;
        self.reserve(req.samples, 0)
    }
    /// Creates a workspace that fits `req`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`try_reserve`](Self::try_reserve).
    pub fn workspace(req: &DistanceRequirements<S>) -> Result<Self, SciOxideError> {
        let mut ws = Self::default();
        ws.try_reserve(req)?;
        Ok(ws)
    }
    /// Computes the distance between the weighted empirical distributions of
    /// `left` and `right`, as [`distribution_distance_with_options`] does.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the workspace is too small for the two supports.
    /// - The errors of [`distribution_distance_with_options`] for invalid data.
    pub fn evaluate<'a, 'b, L: crate::BatchWeights<'a, S>, R: crate::BatchWeights<'b, S>>(
        &mut self,
        left: &[S],
        left_weights: L,
        right: &[S],
        right_weights: R,
        metric: DistributionDistance,
    ) -> Result<S, SciOxideError>
    where
        L::Model: crate::EmpiricalWeights,
        R::Model: crate::EmpiricalWeights,
    {
        self.evaluate_data(
            left,
            crate::statistic::legacy_weights(crate::weights::batch_sealed::Sealed::data(
                left_weights,
            )),
            right,
            crate::statistic::legacy_weights(crate::weights::batch_sealed::Sealed::data(
                right_weights,
            )),
            metric,
        )
    }
}
impl<S: RealScalar> Default for DistanceWorkspace<S> {
    fn default() -> Self {
        Self { atoms: Vec::new() }
    }
}
impl<S: RealScalar> DistanceWorkspace<S> {
    /// Grows the storage to `left + right` points; on error it is unchanged.
    ///
    /// # Errors
    /// Returns size overflow or allocation failure.
    fn reserve(&mut self, left: usize, right: usize) -> Result<(), SciOxideError> {
        let n = left.checked_add(right).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("distance_support"))
        })?;
        if self.atoms.len() < n {
            self.atoms = try_filled(
                n,
                Atom {
                    value: S::ZERO,
                    mass: Scaled::<f64>::ZERO,
                    left: false,
                },
                scioxide_core::quantity!("distance_atoms"),
            )?;
        }
        Ok(())
    }
    /// Computes the distance in O((n + m) log(n + m)). Each side may use mass
    /// weights or frequency weights.
    ///
    /// # Errors
    /// Fails on invalid weights, zero total weight on a side, a non-finite point
    /// with positive weight, a workspace that is too small, or a distance that
    /// cannot be represented in `S`.
    fn evaluate_data(
        &mut self,
        left: &[S],
        left_weights: Option<KernelWeights<'_, S>>,
        right: &[S],
        right_weights: Option<KernelWeights<'_, S>>,
        metric: DistributionDistance,
    ) -> Result<S, SciOxideError> {
        check_weights(
            left_weights,
            left.len(),
            scioxide_core::argument!("left_weights"),
        )?;
        check_weights(
            right_weights,
            right.len(),
            scioxide_core::argument!("right_weights"),
        )?;
        let n = left.len().checked_add(right.len()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("distance_support"))
        })?;
        if self.atoms.len() < n {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient distance workspace",
            ));
        }
        let mut len = 0;
        for (values, weights, is_left, values_argument, weights_argument) in [
            (
                left,
                left_weights,
                true,
                scioxide_core::argument!("left"),
                scioxide_core::argument!("left_weights"),
            ),
            (
                right,
                right_weights,
                false,
                scioxide_core::argument!("right"),
                scioxide_core::argument!("right_weights"),
            ),
        ] {
            let start = len;
            let mut total = Compensated::<Scaled<f64>>::zero();
            for (i, &x) in values.iter().enumerate() {
                let w = mass(weights, i, weights_argument)?;
                if w.is_zero() {
                    continue;
                }
                if !x.is_finite() {
                    return Err(invalid(
                        values_argument,
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "nonfinite distance support",
                    ));
                }
                self.atoms[len] = Atom {
                    value: x,
                    mass: w,
                    left: is_left,
                };
                len += 1;
                total = total.added(w)?;
            }
            let total = total.total()?;
            if total.is_zero() {
                return Err(invalid(
                    weights_argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "empty positive-mass distance support",
                ));
            }
            for atom in &mut self.atoms[start..len] {
                atom.mass = atom.mass.div(total)?;
            }
        }
        let atoms = &mut self.atoms[..len];
        atoms.sort_unstable_by(|a, b| finite_cmp(&a.value, &b.value));
        let mut cdf_with_options = Compensated::<Scaled<f64>>::zero();
        let mut integral = Compensated::<Scaled<f64>>::zero();
        let mut i = 0;
        while i < len {
            let x = atoms[i].value;
            while i < len && atoms[i].value == x {
                cdf_with_options = cdf_with_options.added(if atoms[i].left {
                    atoms[i].mass
                } else {
                    -atoms[i].mass
                })?;
                i += 1;
            }
            if i == len {
                break;
            }
            let gap =
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(atoms[i].value))?
                    .sub(Scaled::<f64>::from_value(
                        scioxide_core::RealScalar::into_f64(x),
                    )?)?;
            let delta = cdf_with_options.total()?;
            let value = match metric {
                DistributionDistance::Wasserstein => delta.abs(),
                DistributionDistance::Energy => delta.square()?,
            };
            integral = integral.added(gap.mul(value)?)?;
        }
        let integral = integral.total()?;
        let value = match metric {
            DistributionDistance::Wasserstein => integral,
            DistributionDistance::Energy => integral.mul(Scaled::<f64>::from_value(2.)?)?.sqrt()?,
        };
        value.round_to()
    }
}
/// Computes the Wasserstein-1 or energy distance between two weighted samples
/// on the real line.
///
/// Each sample defines a discrete distribution: its weights are divided by
/// their sum, and zero-weight points are ignored. The distance is computed
/// exactly from the two empirical CDFs, as a sum over the gaps between the
/// sorted support points, in O((n + m) log(n + m)) for `n` and `m` points. Use
/// [`DistanceWorkspace::evaluate`] to reuse the sorting storage.
///
/// # Errors
///
/// - `InvalidInput` if a weight slice has a different length from its points.
/// - `InvalidInput` if a mass weight is negative, NaN or infinite.
/// - `InvalidInput` if a sample has zero total weight, including an empty sample.
/// - `InvalidInput` if a point with positive weight is NaN or infinite.
/// - `ResourceLimit` if the sorting storage exceeds the byte limit in `options`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{DistributionDistance, StatisticsOptions, Unweighted, distribution_distance_with_options};
///
/// // Shifting a distribution by 1 moves it a Wasserstein-1 distance of exactly 1.
/// let d = distribution_distance_with_options(
///     &[0.0_f64, 2.0],
///     Unweighted,
///     &[1.0, 3.0],
///     Unweighted,
///     DistributionDistance::Wasserstein,
///     StatisticsOptions::new(),
/// )?;
/// assert!((d - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn distribution_distance_with_options<
    'a,
    'b,
    S: RealScalar,
    L: crate::BatchWeights<'a, S>,
    R: crate::BatchWeights<'b, S>,
>(
    left: &[S],
    left_weights: L,
    right: &[S],
    right_weights: R,
    metric: DistributionDistance,
    options: crate::StatisticsOptions,
) -> Result<S, SciOxideError>
where
    L::Model: crate::EmpiricalWeights,
    R::Model: crate::EmpiricalWeights,
{
    let req = DistanceWorkspace::<S>::workspace_requirements(left.len(), right.len(), options)?;
    let mut workspace = DistanceWorkspace::workspace(&req)?;
    workspace.evaluate(left, left_weights, right, right_weights, metric)
}

/// Computes the result of [`distribution_distance_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the configured form's errors for these defaults.
pub fn distribution_distance<S: RealScalar>(
    left: &[S],
    right: &[S],
    metric: DistributionDistance,
) -> Result<S, SciOxideError> {
    distribution_distance_with_options(
        left,
        crate::Unweighted,
        right,
        crate::Unweighted,
        metric,
        Default::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &[
                "options",
                "weights",
                "left_weights",
                "right_weights",
                "direction",
                "ties",
            ],
            scioxide_core::argument!("left"),
        )
    })
}
