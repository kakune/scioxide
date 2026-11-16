//! Dense multivariate estimators: covariance, shrinkage covariance, principal
//! component analysis and canonical correlation analysis.
//!
//! Rows are observations and columns are variables. The sample covariance of
//! columns `j` and `k` is
//!
//! ```text
//! Σ_jk = (1/(n − 1)) Σᵢ conj(x_ij − x̄_j) (x_ik − x̄_k),
//! ```
//!
//! where `n` is the number of rows and `x̄_j` is the mean of column `j`.
//! Centering uses compensated column means, so data with a large common offset
//! keeps its small deviations.
//!
//! ```
//! use scioxide_linalg::{Matrix, RankPolicy};
//! use scioxide_stats::{MultivariateOptions, PreparedPca, sample_covariance_with_options, shrink_covariance};
//!
//! // Two observations 2 apart, offset by 2⁵³: the sample variance is 2.
//! let base = 2_f64.powi(53);
//! let data = Matrix::from_vec(2, 1, vec![base, base + 2.])?;
//! let (_, covariance) = sample_covariance_with_options(data.view(), MultivariateOptions::new())?;
//! assert_eq!(covariance.as_slice(), &[2.]);
//! let pca = PreparedPca::with_options(
//!     data.view(),
//!     MultivariateOptions::new()
//!         .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
//!         .with_max_bytes(usize::MAX),
//! )?;
//! assert!((pca.variances()[0] - 2.).abs() < 1e-14);
//! assert_eq!(shrink_covariance(covariance.view(), 0.)?, covariance);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
use scioxide_core::{
    Complex, ExecutionContext, NumericScalar, RealScalar, Scaled, ScaledAccumulator, ScaledValue,
    SciOxideError,
};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, Matrix, RankPolicy, ThinSvdDecomposition,
};

mod cca;
mod center;
mod covariance;
mod pca;
mod storage;
#[cfg(test)]
mod tests;

pub use cca::{CcaWorkspace, PreparedCca};
pub(crate) use center::{CenteredData, centered};
pub(crate) use covariance::frequency_covariance_with_options;
pub use covariance::{
    ledoit_wolf_covariance, ledoit_wolf_covariance_with_options, sample_covariance,
    sample_covariance_with_options, shrink_covariance,
};
pub use pca::{PcaScaling, PcaWorkspace, PreparedPca};
pub use storage::MultivariateRequirements;
pub(crate) use storage::reserve_multivariate;

pub(crate) fn factor<T: NumericScalar>(
    data: &Matrix<T>,
    context: ExecutionContext<'_>,
) -> Result<ThinSvdDecomposition<T>, SciOxideError> {
    let mut result = ThinSvdDecomposition::empty();
    let mut work = LinalgWorkspace::new();
    scioxide_linalg::ThinSvdDecomposition::compute_into(
        data,
        &LinalgExecutionOptions::default(),
        &mut result,
        &mut work,
        context,
    )?;
    Ok(result)
}

type Extended = ScaledValue<Complex<f64>>;

fn extended<T: NumericScalar>(value: T) -> Result<Extended, SciOxideError> {
    Extended::from_value(Complex::new(
        value.real_component().try_to_f64_exact()?,
        value.imaginary_component().try_to_f64_exact()?,
    ))
}

fn project<T: NumericScalar>(value: Extended) -> Result<T, SciOxideError> {
    Ok(T::from_components(
        value.real().round_to()?,
        value.imaginary().round_to()?,
    ))
}

fn conjugate(value: Extended) -> Extended {
    Extended::from_components(value.real(), -value.imaginary())
}

fn norm_squared(value: Extended) -> Result<Scaled<f64>, SciOxideError> {
    value.real().square()?.add(value.imaginary().square()?)
}

/// Returns a compensated center as a renormalized `[high, low]` pair of `f64`,
/// or `None` if it is outside the `f64` range.
fn normalized_native(center: ScaledAccumulator<f64>) -> Option<[f64; 2]> {
    let [high, low] = crate::arithmetic::native_center(center)?;
    let (high, low) = scioxide_core::backend::two_sum(high, low);
    Some([high, low])
}

// Statistical centers compose core's compensated components. In particular,
// subtraction uses the same low-part-preserving operation as scalar moments.
#[derive(Clone, Copy)]
pub(crate) struct PreciseCenter {
    real: ScaledAccumulator<f64>,
    imaginary: ScaledAccumulator<f64>,
}

impl PreciseCenter {
    const ZERO: Self = Self {
        real: ScaledAccumulator::ZERO,
        imaginary: ScaledAccumulator::ZERO,
    };

    fn from_value(value: Extended) -> Self {
        Self {
            real: ScaledAccumulator::from_scaled(value.real()),
            imaginary: ScaledAccumulator::from_scaled(value.imaginary()),
        }
    }

    fn add(&mut self, value: Extended) -> Result<(), SciOxideError> {
        self.real.add_scaled(value.real())?;
        self.imaginary.add_scaled(value.imaginary())
    }

    fn value(self) -> Result<Extended, SciOxideError> {
        Ok(Extended::from_components(
            self.real.total()?,
            self.imaginary.total()?,
        ))
    }

    /// Returns the normalized `f64` parts of the real component, if representable.
    fn native_real(self) -> Option<[f64; 2]> {
        normalized_native(self.real)
    }

    /// Returns the normalized `f64` parts of both components, if representable.
    fn native_parts(self) -> Option<[[f64; 2]; 2]> {
        Some([
            normalized_native(self.real)?,
            normalized_native(self.imaginary)?,
        ])
    }

    fn difference(self, value: Extended) -> Result<Extended, SciOxideError> {
        Ok(Extended::from_components(
            crate::arithmetic::center_difference(
                self.real,
                ScaledAccumulator::from_scaled(value.real()),
            )?,
            crate::arithmetic::center_difference(
                self.imaginary,
                ScaledAccumulator::from_scaled(value.imaginary()),
            )?,
        ))
    }
}

/// Options for the multivariate estimators: a numerical rank policy and a byte limit.
///
/// The rank policy decides how many singular values [`PreparedPca`] and
/// [`PreparedCca`] treat as nonzero. The covariance estimators use only the byte
/// limit.
#[derive(Clone, Copy, Debug)]
pub struct MultivariateOptions<S: RealScalar> {
    rank_policy: RankPolicy<S>,
    max_bytes: usize,
}
impl<S: RealScalar> Default for MultivariateOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> MultivariateOptions<S> {
    /// Returns the default options: `RankPolicy::DimensionScaledEpsilon` and no byte limit.
    pub const fn new() -> Self {
        Self {
            rank_policy: RankPolicy::DimensionScaledEpsilon,
            max_bytes: usize::MAX,
        }
    }
    /// Sets the policy that decides the numerical rank of the centered data.
    pub const fn with_rank_policy(mut self, policy: RankPolicy<S>) -> Self {
        self.rank_policy = policy;
        self
    }
    /// Sets the limit, in bytes, on the memory used to prepare an estimator, on its
    /// result and on the workspaces it requests.
    ///
    /// A request above the limit fails with `ResourceLimit`.
    pub const fn with_max_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = bytes;
        self
    }
    /// Returns the rank policy.
    pub const fn rank_policy(self) -> RankPolicy<S> {
        self.rank_policy
    }
    /// Returns the byte limit.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}
