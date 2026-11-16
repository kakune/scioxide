//! Weight models and the weight storage accepted by statistics.
use core::marker::PhantomData;
use scioxide_core::RealScalar;

mod sealed {
    pub trait Sealed {}
}

/// The statistical meaning of observation weights.
///
/// The models are [`Unweighted`], [`Frequency`], [`Reliability`], [`Analytic`]
/// and [`Mass`]. They differ in which estimators are defined and in the
/// divisor of the sample variance. Each function states the models it accepts
/// through a trait bound, so an unsupported model is a compile-time error.
/// This trait is sealed.
pub trait WeightModel:
    sealed::Sealed + Copy + Eq + Send + Sync + core::fmt::Debug + 'static
{
    /// Whether observations carry weights (`false` only for [`Unweighted`]).
    const IS_WEIGHTED: bool;
}

macro_rules! models {
    ($($name:ident($weighted:literal): $doc:literal),* $(,)?) => {$ (
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl WeightModel for $name { const IS_WEIGHTED: bool = $weighted; }
    )* };
}
models! {
    Unweighted(false): "Unweighted observations: each observation counts once.",
    Frequency(true): "Frequency weights: integer replication counts.\n\n\
        A count `wᵢ` means that observation `xᵢ` occurs `wᵢ` times. Estimators \
        equal those of the expanded sample of size `N = Σ wᵢ`, so the sample \
        variance has divisor `N − 1`. Batch views take counts as `u64`. \
        Regression takes real weights, which must be integers. A zero count \
        omits the observation.",
    Mass(true): "Relative masses: nonnegative weights that define a discrete \
        distribution.\n\n\
        Observation `xᵢ` has probability `wᵢ / W`, where `W = Σ wᵢ`. Masses \
        support weighted means, quantiles (the inverse of the weighted empirical \
        CDF), expectiles and other summaries of that distribution. They carry \
        no sampling model, so sample variances and standard errors are \
        unavailable. A zero mass omits the observation.",
    Reliability(true): "Reliability weights: nonnegative relative weights with \
        an effective sample size.\n\n\
        Only relative weights matter: the mean, the sample variance and the \
        effective sample size do not change when all weights are multiplied by \
        a positive constant. With `W = Σ wᵢ` and `Q = Σ wᵢ²`, the Kish \
        effective sample size is `n_eff = W²/Q`, and the sample variance is \
        `Σ wᵢ (xᵢ − x̄)² / (W − Q/W)`, where `x̄` is the weighted mean. A zero \
        weight omits the observation.\n\n\
        # References\n\n\
        L. Kish, *Survey Sampling*, Wiley, 1965.",
    Analytic(true): "Analytic weights: relative precisions (inverse variances).\n\n\
        Observation `xᵢ` has variance `σ²/wᵢ` with a common unknown noise scale \
        `σ²`. The noise scale is estimated by `σ̂² = Σ wᵢ (xᵢ − x̄)² / (n − 1)`, \
        where `x̄` is the weighted mean and `n` is the number of positive \
        weights. A zero weight omits the observation.",
}

/// Weight storage tagged with its weight model `M`.
///
/// `D` is usually a borrowed slice: `&[u64]` for [`Frequency`] counts and a
/// slice of the real scalar type for [`Mass`], [`Reliability`] and
/// [`Analytic`]. Attach weights to a view with
/// [`BatchView::with_weights`](crate::BatchView::with_weights). In-place
/// selection functions take mutable storage and permute observations and
/// weights together.
///
/// Construction does not read the weights. The function that uses them checks
/// the length and rejects negative, non-finite and, for frequency counts,
/// non-integer values.
///
/// ```
/// use scioxide_stats::{Frequency, Weights};
///
/// let counts = [1u64, 2, 0];
/// let weights = Weights::<_, Frequency>::new(&counts[..]);
/// assert_eq!(*weights.storage(), &counts[..]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Weights<D, M: WeightModel> {
    storage: D,
    model: PhantomData<M>,
}
impl<D, M: WeightModel> Weights<D, M> {
    /// Wraps `storage` as weights of model `M`.
    pub const fn new(storage: D) -> Self {
        Self {
            storage,
            model: PhantomData,
        }
    }
    /// Returns a reference to the wrapped storage.
    pub const fn storage(&self) -> &D {
        &self.storage
    }
    /// Returns the wrapped storage.
    pub fn into_storage(self) -> D {
        self.storage
    }
}

/// Weight models that define an empirical distribution: [`Unweighted`],
/// [`Frequency`] and [`Mass`].
///
/// Quantiles, empirical CDFs and other order-based estimators accept these
/// models.
pub trait EmpiricalWeights: WeightModel {}
impl EmpiricalWeights for Unweighted {}
impl EmpiricalWeights for Frequency {}
impl EmpiricalWeights for Mass {}

/// Weight models accepted by isotonic regression: [`Unweighted`] and [`Mass`].
///
/// Frequency counts are rejected at compile time:
///
/// ```compile_fail
/// use scioxide_core::{ExecutionContext, StridedLayout};
/// use scioxide_stats::{Axis, BatchView, Frequency, IsotonicWorkspace, StatisticsOptions, Weights};
/// let view = BatchView::new(&[1.0_f64, 2.0],
///     StridedLayout::new(&[2], &[1]).unwrap(), Axis::Index(0)).unwrap()
///     .with_weights(Weights::<_, Frequency>::new(&[1u64, 1])).unwrap();
/// IsotonicWorkspace::workspace_requirements(view, StatisticsOptions::new(), ExecutionContext::sequential());
/// ```
pub trait IsotonicWeights: EmpiricalWeights {}
impl IsotonicWeights for Unweighted {}
impl IsotonicWeights for Mass {}

/// Weight models with a normal likelihood, accepted by conjugate normal
/// updates: [`Frequency`] and [`Analytic`].
///
/// Reliability weights define no likelihood and are rejected at compile time:
///
/// ```compile_fail
/// use scioxide_stats::{MomentOrder, NormalInverseGammaPrior, OnlineWeightedMoments, Reliability};
/// let moments = OnlineWeightedMoments::<f64, Reliability>::new(MomentOrder::Second);
/// NormalInverseGammaPrior::reference().update_weighted(&moments);
/// ```
pub trait ConjugateWeights: crate::MomentWeights {}
impl ConjugateWeights for Frequency {}
impl ConjugateWeights for Analytic {}

/// Row weights accepted by linear, operator and generalized linear regression.
///
/// Implemented for [`Unweighted`] and for borrowed real slices of
/// [`Frequency`] counts or [`Analytic`] precisions. Mass and reliability
/// weights define no regression likelihood and do not compile. Preparing a
/// regression checks that there is one weight per row and that each weight is
/// finite and nonnegative, and an integer for frequency counts.
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_stats::{Analytic, PreparedRegression, Weights};
///
/// // Weighted least squares on an intercept-only design gives the weighted
/// // mean (1·1 + 2·2 + 1·3) / 4 = 2.
/// let design = Matrix::from_vec(3, 1, vec![1.0_f64; 3])?;
/// let weights = [1.0, 2.0, 1.0];
/// let plan = PreparedRegression::new(
///     design.view(),
///     Weights::<_, Analytic>::new(&weights[..]),
///     Default::default(),
/// )?;
/// assert!((plan.fit(&[1.0, 2.0, 3.0])?.coefficients().as_slice()[0] - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// ```compile_fail
/// use scioxide_linalg::Matrix;
/// use scioxide_stats::{PreparedRegression, Reliability, Weights};
/// let design = Matrix::from_vec(2, 1, vec![1.0_f64; 2]).unwrap();
/// let weights = [1.0, 2.0];
/// PreparedRegression::new(design.view(),
///     Weights::<_, Reliability>::new(&weights[..]), Default::default());
/// ```
pub trait RegressionWeights<'a, S: RealScalar>: regression_sealed::Sealed<'a, S> {
    /// Weight model of the rows.
    type Model: WeightModel;
}
pub(crate) mod regression_sealed {
    use super::*;
    pub trait Sealed<'a, S: RealScalar> {
        fn rows(self) -> RegressionRows<'a, S>;
    }
}
/// Regression row weights with their model resolved.
#[derive(Clone, Copy, Debug)]
pub enum RegressionRows<'a, S: RealScalar> {
    /// Unit weight on every row.
    Unit,
    /// Integer replication counts.
    Frequency(&'a [S]),
    /// Relative precisions.
    Analytic(&'a [S]),
}
impl<S: RealScalar> RegressionRows<'_, S> {
    /// Returns the weight of `row` after checking that it is finite,
    /// nonnegative and, for counts, an integer.
    pub(crate) fn weight(self, row: usize) -> Result<S, scioxide_core::SciOxideError> {
        let (w, frequency) = match self {
            Self::Unit => return Ok(S::ONE),
            Self::Frequency(w) => (w[row], true),
            Self::Analytic(w) => (w[row], false),
        };
        if !w.is_finite() || w < S::ZERO || (frequency && w != w.floor()) {
            return Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "weights must be finite nonnegative, and integral for frequency counts",
            ));
        }
        Ok(w)
    }
    /// Whether the rows carry frequency counts.
    pub(crate) const fn is_frequency(self) -> bool {
        matches!(self, Self::Frequency(_))
    }
    /// Whether there is one weight per row.
    pub(crate) fn matches(self, rows: usize) -> bool {
        match self {
            Self::Unit => true,
            Self::Frequency(w) | Self::Analytic(w) => w.len() == rows,
        }
    }
}
impl<'a, S: RealScalar> regression_sealed::Sealed<'a, S> for Unweighted {
    fn rows(self) -> RegressionRows<'a, S> {
        RegressionRows::Unit
    }
}
impl<'a, S: RealScalar> RegressionWeights<'a, S> for Unweighted {
    type Model = Unweighted;
}
macro_rules! regression_rows {
    ($model:ident) => {
        impl<'a, S: RealScalar> regression_sealed::Sealed<'a, S> for Weights<&'a [S], $model> {
            fn rows(self) -> RegressionRows<'a, S> {
                RegressionRows::$model(self.storage)
            }
        }
        impl<'a, S: RealScalar> RegressionWeights<'a, S> for Weights<&'a [S], $model> {
            type Model = $model;
        }
        impl<'a, S: RealScalar, const N: usize> regression_sealed::Sealed<'a, S>
            for Weights<&'a [S; N], $model>
        {
            fn rows(self) -> RegressionRows<'a, S> {
                RegressionRows::$model(&self.storage[..])
            }
        }
        impl<'a, S: RealScalar, const N: usize> RegressionWeights<'a, S>
            for Weights<&'a [S; N], $model>
        {
            type Model = $model;
        }
        impl<'a, S: RealScalar> regression_sealed::Sealed<'a, S>
            for Weights<&'a alloc::vec::Vec<S>, $model>
        {
            fn rows(self) -> RegressionRows<'a, S> {
                RegressionRows::$model(self.storage.as_slice())
            }
        }
        impl<'a, S: RealScalar> RegressionWeights<'a, S>
            for Weights<&'a alloc::vec::Vec<S>, $model>
        {
            type Model = $model;
        }
    };
}
regression_rows!(Frequency);
regression_rows!(Analytic);

/// Borrowed observation weights with their model resolved.
#[derive(Clone, Copy, Debug)]
pub enum WeightData<'a, S: RealScalar> {
    /// No weights.
    None,
    /// Replication counts.
    Frequency(&'a [u64]),
    /// Relative masses.
    Mass(&'a [S]),
    /// Reliability weights.
    Reliability(&'a [S]),
    /// Analytic precisions.
    Analytic(&'a [S]),
}
impl<S: RealScalar> WeightData<'_, S> {
    pub(crate) fn len(self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::Frequency(w) => Some(w.len()),
            Self::Mass(w) | Self::Reliability(w) | Self::Analytic(w) => Some(w.len()),
        }
    }
}

/// Weight storage accepted by [`BatchView::with_weights`](crate::BatchView::with_weights)
/// and [`statistic`](fn@crate::statistic).
///
/// Implemented for [`Unweighted`], for `Weights<D, Frequency>` with `D` one of
/// `&[u64]`, `&[u64; N]` or `&Vec<u64>`, and for `Weights<D, M>` with `M` one
/// of [`Mass`], [`Reliability`] or [`Analytic`] and `D` one of `&[S]`,
/// `&[S; N]` or `&Vec<S>`. This trait is sealed.
pub trait BatchWeights<'a, S: RealScalar>: batch_sealed::Sealed<'a, S> {
    /// Weight model of the resulting view.
    type Model: WeightModel;
}
pub(crate) mod batch_sealed {
    use super::*;
    pub trait Sealed<'a, S: RealScalar> {
        fn data(self) -> WeightData<'a, S>;
    }
}
macro_rules! borrowed {
    ($model:ident, $value:ty) => {
        impl<'a, S: RealScalar> batch_sealed::Sealed<'a, S> for Weights<&'a [$value], $model> {
            fn data(self) -> WeightData<'a, S> {
                WeightData::$model(self.storage)
            }
        }
        impl<'a, S: RealScalar> BatchWeights<'a, S> for Weights<&'a [$value], $model> {
            type Model = $model;
        }
    };
}
borrowed!(Frequency, u64);
borrowed!(Mass, S);
borrowed!(Reliability, S);
borrowed!(Analytic, S);

impl<'a, S: RealScalar> batch_sealed::Sealed<'a, S> for Unweighted {
    fn data(self) -> WeightData<'a, S> {
        WeightData::None
    }
}
impl<'a, S: RealScalar> BatchWeights<'a, S> for Unweighted {
    type Model = Unweighted;
}

macro_rules! borrowed_array {
    ($model:ident, $value:ty) => {
        impl<'a, S: RealScalar, const N: usize> batch_sealed::Sealed<'a, S>
            for Weights<&'a [$value; N], $model>
        {
            fn data(self) -> WeightData<'a, S> {
                WeightData::$model(&self.storage[..])
            }
        }
        impl<'a, S: RealScalar, const N: usize> BatchWeights<'a, S>
            for Weights<&'a [$value; N], $model>
        {
            type Model = $model;
        }
        impl<'a, S: RealScalar> batch_sealed::Sealed<'a, S>
            for Weights<&'a alloc::vec::Vec<$value>, $model>
        {
            fn data(self) -> WeightData<'a, S> {
                WeightData::$model(self.storage.as_slice())
            }
        }
        impl<'a, S: RealScalar> BatchWeights<'a, S>
            for Weights<&'a alloc::vec::Vec<$value>, $model>
        {
            type Model = $model;
        }
    };
}
borrowed_array!(Frequency, u64);
borrowed_array!(Mass, S);
borrowed_array!(Reliability, S);
borrowed_array!(Analytic, S);

pub(crate) fn frequency_scalar<S: RealScalar>(
    count: u64,
) -> Result<S, scioxide_core::SciOxideError> {
    let bits = u64::BITS - count.leading_zeros();
    if bits > S::MANTISSA_DIGITS && count.trailing_zeros() < bits - S::MANTISSA_DIGITS {
        return Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "frequency count is not exactly representable",
        ));
    }
    S::try_from_i64_exact(count as i64)
}
pub(crate) fn frequency_index(count: u64) -> Result<usize, scioxide_core::SciOxideError> {
    usize::try_from(count).map_err(|_| {
        scioxide_core::SciOxideError::size_overflow(scioxide_core::quantity!("frequency_count"))
    })
}
