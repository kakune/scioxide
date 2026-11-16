//! Absolute and relative tolerances for convergence and error tests.
//!
//! A tolerance pair `(atol, rtol)` accepts a magnitude `m` at scale `s` when
//! `m ≤ atol + rtol·s`. [`Tolerance`] is the general pair. The role types
//! ([`StepTolerance`], [`ResidualTolerance`], [`GradientTolerance`],
//! [`RankThreshold`], [`FeasibilityTolerance`]) hold the same pair but name its
//! use in a solver's options. [`ErrorTolerance`] tests a vector of local error
//! estimates componentwise, as ODE and quadrature codes do.
//!
//! All thresholds are finite and nonnegative. Below, `ε` is the machine
//! epsilon of the scalar type. Iteration and evaluation limits are separate;
//! see [`crate::Budget`].

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::arithmetic::{
    RangeAccumulator, ScaledNorm, absolute_difference_le, checked_final_rounding,
    checked_float_quotient,
};
use crate::error::{InvalidInputReason, NonFiniteOrigin, SciOxideError};
use crate::scalar::RealScalar;

/// An absolute and a relative tolerance, `atol` and `rtol`, for convergence tests.
///
/// A magnitude `m` at scale `s` passes when `m ≤ atol + rtol·s`. Both values
/// are finite and nonnegative.
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
///
/// // 5e-7 ≤ 1e-8 + 1e-6 · 1.0
/// let tolerance = Tolerance::<f64>::new(1e-8, 1e-6).unwrap();
/// assert_eq!(tolerance.absolute(), 1e-8);
/// assert!(tolerance.is_residual_ok(5e-7, 1.0).unwrap());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Tolerance<S: RealScalar> {
    absolute: S,
    relative: S,
}

impl<S: RealScalar> From<RelativeTolerance<S>> for Tolerance<S> {
    /// Converts a relative tolerance to a [`Tolerance`] with `atol = 0`.
    ///
    /// ```
    /// use scioxide_core::{RelativeTolerance, Tolerance};
    ///
    /// let tolerance: Tolerance<f64> = RelativeTolerance::default().into();
    /// assert_eq!(tolerance.absolute(), 0.0);
    /// assert_eq!(tolerance.relative(), f64::EPSILON);
    /// ```
    fn from(relative: RelativeTolerance<S>) -> Self {
        Self {
            absolute: S::ZERO,
            relative: relative.relative(),
        }
    }
}

/// A relative tolerance alone, finite and nonnegative.
///
/// Zero asks for the best accuracy the algorithm can reach; an algorithm that
/// has a lower limit documents it. The default is the machine epsilon.
///
/// ```
/// let tolerance = scioxide_core::RelativeTolerance::new(1e-8_f64)?;
/// assert_eq!(tolerance.relative(), 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RelativeTolerance<S: RealScalar>(S);

impl<S: RealScalar> RelativeTolerance<S> {
    /// Creates a relative tolerance.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `relative` is negative.
    /// - `NonFinite` if `relative` is NaN or infinite.
    pub fn new(relative: S) -> Result<Self, SciOxideError> {
        validate_nonnegative(relative, crate::argument!("relative"))?;
        Ok(Self(relative))
    }

    /// Returns the relative tolerance.
    pub const fn relative(self) -> S {
        self.0
    }
}

impl<S: RealScalar> Default for RelativeTolerance<S> {
    /// Returns the machine epsilon of `S`.
    fn default() -> Self {
        Self(S::EPSILON)
    }
}

impl<S: RealScalar> Tolerance<S> {
    /// Creates the tolerance pair `(atol, rtol) = (absolute, relative)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if either value is negative.
    /// - `NonFinite` if either value is NaN or infinite.
    pub fn new(absolute: S, relative: S) -> Result<Self, SciOxideError> {
        validate_nonnegative(absolute, crate::argument!("absolute"))?;
        validate_nonnegative(relative, crate::argument!("relative"))?;
        Ok(Self { absolute, relative })
    }

    /// Returns `atol = rtol = √ε`, where `ε` is the machine epsilon of `S`.
    ///
    /// Solver options state whether they use this as their default.
    pub fn from_sqrt_epsilon() -> Self {
        let threshold = S::SQRT_EPSILON;
        Self {
            absolute: threshold,
            relative: threshold,
        }
    }

    /// Returns `atol = rtol = ε`, the machine epsilon of `S`.
    ///
    /// This suits deflation tests in backward-stable dense factorizations,
    /// which work at full precision.
    pub fn from_machine_epsilon() -> Self {
        let threshold = S::EPSILON;
        Self {
            absolute: threshold,
            relative: threshold,
        }
    }

    /// Returns the absolute tolerance `atol`.
    #[must_use]
    pub fn absolute(&self) -> S {
        self.absolute
    }

    /// Returns the relative tolerance `rtol`.
    #[must_use]
    pub fn relative(&self) -> S {
        self.relative
    }

    /// Returns a copy with a different absolute tolerance.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `absolute` is negative.
    /// - `NonFinite` if `absolute` is NaN or infinite.
    pub fn with_absolute(self, absolute: S) -> Result<Self, SciOxideError> {
        validate_nonnegative(absolute, crate::argument!("absolute"))?;
        Ok(Self { absolute, ..self })
    }

    /// Returns a copy with a different relative tolerance.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `relative` is negative.
    /// - `NonFinite` if `relative` is NaN or infinite.
    pub fn with_relative(self, relative: S) -> Result<Self, SciOxideError> {
        validate_nonnegative(relative, crate::argument!("relative"))?;
        Ok(Self { relative, ..self })
    }

    /// Returns the threshold `atol + rtol·scale`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `scale` is negative.
    /// - `NonFinite` if `scale` is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if the threshold exceeds the largest
    ///   finite value.
    pub fn threshold(&self, scale: S) -> Result<S, SciOxideError> {
        validate_nonnegative(scale, crate::argument!("scale"))?;
        threshold_value(self.absolute, self.relative, scale)
    }

    /// Tests two successive iterates: `|current − previous| ≤ atol + rtol·max(|current|, |previous|)`.
    ///
    /// The difference is compared without being formed, so it cannot overflow.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if either iterate is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if the threshold exceeds the largest
    ///   finite value.
    pub fn is_converged(&self, current: S, previous: S) -> Result<bool, SciOxideError> {
        if !current.is_finite() || !previous.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "iterates must be finite",
            ));
        }
        let scale = current.abs().max(previous.abs());
        let threshold = self.threshold(scale)?;
        // Compare without forming `current - previous`, which can overflow for
        // large opposite-sign iterates even when both are finite.
        Ok(absolute_difference_le(current, previous, threshold))
    }

    /// Tests a residual norm: `residual_norm ≤ atol + rtol·problem_scale`.
    ///
    /// A large residual returns `Ok(false)`; only invalid arguments are errors.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `residual_norm` or `problem_scale` is negative.
    /// - `NonFinite` if either value is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if the threshold exceeds the largest
    ///   finite value.
    pub fn is_residual_ok(
        &self,
        residual_norm: S,
        problem_scale: S,
    ) -> Result<bool, SciOxideError> {
        self.residual_for(
            residual_norm,
            problem_scale,
            crate::argument!("residual_norm"),
            crate::argument!("problem_scale"),
        )
    }
    fn residual_for(
        &self,
        magnitude: S,
        scale: S,
        argument: crate::Argument,
        scale_argument: crate::Argument,
    ) -> Result<bool, SciOxideError> {
        validate_nonnegative(magnitude, argument)?;
        validate_nonnegative(scale, scale_argument)?;
        Ok(magnitude <= threshold_value(self.absolute, self.relative, scale)?)
    }
}

impl<S: RealScalar> Tolerance<S> {
    /// Returns `atol = 10^(−abs_places)` and `rtol = 10^(−rel_places)`.
    ///
    /// A large exponent underflows to a zero tolerance, which is valid.
    pub fn from_decimal_places(abs_places: u32, rel_places: u32) -> Self {
        fn inverse_power_of_ten<S: RealScalar>(places: u32) -> S {
            i32::try_from(places).map_or_else(|_| S::ZERO, |power| S::TEN.powi(-power))
        }

        Self {
            absolute: inverse_power_of_ten(abs_places),
            relative: inverse_power_of_ten(rel_places),
        }
    }
}

/// Vector norm used by [`ErrorTolerance`] to combine normalized component errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum NormSelection {
    /// Sum of component magnitudes.
    L1,
    /// Euclidean norm.
    L2,
    /// Maximum component magnitude.
    Infinity,
    /// Root mean square of component magnitudes, `√(Σ xᵢ² / n)`.
    RootMeanSquare,
}

/// An absolute tolerance, either one value for all components or one value per component.
///
/// All values are finite and nonnegative. [`Self::try_vector`] copies the
/// values and so allocates. An empty vector is valid for a problem with no
/// components.
///
/// ```
/// use scioxide_core::AbsoluteTolerance;
///
/// let tolerance = AbsoluteTolerance::try_vector(&[1e-8_f64, 1e-6])?;
/// assert_eq!(tolerance.component(1, 2)?, 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct AbsoluteTolerance<S: RealScalar>(AbsolutePolicy<S>);

#[derive(Debug, Clone, PartialEq)]
enum AbsolutePolicy<S: RealScalar> {
    /// One absolute tolerance broadcast to every component.
    Scalar(S),
    /// One absolute tolerance per component.
    Vector(Box<[S]>),
}

impl<S: RealScalar> AbsoluteTolerance<S> {
    /// Creates one absolute tolerance for all components.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `value` is negative.
    /// - `NonFinite` if `value` is NaN or infinite.
    pub fn scalar(value: S) -> Result<Self, SciOxideError> {
        validate_nonnegative(value, crate::argument!("value"))?;
        Ok(Self(AbsolutePolicy::Scalar(value)))
    }

    /// Creates one absolute tolerance per component, copying `values`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if a value is negative.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `AllocationFailure` if the copy cannot be stored.
    pub fn try_vector(values: &[S]) -> Result<Self, SciOxideError> {
        Self::validated_vector(values, crate::argument!("values"))
    }

    // Validation and fallible copy reported by the public entry point
    // whose component tolerances are `argument`.
    fn validated_vector(values: &[S], argument: crate::Argument) -> Result<Self, SciOxideError> {
        values
            .iter()
            .copied()
            .try_for_each(|value| validate_nonnegative(value, argument))?;
        let mut owned = Vec::new();
        owned.try_reserve_exact(values.len()).map_err(|_| {
            SciOxideError::allocation_failure(
                crate::quantity!("absolute_tolerance_vector"),
                values.len(),
            )
        })?;
        owned.extend_from_slice(values);
        Ok(Self(AbsolutePolicy::Vector(owned.into_boxed_slice())))
    }

    /// Returns the tolerance of component `index` in a problem with `dimension` components.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if this is a per-component tolerance whose length
    ///   is not `dimension`.
    /// - `OutOfBounds` if `index >= dimension`.
    pub fn component(&self, index: usize, dimension: usize) -> Result<S, SciOxideError> {
        self.validate_dimension(dimension)?;
        if index >= dimension {
            return Err(SciOxideError::out_of_bounds(
                crate::quantity!("absolute_tolerance_components"),
                index,
                dimension,
            ));
        }
        match &self.0 {
            AbsolutePolicy::Scalar(value) => Ok(*value),
            AbsolutePolicy::Vector(values) => Ok(values[index]),
        }
    }

    fn validate_dimension(&self, dimension: usize) -> Result<(), SciOxideError> {
        if let AbsolutePolicy::Vector(values) = &self.0
            && values.len() != dimension
        {
            return Err(SciOxideError::dimension_mismatch(
                crate::Dimensions::one(dimension),
                crate::Dimensions::one(values.len()),
            ));
        }
        Ok(())
    }

    /// Returns the per-component values, or `None` for a single value.
    pub fn as_vector(&self) -> Option<&[S]> {
        match &self.0 {
            AbsolutePolicy::Scalar(_) => None,
            AbsolutePolicy::Vector(values) => Some(values),
        }
    }
}

fn validate_nonnegative<S: RealScalar>(
    value: S,
    argument: crate::Argument,
) -> Result<(), SciOxideError> {
    if !value.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "tolerance value must be finite",
        ));
    }
    if value < S::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            InvalidInputReason::OutOfRange,
            "tolerance must be nonnegative",
        ));
    }
    Ok(())
}

#[inline]
fn threshold_value<S: RealScalar>(absolute: S, relative: S, scale: S) -> Result<S, SciOxideError> {
    let relative = checked_final_rounding(
        relative * scale,
        "relative tolerance is outside the finite scalar range",
    )?;
    checked_final_rounding(
        absolute + relative,
        "combined tolerance is outside the finite scalar range",
    )
}

macro_rules! role_tolerance {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[must_use]
        pub struct $name<S: RealScalar>(Tolerance<S>);

        impl<S: RealScalar> $name<S> {
            /// Creates the tolerance pair `(atol, rtol) = (absolute, relative)`.
            ///
            /// # Errors
            ///
            /// The errors of [`Tolerance::new`].
            pub fn new(absolute: S, relative: S) -> Result<Self, SciOxideError> {
                Tolerance::new(absolute, relative).map(Self)
            }

            /// Wraps an existing [`Tolerance`].
            pub const fn from_tolerance(tolerance: Tolerance<S>) -> Self {
                Self(tolerance)
            }

            /// Returns the underlying [`Tolerance`].
            pub const fn tolerance(self) -> Tolerance<S> {
                self.0
            }

            /// Returns the absolute tolerance, in the units of the tested quantity.
            pub fn absolute(self) -> S {
                self.0.absolute()
            }

            /// Returns the dimensionless relative tolerance.
            pub fn relative(self) -> S {
                self.0.relative()
            }

            /// Returns the threshold `atol + rtol·scale`.
            ///
            /// # Errors
            ///
            /// The errors of [`Tolerance::threshold`].
            pub fn threshold(self, scale: S) -> Result<S, SciOxideError> {
                self.0.threshold(scale)
            }

            /// Tests `magnitude ≤ atol + rtol·scale`.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` (`OutOfRange`) if `magnitude` or `scale` is negative.
            /// - `NonFinite` if either value is NaN or infinite.
            /// - `NumericalInstability` (`Overflow`) if the threshold exceeds the
            ///   largest finite value.
            pub fn is_satisfied(self, magnitude: S, scale: S) -> Result<bool, SciOxideError> {
                self.0.residual_for(
                    magnitude,
                    scale,
                    crate::argument!("magnitude"),
                    crate::argument!("scale"),
                )
            }
        }

        impl<S: RealScalar> Default for $name<S> {
            fn default() -> Self {
                Self(Tolerance::from_sqrt_epsilon())
            }
        }
    };
}

role_tolerance!(
    StepTolerance,
    "Tolerance on the change in the iterate between steps. The default is `atol = rtol = √ε`."
);
role_tolerance!(
    ResidualTolerance,
    "Tolerance on a residual norm. The default is `atol = rtol = √ε`."
);
role_tolerance!(
    GradientTolerance,
    "Tolerance on the norm of an objective gradient. The default is `atol = rtol = √ε`."
);
role_tolerance!(
    RankThreshold,
    "Tolerance used to decide numerical rank. The default is `atol = rtol = √ε`."
);
role_tolerance!(
    FeasibilityTolerance,
    "Tolerance on constraint violation. The default is `atol = rtol = √ε`."
);

/// Tolerance in `[0, 0.5)` on the distance of a value from the nearest integer.
///
/// Because the tolerance is below one half, a value is near at most one
/// integer, and half-integers are never accepted. The default is `√ε`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct IntegralityTolerance<S: RealScalar>(S);

impl<S: RealScalar> IntegralityTolerance<S> {
    /// Creates an integrality tolerance.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `absolute` is negative or at least 0.5.
    /// - `NonFinite` if `absolute` is NaN or infinite.
    pub fn new(absolute: S) -> Result<Self, SciOxideError> {
        validate_nonnegative(absolute, crate::argument!("absolute"))?;
        // Below one half, a value lies within tolerance of at most one integer, so
        // "nearly integral" identifies a unique rounding target.
        if absolute >= S::HALF {
            return Err(SciOxideError::invalid_input(
                crate::argument!("absolute"),
                InvalidInputReason::OutOfRange,
                "integrality tolerance must be less than one half",
            ));
        }
        Ok(Self(absolute))
    }

    /// Returns the tolerance.
    pub const fn absolute(self) -> S {
        self.0
    }

    /// Tests `distance ≤ tolerance`, where `distance` is the distance to the nearest integer.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `distance` is negative.
    /// - `NonFinite` if `distance` is NaN or infinite.
    pub fn is_satisfied(self, distance: S) -> Result<bool, SciOxideError> {
        validate_nonnegative(distance, crate::argument!("distance"))?;
        Ok(distance <= self.0)
    }
}

impl<S: RealScalar> Default for IntegralityTolerance<S> {
    fn default() -> Self {
        Self(S::SQRT_EPSILON)
    }
}

/// Componentwise error tolerance for a vector of error estimates, as in ODE and quadrature codes.
///
/// Component `i` of an error vector `e` with scales `s` is normalized as
///
/// ```text
/// rᵢ = |eᵢ| / (atolᵢ + rtol·sᵢ)
/// ```
///
/// and the test passes when `‖r‖ ≤ 1` in the selected [`NormSelection`]. The
/// absolute tolerance may differ per component; the relative tolerance is one
/// value. A zero threshold accepts only an exactly zero error. An empty vector
/// has norm zero. The default is `atol = rtol = √ε` with the root-mean-square
/// norm.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ErrorTolerance<S: RealScalar> {
    absolute: AbsoluteTolerance<S>,
    relative: S,
    norm: NormSelection,
}

/// Precomputed thresholds `atolᵢ + rtol·sᵢ`, for testing several error vectors with the same scales.
///
/// Created by [`ErrorTolerance::prepare_norm`]; it borrows the tolerance and
/// the threshold buffer.
#[derive(Debug, Clone, Copy)]
pub struct PreparedErrorNorm<'a, S: RealScalar> {
    tolerance: &'a ErrorTolerance<S>,
    thresholds: &'a [S],
}

impl<S: RealScalar> PreparedErrorNorm<'_, S> {
    /// Tests whether the normalized error norm is at most one; `error(i)` returns component `i`.
    ///
    /// `error` is called once per component, in index order, including after a
    /// component has already failed. With the infinity norm each `|eᵢ|` is
    /// compared with its threshold directly, so no quotient can overflow.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an error component is NaN or infinite.
    /// - For norms other than the infinity norm, `NumericalInstability`
    ///   (`Overflow`) if a quotient or the norm exceeds the largest finite value.
    #[inline(always)]
    pub fn is_satisfied_by(
        &self,
        mut error: impl FnMut(usize) -> S,
    ) -> Result<bool, SciOxideError> {
        if self.tolerance.norm != NormSelection::Infinity {
            return self
                .tolerance
                .normalized_values(
                    self.thresholds.len(),
                    |i| (error(i), S::ZERO),
                    Some(self.thresholds),
                )
                .map(|value| value.is_some_and(|norm| norm <= S::ONE));
        }
        let mut within = true;
        for (i, &threshold) in self.thresholds.iter().enumerate() {
            let value = error(i);
            if !value.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "error component must be finite",
                ));
            }
            within &= value.abs() <= threshold;
        }
        Ok(within)
    }

    /// Returns the normalized error norm `‖r‖`; `error(i)` returns component `i`.
    ///
    /// `error` is called once per component, in index order.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an error component is NaN or infinite; this is checked
    ///   for every component before an overflow is reported.
    /// - `NumericalInstability` (`Overflow`) if a nonzero error meets a zero
    ///   threshold, or a quotient or the norm exceeds the largest finite value.
    #[inline(always)]
    pub fn normalized_norm_by(
        &self,
        mut error: impl FnMut(usize) -> S,
    ) -> Result<S, SciOxideError> {
        self.tolerance
            .normalized_values(
                self.thresholds.len(),
                |i| (error(i), S::ZERO),
                Some(self.thresholds),
            )?
            .ok_or_else(unbounded_error_norm)
    }
}

fn unbounded_error_norm() -> SciOxideError {
    SciOxideError::numerical_instability(
        crate::NumericalIssue::Overflow,
        "a nonzero error divided by an exact zero tolerance is unbounded",
    )
}

impl<S: RealScalar> ErrorTolerance<S> {
    /// Creates a tolerance with one absolute tolerance for all components.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `absolute` or `relative` is negative.
    /// - `NonFinite` if either is NaN or infinite.
    pub fn scalar(absolute: S, relative: S, norm: NormSelection) -> Result<Self, SciOxideError> {
        validate_nonnegative(relative, crate::argument!("relative"))?;
        validate_nonnegative(absolute, crate::argument!("absolute"))?;
        Ok(Self {
            absolute: AbsoluteTolerance(AbsolutePolicy::Scalar(absolute)),
            relative,
            norm,
        })
    }

    /// Creates a tolerance with one absolute tolerance per component, copying `absolute`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `relative` or an absolute value is negative.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `AllocationFailure` if the copy cannot be stored.
    pub fn try_vector(
        absolute: &[S],
        relative: S,
        norm: NormSelection,
    ) -> Result<Self, SciOxideError> {
        validate_nonnegative(relative, crate::argument!("relative"))?;
        Ok(Self {
            absolute: AbsoluteTolerance::validated_vector(absolute, crate::argument!("absolute"))?,
            relative,
            norm,
        })
    }

    /// Creates a componentwise tolerance from a [`Tolerance`] pair and a norm.
    pub fn from_tolerance(tolerance: Tolerance<S>, norm: NormSelection) -> Self {
        Self {
            absolute: AbsoluteTolerance(AbsolutePolicy::Scalar(tolerance.absolute())),
            relative: tolerance.relative(),
            norm,
        }
    }

    /// Returns the absolute tolerance.
    pub const fn absolute(&self) -> &AbsoluteTolerance<S> {
        &self.absolute
    }

    /// Returns the relative tolerance.
    pub const fn relative(&self) -> S {
        self.relative
    }

    /// Returns the norm applied to the normalized components.
    pub const fn norm(&self) -> NormSelection {
        self.norm
    }

    /// Returns the threshold `atolᵢ + rtol·scale` of component `index` in a problem with `dimension` components.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `scale` is negative.
    /// - `NonFinite` if `scale` is NaN or infinite.
    /// - `DimensionMismatch` or `OutOfBounds` as in [`AbsoluteTolerance::component`].
    /// - `NumericalInstability` (`Overflow`) if the threshold exceeds the
    ///   largest finite value.
    pub fn threshold(&self, index: usize, dimension: usize, scale: S) -> Result<S, SciOxideError> {
        self.threshold_for(index, dimension, scale, crate::argument!("scale"))
    }
    #[inline(always)]
    fn threshold_for(
        &self,
        index: usize,
        dimension: usize,
        scale: S,
        argument: crate::Argument,
    ) -> Result<S, SciOxideError> {
        validate_nonnegative(scale, argument)?;
        let absolute = self.absolute.component(index, dimension)?;
        threshold_value(absolute, self.relative, scale)
    }

    /// Computes the thresholds `atolᵢ + rtol·scalesᵢ` into `storage`, for reuse with several error vectors.
    ///
    /// `storage` must have the same length as `scales`. On error its contents
    /// are unspecified.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `storage` and `scales` differ in length, or a
    ///   per-component absolute tolerance has another length.
    /// - `InvalidInput` (`OutOfRange`) if a scale is negative.
    /// - `NonFinite` if a scale is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a threshold exceeds the largest
    ///   finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ErrorTolerance, SciOxideError, NormSelection};
    ///
    /// // Thresholds are 1 + 0.5·2 = 2, so the errors [3, 4] normalize to [1.5, 2].
    /// let tolerance = ErrorTolerance::scalar(1.0_f64, 0.5, NormSelection::Infinity)?;
    /// let mut storage = [0.0; 2];
    /// let norm = tolerance.prepare_norm(&[2.0, 2.0], &mut storage)?;
    /// assert_eq!(norm.normalized_norm_by(|i| [3.0, 4.0][i])?, 2.0);
    /// assert!(norm.is_satisfied_by(|_| 1.0)?);
    /// # Ok::<(), SciOxideError>(())
    /// ```
    pub fn prepare_norm<'a>(
        &'a self,
        scales: &[S],
        storage: &'a mut [S],
    ) -> Result<PreparedErrorNorm<'a, S>, SciOxideError> {
        if scales.len() != storage.len() {
            return Err(SciOxideError::dimension_mismatch(
                scales.len(),
                storage.len(),
            ));
        }
        self.absolute.validate_dimension(scales.len())?;
        for (i, (out, &scale)) in storage.iter_mut().zip(scales).enumerate() {
            validate_nonnegative(scale, crate::argument!("scales"))?;
            let absolute = match &self.absolute.0 {
                AbsolutePolicy::Scalar(value) => *value,
                AbsolutePolicy::Vector(values) => values[i],
            };
            *out = threshold_value(absolute, self.relative, scale)?;
        }
        Ok(PreparedErrorNorm {
            tolerance: self,
            thresholds: storage,
        })
    }

    /// Returns the normalized error norm `‖r‖` with `rᵢ = |errorsᵢ| / (atolᵢ + rtol·scalesᵢ)`.
    ///
    /// An empty vector has norm zero. The cost is O(n) time and constant memory.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `errors` and `scales` differ in length, or a
    ///   per-component absolute tolerance has another length.
    /// - `NonFinite` if an error or scale is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if a scale is negative.
    /// - `NumericalInstability` (`Overflow`) if a nonzero error meets a zero
    ///   threshold, or a threshold, a quotient or the norm exceeds the largest
    ///   finite value.
    #[inline]
    pub fn normalized_norm(&self, errors: &[S], scales: &[S]) -> Result<S, SciOxideError> {
        if let ([error], [scale]) = (errors, scales) {
            self.absolute.validate_dimension(1)?;
            if !error.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "error component must be finite",
                ));
            }
            let threshold = self.threshold_for(0, 1, *scale, crate::argument!("scales"))?;
            let magnitude = error.abs();
            if threshold != S::ZERO {
                return checked_float_quotient(magnitude, threshold);
            }
            return if magnitude == S::ZERO {
                Ok(S::ZERO)
            } else {
                Err(SciOxideError::numerical_instability(
                    crate::NumericalIssue::Overflow,
                    "a nonzero error divided by an exact zero tolerance is unbounded",
                ))
            };
        }
        self.normalized_norm_finite(errors, scales)?.ok_or_else(|| {
            SciOxideError::numerical_instability(
                crate::NumericalIssue::Overflow,
                "a nonzero error divided by an exact zero tolerance is unbounded",
            )
        })
    }
    fn normalized_norm_finite(
        &self,
        errors: &[S],
        scales: &[S],
    ) -> Result<Option<S>, SciOxideError> {
        if errors.len() != scales.len() {
            return Err(SciOxideError::dimension_mismatch(
                crate::Dimensions::one(errors.len()),
                crate::Dimensions::one(scales.len()),
            ));
        }

        self.absolute.validate_dimension(errors.len())?;

        self.normalized_values(errors.len(), |i| (errors[i], scales[i]), None)
    }

    /// Returns the normalized error norm, where `component(i)` returns the pair `(errorᵢ, scaleᵢ)`.
    ///
    /// This is [`Self::normalized_norm`] without building the two slices.
    /// `component` is called once per index, in order.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::normalized_norm`]; `DimensionMismatch` refers to
    /// `dimension`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ErrorTolerance, SciOxideError, NormSelection};
    ///
    /// // Thresholds are 1 + 0.5·2 = 2; max(3/2, 4/2) = 2.
    /// let tolerance = ErrorTolerance::scalar(1.0_f64, 0.5, NormSelection::Infinity)?;
    /// let components = [(3.0, 2.0), (4.0, 2.0)];
    /// assert_eq!(tolerance.normalized_norm_by(2, |i| components[i])?, 2.0);
    /// # Ok::<(), SciOxideError>(())
    /// ```
    #[inline(always)]
    pub fn normalized_norm_by(
        &self,
        dimension: usize,
        mut component: impl FnMut(usize) -> (S, S),
    ) -> Result<S, SciOxideError> {
        self.absolute.validate_dimension(dimension)?;
        if dimension == 1 {
            let (error, scale) = component(0);
            return self.normalized_norm(&[error], &[scale]);
        }
        self.normalized_values(dimension, component, None)?
            .ok_or_else(|| {
                SciOxideError::numerical_instability(
                    crate::NumericalIssue::Overflow,
                    "a nonzero error divided by an exact zero tolerance is unbounded",
                )
            })
    }

    #[inline(always)]
    fn normalized_values(
        &self,
        dimension: usize,
        component: impl FnMut(usize) -> (S, S),
        thresholds: Option<&[S]>,
    ) -> Result<Option<S>, SciOxideError> {
        // Construct only the selected norm state. In particular, infinity
        // norms never initialize the exact L1 accumulator's limb arrays.
        match self.norm {
            NormSelection::Infinity => self.normalized_norm_accumulate(
                dimension,
                component,
                S::ZERO,
                |maximum, ratio| {
                    // Both operands are validated finite, nonnegative ratios.
                    if ratio > *maximum {
                        *maximum = ratio;
                    }
                    Ok(())
                },
                Ok,
                thresholds,
            ),
            NormSelection::L1 => self.normalized_l1(dimension, component, thresholds),
            NormSelection::L2 | NormSelection::RootMeanSquare => {
                self.normalized_l2(dimension, component, thresholds)
            }
        }
    }

    #[inline(never)]
    fn normalized_l2(
        &self,
        dimension: usize,
        component: impl FnMut(usize) -> (S, S),
        thresholds: Option<&[S]>,
    ) -> Result<Option<S>, SciOxideError> {
        self.normalized_norm_accumulate(
            dimension,
            component,
            ScaledNorm::new(),
            |sum, ratio| sum.add_magnitude(ratio),
            |sum| {
                if self.norm == NormSelection::RootMeanSquare {
                    if dimension == 0 {
                        return Ok(S::ZERO);
                    }
                    crate::checked_float_product(
                        sum.scale(),
                        (sum.sum_squares() / S::try_from_usize_exact(dimension)?).sqrt(),
                    )
                } else {
                    sum.finish()
                }
            },
            thresholds,
        )
    }

    // Keep exact L1 limb storage out of the scalar/infinity norm stack frame.
    #[inline(never)]
    fn normalized_l1(
        &self,
        dimension: usize,
        component: impl FnMut(usize) -> (S, S),
        thresholds: Option<&[S]>,
    ) -> Result<Option<S>, SciOxideError> {
        self.normalized_norm_accumulate(
            dimension,
            component,
            RangeAccumulator::new(),
            |sum, ratio| sum.add(ratio),
            |sum| sum.finish(),
            thresholds,
        )
    }

    #[inline(always)]
    fn normalized_norm_accumulate<A>(
        &self,
        dimension: usize,
        mut component: impl FnMut(usize) -> (S, S),
        mut accumulator: A,
        mut add: impl FnMut(&mut A, S) -> Result<(), SciOxideError>,
        finish: impl FnOnce(A) -> Result<S, SciOxideError>,
        thresholds: Option<&[S]>,
    ) -> Result<Option<S>, SciOxideError> {
        // Validate every later component even after an arithmetic failure or
        // an exact-zero threshold, retaining the public error precedence.
        let mut unbounded = false;
        let mut arithmetic_error = None;
        for index in 0..dimension {
            let (error, scale) = component(index);
            // The immutable policy and shape were admitted before reduction.
            // Ordinary components need only one combined range/input guard;
            // exceptional components retain the diagnostic precedence below.
            let absolute = if thresholds.is_some() {
                S::ZERO
            } else {
                match &self.absolute.0 {
                    AbsolutePolicy::Scalar(value) => *value,
                    AbsolutePolicy::Vector(values) => values[index],
                }
            };
            if !unbounded && arithmetic_error.is_none() {
                let threshold = thresholds.map_or_else(
                    || crate::fast_mul_add(self.relative, scale, absolute),
                    |values| values[index],
                );
                let ratio = error.abs() / threshold;
                if ratio.is_finite()
                    & (thresholds.is_some()
                        | ((scale >= S::ZERO) & scale.is_finite() & threshold.is_finite()))
                {
                    if let Err(error) = add(&mut accumulator, ratio) {
                        arithmetic_error = Some(error);
                    }
                    continue;
                }
            }
            if !error.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "error component must be finite",
                ));
            }
            let threshold = if let Some(values) = thresholds {
                values[index]
            } else {
                validate_nonnegative(scale, crate::argument!("scales"))?;
                threshold_value(absolute, self.relative, scale)?
            };
            let magnitude = error.abs();
            if threshold == S::ZERO && magnitude != S::ZERO {
                unbounded = true;
            }
            if unbounded || arithmetic_error.is_some() {
                continue;
            }
            let normalized = if threshold == S::ZERO {
                Ok(S::ZERO)
            } else {
                checked_final_rounding(
                    magnitude / threshold,
                    "quotient is outside the finite scalar range",
                )
            };
            match normalized {
                Ok(ratio) => {
                    if let Err(error) = add(&mut accumulator, ratio) {
                        arithmetic_error = Some(error);
                    }
                }
                Err(error) => arithmetic_error = Some(error),
            }
        }
        if unbounded {
            return Ok(None);
        }
        if let Some(error) = arithmetic_error {
            return Err(error);
        }
        let norm = finish(accumulator)?;
        Ok(Some(norm))
    }

    /// Tests whether the normalized error norm `‖r‖` is at most one.
    ///
    /// A nonzero error against a zero threshold returns `Ok(false)`, not an
    /// error.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `errors` and `scales` differ in length, or a
    ///   per-component absolute tolerance has another length.
    /// - `NonFinite` if an error or scale is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if a scale is negative.
    /// - `NumericalInstability` (`Overflow`) if a threshold, a quotient or the
    ///   norm exceeds the largest finite value.
    #[inline(always)]
    pub fn is_satisfied(&self, errors: &[S], scales: &[S]) -> Result<bool, SciOxideError> {
        if let ([error], [scale]) = (errors, scales) {
            self.absolute.validate_dimension(1)?;
            if !error.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "error component must be finite",
                ));
            }
            let threshold = self.threshold_for(0, 1, *scale, crate::argument!("scales"))?;
            let magnitude = error.abs();
            // For positive IEEE values, the next value above a threshold
            // divided by that threshold rounds strictly above one. Comparing
            // directly is therefore equivalent to the rounded-ratio predicate.
            if magnitude <= threshold {
                return Ok(true);
            }
            if threshold == S::ZERO {
                return Ok(false);
            }
            // Preserve representability errors for extreme ratios. These
            // ordinary ranges prove a finite quotient in both scalar widths.
            if threshold >= S::ONE || threshold >= S::EPSILON && magnitude <= S::ONE / S::EPSILON {
                return Ok(false);
            }
            return checked_float_quotient(magnitude, threshold).map(|ratio| ratio <= S::ONE);
        }
        Ok(self
            .normalized_norm_finite(errors, scales)?
            .is_some_and(|norm| norm <= S::ONE))
    }
}

impl<S: RealScalar> Default for ErrorTolerance<S> {
    fn default() -> Self {
        let tolerance = Tolerance::from_sqrt_epsilon();
        Self::from_tolerance(tolerance, NormSelection::RootMeanSquare)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_error_norms_match_closed_form_and_keep_validation_precedence() {
        for (norm, expected) in [
            (NormSelection::Infinity, 2.0_f64),
            (NormSelection::L1, 3.5),
            (NormSelection::L2, 2.5),
            (NormSelection::RootMeanSquare, (3.125_f64).sqrt()),
        ] {
            let tolerance = ErrorTolerance::scalar(1.0, 0.5, norm).unwrap();
            let values = [(3.0, 2.0), (4.0, 2.0)];
            let actual = tolerance.normalized_norm_by(2, |i| values[i]).unwrap();
            assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected);
            let mut thresholds = [0.0; 2];
            let prepared = tolerance
                .prepare_norm(&[2.0, 2.0], &mut thresholds)
                .unwrap();
            for errors in [[3.0, 4.0], [-3.0, -4.0]] {
                assert!(!prepared.is_satisfied_by(|i| errors[i]).unwrap());
                assert!(prepared.is_satisfied_by(|i| errors[i] * 0.1).unwrap());
                let actual = prepared.normalized_norm_by(|i| errors[i]).unwrap();
                assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected);
            }
            assert_eq!(
                tolerance.normalized_norm_by(0, |_| unreachable!()).unwrap(),
                0.0
            );
        }
        let exact = ErrorTolerance::scalar(0.0_f64, 0.0, NormSelection::Infinity).unwrap();
        let error = exact
            .normalized_norm_by(2, |i| if i == 0 { (1.0, 0.0) } else { (f64::NAN, 0.0) })
            .unwrap_err();
        assert_eq!(error.kind(), crate::SciOxideErrorKind::NonFinite);
        let vector =
            ErrorTolerance::try_vector(&[1.0_f64, 1.0], 0.0, NormSelection::Infinity).unwrap();
        let mut calls = 0;
        assert!(
            vector
                .normalized_norm_by(3, |_| {
                    calls += 1;
                    (0.0, 0.0)
                })
                .is_err()
        );
        assert_eq!(calls, 0);
        assert!(vector.prepare_norm(&[0.0; 2], &mut [0.0; 3]).is_err());
        assert!(
            vector
                .prepare_norm(&[f64::NAN, 0.0], &mut [0.0; 2])
                .is_err()
        );
        assert!(vector.prepare_norm(&[-1.0, 0.0], &mut [0.0; 2]).is_err());
        let mut thresholds = [0.0; 2];
        let prepared = exact.prepare_norm(&[0.0; 2], &mut thresholds).unwrap();
        assert!(prepared.is_satisfied_by(|_| 0.0).unwrap());
        assert!(!prepared.is_satisfied_by(|_| 1.0).unwrap());
        assert!(
            prepared
                .is_satisfied_by(|i| if i == 0 { 1.0 } else { f64::NAN })
                .is_err()
        );
        assert_eq!(prepared.normalized_norm_by(|_| 0.0).unwrap(), 0.0);
        assert!(prepared.normalized_norm_by(|_| 1.0).is_err());
        let error = prepared
            .normalized_norm_by(|i| if i == 0 { 1.0 } else { f64::NAN })
            .unwrap_err();
        assert_eq!(error.kind(), crate::SciOxideErrorKind::NonFinite);
        let tiny = ErrorTolerance::scalar(f64::MIN_POSITIVE, 0.0, NormSelection::Infinity).unwrap();
        let mut threshold = [0.0];
        let prepared = tiny.prepare_norm(&[0.0], &mut threshold).unwrap();
        assert!(!prepared.is_satisfied_by(|_| f64::MAX).unwrap());
        let rms = ErrorTolerance::scalar(1.0, 0.0, NormSelection::RootMeanSquare).unwrap();
        let large = f64::MAX / 2.0;
        let actual = rms.normalized_norm_by(16, |_| (large, 0.0)).unwrap();
        assert!((actual / large - 1.0).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn scalar_predicates_match_rounded_ratios_at_adjacent_and_extreme_thresholds() {
        fn check<S: RealScalar>() {
            let tiny = S::MIN_POSITIVE * S::EPSILON;
            for norm in [
                NormSelection::L1,
                NormSelection::L2,
                NormSelection::RootMeanSquare,
                NormSelection::Infinity,
            ] {
                for threshold in [
                    S::ZERO,
                    tiny,
                    S::MIN_POSITIVE,
                    S::EPSILON,
                    S::SQRT_EPSILON,
                    S::ONE,
                    S::MAX,
                ] {
                    let tolerance = ErrorTolerance::scalar(threshold, S::ZERO, norm).unwrap();
                    for error in [
                        S::ZERO,
                        tiny,
                        S::ONE,
                        S::MAX,
                        threshold,
                        threshold.next_after(S::ZERO),
                        threshold.next_after(S::INFINITY),
                    ] {
                        if !error.is_finite() {
                            continue;
                        }
                        let actual = tolerance.is_satisfied(&[error], &[S::ONE]);
                        if threshold == S::ZERO && error != S::ZERO {
                            assert!(!actual.unwrap());
                            continue;
                        }
                        match (actual, tolerance.normalized_norm(&[error], &[S::ONE])) {
                            (Ok(actual), Ok(ratio)) => assert_eq!(actual, ratio <= S::ONE),
                            (actual, expected) => {
                                assert_eq!(actual.unwrap_err().kind(), expected.unwrap_err().kind())
                            }
                        }
                    }
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn unbounded_predicate_still_validates_every_component() {
        for norm in [
            NormSelection::L1,
            NormSelection::L2,
            NormSelection::Infinity,
            NormSelection::RootMeanSquare,
        ] {
            let tolerance =
                ErrorTolerance::try_vector(&[f64::MIN_POSITIVE, 0.0], 0.0, norm).unwrap();
            assert!(
                !tolerance
                    .is_satisfied(&[f64::MAX, 1.0], &[1.0, 1.0])
                    .unwrap()
            );
            let error = tolerance
                .is_satisfied(&[f64::MAX, f64::NAN], &[1.0, 1.0])
                .unwrap_err();
            assert_eq!(error.kind(), crate::SciOxideErrorKind::NonFinite);
            let tolerance = ErrorTolerance::scalar(0.0, 0.0, norm).unwrap();
            assert!(!tolerance.is_satisfied(&[1.0, 0.0], &[1.0, 1.0]).unwrap());
            assert!(
                tolerance
                    .is_satisfied(&[1.0, 0.0], &[1.0, f64::NAN])
                    .is_err()
            );
        }
    }

    #[test]
    fn construction_and_accessors_preserve_invariants() {
        let tolerance = Tolerance::<f64>::new(1e-8, 1e-6).unwrap();

        assert_eq!(tolerance.absolute(), 1e-8);
        assert_eq!(tolerance.relative(), 1e-6);
        assert_eq!(tolerance.with_absolute(1e-10).unwrap().absolute(), 1e-10);
        assert_eq!(tolerance.with_relative(1e-4).unwrap().relative(), 1e-4);
    }

    #[test]
    fn construction_rejects_negative_and_non_finite_values() {
        for result in [
            Tolerance::<f64>::new(-1.0, 0.0),
            Tolerance::<f64>::new(0.0, -1.0),
            Tolerance::<f64>::new(f64::NAN, 0.0),
            Tolerance::<f64>::new(0.0, f64::INFINITY),
        ] {
            assert!(matches!(
                (result).as_ref().unwrap_err().view(),
                crate::SciOxideErrorView::InvalidInput { .. }
                    | crate::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }

    #[test]
    fn named_default_policies_support_both_precisions() {
        let f64_tolerance = Tolerance::<f64>::from_sqrt_epsilon();
        let f32_tolerance = Tolerance::<f32>::from_sqrt_epsilon();
        let ode = Tolerance::<f64>::from_decimal_places(6, 3);

        assert_eq!(f64_tolerance.absolute(), <f64 as RealScalar>::SQRT_EPSILON);
        assert_eq!(f32_tolerance.relative(), <f32 as RealScalar>::SQRT_EPSILON);
        assert!((ode.absolute() - 1e-6).abs() < f64::EPSILON);
        assert!((ode.relative() - 1e-3).abs() < f64::EPSILON);
    }

    #[test]
    fn residual_uses_explicit_relative_scale() {
        let tolerance = Tolerance::<f64>::new(1e-8, 1e-3).unwrap();

        assert!(!tolerance.is_residual_ok(0.5, 1.0).unwrap());
        assert!(tolerance.is_residual_ok(0.5, 1_000.0).unwrap());
    }

    #[test]
    fn valid_non_convergence_is_not_an_error() {
        let tolerance = Tolerance::<f64>::new(1e-12, 1e-12).unwrap();

        assert!(!tolerance.is_residual_ok(1.0, 1.0).unwrap());
        assert!(!tolerance.is_converged(2.0, 1.0).unwrap());
    }

    #[test]
    fn malformed_residuals_and_scales_are_errors() {
        let tolerance = Tolerance::<f64>::from_sqrt_epsilon();

        assert!(matches!(
            (tolerance.is_residual_ok(-1.0, 1.0))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(matches!(
            (tolerance.is_residual_ok(f64::NAN, 1.0))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
        assert!(matches!(
            (tolerance.is_residual_ok(1.0, -1.0))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(matches!(
            (tolerance.is_residual_ok(1.0, f64::INFINITY))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn threshold_overflow_is_an_error() {
        let tolerance = Tolerance::<f64>::new(f64::MAX, f64::MAX).unwrap();

        assert!(matches!(
            (tolerance.threshold(2.0)).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::NumericalInstability {
                issue: crate::NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn non_finite_and_opposite_maximum_iterates_have_distinct_outcomes() {
        let tolerance = Tolerance::<f64>::from_sqrt_epsilon();

        assert!(matches!(
            (tolerance.is_converged(f64::NAN, 1.0))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
        assert!(!tolerance.is_converged(f64::MAX, -f64::MAX).unwrap());
    }

    #[test]
    fn comparisons_cover_adjacent_subnormal_and_maximum_scales() {
        let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);
        let ulp = adjacent - 1.0;
        let exact = Tolerance::<f64>::new(ulp, 0.0).unwrap();
        assert!(exact.is_converged(adjacent, 1.0).unwrap());

        let minimum = f64::from_bits(1);
        let subnormal = Tolerance::<f64>::new(0.0, 1.0).unwrap();
        assert_eq!(subnormal.threshold(minimum).unwrap(), minimum);
        assert!(subnormal.is_converged(minimum, 0.0).unwrap());

        let maximum = Tolerance::<f64>::new(0.0, f64::EPSILON).unwrap();
        assert!(maximum.threshold(f64::MAX).unwrap().is_finite());
        assert!(!maximum.is_converged(f64::MAX, -f64::MAX).unwrap());

        let adjacent_f32 = f32::from_bits(1.0_f32.to_bits() + 1);
        let exact_f32 = Tolerance::<f32>::new(adjacent_f32 - 1.0, 0.0).unwrap();
        assert!(exact_f32.is_converged(adjacent_f32, 1.0).unwrap());
        let minimum_f32 = f32::from_bits(1);
        let subnormal_f32 = Tolerance::<f32>::new(0.0, 1.0).unwrap();
        assert_eq!(subnormal_f32.threshold(minimum_f32).unwrap(), minimum_f32);
        assert!(
            !Tolerance::<f32>::from_sqrt_epsilon()
                .is_converged(f32::MAX, -f32::MAX)
                .unwrap()
        );
    }

    #[test]
    fn role_tolerances_retain_units_and_component_error_norms_are_centralized() {
        let residual = ResidualTolerance::new(1e-8_f64, 1e-6).unwrap();
        assert_eq!(residual.absolute(), 1e-8);
        assert_eq!(residual.relative(), 1e-6);

        let rms = ErrorTolerance::try_vector(&[1.0_f64, 2.0], 0.0, NormSelection::RootMeanSquare)
            .unwrap();
        assert_eq!(rms.normalized_norm(&[1.0, 2.0], &[0.0, 0.0]).unwrap(), 1.0);
        assert!(rms.is_satisfied(&[1.0, 2.0], &[0.0, 0.0]).unwrap());
        assert!(!rms.is_satisfied(&[2.0, 4.0], &[0.0, 0.0]).unwrap());

        let exact = ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap();
        assert!(exact.is_satisfied(&[0.0], &[0.0]).unwrap());
        assert!(!exact.is_satisfied(&[f64::from_bits(1)], &[0.0]).unwrap());
        assert!(matches!(
            (exact.normalized_norm(&[f64::from_bits(1)], &[0.0]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NumericalInstability {
                issue: crate::NumericalIssue::Overflow,
                ..
            }
        ));
    }
    #[test]
    fn integrality_tolerance_excludes_half_integers() {
        macro_rules! check {
            ($scalar:ty) => {{
                for invalid in [-1.0, 0.5, 1.0, <$scalar>::MAX] {
                    assert!(matches!(
                        IntegralityTolerance::<$scalar>::new(invalid),
                        Err(error) if matches!(error.view(), crate::SciOxideErrorView::InvalidInput { .. })
                    ));
                }
                for invalid in [<$scalar>::NAN, <$scalar>::INFINITY] {
                    assert!(matches!((IntegralityTolerance :: new (invalid)).as_ref().unwrap_err().view(), crate :: SciOxideErrorView :: NonFinite { .. }));
                }
                let below_half = <$scalar>::from_bits((0.5 as $scalar).to_bits() - 1);
                let tolerance = IntegralityTolerance::new(below_half).unwrap();
                assert!(tolerance.is_satisfied(below_half).unwrap());
                assert!(!tolerance.is_satisfied(0.5).unwrap());
                assert!(
                    IntegralityTolerance::<$scalar>::new(0.0)
                        .unwrap()
                        .is_satisfied(0.0)
                        .unwrap()
                );
            }};
        }
        check!(f32);
        check!(f64);
    }

    #[test]
    fn shared_validation_reports_parameter_and_non_finite_origin() {
        macro_rules! check {
            ($scalar:ty, $role:ident) => {
                for (absolute, relative, argument) in
                    [(-1.0, 0.0, "absolute"), (0.0, -1.0, "relative")]
                {
                    let error = $role::<$scalar>::new(absolute, relative).unwrap_err();
                    let crate::SciOxideErrorView::InvalidInput {
                        argument: _view_argument,
                        reason: _view_reason,
                        detail: _view_detail,
                    } = error.view()
                    else {
                        unreachable!("unexpected error kind")
                    };

                    assert_eq!(_view_argument.as_str(), argument);
                }
                for (absolute, relative, _argument) in [
                    (<$scalar>::NAN, 0.0, "absolute"),
                    (0.0, <$scalar>::INFINITY, "relative"),
                ] {
                    let error = $role::<$scalar>::new(absolute, relative).unwrap_err();
                    let crate::SciOxideErrorView::NonFinite {
                        origin: _view_origin,
                        detail: _view_detail,
                    } = error.view()
                    else {
                        unreachable!("unexpected error kind")
                    };

                    assert_eq!(_view_origin, NonFiniteOrigin::Input);
                    assert_eq!(_view_detail, "tolerance value must be finite");
                }
                let tolerance = $role::<$scalar>::new(<$scalar>::MAX, 1.0).unwrap();
                let error = tolerance.threshold(<$scalar>::MAX).unwrap_err();
                assert!(matches!(
                    error.view(),
                    crate::SciOxideErrorView::NumericalInstability {
                        issue: crate::NumericalIssue::Overflow,
                        ..
                    }
                ));

                let zero = $role::<$scalar>::new(-0.0, <$scalar>::from_bits(1)).unwrap();
                assert_eq!(zero.absolute().to_bits(), (-0.0 as $scalar).to_bits());
                assert_eq!(zero.relative().to_bits(), 1);
            };
        }
        macro_rules! roles {
            ($scalar:ty) => {
                check!($scalar, Tolerance);
                check!($scalar, StepTolerance);
                check!($scalar, ResidualTolerance);
                check!($scalar, GradientTolerance);
                check!($scalar, RankThreshold);
                check!($scalar, FeasibilityTolerance);
            };
        }
        roles!(f32);
        roles!(f64);
    }

    #[test]
    fn component_bounds_and_empty_vector_shapes_are_validated() {
        for tolerance in [
            AbsoluteTolerance::scalar(0.0_f64).unwrap(),
            AbsoluteTolerance::try_vector(&[]).unwrap(),
        ] {
            assert!(matches!(
                tolerance.component(0, 0).unwrap_err().view(),
                crate::SciOxideErrorView::OutOfBounds { .. }
            ));
        }
        let scalar = AbsoluteTolerance::scalar(1.0_f64).unwrap();
        assert!(matches!(
            scalar.component(usize::MAX, 1).unwrap_err().view(),
            crate::SciOxideErrorView::OutOfBounds { .. }
        ));
        let mismatch = ErrorTolerance::try_vector(&[1.0_f64], 0.0, NormSelection::L2).unwrap();
        assert!(matches!(
            mismatch.normalized_norm(&[], &[]).unwrap_err().view(),
            crate::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert!(matches!(
            mismatch.is_satisfied(&[], &[]).unwrap_err().view(),
            crate::SciOxideErrorView::DimensionMismatch { .. }
        ));
        let empty = ErrorTolerance::try_vector(&[], 0.0_f64, NormSelection::L2).unwrap();
        assert_eq!(empty.normalized_norm(&[], &[]).unwrap(), 0.0);
        assert!(empty.is_satisfied(&[], &[]).unwrap());
    }

    #[test]
    fn builders_and_error_norms_report_invalid_arguments() {
        let tolerance = Tolerance::<f64>::new(1.0, 0.0).unwrap();

        let error = tolerance.is_residual_ok(0.0, -1.0).unwrap_err();
        assert_eq!(
            match error.view() {
                crate::SciOxideErrorView::InvalidInput { argument, .. } => Some(argument),
                _ => None,
            }
            .unwrap(),
            crate::argument!("problem_scale")
        );
        let tolerance = ErrorTolerance::scalar(f64::MAX, 1.0, NormSelection::Infinity).unwrap();
        let error = tolerance.normalized_norm(&[0.0], &[f64::MAX]).unwrap_err();
        assert!(matches!(
            error.view(),
            crate::SciOxideErrorView::NumericalInstability {
                issue: crate::NumericalIssue::Overflow,
                ..
            }
        ));

        let error = tolerance.is_satisfied(&[0.0], &[-1.0]).unwrap_err();
        let crate::SciOxideErrorView::InvalidInput {
            argument: _view_argument,
            reason: _view_reason,
            detail: _view_detail,
        } = error.view()
        else {
            unreachable!("unexpected error kind")
        };

        assert_eq!(_view_argument, crate::argument!("scales"));
    }
}

#[cfg(test)]
mod relative_tests {
    use super::*;
    #[test]
    fn relative_thresholds_preserve_validation_boundaries() {
        for value in [0.0, f64::MIN_POSITIVE, 1e-12, f64::MAX] {
            let relative = RelativeTolerance::new(value).unwrap();
            assert_eq!(relative.relative(), value);
            let converted: Tolerance<f64> = relative.into();
            assert_eq!(converted.absolute(), 0.0);
            assert_eq!(converted.relative(), value);
        }
        assert_eq!(
            RelativeTolerance::new(-1.0_f64).unwrap_err().kind(),
            crate::SciOxideErrorKind::InvalidInput
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                RelativeTolerance::new(value).unwrap_err().kind(),
                crate::SciOxideErrorKind::NonFinite
            );
        }
        assert_eq!(RelativeTolerance::<f32>::default().relative(), f32::EPSILON);
        assert_eq!(RelativeTolerance::<f64>::default().relative(), f64::EPSILON);
        let single: Tolerance<f32> = RelativeTolerance::default().into();
        assert_eq!(single.absolute(), 0.0);
        assert_eq!(single.relative(), f32::EPSILON);
    }
}
