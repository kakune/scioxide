//! Solve identities, accepted-state metadata, residual reports and estimates.

use crate::{
    InvalidInputReason, NonFiniteOrigin, NormSelection, RealScalar, SciOxideError,
    checked_float_quotient,
};
use core::fmt;

/// Identity of an algorithm: family, method and version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct AlgorithmId {
    family: crate::ValidatedIdentifier<'static>,
    method: crate::ValidatedIdentifier<'static>,
    version: u32,
}

impl AlgorithmId {
    /// Creates an identity from a family name, a method name and a version.
    ///
    /// Any `u32` is a valid version. The [`algorithm_id!`](crate::algorithm_id)
    /// macro checks the names at compile time.
    ///
    /// # Errors
    ///
    /// - [`crate::IdentifierError`] if `family` or `method` is not a valid
    ///   [`crate::AlgorithmComponent`].
    pub const fn new(
        family: &'static str,
        method: &'static str,
        version: u32,
    ) -> Result<Self, crate::IdentifierError> {
        let family = match crate::AlgorithmComponent::new(family) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        let method = match crate::AlgorithmComponent::new(method) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        Ok(Self::from_components(family, method, version))
    }

    /// Creates an identity from components that are already validated.
    ///
    /// Use it to combine a family and a method chosen separately at run time.
    ///
    /// ```
    /// use scioxide_core::{AlgorithmId, algorithm_component};
    ///
    /// let method = if cfg!(debug_assertions) {
    ///     algorithm_component!("checked")
    /// } else {
    ///     algorithm_component!("fast")
    /// };
    /// let identity = AlgorithmId::from_components(algorithm_component!("linear"), method, 2);
    /// assert_eq!(identity.family(), "linear");
    /// assert_eq!(identity.version(), 2);
    /// ```
    pub const fn from_components(
        family: crate::AlgorithmComponent,
        method: crate::AlgorithmComponent,
        version: u32,
    ) -> Self {
        Self {
            family: crate::ValidatedIdentifier::component(family),
            method: crate::ValidatedIdentifier::component(method),
            version,
        }
    }

    /// Returns the family name.
    pub const fn family(self) -> &'static str {
        self.family.as_str()
    }

    /// Returns the method name within the family.
    pub const fn method(self) -> &'static str {
        self.method.as_str()
    }

    /// Returns the version.
    pub const fn version(self) -> u32 {
        self.version
    }
}

impl fmt::Display for AlgorithmId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}/{}@{}",
            self.family, self.method, self.version
        )
    }
}

/// Token you choose to identify the mathematical problem of a solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct ProblemToken(u128);

impl ProblemToken {
    /// Creates a token from `value`.
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Returns the token value.
    pub const fn get(self) -> u128 {
        self.0
    }
}

/// Identity of a solve: problem, algorithm, shape and options.
///
/// Resuming from a checkpoint requires the same identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct SolveIdentity {
    problem: ProblemToken,
    algorithm: AlgorithmId,
    shape_token: u128,
    options_token: u128,
}

impl SolveIdentity {
    /// Creates an identity; `shape_token` and `options_token` are values you choose.
    pub const fn new(
        problem: ProblemToken,
        algorithm: AlgorithmId,
        shape_token: u128,
        options_token: u128,
    ) -> Self {
        Self {
            problem,
            algorithm,
            shape_token,
            options_token,
        }
    }

    /// Returns the problem token.
    pub const fn problem(self) -> ProblemToken {
        self.problem
    }

    /// Returns the algorithm identity.
    pub const fn algorithm(self) -> AlgorithmId {
        self.algorithm
    }

    /// Returns the shape token.
    pub const fn shape_token(self) -> u128 {
        self.shape_token
    }

    /// Returns the options token.
    pub const fn options_token(self) -> u128 {
        self.options_token
    }
}

/// Number of a solve epoch within one [`crate::SolveState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct SolveEpoch(pub(super) u64);

impl SolveEpoch {
    /// Returns the epoch number.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// How far the last accepted state can be trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum PartialResultTrust {
    /// Finite diagnostic state with no feasibility or residual claim.
    DiagnosticOnly,
    /// State accepted by the algorithm's own acceptance or feasibility rules.
    Admissible,
    /// State whose residual independently passed its configured threshold.
    ResidualChecked,
    /// State backed by an independently verified certificate.
    CertificateVerified,
}

impl fmt::Display for PartialResultTrust {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DiagnosticOnly => "diagnostic-only",
            Self::Admissible => "admissible",
            Self::ResidualChecked => "residual-checked",
            Self::CertificateVerified => "certificate-verified",
        })
    }
}

/// Epoch, revision and trust level of the last accepted state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct AcceptedStateMetadata {
    pub(super) epoch: SolveEpoch,
    pub(super) revision: u64,
    pub(super) trust: PartialResultTrust,
}

impl AcceptedStateMetadata {
    /// Returns the epoch of the accepted state.
    pub const fn epoch(self) -> SolveEpoch {
        self.epoch
    }

    /// Returns the revision, which increases with each accepted state in the epoch.
    pub const fn revision(self) -> u64 {
        self.revision
    }

    /// Returns the trust level of the accepted state.
    pub const fn trust(self) -> PartialResultTrust {
        self.trust
    }
}

/// Rule used to scale a raw residual norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum ResidualScalingRule {
    /// No scaling; the denominator is one.
    Unscaled,
    /// Division by an explicit reference magnitude.
    Reference,
    /// Division by an absolute/relative tolerance threshold.
    ToleranceThreshold,
    /// Componentwise normalization before norm reduction.
    ComponentwiseTolerance,
    /// Normwise backward-error denominator.
    BackwardError,
}

/// Residual scaling rule with its positive denominator.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ResidualScaling<S: RealScalar> {
    rule: ResidualScalingRule,
    denominator: S,
}

impl<S: RealScalar> ResidualScaling<S> {
    /// Creates a scaling from a rule and its denominator.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `denominator` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `denominator` is not positive.
    pub fn new(rule: ResidualScalingRule, denominator: S) -> Result<Self, SciOxideError> {
        if !denominator.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "residual denominator must be finite",
            ));
        }
        if denominator <= S::ZERO {
            return Err(SciOxideError::invalid_input(
                crate::argument!("denominator"),
                InvalidInputReason::OutOfRange,
                "residual denominator must be positive",
            ));
        }
        Ok(Self { rule, denominator })
    }

    /// Returns the unscaled rule with denominator one.
    pub fn unscaled() -> Self {
        Self {
            rule: ResidualScalingRule::Unscaled,
            denominator: S::ONE,
        }
    }

    /// Returns the scaling rule.
    pub const fn rule(self) -> ResidualScalingRule {
        self.rule
    }

    /// Returns the denominator.
    pub const fn denominator(self) -> S {
        self.denominator
    }
}

/// Raw and scaled residual norms, with the norm and scaling used.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ResidualReport<S: RealScalar> {
    raw_norm: S,
    scaled_norm: S,
    norm: NormSelection,
    scaling: ResidualScaling<S>,
}

impl<S: RealScalar> ResidualReport<S> {
    /// Creates a report; the scaled norm is `raw_norm / scaling.denominator()`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `raw_norm` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `raw_norm` is negative.
    /// - `NumericalInstability` (`Overflow`) if the scaled norm overflows.
    pub fn new(
        raw_norm: S,
        norm: NormSelection,
        scaling: ResidualScaling<S>,
    ) -> Result<Self, SciOxideError> {
        if !raw_norm.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "raw residual norm must be finite",
            ));
        }
        if raw_norm < S::ZERO {
            return Err(SciOxideError::invalid_input(
                crate::argument!("raw_norm"),
                InvalidInputReason::OutOfRange,
                "raw residual norm must be nonnegative",
            ));
        }
        let scaled_norm = checked_float_quotient(raw_norm, scaling.denominator)?;
        Ok(Self {
            raw_norm,
            scaled_norm,
            norm,
            scaling,
        })
    }

    /// Creates a report without scaling.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `raw_norm` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `raw_norm` is negative.
    pub fn unscaled(raw_norm: S, norm: NormSelection) -> Result<Self, SciOxideError> {
        Self::new(raw_norm, norm, ResidualScaling::unscaled())
    }

    /// Returns the raw residual norm in problem units.
    pub const fn raw_norm(self) -> S {
        self.raw_norm
    }

    /// Returns the dimensionless scaled residual norm.
    pub const fn scaled_norm(self) -> S {
        self.scaled_norm
    }

    /// Returns the norm applied to the residual vector.
    pub const fn norm(self) -> NormSelection {
        self.norm
    }

    /// Returns the scaling rule and denominator.
    pub const fn scaling(self) -> ResidualScaling<S> {
        self.scaling
    }
}

/// Reliability of an estimate that is not a proven bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum EstimateReliability {
    /// Heuristic indication without a calibrated coverage statement.
    Heuristic,
    /// Estimate justified by a documented numerical model.
    ModelBased,
    /// Estimate calibrated by a documented statistical procedure.
    StatisticallyCalibrated,
}

/// A finite estimate with its reliability.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ReliableEstimate<S: RealScalar> {
    value: S,
    reliability: EstimateReliability,
}

fn validate_estimate<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "estimate must be finite",
        ))
    }
}

impl<S: RealScalar> ReliableEstimate<S> {
    /// Creates an estimate with the given reliability.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    pub fn new(value: S, reliability: EstimateReliability) -> Result<Self, SciOxideError> {
        Ok(Self {
            value: validate_estimate(value)?,
            reliability,
        })
    }

    /// Returns the estimated value.
    pub const fn value(self) -> S {
        self.value
    }

    /// Returns the reliability.
    pub const fn reliability(self) -> EstimateReliability {
        self.reliability
    }
}

#[cfg(test)]
mod tests {
    use super::AlgorithmId;
    use alloc::string::ToString;

    #[test]
    fn composed_identities_equal_validated_construction() {
        let family = crate::algorithm_component!("special-function");
        let method = crate::algorithm_component!("series.v2");
        let composed = AlgorithmId::from_components(family, method, 3);
        assert_eq!(
            Ok(composed),
            AlgorithmId::new("special-function", "series.v2", 3)
        );
        assert_eq!(composed.to_string(), "special-function/series.v2@3");
        assert!(crate::AlgorithmComponent::new("bad method").is_err());
        assert!(crate::AlgorithmComponent::new("").is_err());
    }
}
