//! Choice of derivative source for solvers.

use crate::DifferenceScheme;
use scioxide_core::{InvalidInputReason, SciOxideError};
/// Names a derivative source that [`DerivativeStrategy::select`] asks the
/// provider for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum DerivativeCandidate {
    /// A derivative you supply in closed form.
    Analytic,
    /// A derivative computed by automatic differentiation.
    AutomaticDifferentiation,
    /// A finite-difference approximation with the given scheme.
    FiniteDifference(DifferenceScheme),
}

/// Selects which derivative sources a solver tries, and in which order.
///
/// A strategy with several sources tries them in the order of its name. It
/// falls back to the next source only when the provider returns `Ok(None)`;
/// an error from the provider is returned at once.
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::{DerivativeCandidate, DerivativeStrategy};
///
/// // An analytic derivative is available, so finite differences are not tried.
/// let (candidate, value) = DerivativeStrategy::AnalyticOrForward.select(|candidate| {
///     Ok(match candidate {
///         DerivativeCandidate::Analytic => Some(4.0_f64),
///         _ => None,
///     })
/// })?;
/// assert_eq!(candidate, DerivativeCandidate::Analytic);
/// assert!((value - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum DerivativeStrategy {
    /// Uses only an analytic derivative.
    Analytic,
    /// Uses only automatic differentiation.
    Automatic,
    /// Uses only forward differences.
    Forward,
    /// Uses only central differences.
    Central,
    /// Tries an analytic derivative, then forward differences. This is the
    /// default.
    #[default]
    AnalyticOrForward,
    /// Tries an analytic derivative, then central differences.
    AnalyticOrCentral,
    /// Tries an analytic derivative, then automatic differentiation, then
    /// forward differences.
    AnalyticOrAutomaticOrForward,
}

impl DerivativeStrategy {
    #[inline]
    fn candidates(self) -> impl Iterator<Item = DerivativeCandidate> {
        let sources = match self {
            Self::Analytic => [Some(DerivativeCandidate::Analytic), None, None],
            Self::Automatic => [
                Some(DerivativeCandidate::AutomaticDifferentiation),
                None,
                None,
            ],
            Self::Forward => [
                Some(DerivativeCandidate::FiniteDifference(
                    DifferenceScheme::Forward,
                )),
                None,
                None,
            ],
            Self::Central => [
                Some(DerivativeCandidate::FiniteDifference(
                    DifferenceScheme::Central,
                )),
                None,
                None,
            ],
            Self::AnalyticOrForward => [
                Some(DerivativeCandidate::Analytic),
                Some(DerivativeCandidate::FiniteDifference(
                    DifferenceScheme::Forward,
                )),
                None,
            ],
            Self::AnalyticOrCentral => [
                Some(DerivativeCandidate::Analytic),
                Some(DerivativeCandidate::FiniteDifference(
                    DifferenceScheme::Central,
                )),
                None,
            ],
            Self::AnalyticOrAutomaticOrForward => [
                Some(DerivativeCandidate::Analytic),
                Some(DerivativeCandidate::AutomaticDifferentiation),
                Some(DerivativeCandidate::FiniteDifference(
                    DifferenceScheme::Forward,
                )),
            ],
        };
        sources.into_iter().flatten()
    }
    /// Returns the first candidate the provider supplies, with its result.
    ///
    /// `provider` is called with each candidate in order. It returns
    /// `Ok(Some(result))` to supply a result and `Ok(None)` if that source is
    /// unavailable. The result is returned unchanged: `select` makes no
    /// evaluations, counts nothing and writes no output of its own. At most
    /// three candidates are tried.
    ///
    /// # Errors
    ///
    /// - Errors returned by `provider`, at once and with their kind unchanged.
    /// - `InvalidInput` ([`InvalidInputReason::UnsupportedConfiguration`]) if
    ///   no candidate supplies a result.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::{DerivativeCandidate, DerivativeStrategy, DifferenceScheme};
    ///
    /// // No analytic derivative, so the strategy falls back to forward differences.
    /// let (candidate, derivative) =
    ///     DerivativeStrategy::AnalyticOrForward.select(|candidate| match candidate {
    ///         DerivativeCandidate::FiniteDifference(_) => Ok(Some(2.0_f64)),
    ///         _ => Ok(None),
    ///     })?;
    /// let forward = DerivativeCandidate::FiniteDifference(DifferenceScheme::Forward);
    /// assert_eq!(candidate, forward);
    /// assert!((derivative - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline]
    pub fn select<T, F>(self, mut provider: F) -> Result<(DerivativeCandidate, T), SciOxideError>
    where
        F: FnMut(DerivativeCandidate) -> Result<Option<T>, SciOxideError>,
    {
        for candidate in self.candidates() {
            if let Some(result) = provider(candidate)? {
                return Ok((candidate, result));
            }
        }
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("strategy"),
            InvalidInputReason::UnsupportedConfiguration,
            "none of the configured derivative sources is available",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    #[test]
    fn strategies_preserve_provider_results_order_errors_and_exhaustion() {
        use DerivativeCandidate::{Analytic, AutomaticDifferentiation, FiniteDifference};
        let cases: &[(DerivativeStrategy, &[DerivativeCandidate])] = &[
            (DerivativeStrategy::Analytic, &[Analytic]),
            (DerivativeStrategy::Automatic, &[AutomaticDifferentiation]),
            (
                DerivativeStrategy::Forward,
                &[FiniteDifference(DifferenceScheme::Forward)],
            ),
            (
                DerivativeStrategy::Central,
                &[FiniteDifference(DifferenceScheme::Central)],
            ),
            (
                DerivativeStrategy::AnalyticOrForward,
                &[Analytic, FiniteDifference(DifferenceScheme::Forward)],
            ),
            (
                DerivativeStrategy::AnalyticOrAutomaticOrForward,
                &[
                    Analytic,
                    AutomaticDifferentiation,
                    FiniteDifference(DifferenceScheme::Forward),
                ],
            ),
        ];
        for &(strategy, expected) in cases {
            for &chosen in expected {
                let mut attempts = Vec::new();
                // Selection must return the provider's owned result without
                // replacing its metadata or copying its storage.
                let mut payload = Some(vec![37_u8, 42]);
                let storage = payload.as_ref().unwrap().as_ptr();
                let (candidate, result) = strategy
                    .select(|candidate| {
                        attempts.push(candidate);
                        Ok(if candidate == chosen {
                            payload.take()
                        } else {
                            None
                        })
                    })
                    .unwrap();
                assert_eq!(candidate, chosen);
                assert_eq!(result, [37, 42]);
                assert_eq!(result.as_ptr(), storage);
                let index = expected
                    .iter()
                    .position(|&candidate| candidate == chosen)
                    .unwrap();
                assert_eq!(attempts, expected[..=index]);

                attempts.clear();
                let error = strategy
                    .select::<u8, _>(|candidate| {
                        attempts.push(candidate);
                        if candidate == chosen {
                            Err(SciOxideError::domain_error(
                                scioxide_core::argument!("x"),
                                scioxide_core::DomainReason::OutsideDomain,
                                "provider failure",
                            ))
                        } else {
                            Ok(None)
                        }
                    })
                    .unwrap_err();
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::DomainError {
                        reason: scioxide_core::DomainReason::OutsideDomain,
                        ..
                    }
                ));
                assert_eq!(attempts, expected[..=index]);
            }
            let mut attempts = Vec::new();
            let error = strategy
                .select::<u8, _>(|candidate| {
                    attempts.push(candidate);
                    Ok(None)
                })
                .unwrap_err();
            assert_eq!(attempts, expected);
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ));
        }
        assert_eq!(
            DerivativeStrategy::default(),
            DerivativeStrategy::AnalyticOrForward
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod central_selection_tests {
    use super::*;
    #[test]
    fn analytic_or_central_selects_in_order() {
        let mut seen = alloc::vec::Vec::new();
        let (source, value) = DerivativeStrategy::AnalyticOrCentral
            .select(|candidate| {
                seen.push(candidate);
                Ok(Some(4.0_f64))
            })
            .unwrap();
        assert_eq!(source, DerivativeCandidate::Analytic);
        assert_eq!(seen, [DerivativeCandidate::Analytic]);
        assert!((value - 4.0).abs() < 1e-14);
        seen.clear();
        let (source, value) = DerivativeStrategy::AnalyticOrCentral
            .select(|candidate| {
                seen.push(candidate);
                Ok(match candidate {
                    DerivativeCandidate::Analytic => None,
                    _ => Some(4.0_f64),
                })
            })
            .unwrap();
        let central = DerivativeCandidate::FiniteDifference(DifferenceScheme::Central);
        assert_eq!(source, central);
        assert_eq!(seen, [DerivativeCandidate::Analytic, central]);
        assert!((value - 4.0).abs() < 1e-14);
    }
}
