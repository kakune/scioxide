//! Range checks for classical orthogonal three-term recurrences.
//!
//! The crate-level range-check module defines arithmetic tiers, point checks
//! and coefficient views. This module adds the recurrence- and family-specific
//! bounds that let a blocked tier skip intermediate checks. Prepared recurrence
//! data is classified once, and family bounds follow from the family
//! parameters and the requested degree.

use alloc::vec::Vec;
use core::ops::Deref;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

use crate::OrthogonalFamily;
use crate::admission::{CheckpointTier, CoefficientView, moderate, native_range};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClassicalRecurrence {
    ChebyshevT,
    ChebyshevU,
    Other,
}

impl ClassicalRecurrence {
    /// Classify normalized recurrence terms `(a_n, b_n, c_n)`. Chebyshev T and
    /// U share every term `(2, 0, 1)` after the first, which is `(1, 0, c)` or
    /// `(2, 0, c)` with `c` zero or one.
    fn of<R: RealScalar>(values: &[(R, R, R)]) -> Self {
        let tail = values
            .iter()
            .skip(1)
            .all(|&(a, b, c)| a == R::TWO && b == R::ZERO && c == R::ONE);
        match values.first() {
            Some(&(a, b, c)) if tail && b == R::ZERO && (c == R::ZERO || c == R::ONE) => {
                if a == R::ONE {
                    Self::ChebyshevT
                } else if a == R::TWO {
                    Self::ChebyshevU
                } else {
                    Self::Other
                }
            }
            _ => Self::Other,
        }
    }
}

/// Recurrence terms with their range classification. Callers can borrow the
/// terms but cannot mutate them after classification.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RecurrenceData<R: RealScalar> {
    values: Vec<(R, R, R)>,
    moderate: bool,
    classical: ClassicalRecurrence,
}

impl<R: RealScalar> RecurrenceData<R> {
    pub(crate) fn new(values: Vec<(R, R, R)>) -> Self {
        let moderate = values
            .iter()
            .all(|&(a, b, c)| moderate(a) & moderate(b) & moderate(c));
        let classical = ClassicalRecurrence::of(&values);
        Self {
            values,
            moderate,
            classical,
        }
    }

    /// Keeps the first `count` terms, appending `term(n)` beyond the prepared
    /// length, and classify the result as [`Self::new`] would.
    pub(crate) fn resized(
        &self,
        count: usize,
        term: impl Fn(usize) -> Result<(R, R, R), SciOxideError>,
    ) -> Result<Self, SciOxideError> {
        let prefix = count.min(self.values.len());
        let mut values = scioxide_core::backend::try_with_capacity(
            count,
            scioxide_core::quantity!("recurrence_terms"),
        )?;
        values.extend_from_slice(&self.values[..prefix]);
        for n in prefix..count {
            values.push(term(n)?);
        }
        Ok(Self::new(values))
    }

    pub(crate) fn series<'a, C: NumericScalar<Real = R>>(
        &'a self,
        coefficients: CoefficientView<'a, C>,
    ) -> SeriesAdmission<'a, C> {
        SeriesAdmission {
            recurrence: self,
            coefficients,
        }
    }

    pub(crate) fn classical(&self) -> ClassicalRecurrence {
        self.classical
    }
}

impl<R: RealScalar> Deref for RecurrenceData<R> {
    type Target = [(R, R, R)];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// A coefficient series paired with the recurrence it is evaluated through.
#[derive(Clone, Copy)]
pub(crate) struct SeriesAdmission<'a, C: NumericScalar> {
    recurrence: &'a RecurrenceData<C::Real>,
    coefficients: CoefficientView<'a, C>,
}

impl<'a, C: NumericScalar> SeriesAdmission<'a, C> {
    pub(crate) fn coefficients(self) -> &'a [C] {
        self.coefficients.values()
    }

    pub(crate) fn recurrence(self) -> &'a RecurrenceData<C::Real> {
        self.recurrence
    }

    /// Moderate recurrence terms and bounded coefficients allow narrow blocks.
    /// Lane-paired classical Chebyshev T terms are integer factors.
    #[inline(always)]
    pub(crate) fn tier(self, points: impl IntoIterator<Item = C>, lanes: bool) -> CheckpointTier {
        CheckpointTier::blocked(
            self.recurrence.moderate && self.coefficients.bounded(),
            lanes && self.recurrence.classical == ClassicalRecurrence::ChebyshevT,
            points,
        )
    }
}

/// The scalar-family bound follows directly from the family and requested degree.
/// At degree <= 2^14, unparameterized coefficients are zero or in [2^-16,2^16].
/// Positive Gegenbauer lambda in [2^-16,2^14] has the same bound. Chebyshev and
/// Hermite terms are integer factors.
#[inline(always)]
pub(crate) fn family_tier<R: RealScalar>(
    family: OrthogonalFamily<R>,
    degree: usize,
    point: R,
) -> CheckpointTier {
    const MAX_CLASSICAL_DEGREE: usize = 1 << 14;
    let bounded = degree <= MAX_CLASSICAL_DEGREE
        && match family {
            OrthogonalFamily::ChebyshevT
            | OrthogonalFamily::ChebyshevU
            | OrthogonalFamily::Legendre
            | OrthogonalFamily::HermiteH
            | OrthogonalFamily::HermiteHe
            | OrthogonalFamily::Laguerre => true,
            OrthogonalFamily::Gegenbauer { lambda } => {
                (1.0 / 65536.0..=16384.0).contains(&lambda.into_f64())
            }
            OrthogonalFamily::GeneralizedLaguerre { alpha } => {
                (0.0..=16384.0).contains(&alpha.into_f64()) && moderate(alpha)
            }
            OrthogonalFamily::Jacobi { alpha, beta } => {
                // a >= 1 and c >= 2^-16 (after the special first step).
                // With 0 <= alpha,beta <= 16, |b| <= 16. Nonzero sum and
                // difference >= 2^-4 imply |b| >= 2^-44 through degree 2^14.
                // Its quantum >= 2^-96 is above the a*x quantum >= 2^-120;
                // the existing conservative 2^-136 factor bound still holds.
                let alpha = alpha.into_f64();
                let beta = beta.into_f64();
                let sum = alpha + beta;
                let difference = (alpha - beta).abs();
                (0.0..=16.0).contains(&alpha)
                    && (0.0..=16.0).contains(&beta)
                    && (sum == 0.0 || sum >= 0.0625)
                    && (difference == 0.0 || difference >= 0.0625)
            }
        };
    let integer = matches!(
        family,
        OrthogonalFamily::ChebyshevT
            | OrthogonalFamily::ChebyshevU
            | OrthogonalFamily::HermiteH
            | OrthogonalFamily::HermiteHe
    );
    CheckpointTier::blocked(bounded, integer, [point])
}

/// Four-product coefficient formulas need another square root of headroom.
#[inline]
pub(crate) fn four_product_values<R: RealScalar>(values: impl IntoIterator<Item = R>) -> bool {
    let upper = R::MAX.sqrt().sqrt() / (R::TWO + R::TWO);
    values
        .into_iter()
        .all(|value| native_range(value) && value.abs() <= upper)
}

#[cfg(test)]
mod tests {
    use super::{ClassicalRecurrence, RecurrenceData, family_tier};
    use crate::OrthogonalFamily as F;
    use crate::admission::{CheckpointTier, CoefficientView, Coefficients};

    #[test]
    fn family_certificates_cover_parameters_and_reject_unproved_cases() {
        for family in [
            F::GeneralizedLaguerre { alpha: 0.5 },
            F::Jacobi {
                alpha: 0.5,
                beta: 1.5,
            },
        ] {
            assert_ne!(family_tier(family, 50, 0.5).steps(), 0);
            assert_eq!(family_tier(family, 16385, 0.5).steps(), 0);
        }
        for family in [
            F::GeneralizedLaguerre { alpha: -0.5 },
            F::Jacobi {
                alpha: 0.5,
                beta: 0.5 + 2.0_f64.powi(-20),
            },
            F::Jacobi {
                alpha: 32.0,
                beta: 0.5,
            },
        ] {
            assert_eq!(family_tier(family, 50, 0.5).steps(), 0);
        }
    }

    fn recurrence(family: F<f64>) -> RecurrenceData<f64> {
        RecurrenceData::new((0..16).map(|n| family.recurrence(n).unwrap()).collect())
    }

    #[test]
    fn tiers_follow_the_bound_data_and_classical_recurrence() {
        let chebyshev = recurrence(F::ChebyshevT);
        let legendre = recurrence(F::Legendre);
        let mut prepared = chebyshev.to_vec();
        prepared[0].2 = 0.0;
        let prepared = RecurrenceData::new(prepared);
        assert_eq!(prepared.classical(), ClassicalRecurrence::ChebyshevT);

        let coefficients = Coefficients::new(vec![0.25_f64; 16]);
        assert_eq!(
            chebyshev
                .series(coefficients.view())
                .tier([0.5], true)
                .steps(),
            12
        );
        assert_eq!(
            chebyshev
                .series(coefficients.view())
                .tier([0.5], false)
                .steps(),
            6
        );
        assert_eq!(
            legendre
                .series(coefficients.view())
                .tier([0.5], true)
                .steps(),
            6
        );
        for bad in [2.0_f64.powi(-65), 131072.0, f64::from_bits(1)] {
            let mut values = vec![0.25; 16];
            values[7] = bad;
            let owned = Coefficients::new(values.clone());
            for view in [owned.view(), CoefficientView::new(&values)] {
                assert_eq!(
                    chebyshev.series(view).tier([0.5], true),
                    CheckpointTier::CHECKED
                );
            }
        }
        let mut terms = chebyshev.to_vec();
        terms[7].0 = 131072.0;
        let inadmissible = RecurrenceData::new(terms);
        assert_eq!(
            inadmissible.series(coefficients.view()).tier([0.5], true),
            CheckpointTier::CHECKED
        );
        assert_eq!(
            chebyshev
                .series(coefficients.view())
                .tier([0.5, 131072.0], true),
            CheckpointTier::CHECKED
        );
    }

    #[test]
    fn scalar_family_proof_enforces_degree_parameter_and_width_bounds() {
        assert_eq!(family_tier(F::HermiteH, 16384, 0.5_f64).steps(), 12);
        assert_eq!(family_tier(F::HermiteH, 16385, 0.5_f64).steps(), 0);
        assert_eq!(family_tier(F::ChebyshevT, 16, 0.5_f32).steps(), 0);
        for lambda in [1.0 / 65536.0, 16384.0] {
            assert_eq!(
                family_tier(F::Gegenbauer { lambda }, 16384, 0.5_f64).steps(),
                6
            );
        }
        for lambda in [-0.25, 1.0 / 131072.0, 32768.0] {
            assert_eq!(
                family_tier(F::Gegenbauer { lambda }, 16, 0.5_f64).steps(),
                0
            );
        }
    }
    #[test]
    fn extending_a_classified_prefix_checks_the_new_terms() {
        let chebyshev = recurrence(F::ChebyshevT);
        let coefficients = Coefficients::new(vec![0.25_f64; 18]);
        let enlarged = chebyshev.resized(18, |_| Ok((131072.0, 0.0, 1.0))).unwrap();
        assert_eq!(
            enlarged.series(coefficients.view()).tier([0.5], true),
            CheckpointTier::CHECKED
        );
        let recovered = enlarged.resized(8, |_| unreachable!()).unwrap();
        let short = Coefficients::new(vec![0.25_f64; 8]);
        assert_eq!(recovered.series(short.view()).tier([0.5], true).steps(), 12);
        let original = chebyshev.resized(8, |_| unreachable!()).unwrap();
        assert_eq!(recovered, original);
    }
}
