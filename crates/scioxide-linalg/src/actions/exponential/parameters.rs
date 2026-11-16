//! Al-Mohy--Higham (2011), Code Fragment 3.1, without forming matrix powers.
use scioxide_core::{RealScalar, SciOxideError};

// Double-precision theta_m: Higham, Computing Matrix Functions, Table A.3
// (m <= 30), and Al-Mohy--Higham, SISC 33 (2011), Table 3.1 (m > 30).
const THETA: &[(usize, f64)] = &[
    (1, 2.29e-16),
    (2, 2.58e-8),
    (3, 1.39e-5),
    (4, 3.40e-4),
    (5, 2.40e-3),
    (6, 9.07e-3),
    (7, 2.38e-2),
    (8, 5.00e-2),
    (9, 8.96e-2),
    (10, 1.44e-1),
    (11, 2.14e-1),
    (12, 3.00e-1),
    (13, 4.00e-1),
    (14, 5.14e-1),
    (15, 6.41e-1),
    (16, 7.81e-1),
    (17, 9.31e-1),
    (18, 1.09),
    (19, 1.26),
    (20, 1.44),
    (21, 1.62),
    (22, 1.82),
    (23, 2.01),
    (24, 2.22),
    (25, 2.43),
    (26, 2.64),
    (27, 2.86),
    (28, 3.08),
    (29, 3.31),
    (30, 3.54),
    (35, 4.7),
    (40, 6.0),
    (45, 7.2),
    (50, 8.5),
    (55, 9.9),
];
// Single-precision entries from Table 3.1. Other degrees conservatively use
// the double table; rounding the tabulated constants down protects the bound.
const SINGLE: &[(usize, f64)] = &[
    (5, 0.13),
    (10, 1.0),
    (15, 2.2),
    (20, 3.6),
    (25, 4.9),
    (30, 6.3),
    (35, 7.7),
    (40, 9.1),
    (45, 11.0),
    (50, 12.0),
    (55, 13.0),
];

#[derive(Clone, Copy)]
pub(super) struct Parameters {
    pub(super) degree: usize,
    pub(super) steps: usize,
}

fn theta<R: RealScalar>(degree: usize, tolerance: R) -> Result<R, SciOxideError> {
    // Unlisted degrees use the preceding entry, since theta_m is monotone.
    // This also allows every user term cap and phi index through 55.
    let double = THETA
        .iter()
        .rev()
        .find(|&&(m, _)| m <= degree)
        .map_or(0.0, |&(_, t)| t);
    let value = if R::EPSILON >= R::try_from_f64_lossy(f64::from(f32::EPSILON))? {
        SINGLE
            .iter()
            .rev()
            .find(|&&(m, _)| m <= degree)
            .map_or(double, |&(_, t)| t)
    } else {
        double
    };
    let unit = R::EPSILON * R::HALF;
    // h_(m+1)(theta)/theta starts at degree m with nonnegative
    // coefficients. Shrinking theta by r^(1/m) reduces the majorant by
    // at least r, preserving a requested tolerance tighter than roundoff.
    let adjustment = if tolerance > R::ZERO && tolerance < unit {
        ((tolerance / unit).ln() / crate::checked::rounded_count::<R>(degree)?).exp()
    } else {
        R::ONE
    };
    Ok(R::try_from_f64_lossy(value)?.next_after(R::ZERO) * adjustment)
}

/// None means the caller's work allowance stopped a power estimate. The
/// caller handles an insufficient degree cap before entering this routine.
pub(super) fn select<R: RealScalar>(
    bound: R,
    terms: usize,
    minimum: usize,
    columns: usize,
    tolerance: R,
    mut power_norm: impl FnMut(usize) -> Result<Option<R>, SciOxideError>,
) -> Result<Option<Parameters>, SciOxideError> {
    let cap = terms;
    let mut best = Parameters {
        degree: 0,
        steps: 1,
    };
    let mut cost = usize::MAX;
    let mut consider = |degree: usize, theta: R, alpha: R| -> Result<(), SciOxideError> {
        if degree < minimum || degree > cap {
            return Ok(());
        }
        let scaled = (alpha / theta).ceil().max(R::ONE);
        // Degrees with unrepresentable scaling cannot beat an admitted one.
        if let Ok(steps) = scaled.try_to_usize_exact()
            && let Some(candidate) = degree.checked_mul(steps)
            && candidate < cost
        {
            best = Parameters { degree, steps };
            cost = candidate;
        }
        Ok(())
    };
    let max_degree = cap;
    let pmax = (2..=8)
        .filter(|p| p * (p - 1) <= max_degree + 1)
        .max()
        .unwrap_or(1);
    // Equation (3.13), ell=1, matching the shared block-free estimator.
    let estimate_cost = crate::checked::rounded_count::<R>(2 * pmax * (pmax + 3))?
        * theta::<R>(max_degree, tolerance)?
        / (crate::checked::rounded_count::<R>(columns.max(1))?
            * crate::checked::rounded_count::<R>(max_degree)?);
    if bound <= estimate_cost || pmax < 2 {
        for m in 1..=cap {
            consider(m, theta::<R>(m, tolerance)?, bound)?;
        }
    } else {
        let Some(mut previous) = power_norm(2)? else {
            return Ok(None);
        };
        for p in 2..=pmax {
            let Some(next) = power_norm(p + 1)? else {
                return Ok(None);
            };
            let alpha = previous.max(next).min(bound);
            for m in 1..=cap {
                if m + 1 >= p * (p - 1) {
                    consider(m, theta::<R>(m, tolerance)?, alpha)?;
                }
            }
            previous = next;
        }
    }
    Ok(Some(best))
}
