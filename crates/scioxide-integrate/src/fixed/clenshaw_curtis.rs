//! Clenshaw–Curtis quadrature on Chebyshev extrema.

use alloc::sync::Arc;
use alloc::vec::Vec;

use scioxide_core::backend::{
    checked_add, checked_mul, checked_product, try_filled, try_with_capacity,
};
use scioxide_core::{Interval, RealScalar, SciOxideError};

use super::{FixedRule, FixedRuleNode, interval_map};
use crate::kernel::PreparedRule;
use scioxide_core::PromotingCompensatedSum;

/// Clenshaw–Curtis quadrature with `n + 1` nodes on a finite interval.
///
/// The rule integrates exactly the polynomial that interpolates `f` at the
/// Chebyshev extrema
///
/// ```text
/// tₖ = cos(kπ/n),   k = 0, …, n,
/// ```
///
/// of the reference interval `[−1, 1]`, mapped affinely to `[a, b]`. It is exact
/// for polynomials of degree `n`, its weights are positive, and it includes
/// both endpoints. The nodes are listed from `b` to `a`. The rule has no
/// embedded error estimate. Apply it through [`FixedRule`].
///
/// Construction computes the weights from their explicit cosine series in
/// `f64` with compensated summation, in O(n²) operations, and rounds nodes and
/// weights to `S`. Clones share this storage.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{ClenshawCurtis, FixedRule};
///
/// // ∫₀^π sin x dx = 2
/// let rule = ClenshawCurtis::<f64>::new(16)?;
/// let (value, report) = rule.integrate(
///     |x: f64| Ok(x.sin()),
///     Interval::new(0.0, core::f64::consts::PI)?,
/// )?;
/// assert!((value - 2.0).abs() < 1e-12);
/// assert_eq!(report.evaluations(), 17);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - C. W. Clenshaw and A. R. Curtis, "A method for numerical integration on an
///   automatic computer", *Numer. Math.* 2, 197–205, 1960.
///   <https://doi.org/10.1007/BF01386223>
/// - J. Waldvogel, "Fast construction of the Fejér and Clenshaw–Curtis
///   quadrature rules", *BIT* 46, 195–202, 2006.
///   <https://doi.org/10.1007/s10543-006-0045-4>
#[derive(Debug, Clone)]
#[must_use]
pub struct ClenshawCurtis<S: RealScalar> {
    rule: Arc<PreparedRule<S>>,
}

impl<S: RealScalar> ClenshawCurtis<S> {
    /// Creates the rule with `n + 1` nodes.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (argument `n`) if `n == 0`.
    /// - `SizeOverflow` if `n²` or another index computation overflows `usize`.
    /// - `ScalarConversion` if an index is not exactly representable in `f64`.
    /// - `AllocationFailure` if the storage cannot be reserved.
    /// - `NumericalInstability` if a weight rounded to `S` is not positive and
    ///   finite.
    pub fn new(n: usize) -> Result<Self, SciOxideError> {
        if n == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("n"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Clenshaw-Curtis requires n >= 1",
            ));
        }
        let (nodes, weights) = canonical_rule(n)?;
        let convert = |values: Vec<f64>| -> Result<Vec<S>, SciOxideError> {
            let mut converted =
                try_with_capacity(values.len(), scioxide_core::quantity!("rule_coefficients"))?;
            for value in values {
                converted.push(S::try_from_f64_lossy(value)?);
            }
            Ok(converted)
        };
        Ok(Self {
            rule: Arc::new(PreparedRule::new(
                convert(nodes)?,
                convert(weights)?,
                scioxide_core::argument!("n"),
            )?),
        })
    }

    /// Returns `n`, the polynomial degree integrated exactly; the rule has
    /// `n + 1` nodes.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.rule.len() - 1
    }
}

/// Nodes and weights on `[-1, 1]` in `f64`.
fn canonical_rule(n: usize) -> Result<(Vec<f64>, Vec<f64>), SciOxideError> {
    let node_count = checked_add(n, 1, scioxide_core::quantity!("clenshaw_curtis_node_count"))?;
    let n_squared = checked_mul(
        n,
        n,
        scioxide_core::quantity!("clenshaw_curtis_endpoint_weight"),
    )?;
    let n_scalar = usize_to_f64(n)?;
    let n_squared_scalar = usize_to_f64(n_squared)?;

    // Nodes are the Chebyshev extrema cos(k pi / n), stored in decreasing order (the
    // fixed-rule validator accepts either monotone direction). Only half are computed
    // and mirrored so the rule is exactly symmetric, and the centre node is pinned to 0
    // because cos(pi/2) rounds to ~6e-17 rather than zero.
    let mut nodes = try_filled(
        node_count,
        0.0_f64,
        scioxide_core::quantity!("clenshaw_curtis_nodes"),
    )?;
    for k in 0..=n / 2 {
        let angle = usize_to_f64(k)? * core::f64::consts::PI / n_scalar;
        let node = <f64 as RealScalar>::cos(angle);
        nodes[k] = node;
        nodes[n - k] = -node;
    }
    if n.is_multiple_of(2) {
        nodes[n / 2] = 0.0;
    }

    // Closed-form weights (Waldvogel / Trefethen, "Is Gauss quadrature better than
    // Clenshaw-Curtis?"): w_0 = w_n = 1/(n^2 - 1) for even n and 1/n^2 for odd n, and
    //     w_k = (2/n) [1 - sum_j 2 cos(2 j theta_k) / (4 j^2 - 1)
    //                    - (n even) cos(n theta_k) / (n^2 - 1)]
    // with theta_k = k pi / n. The alternating cosine series is accumulated with
    // compensation because its terms partially cancel for large n.
    let endpoint_weight = if n.is_multiple_of(2) {
        1.0 / (n_squared_scalar - 1.0)
    } else {
        1.0 / n_squared_scalar
    };
    let mut weights = try_filled(
        node_count,
        0.0_f64,
        scioxide_core::quantity!("clenshaw_curtis_weights"),
    )?;
    weights[0] = endpoint_weight;
    weights[n] = endpoint_weight;

    let upper = if n.is_multiple_of(2) {
        n / 2 - 1
    } else {
        (n - 1) / 2
    };
    for k in 1..=n / 2 {
        let theta = usize_to_f64(k)? * core::f64::consts::PI / n_scalar;
        let mut coefficient = PromotingCompensatedSum::ZERO;
        coefficient.add_value(1.0)?;
        for j in 1..=upper {
            let four_j_squared = checked_product(
                &[4, j, j],
                scioxide_core::quantity!("clenshaw_curtis_weight_index"),
            )?;
            let denominator = usize_to_f64(four_j_squared)? - 1.0;
            let angle = 2.0 * usize_to_f64(j)? * theta;
            coefficient.add_value(-2.0 * <f64 as RealScalar>::cos(angle) / denominator)?;
        }
        if n.is_multiple_of(2) {
            coefficient.add_value(
                -<f64 as RealScalar>::cos(n_scalar * theta) / (n_squared_scalar - 1.0),
            )?;
        }
        let weight = 2.0 * coefficient.round()? / n_scalar;
        weights[k] = weight;
        weights[n - k] = weight;
    }
    Ok((nodes, weights))
}

fn usize_to_f64(value: usize) -> Result<f64, SciOxideError> {
    <f64 as RealScalar>::try_from_usize_exact(value)
}

impl<S: RealScalar> FixedRule<S> for ClenshawCurtis<S> {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "ClenshawCurtis", 1)
    }

    fn node_count(&self) -> usize {
        self.rule.len()
    }

    fn map_nodes_into(
        &self,
        interval: Interval<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<(), SciOxideError> {
        self.rule
            .map_into(interval_map(interval)?, true, destination)
            .map(|_| ())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn zero_degree_is_rejected() {
        assert!(matches!(
            ClenshawCurtis::<f64>::new(0).err().unwrap().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn canonical_weights_are_symmetric_and_integrate_constants() {
        for n in [1_usize, 2, 7, 16] {
            let rule = ClenshawCurtis::<f64>::new(n).unwrap();
            assert_eq!(rule.degree(), n);
            let weights = rule.rule.weights();
            let total: f64 = weights.iter().sum();
            assert!((total - 2.0).abs() < 1e-14, "n = {n}: {total}");
            assert!(
                weights
                    .iter()
                    .zip(weights.iter().rev())
                    .all(|(a, b)| a == b)
            );
        }
    }

    #[test]
    fn polynomial_exactness_in_f64_and_f32() {
        let rule = ClenshawCurtis::<f64>::new(4).unwrap();
        let (value, _) = rule
            .integrate(|x: f64| Ok(x.powi(4)), Interval::new(-1.0, 1.0).unwrap())
            .unwrap();
        assert!((value - 0.4).abs() < 1e-14, "got {value}");
        let rule = ClenshawCurtis::<f32>::new(4).unwrap();
        let (value, _) = rule
            .integrate(|x: f32| Ok(x), Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        assert!((value - 0.5).abs() < 1e-6, "got {value}");
    }
}
