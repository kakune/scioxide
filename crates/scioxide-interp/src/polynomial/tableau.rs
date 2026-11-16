//! Neville-type tableau used by Neville polynomial and Bulirsch–Stoer rational
//! interpolation.
//!
//! Both build corrections `C_{m,i}` and `D_{m,i}` in place and walk a centred
//! path from the node nearest the query (Stoer and Bulirsch, §2.1–2.2); only
//! the per-entry update differs.

use scioxide_core::{RealScalar, SciOxideError};

use crate::checked::{intermediate, output};
use crate::protocol::NodeSeparation;

/// Evaluate one component's tableau at `t`, returning the value and the last
/// path correction (zero at a node).
///
/// `c` and `d` hold the samples on entry and are overwritten. A query that
/// coincides with a node under `separation` returns that sample verbatim.
/// `step(c, d, index, order)` replaces `c[index]`, `d[index]` by order `order`
/// from the order-`order - 1` entries `c[index + 1]` and `d[index]`.
pub(super) fn tableau<S: RealScalar>(
    x: &[S],
    t: S,
    separation: NodeSeparation<S>,
    c: &mut [S],
    d: &mut [S],
    mut step: impl FnMut(&mut [S], &mut [S], usize, usize) -> Result<(), SciOxideError>,
) -> Result<(S, S), SciOxideError> {
    // A node hit returns the sample exactly: the tableau would reproduce it only
    // up to rounding, and its ratios divide by vanishing `t - x_i`.
    for (index, &node) in x.iter().enumerate() {
        if separation.coincides(t, node)? {
            return Ok((output(c[index])?, S::ZERO));
        }
    }
    // Starting at the nearest node keeps the accumulated corrections small. Any
    // start is exact in exact arithmetic, so the distance only ranks nodes.
    let closest = nearest_node(x, t);
    let mut result = c[closest];
    let mut correction = S::ZERO;
    // Nodes to the left of the current correction path.
    let mut left = closest;
    for order in 1..x.len() {
        for index in 0..x.len() - order {
            step(c, d, index, order)?;
        }
        // Centred path: go right (`c`) while fewer than half of the remaining
        // entries lie left of the path (`2 left < remaining`), else go left (`d`).
        let remaining = x.len() - order;
        correction = if left < remaining - left.min(remaining) {
            c[left]
        } else {
            left -= 1;
            d[left]
        };
        result = intermediate(result + correction)?;
    }
    Ok((output(result)?, correction))
}

/// Only the two bracketing nodes can be nearest. At most one of their
/// distances can overflow: the node on the query's side of zero has a finite
/// distance. Thus ordinary differences rank them even across the full range.
fn nearest_node<S: RealScalar>(nodes: &[S], query: S) -> usize {
    let right = nodes.partition_point(|&node| node < query);
    if right == 0 {
        0
    } else if right == nodes.len() || query - nodes[right - 1] <= nodes[right] - query {
        right - 1
    } else {
        right
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Neville's polynomial update, the simplest admissible step.
    fn neville(x: &[f64], t: f64, samples: &[f64]) -> (f64, f64) {
        let (mut c, mut d) = (samples.to_vec(), samples.to_vec());
        tableau(x, t, NodeSeparation::Exact, &mut c, &mut d, |c, d, i, m| {
            let w = c[i + 1] - d[i];
            let h = x[i] - x[i + m];
            c[i] = (x[i] - t) * w / h;
            d[i] = (x[i + m] - t) * w / h;
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn scaffold_reproduces_polynomials_nodes_and_errors() {
        let x = [-1.0, 0.0, 1.0, 2.0];
        let samples = x.map(|v: f64| v * v * v - v);
        for t in [-1.5, -0.25, 0.5, 1.75, 3.0] {
            let (value, correction) = neville(&x, t, &samples);
            assert!((value - (t * t * t - t)).abs() <= 1e-13);
            assert!(correction.is_finite());
        }
        assert_eq!(neville(&x, 1.0, &samples), (0.0, 0.0));
        assert_eq!(neville(&[2.0], 5.0, &[7.0]), (7.0, 0.0));
        let tolerant = NodeSeparation::Tolerant(scioxide_core::Tolerance::new(1e-6, 0.0).unwrap());
        let (mut c, mut d) = (samples.to_vec(), samples.to_vec());
        let hit = tableau(
            &x,
            2.0 + 1e-9,
            tolerant,
            &mut c,
            &mut d,
            |_, _, _, _| unreachable!(),
        );
        assert_eq!(hit.unwrap(), (samples[3], 0.0));
        let (mut c, mut d) = (samples.to_vec(), samples.to_vec());
        let failure = tableau(
            &x,
            0.5,
            NodeSeparation::Exact,
            &mut c,
            &mut d,
            |_, _, _, _| {
                Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "pole",
                ))
            },
        );
        assert!(failure.is_err());
        assert_eq!(nearest_node(&[-f64::MAX, f64::MAX], 0.0), 0);
        assert_eq!(nearest_node(&[-f64::MAX, f64::MAX], f64::MAX / 2.0), 1);
        assert_eq!(nearest_node(&[-f64::MAX, f64::MAX], -f64::MAX / 2.0), 0);
        assert_eq!(nearest_node(&[1.0, 2.0], -f64::MAX), 0);
        assert_eq!(nearest_node(&[1.0, 2.0], f64::MAX), 1);
    }
}
