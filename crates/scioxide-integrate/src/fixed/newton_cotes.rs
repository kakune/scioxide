//! Composite Newton–Cotes rules with weights computed in exact integer arithmetic.

use scioxide_core::{
    AffineValue, AffineValueMut, Budget, Interval, InvalidInputReason, RealScalar, Scaled,
    ScaledAccumulator, SciOxideError,
};

use super::{FixedRule, FixedRuleNode, degenerate_interval};
use crate::geometry::IntervalGeometry;

/// Largest supported stencil; larger ones have large weights of alternating sign.
const MAX_NODES: usize = 11;

/// Selects whether a Newton–Cotes stencil includes the endpoints of its panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewtonCotesEndpoints {
    /// Closed rule: `n` nodes at `0, …, n − 1` on the panel `[0, n − 1]`.
    /// Adjacent panels share their common endpoint node.
    Closed,
    /// Open rule: `n` nodes at `1, …, n` on the panel `[0, n + 1]`. The panel
    /// endpoints are not sampled.
    Open,
}

/// Composite Newton–Cotes quadrature on equally spaced nodes.
///
/// The interval `[a, b]` is split into `panels` equal panels. On each panel the
/// rule integrates the polynomial that interpolates `f` at `n` equally spaced
/// nodes (the stencil), so it is exact for polynomials of degree
/// [`Self::exactness_degree`]. The grid spacing is `h = (b − a)/G`, where `G`
/// is `panels·(n − 1)` for a closed and `panels·(n + 1)` for an open stencil.
/// A closed rule calls `f` `panels·(n − 1) + 1` times, because adjacent panels
/// share an endpoint, and an open rule `panels·n` times.
///
/// | Stencil | Rule |
/// |---|---|
/// | 1 node, open | midpoint |
/// | 2 nodes, closed | trapezoid |
/// | 3 nodes, closed | Simpson |
/// | 4 nodes, closed | Simpson 3/8 |
/// | 5 nodes, closed | Boole |
///
/// Stencils have 1–11 open or 2–11 closed nodes. Larger stencils are not
/// offered: their weights have alternating signs and grow large, which
/// amplifies errors in the samples.
///
/// The weights are computed as exact rational numbers, and each mapped weight
/// is rounded once. The nodes are placed from exact grid fractions, so the
/// endpoints are reproduced exactly and rounding does not accumulate across
/// panels. The rule has no embedded error estimate;
/// [`Self::error_coefficient`] gives the leading term of the truncation error.
/// Apply it through [`FixedRule`], or to samples on an equally spaced grid
/// through [`Self::integrate_samples_into`].
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, Interval};
/// use scioxide_integrate::{FixedRule, NewtonCotes, NewtonCotesEndpoints};
///
/// // Composite Simpson is exact for cubics: ∫₀² x³ dx = 4.
/// let simpson = NewtonCotes::<f64>::new(3, NewtonCotesEndpoints::Closed, Budget::try_from(4)?)?;
/// assert_eq!(simpson.node_count(), 9);
/// let (value, report) = simpson.integrate(|x: f64| Ok(x * x * x), Interval::new(0.0, 2.0)?)?;
/// assert!((value - 4.0).abs() < 1e-14);
/// assert!(report.error_estimate().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed.,
/// Academic Press, 1984.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct NewtonCotes<S: RealScalar> {
    endpoints: NewtonCotesEndpoints,
    len: usize,
    panels: Budget,
    count: usize,
    intervals: usize,
    weights: [S; MAX_NODES],
    numerators: [S; MAX_NODES],
    boundary: S,
    denominator: S,
    divisor: S,
    grid: S,
    error_order: usize,
    error_coefficient: S,
    amplification: S,
}

impl<S: RealScalar> NewtonCotes<S> {
    /// Creates the composite rule with a `nodes`-point stencil on `panels` panels.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (argument `nodes`) if `nodes` is outside 1–11 for an open
    ///   or 2–11 for a closed stencil.
    /// - `SizeOverflow` if the number of grid intervals `G` or of nodes
    ///   overflows.
    /// - `ScalarConversion` if `G` is not exactly representable in `S`.
    pub fn new(
        nodes: usize,
        endpoints: NewtonCotesEndpoints,
        panels: Budget,
    ) -> Result<Self, SciOxideError> {
        let minimum = match endpoints {
            NewtonCotesEndpoints::Closed => 2,
            NewtonCotesEndpoints::Open => 1,
        };
        if !(minimum..=MAX_NODES).contains(&nodes) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("nodes"),
                InvalidInputReason::OutOfRange,
                "supported node counts are 1..=11 open and 2..=11 closed",
            ));
        }
        let stencil = Stencil::new(nodes, endpoints);
        let width = match endpoints {
            NewtonCotesEndpoints::Closed => nodes - 1,
            NewtonCotesEndpoints::Open => nodes + 1,
        };
        let intervals = scioxide_core::backend::checked_mul(
            panels.get(),
            width,
            scioxide_core::quantity!("newton_cotes_grid"),
        )?;
        let count = match endpoints {
            NewtonCotesEndpoints::Closed => intervals.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("newton_cotes_nodes"))
            })?,
            NewtonCotesEndpoints::Open => scioxide_core::backend::checked_mul(
                panels.get(),
                nodes,
                scioxide_core::quantity!("newton_cotes_nodes"),
            )?,
        };
        // Every grid index up to `intervals` must convert exactly. Two consecutive
        // representable integers imply that all smaller integers are representable.
        let grid = S::try_from_usize_exact(intervals)?;
        S::try_from_usize_exact(intervals - 1)?;
        let reduced = stencil.reduced();
        let divisor = reduced
            .denominator
            .checked_mul(intervals as i128)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("newton_cotes_grid"))
            })?;
        let mut weights = [S::ZERO; MAX_NODES];
        let mut numerators = [S::ZERO; MAX_NODES];
        for index in 0..nodes {
            weights[index] = ratio(stencil.numerators[index], stencil.denominator)?;
            numerators[index] = integer(reduced.numerators[index])?;
        }
        Ok(Self {
            endpoints,
            len: nodes,
            panels,
            count,
            intervals,
            weights,
            numerators,
            boundary: integer(reduced.numerators[0] + reduced.numerators[nodes - 1])?,
            denominator: integer(reduced.denominator)?,
            divisor: integer(divisor)?,
            grid,
            error_order: stencil.error_order,
            error_coefficient: ratio(stencil.error_moment, stencil.error_denominator)?,
            amplification: ratio(stencil.absolute, stencil.denominator * width as i128)?,
        })
    }

    /// Returns the weights of one panel for unit spacing `h = 1`, one per
    /// stencil node; Simpson's rule gives `[1/3, 4/3, 1/3]`.
    pub fn weights(&self) -> &[S] {
        &self.weights[..self.len]
    }

    /// Returns whether the stencil is closed or open.
    pub const fn endpoints(&self) -> NewtonCotesEndpoints {
        self.endpoints
    }

    /// Returns the number of panels.
    pub const fn panels(&self) -> Budget {
        self.panels
    }

    /// Returns the highest polynomial degree the rule integrates exactly, in
    /// exact arithmetic.
    pub const fn exactness_degree(&self) -> usize {
        self.error_order - 1
    }

    /// Returns the order `m` of the derivative in the leading truncation error
    /// term (see [`Self::error_coefficient`]).
    pub const fn error_order(&self) -> usize {
        self.error_order
    }

    /// Returns the coefficient `C` of the truncation error of one panel,
    ///
    /// ```text
    /// E = ∫ f dx − Σⱼ h wⱼ f(xⱼ) = C h^(m+1) f⁽ᵐ⁾(ξ),
    /// ```
    ///
    /// where `h` is the grid spacing, `wⱼ` are the [unit-spacing weights](Self::weights),
    /// `m` is [`Self::error_order`] and `ξ` is a point in the panel. Simpson's
    /// rule gives `C = −1/90` and `m = 4`. The term holds for `f` with a continuous
    /// `m`-th derivative and is not an error bound for the computed value.
    pub const fn error_coefficient(&self) -> S {
        self.error_coefficient
    }

    /// Returns `Σⱼ |wⱼ| / W`, the sum of the absolute unit-spacing weights divided
    /// by the panel width `W` (`n − 1` closed, `n + 1` open).
    ///
    /// An absolute error `δ` in every sample changes the result by at most this
    /// factor times `δ |b − a|`. It is 1 when all weights are positive.
    pub const fn weight_amplification(&self) -> S {
        self.amplification
    }

    /// Applies the composite rule to samples on an equally spaced grid and writes
    /// the result to `output`.
    ///
    /// `samples` holds one value per node, in the order of
    /// [`FixedRule::map_nodes_into`], and `spacing` is the signed grid spacing
    /// `h = (b − a)/G`. Open stencils skip the panel endpoints, so consecutive
    /// samples across a panel boundary are `2h` apart. `scratch` needs one entry
    /// per real component of the output.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `spacing` or a sample is NaN or infinite.
    /// - `DimensionMismatch` if `samples.len()` is not the node count, a sample
    ///   does not have the component count of `output`, or `scratch` is shorter
    ///   than that count.
    /// - `NumericalInstability` if the result overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Budget;
    /// use scioxide_integrate::{NewtonCotes, NewtonCotesEndpoints};
    ///
    /// // Composite trapezoid with h = 0.5 on samples of (x², 1) at x = 0, 0.5, 1
    /// let trapezoid = NewtonCotes::<f64>::new(2, NewtonCotesEndpoints::Closed, Budget::try_from(2)?)?;
    /// let samples = [[0.0, 1.0], [0.25, 1.0], [1.0, 1.0]];
    /// let mut output = [0.0; 2];
    /// trapezoid.integrate_samples_into(&samples, 0.5, &mut output, &mut [0.0; 2])?;
    /// assert_eq!(output, [0.375, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn integrate_samples_into<V, O>(
        &self,
        samples: &[V],
        spacing: S,
        output: &mut O,
        scratch: &mut [S],
    ) -> Result<(), SciOxideError>
    where
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        scioxide_core::validate_value(&spacing)?;
        crate::kernel::shape(self.count, samples.len())?;
        let components = output.dimension()?;
        if scratch.len() < components {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(components),
                scioxide_core::Dimensions::one(scratch.len()),
            ));
        }
        for sample in samples {
            scioxide_core::check_value_shape(sample, components)?;
            scioxide_core::validate_value(sample)?;
        }
        let spacing = Scaled::from_value(spacing)?;
        let denominator = Scaled::from_value(self.denominator)?;
        for (component, target) in scratch[..components].iter_mut().enumerate() {
            let mut sum = ScaledAccumulator::ZERO;
            for (index, sample) in samples.iter().enumerate() {
                sum.add_product(
                    Scaled::from_value(self.grid_node(index).1)?,
                    Scaled::from_value(sample.component(component)?)?,
                )?;
            }
            *target = sum.total()?.mul(spacing)?.div(denominator)?.round()?;
        }
        scioxide_core::write_value(&scratch[..components], output)
    }

    /// Grid index and exact reduced weight numerator of node `index`.
    fn grid_node(&self, index: usize) -> (usize, S) {
        match self.endpoints {
            NewtonCotesEndpoints::Closed => {
                let local = index % (self.len - 1);
                let numerator = if index == self.intervals {
                    self.numerators[self.len - 1]
                } else if local == 0 && index != 0 {
                    self.boundary
                } else {
                    self.numerators[local]
                };
                (index, numerator)
            }
            NewtonCotesEndpoints::Open => {
                let (panel, local) = (index / self.len, index % self.len);
                (panel * (self.len + 1) + local + 1, self.numerators[local])
            }
        }
    }
}

impl<S: RealScalar> FixedRule<S> for NewtonCotes<S> {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "NewtonCotes", 1)
    }

    fn node_count(&self) -> usize {
        self.count
    }

    fn map_nodes_into(
        &self,
        interval: Interval<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<(), SciOxideError> {
        crate::kernel::shape(self.count, destination.len())?;
        let geometry = IntervalGeometry::prepare(interval)?.ok_or_else(degenerate_interval)?;
        for (index, slot) in destination.iter_mut().enumerate() {
            let (grid, numerator) = self.grid_node(index);
            // The span stays in extended range and is divided last, so huge intervals do
            // not overflow and each weight is one rounding of its exact rational value.
            let weight = if let Some(span) = geometry.native_span() {
                let weight = (span / self.divisor) * numerator;
                if weight.is_finite() && weight.abs() >= S::MIN_POSITIVE {
                    weight
                } else {
                    geometry
                        .span()?
                        .mul(Scaled::from_value(numerator)?)?
                        .div(Scaled::from_value(self.divisor)?)?
                        .round()?
                }
            } else {
                geometry
                    .span()?
                    .mul(Scaled::from_value(numerator)?)?
                    .div(Scaled::from_value(self.divisor)?)?
                    .round()?
            };
            *slot = FixedRuleNode::new(
                geometry.point_at_fraction(S::try_from_usize_exact(grid)? / self.grid)?,
                weight,
                None,
            );
        }
        Ok(())
    }
}

/// Exact integer weights of one stencil over a shared denominator.
struct Stencil {
    numerators: [i128; MAX_NODES],
    denominator: i128,
    absolute: i128,
    error_order: usize,
    error_moment: i128,
    error_denominator: i128,
}

/// Weight numerators and denominator reduced by their common divisor.
struct Reduced {
    numerators: [i128; MAX_NODES],
    denominator: i128,
}

impl Stencil {
    fn new(nodes: usize, endpoints: NewtonCotesEndpoints) -> Self {
        let start = usize::from(endpoints == NewtonCotesEndpoints::Open);
        let width = if start == 0 { nodes - 1 } else { nodes + 1 };
        // lcm(1,...,13). All bounded products below fit i128 at the admitted order.
        const COMMON: i128 = 360_360;
        let factorial = (1..nodes as i128).product::<i128>();
        let denominator = COMMON * factorial;
        let mut numerators = [0_i128; MAX_NODES];
        // Weight j is the integral over [0, width] of the Lagrange basis polynomial
        // `prod_{k != j} (x - x_k) / (x_j - x_k)`. The numerator polynomial is expanded
        // in exact integer coefficients and integrated term-by-term; `COMMON / (power + 1)`
        // is an exact integer because power + 1 <= 12 divides lcm(1, ..., 13). The divisor
        // `prod (j - k)` has magnitude j!(n-1-j)!, which divides (n-1)!, so every weight is
        // an exact integer ratio over the shared `denominator`.
        for (j, slot) in numerators.iter_mut().enumerate().take(nodes) {
            let mut polynomial = [0_i128; 12];
            polynomial[0] = 1;
            let mut divisor = 1_i128;
            for (degree, k) in (0..nodes).filter(|&k| k != j).enumerate() {
                let node = (k + start) as i128;
                // Multiply the running polynomial by (x - node) in place, highest power first
                // so each coefficient reads its not-yet-updated lower neighbour.
                for power in (1..=degree + 1).rev() {
                    polynomial[power] = polynomial[power - 1] - node * polynomial[power];
                }
                polynomial[0] *= -node;
                divisor *= j as i128 - k as i128;
            }
            let numerator: i128 = polynomial[..nodes]
                .iter()
                .enumerate()
                .map(|(power, coefficient)| {
                    coefficient
                        * (width as i128).pow((power + 1) as u32)
                        * (COMMON / (power as i128 + 1))
                })
                .sum();
            *slot = numerator * (factorial / divisor);
        }
        // Symmetric stencils with an odd node count also integrate the next odd monomial
        // exactly, so the leading error term is one derivative higher than the node count.
        // Its coefficient is the defect on x^order divided by order!, following the
        // Peano-kernel form `E = C h^(order+1) f^(order)(xi)`.
        let order = nodes + usize::from(!nodes.is_multiple_of(2));
        let error_moment = (width as i128).pow((order + 1) as u32)
            * (denominator / (order as i128 + 1))
            - numerators[..nodes]
                .iter()
                .enumerate()
                .map(|(j, n)| n * ((j + start) as i128).pow(order as u32))
                .sum::<i128>();
        Self {
            numerators,
            denominator,
            absolute: numerators[..nodes].iter().map(|n| n.abs()).sum(),
            error_order: order,
            error_moment,
            error_denominator: denominator * (1..=order as i128).product::<i128>(),
        }
    }

    fn reduced(&self) -> Reduced {
        let common = self
            .numerators
            .iter()
            .fold(self.denominator, |common, &numerator| {
                gcd(common, numerator)
            });
        Reduced {
            numerators: self.numerators.map(|numerator| numerator / common),
            denominator: self.denominator / common,
        }
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    (a, b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn ratio<S: RealScalar>(numerator: i128, denominator: i128) -> Result<S, SciOxideError> {
    // Integer conversion is explicitly lossy; the final division is native to S.
    Ok(integer::<S>(numerator)? / integer::<S>(denominator)?)
}

fn integer<S: RealScalar>(value: i128) -> Result<S, SciOxideError> {
    let negative = value < 0;
    let magnitude = value.unsigned_abs();
    // `RealScalar` only offers exact conversion from `usize`, which may be 32 bits wide, so
    // the 128-bit magnitude is rebuilt from 16-bit limbs by Horner's rule. Each limb is
    // exact; only the final accumulation can round, which is the intended lossy step.
    let radix = S::try_from_usize_exact(65536)?;
    let mut result = S::ZERO;
    for shift in (0..8).rev() {
        result = result * radix
            + S::try_from_usize_exact(((magnitude >> (shift * 16)) & 65535) as usize)?;
    }
    Ok(if negative { -result } else { result })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn rule(nodes: usize, endpoints: NewtonCotesEndpoints, panels: usize) -> NewtonCotes<f64> {
        NewtonCotes::new(nodes, endpoints, Budget::try_from(panels).unwrap()).unwrap()
    }

    #[test]
    fn classical_stencils_reduce_to_their_textbook_fractions() {
        let simpson = Stencil::new(3, NewtonCotesEndpoints::Closed).reduced();
        assert_eq!(simpson.numerators[..3], [1, 4, 1]);
        assert_eq!(simpson.denominator, 3);
        let boole = Stencil::new(5, NewtonCotesEndpoints::Closed).reduced();
        assert_eq!(boole.numerators[..5], [14, 64, 24, 64, 14]);
        assert_eq!(boole.denominator, 45);
        let midpoint = Stencil::new(1, NewtonCotesEndpoints::Open).reduced();
        assert_eq!(midpoint.numerators[..1], [2]);
        assert_eq!(midpoint.denominator, 1);
    }

    #[test]
    fn composite_layout_shares_closed_endpoints_and_skips_open_ones() {
        let interval = Interval::new(0.0, 1.0).unwrap();
        let closed = rule(3, NewtonCotesEndpoints::Closed, 2);
        assert_eq!(closed.node_count(), 5);
        let mut nodes = alloc::vec![FixedRuleNode::new(0.0, 0.0, None); 5];
        closed.map_nodes_into(interval, &mut nodes).unwrap();
        assert_eq!(
            nodes
                .iter()
                .map(|node| node.point())
                .collect::<alloc::vec::Vec<_>>(),
            [0.0, 0.25, 0.5, 0.75, 1.0]
        );
        assert_eq!(nodes[2].primary_weight(), 2.0 / 12.0);
        let open = rule(1, NewtonCotesEndpoints::Open, 4);
        assert_eq!(open.node_count(), 4);
        let mut nodes = [FixedRuleNode::new(0.0, 0.0, None); 4];
        open.map_nodes_into(interval, &mut nodes).unwrap();
        assert_eq!(nodes[3].point(), 0.875);
        assert_eq!(nodes[3].primary_weight(), 0.25);
        assert!(open.map_nodes_into(interval, &mut nodes[..3]).is_err());
    }

    #[test]
    fn unsupported_stencils_and_grids_are_rejected() {
        for (nodes, endpoints) in [
            (1, NewtonCotesEndpoints::Closed),
            (0, NewtonCotesEndpoints::Open),
            (12, NewtonCotesEndpoints::Open),
        ] {
            assert!(matches!(
                NewtonCotes::<f64>::new(nodes, endpoints, Budget::ONE)
                    .err()
                    .unwrap()
                    .view(),
                scioxide_core::SciOxideErrorView::InvalidInput { .. }
            ));
        }
        assert!(
            NewtonCotes::<f32>::new(
                2,
                NewtonCotesEndpoints::Closed,
                Budget::try_from((1 << 24) + 2).unwrap()
            )
            .is_err()
        );
    }
}
