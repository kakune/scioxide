//! Gauss–Kronrod pairs: tabulated nodes and weights and the embedded error
//! estimate. The tables store only the nonnegative half of each symmetric rule.

#![allow(clippy::excessive_precision)]

use scioxide_core::{Interval, RealScalar, Scaled, ScaledAccumulator, SciOxideError};

use crate::fixed::{FixedRule, FixedRuleNode, interval_map};
use crate::kernel::{AffineMap1D, checked_intermediate};

const MAX_POSITIVE_NODES: usize = 25;
/// Maximum callback points in any supported Kronrod pair.
pub(crate) const MAX_NODES: usize = 2 * MAX_POSITIVE_NODES + 1;
const SAMPLE_INDICES: [usize; MAX_NODES] = {
    let mut indices = [0; MAX_NODES];
    let mut index = 0;
    while index < indices.len() {
        indices[index] = index;
        index += 1;
    }
    indices
};

/// Selects a Gauss–Kronrod pair.
///
/// `G<n>K<2n+1>` is the `n`-point Gauss–Legendre rule together with its
/// `(2n + 1)`-point Kronrod extension, which reuses the `n` Gauss nodes. The
/// Gauss rule is exact for polynomials of degree `2n − 1`; the Kronrod degree
/// is given on each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussKronrodRule {
    /// 7-point Gauss and 15-point Kronrod rule, exact for degree 23.
    G7K15,
    /// 10-point Gauss and 21-point Kronrod rule, exact for degree 31.
    G10K21,
    /// 15-point Gauss and 31-point Kronrod rule, exact for degree 47.
    G15K31,
    /// 20-point Gauss and 41-point Kronrod rule, exact for degree 61.
    G20K41,
    /// 25-point Gauss and 51-point Kronrod rule, exact for degree 77.
    G25K51,
}

impl GaussKronrodRule {
    pub(crate) const fn rule(self) -> &'static KronrodTable {
        match self {
            Self::G7K15 => &G7K15_RULE,
            Self::G10K21 => &G10K21_RULE,
            Self::G15K31 => &G15K31_RULE,
            Self::G20K41 => &G20K41_RULE,
            Self::G25K51 => &G25K51_RULE,
        }
    }
}

/// Gauss–Kronrod quadrature on a finite interval, with an error estimate.
///
/// The Kronrod rule gives the integral and the embedded Gauss rule the error
/// estimate. With the mapped Kronrod weights `wᵢ`, Gauss weights `ŵᵢ` (zero at
/// the Kronrod-only nodes) and nodes `xᵢ` on `[a, b]`,
///
/// ```text
/// I = Σ wᵢ f(xᵢ),   G = Σ ŵᵢ f(xᵢ),
/// ρ = Σ wᵢ |f(xᵢ) − I/(b − a)|,
/// E = ρ · min(1, (200 |I − G| / ρ)^(3/2)),
/// ```
///
/// where `ρ` approximates `∫ |f − mean f| dx`. The rule returns `I` with the
/// discretization error `E`, or `|I − G|` if `ρ = 0`. The roundoff error is
/// `50 ε Σ wᵢ |f(xᵢ)|`, where `ε` is the machine epsilon of `S`. The scaling
/// of `|I − G|` is empirical: it is smaller than `|I − G|` when the two rules
/// agree closely and never larger than `ρ`. See
/// [Error estimates](crate#error-estimates).
///
/// A call evaluates `f` at the `2n + 1` Kronrod nodes of the chosen
/// [`GaussKronrodRule`]. The nodes and weights come from tables and are
/// rounded to `S` once, by [`Self::new`]. Apply the rule through [`FixedRule`];
/// its [mapped nodes](FixedRule::map_nodes_into) carry `wᵢ` as the primary and
/// `ŵᵢ` as the embedded weight. [`AdaptiveIntegrator`](crate::AdaptiveIntegrator)
/// applies Gauss–Kronrod rules adaptively on subintervals.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{FixedRule, GaussKronrod, GaussKronrodRule};
///
/// // ∫₀¹ eˣ dx = e − 1
/// let rule = GaussKronrod::new(GaussKronrodRule::G7K15)?;
/// let (value, report) = rule.integrate(|x: f64| Ok(x.exp()), Interval::new(0.0, 1.0)?)?;
/// assert!((value - (1.0_f64.exp() - 1.0)).abs() < 1e-15);
/// assert!(report.error_estimate().unwrap() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. S. Kronrod, *Nodes and Weights of Quadrature Formulas*, Consultants
///   Bureau, New York, 1965.
/// - R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner,
///   *QUADPACK: A Subroutine Package for Automatic Integration*, Springer,
///   1983.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct GaussKronrod<S: RealScalar> {
    rule: KronrodRule<S>,
}

impl<S: RealScalar> GaussKronrod<S> {
    /// Creates the pair `pair` with nodes and weights rounded to `S`.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if a tabulated node or weight cannot be converted
    ///   to `S`.
    pub fn new(pair: GaussKronrodRule) -> Result<Self, SciOxideError> {
        let table = pair.rule();
        let zero = KronrodNode {
            abscissa: S::ZERO,
            gauss_weight: S::ZERO,
            kronrod_weight: S::ZERO,
        };
        let mut nodes = [zero; MAX_POSITIVE_NODES];
        for (slot, node) in nodes.iter_mut().zip(table.positive_nodes) {
            *slot = KronrodNode {
                abscissa: S::try_from_f64_lossy(node.abscissa)?,
                gauss_weight: S::try_from_f64_lossy(node.gauss_weight)?,
                kronrod_weight: S::try_from_f64_lossy(node.kronrod_weight)?,
            };
        }
        let center_gauss_weight = S::try_from_f64_lossy(table.center_gauss_weight)?;
        let center_kronrod_weight = S::try_from_f64_lossy(table.center_kronrod_weight)?;
        let min_weight = nodes[..table.positive_nodes.len()]
            .iter()
            .flat_map(|n| [n.gauss_weight, n.kronrod_weight])
            .chain([center_gauss_weight, center_kronrod_weight])
            .filter(|&w| w > S::ZERO)
            .fold(S::ONE, |m, w| m.min(w));
        let count = table.positive_nodes.len();
        let separation = nodes[..count].windows(2).fold(
            (S::ONE - nodes[0].abscissa).min(nodes[count - 1].abscissa),
            |gap, pair| gap.min(pair[0].abscissa - pair[1].abscissa),
        );
        Ok(Self {
            rule: KronrodRule {
                nodes,
                count: table.positive_nodes.len(),
                range: scioxide_core::backend::ShortReductionRange::new(2 * count + 1),
                min_weight,
                separation,
                center_gauss_weight,
                center_kronrod_weight,
            },
        })
    }

    pub(crate) const fn rule(&self) -> &KronrodRule<S> {
        &self.rule
    }

    /// Maps the nodes to `interval`; returns whether their geometry is known to
    /// be valid without a separate check.
    #[inline(always)]
    fn mapped_nodes(
        &self,
        interval: Interval<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<bool, SciOxideError> {
        self.mapped_nodes_in(interval_map(interval)?, destination)
    }

    #[inline(always)]
    fn mapped_nodes_in(
        &self,
        map: AffineMap1D<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<bool, SciOxideError> {
        self.rule.fixed_nodes(map, destination)
    }
}

#[derive(Debug, Clone, Copy)]
struct KronrodNode<S = f64> {
    abscissa: S,
    gauss_weight: S,
    kronrod_weight: S,
}

const fn node(abscissa: f64, gauss_weight: f64, kronrod_weight: f64) -> KronrodNode {
    KronrodNode {
        abscissa,
        gauss_weight,
        kronrod_weight,
    }
}

/// Tabulated pair on `[-1, 1]`, used by the fixed and adaptive integrators.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KronrodTable {
    positive_nodes: &'static [KronrodNode],
    center_gauss_weight: f64,
    center_kronrod_weight: f64,
}

/// Tabulated pair converted to `S`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KronrodRule<S: RealScalar> {
    nodes: [KronrodNode<S>; MAX_POSITIVE_NODES],
    count: usize,
    range: scioxide_core::backend::ShortReductionRange<S>,
    min_weight: S,
    separation: S,
    center_gauss_weight: S,
    center_kronrod_weight: S,
}

/// Integral, discretization error and roundoff error of one application.
pub(crate) struct KronrodEstimate<S: RealScalar> {
    value: S,
    error: S,
    roundoff_floor: S,
}

impl<S: RealScalar> KronrodEstimate<S> {
    pub(crate) const fn value(&self) -> S {
        self.value
    }

    pub(crate) const fn error(&self) -> S {
        self.error
    }

    pub(crate) const fn roundoff_floor(&self) -> S {
        self.roundoff_floor
    }
}

impl KronrodTable {
    pub(crate) const fn evaluation_cost(&self) -> usize {
        self.positive_nodes.len() * 2 + 1
    }
}

impl<S: RealScalar> KronrodRule<S> {
    pub(crate) const fn evaluation_cost(&self) -> usize {
        self.count * 2 + 1
    }

    /// Returns the midpoint and half-width of `map` when every mapped node and
    /// weight is representable and the nodes are separated, so the nodes can be
    /// computed while sampling.
    #[inline(always)]
    pub(crate) fn native_frame(&self, map: AffineMap1D<S>) -> Option<(S, S)> {
        let (middle, half) =
            map.native_frame(self.nodes[self.count - 1].abscissa, self.min_weight, S::ONE)?;
        crate::kernel::separated(half * self.separation, middle.abs() + half)
            .then_some((middle, half))
    }

    #[inline(always)]
    pub(crate) fn fixed_nodes(
        &self,
        map: AffineMap1D<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<bool, SciOxideError> {
        crate::kernel::shape(self.evaluation_cost(), destination.len())?;
        self.map_stencil::<true>(map, |index, point, primary, embedded| {
            destination[index] = FixedRuleNode::new(point, primary, Some(embedded));
            Ok(())
        })
    }

    /// Evaluates `function` at the mapped nodes, left to right, computing each
    /// node as in `map_stencil`.
    #[inline(always)]
    pub(crate) fn sample_native_scalar<F, V>(
        &self,
        middle: S,
        half: S,
        function: &F,
        rows: &mut [S],
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: scioxide_core::AffineValue<Scale = S>,
    {
        crate::evaluation::sample_scalar::<S, usize, 0, false>(
            &SAMPLE_INDICES[..self.evaluation_cost()],
            rows,
            context,
            {
                #[inline(always)]
                |&index: &usize| {
                    let point = if index < self.count {
                        middle - half * self.nodes[index].abscissa
                    } else if index == self.count {
                        middle
                    } else {
                        middle + half * self.nodes[2 * self.count - index].abscissa
                    };
                    crate::evaluation::scalar_value(&function(point)?)
                }
            },
            |_| S::ONE,
        )?;
        Ok(())
    }

    /// Writes the mapped nodes in ascending order.
    #[inline]
    pub(crate) fn points(
        &self,
        map: AffineMap1D<S>,
        points: &mut [S],
    ) -> Result<bool, SciOxideError> {
        crate::kernel::shape(self.evaluation_cost(), points.len())?;
        self.map_stencil::<false>(map, |index, point, _, _| {
            points[index] = point;
            Ok(())
        })
    }

    /// Emits each mapped node with its Kronrod and Gauss weights; returns whether
    /// the nodes are known to be separated.
    #[inline]
    fn map_stencil<const WEIGHTS: bool>(
        &self,
        map: AffineMap1D<S>,
        mut emit: impl FnMut(usize, S, S, S) -> Result<(), SciOxideError>,
    ) -> Result<bool, SciOxideError> {
        if let Some((middle, half)) =
            map.native_frame(self.nodes[self.count - 1].abscissa, self.min_weight, S::ONE)
        {
            emit(
                self.count,
                middle,
                half * self.center_kronrod_weight,
                half * self.center_gauss_weight,
            )?;
            for (i, node) in self.nodes[..self.count].iter().enumerate() {
                let offset = half * node.abscissa;
                let primary = half * node.kronrod_weight;
                let embedded = half * node.gauss_weight;
                emit(i, middle - offset, primary, embedded)?;
                emit(2 * self.count - i, middle + offset, primary, embedded)?;
            }
            return Ok(crate::kernel::separated(
                half * self.separation,
                middle.abs() + half,
            ));
        }
        emit(
            self.count,
            map.point(S::ZERO)?,
            if WEIGHTS {
                map.weight(self.center_kronrod_weight)?
            } else {
                S::ZERO
            },
            if WEIGHTS {
                map.weight(self.center_gauss_weight)?
            } else {
                S::ZERO
            },
        )?;
        for (i, node) in self.nodes[..self.count].iter().enumerate() {
            let primary = if WEIGHTS {
                map.weight(node.kronrod_weight)?
            } else {
                S::ZERO
            };
            let embedded = if WEIGHTS {
                map.weight(node.gauss_weight)?
            } else {
                S::ZERO
            };
            emit(i, map.point(-node.abscissa)?, primary, embedded)?;
            emit(
                2 * self.count - i,
                map.point(node.abscissa)?,
                primary,
                embedded,
            )?;
        }
        Ok(false)
    }

    #[inline(always)]
    pub(crate) fn apply_samples<const PAIRS: usize>(
        &self,
        scale: scioxide_core::PromotingProduct<S>,
        mut sample_at: impl FnMut(usize) -> Result<S, SciOxideError>,
    ) -> Result<KronrodEstimate<S>, SciOxideError> {
        let mut values = [S::ZERO; 51];
        for (i, value) in values[..self.evaluation_cost()].iter_mut().enumerate() {
            *value = sample_at(i)?;
        }
        self.apply_values::<PAIRS>(scale, &values[..self.evaluation_cost()])
    }

    /// Computes the estimate from the sampled values in ascending node order.
    #[inline(always)]
    pub(crate) fn apply_values<const PAIRS: usize>(
        &self,
        scale: scioxide_core::PromotingProduct<S>,
        values: &[S],
    ) -> Result<KronrodEstimate<S>, SciOxideError> {
        // Adaptive G7K15 knows its immutable pair order at preparation. Other
        // callers use PAIRS = 0 for the same kernel's dynamic dispatch.
        if PAIRS != 0 {
            crate::kernel::shape(PAIRS, self.count)?;
        }
        let count = if PAIRS == 0 { self.count } else { PAIRS };
        crate::kernel::shape(2 * count + 1, values.len())?;
        let safe = |x| self.range.admits(x);
        let mut signs = 0_u8;
        if let Some(half) = scale.native().filter(|h| *h >= S::ZERO && safe(*h))
            && self.min_weight >= S::EPSILON
            && values.iter().fold(true, |valid, &x| {
                signs |= u8::from(x.is_sign_negative()) + 1;
                valid & safe(x)
            })
        {
            return match count {
                7 => self.native_samples::<7, 15>(half, values, signs),
                10 => self.native_samples::<10, 21>(half, values, signs),
                15 => self.native_samples::<15, 31>(half, values, signs),
                20 => self.native_samples::<20, 41>(half, values, signs),
                25 => self.native_samples::<25, 51>(half, values, signs),
                _ => Err(crate::kernel::invalid(
                    scioxide_core::argument!("rule"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "unsupported prepared Kronrod order",
                )),
            };
        }
        self.extended_values(scale, values)
    }

    #[inline(always)]
    fn native_samples<const N: usize, const M: usize>(
        &self,
        half: S,
        values: &[S],
        signs: u8,
    ) -> Result<KronrodEstimate<S>, SciOxideError> {
        let values = values.first_chunk::<M>().ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("samples"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "retained Kronrod row does not match its prepared order",
            )
        })?;
        let (mut kronrod, mut gauss) = if signs != 3 {
            // Positive quadrature weights preserve a common sample sign.
            // These short lane sums have no cancellation; their error is
            // covered by the unchanged 50-epsilon absolute-mass floor.
            let [kronrod, gauss] = scioxide_core::backend::four_lane_sums(N, |i| {
                let pair = values[i] + values[2 * N - i];
                [
                    self.nodes[i].kronrod_weight * pair,
                    self.nodes[i].gauss_weight * pair,
                ]
            });
            ([kronrod, S::ZERO], [gauss, S::ZERO])
        } else {
            let Some([kronrod]) = scioxide_core::backend::compensated_indexed(N, |i| {
                [self.nodes[i].kronrod_weight * (values[i] + values[2 * N - i])]
            }) else {
                return self
                    .extended_values(scioxide_core::PromotingProduct::from_value(half)?, values);
            };
            let Some([gauss]) = scioxide_core::backend::compensated_indexed(N / 2, |j| {
                let i = 2 * j + 1;
                [self.nodes[i].gauss_weight * (values[i] + values[2 * N - i])]
            }) else {
                return self
                    .extended_values(scioxide_core::PromotingProduct::from_value(half)?, values);
            };
            (kronrod.parts(), gauss.parts())
        };
        if N % 2 == 1 {
            gauss = add_bounded_parts(gauss, self.center_gauss_weight * values[N]);
        }
        kronrod = add_bounded_parts(kronrod, self.center_kronrod_weight * values[N]);
        let kronrod_value = kronrod[0] + kronrod[1];
        let absolute = if signs != 3 {
            // With a common sample sign and positive weights, this is the
            // identical weighted absolute sum, including the center residual.
            kronrod_value.abs()
        } else {
            let [absolute] = scioxide_core::backend::four_lane_sums(N, |i| {
                [self.nodes[i].kronrod_weight * (values[i].abs() + values[2 * N - i].abs())]
            });
            let [hi, lo] = add_bounded_parts(
                [absolute, S::ZERO],
                self.center_kronrod_weight * values[N].abs(),
            );
            hi + lo
        };
        let difference = add_bounded_parts(kronrod, -gauss[0]);
        let difference = if gauss[1] == S::ZERO {
            difference
        } else {
            add_bounded_parts(difference, -gauss[1])
        };
        let raw = (difference[0] + difference[1]).abs() * half;
        let resabs = absolute * half;
        // A zero embedded difference already fixes the discretization estimate;
        // its deviation mass cannot change that result. Avoid a second retained
        // sample traversal in this exact case, independently of tolerance.
        let error = if raw == S::ZERO {
            raw
        } else {
            let mean = kronrod_value * S::HALF;
            let [centered] = scioxide_core::backend::four_lane_sums(N, |i| {
                [self.nodes[i].kronrod_weight
                    * ((values[i] - mean).abs() + (values[2 * N - i] - mean).abs())]
            });
            let centered = checked_intermediate(
                centered + self.center_kronrod_weight * (values[N] - mean).abs(),
            )?;
            let resasc = centered * half;
            if resasc == S::ZERO {
                raw
            } else {
                rescale_nonzero_error(raw, resasc, S::from_f64_ieee(200.0))
            }
        };
        // Admission bounds every nonzero sample and half-span below/above
        // square-root range limits, with min_weight >= epsilon. A nonzero
        // 50*epsilon*resabs is therefore normal; all upper products fit.
        let roundoff_floor = (S::from_f64_ieee(50.0) * S::EPSILON) * resabs.abs();
        Ok(KronrodEstimate {
            value: kronrod_value * half,
            error,
            roundoff_floor,
        })
    }

    #[cold]
    #[inline(never)]
    fn extended_values(
        &self,
        scale: scioxide_core::PromotingProduct<S>,
        values: &[S],
    ) -> Result<KronrodEstimate<S>, SciOxideError> {
        let center_value = values[self.count];
        let center_gauss_weight: S = self.center_gauss_weight;
        let center_kronrod_weight: S = self.center_kronrod_weight;
        let mut gauss = ScaledAccumulator::ZERO;
        gauss.add_product(
            Scaled::from_value(center_gauss_weight)?,
            Scaled::from_value(center_value)?,
        )?;
        let mut kronrod = ScaledAccumulator::ZERO;
        kronrod.add_product(
            Scaled::from_value(center_kronrod_weight)?,
            Scaled::from_value(center_value)?,
        )?;
        let mut resabs = ScaledAccumulator::ZERO;
        resabs.add_product(
            Scaled::from_value(center_kronrod_weight)?,
            Scaled::from_value(center_value.abs())?,
        )?;

        for (index, node) in self.nodes[..self.count].iter().enumerate() {
            let lower_value = values[index];
            let upper_value = values[2 * self.count - index];

            // Symmetric nodes share a weight, so each pair is summed first (one multiply per
            // pair). Kronrod-only nodes have a zero Gauss weight and are skipped for Gauss.
            let pair_sum =
                Scaled::from_value(lower_value)?.add(Scaled::from_value(upper_value)?)?;
            if node.gauss_weight != S::ZERO {
                let gauss_weight: S = node.gauss_weight;
                gauss.add_product(Scaled::from_value(gauss_weight)?, pair_sum)?;
            }
            let kronrod_weight: S = node.kronrod_weight;
            kronrod.add_product(Scaled::from_value(kronrod_weight)?, pair_sum)?;
            let pair_abs = Scaled::from_value(lower_value.abs())?
                .add(Scaled::from_value(upper_value.abs())?)?;
            resabs.add_product(Scaled::from_value(kronrod_weight)?, pair_abs)?;
        }

        // QUADPACK `resasc` = integral of |f - mean(f)| with the Kronrod weights: canonical
        // weights sum to 2, so the mean is half the canonical Kronrod value. It measures how
        // much the integrand varies across the region and scales the error heuristic below.
        let gauss_value = gauss.total()?;
        let kronrod_value = kronrod.total()?;
        let mean = kronrod_value.shifted(-1)?.round()?;
        let mut resasc = ScaledAccumulator::ZERO;
        resasc.add_product(
            Scaled::from_value(center_kronrod_weight)?,
            Scaled::from_value(center_value)?
                .sub(Scaled::from_value(mean)?)?
                .abs(),
        )?;
        for (index, node) in self.nodes[..self.count].iter().enumerate() {
            let deviation = Scaled::from_value(values[index])?
                .sub(Scaled::from_value(mean)?)?
                .abs()
                .add(
                    Scaled::from_value(values[2 * self.count - index])?
                        .sub(Scaled::from_value(mean)?)?
                        .abs(),
                )?;
            resasc.add_product(Scaled::from_value(node.kronrod_weight)?, deviation)?;
        }

        // All sums are formed on the canonical [-1, 1] rule and scaled by |h/2| once at the
        // end, which saves a multiplication per node and keeps the Gauss/Kronrod difference
        // free of per-node weight rounding.
        let absolute_half_width = scale.scaled()?.abs();
        let result = kronrod_value.mul(absolute_half_width)?.round()?;
        let scaled_resabs =
            crate::kernel::diagnostic(resabs.total()?.mul(absolute_half_width)?.round())?;
        let scaled_resasc =
            crate::kernel::diagnostic(resasc.total()?.mul(absolute_half_width)?.round())?;
        let raw_error = crate::kernel::diagnostic(
            kronrod_value
                .sub(gauss_value)?
                .abs()
                .mul(absolute_half_width)?
                .round(),
        )?;
        let (error, roundoff_floor) = refine_error(raw_error, scaled_resabs, scaled_resasc)?;

        Ok(KronrodEstimate {
            value: result,
            error,
            roundoff_floor,
        })
    }
}

/// Adds `value` to a sum stored as a high and a low part, keeping the rounding
/// error in the low part.
#[inline(always)]
fn add_bounded_parts<S: RealScalar>(parts: [S; 2], value: S) -> [S; 2] {
    let (sum, error) = scioxide_core::backend::two_sum(parts[0], value);
    [sum, parts[1] + error]
}

fn refine_error<S: RealScalar>(raw: S, resabs: S, resasc: S) -> Result<(S, S), SciOxideError> {
    let zero = S::ZERO;
    let mut error = raw;
    // QUADPACK's empirical scaling `resasc * min(1, (200 |K - G| / resasc)^1.5)`. The raw
    // |K - G| is the error of the lower-order Gauss rule and grossly overestimates the Kronrod
    // error once the rules agree; the 3/2 power reflects the observed faster decay while the
    // `resasc` cap keeps the estimate pessimistic when the rules disagree badly. The
    // roundoff floor is reported separately and combined by `max` downstream, matching
    // QUADPACK's `max(epmach * 50 * resabs, abserr)` step.
    if resasc != zero && raw != zero {
        let two_hundred = S::try_from_f64_lossy(200.0)?;
        error = rescale_nonzero_error(raw, resasc, two_hundred);
    }

    let roundoff_floor = crate::kernel::roundoff_floor(resabs)?;
    Ok((checked_intermediate(error)?, roundoff_floor))
}

/// Returns `resasc · min(1, (factor · raw / resasc)^(3/2))` for nonzero `raw`
/// and `resasc`; an overflowing `factor · raw` selects `resasc`.
#[inline(always)]
fn rescale_nonzero_error<S: RealScalar>(raw: S, resasc: S, two_hundred: S) -> S {
    let scaled = two_hundred * raw;
    if scaled >= resasc {
        resasc
    } else {
        let ratio = scaled / resasc;
        resasc * (ratio * ratio.sqrt())
    }
}

// Coefficients are the positive half of QUADPACK's DQK15, DQK21, DQK31,
// DQK41, and DQK51 tables. A zero Gauss weight marks a Kronrod-only node.
const G7K15_NODES: &[KronrodNode] = &[
    node(
        9.91455371120812639206854697526328516642e-1,
        0.0,
        2.29353220105292249637320080589695919936e-2,
    ),
    node(
        9.49107912342758524526189684047851262401e-1,
        1.29484966168869693270611432679082018329e-1,
        6.30920926299785532907006631892042866651e-2,
    ),
    node(
        8.64864423359769072789712788640926201211e-1,
        0.0,
        1.04790010322250183839876322541518017444e-1,
    ),
    node(
        7.41531185599394439863864773280788407074e-1,
        2.79705391489276667901467771423779582487e-1,
        1.40653259715525918745189590510237920400e-1,
    ),
    node(
        5.86087235467691130294144838258729598437e-1,
        0.0,
        1.69004726639267902826583426598550284106e-1,
    ),
    node(
        4.05845151377397166906606412076961463347e-1,
        3.81830050505118944950369775488975133878e-1,
        1.90350578064785409913256402421013682826e-1,
    ),
    node(
        2.07784955007898467600689403773244913480e-1,
        0.0,
        2.04432940075298892414161999234649084717e-1,
    ),
];
const G7K15_RULE: KronrodTable = KronrodTable {
    positive_nodes: G7K15_NODES,
    center_gauss_weight: 4.17959183673469387755102040816326530612e-1,
    center_kronrod_weight: 2.09482141084727828012999174891714263698e-1,
};

const G10K21_NODES: &[KronrodNode] = &[
    node(
        9.95657163025808080735527280689002847921e-1,
        0.0,
        1.16946388673718742780643960621920483962e-2,
    ),
    node(
        9.73906528517171720077964012084452053428e-1,
        6.66713443086881375935688098933317928579e-2,
        3.25581623079647274788189724593897606174e-2,
    ),
    node(
        9.30157491355708226001207180059508346225e-1,
        0.0,
        5.47558965743519960313813002445801763737e-2,
    ),
    node(
        8.65063366688984510732096688423493048528e-1,
        1.49451349150580593145776339657697332403e-1,
        7.50396748109199527670431409161900093952e-2,
    ),
    node(
        7.80817726586416897063717578345042377163e-1,
        0.0,
        9.31254545836976055350654650833663443900e-2,
    ),
    node(
        6.79409568299024406234327365114873575769e-1,
        2.19086362515982043995534934228163192459e-1,
        1.09387158802297641899210590325804960272e-1,
    ),
    node(
        5.62757134668604683339000099272694140843e-1,
        0.0,
        1.23491976262065851077958109831074159512e-1,
    ),
    node(
        4.33395394129247190799265943165784162200e-1,
        2.69266719309996355091226921569469352860e-1,
        1.34709217311473325928054001771706832761e-1,
    ),
    node(
        2.94392862701460198131126603103865566163e-1,
        0.0,
        1.42775938577060080797094273138717060886e-1,
    ),
    node(
        1.48874338981631210884826001129719984618e-1,
        2.95524224714752870173892994651338329421e-1,
        1.47739104901338491374841515972068045524e-1,
    ),
];
const G10K21_RULE: KronrodTable = KronrodTable {
    positive_nodes: G10K21_NODES,
    center_gauss_weight: 0.0,
    center_kronrod_weight: 1.49445554002916905664936468389821203745e-1,
};

const G15K31_NODES: &[KronrodNode] = &[
    node(
        9.98002298693397060285172840152271209073e-1,
        0.0,
        5.37747987292334898779205143012764981831e-3,
    ),
    node(
        9.87992518020485428489565718586612581147e-1,
        3.07532419961172683546283935772044177217e-2,
        1.50079473293161225383747630758072680946e-2,
    ),
    node(
        9.67739075679139134257347978784337225283e-1,
        0.0,
        2.54608473267153201868740010196533593973e-2,
    ),
    node(
        9.37273392400705904307758947710209471244e-1,
        7.03660474881081247092674164506673384667e-2,
        3.53463607913758462220379484783600481226e-2,
    ),
    node(
        8.97264532344081900882509656454495882832e-1,
        0.0,
        4.45897513247648766082272993732796902233e-2,
    ),
    node(
        8.48206583410427216200648320774216851366e-1,
        1.07159220467171935011869546685869303416e-1,
        5.34815246909280872653431472394302967716e-2,
    ),
    node(
        7.90418501442465932967649294817947346862e-1,
        0.0,
        6.20095678006706402851392309608029321904e-2,
    ),
    node(
        7.24417731360170047416186054613938009631e-1,
        1.39570677926154314447804794511028322521e-1,
        6.98541213187282587095200770991474757860e-2,
    ),
    node(
        6.50996741297416970533735895313274692547e-1,
        0.0,
        7.68496807577203788944327774826590067221e-2,
    ),
    node(
        5.70972172608538847537226737253910641238e-1,
        1.66269205816993933553200860481208811131e-1,
        8.30805028231330210382892472861037896016e-2,
    ),
    node(
        4.85081863640239680693655740232350612866e-1,
        0.0,
        8.85644430562117706472754436937743032123e-2,
    ),
    node(
        3.94151347077563369897207370981045468363e-1,
        1.86161000015562211026800561866422824506e-1,
        9.31265981708253212254868727473457185619e-2,
    ),
    node(
        2.99180007153168812166780024266388962662e-1,
        0.0,
        9.66427269836236785051799076275893351367e-2,
    ),
    node(
        2.01194093997434522300628303394596207813e-1,
        1.98431485327111576456118326443839324819e-1,
        9.91735987217919593323931734846031310596e-2,
    ),
    node(
        1.01142066918717499027074231447392338787e-1,
        0.0,
        1.00769845523875595044946662617569721916e-1,
    ),
];
const G15K31_RULE: KronrodTable = KronrodTable {
    positive_nodes: G15K31_NODES,
    center_gauss_weight: 2.02578241925561272880620199967519314839e-1,
    center_kronrod_weight: 1.01330007014791549017374792767492546771e-1,
};

const G20K41_NODES: &[KronrodNode] = &[
    node(
        9.98859031588277663838315576545863010000e-1,
        0.0,
        3.07358371852053150121829324603098748803e-3,
    ),
    node(
        9.93128599185094924786122388471320278223e-1,
        1.76140071391521183118619623518528163621e-2,
        8.60026985564294219866178795010234725213e-3,
    ),
    node(
        9.81507877450250259193342994720216944567e-1,
        0.0,
        1.46261692569712529837879603088683561639e-2,
    ),
    node(
        9.63971927277913791267666131197277221912e-1,
        4.06014298003869413310399522749321098791e-2,
        2.03883734612665235980102314327547051228e-2,
    ),
    node(
        9.40822633831754753519982722212443380274e-1,
        0.0,
        2.58821336049511588345050670961531429995e-2,
    ),
    node(
        9.12234428251325905867752441203298113049e-1,
        6.26720483341090635695065351870416063516e-2,
        3.12873067770327989585431193238007378878e-2,
    ),
    node(
        8.78276811252281976077442995113078466711e-1,
        0.0,
        3.66001697582007980305572407072110084875e-2,
    ),
    node(
        8.39116971822218823394529061701520685330e-1,
        8.32767415767047487247581432220462061002e-2,
        4.16688733279736862637883059368947380440e-2,
    ),
    node(
        7.95041428837551198350638833272787942959e-1,
        0.0,
        4.64348218674976747202318809261075168421e-2,
    ),
    node(
        7.46331906460150792614305070355641590311e-1,
        1.01930119817240435036750135480349876167e-1,
        5.09445739237286919327076700503449486648e-2,
    ),
    node(
        6.93237656334751384805490711845931533386e-1,
        0.0,
        5.51951053482859947448323724197773291948e-2,
    ),
    node(
        6.36053680726515025452836696226285936743e-1,
        1.18194531961518417312377377711382287005e-1,
        5.91114008806395723749672206485942171364e-2,
    ),
    node(
        5.75140446819710315342946036586425132814e-1,
        0.0,
        6.26532375547811680258701221742549805858e-2,
    ),
    node(
        5.10867001950827098004364050955250998425e-1,
        1.31688638449176626898494499748163134916e-1,
        6.58345971336184221115635569693979431472e-2,
    ),
    node(
        4.43593175238725103199992213492640107840e-1,
        0.0,
        6.86486729285216193456234118853678017155e-2,
    ),
    node(
        3.73706088715419560672548177024927237396e-1,
        1.42096109318382051329298325067164933035e-1,
        7.10544235534440683057903617232101674129e-2,
    ),
    node(
        3.01627868114913004320555356858592260615e-1,
        0.0,
        7.30306903327866674951894176589131127606e-2,
    ),
    node(
        2.27785851141645078080496195368574624743e-1,
        1.49172986472603746787828737001969436693e-1,
        7.45828754004991889865814183624875286161e-2,
    ),
    node(
        1.52605465240922675505220241022677527912e-1,
        0.0,
        7.57044976845566746595427753766165582634e-2,
    ),
    node(
        7.65265211334973337546404093988382110048e-2,
        1.52753387130725850698084331955097593492e-1,
        7.63778676720807367055028350380610018008e-2,
    ),
];
const G20K41_RULE: KronrodTable = KronrodTable {
    positive_nodes: G20K41_NODES,
    center_gauss_weight: 0.0,
    center_kronrod_weight: 7.66007119179996564450499015301017408279e-2,
};

const G25K51_NODES: &[KronrodNode] = &[
    node(
        9.99262104992609834193457486540340593705e-1,
        0.0,
        1.98738389233031592650785188284340988943e-3,
    ),
    node(
        9.95556969790498097908784946893901617258e-1,
        1.13937985010262879479029641132347736033e-2,
        5.56193213535671375804023690106552207018e-3,
    ),
    node(
        9.88035794534077247637331014577406227072e-1,
        0.0,
        9.47397338617415160720771052365532387165e-3,
    ),
    node(
        9.76663921459517511498315386479594067745e-1,
        2.63549866150321372619018152952991449360e-2,
        1.32362291955716748136564058469762380776e-2,
    ),
    node(
        9.61614986425842512418130033660167241692e-1,
        0.0,
        1.68478177091282982315166675363363158404e-2,
    ),
    node(
        9.42974571228974339414011169658470531905e-1,
        4.09391567013063126556234877116459536608e-2,
        2.04353711458828354565682922359389736788e-2,
    ),
    node(
        9.20747115281701561746346084546330631575e-1,
        0.0,
        2.40099456069532162200924891648810813929e-2,
    ),
    node(
        8.94991997878275368851042006782804954175e-1,
        5.49046959758351919259368915404733241601e-2,
        2.74753175878517378029484555178110786148e-2,
    ),
    node(
        8.65847065293275595448996969588340088203e-1,
        0.0,
        3.07923001673874888911090202152285856009e-2,
    ),
    node(
        8.33442628760834001421021108693569569461e-1,
        6.80383338123569172071871856567079685547e-2,
        3.40021302743293378367487952295512032257e-2,
    ),
    node(
        7.97873797998500059410410904994306569409e-1,
        0.0,
        3.71162714834155435603306253676198759960e-2,
    ),
    node(
        7.59259263037357630577282865204360976388e-1,
        8.01407003350010180132349596691113022902e-2,
        4.00838255040323820748392844670756464014e-2,
    ),
    node(
        7.17766406813084388186654079773297780598e-1,
        0.0,
        4.28728450201700494768957924394951611020e-2,
    ),
    node(
        6.73566368473468364485120633247622175883e-1,
        9.10282619829636498114972207028916533810e-2,
        4.55029130499217889098705847526603930437e-2,
    ),
    node(
        6.26810099010317412788122681624517881020e-1,
        0.0,
        4.79825371388367139063922557569147549836e-2,
    ),
    node(
        5.77662930241222967723689841612654067396e-1,
        1.00535949067050644202206890392685826988e-1,
        5.02776790807156719633252594334400844406e-2,
    ),
    node(
        5.26325284334719182599623778158010178037e-1,
        0.0,
        5.23628858064074758643667121378727148874e-2,
    ),
    node(
        4.73002731445714960522182115009192041332e-1,
        1.08519624474263653116093957050116619340e-1,
        5.42511298885454901445433704598756068261e-2,
    ),
    node(
        4.17885382193037748851814394594572487093e-1,
        0.0,
        5.59508112204123173082406863827473468203e-2,
    ),
    node(
        3.61172305809387837735821730127640667422e-1,
        1.14858259145711648339325545869555808641e-1,
        5.74371163615678328535826939395064719948e-2,
    ),
    node(
        3.03089538931107830167478909980339329200e-1,
        0.0,
        5.86896800223942079619741758567877641398e-2,
    ),
    node(
        2.43866883720988432045190362797451586406e-1,
        1.19455763535784772228178126512901047390e-1,
        5.97203403241740599790992919325618538354e-2,
    ),
    node(
        1.83718939421048892015969888759528415785e-1,
        0.0,
        6.05394553760458629453602675175654271623e-2,
    ),
    node(
        1.22864692610710396387359818808036805532e-1,
        1.22242442990310041688959518945851505835e-1,
        6.11285097170530483058590304162927119227e-2,
    ),
    node(
        6.15444830056850788865463923667966312817e-2,
        0.0,
        6.14711898714253166615441319652641775865e-2,
    ),
];
const G25K51_RULE: KronrodTable = KronrodTable {
    positive_nodes: G25K51_NODES,
    center_gauss_weight: 1.23176053726715451203902873079050142438e-1,
    center_kronrod_weight: 6.15808180678329350787598242400645531904e-2,
};

impl<S: RealScalar> FixedRule<S> for GaussKronrod<S> {
    #[inline(always)]
    fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        interval: Interval<S>,
        out: &mut O,
        workspace: &mut crate::FixedRuleWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<crate::IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: scioxide_core::AffineValue<Scale = S>,
        O: scioxide_core::AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        crate::fixed::apply_kronrod(self, interval, f, out, workspace, scope.into())
    }

    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "GaussKronrod", 1)
    }

    fn node_count(&self) -> usize {
        self.rule().evaluation_cost()
    }

    fn map_nodes_into(
        &self,
        interval: Interval<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<(), SciOxideError> {
        self.mapped_nodes(interval, destination).map(|_| ())
    }

    #[inline(always)]
    fn estimate_embedded(
        &self,
        interval: Interval<S>,
        nodes: &[FixedRuleNode<S>],
        sample: &dyn Fn(usize) -> S,
    ) -> Result<(S, crate::IntegrationErrorComponents<S>), SciOxideError> {
        crate::kernel::shape(self.node_count(), nodes.len())?;
        let estimate = self
            .rule
            .apply_samples::<0>(interval_map(interval)?.half_weight()?, |i| Ok(sample(i)))?;
        Ok((
            estimate.value(),
            crate::IntegrationErrorComponents::discretization_only(estimate.error())
                .with_roundoff(estimate.roundoff_floor()),
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::NumericScalar;

    const RULES: [(GaussKronrodRule, usize, usize); 5] = [
        (GaussKronrodRule::G7K15, 7, 23),
        (GaussKronrodRule::G10K21, 10, 31),
        (GaussKronrodRule::G15K31, 15, 47),
        (GaussKronrodRule::G20K41, 20, 61),
        (GaussKronrodRule::G25K51, 25, 77),
    ];

    fn table_moment(rule: &KronrodTable, degree: i32, gauss: bool) -> f64 {
        let center_weight = if gauss {
            rule.center_gauss_weight
        } else {
            rule.center_kronrod_weight
        };
        let center = if degree == 0 { center_weight } else { 0.0 };
        center
            + rule
                .positive_nodes
                .iter()
                .map(|node| {
                    let weight = if gauss {
                        node.gauss_weight
                    } else {
                        node.kronrod_weight
                    };
                    weight * (node.abscissa.powi(degree) + (-node.abscissa).powi(degree))
                })
                .sum::<f64>()
    }

    #[test]
    fn canonical_tables_have_valid_symmetry_weights_and_nesting() {
        for (pair, gauss_order, _) in RULES {
            let rule = pair.rule();
            assert_eq!(rule.evaluation_cost(), 2 * gauss_order + 1);
            assert_eq!(rule.positive_nodes.len(), gauss_order);
            assert!(
                rule.positive_nodes
                    .windows(2)
                    .all(|nodes| nodes[0].abscissa > nodes[1].abscissa)
            );
            assert!(rule.positive_nodes.iter().all(|node| {
                node.abscissa > 0.0
                    && node.abscissa < 1.0
                    && node.gauss_weight >= 0.0
                    && node.kronrod_weight > 0.0
                    && node.abscissa + (-node.abscissa) == 0.0
            }));
            assert!(
                rule.positive_nodes
                    .iter()
                    .enumerate()
                    .all(|(index, node)| (node.gauss_weight > 0.0) == (index % 2 == 1))
            );

            let center_is_gauss = rule.center_gauss_weight > 0.0;
            assert_eq!(center_is_gauss, gauss_order % 2 == 1);
            let nested_gauss_count = 2 * rule
                .positive_nodes
                .iter()
                .filter(|node| node.gauss_weight > 0.0)
                .count()
                + usize::from(center_is_gauss);
            assert_eq!(nested_gauss_count, gauss_order);

            let gauss_weight_sum = rule.center_gauss_weight
                + 2.0
                    * rule
                        .positive_nodes
                        .iter()
                        .map(|node| node.gauss_weight)
                        .sum::<f64>();
            let kronrod_weight_sum = rule.center_kronrod_weight
                + 2.0
                    * rule
                        .positive_nodes
                        .iter()
                        .map(|node| node.kronrod_weight)
                        .sum::<f64>();
            assert!((gauss_weight_sum - 2.0).abs() < 1.0e-14);
            assert!((kronrod_weight_sum - 2.0).abs() < 1.0e-14);
        }
    }

    #[test]
    fn canonical_tables_match_all_documented_polynomial_moments() {
        for (pair, gauss_order, kronrod_degree) in RULES {
            let rule = pair.rule();
            for degree in 0..=2 * gauss_order - 1 {
                let expected = if degree % 2 == 0 {
                    2.0 / (degree + 1) as f64
                } else {
                    0.0
                };
                let actual = table_moment(rule, degree as i32, true);
                assert!(
                    (actual - expected).abs() < 4.0e-14,
                    "Gauss moment {degree} for {pair:?}: {actual} != {expected}"
                );
            }
            for degree in 0..=kronrod_degree {
                let expected = if degree % 2 == 0 {
                    2.0 / (degree + 1) as f64
                } else {
                    0.0
                };
                let actual = table_moment(rule, degree as i32, false);
                assert!(
                    (actual - expected).abs() < 4.0e-14,
                    "Kronrod moment {degree} for {pair:?}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn canonical_error_refinement_scales_and_applies_the_roundoff_floor() {
        let raw_f64 = 1.0e-8_f64;
        let resasc_f64 = 1.0e-3_f64;
        let expected_f64 = resasc_f64 * (200.0 * raw_f64 / resasc_f64).powf(1.5);
        let (refined_f64, _) = refine_error(raw_f64, 2.0_f64, resasc_f64).unwrap();
        assert!((refined_f64 - expected_f64).abs() <= 4.0 * f64::EPSILON * expected_f64);
        let (_, floor_f64) = refine_error(0.0_f64, 2.0, 0.0).unwrap();
        assert_eq!(floor_f64, 100.0 * f64::EPSILON);

        let raw_f32 = 1.0e-5_f32;
        let resasc_f32 = 1.0e-2_f32;
        let expected_f32 = resasc_f32 * (200.0 * raw_f32 / resasc_f32).powf(1.5);
        let (refined_f32, _) = refine_error(raw_f32, 2.0_f32, resasc_f32).unwrap();
        assert!((refined_f32 - expected_f32).abs() <= 4.0 * f32::EPSILON * expected_f32);
        let (_, floor_f32) = refine_error(0.0_f32, 2.0, 0.0).unwrap();
        assert_eq!(floor_f32, 100.0 * f32::EPSILON);
    }

    #[test]
    fn admitted_stencils_remain_distinct_at_scalar_spacing_boundaries() {
        fn check<S: RealScalar>() {
            let mut admitted = 0;
            let mut checked = 0;
            for (pair, _, _) in RULES {
                let plan = GaussKronrod::<S>::new(pair).unwrap();
                for power in 0..19 {
                    let upper = S::ONE + S::EPSILON * S::try_from_usize_exact(1 << power).unwrap();
                    let interval = Interval::new(S::ONE, upper).unwrap();
                    let map = interval_map(interval).unwrap();
                    let mut points = [S::ZERO; 51];
                    let points = &mut points[..plan.node_count()];
                    if plan.rule().points(map, points).unwrap() {
                        admitted += 1;
                        assert!(plan.rule().native_frame(map).is_some());
                        let mut metadata = [FixedRuleNode::new(S::ZERO, S::ZERO, None); 51];
                        assert!(
                            plan.rule()
                                .fixed_nodes(map, &mut metadata[..points.len()])
                                .unwrap()
                        );
                        for (point, node) in points.iter().zip(metadata) {
                            assert_eq!(
                                point.into_f64().to_bits(),
                                node.point().into_f64().to_bits()
                            );
                            assert!(node.primary_weight() >= S::MIN_POSITIVE);
                            assert!(node.embedded_weight().unwrap().is_finite());
                        }
                        assert!(points[0] > interval.lower());
                        assert!(points[points.len() - 1] < interval.upper());
                        assert!(points.windows(2).all(|p| p[0] < p[1]));
                    } else {
                        checked += 1;
                        assert!(plan.rule().native_frame(map).is_none());
                    }
                }
            }
            assert!(admitted > 0 && checked > 0);
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn scalar_native_coordinates_and_evidence_match_the_complete_metadata_path() {
        fn check<S: RealScalar + NumericScalar<Real = S> + Default>() {
            use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
            for (pair, _, _) in RULES {
                let plan = GaussKronrod::<S>::new(pair).unwrap();
                let mut workspace = plan.workspace(1).unwrap();
                for (lower, upper) in [(0.0, 1.0), (-2.0, 4.0), (17.0, 33.0)] {
                    let interval =
                        Interval::new(S::from_f64_ieee(lower), S::from_f64_ieee(upper)).unwrap();
                    let mut nodes = [FixedRuleNode::new(S::ZERO, S::ZERO, None); 51];
                    let nodes = &mut nodes[..plan.node_count()];
                    let map = interval_map(interval).unwrap();
                    assert!(plan.rule().native_frame(map).is_some());
                    assert!(plan.rule().fixed_nodes(map, nodes).unwrap());
                    let function = |x: S| S::ONE / (S::ONE + x * x);
                    let (expected, evidence) = plan
                        .estimate_embedded(interval, nodes, &|i| function(nodes[i].point()))
                        .unwrap();
                    let calls = AtomicUsize::new(0);
                    let mut output = S::ZERO;
                    let report = plan
                        .integrate_into(
                            |point: S| {
                                let index = calls.fetch_add(1, Relaxed);
                                assert_eq!(
                                    point.into_f64().to_bits(),
                                    nodes[index].point().into_f64().to_bits()
                                );
                                Ok(function(point))
                            },
                            interval,
                            &mut output,
                            &mut workspace,
                            scioxide_core::ExecutionContext::sequential_native(),
                        )
                        .unwrap();
                    assert_eq!(output.into_f64().to_bits(), expected.into_f64().to_bits());
                    assert_eq!(report.error_components(), Some(&evidence));
                    assert_eq!(report.evaluations(), nodes.len());
                    assert_eq!(calls.load(Relaxed), nodes.len());
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    fn integrate<S: RealScalar + NumericScalar<Real = S> + Default>(
        pair: GaussKronrodRule,
        f: impl Fn(S) -> S + Sync,
        lower: S,
        upper: S,
    ) -> (S, crate::IntegrationReport<S>) {
        GaussKronrod::new(pair)
            .unwrap()
            .integrate(|x: S| Ok(f(x)), Interval::new(lower, upper).unwrap())
            .unwrap()
    }

    #[test]
    fn fixed_pairs_integrate_smooth_functions_in_f64_and_f32() {
        let (value, report) = integrate(GaussKronrodRule::G7K15, |x: f64| x, 0.0, 1.0);
        assert!((value - 0.5).abs() < 1e-15, "got {value}");
        assert!(report.error_estimate().unwrap() < 1e-14);
        assert_eq!(report.evaluations(), 15);
        let (value, report) = integrate(
            GaussKronrodRule::G7K15,
            f64::sin,
            0.0,
            core::f64::consts::PI,
        );
        assert!((value - 2.0).abs() < 1e-14, "got {value}");
        assert!(report.error_estimate().unwrap() < 1e-8);
        let (value, _) = integrate(GaussKronrodRule::G10K21, |x: f64| x * x, 0.0, 1.0);
        assert!((value - 1.0 / 3.0).abs() < 1e-15, "got {value}");
        let (value, report) = integrate(GaussKronrodRule::G25K51, |x: f32| x.exp(), 0.0, 1.0);
        assert!(
            (value - (core::f32::consts::E - 1.0)).abs() < 1e-6,
            "got {value}"
        );
        assert_eq!(report.evaluations(), 51);
    }
}
