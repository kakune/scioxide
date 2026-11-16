//! Regional quadrature rules without side effects. Sampling and output writing live in `solve`.
use super::solve::REGION_NODES;
use super::{EndpointWeight, FourierWeight, Weight, Weighting, moments};
use crate::IntegrationTermination;
use crate::adaptive::regional::{Preparation, Prepared};
use crate::kernel::{AffineMap1D, checked_intermediate, roundoff_floor};
use scioxide_core::backend::checked_mul;
use scioxide_core::{Interval, PromotingProduct, RealScalar, Scaled, SciOxideError};

/// Zeroth Cauchy moment `ln(|b-c|/|c-a|)`, retaining overflowing distances
/// and the small difference of nearly equal distances before taking `ln(1+x)`.
fn cauchy_mass<S: RealScalar>(interval: Interval<S>, pole: S) -> Result<S, SciOxideError> {
    let (lower, upper) = (interval.lower(), interval.upper());
    let left = Scaled::from_value(pole)?
        .sub(Scaled::from_value(lower)?)?
        .abs();
    let right = Scaled::from_value(upper)?
        .sub(Scaled::from_value(pole)?)?
        .abs();
    let mut difference = scioxide_core::RangeAccumulator::new();
    if pole <= lower {
        difference.add(upper)?;
        difference.add(-lower)?;
    } else if pole >= upper {
        difference.add(lower)?;
        difference.add(-upper)?;
    } else {
        difference.add(lower)?;
        difference.add(upper)?;
        difference.add(-pole)?;
        difference.add(-pole)?;
    }
    let relative = difference.finish_scaled()?.div(left)?;
    if relative.abs_le(Scaled::from_value(S::HALF)?) {
        Ok(relative.round()?.ln_1p())
    } else {
        right.div(left)?.ln_abs()
    }
}

/// One region's call-local payload: its validated geometry, node count and
/// the compact [`Source`] of its moment functional. Measurement maps the nodes,
/// evaluates the sample factors and forms the functional in place, so
/// preparation, preflight storage and paired children move only a few words.
#[derive(Clone, Copy)]
pub(crate) struct Rule<S: RealScalar> {
    interval: Interval<S>,
    map: AffineMap1D<S>,
    count: usize,
    moments: Option<Source<S>>,
}

/// The region-specific part of a moment rule. Together with the plan's
/// immutable unit-interval tables it determines the region's moments and its
/// nodal functional.
#[derive(Clone, Copy)]
enum Source<S: RealScalar> {
    /// The singular side, `ln(width)` for a logarithmic weight (one otherwise)
    /// and the physical scale `width^(exponent + 1)`.
    Endpoint {
        left: bool,
        logarithm: S,
        scale: MomentScale<S>,
    },
    /// The pole mapped to `[-1, 1]` and the zeroth moment from physical distances.
    Cauchy { pole: S, mass: S },
    /// The stored depth, its even/odd orientation factors, phase floor and the
    /// bound of `∫|w|` for the amplitude-resolution term.
    Fourier {
        depth: usize,
        factors: [S; 2],
        floor: S,
        resolution: S,
    },
}

/// Nodes, sample factors and contraction scale of one region, formed in
/// place where the region is measured.
#[derive(Clone, Copy)]
pub(crate) struct Stencil<S: RealScalar> {
    pub(crate) points: [S; REGION_NODES],
    pub(crate) count: usize,
    factors: [S; REGION_NODES],
    unit_factors: bool,
    /// Whether a positive sample factor fell below the normal range.
    flushed: bool,
    /// Bound of a Kronrod rule's absolute sensitivity to its sample values;
    /// moment rules take theirs from the formed functional.
    sensitivity: S,
    scale: PromotingProduct<S>,
}

impl<S: RealScalar> Stencil<S> {
    /// An unformed stencil, the target of [`Rule::stencil`].
    #[inline(always)]
    pub(crate) fn unformed() -> Self {
        Self {
            points: [S::ZERO; REGION_NODES],
            count: 0,
            factors: [S::ONE; REGION_NODES],
            unit_factors: true,
            flushed: false,
            sensitivity: S::ZERO,
            scale: PromotingProduct::ONE,
        }
    }
}

impl<S: RealScalar> Weighting<S> {
    /// Checks one region: its weight selection, geometry and stops. Nodes are
    /// validated here, before any callback, and mapped again when measured.
    pub(super) fn rule(
        &self,
        interval: Interval<S>,
        root: usize,
        depth: usize,
        providers_per_sample: usize,
    ) -> Result<Preparation<Rule<S>>, SciOxideError> {
        let Some(map) = AffineMap1D::from_interval(interval)? else {
            return Ok(Preparation::Stop(IntegrationTermination::GeometryFailure));
        };
        let source = if matches!(self.weight, Weight::Ordinary) {
            None
        } else {
            match self.source(interval, map, root, depth)? {
                Ok(source) => source,
                Err(stop) => return Ok(Preparation::Stop(stop)),
            }
        };
        let rule = Rule {
            interval,
            map,
            count: if source.is_some() {
                25
            } else {
                self.kronrod.rule().evaluation_cost()
            },
            moments: source,
        };
        // Moment nodes are admitted by a separating native frame; Kronrod
        // nodes, and moment nodes outside such a frame, are mapped once here.
        let admitted = source.is_some()
            && map
                .native_frame(self.cosines[1][11], S::ONE, S::ONE)
                .is_some_and(|(middle, half)| {
                    crate::kernel::separated(
                        half * (S::ONE - self.cosines[1][1]),
                        middle.abs() + half,
                    )
                });
        if !admitted {
            let mut points = [S::ZERO; REGION_NODES];
            let separated = rule.nodes(self, &mut points)?;
            // Moment nodes are closed and descending; Kronrod points are ascending.
            if !separated
                && !crate::kernel::mapped_geometry(
                    points[..rule.count].iter().copied(),
                    interval,
                    source.is_some(),
                )
            {
                return Ok(Preparation::Stop(IntegrationTermination::GeometryFailure));
            }
        }
        Ok(Preparation::Ready(Prepared {
            evaluations: rule.count,
            providers: checked_mul(
                rule.count,
                providers_per_sample,
                scioxide_core::quantity!("regional_providers"),
            )?,
            rule,
        }))
    }

    /// The moment source of a weighted region, `None` for Kronrod nodes, or the
    /// refinement stop of a strongly oscillating region beyond the moment levels.
    fn source(
        &self,
        interval: Interval<S>,
        map: AffineMap1D<S>,
        root: usize,
        depth: usize,
    ) -> Result<Result<Option<Source<S>>, IntegrationTermination>, SciOxideError> {
        let half = match map.native_half_span() {
            Some(half) => half,
            None => map.scaled_weight(S::ONE)?.round()?,
        };
        let middle = map.point(S::ZERO)?;
        Ok(Ok(match &self.weight {
            Weight::Ordinary => None,
            Weight::Endpoint {
                specification: w, ..
            } => {
                let is_left =
                    interval.lower() == self.interval.lower() && (w.alpha != S::ZERO || w.left_log);
                let is_right =
                    interval.upper() == self.interval.upper() && (w.beta != S::ZERO || w.right_log);
                // Only a region touching a singular physical endpoint uses moments. With
                // x = a + w s for the region width w and s = (1 + t)/2, the weight is
                // (x - a)^alpha = w^alpha s^alpha and dx = w ds, so the unit-width canonical
                // moments scale by w^(alpha + 1). The moment rule applies that scale in
                // extended range, so neither factor underflows or overflows on its own.
                // For a logarithmic weight ln(x - a) = ln(w) + ln(s) splits into ln(w) times
                // the power moment plus the canonical log moment. The opposite endpoint's
                // weight is smooth on this region and is folded into the sample factors.
                if is_left || is_right {
                    let (alpha, logarithm) = if is_left {
                        (w.alpha, w.left_log)
                    } else {
                        (w.beta, w.right_log)
                    };
                    let width = interval.width()?;
                    Some(Source::Endpoint {
                        left: is_left,
                        logarithm: if logarithm { width.ln() } else { S::ONE },
                        scale: MomentScale::power(width, alpha + S::ONE)?,
                    })
                } else {
                    None
                }
            }
            Weight::Cauchy(pole) => {
                // Principal-value moments integrate an interior pole analytically. As in
                // QUADPACK QAWC they also serve exterior poles within 1.1 half-widths of
                // the centre, where 1/(x - c) is too steep for the ordinary rule; farther
                // poles use the ordinary rule with an explicit factor (see
                // `Rule::stencil`). The decision uses physical positions: the canonical
                // position of a pole within about epsilon times the width of an endpoint
                // rounds to +-1. The zeroth moment always comes from physical distances.
                let (lower, upper) = (interval.lower(), interval.upper());
                let interior = lower < *pole && *pole < upper;
                let canonical = crate::kernel::diagnostic(
                    Scaled::from_value(*pole)?
                        .sub(Scaled::from_value(middle)?)?
                        .div(map.scaled_weight(S::ONE)?)?
                        .round(),
                )?;
                if interior
                    || (*pole != lower && *pole != upper && canonical.abs() < S::from_f64_ieee(1.1))
                {
                    Some(Source::Cauchy {
                        pole: if interior {
                            canonical.max(-S::ONE).min(S::ONE)
                        } else {
                            canonical
                        },
                        mass: cauchy_mass(interval, *pole)?,
                    })
                } else {
                    None
                }
            }
            Weight::Fourier {
                omega,
                weight,
                levels,
                table,
            } => {
                // Weak oscillation needs no prepared moments, at any depth.
                if omega.abs() * half > S::TWO {
                    if depth >= *levels {
                        return Ok(Err(IntegrationTermination::RefinementLimit));
                    }
                    match table.index(root, depth) {
                        Some(stored) => {
                            let (factors, floor) =
                                moments::fourier_orientation(*omega, *weight, half, middle)?;
                            Some(Source::Fourier {
                                depth: stored,
                                factors,
                                floor,
                                // |cos| and |sin| never exceed one, so the width bounds
                                // |w|'s mass.
                                resolution: (S::TWO * half).min(S::MAX),
                            })
                        }
                        None => None,
                    }
                } else {
                    None
                }
            }
        }))
    }
}

/// Sample factors `w_l(x) w_r(x)` of the physical endpoint weights, with
/// `w_l = (x - a)^alpha [ln(x - a)]` and `w_r = (b - x)^beta [ln(b - x)]`. The
/// moment side (`Some(true)` left, `Some(false)` right) contributes one. Each
/// power and logarithm runs as its own pass only when it is present, so no unused
/// transcendental is evaluated, and every factor keeps the product order
/// `(power * log) * (power * log)`. Returns whether a positive power product
/// fell below the normal range: such a weight has lost absolute precision and
/// the estimate then carries an underflow floor.
fn endpoint_factors<S: RealScalar>(
    weight: &EndpointWeight<S>,
    interval: Interval<S>,
    moment_side: Option<bool>,
    points: &[S],
    factors: &mut [S],
) -> Result<bool, SciOxideError> {
    let (a, b) = (interval.lower(), interval.upper());
    let mut left = [S::ONE; REGION_NODES];
    let mut right = [S::ONE; REGION_NODES];
    let (sample_left, sample_right) = (moment_side != Some(true), moment_side != Some(false));
    // A zero exponent's power is exactly one.
    if sample_left && weight.alpha != S::ZERO {
        for (power, &x) in left.iter_mut().zip(points) {
            *power = (x - a).powf(weight.alpha);
        }
    }
    if sample_right && weight.beta != S::ZERO {
        for (power, &x) in right.iter_mut().zip(points) {
            *power = (b - x).powf(weight.beta);
        }
    }
    let flushed = points
        .iter()
        .zip(left.iter().zip(&right))
        .any(|(&x, (&l, &r))| x - a > S::ZERO && b - x > S::ZERO && l * r < S::MIN_POSITIVE);
    if sample_left && weight.left_log {
        for (factor, &x) in left.iter_mut().zip(points) {
            *factor *= (x - a).ln();
        }
    }
    if sample_right && weight.right_log {
        for (factor, &x) in right.iter_mut().zip(points) {
            *factor *= (b - x).ln();
        }
    }
    for ((factor, &l), &r) in factors.iter_mut().zip(&left).zip(&right) {
        *factor = checked_intermediate(l * r)?;
    }
    Ok(flushed)
}

impl<S: RealScalar> Rule<S> {
    /// Whether measurement contracts a moment functional (else Kronrod nodes).
    pub(super) const fn has_moments(&self) -> bool {
        self.moments.is_some()
    }

    /// Map this rule's nodes into `points`: the 25 Chebyshev extrema
    /// (endpoints included, descending) for a moment rule, otherwise the open
    /// ascending Kronrod nodes. Returns whether a native frame separates them.
    #[inline(always)]
    fn nodes(
        &self,
        weighting: &Weighting<S>,
        points: &mut [S; REGION_NODES],
    ) -> Result<bool, SciOxideError> {
        if self.moments.is_none() {
            let rule = weighting.kronrod.rule();
            return rule.points(self.map, &mut points[..rule.evaluation_cost()]);
        }
        let cosines = &weighting.cosines;
        let native = self.map.native_frame(cosines[1][11], S::ONE, S::ONE);
        match native {
            Some((middle, half)) => {
                for (p, &node) in points.iter_mut().zip(&cosines[1]) {
                    *p = middle + half * node;
                }
            }
            None => {
                for (p, &node) in points[1..24].iter_mut().zip(&cosines[1][1..24]) {
                    *p = self.map.point(node)?;
                }
            }
        }
        // The extreme nodes are the endpoints themselves.
        points[0] = self.interval.upper();
        points[24] = self.interval.lower();
        Ok(native.is_some_and(|(middle, half)| {
            crate::kernel::separated(half * (S::ONE - cosines[1][1]), middle.abs() + half)
        }))
    }

    /// Form this region's nodes, sample factors and scale in `stencil`.
    /// Moment rules integrate their singular or oscillating weight
    /// analytically; Kronrod nodes carry the weight in their sample factors.
    #[inline(always)]
    pub(super) fn stencil(
        &self,
        weighting: &Weighting<S>,
        stencil: &mut Stencil<S>,
    ) -> Result<(), SciOxideError> {
        self.nodes(weighting, &mut stencil.points)?;
        stencil.count = self.count;
        let sampled = &stencil.points[..self.count];
        let factors = &mut stencil.factors;
        let mut flushed = false;
        let mut unit = true;
        match (&weighting.weight, self.moments) {
            (Weight::Cauchy(pole), None) => {
                for (factor, &x) in factors.iter_mut().zip(sampled) {
                    *factor = checked_intermediate(S::ONE / (x - *pole))?;
                }
                unit = false;
            }
            (Weight::Fourier { omega, weight, .. }, None) => {
                for (factor, &x) in factors.iter_mut().zip(sampled) {
                    let phase = checked_intermediate(*omega * x)?;
                    *factor = checked_intermediate(match weight {
                        FourierWeight::Cosine => phase.cos(),
                        FourierWeight::Sine => phase.sin(),
                    })?;
                }
                unit = false;
            }
            (
                Weight::Endpoint {
                    specification: w, ..
                },
                source,
            ) => {
                let side = match source {
                    Some(Source::Endpoint { left, .. }) => Some(left),
                    _ => None,
                };
                flushed = endpoint_factors(
                    w,
                    weighting.interval,
                    side,
                    sampled,
                    &mut factors[..self.count],
                )?;
                unit = false;
            }
            _ => {}
        }
        stencil.unit_factors = unit || factors[..self.count].iter().all(|&f| f == S::ONE);
        stencil.flushed = flushed;
        stencil.scale = self.map.half_weight()?;
        stencil.sensitivity = match (self.moments, &weighting.weight) {
            (Some(_), _) | (None, Weight::Ordinary) => S::ZERO,
            // Kronrod weights sum to two half-widths.
            (None, _) => {
                crate::kernel::diagnostic(stencil.scale.multiplied(S::TWO)?.abs().round())?
            }
        };
        Ok(())
    }

    /// This moment rule's functional, from its [`Source`] and the plan's
    /// unit-interval tables. A Fourier depth's half functionals are borrowed from
    /// the plan; other functionals are formed in place in `general`.
    #[inline(always)]
    pub(super) fn functional<'a>(
        &self,
        weighting: &'a Weighting<S>,
        general: &'a mut Option<MomentRule<S>>,
    ) -> Result<Functional<'a, S>, SciOxideError> {
        let cosines = &weighting.cosines;
        match (self.moments, &weighting.weight) {
            (
                Some(Source::Endpoint {
                    left,
                    logarithm,
                    scale,
                }),
                Weight::Endpoint {
                    left: lower,
                    right: upper,
                    ..
                },
            ) => {
                let rule = general.insert(MomentRule::UNFORMED);
                if left { lower } else { upper }.form(rule, logarithm, scale, cosines)?;
                Ok(Functional::Rule(rule))
            }
            (Some(Source::Cauchy { pole, mass }), Weight::Cauchy(_)) => {
                let rule = general.insert(MomentRule::UNFORMED);
                rule.set(&moments::cauchy(pole, mass)?, cosines, None, S::ZERO, None)?;
                Ok(Functional::Rule(rule))
            }
            (
                Some(Source::Fourier {
                    depth,
                    factors,
                    floor,
                    resolution,
                }),
                Weight::Fourier { table, .. },
            ) => table.depths[depth].functional(general, factors, floor, resolution, cosines),
            _ => Err(crate::kernel::invalid(
                scioxide_core::argument!("rule"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "moment source does not match the prepared weight",
            )),
        }
    }
}

/// A region's formed moment functional.
pub(super) enum Functional<'a, S: RealScalar> {
    /// Nodal (or extended) functionals formed for the region.
    Rule(&'a MomentRule<S>),
    /// A Fourier depth's half functionals, borrowed from the plan and
    /// oriented by the region's even/odd factors.
    Parity {
        parts: MomentParts<S>,
        halves: &'a Functionals<S>,
        factors: [S; 2],
    },
}

impl<S: RealScalar> Functional<'_, S> {
    #[inline(always)]
    fn estimate(
        &self,
        values: &[S],
        top: &[[S; 2]; 25],
        cosines: &[[S; 25]; 25],
    ) -> Result<[S; 4], SciOxideError> {
        match self {
            Self::Rule(rule) => rule.estimate(values, top, cosines),
            Self::Parity {
                parts,
                halves,
                factors,
            } => {
                crate::kernel::shape(25, values.len())?;
                match MomentRule::contract(parts, halves, Some(*factors), values, top) {
                    Some(estimate) => Ok(estimate),
                    None => MomentRule {
                        parts: *parts,
                        weights: **halves,
                        parity: Some(*factors),
                        extended: None,
                    }
                    .extended(values, top, cosines),
                }
            }
        }
    }

    fn sensitivity(&self) -> Result<S, SciOxideError> {
        match self {
            Self::Rule(rule) => rule.parts.sensitivity(),
            Self::Parity { parts, .. } => parts.sensitivity(),
        }
    }
}

impl<S: RealScalar> Stencil<S> {
    /// Value, discretization, arithmetic floor and propagated moment error;
    /// `functional` is the region's formed [`Rule::functional`].
    ///
    /// A sample factor or factored sample that leaves the normal range is wrong
    /// by at most `MIN_POSITIVE * (1 + |f|)`; the floor then adds that error
    /// times the rule's sensitivity, so an underflowed weighted region
    /// contributes its rounded value with a correct roundoff model.
    #[inline(always)]
    pub(super) fn estimate(
        &self,
        weighting: &Weighting<S>,
        functional: Option<&Functional<'_, S>>,
        samples: &[S],
        component: usize,
        dimension: usize,
    ) -> Result<[S; 4], SciOxideError> {
        // Samplers already validate callback rows. An identity smooth factor
        // needs no second multiplication or staging copy for scalar values.
        let mut transformed;
        let mut lost = self.flushed;
        let values = if self.unit_factors && dimension == 1 {
            &samples[..self.count]
        } else {
            transformed = [S::ZERO; REGION_NODES];
            for (i, v) in transformed[..self.count].iter_mut().enumerate() {
                let sample = samples[i * dimension + component];
                *v = checked_intermediate(sample * self.factors[i])?;
                lost |=
                    v.abs() < S::MIN_POSITIVE && sample != S::ZERO && self.factors[i] != S::ZERO;
            }
            &transformed[..self.count]
        };
        let mut estimate = match functional {
            None => {
                let rule = weighting.kronrod.rule();
                let e = rule.apply_values::<0>(self.scale, values)?;
                [e.value(), e.error(), e.roundoff_floor(), S::ZERO]
            }
            Some(moments) => moments.estimate(&values[..25], &weighting.top, &weighting.cosines)?,
        };
        if lost {
            let amplitude = (0..self.count).fold(S::ZERO, |largest, i| {
                largest.max(samples[i * dimension + component].abs())
            });
            let sensitivity = match functional {
                Some(moments) => moments.sensitivity()?,
                None => self.sensitivity,
            };
            let underflow = ((S::ONE + amplitude) * S::MIN_POSITIVE) * sensitivity;
            estimate[2] = (estimate[2] + underflow).min(S::MAX);
        }
        Ok(estimate)
    }
}

/// Unit-width moments of one singular endpoint, oriented toward it:
/// the right side maps `t -> -t`, multiplying odd orders by -1. When every
/// unit-width operand is native, the anchored nodal functionals of the power and
/// logarithmic parts are prepared once, so a region forms
/// `ln(width) * power + log` in one pass over its 50 weights instead of
/// transforming its moments.
pub(crate) struct EndpointSide<S: RealScalar> {
    left: bool,
    logarithm: bool,
    power: [S; 25],
    log: [S; 25],
    functionals: Option<[Functionals<S>; 2]>,
}

/// High and nested-difference nodal weights.
type Functionals<S> = [[S; 2]; 25];

impl<S: RealScalar> EndpointSide<S> {
    pub(crate) fn new(
        moments: moments::EndpointMoments<S>,
        left: bool,
        logarithm: bool,
        cosines: &[[S; 25]; 25],
    ) -> Self {
        let orient = |values: [S; 25]| -> [S; 25] {
            core::array::from_fn(|k| {
                if !left && k % 2 == 1 {
                    -values[k]
                } else {
                    values[k]
                }
            })
        };
        let power = orient(moments.power);
        let log = if logarithm {
            orient(moments.log)
        } else {
            [S::ZERO; 25]
        };
        let mut functionals = [[[S::ZERO; 2]; 25]; 2];
        let native = admitted(&power)
            && admitted(&log)
            && native_functional(&mut functionals[0], &power, 1, cosines)
            && native_functional(&mut functionals[1], &log, 1, cosines);
        Self {
            left,
            logarithm,
            power,
            log,
            functionals: native.then_some(functionals),
        }
    }

    /// Form the rule of a region whose logarithmic scale is `logarithm`:
    /// moments `logarithm * power + log` with errors
    /// `50 eps (|logarithm * power| + |log|)`.
    #[inline(always)]
    fn form(
        &self,
        functional: &mut MomentRule<S>,
        logarithm: S,
        scale: MomentScale<S>,
        cosines: &[[S; 25]; 25],
    ) -> Result<(), SciOxideError> {
        let fifty = S::from_f64_ieee(50.0) * S::EPSILON;
        let mut values = [S::ZERO; 25];
        let mut errors = values;
        let mut regular = true;
        for k in 0..25 {
            let p = self.power[k] * logarithm;
            let l = self.log[k];
            let magnitude = p.abs() + l.abs();
            values[k] = p + l;
            errors[k] = fifty * magnitude;
            // A rounding floor below the normal range takes the extended floor.
            regular &= values[k].is_finite()
                & errors[k].is_finite()
                & ((errors[k] >= S::MIN_POSITIVE) | (magnitude == S::ZERO));
        }
        if !regular {
            for k in 0..25 {
                let p = self.power[k] * logarithm;
                let l = self.log[k];
                values[k] = checked_intermediate(p + l)?;
                errors[k] = roundoff_floor(p.abs() + l.abs())?;
            }
        }
        let moments = moments::Moments::new(values, errors);
        let anchor = Some(if self.left { 24 } else { 0 });
        if moments.native
            && let Some([power, log]) = &self.functionals
        {
            let weights = &mut functional.weights;
            if self.logarithm {
                for ((weight, power), log) in weights.iter_mut().zip(power).zip(log) {
                    *weight = [logarithm * power[0] + log[0], logarithm * power[1] + log[1]];
                }
            } else {
                *weights = *power;
            }
            if admitted(weights.as_flattened()) {
                functional.set_native(&moments, anchor, S::ZERO, Some(scale));
                return Ok(());
            }
        }
        functional.set(&moments, cosines, anchor, S::ZERO, Some(scale))
    }
}

/// One stored strongly oscillating depth of an initial segment: its normalized
/// phase `|omega| h` and, when native, the parity tables of its moments.
///
/// Oriented moments are `factor_(k mod 2) m_k` (see
/// [`moments::fourier_orientation`]), so the region's nodal functional is
/// `factors[0] * even + factors[1] * odd`, where `even` and `odd` are the
/// functionals of the even and odd orders. Even orders are symmetric
/// in the node index and odd orders antisymmetric, so only nodes `0..=12` are
/// stored, and a region contracts pair sums and differences of its samples
/// against them, scaling the two parity totals by its factors. The moment
/// scalars of the error model follow in closed form from per-parity sums,
/// without orienting or transforming 25 moments. Depths without a native
/// table, and regions whose oriented moments could leave the native range,
/// recompute the unit-interval moments from the phase.
pub(crate) struct FourierDepth<S: RealScalar> {
    parameter: S,
    table: Option<ParityTable<S>>,
}

/// Half functionals and closed-form sums of one depth's unit-interval moments.
struct ParityTable<S: RealScalar> {
    /// Nodes `0..=12` of the even functional, then nodes `0..12` of the odd
    /// one, whose antisymmetry makes node 12 zero.
    halves: Functionals<S>,
    /// `m_0` and its error `e_0`.
    zeroth: [S; 2],
    /// Per parity: `[sum |m_k|, sum e_k^2, sum e_k |m_k|, sum m_k^2]`.
    sums: [[S; 4]; 2],
    /// Per parity: `[min |m_k|, max |m_k|, min e_k, max e_k]`.
    extremes: [[S; 4]; 2],
}

impl<S: RealScalar> FourierDepth<S> {
    pub(crate) fn new(parameter: S, cosines: &[[S; 25]; 25]) -> Result<Self, SciOxideError> {
        Ok(Self {
            parameter,
            table: ParityTable::new(&moments::fourier(parameter)?, cosines),
        })
    }

    /// The functional of a region oriented by `factors`: the borrowed half
    /// functionals when native, otherwise one formed in `general` from the
    /// recomputed unit-interval moments.
    #[inline(always)]
    fn functional<'a>(
        &'a self,
        general: &'a mut Option<MomentRule<S>>,
        factors: [S; 2],
        floor: S,
        resolution: S,
        cosines: &[[S; 25]; 25],
    ) -> Result<Functional<'a, S>, SciOxideError> {
        if let Some(table) = &self.table
            && let Some(parts) = table.parts(factors, floor, resolution)
        {
            return Ok(Functional::Parity {
                parts,
                halves: &table.halves,
                factors,
            });
        }
        let moments = moments::orient(moments::fourier(self.parameter)?, factors, floor)?;
        let rule = general.insert(MomentRule::UNFORMED);
        rule.set(&moments, cosines, None, resolution, None)?;
        Ok(Functional::Rule(rule))
    }
}

impl<S: RealScalar> ParityTable<S> {
    /// Tables of native, nonzero unit-interval moments with native functionals.
    fn new(moments: &moments::Moments<S>, cosines: &[[S; 25]; 25]) -> Option<Self> {
        if !moments.native
            || moments
                .values
                .iter()
                .chain(&moments.errors)
                .any(|&v| v == S::ZERO)
        {
            return None;
        }
        let parity = |odd: usize| -> [S; 25] {
            core::array::from_fn(|k| {
                if k % 2 == odd {
                    moments.values[k]
                } else {
                    S::ZERO
                }
            })
        };
        let mut even = [[S::ZERO; 2]; 25];
        let mut odd = even;
        if !native_functional(&mut even, &parity(0), 0, cosines)
            || !native_functional(&mut odd, &parity(1), 0, cosines)
        {
            return None;
        }
        let mut sums = [[S::ZERO; 4]; 2];
        let mut extremes = [[S::MAX, S::ZERO, S::MAX, S::ZERO]; 2];
        for (k, (&value, &error)) in moments.values.iter().zip(&moments.errors).enumerate() {
            let magnitude = value.abs();
            let [mass, errors, products, squares] = &mut sums[k % 2];
            *mass += magnitude;
            *errors += error * error;
            *products += error * magnitude;
            *squares += magnitude * magnitude;
            let [low, high, low_error, high_error] = &mut extremes[k % 2];
            *low = low.min(magnitude);
            *high = high.max(magnitude);
            *low_error = low_error.min(error);
            *high_error = high_error.max(error);
        }
        Some(Self {
            halves: core::array::from_fn(|i| if i < 13 { even[i] } else { odd[i - 13] }),
            zeroth: [moments.values[0], moments.errors[0]],
            sums,
            extremes,
        })
    }

    /// The scalars of the oriented moments `factors[k mod 2] m_k` with errors
    /// `|factor| e_k + (floor + 50 eps |factor|) |m_k|`, or `None` when an
    /// oriented moment or error could leave the native range.
    #[inline(always)]
    fn parts(&self, factors: [S; 2], floor: S, resolution: S) -> Option<MomentParts<S>> {
        let range = scioxide_core::backend::ShortReductionRange::new(25);
        let fifty = S::from_f64_ieee(50.0) * S::EPSILON;
        // Every oriented magnitude lies between its parity's extreme products;
        // both ends inside the native range admit the whole parity.
        let inside = |low: S, high: S| {
            (high == S::ZERO) | ((low > S::ZERO) & range.admits_all(&[low, high]))
        };
        let mut native = true;
        let mut squares = S::ZERO;
        let mut mass = S::ZERO;
        for ((&factor, sums), extremes) in factors.iter().zip(&self.sums).zip(&self.extremes) {
            let (scale, [low, high, low_error, high_error]) = (factor.abs(), *extremes);
            let shift = floor + fifty * scale;
            native &= inside(scale * low, scale * high)
                & inside(
                    scale * low_error + shift * low,
                    scale * high_error + shift * high,
                );
            let [magnitudes, errors, products, moments] = *sums;
            squares += (scale * scale) * errors
                + (S::TWO * scale * shift) * products
                + (shift * shift) * moments;
            mass += scale * magnitudes;
        }
        let error_norm = squares.sqrt();
        if !(native & error_norm.is_finite() & mass.is_finite()) {
            return None;
        }
        let constant = factors[0] * self.zeroth[0];
        Some(MomentParts {
            constant,
            zeroth_error: self.zeroth[1] * factors[0].abs()
                + floor * self.zeroth[0].abs()
                + fifty * constant.abs(),
            mass,
            variation: mass,
            error_norm: error_norm * (S::ONE + S::from_f64_ieee(32.0) * S::EPSILON),
            resolution,
            scale: None,
            anchor: 12,
        })
    }
}

/// Whether every value is zero or in the native short-reduction range.
#[inline(always)]
fn admitted<S: RealScalar>(values: &[S]) -> bool {
    scioxide_core::backend::ShortReductionRange::new(25).admits_all(values)
}

/// Write into `weights` the nodal weights of the degree-24 rule and of its
/// difference from the nested degree-12 rule for checked moments `values`, by
/// the transposed native DCT-I; `start = 1` drops `m_0`, which
/// endpoint-modified rules restore separately. False when a weight leaves the
/// native range.
#[inline(always)]
fn native_functional<S: RealScalar>(
    weights: &mut Functionals<S>,
    values: &[S; 25],
    start: usize,
    cosines: &[[S; 25]; 25],
) -> bool {
    let mut high_moments = *values;
    if start == 1 {
        high_moments[0] = S::ZERO;
    }
    let mut nested = [S::ZERO; 25];
    for (slot, &moment) in nested.iter_mut().step_by(2).zip(&high_moments[..13]) {
        *slot = moment;
    }
    let (_, high) = super::chebyshev::coefficients(&high_moments, cosines);
    let low = super::chebyshev::coarse_coefficients(&nested, cosines);
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight = [
            high[i],
            high[i] - if i % 2 == 0 { low[i / 2] } else { S::ZERO },
        ];
    }
    admitted(weights.as_flattened())
}

/// Physical scale of a unit-interval moment rule and its relative uncertainty.
#[derive(Clone, Copy)]
pub(crate) struct MomentScale<S: RealScalar> {
    factor: PromotingProduct<S>,
    uncertainty: S,
}

impl<S: RealScalar> MomentScale<S> {
    /// `base^exponent` for a positive base. A normal IEEE power keeps the
    /// elementary accuracy. Otherwise `exp(exponent * ln(base))` is formed in
    /// extended range; the rounding of its log adds a relative uncertainty of
    /// `4 * epsilon * (8 + |log|)`, which the arithmetic floor carries.
    pub(crate) fn power(base: S, exponent: S) -> Result<Self, SciOxideError> {
        let direct = base.powf(exponent);
        if direct.is_normal() {
            return Ok(Self {
                factor: PromotingProduct::from_value(direct)?,
                uncertainty: S::ZERO,
            });
        }
        let log = checked_intermediate(exponent * base.ln())?;
        Ok(Self {
            factor: PromotingProduct::from_scaled(Scaled::from_log(S::ONE, log)?),
            uncertainty: S::from_f64_ieee(4.0) * S::EPSILON * (S::from_f64_ieee(8.0) + log.abs()),
        })
    }
}

/// The transpose of the normalized DCT-I contracts the same degree-12/24
/// interpolants without transforming each sampled component. Only preparation
/// forms the nodal functionals, from moments or the plan's unit-interval tables.
/// The common error model is independent of all tolerances.
///
/// Both arithmetic representations stay inline, so promotion never allocates
/// during execution: moments outside the native range keep only their compact
/// form, which each (cold) extended replay transforms.
#[derive(Clone, Copy)]
pub(crate) struct MomentRule<S: RealScalar> {
    parts: MomentParts<S>,
    /// Native nodal weights of the high and nested-difference functionals,
    /// or the half functionals of a [`FourierDepth`] when `parity` is set.
    weights: Functionals<S>,
    /// Even and odd orientation factors of half functionals.
    parity: Option<[S; 2]>,
    /// Moments outside the native range, which replace `weights`.
    extended: Option<ExtendedMoments<S>>,
}

/// Scalars of a moment rule.
#[derive(Clone, Copy)]
pub(super) struct MomentParts<S: RealScalar> {
    /// `m_0`, restored as `f_anchor * m_0`.
    constant: S,
    /// Uncertainty of `m_0`; the others are folded into the error norm.
    zeroth_error: S,
    /// Absolute moment mass `sum |m_k|`.
    mass: S,
    /// Variation mass `sum_(k >= start) |m_k|` of the arithmetic floor;
    /// endpoint-modified rules start at one.
    variation: S,
    /// Inflated native `norm2(e_k)` over the orders `k >= start`.
    error_norm: S,
    /// Bound of `∫|w|` in unit-interval result units; zero disables the
    /// amplitude-resolution term.
    resolution: S,
    scale: Option<MomentScale<S>>,
    anchor: usize,
}

impl<S: RealScalar> MomentParts<S> {
    /// Bound of the physical result's absolute sensitivity to its samples: the
    /// constant restoration and both nodal functionals are bounded by five times
    /// the absolute moment mass.
    fn sensitivity(&self) -> Result<S, SciOxideError> {
        let canonical = S::from_f64_ieee(5.0) * self.mass;
        Ok(match self.scale {
            None => canonical.min(S::MAX),
            Some(scale) => crate::kernel::diagnostic(
                scale
                    .factor
                    .multiplied(canonical.min(S::MAX))?
                    .abs()
                    .round(),
            )?,
        })
    }
}

/// Moments of the high functional (`m_0` is zero when anchored) with their
/// error norm and variation mass in extended range.
#[derive(Clone, Copy)]
struct ExtendedMoments<S: RealScalar> {
    moments: [S; 25],
    error_norm: Scaled<S>,
    variation: Scaled<S>,
}

/// Extended-range nodal weights of the high and nested-difference functionals
/// of `high_moments`, by the direct DCT-I.
fn extended_functional<S: RealScalar>(
    high_moments: &[S; 25],
    cosines: &[[S; 25]; 25],
) -> Result<[[Scaled<S>; 2]; 25], SciOxideError> {
    let mut nested = [S::ZERO; 25];
    for (slot, &moment) in nested.iter_mut().step_by(2).zip(&high_moments[..13]) {
        *slot = moment;
    }
    let (_, high) = super::chebyshev::extended_coefficients(high_moments, cosines)?;
    let (low, _) = super::chebyshev::extended_coefficients(&nested, cosines)?;
    let mut weights = [[Scaled::ZERO; 2]; 25];
    for (i, slot) in weights.iter_mut().enumerate() {
        let high = high[i].scaled()?;
        let low = if i % 2 == 0 {
            low[i / 2].scaled()?
        } else {
            Scaled::ZERO
        };
        *slot = [high, high.sub(low)?];
    }
    Ok(weights)
}

/// Rows of the degree-23 and degree-24 coefficients of the interpolant at the
/// 25 Chebyshev extrema: `c_k = (2/24) sum'' f_j T_k(t_j)`, with the last
/// coefficient halved. Both rows annihilate constants.
pub(crate) fn top_coefficient_rows<S: RealScalar>(cosines: &[[S; 25]; 25]) -> [[S; 2]; 25] {
    let twelfth = S::ONE / S::from_f64_ieee(12.0);
    core::array::from_fn(|j| {
        let end = if j == 0 || j == 24 { S::HALF } else { S::ONE };
        [
            cosines[23][j] * twelfth * end,
            cosines[24][j] * twelfth * S::HALF * end,
        ]
    })
}

/// Frequency-independent amplitude-resolution bound `2 (|c_23| + |c_24|) mass`.
/// The interpolation error of a resolved amplitude is at most about twice its
/// remaining Chebyshev tail, which the top two coefficients overestimate; an
/// interior jump, kink or cusp keeps them large however fast the weight
/// oscillates. Callers pass the weight's absolute mass `∫|w|` and an `amplitude`
/// bounding the contracted samples, which bounds the rows' rounding: each
/// coefficient first discounts `32 epsilon amplitude`, so resolved samples whose
/// top coefficients are only rounding noise add nothing beyond the arithmetic floor.
#[inline(always)]
pub(crate) fn resolution_bound<S: RealScalar>(top: [S; 2], amplitude: S, mass: S) -> S {
    let noise = S::from_f64_ieee(32.0) * S::EPSILON * amplitude;
    let excess = (top[0].abs() - noise).max(S::ZERO) + (top[1].abs() - noise).max(S::ZERO);
    ((S::TWO * excess) * mass).min(S::MAX)
}

impl<S: RealScalar> MomentRule<S> {
    /// An unformed rule, the target of in-place formation.
    pub(crate) const UNFORMED: Self = Self {
        parts: MomentParts {
            constant: S::ZERO,
            zeroth_error: S::ZERO,
            mass: S::ZERO,
            variation: S::ZERO,
            error_norm: S::ZERO,
            resolution: S::ZERO,
            scale: None,
            anchor: 12,
        },
        weights: [[S::ZERO; 2]; 25],
        parity: None,
        extended: None,
    };

    pub(crate) fn new(
        moments: moments::Moments<S>,
        cosines: &[[S; 25]; 25],
        anchor: Option<usize>,
        resolution: S,
        scale: Option<MomentScale<S>>,
    ) -> Result<Self, SciOxideError> {
        let mut rule = Self::UNFORMED;
        rule.set(&moments, cosines, anchor, resolution, scale)?;
        Ok(rule)
    }

    /// Form the rule of `moments` in place: compensated DCT-I projections when the
    /// moments and their weights are native, otherwise the extended form.
    pub(crate) fn set(
        &mut self,
        moments: &moments::Moments<S>,
        cosines: &[[S; 25]; 25],
        anchor: Option<usize>,
        resolution: S,
        scale: Option<MomentScale<S>>,
    ) -> Result<(), SciOxideError> {
        use scioxide_core::ScaledAccumulator;
        let start = usize::from(anchor.is_some());
        if moments.native && native_functional(&mut self.weights, &moments.values, start, cosines) {
            self.set_native(moments, anchor, resolution, scale);
            return Ok(());
        }
        let mut high_moments = moments.values;
        if start == 1 {
            high_moments[0] = S::ZERO;
        }
        let mut square = ScaledAccumulator::ZERO;
        for &error in &moments.errors[start..] {
            let error = Scaled::from_value(error)?;
            square.add_product(error, error)?;
        }
        let mut variation = ScaledAccumulator::ZERO;
        for &moment in &moments.values[start..] {
            variation.add_value(moment.abs())?;
        }
        self.parts = MomentParts {
            constant: moments.values[0],
            zeroth_error: moments.errors[0],
            mass: moments.mass[0],
            variation: moments.mass[start],
            error_norm: S::ZERO,
            resolution,
            scale,
            anchor: anchor.unwrap_or(12),
        };
        self.parity = None;
        self.extended = Some(ExtendedMoments {
            moments: high_moments,
            error_norm: square.total()?.sqrt()?.mul(Scaled::from_value(
                S::ONE + S::from_f64_ieee(32.0) * S::EPSILON,
            )?)?,
            variation: variation.total()?,
        });
        Ok(())
    }

    /// The scalars of native `moments` whose checked weights are already in place.
    #[inline(always)]
    fn set_native(
        &mut self,
        moments: &moments::Moments<S>,
        anchor: Option<usize>,
        resolution: S,
        scale: Option<MomentScale<S>>,
    ) {
        let start = usize::from(anchor.is_some());
        let [square] = scioxide_core::backend::four_lane_sums(moments.errors.len() - start, |i| {
            let e = moments.errors[start + i];
            [e * e]
        });
        self.parts = MomentParts {
            constant: moments.values[0],
            zeroth_error: moments.errors[0],
            mass: moments.mass[0],
            variation: moments.mass[start],
            error_norm: square.sqrt() * (S::ONE + S::from_f64_ieee(32.0) * S::EPSILON),
            resolution,
            scale,
            anchor: anchor.unwrap_or(12),
        };
        self.parity = None;
        self.extended = None;
    }

    /// One contraction of the stored samples against both moment functionals.
    /// With g_j = f_j - f_anchor, DCT-I Parseval gives
    /// ||c(g)||_2 <= sqrt(sum'' g_j^2 / 12). Cauchy-Schwarz therefore bounds
    /// sum |c_k| e_k by that norm times ||e||_2. Norms carry 32-epsilon
    /// inflation; the anchor contributes |f_anchor| e_0 separately.
    ///
    /// The arithmetic floor is 128 eps max|g| sum|m_k| plus
    /// 50 eps |f_anchor m_0|. It covers preparation, nodal reductions and
    /// constant restoration, including cancellation. Endpoint-modified moments
    /// exclude m_0 from both the variation mass and error norm. Every arithmetic
    /// path uses this same model and the same fourfold nested difference. The
    /// discretization estimate is the larger of that difference and the
    /// amplitude-resolution bound, when enabled. A physical scale multiplies
    /// every component once, in extended range when its products leave the normal
    /// range, and its relative uncertainty times the value joins the floor.
    /// `top` holds the [`top_coefficient_rows`]; a disabled resolution
    /// term (zero mass) contributes exactly zero, so its rows are not contracted.
    #[inline(always)]
    pub(crate) fn estimate(
        &self,
        values: &[S],
        top: &[[S; 2]; 25],
        cosines: &[[S; 25]; 25],
    ) -> Result<[S; 4], SciOxideError> {
        crate::kernel::shape(25, values.len())?;
        if self.extended.is_none()
            && let Some(estimate) =
                Self::contract(&self.parts, &self.weights, self.parity, values, top)
        {
            return Ok(estimate);
        }
        self.extended(values, top, cosines)
    }

    /// The native contraction, or `None` when a sample, product or diagnostic
    /// leaves the native range.
    #[inline(always)]
    fn contract(
        parts: &MomentParts<S>,
        weights: &Functionals<S>,
        parity: Option<[S; 2]>,
        values: &[S],
        top: &[[S; 2]; 25],
    ) -> Option<[S; 4]> {
        let range = scioxide_core::backend::ShortReductionRange::new(25);
        let anchor = values[parts.anchor];
        let mut deviations = [S::ZERO; 25];
        for (g, &value) in deviations.iter_mut().zip(values) {
            *g = value - anchor;
        }
        if !(range.admits(anchor) & range.admits_all(&deviations)) {
            return None;
        }
        // Term i joins lane i % 4; endpoint squares are halved. Admitted
        // deviations are finite, so a comparison selects the larger magnitude.
        let larger = |a: S, b: S| if b > a { b } else { a };
        let mut amplitudes = [S::ZERO; 4];
        let mut squares = [S::ZERO; 4];
        for (block, chunk) in deviations[..24].chunks_exact(4).enumerate() {
            for (lane, &g) in chunk.iter().enumerate() {
                let half = if block == 0 && lane == 0 {
                    S::HALF
                } else {
                    S::ONE
                };
                amplitudes[lane] = larger(amplitudes[lane], g.abs());
                squares[lane] += (g * g) * half;
            }
        }
        let last = deviations[24];
        amplitudes[0] = larger(amplitudes[0], last.abs());
        squares[0] += (last * last) * S::HALF;
        let (high, difference, coefficients) = if let Some([even_factor, odd_factor]) = parity {
            // Half functionals: even orders weight the pair sums
            // `g_i + g_(24-i)` and odd orders the pair differences, as do the
            // even degree-24 and odd degree-23 coefficient rows; node 12 pairs
            // with itself and contributes no difference.
            let (even, odd) = weights.split_at(13);
            let [high_even, high_odd, low_even, low_odd, top23, top24] =
                scioxide_core::backend::four_lane_sums(12, |i| {
                    let (sum, difference) = (
                        deviations[i] + deviations[24 - i],
                        deviations[i] - deviations[24 - i],
                    );
                    [
                        even[i][0] * sum,
                        odd[i][0] * difference,
                        even[i][1] * sum,
                        odd[i][1] * difference,
                        top[i][0] * difference,
                        top[i][1] * sum,
                    ]
                });
            let middle = deviations[12];
            (
                even_factor * (high_even + even[12][0] * middle) + odd_factor * high_odd,
                even_factor * (low_even + even[12][1] * middle) + odd_factor * low_odd,
                (parts.resolution != S::ZERO).then_some([top23, top24 + top[12][1] * middle]),
            )
        } else if parts.resolution == S::ZERO {
            let [high, difference] = scioxide_core::backend::four_lane_sums(25, |i| {
                let g = deviations[i];
                [weights[i][0] * g, weights[i][1] * g]
            });
            (high, difference, None)
        } else {
            let [high, difference, top23, top24] =
                scioxide_core::backend::four_lane_sums(25, |i| {
                    let g = deviations[i];
                    [
                        weights[i][0] * g,
                        weights[i][1] * g,
                        top[i][0] * g,
                        top[i][1] * g,
                    ]
                });
            (high, difference, Some([top23, top24]))
        };
        let amplitude = amplitudes.into_iter().fold(S::ZERO, larger);
        let square = (squares[0] + squares[1]) + (squares[2] + squares[3]);
        let constant = anchor * parts.constant;
        let norm = (square / S::from_f64_ieee(12.0)).sqrt()
            * (S::ONE + S::from_f64_ieee(32.0) * S::EPSILON);
        let floor = (S::from_f64_ieee(128.0) * S::EPSILON * amplitude) * parts.variation
            + (S::from_f64_ieee(50.0) * S::EPSILON) * constant.abs();
        let nested = S::from_f64_ieee(4.0) * difference.abs();
        let canonical = [
            high + constant,
            match coefficients {
                Some(top) => nested.max(resolution_bound(top, amplitude, parts.resolution)),
                None => nested,
            },
            floor,
            norm * parts.error_norm + anchor.abs() * parts.zeroth_error,
        ];
        let Some(scale) = parts.scale else {
            return Some(canonical);
        };
        let factor = scale.factor.native()?;
        let magnitude = factor.abs();
        let scaled = [
            canonical[0] * factor,
            canonical[1] * magnitude,
            canonical[2] * magnitude,
            canonical[3] * magnitude,
        ];
        // Only normal (or exactly zero) products keep the floor's
        // relative accuracy; others replay in extended range.
        if !scaled
            .iter()
            .zip(&canonical)
            .all(|(&product, &operand)| product.is_normal() || operand == S::ZERO)
        {
            return None;
        }
        let [value, discretization, floor, uncertainty] = scaled;
        let floor = floor + scale.uncertainty * value.abs();
        floor
            .is_finite()
            .then_some([value, discretization, floor, uncertainty])
    }

    #[cold]
    #[inline(never)]
    fn extended(
        &self,
        values: &[S],
        top_rows: &[[S; 2]; 25],
        cosines: &[[S; 25]; 25],
    ) -> Result<[S; 4], SciOxideError> {
        use scioxide_core::ScaledAccumulator;
        let parts = &self.parts;
        // Retain original factors through extended replay; callbacks are never
        // repeated and neither products nor normalization lose tiny operands.
        let (functional, error_norm, variation) = match &self.extended {
            None => {
                let mut converted = [[Scaled::ZERO; 2]; 25];
                if let Some([even_factor, odd_factor]) = self.parity {
                    // Nodal weights `factor_even * even +- factor_odd * odd`.
                    let (even, odd) = self.weights.split_at(13);
                    let (even_factor, odd_factor) = (
                        Scaled::from_value(even_factor)?,
                        Scaled::from_value(odd_factor)?,
                    );
                    for (i, even) in even.iter().enumerate() {
                        for c in 0..2 {
                            let symmetric = even_factor.mul(Scaled::from_value(even[c])?)?;
                            let antisymmetric = match odd.get(i) {
                                Some(odd) => odd_factor.mul(Scaled::from_value(odd[c])?)?,
                                None => Scaled::ZERO,
                            };
                            converted[i][c] = symmetric.add(antisymmetric)?;
                            converted[24 - i][c] = symmetric.sub(antisymmetric)?;
                        }
                    }
                } else {
                    for (slot, weight) in converted.iter_mut().zip(&self.weights) {
                        *slot = [
                            Scaled::from_value(weight[0])?,
                            Scaled::from_value(weight[1])?,
                        ];
                    }
                }
                (
                    converted,
                    Scaled::from_value(parts.error_norm)?,
                    Scaled::from_value(parts.variation)?,
                )
            }
            Some(extended) => (
                extended_functional(&extended.moments, cosines)?,
                extended.error_norm,
                extended.variation,
            ),
        };
        let anchor = Scaled::from_value(values[parts.anchor])?;
        let mut high = ScaledAccumulator::ZERO;
        let mut difference = ScaledAccumulator::ZERO;
        let mut top = [ScaledAccumulator::ZERO; 2];
        let mut square = ScaledAccumulator::ZERO;
        let mut amplitude = Scaled::ZERO;
        for (i, &value) in values.iter().enumerate() {
            let value = Scaled::from_value(value)?;
            let g = value.sub(anchor)?;
            let weights = functional[i];
            high.add_product(weights[0], g)?;
            difference.add_product(weights[1], g)?;
            for (sum, &row) in top.iter_mut().zip(&top_rows[i]) {
                sum.add_product(Scaled::from_value(row)?, g)?;
            }
            square.add_scaled(
                g.square()?
                    .shifted(if i == 0 || i == 24 { -1 } else { 0 })?,
            )?;
            if g.abs_cmp(amplitude).is_gt() {
                amplitude = g.abs();
            }
        }
        let constant = anchor.mul(Scaled::from_value(parts.constant)?)?;
        let norm = square
            .total()?
            .div(Scaled::from_value(S::from_f64_ieee(12.0))?)?
            .sqrt()?
            .mul(Scaled::from_value(
                S::ONE + S::from_f64_ieee(32.0) * S::EPSILON,
            )?)?;
        let uncertainty = norm
            .mul(error_norm)?
            .add(anchor.abs().mul(Scaled::from_value(parts.zeroth_error)?)?)?;
        let floor = amplitude
            .mul(variation)?
            .mul(Scaled::from_value(S::from_f64_ieee(128.0) * S::EPSILON)?)?
            .add(
                constant
                    .abs()
                    .mul(Scaled::from_value(S::from_f64_ieee(50.0) * S::EPSILON)?)?,
            )?;
        let nested = difference.total()?.abs().shifted(2)?;
        let noise = amplitude.mul(Scaled::from_value(S::from_f64_ieee(32.0) * S::EPSILON)?)?;
        let mut excess = Scaled::ZERO;
        for sum in top {
            let coefficient = sum.total()?.abs().sub(noise)?;
            if !coefficient.is_sign_negative() {
                excess = excess.add(coefficient)?;
            }
        }
        let resolution = excess
            .shifted(1)?
            .mul(Scaled::from_value(parts.resolution)?)?;
        let discretization = if resolution.abs_cmp(nested).is_gt() {
            resolution
        } else {
            nested
        };
        let mut value = high.total()?.add(constant)?;
        let mut diagnostics = [discretization, floor, uncertainty];
        if let Some(scale) = parts.scale {
            let factor = scale.factor.scaled()?;
            value = value.mul(factor)?;
            for diagnostic in &mut diagnostics {
                *diagnostic = diagnostic.mul(factor.abs())?;
            }
            diagnostics[1] =
                diagnostics[1].add(value.abs().mul(Scaled::from_value(scale.uncertainty)?)?)?;
        }
        let project = |value: Scaled<S>| -> Result<S, SciOxideError> {
            let rounded = crate::kernel::diagnostic(value.round())?;
            Ok(if value.is_zero() {
                S::ZERO
            } else {
                rounded.max(S::MIN_SUBNORMAL)
            })
        };
        let [discretization, floor, uncertainty] = diagnostics;
        Ok([
            value.round()?,
            project(discretization)?,
            project(floor)?,
            project(uncertainty)?,
        ])
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod contraction_tests {
    use super::*;
    use crate::weighted::number;

    #[test]
    fn parseval_uncertainty_covers_known_polynomial_coefficients() {
        fn check<S: RealScalar>() -> Result<(), SciOxideError> {
            let cosines = moments::cosines::<S>()?;
            let rows = top_coefficient_rows(&cosines);
            let mut coefficients = [S::ZERO; 25];
            for (k, value) in [(0, 1.0), (1, -0.5), (2, 0.25), (7, 0.125), (24, 0.0625)] {
                coefficients[k] = S::from_f64_ieee(value);
            }
            let values = core::array::from_fn::<_, 25, _>(|j| {
                coefficients
                    .iter()
                    .zip(&cosines)
                    .fold(S::ZERO, |sum, (&c, row)| sum + c * row[j])
            });
            let ordinary = core::array::from_fn::<_, 25, _>(|k| {
                if k % 2 == 1 {
                    S::ZERO
                } else {
                    S::TWO / (S::ONE - S::from_f64_ieee((k * k) as f64))
                }
            });
            let errors =
                core::array::from_fn::<_, 25, _>(|k| S::from_f64_ieee((k + 1) as f64) * S::EPSILON);
            for anchored in [false, true] {
                let mut modified = ordinary;
                if anchored {
                    for (k, moment) in modified.iter_mut().enumerate().skip(1) {
                        *moment -= if k % 2 == 0 { S::TWO } else { -S::TWO };
                    }
                }
                let rule = MomentRule::new(
                    moments::Moments::new(modified, errors),
                    &cosines,
                    anchored.then_some(24),
                    S::ZERO,
                    None,
                )?;
                let estimate = rule.estimate(&values, &rows, &cosines)?;
                let mut uncertainty = coefficients
                    .iter()
                    .zip(&errors)
                    .skip(usize::from(anchored))
                    .fold(S::ZERO, |sum, (&c, &e)| sum + c.abs() * e);
                if anchored {
                    uncertainty += values[24].abs() * errors[0];
                }
                assert!(estimate[3] >= uncertainty);
                let expected = coefficients
                    .iter()
                    .zip(ordinary)
                    .fold(S::ZERO, |sum, (&c, m)| sum + c * m);
                assert!((estimate[0] - expected).abs() <= estimate[2]);
                let small = S::MIN_POSITIVE.sqrt() * S::EPSILON;
                let large = S::ONE / small;
                let extended = MomentRule::new(
                    moments::Moments::new(modified.map(|v| v * large), errors.map(|e| e * large)),
                    &cosines,
                    anchored.then_some(24),
                    S::ZERO,
                    None,
                )?
                .estimate(&values.map(|v| v * small), &rows, &cosines)?;
                assert!(extended[3] >= uncertainty);
                assert!((extended[0] - expected).abs() <= extended[2]);
                for c in [2, 3] {
                    assert!(
                        (estimate[c] - extended[c]).abs()
                            <= S::from_f64_ieee(128.0) * S::EPSILON * estimate[c]
                    );
                }
            }
            Ok(())
        }
        check::<f32>().unwrap();
        check::<f64>().unwrap();
    }

    #[test]
    fn extended_transform_preserves_subnormal_samples_and_wide_products() {
        fn check<S: RealScalar>() -> Result<(), SciOxideError> {
            let cosines = moments::cosines()?;
            let rows = top_coefficient_rows(&cosines);
            for (amplitude, mass) in [
                (S::MIN_SUBNORMAL, S::ONE),
                (S::MIN_SUBNORMAL, S::MAX),
                (S::MAX, S::MIN_POSITIVE),
            ] {
                let mut values = [S::ZERO; 25];
                values[0] = mass;
                let moments = moments::Moments::new(values, [S::ZERO; 25]);
                let estimate = MomentRule::new(moments, &cosines, None, S::ZERO, None)?.estimate(
                    &[amplitude; 25],
                    &rows,
                    &cosines,
                )?;
                let expected = amplitude * mass;
                assert!(
                    (estimate[0] - expected).abs()
                        <= S::from_f64_ieee(32.0) * S::EPSILON * expected.abs()
                );
                assert!(estimate[2] > S::ZERO && estimate[2].is_finite());
                assert_eq!(estimate[3], S::ZERO);
            }
            let mut values = [S::ZERO; 25];
            values[0] = S::MAX;
            let moments = moments::Moments::new(values, [S::ZERO; 25]);
            assert!(
                MomentRule::new(moments, &cosines, None, S::ZERO, None)?
                    .estimate(&[S::MAX; 25], &rows, &cosines)
                    .is_err()
            );
            Ok(())
        }
        check::<f32>().unwrap();
        check::<f64>().unwrap();
    }

    #[test]
    fn identical_regions_keep_all_estimate_bits_across_tolerance_policies() {
        use crate::{EndpointWeight, WeightedQuadratureOptions, WeightedQuadraturePlan};
        use scioxide_core::{ErrorTolerance, NormSelection};
        fn check<S: RealScalar>() -> Result<(), SciOxideError> {
            let interval = Interval::new(S::ZERO, S::ONE)?;
            let tight = S::EPSILON;
            let mut reference = [None; 4];
            for (absolute, relative) in [(S::ONE, S::ZERO), (S::ZERO, tight), (tight, tight)] {
                let options =
                    WeightedQuadratureOptions::new(
                        crate::SubdivisionOptions::default().with_tolerance(
                            ErrorTolerance::scalar(absolute, relative, NormSelection::Infinity)?,
                        ),
                    );
                let plans = [
                    WeightedQuadraturePlan::endpoint(
                        interval,
                        &[],
                        EndpointWeight::new(-S::HALF, S::ZERO, false, false)?,
                        options.clone(),
                    )?,
                    WeightedQuadraturePlan::endpoint(
                        interval,
                        &[],
                        EndpointWeight::new(S::HALF, S::ZERO, true, false)?,
                        options.clone(),
                    )?,
                    WeightedQuadraturePlan::principal_value(interval, S::HALF, options.clone())?,
                    WeightedQuadraturePlan::fourier(
                        interval,
                        &[],
                        number::<S>(20)?,
                        FourierWeight::Cosine,
                        options,
                    )?,
                ];
                for (index, plan) in plans.iter().enumerate() {
                    let weighting = &plan.weighting;
                    let region = weighting.segments[0];
                    let Preparation::Ready(prepared) = weighting.rule(region, 0, 0, 1)? else {
                        return Err(crate::kernel::invalid(
                            scioxide_core::argument!("region"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "test region was not prepared",
                        ));
                    };
                    let rule = prepared.rule;
                    assert!(rule.has_moments());
                    let mut general = None;
                    let functional = rule.functional(weighting, &mut general)?;
                    let mut stencil = Stencil::unformed();
                    rule.stencil(weighting, &mut stencil)?;
                    let mut samples = [S::ZERO; REGION_NODES];
                    for (sample, &point) in samples.iter_mut().zip(&stencil.points[..stencil.count])
                    {
                        *sample = point.exp();
                    }
                    let actual = stencil
                        .estimate(weighting, Some(&functional), &samples, 0, 1)?
                        .map(|v| v.into_f64().to_bits());
                    if let Some(expected) = reference[index] {
                        assert_eq!(actual, expected);
                    } else {
                        reference[index] = Some(actual);
                    }
                    let mut vector = [S::ZERO; REGION_NODES * 2];
                    for (row, &value) in vector.chunks_exact_mut(2).zip(&samples) {
                        row.copy_from_slice(&[value, -value]);
                    }
                    assert_eq!(
                        stencil
                            .estimate(weighting, Some(&functional), &vector, 0, 2)?
                            .map(|v| v.into_f64().to_bits()),
                        actual
                    );
                }
            }
            Ok(())
        }
        check::<f32>().unwrap();
        check::<f64>().unwrap();
    }

    #[test]
    fn prepared_moment_functionals_match_analytic_chebyshev_moments() {
        fn check<S: RealScalar>() {
            let cosines = moments::cosines().unwrap();
            let rows = top_coefficient_rows(&cosines);
            let ordinary: [S; 25] = core::array::from_fn(|k| {
                if k % 2 == 1 {
                    S::ZERO
                } else {
                    S::TWO / (S::ONE - S::try_from_usize_exact(k * k).unwrap())
                }
            });
            for anchored in [false, true] {
                let mut modified = ordinary;
                if anchored {
                    for (k, moment) in modified.iter_mut().enumerate().skip(1) {
                        *moment -= if k % 2 == 0 { S::TWO } else { -S::TWO };
                    }
                }
                let moments = moments::Moments::new(modified, [S::ZERO; 25]);
                for (degree, values) in cosines.iter().enumerate() {
                    let [value, difference, floor, uncertainty] =
                        MomentRule::new(moments, &cosines, anchored.then_some(24), S::ZERO, None)
                            .unwrap()
                            .estimate(values, &rows, &cosines)
                            .unwrap();
                    let alias = if degree <= 12 { degree } else { 24 - degree };
                    let allowance = S::try_from_usize_exact(512).unwrap() * S::EPSILON;
                    assert!((value - ordinary[degree]).abs() <= allowance);
                    assert!(
                        (difference
                            - (S::TWO * S::TWO) * (ordinary[degree] - ordinary[alias]).abs())
                        .abs()
                            <= allowance
                    );
                    assert!((value - ordinary[degree]).abs() <= floor);
                    assert_eq!(uncertainty, S::ZERO);
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn canonical_tables_match_per_region_moment_transforms() {
        fn agree<S: RealScalar>(
            table: &MomentRule<S>,
            general: &MomentRule<S>,
        ) -> Result<(), SciOxideError> {
            let cosines = moments::cosines::<S>()?;
            let rows = top_coefficient_rows(&cosines);
            let relative = S::from_f64_ieee(256.0) * S::EPSILON;
            let smooth: [S; 25] = core::array::from_fn(|j| (cosines[1][j] * S::HALF).exp());
            let step: [S; 25] = core::array::from_fn(|j| if j <= 12 { S::ONE } else { -S::HALF });
            for samples in [smooth, step] {
                let a = table.estimate(&samples, &rows, &cosines)?;
                let b = general.estimate(&samples, &rows, &cosines)?;
                let floor = a[2] + b[2];
                assert!((a[0] - b[0]).abs() <= floor, "{a:?} versus {b:?}");
                assert!(
                    (a[1] - b[1]).abs() <= S::from_f64_ieee(8.0) * floor + relative * a[1].abs(),
                    "{a:?} versus {b:?}"
                );
                for c in [2, 3] {
                    assert!(
                        (a[c] - b[c]).abs() <= relative * a[c].abs(),
                        "{a:?} versus {b:?}"
                    );
                }
            }
            Ok(())
        }
        fn check<S: RealScalar>() -> Result<(), SciOxideError> {
            let cosines = moments::cosines::<S>()?;
            // Fourier depths: half functionals and closed-form scalars against
            // oriented, transformed canonical moments, in both regimes.
            for (parameter, middle, sign) in
                [(3.0, 0.3, 1.0), (25.0, -2.0, -1.0), (400.0, 7.5, 1.0)]
            {
                let parameter = S::from_f64_ieee(parameter);
                let depth = FourierDepth::new(parameter, &cosines)?;
                assert!(depth.table.is_some());
                for weight in [FourierWeight::Cosine, FourierWeight::Sine] {
                    let half = S::from_f64_ieee(0.25);
                    let omega = S::from_f64_ieee(sign) * parameter / half;
                    let (factors, floor) = moments::fourier_orientation(
                        omega,
                        weight,
                        half,
                        S::from_f64_ieee(middle),
                    )?;
                    let resolution = S::TWO * half;
                    let mut unused = None;
                    let Functional::Parity {
                        parts,
                        halves,
                        factors,
                    } = depth.functional(&mut unused, factors, floor, resolution, &cosines)?
                    else {
                        return Err(crate::kernel::invalid(
                            scioxide_core::argument!("depth"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "a native depth forms its parity functional",
                        ));
                    };
                    let table = MomentRule {
                        parts,
                        weights: *halves,
                        parity: Some(factors),
                        extended: None,
                    };
                    let oriented = moments::orient(moments::fourier(parameter)?, factors, floor)?;
                    let general = MomentRule::new(oriented, &cosines, None, resolution, None)?;
                    assert!(general.parity.is_none() && general.extended.is_none());
                    agree(&table, &general)?;
                }
            }
            // Endpoint sides: `ln(width) * power + log` functionals against the
            // transform of the same region moments.
            for (alpha, left, logarithm, width) in [
                (-0.5, true, true, 0.5),
                (0.75, false, true, 3.0),
                (-0.25, false, false, 1e-3),
            ] {
                let alpha = S::from_f64_ieee(alpha);
                let side = EndpointSide::new(moments::endpoint(alpha)?, left, logarithm, &cosines);
                assert!(side.functionals.is_some());
                let width = S::from_f64_ieee(width);
                let scale = MomentScale::power(width, alpha + S::ONE)?;
                let log = if logarithm { width.ln() } else { S::ONE };
                let mut table = MomentRule::UNFORMED;
                side.form(&mut table, log, scale, &cosines)?;
                let values = core::array::from_fn(|k| side.power[k] * log + side.log[k]);
                let errors = core::array::from_fn(|k| {
                    S::from_f64_ieee(50.0)
                        * S::EPSILON
                        * ((side.power[k] * log).abs() + side.log[k].abs())
                });
                let anchor = Some(if left { 24 } else { 0 });
                let general = MomentRule::new(
                    moments::Moments::new(values, errors),
                    &cosines,
                    anchor,
                    S::ZERO,
                    Some(scale),
                )?;
                agree(&table, &general)?;
            }
            Ok(())
        }
        check::<f32>().unwrap();
        check::<f64>().unwrap();
    }

    #[test]
    fn top_coefficient_rows_isolate_degrees_23_and_24() {
        fn check<S: RealScalar>() {
            let cosines = moments::cosines::<S>().unwrap();
            let rows = top_coefficient_rows(&cosines);
            let allowance = S::from_f64_ieee(64.0) * S::EPSILON;
            for (degree, values) in cosines.iter().enumerate() {
                for c in 0..2 {
                    let coefficient = values
                        .iter()
                        .zip(&rows)
                        .fold(S::ZERO, |sum, (&value, row)| sum + value * row[c]);
                    let expected = if degree == 23 + c { S::ONE } else { S::ZERO };
                    assert!(
                        (coefficient - expected).abs() <= allowance,
                        "degree {degree}, row {c}: {coefficient:?}"
                    );
                }
            }
            assert_eq!(
                resolution_bound([S::ONE, -S::ONE], S::ONE, S::ZERO),
                S::ZERO
            );
            assert_eq!(
                resolution_bound([S::HALF, -S::HALF], S::ZERO, S::ONE),
                S::TWO
            );
            assert_eq!(resolution_bound([S::MAX, S::MAX], S::ZERO, S::MAX), S::MAX);
            // Coefficients at the rounding level of their samples add nothing.
            let noise = S::from_f64_ieee(32.0) * S::EPSILON;
            assert_eq!(resolution_bound([noise, -noise], S::ONE, S::ONE), S::ZERO);
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn strong_fourier_rules_bound_unresolved_amplitudes_independently_of_frequency() {
        fn check<S: RealScalar>() -> Result<(), SciOxideError> {
            let cosines = moments::cosines::<S>()?;
            let rows = top_coefficient_rows(&cosines);
            for phase in [30.0, 1e3, 1e5] {
                let stored = moments::fourier(S::from_f64_ieee(phase))?;
                let half = S::HALF;
                let oriented = moments::orient_fourier(
                    stored,
                    S::from_f64_ieee(2.0 * phase),
                    FourierWeight::Sine,
                    half,
                    half,
                )?;
                let disabled = MomentRule::new(oriented, &cosines, None, S::ZERO, None)?;
                let enabled = MomentRule::new(oriented, &cosines, None, S::TWO * half, None)?;
                // A smooth amplitude keeps every estimate bit.
                let smooth: [S; 25] = core::array::from_fn(|j| cosines[1][j].exp());
                assert_eq!(
                    enabled
                        .estimate(&smooth, &rows, &cosines)?
                        .map(|v| v.into_f64().to_bits()),
                    disabled
                        .estimate(&smooth, &rows, &cosines)?
                        .map(|v| v.into_f64().to_bits())
                );
                // A jump between nodes 12 and 13 keeps its top coefficients at
                // every frequency, while the nested difference decays with it.
                let step: [S; 25] =
                    core::array::from_fn(|j| if j <= 12 { S::ONE } else { S::ZERO });
                let top = [0, 1].map(|c| {
                    step.iter()
                        .zip(&rows)
                        .fold(S::ZERO, |sum, (&value, row)| sum + value * row[c])
                });
                let bound = resolution_bound(top, S::ONE, S::ONE);
                assert!(bound > S::from_f64_ieee(0.03), "{bound:?}");
                let estimate = enabled.estimate(&step, &rows, &cosines)?;
                let rounding = S::from_f64_ieee(64.0) * S::EPSILON * bound;
                assert!(
                    estimate[1] >= bound - rounding,
                    "phase {phase}: {estimate:?}"
                );
                assert!(estimate[1] >= disabled.estimate(&step, &rows, &cosines)?[1]);
                let scaled =
                    enabled.estimate(&step.map(|v| v * S::MIN_POSITIVE), &rows, &cosines)?;
                assert!(scaled[1] > S::ZERO && scaled[1] <= estimate[1]);
            }
            Ok(())
        }
        check::<f32>().unwrap();
        check::<f64>().unwrap();
    }

    #[test]
    fn extended_moment_scales_keep_tiny_and_huge_regions_representable() {
        fn check<S: RealScalar>() -> Result<(), SciOxideError> {
            let cosines = moments::cosines::<S>()?;
            let rows = top_coefficient_rows(&cosines);
            let mut values = [S::ZERO; 25];
            values[0] = S::ONE;
            let unit = moments::Moments::new(values, [S::ZERO; 25]);
            // Both scales leave the IEEE range while the weighted values stay normal.
            let tiny = S::MIN_POSITIVE.sqrt() * S::EPSILON;
            let huge = S::MAX.sqrt().sqrt();
            for (base, exponent, amplitude) in
                [(tiny, 2, S::ONE / tiny), (huge, 5, S::ONE / (huge * huge))]
            {
                let scale = MomentScale::power(base, S::from_f64_ieee(f64::from(exponent)))?;
                assert!(scale.factor.native().is_none());
                assert!(scale.uncertainty > S::ZERO);
                let rule = MomentRule::new(unit, &cosines, None, S::ZERO, Some(scale))?;
                let [value, _, floor, _] = rule.estimate(&[amplitude; 25], &rows, &cosines)?;
                let mut expected = Scaled::from_value(amplitude)?;
                for _ in 0..exponent {
                    expected = expected.mul(Scaled::from_value(base)?)?;
                }
                let expected = expected.round()?;
                assert!(
                    (value - expected).abs() <= floor && floor > S::ZERO,
                    "{value:?} versus {expected:?} with floor {floor:?}"
                );
            }
            // A region whose weighted mass underflows rounds to zero with a
            // positive floor, never an exact zero.
            let scale = MomentScale::power(S::MIN_POSITIVE, S::from_f64_ieee(4.0))?;
            let rule = MomentRule::new(unit, &cosines, None, S::ZERO, Some(scale))?;
            let [value, _, floor, _] = rule.estimate(&[S::ONE; 25], &rows, &cosines)?;
            assert_eq!(value, S::ZERO);
            assert!(floor >= S::MIN_SUBNORMAL);
            Ok(())
        }
        check::<f32>().unwrap();
        check::<f64>().unwrap();
    }
}
