//! Application of fixed rules: node mapping, checks, sampling and weighted sums.

use scioxide_core::{
    AffineValue, AffineValueMut, ExecutionPolicy, Interval, NonFiniteOrigin, PartialResultTrust,
    RealScalar, SciOxideError, SolveBudgets, SolveScope, write_value,
};

use super::{FixedRule, FixedRuleNode, FixedRuleWorkspace};
use crate::report::{IntegrationErrorComponents, IntegrationReport, IntegrationTermination};
use crate::workspace::Outcome;

/// Result of mapping a rule: nodes written to scratch (with whether their
/// geometry is already known to be valid), or a native stencil.
pub(crate) enum Mapping<'a, S: RealScalar> {
    Stored(bool),
    Native(crate::kernel::NativeStencil<'a, S>),
}

/// Domain on which mapped nodes must lie, strictly monotone.
#[derive(Clone, Copy)]
pub(crate) enum Geometry<S: RealScalar> {
    /// Closed finite interval; a single node may touch an endpoint only if permitted.
    Interval {
        interval: Interval<S>,
        single_endpoint: bool,
    },
    /// Open half-line above `lower`, with strictly increasing nodes.
    HalfLine { lower: S },
    /// The whole real line.
    RealLine,
}

impl<S: RealScalar> Geometry<S> {
    fn degenerate(self) -> bool {
        matches!(self, Self::Interval { interval, .. } if interval.lower() == interval.upper())
    }

    pub(crate) fn validator(self, count: usize) -> crate::kernel::MappedPoints<S> {
        use crate::kernel::{MappedPoints, PointDomain};
        let domain = match self {
            Self::Interval {
                interval,
                single_endpoint,
            } => PointDomain::Interval {
                interval,
                closed: count != 1 || single_endpoint,
            },
            Self::HalfLine { lower } => PointDomain::HalfLine { lower },
            Self::RealLine => PointDomain::RealLine,
        };
        MappedPoints::new(domain, matches!(self, Self::HalfLine { .. }))
    }
}

/// Maps, samples and sums a fixed rule; the output is written only on success.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn apply<'rule, S, R, F, V, O>(
    rule: &R,
    mut map_rule: impl FnMut(&mut [FixedRuleNode<S>]) -> Result<Mapping<'rule, S>, SciOxideError>,
    geometry: Geometry<S>,
    native_stencil: Option<crate::kernel::NativeStencil<'rule, S>>,
    function: &F,
    out: &mut O,
    workspace: &mut FixedRuleWorkspace<S>,
    scope: SolveScope<'_, '_>,
) -> Result<IntegrationReport<S>, SciOxideError>
where
    S: RealScalar + 'rule,
    R: FixedRule<S> + ?Sized,
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
    V: AffineValue<Scale = S>,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    let count = rule.node_count();
    let components = workspace.requirements.components();
    if components == 1
        && count <= 64
        && !geometry.degenerate()
        && let Some(stencil) = native_stencil
    {
        return apply_prepared(
            rule,
            count,
            components,
            geometry,
            out,
            workspace,
            scope,
            |_| Ok(Some(stencil)),
            |stencil, _, rows, values, _, cx| {
                values[0] = stencil.sample(function, rows, cx)?;
                Ok(false)
            },
        );
    }
    apply_prepared(
        rule,
        count,
        components,
        geometry,
        out,
        workspace,
        scope,
        |nodes| {
            let mapping = match native_stencil {
                Some(stencil) => Mapping::Native(stencil),
                None => map_rule(nodes)?,
            };
            let admitted = match mapping {
                Mapping::Native(stencil) => {
                    stencil.map_into(nodes);
                    true
                }
                Mapping::Stored(admitted) => admitted,
            };
            if admitted {
                Ok(Some(nodes[0].embedded_weight().is_some()))
            } else {
                validate_nodes(count, geometry, nodes)
            }
        },
        |embedded, nodes, rows, values, errors, cx| {
            if components == 1 && !embedded && count <= 64 {
                let [sum] = crate::evaluation::sample_scalar::<S, _, 1, false>(
                    nodes,
                    rows,
                    cx,
                    {
                        #[inline(always)]
                        |node: &FixedRuleNode<S>| {
                            crate::evaluation::scalar_value(&function(node.point())?)
                        }
                    },
                    |node| node.primary_weight(),
                )?;
                values[0] = sum.round()?;
            } else {
                sample_metadata(function, nodes, components, rows, cx)?;
                for (component, (value, error)) in
                    values.iter_mut().zip(errors.iter_mut()).enumerate()
                {
                    let sample = |index: usize| rows[index * components + component];
                    if embedded {
                        let Geometry::Interval { interval, .. } = geometry else {
                            return Err(crate::kernel::invalid(
                                scioxide_core::argument!("rule"),
                                scioxide_core::InvalidInputReason::Inconsistent,
                                "embedded fixed rules require a finite interval",
                            ));
                        };
                        (*value, *error) = rule.estimate_embedded(interval, nodes, &sample)?;
                    } else {
                        *value = weighted_value(nodes, sample)?;
                    }
                }
            }
            Ok(embedded)
        },
    )
}

/// Applies a Gauss–Kronrod rule with its symmetric node layout.
#[inline(always)]
pub(crate) fn apply_kronrod<S, F, V, O>(
    rule: &crate::GaussKronrod<S>,
    interval: Interval<S>,
    function: &F,
    out: &mut O,
    workspace: &mut FixedRuleWorkspace<S>,
    scope: SolveScope<'_, '_>,
) -> Result<IntegrationReport<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
    V: AffineValue<Scale = S>,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    let count = rule.node_count();
    let components = workspace.requirements.components();
    let geometry = Geometry::Interval {
        interval,
        single_endpoint: false,
    };
    let canonical = rule.rule();
    if components == 1
        && let Some(map) = crate::kernel::AffineMap1D::native(interval)
        && let Some((middle, half)) = canonical.native_frame(map)
    {
        return apply_prepared(
            rule,
            count,
            components,
            geometry,
            out,
            workspace,
            scope,
            |_| Ok(Some(map.half_weight()?)),
            |scale, _, rows, values, errors, cx| {
                canonical.sample_native_scalar(middle, half, function, rows, cx)?;
                let estimate = canonical.apply_values::<0>(scale, rows)?;
                values[0] = estimate.value();
                errors[0] = IntegrationErrorComponents::discretization_only(estimate.error())
                    .with_roundoff(estimate.roundoff_floor());
                Ok(true)
            },
        );
    }
    apply_prepared(
        rule,
        count,
        components,
        geometry,
        out,
        workspace,
        scope,
        |nodes| {
            let map = super::interval_map(interval)?;
            let admitted = canonical.fixed_nodes(map, nodes)?;
            let scale = map.half_weight()?;
            let mapped = if admitted {
                Some(true)
            } else {
                validate_nodes(count, geometry, nodes)?
            };
            Ok(mapped.map(|_| scale))
        },
        |scale, nodes, rows, values, errors, cx| {
            sample_metadata(function, nodes, components, rows, cx)?;
            for (component, (value, error)) in values.iter_mut().zip(errors.iter_mut()).enumerate()
            {
                let estimate = if components == 1 {
                    canonical.apply_values::<0>(scale, rows)?
                } else {
                    canonical.apply_samples::<0>(scale, |index| {
                        Ok(rows[index * components + component])
                    })?
                };
                *value = estimate.value();
                *error = IntegrationErrorComponents::discretization_only(estimate.error())
                    .with_roundoff(estimate.roundoff_floor());
            }
            Ok(true)
        },
    )
}

#[inline(always)]
fn sample_metadata<S, F, V>(
    function: &F,
    nodes: &[FixedRuleNode<S>],
    components: usize,
    rows: &mut [S],
    cx: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
    V: AffineValue<Scale = S>,
{
    if components == 1 && nodes.len() <= 64 {
        crate::evaluation::sample_scalar::<S, _, 0, false>(
            nodes,
            rows,
            cx,
            {
                #[inline(always)]
                |node: &FixedRuleNode<S>| crate::evaluation::scalar_value(&function(node.point())?)
            },
            |_| S::ONE,
        )?;
        Ok(())
    } else {
        crate::evaluation::sample_rows(
            nodes.len(),
            components,
            rows,
            ExecutionPolicy::Sequential,
            cx,
            &|index, row| crate::evaluation::write_row(&function(nodes[index].point())?, row),
        )
    }
}

/// Input checks, node preparation, budget check, sampling and output write.
/// `prepare` checks the geometry before the budget is charged. `sample` returns
/// whether it filled per-component error estimates; the output is written only
/// after sampling succeeds.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[inline(always)]
fn apply_prepared<S, R, O, P>(
    rule: &R,
    count: usize,
    components: usize,
    geometry: Geometry<S>,
    out: &mut O,
    workspace: &mut FixedRuleWorkspace<S>,
    scope: SolveScope<'_, '_>,
    prepare: impl FnOnce(&mut [FixedRuleNode<S>]) -> Result<Option<P>, SciOxideError>,
    sample: impl FnOnce(
        P,
        &[FixedRuleNode<S>],
        &mut [S],
        &mut [S],
        &mut [IntegrationErrorComponents<S>],
        &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError>,
) -> Result<IntegrationReport<S>, SciOxideError>
where
    S: RealScalar,
    R: FixedRule<S> + ?Sized,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    crate::execution::preflight(
        &mut workspace.attempts,
        &mut workspace.published_errors,
        out,
        components,
        None,
        || {
            Ok(())
                .and_then(|()| {
                    if count == 0 {
                        Err(crate::kernel::invalid(
                            scioxide_core::argument!("rule"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "fixed rule must expose at least one node",
                        ))
                    } else {
                        Ok(())
                    }
                })
                .and_then(|()| {
                    if count > workspace.nodes.len() {
                        crate::kernel::shape(count, workspace.nodes.len())
                    } else {
                        Ok(())
                    }
                })
        },
    )?;
    let FixedRuleWorkspace {
        attempts,
        nodes,
        rows,
        values,
        errors,
        published_errors,
        ..
    } = workspace;
    crate::execution::integrate(
        attempts,
        published_errors,
        scope,
        rule.algorithm_id(),
        count,
        SolveBudgets::unbounded(),
        {
            #[inline(always)]
            |cx| {
                if components == 0 || geometry.degenerate() {
                    return crate::execution::zero(
                        &mut values[..components],
                        out,
                        if components == 0 {
                            IntegrationTermination::FixedRuleCompleted
                        } else {
                            IntegrationTermination::DegenerateDomain
                        },
                        cx,
                    );
                }
                let nodes = &mut nodes[..count];
                let Some(prepared) = prepare(nodes)? else {
                    return Ok(Outcome::Unpublished(
                        IntegrationTermination::GeometryFailure,
                    ));
                };
                if let Some(stop) = crate::workspace::admit(cx, count, false)? {
                    return Ok(Outcome::Unpublished(stop));
                }
                let embedded = sample(
                    prepared,
                    nodes,
                    &mut rows[..count * components],
                    values,
                    errors,
                    cx,
                )?;
                cx.commit_accepted(PartialResultTrust::Admissible)?;
                write_value(&values[..components], out)?;
                Ok(Outcome::Published {
                    termination: IntegrationTermination::FixedRuleCompleted,
                    subdivisions: 0,
                    errors: embedded
                        .then(|| IntegrationErrorComponents::envelope(&errors[..components]))
                        .flatten(),
                })
            }
        },
    )
}

/// Checks mapped nodes and weights when the mapping could not prove them valid.
#[inline(never)]
fn validate_nodes<S: RealScalar>(
    count: usize,
    geometry: Geometry<S>,
    nodes: &[FixedRuleNode<S>],
) -> Result<Option<bool>, SciOxideError> {
    let mut validator = geometry.validator(count);
    let mut embedded = None;
    for node in nodes.iter().copied() {
        if !node.primary_weight().is_finite()
            || node
                .embedded_weight()
                .is_some_and(|weight| !weight.is_finite())
        {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Output,
                "fixed rule exposed non-finite mapped metadata",
            ));
        }
        if !validator.push(node.point())? {
            return Ok(None);
        }
        let has_embedded = node.embedded_weight().is_some();
        if embedded.is_some_and(|embedded| embedded != has_embedded) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("rule"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "embedded weights must be present for either every node or no node",
            ));
        }
        embedded = Some(has_embedded);
    }
    Ok(Some(embedded.unwrap_or(false)))
}

/// Compensated weighted sum `Σ wᵢ f(xᵢ)` with extended-range promotion.
fn weighted_value<S: RealScalar>(
    nodes: &[FixedRuleNode<S>],
    sample: impl Fn(usize) -> S,
) -> Result<S, SciOxideError> {
    weighted_sum(nodes, sample, false)?.round()
}

fn weighted_sum<S: RealScalar>(
    nodes: &[FixedRuleNode<S>],
    sample: impl Fn(usize) -> S,
    absolute: bool,
) -> Result<scioxide_core::PromotingCompensatedSum<S>, SciOxideError> {
    scioxide_core::PromotingCompensatedSum::from_products(nodes.iter().enumerate().map(
        |(i, node)| {
            let (weight, value) = (node.primary_weight(), sample(i));
            if absolute {
                (weight.abs(), value.abs())
            } else {
                (weight, value)
            }
        },
    ))
}

/// Default embedded estimate: the primary sum, the primary/embedded difference
/// and the roundoff error `50 ε Σ |wᵢ f(xᵢ)|`.
pub(crate) fn embedded_estimate<S: RealScalar>(
    nodes: &[FixedRuleNode<S>],
    sample: &dyn Fn(usize) -> S,
) -> Result<(S, IntegrationErrorComponents<S>), SciOxideError> {
    let primary = weighted_sum(nodes, sample, false)?;
    let secondary = scioxide_core::PromotingCompensatedSum::from_products(
        nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.embedded_weight().unwrap_or(S::ZERO), sample(i))),
    )?;
    let absolute = crate::kernel::diagnostic(weighted_sum(nodes, sample, true)?.round())?;
    Ok((
        primary.round()?,
        IntegrationErrorComponents::discretization_only(
            crate::kernel::diagnostic(primary.difference(secondary))?.abs(),
        )
        .with_roundoff(crate::kernel::roundoff_floor(absolute)?),
    ))
}
