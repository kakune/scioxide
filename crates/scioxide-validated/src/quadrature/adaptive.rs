use super::*;
use scioxide_core::{NormSelection, checked_midpoint};

/// Radius and scale of a finite enclosure.
pub(super) fn radius_scale<R: RealScalar>(value: IntervalBox<R>) -> Result<(R, R)> {
    let mut radius = R::ZERO;
    let mut scale = R::ZERO;
    for interval in [value.re, value.im] {
        let (lo, hi) = finite_endpoints(interval, "quadrature radius of an unbounded enclosure")?;
        let center = checked_midpoint(lo, hi)?;
        radius = radius
            .max(upper(D::point(center)?.sub_admitted(&D::point(lo)?)?))
            .max(upper(D::point(hi)?.sub_admitted(&D::point(center)?)?));
        scale = scale.max(center.abs());
    }
    Ok((radius, scale))
}
/// Radius-to-tolerance ratio.
pub(super) fn ratio<R: RealScalar>(
    local: IntervalBox<R>,
    total: IntervalBox<R>,
    tolerance: &ErrorTolerance<R>,
    component: usize,
    components: usize,
) -> Result<R> {
    let (radius, _) = radius_scale(local)?;
    radius_ratio(radius, total, tolerance, component, components)
}
/// Normalize a cached radius using the current total scale.
pub(super) fn radius_ratio<R: RealScalar>(
    radius: R,
    total: IntervalBox<R>,
    tolerance: &ErrorTolerance<R>,
    component: usize,
    components: usize,
) -> Result<R> {
    if radius == R::ZERO {
        return Ok(R::ZERO);
    }
    let (_, scale) = radius_scale(total)?;
    let threshold = D::point(tolerance.absolute().component(component, components)?)?
        .add_admitted(&D::point(tolerance.relative())?.mul_admitted(&D::point(scale)?)?)?;
    // Using the lower endpoint makes successful tolerance tests conservative.
    let lower = threshold.bounds().map_or(R::ZERO, |(lo, _)| lo);
    if lower == R::ZERO {
        return Ok(R::INFINITY);
    }
    Ok(upper(D::point(radius)?.div_admitted(&D::point(lower)?)?))
}
/// Write the outward totals of exact running `sums`; whether every total is finite.
fn finish<R: RealScalar>(sums: &[BoxSum<R>], total: &mut [IntervalBox<R>]) -> Result<bool> {
    for (total, sum) in total.iter_mut().zip(sums) {
        *total = pack(sum.clone().finish()?);
    }
    Ok(finite(total))
}
/// Whether all `bounds` are finite.
fn finite<R: RealScalar>(bounds: &[IntervalBox<R>]) -> bool {
    bounds.iter().all(|&value| unpack(value).finite())
}
/// Whether `total` meets `tolerance`.
fn tolerance_reached<R: RealScalar>(
    total: &[IntervalBox<R>],
    tolerance: &ErrorTolerance<R>,
) -> Result<bool> {
    if total.is_empty() {
        return Ok(true);
    }
    let mut maximum = R::ZERO;
    for (index, &value) in total.iter().enumerate() {
        maximum = maximum.max(ratio(value, value, tolerance, index, total.len())?);
    }
    if !maximum.is_finite() {
        return Ok(false);
    }
    if maximum == R::ZERO {
        return Ok(true);
    }
    if tolerance.norm() == NormSelection::Infinity {
        return Ok(maximum <= R::ONE);
    }
    let mut sum = D::point(R::ZERO)?;
    for (index, &value) in total.iter().enumerate() {
        let normalized = D::point(ratio(value, value, tolerance, index, total.len())?)?
            .div_admitted(&D::point(maximum)?)?;
        sum = sum.add_admitted(&if tolerance.norm() == NormSelection::L1 {
            normalized
        } else {
            normalized.square_admitted()?
        })?;
    }
    if tolerance.norm() == NormSelection::RootMeanSquare {
        sum = sum.div_admitted(&D::point(R::try_from_usize_exact(total.len())?)?)?;
    }
    if tolerance.norm() != NormSelection::L1 {
        sum = sum.sqrt_admitted()?;
    }
    Ok(upper(sum.mul_admitted(&D::point(maximum)?)?) <= R::ONE)
}

impl<S: NumericScalar> QuadratureVerifier<'_, S> {
    /// Generates a certificate of at most `max_panels` panels; see
    /// [`Self::generate_into`].
    ///
    /// Allocates the workspace and the certificate and runs sequentially. The
    /// certificate is present exactly when [`QuadratureGenerationReport::has_enclosure`]
    /// is true. It is a proposal: pass it to [`Self::verify`] to certify it.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::generate_into`].
    /// - `SizeOverflow` or `AllocationFailure` if the certificate cannot be
    ///   sized or reserved.
    pub fn generate(
        &self,
        tolerance: &ErrorTolerance<S::Real>,
        max_panels: usize,
    ) -> Result<(
        QuadratureGenerationReport,
        Option<QuadratureCertificate<S::Real>>,
    )> {
        let mut certificate = QuadratureCertificate::new(max_panels, self.components)?;
        let report = self.generate_into(
            tolerance,
            &mut certificate,
            &mut QuadratureWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok((report, report.has_enclosure().then_some(certificate)))
    }
    /// Generates a certificate by adaptive bisection and writes it to `out`.
    ///
    /// Each step bisects the panel of a regular piece whose bound has the
    /// largest radius relative to `tolerance`. A panel whose bound is not
    /// finite, for example because the interval extension overestimates on a
    /// wide panel, is left out of the total and bisected first; a certificate
    /// exists only once every panel bound and the total are finite. Generation
    /// ends with `Inconclusive(Unbounded)` when the bound of a singular or tail
    /// piece, of a panel too narrow to bisect, or the total is not finite; a
    /// certificate found earlier is kept. Singular and tail pieces are never
    /// subdivided.
    ///
    /// The radius of a complex component is the larger of its real and
    /// imaginary radii, and `tolerance` combines the components. Generation
    /// stops with [`RefinementLimit::Capacity`](crate::RefinementLimit::Capacity)
    /// when a split would not fit in `out`, and a budget or stop keeps the last
    /// certificate. `out` changes only if the report has an enclosure; on error
    /// it is unchanged, but the oracle may have run. For `N` panels and `c`
    /// components, bisection costs O(cN log N) time and O(cN) storage, besides
    /// the oracle.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` cannot hold the pieces of the partition
    ///   or has the wrong number of components, or if `tolerance` has the wrong
    ///   number of components.
    /// - `InvalidInput` if the oracle returns an empty or malformed enclosure,
    ///   or if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - Errors of the oracle, and those of [`QuadratureWorkspace::try_reserve`]
    ///   if the workspace must grow.
    pub fn generate_into<'s, 'c>(
        &self,
        tolerance: &ErrorTolerance<S::Real>,
        out: &mut QuadratureCertificate<S::Real>,
        workspace: &mut QuadratureWorkspace<S::Real>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<QuadratureGenerationReport> {
        let capacity = out.capacity();
        let components = self.components;
        let report = workspace.core.run(
            scope.into(),
            self.identity(GENERATE),
            &self.options,
            || {
                let requirements =
                    self.workspace_requirements(ProofRequest::refinement(capacity))?;
                crate::environment::check()?;
                crate::shell::at_least(self.pieces.len(), capacity)?;
                crate::shell::exact(components, out.components())?;
                if let Some(values) = tolerance.absolute().as_vector() {
                    crate::shell::exact(components, values.len())?;
                }
                Ok(requirements)
            },
            |_, _| Ok(()),
            |lanes, cover, _, requirements, context| {
                self.refine(
                    tolerance,
                    capacity,
                    &mut cover.view(requirements),
                    &mut lanes[0].view(components),
                    context,
                )
            },
            |report| report.termination == QuadratureTermination::ToleranceReached,
        )?;
        if report.has_enclosure() {
            out.publish(workspace.staged(report.panels, components));
        }
        Ok(report)
    }
    /// Bisect within `max_panels` cover panels, staging the cover and total.
    ///
    /// A panel is pending while its bounds are not finite: it is scheduled first and
    /// kept out of the running sums, and the staged total covers the domain only when
    /// no panel is pending.
    fn refine(
        &self,
        tolerance: &ErrorTolerance<S::Real>,
        max_panels: usize,
        cover: &mut workspace::CoverScratch<'_, S::Real>,
        lane: &mut workspace::LaneScratch<'_, S::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<QuadratureGenerationReport> {
        const UNBOUNDED: QuadratureTermination =
            QuadratureTermination::Inconclusive(QuadratureInconclusive::Unbounded);
        let c = self.components;
        let unpublished = |termination| {
            Ok(QuadratureGenerationReport {
                termination,
                panels: 0,
            })
        };
        for sum in cover.refinement.sums.iter_mut() {
            *sum = BoxSum::new(Box2::zero()?, scioxide_core::argument!("self"))?;
        }
        let mut pending = 0;
        for (piece, original) in self.pieces.iter().enumerate() {
            let panel = QuadraturePanel {
                piece,
                domain: original.domain,
            };
            cover.panels[piece] = panel;
            let bounds = &mut cover.bounds[piece * c..(piece + 1) * c];
            // Only regular pieces subdivide, so their children are regular too.
            match self.local_bound(panel, bounds, lane.scratch, context)? {
                None => {
                    for (sum, &value) in cover.refinement.sums.iter_mut().zip(bounds.iter()) {
                        sum.update(unpack(value), false)?;
                    }
                    if original.assumption.regular() {
                        cover.refinement.insert(piece, cover.panels, Some(bounds))?;
                    }
                }
                Some(QuadratureInconclusive::Unbounded)
                    if original.assumption.regular() && scheduler::bisect(panel)?.is_some() =>
                {
                    bounds.fill(sentinel());
                    cover.refinement.insert(piece, cover.panels, None)?;
                    pending += 1;
                }
                Some(reason) => return unpublished(QuadratureTermination::Inconclusive(reason)),
            }
        }
        let mut count = self.pieces.len();
        if pending == 0 && !finish(cover.refinement.sums, lane.total)? {
            return unpublished(UNBOUNDED);
        }
        let termination = loop {
            if let Some(reason) = stop(context)? {
                break QuadratureTermination::Inconclusive(reason);
            }
            if pending == 0 && tolerance_reached(lane.total, tolerance)? {
                break QuadratureTermination::ToleranceReached;
            }
            if let Some(reason) = iteration(context)? {
                break QuadratureTermination::Inconclusive(reason);
            }
            let Some(index) = cover
                .refinement
                .select(cover.panels, lane.total, tolerance)?
            else {
                break if cover.refinement.unresolved {
                    QuadratureTermination::Limit(crate::RefinementLimit::Resolution)
                } else {
                    QuadratureTermination::FixedBounds
                };
            };
            if count == max_panels {
                break QuadratureTermination::Limit(crate::RefinementLimit::Capacity);
            }
            let children = scheduler::bisect(cover.panels[index])?
                .ok_or_else(|| breakdown("scheduled quadrature panel cannot be bisected"))?;
            let parent_pending = !finite(&cover.bounds[index * c..(index + 1) * c]);
            // Children of a pending parent may be pending too; once the cover is
            // finite, a child without finite bounds ends refinement and keeps that cover.
            let mut child_pending = [false; 2];
            let mut interrupted = None;
            for (child, &panel) in children.iter().enumerate() {
                let slot = max_panels + child;
                let bounds = &mut cover.bounds[slot * c..(slot + 1) * c];
                match self.local_bound(panel, bounds, lane.scratch, context)? {
                    None => {}
                    Some(QuadratureInconclusive::Unbounded)
                        if parent_pending && scheduler::bisect(panel)?.is_some() =>
                    {
                        bounds.fill(sentinel());
                        child_pending[child] = true;
                    }
                    Some(reason) => {
                        interrupted = Some(reason);
                        break;
                    }
                }
            }
            if let Some(reason) = interrupted {
                break QuadratureTermination::Inconclusive(reason);
            }
            if let Some(reason) = stop(context)? {
                break QuadratureTermination::Inconclusive(reason);
            }
            let remaining = pending - usize::from(parent_pending)
                + child_pending.iter().filter(|&&child| child).count();
            // Sum the prospective cover before replacing its parent. A total that is
            // not finite must preserve the previous finite cover.
            for (component, sum) in cover.refinement.sums.iter_mut().enumerate() {
                if !parent_pending {
                    sum.update(unpack(cover.bounds[index * c + component]), true)?;
                }
                for (slot, &unbounded) in (max_panels..).zip(&child_pending) {
                    if !unbounded {
                        sum.update(unpack(cover.bounds[slot * c + component]), false)?;
                    }
                }
            }
            if remaining == 0 && !finish(cover.refinement.sums, &mut lane.scratch[..c])? {
                break UNBOUNDED;
            }
            cover.refinement.remove(index, cover.panels);
            for (child, destination) in [index, count].into_iter().enumerate() {
                cover.panels[destination] = children[child];
                let slot = max_panels + child;
                cover
                    .bounds
                    .copy_within(slot * c..(slot + 1) * c, destination * c);
                let bounds = &cover.bounds[destination * c..(destination + 1) * c];
                cover.refinement.insert(
                    destination,
                    cover.panels,
                    (!child_pending[child]).then_some(bounds),
                )?;
            }
            if remaining == 0 {
                lane.total.copy_from_slice(&lane.scratch[..c]);
            }
            pending = remaining;
            count += 1;
        };
        let termination = match stop(context)? {
            Some(reason) => QuadratureTermination::Inconclusive(reason),
            None => termination,
        };
        // Nothing covers the domain while a panel is pending.
        if pending != 0 {
            return unpublished(termination);
        }
        cover.refinement.sort(
            &mut cover.panels[..count],
            &mut cover.bounds[..count * c],
            c,
        );
        // Commit the shared ledger before the infallible caller-output writes.
        context.commit_accepted(PartialResultTrust::Admissible)?;
        Ok(QuadratureGenerationReport {
            termination,
            panels: count,
        })
    }
}
