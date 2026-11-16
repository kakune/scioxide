use super::*;

/// Endpoints of a nonempty domain from `argument`.
fn endpoints<R: RealScalar>(
    x: ExtendedInterval<R>,
    argument: Argument,
) -> Result<(Option<R>, Option<R>)> {
    x.bounds()
        .ok_or_else(|| invalid(argument, "empty quadrature domain"))
}
/// Whether a domain has positive width; the empty set has none.
fn positive<R: RealScalar>(x: ExtendedInterval<R>) -> bool {
    match x.bounds() {
        Some((Some(a), Some(b))) => a < b,
        Some(_) => true,
        None => false,
    }
}
/// Derive the domain and validate the partition.
pub(super) fn validate_partition<R: RealScalar>(
    pieces: &[QuadraturePiece<R>],
) -> Result<ExtendedInterval<R>> {
    const PIECES: Argument = scioxide_core::argument!("pieces");
    let malformed = || {
        invalid(
            PIECES,
            "partition must cover the domain exactly in increasing order",
        )
    };
    let first = pieces.first().ok_or_else(malformed)?;
    let last = pieces.last().ok_or_else(malformed)?;
    let (start, _) = first.domain.bounds().ok_or_else(malformed)?;
    let (_, end) = last.domain.bounds().ok_or_else(malformed)?;
    let domain = ExtendedInterval::new(start, end).map_err(|_| malformed())?;
    let degenerate = !positive(domain);
    let mut next = start;
    for (index, piece) in pieces.iter().enumerate() {
        let (a, b) = piece.domain.bounds().ok_or_else(malformed)?;
        if a != next
            || !domain.contains_interval(&piece.domain)
            || !(positive(piece.domain) || degenerate && pieces.len() == 1)
            || (index != 0 && a.is_none())
            || (index + 1 != pieces.len() && b.is_none())
        {
            return Err(invalid(
                PIECES,
                "partition must cover the domain exactly in increasing order",
            ));
        }
        let permitted = match piece.assumption {
            QuadratureAssumption::ConvergentTail => a.is_none() != b.is_none(),
            QuadratureAssumption::IntegrableSingularity => {
                a.is_some() && b.is_some() && !degenerate
            }
            _ => a.is_some() && b.is_some(),
        };
        if !permitted {
            return Err(invalid(
                PIECES,
                "piece domain does not match its proof assumption",
            ));
        }
        next = b;
    }
    if next != end {
        return Err(invalid(PIECES, "partition omits part of the domain"));
    }
    Ok(domain)
}

impl<S: NumericScalar> QuadratureVerifier<'_, S> {
    /// Validate the `certificate` argument.
    fn validate_certificate(&self, c: QuadratureCertificateView<'_, S::Real>) -> Result<()> {
        const PANELS: Argument = scioxide_core::argument!("certificate.panels");
        crate::shell::exact(self.components, c.total.len())?;
        crate::shell::exact(
            checked_mul(
                c.panels.len(),
                self.components,
                scioxide_core::quantity!("certificate.local_bounds"),
            )?,
            c.local_bounds.len(),
        )?;
        let mut index = 0;
        for (piece, original) in self.pieces.iter().enumerate() {
            let (mut next, end) = original
                .domain
                .bounds()
                .ok_or_else(|| breakdown("validated partition piece is empty"))?;
            let start_index = index;
            while let Some(panel) = c.panels.get(index).filter(|p| p.piece == piece) {
                let (a, b) = endpoints(panel.domain, PANELS)?;
                if a != next
                    || !original.domain.contains_interval(&panel.domain)
                    || (!positive(panel.domain) && positive(self.domain))
                    || (!original.assumption.regular()
                        && (panel.domain != original.domain || index != start_index))
                {
                    return Err(invalid(PANELS, "certificate changes original coverage"));
                }
                index += 1;
                next = b;
                if next == end {
                    break;
                }
            }
            if index == start_index || next != end {
                return Err(invalid(
                    PANELS,
                    "certificate omits an original piece or subinterval",
                ));
            }
        }
        if index != c.panels.len() {
            return Err(invalid(
                PANELS,
                "certificate contains duplicate or unordered pieces",
            ));
        }
        validate_boxes::<S>(
            c.local_bounds,
            true,
            scioxide_core::argument!("certificate.local_bounds"),
        )?;
        validate_boxes::<S>(c.total, true, scioxide_core::argument!("certificate.total"))
    }
    /// One oracle query of the plan (`self`).
    pub(super) fn query(
        &self,
        panel: QuadraturePanel<S::Real>,
        domain: ExtendedInterval<S::Real>,
        query: QuadratureQuery,
        out: &mut [IntervalBox<S::Real>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<QuadratureInconclusive>> {
        if let Some(reason) = stop(context)? {
            return Ok(Some(reason));
        }
        let role = if matches!(
            query,
            QuadratureQuery::SecondDerivative | QuadratureQuery::EighthDerivative
        ) {
            EvaluationRole::Jacobian
        } else {
            EvaluationRole::Function
        };
        if !context.record_provider(role)?.is_granted() {
            return Ok(Some(QuadratureInconclusive::EvaluationLimit));
        }
        out.fill(sentinel());
        let request = QuadratureRequest {
            piece: panel.piece,
            domain,
            query,
        };
        let obligation = match self.oracle.enclose_scoped(request, out, context) {
            Ok(obligation) => obligation,
            Err(error) => {
                return match crate::shell::stopped(&error) {
                    Some(reason) => Ok(Some(QuadratureInconclusive::Stopped(reason))),
                    None => Err(error),
                };
            }
        };
        crate::environment::check()?;
        if let Some(reason) = stop(context)? {
            return Ok(Some(reason));
        }
        if obligation == QuadratureObligation::Unavailable {
            return Ok(Some(QuadratureInconclusive::Reason(
                QuadratureProofFailure::MissingObligation,
            )));
        }
        validate_boxes::<S>(out, false, scioxide_core::argument!("self"))?;
        Ok(out
            .iter()
            .any(|&x| !unpack(x).finite())
            .then_some(QuadratureInconclusive::Unbounded))
    }
    /// Enclose a panel integral using its declared regularity and provider bounds;
    /// `Unbounded` reports a bound that is not finite.
    pub(super) fn local_bound(
        &self,
        panel: QuadraturePanel<S::Real>,
        out: &mut [IntervalBox<S::Real>],
        scratch: &mut [IntervalBox<S::Real>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<QuadratureInconclusive>> {
        if let Some(reason) = iteration(context)? {
            return Ok(Some(reason));
        }
        if self.components == 0 || !positive(panel.domain) {
            out.fill(zero()?);
            return Ok(None);
        }
        let assumption = self.pieces[panel.piece].assumption;
        if !assumption.regular() {
            return self.query(panel, panel.domain, QuadratureQuery::Integral, out, context);
        }
        if assumption == QuadratureAssumption::EightTimesContinuouslyDifferentiable {
            return self.gauss_bound(panel, out, scratch, context);
        }
        let (a, b) = finite_endpoints(panel.domain, UNBOUNDED_PANEL)?;
        let h = D::point(b)?.sub_admitted(&D::point(a)?)?;
        if !h.set().is_finite() {
            return Ok(Some(QuadratureInconclusive::Unbounded));
        }
        if assumption == QuadratureAssumption::IntegrableRange {
            if let Some(reason) =
                self.query(panel, panel.domain, QuadratureQuery::Value, out, context)?
            {
                return Ok(Some(reason));
            }
            for x in out.iter_mut() {
                *x = pack(unpack(*x).mul(Box2::real(h)?)?);
            }
        } else {
            let (left, rest) = scratch.split_at_mut(self.components);
            let (right, derivative) = rest.split_at_mut(self.components);
            for (domain, query, output) in [
                (D::point(a)?.set(), QuadratureQuery::Value, &mut *left),
                (D::point(b)?.set(), QuadratureQuery::Value, &mut *right),
                (
                    panel.domain,
                    QuadratureQuery::SecondDerivative,
                    &mut *derivative,
                ),
            ] {
                if let Some(reason) = self.query(panel, domain, query, output, context)? {
                    return Ok(Some(reason));
                }
            }
            let half_h = Box2::real(h.div_admitted(&crate::rectangle::constant(2.0)?)?)?;
            let remainder = Box2::real(
                h.square_admitted()?
                    .mul_admitted(&h)?
                    .div_admitted(&crate::rectangle::constant(12.0)?)?,
            )?;
            for (((y, &a), &b), &d) in out
                .iter_mut()
                .zip(left.iter())
                .zip(right.iter())
                .zip(derivative.iter())
            {
                // Scale before summing, to avoid needless endpoint-value overflow.
                *y = pack(
                    unpack(a)
                        .mul(half_h)?
                        .add(unpack(b).mul(half_h)?)?
                        .sub(unpack(d).mul(remainder)?)?,
                );
            }
        }
        Ok(out
            .iter()
            .any(|&x| !unpack(x).finite())
            .then_some(QuadratureInconclusive::Unbounded))
    }
    /// Checks a certificate against the oracle, from any source.
    ///
    /// The bound of every panel is recomputed from the oracle and must lie in
    /// the recorded bound, and the recorded total must contain the outward sum
    /// of the recorded bounds; see [`QuadratureVerificationStatus`]. No
    /// tolerance is involved. Allocates a workspace and runs sequentially.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths of the certificate do not match the
    ///   plan.
    /// - `InvalidInput` if the panels do not cover the pieces of the plan, a
    ///   bound is empty or unbounded or has a nonzero imaginary part for a real
    ///   problem, the oracle returns an empty or malformed enclosure, or the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - Errors of the oracle, and `ResourceLimit`, `SizeOverflow` or
    ///   `AllocationFailure` for the workspace.
    pub fn verify(
        &self,
        certificate: QuadratureCertificateView<'_, S::Real>,
    ) -> Result<QuadratureVerificationStatus> {
        self.verify_scoped(
            certificate,
            &mut QuadratureWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }
    /// [`Self::verify`] with your workspace, run in `scope`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verify`], and those of
    ///   [`QuadratureWorkspace::try_reserve`] if the workspace must grow.
    pub fn verify_scoped<'s, 'c>(
        &self,
        certificate: QuadratureCertificateView<'_, S::Real>,
        workspace: &mut QuadratureWorkspace<S::Real>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<QuadratureVerificationStatus> {
        workspace.core.run(
            scope.into(),
            self.identity(VERIFY),
            &self.options,
            || {
                let requirements = self.workspace_requirements(ProofRequest::proofs(1))?;
                crate::environment::check()?;
                self.validate_certificate(certificate)?;
                Ok(requirements)
            },
            |_, _| Ok(()),
            |lanes, _, _, _, context| {
                self.verify_in(certificate, &mut lanes[0].view(self.components), context)
            },
            |status| *status == QuadratureVerificationStatus::Verified,
        )
    }
    /// Verifies independent certificates, writing one status per certificate.
    ///
    /// The checks run in parallel under the options' execution policy, and the
    /// statuses do not depend on the policy or the number of workers. After a
    /// cancellation or deadline, every status is `Inconclusive(Stopped)`, even
    /// for checks that finished. On error, `statuses` is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `statuses` and `certificates` differ in length.
    /// - The errors of [`Self::verify_scoped`] for any certificate, and errors
    ///   of the executor.
    pub fn verify_batch_into<'s, 'c>(
        &self,
        certificates: &[QuadratureCertificateView<'_, S::Real>],
        statuses: &mut [QuadratureVerificationStatus],
        workspace: &mut QuadratureWorkspace<S::Real>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<()> {
        let count = statuses.len();
        let components = self.components;
        workspace
            .core
            .verify_batch(
                scope.into(),
                self.identity(VERIFY),
                &self.options,
                || {
                    let requirements =
                        self.workspace_requirements(ProofRequest::proofs(certificates.len()))?;
                    crate::environment::check()?;
                    crate::shell::exact(certificates.len(), count)?;
                    certificates
                        .iter()
                        .try_for_each(|&c| self.validate_certificate(c))?;
                    Ok(requirements)
                },
                statuses,
                |_, _| Ok(crate::shell::Prepared::Ready(())),
                &|index, lane, _, _, context| {
                    self.verify_in(certificates[index], &mut lane.view(components), context)
                },
            )
            .map(|_| ())
    }
    /// Verify a validated certificate inside an existing epoch.
    fn verify_in(
        &self,
        certificate: QuadratureCertificateView<'_, S::Real>,
        lane: &mut workspace::LaneScratch<'_, S::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<QuadratureVerificationStatus> {
        crate::environment::check()?;
        for (index, &panel) in certificate.panels.iter().enumerate() {
            if let Some(reason) = self.local_bound(panel, lane.total, lane.scratch, context)? {
                return Ok(QuadratureVerificationStatus::Inconclusive(reason));
            }
            let start = index * self.components;
            if !certificate.local_bounds[start..start + self.components]
                .iter()
                .zip(lane.total.iter())
                .all(|(&claimed, &fresh)| contains(claimed, fresh))
            {
                return Ok(QuadratureVerificationStatus::Rejected(
                    QuadratureRejection::LocalBound,
                ));
            }
        }
        if let Some(reason) = sum(
            certificate.local_bounds,
            lane.total,
            scioxide_core::argument!("certificate.local_bounds"),
        )? {
            return Ok(QuadratureVerificationStatus::Inconclusive(reason));
        }
        if let Some(reason) = stop(context)? {
            return Ok(QuadratureVerificationStatus::Inconclusive(reason));
        }
        if !certificate
            .total
            .iter()
            .zip(lane.total.iter())
            .all(|(&claimed, &fresh)| contains(claimed, fresh))
        {
            return Ok(QuadratureVerificationStatus::Rejected(
                QuadratureRejection::TotalBound,
            ));
        }
        context.commit_accepted(PartialResultTrust::CertificateVerified)?;
        Ok(QuadratureVerificationStatus::Verified)
    }
}
/// Outward component sums of `bounds` taken from `argument`, which reports every failure.
pub(super) fn sum<R: RealScalar>(
    bounds: &[IntervalBox<R>],
    total: &mut [IntervalBox<R>],
    argument: Argument,
) -> Result<Option<QuadratureInconclusive>> {
    let components = total.len();
    for (component, result) in total.iter_mut().enumerate() {
        let mut sum = BoxSum::new(Box2::zero()?, argument)?;
        for panel in bounds.chunks_exact(components) {
            sum.update(unpack(panel[component]), false)?;
        }
        *result = pack(sum.finish()?);
    }
    Ok(total
        .iter()
        .any(|&x| !unpack(x).finite())
        .then_some(QuadratureInconclusive::Unbounded))
}
