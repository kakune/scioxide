use super::*;
use scioxide_core::AlgebraicScalar;
use scioxide_core::StepTolerance;

/// Node of a root isolation tree: a split of a region, or a leaf with its
/// proved claim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RootIsolationNode<R: RealScalar> {
    /// The region is split at `point` along `axis` into two children, with
    /// node indices `left` and `left + 1`. For `n` unknowns, axes `0..n` are the
    /// real parts and `n..2n` the imaginary parts.
    Split {
        /// Axis to split.
        axis: usize,
        /// Split point, strictly inside the region along `axis`.
        point: R,
        /// Index of the first child; greater than the index of this node.
        left: usize,
    },
    /// A leaf with its proved claim, `Empty` or `Unique`, or `None` if
    /// undecided.
    Leaf(Option<RootClaim>),
}
/// Borrowed root isolation tree: its nodes and the region of each node.
#[derive(Debug, Clone, Copy)]
pub struct RootIsolationView<'a, R: RealScalar> {
    /// Nodes, with the whole domain at node 0.
    pub nodes: &'a [RootIsolationNode<R>],
    /// Region of each node in turn, `n` rectangles per node for `n` unknowns.
    pub regions: &'a [IntervalBox<R>],
}
/// Output of root isolation: a binary tree of regions whose leaves are
/// proved free of roots, proved to contain exactly one root, or undecided.
///
/// The tree holds at most [`Self::capacity`] nodes; isolation stops with
/// [`RefinementLimit::Capacity`](crate::RefinementLimit::Capacity) when
/// another split would not fit. The storage proves nothing by itself: keep the
/// report of the isolation, or check the tree with `verify_isolation`.
///
/// ```
/// use scioxide_validated::RootIsolation;
///
/// let tree = RootIsolation::<f64>::new(1, 15)?;
/// assert_eq!((tree.order(), tree.capacity()), (1, 15));
/// assert!(tree.view().nodes.is_empty());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct RootIsolation<R: RealScalar> {
    nodes: Vec<RootIsolationNode<R>>,
    regions: Vec<IntervalBox<R>>,
    order: usize,
    used: usize,
}
/// Trees are equal when they have the same order, capacity and written nodes and
/// regions.
impl<R: RealScalar> PartialEq for RootIsolation<R> {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (self.view(), other.view());
        self.order == other.order
            && self.capacity() == other.capacity()
            && a.nodes == b.nodes
            && a.regions == b.regions
    }
}
impl<R: RealScalar> core::fmt::Debug for RootIsolation<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let view = self.view();
        f.debug_struct("RootIsolation")
            .field("order", &self.order)
            .field("capacity", &self.capacity())
            .field("nodes", &view.nodes)
            .field("regions", &view.regions)
            .finish()
    }
}
impl<R: RealScalar> RootIsolation<R> {
    /// Allocates an empty tree of up to `capacity` nodes, for `order` unknowns.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `capacity` is zero; the root node needs one.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized
    ///   or reserved.
    pub fn new(order: usize, capacity: usize) -> Result<Self> {
        const REGIONS: scioxide_core::Quantity = scioxide_core::quantity!("regions");
        if capacity == 0 {
            return Err(invalid(
                scioxide_core::argument!("capacity"),
                "isolation output must hold the original domain",
            ));
        }
        Ok(Self {
            nodes: try_filled(
                capacity,
                RootIsolationNode::Leaf(None),
                scioxide_core::quantity!("nodes"),
            )?,
            regions: try_filled(checked_mul(order, capacity, REGIONS)?, zero()?, REGIONS)?,
            order,
            used: 0,
        })
    }
    /// Iterates over regions whose written leaves prove exactly one root, in node order.
    ///
    /// Each slice contains `order()` rectangles. The traversal costs O(nodes)
    /// and allocates nothing; a shared scalar endpoint may occur in two regions.
    pub fn isolated_regions(&self) -> impl Iterator<Item = &[IntervalBox<R>]> + '_ {
        self.view()
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node, RootIsolationNode::Leaf(Some(RootClaim::Unique))))
            .map(|(i, _)| &self.regions[i * self.order..(i + 1) * self.order])
    }
    /// Returns the number of unknowns.
    pub const fn order(&self) -> usize {
        self.order
    }
    /// Returns the largest number of nodes.
    pub fn capacity(&self) -> usize {
        self.nodes.len()
    }
    /// Returns the tree written by the last isolation.
    pub fn view(&self) -> RootIsolationView<'_, R> {
        RootIsolationView {
            nodes: &self.nodes[..self.used],
            regions: &self.regions[..self.used * self.order],
        }
    }
    /// Write a staged tree that fits this output.
    fn publish(&mut self, staged: RootIsolationView<'_, R>) {
        self.nodes[..staged.nodes.len()].copy_from_slice(staged.nodes);
        self.regions[..staged.regions.len()].copy_from_slice(staged.regions);
        self.used = staged.nodes.len();
    }
}
/// Summary of a root isolation tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct RootIsolationReport {
    nodes: usize,
    isolated: usize,
    excluded: usize,
    unresolved: usize,
    termination: RootIsolationTermination,
}
impl RootIsolationReport {
    /// Returns the number of nodes written, or checked by `verify_isolation`.
    pub const fn nodes(&self) -> usize {
        self.nodes
    }
    /// Returns the number of leaves proved to contain exactly one root. Leaves
    /// that touch may contain the same root.
    pub const fn isolated(&self) -> usize {
        self.isolated
    }
    /// Returns the number of leaves proved free of roots.
    pub const fn excluded(&self) -> usize {
        self.excluded
    }
    /// Returns the number of undecided leaves.
    pub const fn unresolved(&self) -> usize {
        self.unresolved
    }
    /// Returns whether every leaf is decided and the leaves with one root account for every root.
    /// In a box tree those leaves are pairwise disjoint and `isolated()` is the
    /// number of roots. In a scalar tree two may share an end, which is then
    /// their common root, counted in both.
    pub const fn is_complete(&self) -> bool {
        matches!(self.termination, RootIsolationTermination::Complete)
    }
    /// Returns the termination status.
    pub const fn termination(&self) -> RootIsolationTermination {
        self.termination
    }
    /// Report of a replay of `view` that ends at node `index` with `termination`:
    /// leaves before it were proved, and it and every later leaf remain unresolved.
    fn interrupted<R: RealScalar>(
        view: RootIsolationView<'_, R>,
        index: usize,
        termination: RootIsolationTermination,
    ) -> Self {
        let (mut isolated, mut excluded, mut unresolved) = (0, 0, 0);
        for (i, node) in view.nodes.iter().enumerate() {
            match node {
                RootIsolationNode::Leaf(Some(RootClaim::Unique)) if i < index => isolated += 1,
                RootIsolationNode::Leaf(Some(RootClaim::Empty)) if i < index => excluded += 1,
                RootIsolationNode::Leaf(_) => unresolved += 1,
                RootIsolationNode::Split { .. } => {}
            }
        }
        Self {
            nodes: index,
            isolated,
            excluded,
            unresolved,
            termination,
        }
    }
}
/// Midpoint of a region from `argument`.
fn midpoint<S: NumericScalar>(
    region: &[IntervalBox<S::Real>],
    center: &mut [S],
    argument: Argument,
) -> Result<()> {
    for (&x, c) in region.iter().zip(center) {
        *c = crate::interval_math::midpoint(x, argument)?;
    }
    Ok(())
}
fn component<R: RealScalar>(region: &[IntervalBox<R>], axis: usize) -> ExtendedInterval<R> {
    if axis < region.len() {
        region[axis].re
    } else {
        region[axis - region.len()].im
    }
}
fn set_component<R: RealScalar>(
    region: &mut [IntervalBox<R>],
    axis: usize,
    value: ExtendedInterval<R>,
) {
    if axis < region.len() {
        region[axis].re = value;
    } else {
        let n = region.len();
        region[axis - n].im = value;
    }
}
/// Widest splittable axis of a subregion of the `domain` argument.
fn widest<S: NumericScalar>(
    region: &[IntervalBox<S::Real>],
    tolerance: StepTolerance<S::Real>,
) -> Result<Option<(usize, S::Real)>> {
    let axes = checked_mul(
        region.len(),
        if !S::IS_REAL { 2 } else { 1 },
        scioxide_core::quantity!("axes"),
    )?;
    let mut best = None;
    let mut width = S::Real::ZERO;
    for axis in 0..axes {
        let (lo, hi) = bounded(
            D::from(component(region, axis)),
            scioxide_core::argument!("domain"),
        )?;
        let w = upper(D::point(hi)?.sub_admitted(&D::point(lo)?)?);
        if w > width {
            let mid = scioxide_core::checked_midpoint(lo, hi)?;
            if mid > lo
                && mid < hi
                && (!w.is_finite() || !tolerance.is_satisfied(w, lo.abs().max(hi.abs()))?)
            {
                best = Some((axis, mid));
                width = w;
            }
        }
    }
    Ok(best)
}
type Verify<'a, S> = dyn FnMut(
        SystemRootCertificateView<'_, S>,
        &mut SolveContext<'_, '_>,
    ) -> Result<RootVerificationReport<<S as NumericScalar>::Real>>
    + 'a;
/// Breadth-first isolation into staging of `w.nodes.len()` nodes. Returns the report
/// and the number of staged nodes.
fn generate<S: NumericScalar>(
    domain: &[IntervalBox<S::Real>],
    shared_ends: bool,
    tolerance: StepTolerance<S::Real>,
    w: &mut workspace::TreeScratch<'_, S>,
    context: &mut SolveContext<'_, '_>,
    verify: &mut Verify<'_, S>,
) -> Result<(RootIsolationReport, usize)> {
    const DOMAIN: Argument = scioxide_core::argument!("domain");
    let n = domain.len();
    let mut used = 1;
    w.nodes.fill(RootIsolationNode::Leaf(None));
    w.regions[..n].copy_from_slice(domain);
    let mut i = 0;
    let mut reason: Option<RootIsolationTermination> = None;
    while i < used {
        if let Some(stop) = iteration(context)? {
            reason = Some(stop.into());
            break;
        }
        w.region.copy_from_slice(&w.regions[i * n..(i + 1) * n]);
        midpoint::<S>(w.region, w.center, DOMAIN)?;
        let report = verify(
            SystemRootCertificateView {
                region: w.region,
                center: w.center,
                claim: RootClaim::Unique,
            },
            context,
        )?;
        if matches!(report.evidence, Some(RootClaim::Unique | RootClaim::Empty)) {
            w.nodes[i] = RootIsolationNode::Leaf(report.evidence);
        } else {
            if let RootVerificationStatus::Inconclusive(stop) = report.status
                && matches!(
                    stop,
                    RootInconclusive::EvaluationLimit
                        | RootInconclusive::IterationLimit
                        | RootInconclusive::Stopped(_)
                )
            {
                reason = Some(stop.into());
                break;
            }
            if let Some((axis, point)) = widest::<S>(w.region, tolerance)? {
                let left = used;
                if checked_add(left, 2, scioxide_core::quantity!("nodes"))? > w.nodes.len() {
                    reason = Some(RootIsolationTermination::Limit(
                        crate::RefinementLimit::Capacity,
                    ));
                    break;
                }
                let (lo, hi) = bounded(D::from(component(w.region, axis)), DOMAIN)?;
                for (child, a, b) in [(left, lo, point), (left + 1, point, hi)] {
                    let target = &mut w.regions[child * n..(child + 1) * n];
                    target.copy_from_slice(w.region);
                    set_component(target, axis, ExtendedInterval::new(Some(a), Some(b))?);
                }
                w.nodes[i] = RootIsolationNode::Split { axis, point, left };
                used += 2;
            } else {
                reason = reason.or(Some(RootIsolationTermination::Limit(
                    crate::RefinementLimit::Resolution,
                )));
            }
        }
        i += 1;
    }
    let view = RootIsolationView {
        nodes: &w.nodes[..used],
        regions: &w.regions[..used * n],
    };
    let report = summarize(view, n, shared_ends, reason, context)?;
    context.commit_accepted(if report.unresolved == 0 {
        PartialResultTrust::CertificateVerified
    } else {
        PartialResultTrust::Admissible
    })?;
    Ok((report, used))
}
/// Coverage summary.
fn summarize<R: RealScalar>(
    view: RootIsolationView<'_, R>,
    n: usize,
    shared_ends: bool,
    mut reason: Option<RootIsolationTermination>,
    context: &mut SolveContext<'_, '_>,
) -> Result<RootIsolationReport> {
    let mut isolated = 0;
    let mut excluded = 0;
    let mut unresolved = 0;
    for (i, node) in view.nodes.iter().enumerate() {
        match node {
            RootIsolationNode::Leaf(Some(RootClaim::Unique)) => {
                isolated += 1;
                if reason.is_none() {
                    for (j, node) in view.nodes[..i].iter().enumerate() {
                        if matches!(node, RootIsolationNode::Leaf(Some(RootClaim::Unique))) {
                            if let Some(stop) = iteration(context)? {
                                reason = Some(stop.into());
                                break;
                            }
                            if !view.regions[i * n..(i + 1) * n]
                                .iter()
                                .zip(&view.regions[j * n..(j + 1) * n])
                                .any(|(&a, &b)| disjoint(a, b))
                                && !shared_ends
                            {
                                reason = Some(
                                    RootInconclusive::Reason(RootProofFailure::Inequality).into(),
                                );
                                break;
                            }
                        }
                    }
                }
            }
            RootIsolationNode::Leaf(Some(RootClaim::Empty)) => excluded += 1,
            RootIsolationNode::Leaf(_) => unresolved += 1,
            _ => {}
        }
    }
    let reason = stop(context)?
        .map(RootIsolationTermination::Inconclusive)
        .or(reason)
        .or((unresolved > 0)
            .then_some(RootInconclusive::Reason(RootProofFailure::Inequality).into()));
    Ok(RootIsolationReport {
        nodes: view.nodes.len(),
        isolated,
        excluded,
        unresolved,
        termination: reason.unwrap_or(RootIsolationTermination::Complete),
    })
}
/// Replay of the `certificate` argument over staging of at least its node count.
fn replay<S: NumericScalar>(
    domain: &[IntervalBox<S::Real>],
    shared_ends: bool,
    view: RootIsolationView<'_, S::Real>,
    w: &mut workspace::TreeScratch<'_, S>,
    context: &mut SolveContext<'_, '_>,
    verify: &mut Verify<'_, S>,
) -> Result<RootIsolationReport> {
    const REGIONS: Argument = scioxide_core::argument!("certificate.regions");
    let n = domain.len();
    for (i, node) in view.nodes.iter().enumerate() {
        if let Some(reason) = iteration(context)? {
            return Ok(RootIsolationReport::interrupted(view, i, reason.into()));
        }
        if let RootIsolationNode::Leaf(Some(claim)) = *node {
            w.region.copy_from_slice(&view.regions[i * n..(i + 1) * n]);
            midpoint::<S>(w.region, w.center, REGIONS)?;
            let report = verify(
                SystemRootCertificateView {
                    region: w.region,
                    center: w.center,
                    claim,
                },
                context,
            )?;
            let termination = match report.status {
                RootVerificationStatus::Proved => continue,
                RootVerificationStatus::Refuted => RootIsolationTermination::Refuted,
                RootVerificationStatus::Inconclusive(reason) => reason.into(),
                RootVerificationStatus::NotRun => {
                    return Err(crate::shell::breakdown("leaf proof did not run"));
                }
            };
            return Ok(RootIsolationReport::interrupted(view, i, termination));
        }
    }
    summarize(view, n, shared_ends, None, context)
}
/// Check all topology, regions and claims before invoking any proof provider.
fn validate_replay<S: NumericScalar>(
    domain: &[IntervalBox<S::Real>],
    view: RootIsolationView<'_, S::Real>,
    w: &mut workspace::TreeScratch<'_, S>,
) -> Result<()> {
    const NODES: Argument = scioxide_core::argument!("certificate.nodes");
    const REGIONS: Argument = scioxide_core::argument!("certificate.regions");
    const MESSAGE: &str = "isolation payload does not match its original domain";
    let n = domain.len();
    if view.regions.len()
        != checked_mul(
            view.nodes.len(),
            n,
            scioxide_core::quantity!("certificate.regions"),
        )?
        || &view.regions[..n] != domain
    {
        return Err(invalid(REGIONS, MESSAGE));
    }
    validate_boxes::<S>(view.regions, true, REGIONS)?;
    w.parents.fill(0);
    for (i, node) in view.nodes.iter().enumerate() {
        w.region.copy_from_slice(&view.regions[i * n..(i + 1) * n]);
        if i > 0 && w.parents[i] != 1 {
            return Err(invalid(
                NODES,
                "disconnected or multiply-owned isolation node",
            ));
        }
        match *node {
            RootIsolationNode::Split { axis, point, left } => {
                if axis
                    >= checked_mul(
                        n,
                        if !S::IS_REAL { 2 } else { 1 },
                        scioxide_core::quantity!("axes"),
                    )?
                    || left <= i
                    || left >= view.nodes.len() - 1
                {
                    return Err(invalid(NODES, "invalid isolation split indices"));
                }
                let (lo, hi) = bounded(D::from(component(w.region, axis)), REGIONS)?;
                if !point.is_finite() || point <= lo || point >= hi {
                    return Err(invalid(NODES, "isolation split must be strictly interior"));
                }
                for (child, a, b) in [(left, lo, point), (left + 1, point, hi)] {
                    set_component(w.region, axis, ExtendedInterval::new(Some(a), Some(b))?);
                    if *w.region != view.regions[child * n..(child + 1) * n] {
                        return Err(invalid(
                            REGIONS,
                            "isolation child does not cover its exact half",
                        ));
                    }
                    w.parents[child] =
                        checked_add(w.parents[child], 1, scioxide_core::quantity!("parents"))?;
                }
            }
            RootIsolationNode::Leaf(None) => {}
            RootIsolationNode::Leaf(Some(RootClaim::Empty | RootClaim::Unique)) => {}
            _ => return Err(invalid(NODES, "invalid isolation leaf claim")),
        }
    }
    Ok(())
}
/// Validate a replayed payload's shape against its domain before any proof work.
fn payload<R: RealScalar>(domain: usize, view: RootIsolationView<'_, R>) -> Result<()> {
    if view.nodes.is_empty() {
        return Err(invalid(
            scioxide_core::argument!("certificate.nodes"),
            "isolation payload has no root node",
        ));
    }
    crate::shell::exact(
        checked_mul(
            view.nodes.len(),
            domain,
            scioxide_core::quantity!("certificate.regions"),
        )?,
        view.regions.len(),
    )
}

/// A real scalar domain as a one-component box.
fn scalar_domain<R: RealScalar>(domain: ExtendedInterval<R>) -> Result<[IntervalBox<R>; 1]> {
    Ok([Complex::new(
        domain,
        ExtendedInterval::new(Some(R::ZERO), Some(R::ZERO))?,
    )])
}

impl<S: NumericScalar> SystemRootVerifier<'_, S> {
    /// Isolates the roots in the box `domain` into a tree of at most
    /// `max_nodes` nodes; see [`Self::isolate_into`] and the
    /// [`SystemRootVerifier`] example.
    ///
    /// Allocates the tree and a workspace, and runs sequentially.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::isolate_into`] and [`RootIsolation::new`].
    pub fn isolate(
        &self,
        domain: &[IntervalBox<S::Real>],
        tolerance: StepTolerance<S::Real>,
        max_nodes: usize,
    ) -> Result<(RootIsolationReport, RootIsolation<S::Real>)> {
        let mut tree = RootIsolation::new(self.order(), max_nodes)?;
        let report = self.isolate_into(
            domain,
            tolerance,
            &mut tree,
            &mut RootWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok((report, tree))
    }
    /// Isolates the roots of `F` in the bounded box `domain` and writes the
    /// tree to `out`.
    ///
    /// The tree is built breadth first. Each region is tested for `Unique` with
    /// [`Self::verify`]; a region proved `Unique` or `Empty` becomes a leaf, and
    /// any other is bisected at the midpoint of its widest component. A region
    /// whose widest component meets `tolerance` stays an undecided leaf; the
    /// tolerance never weakens a proof. Refinement ends when every leaf is
    /// decided, the tree is full, or a budget or stop ends it. Every `Ok`
    /// result, including a stop, writes the tree built so far to `out`; on
    /// error `out` is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `domain` or `out` does not have `n` components.
    /// - `InvalidInput` if `domain` is empty or unbounded, or for the reasons given on
    ///   [`Self::verify`].
    /// - Errors of the operators, and those of [`RootWorkspace::try_reserve`]
    ///   if the workspace must grow.
    pub fn isolate_into<'s, 'c>(
        &self,
        domain: &[IntervalBox<S::Real>],
        tolerance: StepTolerance<S::Real>,
        out: &mut RootIsolation<S::Real>,
        workspace: &mut RootWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<RootIsolationReport> {
        let n = self.order();
        let capacity = out.capacity();
        let (report, used) = workspace.core.run(
            scope.into(),
            self.identity(),
            self.options(),
            || {
                self.providers()?;
                let requirements =
                    self.workspace_requirements(ProofRequest::refinement(capacity))?;
                crate::shell::exact(n, domain.len())?;
                crate::shell::exact(n, out.order())?;
                validate_boxes::<S>(domain, true, scioxide_core::argument!("domain"))?;
                Ok(requirements)
            },
            |_, _| Ok(()),
            |lanes, refinement, _, _, context| {
                let lane = &mut lanes[0];
                let mut tree = refinement.tree(n, capacity);
                generate(
                    domain,
                    false,
                    tolerance,
                    &mut tree,
                    context,
                    &mut |c, cx| self.verify_in(c, lane, cx),
                )
            },
            |(report, _)| report.is_complete(),
        )?;
        out.publish(workspace.core.shared.staged_tree(n, used));
        Ok(report)
    }
    /// Checks a root isolation tree for the box `domain`: that its splits
    /// cover `domain` and that every leaf claim holds.
    ///
    /// Every leaf claim is proved again from `F`. The check stops at the first
    /// leaf whose claim is not proved, with
    /// [`RootIsolationTermination::Refuted`] if it is refuted and
    /// `Inconclusive` otherwise.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `domain` or the tree regions do not have `n`
    ///   components.
    /// - `InvalidInput` if the tree is malformed or does not match `domain`.
    /// - Errors of the operators.
    pub fn verify_isolation(
        &self,
        domain: &[IntervalBox<S::Real>],
        certificate: RootIsolationView<'_, S::Real>,
    ) -> Result<RootIsolationReport> {
        self.verify_isolation_scoped(
            domain,
            certificate,
            &mut RootWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }
    /// [`Self::verify_isolation`] with your workspace, run in `scope`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verify_isolation`], and those of
    ///   [`RootWorkspace::try_reserve`] if the workspace must grow.
    pub fn verify_isolation_scoped<'s, 'c>(
        &self,
        domain: &[IntervalBox<S::Real>],
        certificate: RootIsolationView<'_, S::Real>,
        workspace: &mut RootWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<RootIsolationReport> {
        let n = self.order();
        let nodes = certificate.nodes.len();
        workspace.core.run(
            scope.into(),
            self.identity(),
            self.options(),
            || {
                self.providers()?;
                let requirements = self.workspace_requirements(ProofRequest::refinement(nodes))?;
                crate::shell::exact(n, domain.len())?;
                validate_boxes::<S>(domain, true, scioxide_core::argument!("domain"))?;
                payload(n, certificate)?;
                Ok(requirements)
            },
            |refinement, _| {
                validate_replay::<S>(domain, certificate, &mut refinement.tree(n, nodes))
            },
            |lanes, refinement, _, _, context| {
                let lane = &mut lanes[0];
                let mut tree = refinement.tree(n, nodes);
                replay(
                    domain,
                    false,
                    certificate,
                    &mut tree,
                    context,
                    &mut |c, cx| self.verify_in(c, lane, cx),
                )
            },
            RootIsolationReport::is_complete,
        )
    }
}
impl<R: RealScalar> ScalarRootVerifier<'_, R> {
    /// Isolates the roots in the interval `domain` into a tree of at most
    /// `max_nodes` nodes; see [`Self::isolate_into`].
    ///
    /// Allocates the tree and a workspace, and runs sequentially.
    ///
    /// ```
    /// use scioxide_core::{ExtendedInterval, StepTolerance};
    /// use scioxide_validated::{DecoratedInterval as D, ScalarRootVerifier, ValidatedOptions};
    ///
    /// // f(x) = x² − 1 has the two roots ±1 in [−2, 3].
    /// let (one, two) = (
    ///     D::new(1.0_f64, 1.0)?,
    ///     D::point(2.0)?,
    /// );
    /// let value = |x: D<f64>| x.square()?.sub(&one);
    /// let derivative = |x: D<f64>| x.mul(&two);
    /// let plan = ScalarRootVerifier::new(&value, &derivative, ValidatedOptions::new());
    /// let domain = ExtendedInterval::new(Some(-2.0), Some(3.0))?;
    /// let (report, tree) = plan.isolate(domain, StepTolerance::default(), 63)?;
    /// assert!(report.is_complete() && report.isolated() == 2);
    /// assert!(plan.verify_isolation(domain, tree.view())?.is_complete());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::isolate_into`] and [`RootIsolation::new`].
    pub fn isolate(
        &self,
        domain: ExtendedInterval<R>,
        tolerance: StepTolerance<R>,
        max_nodes: usize,
    ) -> Result<(RootIsolationReport, RootIsolation<R>)> {
        let mut tree = RootIsolation::new(1, max_nodes)?;
        let report = self.isolate_into(
            domain,
            tolerance,
            &mut tree,
            &mut RootWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok((report, tree))
    }
    /// Isolates the roots of `f` in the bounded interval `domain` and writes
    /// the tree to `out`.
    ///
    /// The tree is built as for [`SystemRootVerifier::isolate_into`], with
    /// [`Self::verify`] testing each interval for `Unique`. Undecided leaves
    /// remain in the tree. Every `Ok` result, including a stop, writes the tree
    /// built so far to `out`; on error `out` is unchanged.
    ///
    /// Two unique leaves sharing an end hold the same root: their derivative
    /// enclosures exclude zero and have the same sign at that end, so their union
    /// is strictly monotone. Each leaf contains a root, forcing both roots to be
    /// the shared end; box trees retain the disjointness requirement.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` is not a tree for one unknown.
    /// - `InvalidInput` if `domain` is unbounded or empty, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - Errors of the callbacks, and those of [`RootWorkspace::try_reserve`]
    ///   if the workspace must grow.
    pub fn isolate_into<'s, 'c>(
        &self,
        domain: ExtendedInterval<R>,
        tolerance: StepTolerance<R>,
        out: &mut RootIsolation<R>,
        workspace: &mut RootWorkspace<R>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<RootIsolationReport> {
        let domain = scalar_domain(domain)?;
        let capacity = out.capacity();
        let (report, used) = workspace.core.run(
            scope.into(),
            self.identity(),
            self.options(),
            || {
                let requirements =
                    self.workspace_requirements(ProofRequest::refinement(capacity))?;
                crate::shell::exact(1, out.order())?;
                validate_boxes::<R>(&domain, true, scioxide_core::argument!("domain"))?;
                Ok(requirements)
            },
            |_, _| Ok(()),
            |_, refinement, _, _, context| {
                let mut tree = refinement.tree(1, capacity);
                generate(
                    &domain,
                    true,
                    tolerance,
                    &mut tree,
                    context,
                    &mut |c, cx| self.verify_in(scalar(c), cx),
                )
            },
            |(report, _)| report.is_complete(),
        )?;
        out.publish(workspace.core.shared.staged_tree(1, used));
        Ok(report)
    }
    /// Checks a root isolation tree for the interval `domain`, as
    /// [`SystemRootVerifier::verify_isolation`] does for boxes.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the tree is not for one unknown.
    /// - `InvalidInput` if the tree is malformed or does not match `domain`.
    /// - Errors of the callbacks.
    pub fn verify_isolation(
        &self,
        domain: ExtendedInterval<R>,
        certificate: RootIsolationView<'_, R>,
    ) -> Result<RootIsolationReport> {
        self.verify_isolation_scoped(
            domain,
            certificate,
            &mut RootWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }
    /// [`Self::verify_isolation`] with your workspace, run in `scope`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verify_isolation`], and those of
    ///   [`RootWorkspace::try_reserve`] if the workspace must grow.
    pub fn verify_isolation_scoped<'s, 'c>(
        &self,
        domain: ExtendedInterval<R>,
        certificate: RootIsolationView<'_, R>,
        workspace: &mut RootWorkspace<R>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<RootIsolationReport> {
        let domain = scalar_domain(domain)?;
        let nodes = certificate.nodes.len();
        workspace.core.run(
            scope.into(),
            self.identity(),
            self.options(),
            || {
                let requirements = self.workspace_requirements(ProofRequest::refinement(nodes))?;
                validate_boxes::<R>(&domain, true, scioxide_core::argument!("domain"))?;
                payload(1, certificate)?;
                Ok(requirements)
            },
            |refinement, _| {
                validate_replay::<R>(&domain, certificate, &mut refinement.tree(1, nodes))
            },
            |_, refinement, _, _, context| {
                let mut tree = refinement.tree(1, nodes);
                replay(
                    &domain,
                    true,
                    certificate,
                    &mut tree,
                    context,
                    &mut |c, cx| self.verify_in(scalar(c), cx),
                )
            },
            RootIsolationReport::is_complete,
        )
    }
}
/// The scalar proposal of a one-component box proposal.
fn scalar<R: RealScalar>(c: SystemRootCertificateView<'_, R>) -> ScalarRootCertificate<R> {
    ScalarRootCertificate {
        region: c.region[0].re,
        center: c.center[0],
        claim: c.claim,
    }
}
