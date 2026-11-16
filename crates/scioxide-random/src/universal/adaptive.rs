//! Rejection samplers with adaptive tangent envelopes: adaptive rejection
//! sampling and transformed density rejection.
use super::*;
use crate::law::logadd;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Node {
    x: f64,
    log: f64,
    y: f64,
    slope: f64,
}
#[derive(Debug, Clone, Copy, PartialEq)]
struct Piece {
    node: Node,
    left: f64,
    right: f64,
    log_mass: f64,
    cumulative: f64,
}
#[derive(Debug, Clone)]
struct Table {
    nodes: Vec<Node>,
    pieces: Vec<Piece>,
    candidate: Vec<Node>,
    pending: Vec<Piece>,
    log_total: f64,
    padding: f64,
}

/// Adaptive envelope of an [`AdaptiveRejectionSampler`](crate::AdaptiveRejectionSampler) or
/// [`TransformedDensitySampler`](crate::TransformedDensitySampler), refined as it samples.
///
/// Create it with the sampler's `workspace` method, which allocates room for `table_nodes`
/// tangent points. Each proposal that the squeeze does not accept, so that the density is
/// evaluated, adds a tangent point whether or not it is then accepted, until the table is full, so the envelope fits the density more closely and later draws
/// evaluate the density less often. A refinement that would give an invalid envelope is
/// skipped, and the previous envelope is kept. Refinements made before an error stay in the
/// workspace, whatever the generator state policy. A workspace works only with the sampler that
/// created it.
///
/// See [`AdaptiveRejectionSampler`](crate::AdaptiveRejectionSampler) for an example.
#[derive(Debug)]
pub struct AdaptiveSamplingWorkspace<'a, S: RealScalar> {
    table: Table,
    transform: DensityTransform,
    owner: usize,
    scalar: PhantomData<&'a S>,
}
impl<S: RealScalar> AdaptiveSamplingWorkspace<'_, S> {
    /// Returns the number of tangent points in the envelope.
    pub fn nodes(&self) -> usize {
        self.table.nodes.len()
    }
}

fn table_bytes(capacity: usize) -> Result<usize, SciOxideError> {
    let per_node = 2 * core::mem::size_of::<Node>() + 2 * core::mem::size_of::<Piece>();
    scioxide_core::backend::checked_mul(
        capacity,
        per_node,
        scioxide_core::quantity!("workspace_bytes"),
    )
}
// A sampling workspace holds a copy of the prepared table next to it.
fn workspace_bytes(capacity: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_mul(
        table_bytes(capacity)?,
        2,
        scioxide_core::quantity!("workspace_bytes"),
    )
}
fn reserve(capacity: usize) -> Result<Table, SciOxideError> {
    Ok(Table {
        nodes: scioxide_core::backend::try_with_capacity(
            capacity,
            scioxide_core::quantity!("density_nodes"),
        )?,
        pieces: scioxide_core::backend::try_with_capacity(
            capacity,
            scioxide_core::quantity!("density_pieces"),
        )?,
        candidate: scioxide_core::backend::try_with_capacity(
            capacity,
            scioxide_core::quantity!("candidate_nodes"),
        )?,
        pending: scioxide_core::backend::try_with_capacity(
            capacity,
            scioxide_core::quantity!("candidate_pieces"),
        )?,
        log_total: 0.0,
        padding: 0.0,
    })
}
fn node<S: RealScalar, P: ConcaveDensity<S>>(
    provider: &P,
    x: S,
    t: DensityTransform,
    work: &mut Work,
    argument: scioxide_core::Argument,
) -> Result<Node, SciOxideError> {
    let log = log_density(provider, x, work, argument)?;
    work.evaluation()?;
    let derivative = provider.log_density_derivative(x)?.into_f64();
    if !log.is_finite() || !derivative.is_finite() {
        return Err(invalid(
            argument,
            "adaptive knots need finite log density and derivative",
        ));
    }
    let (y, slope) = match t {
        DensityTransform::Log => (log, derivative),
        DensityTransform::NegativeInverseSqrt => {
            let y = -(-0.5 * log).exp();
            (y, -0.5 * y * derivative)
        }
    };
    if !y.is_finite()
        || !slope.is_finite()
        || (t == DensityTransform::NegativeInverseSqrt && y >= 0.0)
    {
        return Err(invalid(
            argument,
            "unrepresentable transformed density knot",
        ));
    }
    Ok(Node {
        x: x.into_f64(),
        log,
        y,
        slope,
    })
}
fn tangent(n: Node, x: f64) -> f64 {
    n.y + n.slope * (x - n.x)
}
// Integrals are evaluated from endpoint heights, avoiding exp(log density).
fn mass(
    n: Node,
    a: f64,
    b: f64,
    t: DensityTransform,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    let slope = n.slope;
    let result = match t {
        DensityTransform::Log => {
            if slope == 0.0 {
                if !a.is_finite() || !b.is_finite() {
                    return Err(invalid(argument, "flat envelope has nonintegrable tail"));
                }
                n.y + (b - a).ln()
            } else if slope > 0.0 {
                if !b.is_finite() {
                    return Err(invalid(
                        argument,
                        "right log envelope tail is nonintegrable",
                    ));
                }
                let y = tangent(n, b);
                if a == f64::NEG_INFINITY {
                    y - slope.ln()
                } else {
                    y + (-(-slope * (b - a)).exp_m1()).ln() - slope.ln()
                }
            } else {
                if !a.is_finite() {
                    return Err(invalid(argument, "left log envelope tail is nonintegrable"));
                }
                let y = tangent(n, a);
                if b == f64::INFINITY {
                    y - (-slope).ln()
                } else {
                    y + (-(slope * (b - a)).exp_m1()).ln() - (-slope).ln()
                }
            }
        }
        DensityTransform::NegativeInverseSqrt => {
            if (a.is_finite() && tangent(n, a) >= 0.0) || (b.is_finite() && tangent(n, b) >= 0.0) {
                return Err(invalid(argument, "transformed envelope crosses zero"));
            }
            if slope == 0.0 {
                if !a.is_finite() || !b.is_finite() {
                    return Err(invalid(argument, "flat TDR tail is nonintegrable"));
                }
                (b - a).ln() - 2.0 * (-n.y).ln()
            } else if a == f64::NEG_INFINITY {
                if slope <= 0.0 || !b.is_finite() {
                    return Err(invalid(argument, "left TDR tail is nonintegrable"));
                }
                -slope.ln() - (-tangent(n, b)).ln()
            } else if b == f64::INFINITY {
                if slope >= 0.0 {
                    return Err(invalid(argument, "right TDR tail is nonintegrable"));
                }
                -(-slope).ln() - (-tangent(n, a)).ln()
            } else {
                (b - a).ln() - (-tangent(n, a)).ln() - (-tangent(n, b)).ln()
            }
        }
    };
    if result.is_nan() || result == f64::INFINITY {
        return Err(invalid(argument, "unrepresentable envelope mass"));
    }
    Ok(result)
}
fn intersection(
    n: Node,
    next: Node,
    epsilon: f64,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    if n.x >= next.x {
        return Err(invalid(argument, "density knots are not ordered"));
    }
    let secant = (next.y - n.y) / (next.x - n.x);
    if !secant.is_finite() {
        return Err(invalid(argument, "unrepresentable envelope secant"));
    }
    let slack = 64.0 * epsilon * (1.0 + n.slope.abs() + next.slope.abs() + secant.abs());
    if n.slope + slack < next.slope || secant > n.slope + slack || secant < next.slope - slack {
        return Err(invalid(
            argument,
            "density derivative violates ordered secant concavity",
        ));
    }
    let right = if (n.slope - next.slope).abs() <= slack {
        0.5 * n.x + 0.5 * next.x
    } else {
        n.x + (next.y - n.y - next.slope * (next.x - n.x)) / (n.slope - next.slope)
    };
    if !right.is_finite() || right < n.x || right > next.x {
        return Err(invalid(argument, "invalid tangent intersection"));
    }
    Ok(right)
}
fn piece_at(
    nodes: &[Node],
    i: usize,
    bounds: (f64, f64),
    t: DensityTransform,
    epsilon: f64,
    argument: scioxide_core::Argument,
) -> Result<Piece, SciOxideError> {
    let n = nodes[i];
    let left = if i == 0 {
        bounds.0
    } else {
        intersection(nodes[i - 1], n, epsilon, argument)?
    };
    let right = if i + 1 == nodes.len() {
        bounds.1
    } else {
        intersection(n, nodes[i + 1], epsilon, argument)?
    };
    if right < left {
        return Err(invalid(argument, "invalid tangent interval"));
    }
    Ok(Piece {
        node: n,
        left,
        right,
        log_mass: if right == left {
            f64::NEG_INFINITY
        } else {
            mass(n, left, right, t, argument)?
        },
        cumulative: 0.0,
    })
}
fn normalize(
    pieces: &mut [Piece],
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    let total = pieces
        .iter()
        .try_fold(f64::NEG_INFINITY, |a, p| logadd(a, p.log_mass))?;
    if !total.is_finite() {
        return Err(invalid(argument, "invalid envelope normalization"));
    }
    let mut sum = 0.0;
    for p in pieces.iter_mut() {
        // Underflowed segments have zero proposal mass and are skipped by lookup.
        sum += (p.log_mass - total).exp();
        p.cumulative = sum;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(invalid(argument, "invalid envelope normalization"));
    }
    for p in pieces.iter_mut() {
        p.cumulative /= sum;
    }
    Ok(total + sum.ln())
}
fn build(
    nodes: &[Node],
    pieces: &mut Vec<Piece>,
    bounds: (f64, f64),
    t: DensityTransform,
    epsilon: f64,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    if nodes.len() < 2 {
        return Err(invalid(
            argument,
            "adaptive rejection needs at least two knots",
        ));
    }
    pieces.clear();
    for i in 0..nodes.len() {
        pieces.push(piece_at(nodes, i, bounds, t, epsilon, argument)?);
    }
    normalize(pieces, argument)
}
fn padding(nodes: &[Node], epsilon: f64) -> f64 {
    64.0 * epsilon * (1.0 + nodes.iter().map(|n| n.log.abs()).fold(0.0, f64::max))
}
/// Build a candidate in reserved pending storage. Only the inserted tangent
/// and its neighbors need new intersections and integrals; normalization alone
/// visits the rest. Failure leaves every active table field untouched.
fn refine(
    table: &mut Table,
    n: Node,
    bounds: (f64, f64),
    t: DensityTransform,
    epsilon: f64,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    table.candidate.clear();
    table.candidate.extend_from_slice(&table.nodes);
    let i = table.candidate.partition_point(|node| node.x < n.x);
    table.candidate.insert(i, n);
    table.pending.clear();
    for j in 0..table.candidate.len() {
        let p = if j.abs_diff(i) <= 1 {
            piece_at(&table.candidate, j, bounds, t, epsilon, argument)?
        } else {
            table.pieces[if j > i { j - 1 } else { j }]
        };
        table.pending.push(p);
    }
    let total = normalize(&mut table.pending, argument)?;
    core::mem::swap(&mut table.nodes, &mut table.candidate);
    core::mem::swap(&mut table.pieces, &mut table.pending);
    table.log_total = total;
    table.padding = table.padding.max(64.0 * epsilon * (1.0 + n.log.abs()));
    Ok(())
}
fn inverse_piece(
    p: Piece,
    q: f64,
    t: DensityTransform,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    let n = p.node;
    let a = p.left;
    let b = p.right;
    let s = n.slope;
    if q == 0.0 && a.is_finite() {
        return Ok(a);
    }
    if q == 1.0 && b.is_finite() {
        return Ok(b);
    }
    let x = match t {
        DensityTransform::Log => {
            if s == 0.0 {
                a * (1.0 - q) + b * q
            } else if a == f64::NEG_INFINITY {
                b + q.ln() / s
            } else if b == f64::INFINITY {
                a + (-q).ln_1p() / s
            } else if s > 0.0 {
                b + (q + (1.0 - q) * (-s * (b - a)).exp()).ln() / s
            } else {
                a + (-q * (-(s * (b - a)).exp_m1())).ln_1p() / s
            }
        }
        DensityTransform::NegativeInverseSqrt => {
            if s == 0.0 {
                a * (1.0 - q) + b * q
            } else if a == f64::NEG_INFINITY {
                b + ((-tangent(n, b)) / s) * (1.0 - 1.0 / q)
            } else if b == f64::INFINITY {
                a + ((-tangent(n, a)) / (-s)) * (1.0 / (1.0 - q) - 1.0)
            } else {
                let ya = tangent(n, a);
                let yb = tangent(n, b);
                a + q * (b - a) * ya / ((1.0 - q) * yb + q * ya)
            }
        }
    };
    if !x.is_finite() || x < a || x > b {
        return Err(invalid(argument, "invalid envelope inverse"));
    }
    Ok(x)
}
fn proposal(
    table: &Table,
    q: f64,
    t: DensityTransform,
    argument: scioxide_core::Argument,
) -> Result<(f64, usize), SciOxideError> {
    let i = table.pieces.partition_point(|p| p.cumulative <= q);
    let p = *table
        .pieces
        .get(i)
        .ok_or_else(|| invalid(argument, "envelope cumulative table does not cover uniform"))?;
    let previous = if i == 0 {
        0.0
    } else {
        table.pieces[i - 1].cumulative
    };
    let u = (q - previous) / (p.cumulative - previous);
    if !(0.0..=1.0).contains(&u) {
        return Err(invalid(
            argument,
            "envelope inverse rounded to a segment endpoint",
        ));
    }
    Ok((inverse_piece(p, u, t, argument)?, i))
}
fn hat(
    p: Piece,
    x: f64,
    t: DensityTransform,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    let y = tangent(p.node, x);
    match t {
        DensityTransform::Log => Ok(y),
        DensityTransform::NegativeInverseSqrt => {
            if y >= 0.0 {
                return Err(invalid(argument, "TDR hat crosses zero"));
            }
            Ok(-2.0 * (-y).ln())
        }
    }
}
fn squeeze(table: &Table, piece: usize, x: f64, t: DensityTransform) -> f64 {
    let i = if x < table.nodes[piece].x {
        if piece == 0 {
            return f64::NEG_INFINITY;
        }
        piece - 1
    } else {
        piece
    };
    let Some(&b) = table.nodes.get(i + 1) else {
        return f64::NEG_INFINITY;
    };
    let a = table.nodes[i];
    if x < a.x || x > b.x {
        return f64::NEG_INFINITY;
    }
    let q = (x - a.x) / (b.x - a.x);
    let y = a.y * (1.0 - q) + b.y * q;
    match t {
        DensityTransform::Log => y,
        DensityTransform::NegativeInverseSqrt => -2.0 * (-y).ln(),
    }
}

#[derive(Debug, Clone)]
struct Prepared<S: RealScalar, P> {
    provider: P,
    table: Table,
    bounds: (f64, f64),
    options: UniversalSamplingOptions,
    transform: DensityTransform,
    preparation: SamplingDiagnostics,
    scalar: PhantomData<S>,
}
impl<S: RealScalar, P: ConcaveDensity<S>> Prepared<S, P> {
    fn new(
        provider: P,
        knots: &[S],
        options: UniversalSamplingOptions,
        transform: DensityTransform,
    ) -> Result<Self, SciOxideError> {
        let argument = scioxide_core::argument!("provider");
        provider.validate_density()?;
        provider.validate_concavity(transform)?;
        let bounds = support(&provider, argument)?;
        if knots.len() < 2 || knots.len() > options.table_nodes {
            return Err(invalid(
                scioxide_core::argument!("knots"),
                "invalid adaptive knot count",
            ));
        }
        crate::workspace::admit_bytes(table_bytes(options.table_nodes)?, options.max_bytes)?;
        let mut work = Work::preparing(options);
        let mut table = reserve(options.table_nodes)?;
        for x in knots {
            if !x.is_finite()
                || x.into_f64() <= bounds.0
                || x.into_f64() >= bounds.1
                || !provider.contains(*x)?
            {
                return Err(invalid(
                    scioxide_core::argument!("knots"),
                    "adaptive knots must be inside support",
                ));
            }
            table
                .nodes
                .push(node(&provider, *x, transform, &mut work, argument)?);
        }
        table.log_total = build(
            &table.nodes,
            &mut table.pieces,
            bounds,
            transform,
            S::EPSILON.into_f64(),
            argument,
        )?;
        table.padding = padding(&table.nodes, S::EPSILON.into_f64());
        Ok(Self {
            provider,
            table,
            bounds,
            options,
            transform,
            preparation: work.diagnostics,
            scalar: PhantomData,
        })
    }
    fn workspace(&self) -> Result<AdaptiveSamplingWorkspace<'_, S>, SciOxideError> {
        crate::workspace::admit_bytes(
            workspace_bytes(self.options.table_nodes)?,
            self.options.max_bytes,
        )?;
        let mut workspace = AdaptiveSamplingWorkspace {
            table: reserve(self.options.table_nodes)?,
            transform: self.transform,
            owner: core::ptr::from_ref(self).addr(),
            scalar: PhantomData,
        };
        self.restore(&mut workspace);
        Ok(workspace)
    }
    /// Reset a workspace to the prepared table within its reserved storage.
    fn restore(&self, workspace: &mut AdaptiveSamplingWorkspace<'_, S>) {
        let table = &mut workspace.table;
        table.nodes.clear();
        table.nodes.extend_from_slice(&self.table.nodes);
        table.pieces.clear();
        table.pieces.extend_from_slice(&self.table.pieces);
        table.log_total = self.table.log_total;
        table.padding = self.table.padding;
    }
    fn draw<R: Rng<S>>(
        &self,
        rng: &mut R,
        mut adaptive: Option<&mut AdaptiveSamplingWorkspace<'_, S>>,
        argument: scioxide_core::Argument,
    ) -> Result<UniversalSample<S>, SciOxideError> {
        // Reject another plan's tables before consuming RNG. The workspace's
        // borrow prevents moving or replacing its owner while it remains in use.
        if let Some(w) = adaptive.as_deref_mut()
            && (w.owner != core::ptr::from_ref(self).addr()
                || w.transform != self.transform
                || w.table.nodes.capacity() < self.options.table_nodes)
        {
            return Err(invalid(argument, "incompatible adaptive workspace"));
        }
        let mut work = Work::sampling(self.options);
        loop {
            work.proposal()?;
            let q = work.uniform(rng, argument)?;
            let table = adaptive.as_ref().map_or(&self.table, |w| &w.table);
            let (value, piece) = proposal(table, q, self.transform, argument)?;
            let x = coordinate::<S>(value, argument)?;
            let padding = table.padding;
            let xf = x.into_f64();
            if !self.provider.contains(x)? {
                return Err(invalid(
                    argument,
                    "adaptive proposal outside density support",
                ));
            }
            let upper = hat(table.pieces[piece], xf, self.transform, argument)?;
            let threshold = upper + padding + work.uniform(rng, argument)?.ln();
            // Only a proven concavity contract justifies avoiding the callback.
            if threshold <= squeeze(table, piece, xf, self.transform) - padding {
                return Ok(UniversalSample {
                    value: x,
                    diagnostics: work.diagnostics,
                });
            }
            let log = log_density(&self.provider, x, &mut work, argument)?;
            if log > upper + padding {
                return Err(invalid(argument, "density exceeds adaptive envelope"));
            }
            let accepted = threshold <= log;
            if let Some(w) = adaptive.as_deref_mut()
                && log.is_finite()
                // Remote TDR tail knots can make intersections ill-conditioned.
                // Retain the already valid hat when the transformed scale differs
                // from every existing knot by more than 1/sqrt(epsilon).
                && (self.transform == DensityTransform::Log
                    || w.table.nodes.iter().any(|n| (n.log - log).abs() <= -S::EPSILON.into_f64().ln()))
                && w.table.nodes.len() < self.options.table_nodes
                && !w
                    .table
                    .nodes
                    .iter()
                    .any(|n| (n.x - xf).abs() <= 64.0 * S::EPSILON.into_f64() * (1.0 + xf.abs()))
            {
                // Reuse the already evaluated log density; only derivative costs another call.
                work.evaluation()?;
                let derivative = self.provider.log_density_derivative(x)?.into_f64();
                if !derivative.is_finite() {
                    return Err(invalid(argument, "nonfinite adaptive derivative"));
                }
                let (y, slope) = match self.transform {
                    DensityTransform::Log => (log, derivative),
                    DensityTransform::NegativeInverseSqrt => {
                        let y = -(-0.5 * log).exp();
                        (y, -0.5 * y * derivative)
                    }
                };
                if !y.is_finite() || !slope.is_finite() {
                    return Err(invalid(argument, "unrepresentable adaptive node"));
                }
                let n = Node {
                    x: xf,
                    log,
                    y,
                    slope,
                };
                // Refinement is optional: geometric/rounding failures keep the
                // valid previous hat. Callback and work errors above still propagate.
                let _ = refine(
                    &mut w.table,
                    n,
                    self.bounds,
                    self.transform,
                    S::EPSILON.into_f64(),
                    argument,
                );
            }
            if accepted {
                return Ok(UniversalSample {
                    value: x,
                    diagnostics: work.diagnostics,
                });
            }
            work.rejected();
        }
    }
}

macro_rules! sampler {
    ($(#[$attr:meta])* $name:ident, $transform:expr) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        pub struct $name<S: RealScalar, P> {
            prepared: Prepared<S, P>,
        }
        impl<S: RealScalar, P: ConcaveDensity<S>> $name<S, P> {
            /// Creates the sampler with an envelope of tangents at the points `knots`.
            ///
            /// The knots must be increasing, at least two, at most `table_nodes`, and inside the
            /// support. Points on both sides of the mode give an integrable envelope on an
            /// unbounded support.
            ///
            /// It uses the default [`UniversalSamplingOptions`](crate::UniversalSamplingOptions);
            /// [`Self::with_options`] sets the limits.
            ///
            /// # Errors
            ///
            /// - The errors of `provider.validate_density()`,
            ///   `provider.validate_concavity(T)` and `provider.support()`.
            /// - `InvalidInput` if the knot count is outside `2..=table_nodes`, a knot is not
            ///   finite or not inside the support, the knots are not increasing, the log density
            ///   or its derivative at a knot is not finite, the derivatives contradict
            ///   concavity, or the envelope has infinite mass.
            /// - `ResourceLimit` if the tables exceed `max_bytes`, and `ConvergenceFailure` if
            ///   the preparation evaluation limit is exhausted.
            /// - Errors of the density and its derivative, unchanged.
            pub fn new(provider: P, knots: &[S]) -> Result<Self, SciOxideError> {
                Self::with_options(provider, knots, UniversalSamplingOptions::default())
            }

            /// Creates the sampler of [`Self::new`] with the limits of `options`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::new`].
            /// - `InvalidInput` if `options` is invalid; see [`UniversalSamplingOptions`](crate::UniversalSamplingOptions).
            pub fn with_options(
                provider: P,
                knots: &[S],
                options: UniversalSamplingOptions,
            ) -> Result<Self, SciOxideError> {
                Ok(Self {
                    prepared: Prepared::new(provider, knots, options, $transform)?,
                })
            }
            /// Returns the work done while creating the sampler.
            pub fn preparation_diagnostics(&self) -> SamplingDiagnostics {
                self.prepared.preparation
            }
            /// Returns an adaptive workspace that starts from the initial envelope.
            ///
            /// # Errors
            ///
            /// - `ResourceLimit` if the workspace would exceed `max_bytes`.
            /// - `AllocationFailure` if it cannot be allocated.
            pub fn workspace(&self) -> Result<AdaptiveSamplingWorkspace<'_, S>, SciOxideError> {
                self.prepared.workspace()
            }
            /// Draws one sample with the initial envelope, without refinement, and reports the
            /// work it took.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if the density exceeds the envelope or is NaN or `+∞` at a
            ///   proposal, which means the concavity guarantee is false.
            /// - `ConvergenceFailure` if 1,024 proposals are rejected or an evaluation or
            ///   uniform-draw limit is exhausted.
            /// - Errors of the density and of `rng`, unchanged.
            pub fn sample_report<R: Rng<S>>(
                &self,
                rng: &mut R,
            ) -> Result<UniversalSample<S>, SciOxideError> {
                self.prepared
                    .draw(rng, None, scioxide_core::argument!("self"))
            }
            /// Draws one sample with the envelope in `workspace`, refining it at each density
            /// evaluation.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `workspace` was created by another sampler, or a derivative
            ///   at a new tangent point is not finite.
            /// - The errors of [`Self::sample_report`].
            pub fn sample_adaptive<R: Rng<S>>(
                &self,
                rng: &mut R,
                workspace: &mut AdaptiveSamplingWorkspace<'_, S>,
            ) -> Result<UniversalSample<S>, SciOxideError> {
                self.prepared
                    .draw(rng, Some(workspace), scioxide_core::argument!("self"))
            }
            /// Fills `output` with samples drawn in sequence by [`Self::sample_adaptive`].
            ///
            /// The output and generator state follow `policy` (see
            /// [batch sampling](crate#batch-sampling)). Refinements made before an error stay
            /// in `adaptive`, even under [`Rollback`](crate::Rollback).
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::sample_adaptive`].
            /// - `ResourceLimit` if `staging` would exceed `max_bytes`, and `AllocationFailure`
            ///   if it must grow and cannot.
            pub fn sample_adaptive_into<R: Rng<S>>(
                &self,
                rng: &mut R,
                policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
                output: &mut [S],
                adaptive: &mut AdaptiveSamplingWorkspace<'_, S>,
                staging: &mut crate::SamplingWorkspace<S>,
            ) -> Result<(), SciOxideError> {
                crate::workspace::sample_scalars(
                    rng,
                    policy,
                    output,
                    S::ZERO,
                    staging,
                    self.prepared
                        .options
                        .max_bytes
                        .saturating_sub(workspace_bytes(self.prepared.options.table_nodes)?),
                    |rng| Ok(self.sample_adaptive(rng, adaptive)?.value),
                )
            }
            /// Fills `output` with independent samples, sample `i` drawn from the stream
            /// `root.split(i)`.
            ///
            /// Each sample starts from the initial envelope, so the samples do not depend on
            /// the options' execution policy, which only schedules them. `root` does not
            /// advance. The output is written only if every sample succeeds.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::sample_adaptive`], of `root.split`, and of `cx`
            ///   (cancellation).
            /// - `ResourceLimit` if storage would exceed `max_bytes`, and `AllocationFailure` if
            ///   it cannot be allocated.
            pub fn sample_indexed_into<R: Rng<S> + crate::SplittableRng + Send + Sync>(
                &self,
                root: &R,
                output: &mut [S],
                staging: &mut crate::SamplingWorkspace<S>,
                cx: scioxide_core::ExecutionContext<'_>,
            ) -> Result<(), SciOxideError>
            where
                P: Sync,
            {
                let table_bytes = table_bytes(self.prepared.options.table_nodes)?;
                indexed(
                    root,
                    output,
                    staging,
                    cx,
                    self.prepared.options,
                    table_bytes,
                    table_bytes,
                    || self.prepared.workspace(),
                    |rng, scratch| {
                        self.prepared.restore(scratch);
                        Ok(self.sample_adaptive(rng, scratch)?.value)
                    },
                )
            }
            /// Returns the quantile at `u` of the normalized initial envelope (the proposal
            /// density), not of the target law.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `u` is not in `(0, 1)` or the quantile is not representable in
            ///   `S`.
            pub fn proposal_quantile(&self, u: S) -> Result<S, SciOxideError> {
                if !u.is_finite() || u <= S::ZERO || u >= S::ONE {
                    return Err(invalid(
                        scioxide_core::argument!("u"),
                        "hat inverse requires 0 < u < 1",
                    ));
                }
                coordinate(
                    proposal(
                        &self.prepared.table,
                        u.into_f64(),
                        $transform,
                        scioxide_core::argument!("self"),
                    )?.0,
                    scioxide_core::argument!("self"),
                )
            }
        }
        impl<S: RealScalar, P: ConcaveDensity<S>> Distribution<S> for $name<S, P> {
            scalar_batches!(
                self,
                self.prepared.options,
                table_bytes(self.prepared.options.table_nodes)?
            );
            fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
                Ok(self.sample_report(rng)?.value)
            }
        }
    };
}
sampler!(
    /// Sampler for a log-concave density you supply, by adaptive rejection sampling.
    ///
    /// For a density `f` with `ln f` concave, the tangents of `ln f` at points `x₁ < … < xₘ`
    /// lie above `ln f`. Their minimum, exponentiated, is a piecewise exponential envelope that
    /// is sampled exactly by inversion. The chords between neighboring points lie below `ln f`
    /// and give a squeeze that accepts many proposals without evaluating `f`. Each proposal
    /// where `f` is evaluated adds a tangent point to an [`AdaptiveSamplingWorkspace`],
    /// so the rejection rate falls as sampling goes on (Gilks and Wild 1992). A proposal costs
    /// O(log m) plus at most one density and one derivative evaluation, and a refinement costs
    /// O(m). The density may be unnormalized.
    ///
    /// Use [`TransformedDensitySampler`] for densities that are `−1/√f`-concave but not
    /// log-concave, such as heavy-tailed ones.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExtendedInterval;
    /// use scioxide_random::{AdaptiveRejectionSampler, ConcaveDensityFn, DensityTransform};
    /// use scioxide_random::{Distribution, SeedableRng, SplitMix64};
    /// // Unnormalized standard normal density exp(−x²/2): ln f = −x²/2, (ln f)′ = −x.
    /// let (log_pdf, derivative) = (|x: f64| Ok(-0.5 * x * x), |x: f64| Ok(-x));
    /// let line = ExtendedInterval::new(None, None)?;
    /// let density = ConcaveDensityFn::new(log_pdf, derivative, DensityTransform::Log, line);
    /// let sampler = AdaptiveRejectionSampler::new(density, &[-1.0, 0.0, 1.0])?;
    /// let draws = sampler.sample_n(&mut SplitMix64::from_seed(7), 256)?;
    /// // The sample mean has standard error 1/16; allow about five of them.
    /// assert!((draws.iter().sum::<f64>() / 256.0).abs() < 0.3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// W. R. Gilks and P. Wild, "Adaptive rejection sampling for Gibbs sampling", *J. R. Stat.
    /// Soc. Ser. C (Appl. Stat.)* 41(2), 1992, 337–348.
    AdaptiveRejectionSampler,
    DensityTransform::Log
);
sampler!(
    /// Sampler for a `T`-concave density you supply, by transformed density rejection with
    /// `T(f) = −1/√f`.
    ///
    /// For a density `f` with `−1/√f` concave, the tangents of `−1/√f` at points `x₁ < … < xₘ`
    /// lie above it, and transforming their minimum back gives an envelope of pieces
    /// proportional to `1/(a + b x)²` that is sampled exactly by inversion (Hörmann 1995). Every
    /// log-concave density is `−1/√f`-concave, and so are heavier-tailed densities such as
    /// Student's t with `ν ≥ 1`. Squeeze, refinement and cost are as for
    /// [`AdaptiveRejectionSampler`], whose methods this type shares.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExtendedInterval, SciOxideError};
    /// use scioxide_random::{
    ///     ConcaveDensity, DensityTransform, LogDensity, SeedableRng, SplitMix64,
    ///     Support, TransformedDensitySampler, UniversalSamplingOptions, UserDensity,
    /// };
    /// /// Unnormalized Epanechnikov density `1 - x^2` on `[-1, 1]`.
    /// struct Epanechnikov;
    /// impl Support<f64> for Epanechnikov {
    ///     fn support(&self) -> Result<ExtendedInterval<f64>, SciOxideError> {
    ///         ExtendedInterval::new(Some(-1.0), Some(1.0))
    ///     }
    /// }
    /// impl LogDensity<f64> for Epanechnikov {
    ///     fn log_pdf(&self, x: f64) -> Result<f64, SciOxideError> { Ok((1.0 - x * x).ln()) }
    /// }
    /// impl UserDensity<f64> for Epanechnikov {
    ///     fn validate_density(&self) -> Result<(), SciOxideError> { Ok(()) } // mass 4/3
    /// }
    /// impl ConcaveDensity<f64> for Epanechnikov {
    ///     fn log_density_derivative(&self, x: f64) -> Result<f64, SciOxideError> {
    ///         Ok(-2.0 * x / (1.0 - x * x))
    ///     }
    ///     // f is concave, hence log-concave and -1/sqrt(f)-concave.
    ///     fn validate_concavity(&self, _: DensityTransform) -> Result<(), SciOxideError> { Ok(()) }
    /// }
    /// let sampler = TransformedDensitySampler::new(
    ///     Epanechnikov,
    ///     &[-0.5, 0.0, 0.5],
    /// )?;
    /// let mut adaptive = sampler.workspace()?;
    /// let mut rng = SplitMix64::from_seed(7);
    /// let draw = sampler.sample_adaptive(&mut rng, &mut adaptive)?;
    /// assert!((-1.0..=1.0).contains(&draw.value));
    /// assert!(sampler.preparation_diagnostics().evaluations > 0);
    /// # Ok::<(), SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - W. Hörmann, "A rejection technique for sampling from T-concave distributions", *ACM
    ///   Trans. Math. Softw.* 21(2), 1995, 182–193.
    /// - W. Hörmann, J. Leydold and G. Derflinger, *Automatic Nonuniform Random Variate
    ///   Generation*, Springer, 2004, Ch. 4.
    TransformedDensitySampler,
    DensityTransform::NegativeInverseSqrt
);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod table_tests {
    use super::*;
    fn gaussian(x: f64) -> Node {
        Node {
            x,
            log: -0.5 * x * x,
            y: -0.5 * x * x,
            slope: -x,
        }
    }
    #[test]
    fn local_refinement_matches_rebuild_and_failed_candidate_preserves_hat() {
        let argument = scioxide_core::argument!("provider");
        let bounds = (f64::NEG_INFINITY, f64::INFINITY);
        let mut table = reserve(16).unwrap();
        table.nodes.extend([-1.0, 1.0].map(gaussian));
        table.log_total = build(
            &table.nodes,
            &mut table.pieces,
            bounds,
            DensityTransform::Log,
            f64::EPSILON,
            argument,
        )
        .unwrap();
        table.padding = padding(&table.nodes, f64::EPSILON);
        for x in [0.0, 0.3, -3.0, 5.0] {
            refine(
                &mut table,
                gaussian(x),
                bounds,
                DensityTransform::Log,
                f64::EPSILON,
                argument,
            )
            .unwrap();
            let mut expected = Vec::new();
            let total = build(
                &table.nodes,
                &mut expected,
                bounds,
                DensityTransform::Log,
                f64::EPSILON,
                argument,
            )
            .unwrap();
            assert_eq!(expected, table.pieces);
            assert_eq!(total, table.log_total);
            assert_eq!(table.padding, padding(&table.nodes, f64::EPSILON));
        }
        let before = table.clone();
        let mut invalid = gaussian(0.5);
        invalid.slope = 100.0;
        assert!(
            refine(
                &mut table,
                invalid,
                bounds,
                DensityTransform::Log,
                f64::EPSILON,
                argument
            )
            .is_err()
        );
        assert_eq!(table.nodes, before.nodes);
        assert_eq!(table.pieces, before.pieces);
        assert_eq!(table.log_total, before.log_total);
        assert_eq!(table.padding, before.padding);
    }
    #[test]
    fn zero_weight_segments_and_roundoff_slopes_remain_valid() {
        let argument = scioxide_core::argument!("provider");
        let mut table = reserve(8).unwrap();
        table
            .nodes
            .extend([-2000.0, -1.0, 1.0, 2000.0].map(gaussian));
        table.log_total = build(
            &table.nodes,
            &mut table.pieces,
            (f64::NEG_INFINITY, f64::INFINITY),
            DensityTransform::Log,
            f64::EPSILON,
            argument,
        )
        .unwrap();
        assert_eq!(table.pieces[0].cumulative, 0.0);
        for q in [0.0, 0.5, 1.0 - f64::EPSILON] {
            assert!(
                proposal(&table, q, DensityTransform::Log, argument)
                    .unwrap()
                    .0
                    .is_finite()
            );
        }
        let nodes = [
            Node {
                x: 0.25,
                log: 0.0,
                y: 0.0,
                slope: 0.0,
            },
            Node {
                x: 0.75,
                log: 0.0,
                y: 0.0,
                slope: f64::EPSILON,
            },
        ];
        assert!(
            build(
                &nodes,
                &mut table.pieces,
                (0.0, 1.0),
                DensityTransform::Log,
                f64::EPSILON,
                argument
            )
            .is_ok()
        );
    }
}
