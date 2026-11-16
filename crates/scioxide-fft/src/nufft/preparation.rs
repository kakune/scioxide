//! NUFFT plan preparation in named phases.
//!
//! `validate` checks the request. `select` scans coordinate bounds without
//! allocating, into width-independent `Spans`, then searches the smallest
//! kernel width whose aliasing and rounding estimates meet the requested
//! accuracy, checking kernel, stencil and grid limits for each candidate.
//! `budget` checks storage and workspace limits before any allocation; `axes`
//! centres the checked axes, `children` prepares their FFTs, and mode tables,
//! stencils and scaling complete the plan in `prepare`.
use super::*;

/// A validated transform with its point counts, rank and accuracy target.
struct Request<'r, 'a, 'l, S: RealScalar> {
    transform: &'r NufftTransform<'a, 'l, S>,
    points: &'r NdView<'a, 'l, S>,
    modes: &'a [usize],
    frequencies: Option<&'r NdView<'a, 'l, S>>,
    count: usize,
    target_count: usize,
    rank: usize,
    eps: f64,
    sigma: f64,
}

impl<S: RealScalar> Request<'_, '_, '_, S> {
    const fn is_three(&self) -> bool {
        self.frequencies.is_some()
    }
    /// Input and output record lengths.
    const fn lens(&self) -> (usize, usize) {
        if let NufftTransform::Type2 { .. } = self.transform {
            (self.target_count, self.count)
        } else {
            (self.count, self.target_count)
        }
    }
}

fn validate<'r, 'a, 'l, S: RealScalar>(
    transform: &'r NufftTransform<'a, 'l, S>,
    spec: &NufftPlanSpec<S>,
    context: crate::FftPlanningContext<'_, S>,
) -> Result<Request<'r, 'a, 'l, S>, SciOxideError> {
    let (points, modes, frequencies) = match transform {
        NufftTransform::Type1 { points, modes } | NufftTransform::Type2 { points, modes } => {
            (points, *modes, None)
        }
        NufftTransform::Type3 {
            points,
            frequencies,
        } => (points, &[][..], Some(frequencies)),
    };
    if !spec.accuracy.is_finite() || !spec.oversampling.is_finite() {
        return Err(nonfinite());
    }
    if spec.accuracy <= S::ZERO || spec.accuracy >= S::ONE {
        return Err(invalid(
            scioxide_core::InvalidInputReason::OutOfRange,
            scioxide_core::argument!("spec.accuracy"),
            "accuracy must be finite and strictly between zero and one",
        ));
    }
    context.execution().check()?;
    let _ = context.resolve(spec.backend_strategy)?;
    let (count, rank) = geometry(points)?;
    let target_count = if let Some(f) = frequencies {
        let (n, d) = geometry(f)?;
        crate::validation::length(d, rank)?;
        n
    } else {
        crate::validation::length(modes.len(), rank)?;
        product(modes, scioxide_core::quantity!("nufft_mode_count"))?
    };
    let eps = spec.accuracy.try_to_f64_exact()?;
    let sigma = spec.oversampling;
    if sigma < 2.0 {
        return Err(invalid(
            scioxide_core::InvalidInputReason::OutOfRange,
            scioxide_core::argument!("spec.oversampling"),
            "oversampling must be finite and at least two",
        ));
    }
    Ok(Request {
        transform,
        points,
        modes,
        frequencies,
        count,
        target_count,
        rank,
        eps,
        sigma,
    })
}

fn geometry<S: RealScalar>(points: &NdView<'_, '_, S>) -> Result<(usize, usize), SciOxideError> {
    let shape = points.layout().shape().extents();
    crate::validation::length(shape.len(), 2)?;
    // The view has already proved its backing span. Walk its two strides
    // directly instead of decoding a general N-D logical index per value.
    if shape[0] == 0 || shape[1] == 0 {
        return Ok((shape[0], shape[1]));
    }
    let layout = points.layout();
    let strides = layout.strides();
    for j in 0..shape[0] {
        for d in 0..shape[1] {
            if !points.storage()[j * strides[0] + d * strides[1]].is_finite() {
                return Err(nonfinite());
            }
        }
    }
    Ok((shape[0], shape[1]))
}
fn coordinate<S: RealScalar>(
    points: &NdView<'_, '_, S>,
    j: usize,
    d: usize,
) -> Result<f64, SciOxideError> {
    // All callers iterate the validated [points, dimensions] shape.
    let layout = points.layout();
    let strides = layout.strides();
    points.storage()[j * strides[0] + d * strides[1]].try_to_f64_exact()
}
fn bounds<S: RealScalar>(
    points: &NdView<'_, '_, S>,
    d: usize,
) -> Result<(f64, f64), SciOxideError> {
    let count = points.layout().shape().extents()[0];
    if count == 0 {
        return Ok((0.0, 0.0));
    }
    let first = coordinate(points, 0, d)?;
    let (mut low, mut high) = (first, first);
    for j in 1..count {
        let x = coordinate(points, j, d)?;
        low = low.min(x);
        high = high.max(x);
    }
    // Halve before adding so the midpoint of extreme finite bounds cannot
    // overflow; the radius is measured to both ends to absorb rounding of
    // the midpoint.
    let center = low * 0.5 + high * 0.5;
    let radius = (high - center).max(center - low);
    if !radius.is_finite() {
        return Err(crate::validation::arithmetic_overflow(
            "coordinate box cannot be represented",
        ));
    }
    Ok((center, radius))
}

/// Width-independent extent of one axis.
#[derive(Clone, Copy)]
enum Span {
    /// Type 1/2: the requested mode count.
    Modes(usize),
    /// Type 3 with a degenerate point or frequency box: a constant phase.
    Constant,
    /// Type 3: the rescaled point-box radius in base-grid cells.
    Box(usize),
}

impl Span {
    // Types 1/2: the base extent is the number of requested modes. Type 3: the
    // base grid holds the rescaled point box (2 * radius) plus one stencil width.
    fn extents(self, width: usize, sigma: f64) -> Result<(usize, usize), SciOxideError> {
        let base = match self {
            Self::Modes(0) => {
                return Err(invalid(
                    scioxide_core::InvalidInputReason::Empty,
                    scioxide_core::argument!("modes"),
                    "mode extents must be positive",
                ));
            }
            Self::Modes(base) => base,
            Self::Constant => 1,
            Self::Box(radius) => {
                let q = scioxide_core::quantity!("nufft_type3_base_extent");
                checked_add(checked_mul(radius, 2, q)?, checked_add(width, 1, q)?, q)?
            }
        };
        Ok((base, oversampled(base, width, sigma)?))
    }
}

// Oversample by sigma (this is what makes the Kaiser–Bessel aliasing error
// decay at `rate`), keep the grid larger than one stencil so a stencil never
// wraps onto itself, and round up to a power of two so the per-axis FFT is pure
// radix-2/4. A single base cell needs no grid.
fn oversampled(base: usize, width: usize, sigma: f64) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("nufft_grid_extent");
    if base == 1 {
        return Ok(1);
    }
    let minimum = integer((base as f64 * sigma).ceil(), q)?.max(checked_add(width, 2, q)?);
    let grid = minimum
        .checked_next_power_of_two()
        .ok_or_else(|| SciOxideError::size_overflow(q))?;
    // All floating indices and signed offsets below must remain exact.
    if grid > 1_usize << (usize::BITS.min(52) - 2) {
        return Err(SciOxideError::size_overflow(q));
    }
    Ok(grid)
}

/// Centred and rescaled geometry of one axis.
#[derive(Clone, Copy)]
struct Frame {
    span: Span,
    source_center: f64,
    frequency_center: f64,
    rescale: f64,
}

/// Prepared axis geometry for one kernel width.
#[derive(Clone, Copy)]
struct Axis {
    base: usize,
    grid: usize,
    source_center: f64,
    frequency_center: f64,
    rescale: f64,
}

// Type 3: points and frequencies are centred on their boxes, and the points are
// rescaled by sr * sigma / pi so that, after centering, the frequency box maps
// into the oversampled grid's periodic band. A zero radius on either side makes
// the axis a constant phase (grid 1).
fn frame<S: RealScalar>(
    request: &Request<'_, '_, '_, S>,
    d: usize,
) -> Result<Frame, SciOxideError> {
    let Some(frequencies) = request.frequencies else {
        return Ok(Frame {
            span: Span::Modes(request.modes[d]),
            source_center: 0.0,
            frequency_center: 0.0,
            rescale: 1.0,
        });
    };
    let (source_center, xr) = bounds(request.points, d)?;
    let (frequency_center, sr) = bounds(frequencies, d)?;
    let rescale = if xr == 0.0 || sr == 0.0 {
        0.0
    } else {
        sr * (request.sigma / PI)
    };
    if !rescale.is_finite() {
        return Err(crate::validation::arithmetic_overflow(
            "type-3 rescaling overflow",
        ));
    }
    let span = if rescale == 0.0 {
        Span::Constant
    } else {
        Span::Box(integer(
            (xr * rescale).ceil(),
            scioxide_core::quantity!("nufft_type3_base_extent"),
        )?)
    };
    Ok(Frame {
        span,
        source_center,
        frequency_center,
        rescale,
    })
}

impl Frame {
    fn axis(self, width: usize, sigma: f64) -> Result<Axis, SciOxideError> {
        let (base, grid) = self.span.extents(width, sigma)?;
        Ok(Axis {
            base,
            grid,
            source_center: self.source_center,
            frequency_center: self.frequency_center,
            rescale: self.rescale,
        })
    }
}

/// Capacity for non-constant type-3 spans: each multiplies stencil visits by a
/// width of at least four, so more than `usize::BITS / 2` overflow the visit count.
const SPAN_CAPACITY: usize = usize::BITS as usize / 2;

/// Width-independent search extents from one coordinate scan without allocation.
///
/// Type 1/2 spans are the mode counts. Type 3 run-length encodes its axes in
/// order: each non-constant radius with the number of constant axes before it,
/// then the trailing constant count.
#[derive(Clone, Copy)]
struct Spans<'a> {
    modes: &'a [usize],
    runs: [(usize, usize); SPAN_CAPACITY],
    runs_len: usize,
    trailing: usize,
    /// Summed mode extents of type-1/2 axes needing periodic argument reduction.
    phase_scale: f64,
}

impl<'a> Spans<'a> {
    fn scan<S: RealScalar>(request: &Request<'_, 'a, '_, S>) -> Result<Self, SciOxideError> {
        let mut spans = Self {
            modes: &[],
            runs: [(0, 0); SPAN_CAPACITY],
            runs_len: 0,
            trailing: 0,
            phase_scale: 0.0,
        };
        if !request.is_three() {
            spans.modes = request.modes;
            for (d, &modes) in request.modes.iter().enumerate() {
                if (0..request.count)
                    .any(|j| coordinate(request.points, j, d).is_ok_and(|x| x.abs() > PI))
                {
                    spans.phase_scale += modes as f64;
                }
            }
            return Ok(spans);
        }
        for d in 0..request.rank {
            if let Span::Box(radius) = frame(request, d)?.span {
                *spans.runs.get_mut(spans.runs_len).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("nufft_stencil_visits"))
                })? = (spans.trailing, radius);
                spans.runs_len += 1;
                spans.trailing = 0;
            } else {
                spans.trailing += 1;
            }
        }
        Ok(spans)
    }
    /// Every axis span in coordinate-axis order.
    fn iter(&self) -> impl Iterator<Item = Span> + '_ {
        let constants = |n| core::iter::repeat_n(Span::Constant, n);
        self.modes
            .iter()
            .map(|&m| Span::Modes(m))
            .chain(
                self.runs[..self.runs_len]
                    .iter()
                    .flat_map(move |&(n, radius)| {
                        constants(n).chain(core::iter::once(Span::Box(radius)))
                    }),
            )
            .chain(constants(self.trailing))
    }
    /// Base and grid extents of every searched axis at `width`.
    fn extents(
        &self,
        width: usize,
        sigma: f64,
    ) -> impl Iterator<Item = Result<(usize, usize), SciOxideError>> + '_ {
        self.iter().map(move |span| span.extents(width, sigma))
    }
}

/// Grid, stencil and tile sizes for one kernel width, computed without allocation.
#[derive(Clone, Copy)]
struct Sizing {
    visits: usize,
    point_total: usize,
    stencil_visits: usize,
    grid_len: usize,
    base_len: usize,
    lane_len: usize,
    source_bins: usize,
    target_bins: usize,
    tile_len: usize,
}

fn size<S: RealScalar>(
    request: &Request<'_, '_, '_, S>,
    spans: &Spans<'_>,
    width: usize,
    limits: NufftLimits,
) -> Result<Sizing, SciOxideError> {
    let sigma = request.sigma;
    let visits = spans.extents(width, sigma).try_fold(1, |v, extents| {
        let factor = if extents?.1 == 1 { 1 } else { width };
        checked_mul(v, factor, scioxide_core::quantity!("nufft_stencil_visits"))
    })?;
    let point_total = checked_add(
        request.count,
        if request.is_three() {
            request.target_count
        } else {
            0
        },
        scioxide_core::quantity!("nufft_point_count"),
    )?;
    let stencil_visits = checked_mul(
        point_total,
        visits,
        scioxide_core::quantity!("nufft_stencil_visits"),
    )?;
    admit_bytes(
        stencil_visits,
        limits.stencil_visits,
        scioxide_core::quantity!("stencil_visits"),
    )?;
    let (mut grid_len, mut base_len, mut lane_len) = (1, 1, 1);
    let (mut base_extents, mut grid_extents) = ([1; 3], [1; 3]);
    for (d, extents) in spans.extents(width, sigma).enumerate() {
        let (base, grid) = extents?;
        grid_len = checked_mul(
            grid_len,
            grid,
            scioxide_core::quantity!("nufft_grid_points"),
        )?;
        base_len = checked_mul(
            base_len,
            base,
            scioxide_core::quantity!("nufft_base_grid_points"),
        )?;
        lane_len = lane_len.max(grid);
        if d < 3 {
            base_extents[d] = base;
            grid_extents[d] = grid;
        }
    }
    let q = scioxide_core::quantity!("grid_points");
    admit_bytes(grid_len, limits.grid_points, q)?;
    admit_bytes(base_len, limits.grid_points, q)?;
    let dense = request.rank == 3;
    let source_bins = if dense {
        let extents = if request.is_three() {
            base_extents
        } else {
            grid_extents
        };
        spread_bins(extents, request.count)
    } else {
        0
    };
    let target_bins = if dense && request.is_three() {
        spread_bins(grid_extents, request.target_count)
    } else {
        0
    };
    let tile_len = if source_bins + target_bins > 0 {
        let q = scioxide_core::quantity!("nufft_tile");
        let side = checked_add(width, SPREAD_BIN - 1, q)?;
        checked_mul(checked_mul(side, side, q)?, side, q)?
    } else {
        0
    };
    Ok(Sizing {
        visits,
        point_total,
        stencil_visits,
        grid_len,
        base_len,
        lane_len,
        source_bins,
        target_bins,
        tile_len,
    })
}

// Compensated summation and separable FFT stages have logarithmic, not
// point-count, rounding growth. Coordinate construction is in f64, including
// for f32 transforms; phase sensitivity uses axis extents.
fn roundoff<S: RealScalar>(
    sizing: &Sizing,
    phase_scale: f64,
    dimensions: f64,
) -> Result<f64, SciOxideError> {
    let local = if sizing.tile_len > 0 {
        spread_chunk::<S>() as f64 + 4.0
    } else {
        4.0
    };
    Ok(
        S::EPSILON.try_to_f64_exact()? * (local + (sizing.grid_len.max(2) as f64).log2())
            + f64::EPSILON * (2.0 * phase_scale + 8.0 * dimensions),
    )
}

/// The selected kernel width with its estimates, sizes and search extents.
struct Selection<'a> {
    width: usize,
    approximation: f64,
    roundoff: f64,
    sizing: Sizing,
    spans: Spans<'a>,
}

/// Search the smallest kernel width meeting the accuracy request.
///
/// The search starts at the aliasing-only width and widens while the
/// geometry-dependent rounding floor still leaves room below the request;
/// a floor at or above the request is a numerical failure. Nothing is allocated.
fn select<'a, S: RealScalar>(
    request: &Request<'_, 'a, '_, S>,
    limits: NufftLimits,
) -> Result<Selection<'a>, SciOxideError> {
    let sigma = request.sigma;
    let dimensions = request.rank.max(1) as f64;
    // Kaiser–Bessel exponential aliasing rate with a practical prefactor;
    // this is a regression-qualified estimate, not Fourmont's certificate.
    // Compose axis errors multiplicatively; type 3 has two gridding stages.
    let alias = |w: usize| 3.0 * (-PI * (w - 1) as f64 * (1.0 - 1.0 / sigma).sqrt()).exp();
    let stages = |x: f64| x * dimensions * if request.is_three() { 2.0 } else { 1.0 };
    let admit = |w: usize| {
        admit_bytes(
            w,
            limits.kernel_support,
            scioxide_core::quantity!("kernel_support"),
        )
    };
    let mut width = 4;
    while stages(alias(width)) > request.eps {
        width += 1;
    }
    admit(width)?;
    let spans = Spans::scan(request)?;
    loop {
        let sizing = size(request, &spans, width, limits)?;
        let approximation = stages(alias(width).ln_1p()).exp_m1();
        let roundoff = roundoff::<S>(&sizing, spans.phase_scale, dimensions)?;
        if approximation + roundoff <= request.eps {
            return Ok(Selection {
                width,
                approximation,
                roundoff,
                sizing,
                spans,
            });
        }
        if roundoff >= request.eps {
            return Err(numerical(
                scioxide_core::NumericalIssue::LossOfSignificance,
                "requested accuracy is below the planned floating-point envelope",
            ));
        }
        // Spend only the accuracy remaining after this geometry's rounding floor.
        width += 1;
        admit(width)?;
    }
}

/// Centre every axis for the selected width, after all limit checks.
fn axes<S: RealScalar>(
    request: &Request<'_, '_, '_, S>,
    width: usize,
) -> Result<Vec<Axis>, SciOxideError> {
    let mut axes = try_with_capacity(request.rank, scioxide_core::quantity!("nufft_axes"))?;
    for d in 0..request.rank {
        axes.push(frame(request, d)?.axis(width, request.sigma)?);
    }
    Ok(axes)
}

/// Checked storage and fixed workspace figures.
struct Budget {
    own_bytes: usize,
    temporary_bytes: usize,
    fixed_scratch: usize,
    input_len: usize,
    output_len: usize,
}

/// Resident (`own`) and simultaneously live temporary preparation bytes.
fn storage<S: RealScalar>(
    request: &Request<'_, '_, '_, S>,
    selection: &Selection<'_>,
) -> Result<(usize, usize), SciOxideError> {
    let q = scioxide_core::quantity!("nufft_preparation_bytes");
    let (rank, three) = (request.rank, request.is_three());
    let Sizing {
        point_total,
        grid_len,
        base_len,
        source_bins,
        target_bins,
        ..
    } = selection.sizing;
    let point_axes = checked_mul(point_total, rank, q)?;
    let mode_entries = selection
        .spans
        .extents(selection.width, request.sigma)
        .try_fold(0, |sum, extents| checked_add(sum, extents?.0, q))?;
    let stencil_count = if three { 2 } else { 1 };
    let metadata_bytes = checked_add(
        size_of::<NufftPlan<S>>(),
        checked_mul(
            rank,
            3 * size_of::<usize>()
                + size_of::<ComplexCore<S>>()
                + size_of::<ModeAxis<S>>()
                + stencil_count * size_of::<StencilAxis>(),
            q,
        )?,
        q,
    )?;
    let mapping_bytes = checked_mul(mode_entries, size_of::<usize>() + size_of::<S>(), q)?;
    let stencil_bytes = checked_add(
        checked_mul(point_axes, size_of::<PointAxis>(), q)?,
        checked_mul(
            point_total,
            if three { size_of::<Complex<S>>() } else { 0 } + size_of::<usize>(),
            q,
        )?,
        q,
    )?;
    let own_bytes = [
        metadata_bytes,
        mapping_bytes,
        stencil_bytes,
        checked_mul(
            Kernel::coefficient_blocks(selection.width, request.eps),
            size_of::<[f64; 4]>(),
            q,
        )?,
        checked_mul(
            checked_add(source_bins, target_bins, q)?,
            size_of::<SpreadBin>(),
            q,
        )?,
    ]
    .into_iter()
    .try_fold(0, |sum, part| checked_add(sum, part, q))?;
    let bin_bytes = |grid: usize, count: usize, tiles: usize| {
        checked_mul(
            if tiles > 0 {
                tiles
            } else {
                grid.min(4096).min(count.max(1))
            },
            size_of::<usize>(),
            q,
        )
    };
    let ordering_bytes = if three {
        let target_bytes = checked_add(
            checked_mul(
                request.target_count,
                rank * size_of::<PointAxis>() + size_of::<Complex<S>>() + size_of::<usize>(),
                q,
            )?,
            checked_add(
                checked_mul(rank, size_of::<StencilAxis>(), q)?,
                checked_mul(target_bins, size_of::<SpreadBin>(), q)?,
                q,
            )?,
            q,
        )?;
        bin_bytes(grid_len, request.target_count, target_bins)?
            .max(bin_bytes(base_len, request.count, source_bins)?.saturating_sub(target_bytes))
    } else {
        bin_bytes(grid_len, request.count, source_bins)?
    };
    let temporary_bytes = checked_add(
        checked_mul(rank, size_of::<Axis>() + size_of::<usize>(), q)?,
        ordering_bytes,
        q,
    )?;
    Ok((own_bytes, temporary_bytes))
}

/// Check preparation storage and the fixed workspace before any FFT child.
fn budget<S: RealScalar>(
    request: &Request<'_, '_, '_, S>,
    selection: &Selection<'_>,
    limits: NufftLimits,
    has_provider: bool,
) -> Result<Budget, SciOxideError> {
    let q = scioxide_core::quantity!("nufft_preparation_bytes");
    let (own_bytes, temporary_bytes) = storage(request, selection)?;
    admit_bytes(
        checked_add(own_bytes, temporary_bytes, q)?,
        limits.preparation_bytes,
        scioxide_core::quantity!("preparation_bytes"),
    )?;
    let (input_len, output_len) = request.lens();
    let sizing = &selection.sizing;
    let fixed_scratch = [
        input_len.max(output_len),
        checked_mul(sizing.grid_len, 2, q)?,
        if request.is_three() {
            checked_mul(sizing.base_len, 2, q)?
        } else {
            0
        },
        sizing.lane_len,
        checked_mul(
            request.rank,
            selection.width,
            scioxide_core::quantity!("nufft_weights"),
        )?,
        sizing.tile_len,
    ]
    .into_iter()
    .try_fold(0, |sum, part| checked_add(sum, part, q))?;
    admit_bytes(
        checked_mul(fixed_scratch, size_of::<Complex<S>>(), q)?,
        limits.workspace_bytes,
        scioxide_core::quantity!("workspace_bytes"),
    )?;
    let budget = Budget {
        own_bytes,
        temporary_bytes,
        fixed_scratch,
        input_len,
        output_len,
    };
    if !has_provider {
        admit_native_children::<S>(request, selection, &budget, limits)?;
    }
    Ok(budget)
}

/// Check the peak and scratch of native power-of-two children before planning them.
fn admit_native_children<S: RealScalar>(
    request: &Request<'_, '_, '_, S>,
    selection: &Selection<'_>,
    budget: &Budget,
    limits: NufftLimits,
) -> Result<(), SciOxideError> {
    let q = scioxide_core::quantity!("nufft_preparation_bytes");
    let child = size_of::<ComplexCore<S>>();
    let before_children = live_preparation_bytes::<S>(request.rank, 0)?;
    let (mut heap, mut peak, mut scratch) = (0, 0, 0);
    for extents in selection.spans.extents(selection.width, request.sigma) {
        let (resident, child_peak, child_work) =
            ComplexCore::<S>::power_two_requirements(extents?.1)?;
        peak = peak.max(checked_add(
            checked_add(before_children, heap, q)?,
            child_peak - child,
            q,
        )?);
        heap = checked_add(heap, resident - child, q)?;
        scratch = scratch.max(child_work);
    }
    peak = peak.max(checked_add(
        checked_add(budget.own_bytes, budget.temporary_bytes, q)?,
        heap,
        q,
    )?);
    admit_bytes(
        peak,
        limits.preparation_bytes,
        scioxide_core::quantity!("preparation_bytes"),
    )?;
    admit_bytes(
        checked_mul(
            checked_add(budget.fixed_scratch, scratch, q)?,
            size_of::<Complex<S>>(),
            q,
        )?,
        limits.workspace_bytes,
        scioxide_core::quantity!("workspace_bytes"),
    )
}

/// Prepared per-axis FFT children and their extents.
struct Children<S: RealScalar> {
    shape: Vec<usize>,
    grid_shape: Vec<usize>,
    fft: Vec<ComplexCore<S>>,
    child_bytes: usize,
    preparation_bytes: usize,
    child_scratch: usize,
}

fn children<S: RealScalar>(
    axes: &[Axis],
    spec: &NufftPlanSpec<S>,
    context: crate::FftPlanningContext<'_, S>,
    budget: &Budget,
    extents: Argument,
) -> Result<Children<S>, SciOxideError> {
    let q = scioxide_core::quantity!("nufft_preparation_bytes");
    let (rank, limits, child) = (axes.len(), spec.limits, size_of::<ComplexCore<S>>());
    let mut shape = try_with_capacity(rank, scioxide_core::quantity!("nufft_mode_shape"))?;
    let mut grid_shape = try_with_capacity(rank, scioxide_core::quantity!("nufft_grid_shape"))?;
    let mut fft = try_with_capacity(rank, scioxide_core::quantity!("nufft_axis_plans"))?;
    let (mut child_bytes, mut preparation_bytes, mut child_scratch) = (0, 0, 0);
    for axis in axes {
        let live = live_preparation_bytes::<S>(rank, child_bytes)?;
        let remaining = limits.preparation_bytes.saturating_sub(live);
        let prepared = ComplexCore::with_preparation_limit(
            axis.grid,
            spec.execution_policy,
            spec.backend_strategy,
            context.with_provider_limits(crate::FftProviderLimits {
                preparation_bytes: remaining,
                workspace_bytes: limits
                    .workspace_bytes
                    .saturating_sub(budget.fixed_scratch * size_of::<Complex<S>>()),
            }),
            extents,
            scioxide_core::argument!("context"),
            remaining.saturating_add(child),
        )?;
        preparation_bytes =
            preparation_bytes.max(checked_add(live, prepared.preparation_bytes() - child, q)?);
        child_bytes = checked_add(child_bytes, prepared.plan_bytes() - child, q)?;
        admit_bytes(
            checked_add(
                checked_add(budget.own_bytes, budget.temporary_bytes, q)?,
                child_bytes,
                q,
            )?,
            limits.preparation_bytes,
            scioxide_core::quantity!("preparation_bytes"),
        )?;
        child_scratch = child_scratch.max(prepared.bounded_scratch_len());
        shape.push(axis.base);
        grid_shape.push(axis.grid);
        fft.push(prepared);
    }
    Ok(Children {
        shape,
        grid_shape,
        fft,
        child_bytes,
        preparation_bytes,
        child_scratch,
    })
}

/// Centered logical modes mapped to periodic FFT slots with diagonal deconvolution.
fn mode_axes<S: RealScalar>(
    axes: &[Axis],
    grid_strides: &[usize],
    kernel: &Kernel,
) -> Result<Vec<ModeAxis<S>>, SciOxideError> {
    let mut modes = try_with_capacity(axes.len(), scioxide_core::quantity!("nufft_mode_axes"))?;
    for (axis, &stride) in axes.iter().zip(grid_strides) {
        let mut indices =
            try_with_capacity(axis.base, scioxide_core::quantity!("nufft_mode_indices"))?;
        let mut weights: Vec<S> =
            try_with_capacity(axis.base, scioxide_core::quantity!("nufft_mode_weights"))?;
        for j in 0..axis.base {
            let k = j as i64 - (axis.base / 2) as i64;
            indices.push(k.rem_euclid(axis.grid as i64) as usize * stride);
            let weight = if axis.grid == 1 {
                1.0
            } else {
                1.0 / kernel.fourier(TAU * k as f64 / axis.grid as f64)
            };
            weights.push(scalar(weight)?);
        }
        modes.push(ModeAxis { indices, weights });
    }
    Ok(modes)
}

fn prepared_geometry<S: RealScalar>(
    request: &Request<'_, '_, '_, S>,
    axes: &[Axis],
    (grid_strides, base_strides): (&[usize], &[usize]),
    kernel: &Kernel,
    direction: FftDirection,
) -> Result<PreparedGeometry<S>, SciOxideError> {
    let sign = if direction == FftDirection::Forward {
        -1.0
    } else {
        1.0
    };
    Ok(match request.transform {
        NufftTransform::Type1 { points, .. } => {
            PreparedGeometry::One(make_stencil::<S, false, false>(
                points,
                axes,
                grid_strides,
                kernel,
                sign,
            )?)
        }
        NufftTransform::Type2 { points, .. } => {
            PreparedGeometry::Two(make_stencil::<S, false, false>(
                points,
                axes,
                grid_strides,
                kernel,
                sign,
            )?)
        }
        NufftTransform::Type3 {
            points,
            frequencies,
        } => PreparedGeometry::Three {
            source: make_stencil::<S, true, false>(points, axes, base_strides, kernel, sign)?,
            target: make_stencil::<S, true, true>(frequencies, axes, grid_strides, kernel, sign)?,
        },
    })
}

/// Operator scale and the exponent headroom reserved for bounded execution.
struct Scaling<S: RealScalar> {
    scale: S,
    headroom: i32,
    bounded: bool,
}

fn scaling<S: RealScalar>(
    spec: &NufftPlanSpec<S>,
    (input_len, output_len): (usize, usize),
    grid_shape: &[usize],
    visits: usize,
    modes: &[ModeAxis<S>],
) -> Result<Scaling<S>, SciOxideError> {
    let (forward, inverse) = spec.normalization.fourier_scales(input_len.max(1))?;
    let scale: S = if spec.direction == FftDirection::Forward {
        forward
    } else {
        inverse
    };
    let mode_bound = modes.iter().fold(scale.abs(), |bound, axis| {
        bound * axis.weights.iter().fold(S::ONE, |m, w| m.max(w.abs()))
    });
    // Spread/interpolation visit at most stencil_visits terms. Reserve
    // both stages, every grid axis, complex rotations and deconvolution.
    let records = crate::scaling::growth_exponent(input_len.max(output_len));
    let headroom = records
        + grid_shape
            .iter()
            .map(|&n| crate::scaling::growth_exponent(n))
            .sum::<i32>()
        + 2 * crate::scaling::growth_exponent(visits)
        + crate::scaling::range_exponent(mode_bound)
        + 4;
    let bounded = S::ONE.scalbn(headroom).is_finite();
    Ok(Scaling {
        scale,
        headroom: if bounded { headroom } else { records },
        bounded,
    })
}

/// Prepare a plan by composing the named phases.
pub(super) fn prepare<S: RealScalar>(
    transform: NufftTransform<'_, '_, S>,
    spec: NufftPlanSpec<S>,
    context: crate::FftPlanningContext<'_, S>,
) -> Result<NufftPlan<S>, SciOxideError> {
    let request = validate(&transform, &spec, context)?;
    let selection = select(&request, spec.limits)?;
    let budget = budget(&request, &selection, spec.limits, context.has_provider())?;
    let extents = if request.is_three() {
        scioxide_core::argument!("frequencies")
    } else {
        scioxide_core::argument!("modes")
    };
    let axes = axes(&request, selection.width)?;
    let children = children(&axes, &spec, context, &budget, extents)?;
    let grid_strides = crate::tensor::strides(&children.grid_shape)?;
    let base_strides = crate::tensor::strides(&children.shape)?;
    // A provider can satisfy every FFT child even when the requested CPU
    // fallback is unavailable. NUFFT's own rows use only verified capabilities.
    let backend = context.resolve(spec.backend_strategy)?;
    let processor = context.execution().processor();
    let backend = if backend.is_available(processor) {
        backend
    } else {
        scioxide_core::backend::KernelBackend::Scalar
    };
    let dispatch = scioxide_core::backend::KernelDispatch::for_backend(backend, processor)?;
    let kernel = Kernel::new(selection.width, request.sigma, request.eps, dispatch)?;
    let modes = mode_axes(&axes, &grid_strides, &kernel)?;
    let geometry = prepared_geometry(
        &request,
        &axes,
        (&grid_strides, &base_strides),
        &kernel,
        spec.direction,
    )?;
    let lens = (budget.input_len, budget.output_len);
    let scaling = scaling(
        &spec,
        lens,
        &children.grid_shape,
        selection.sizing.visits,
        &modes,
    )?;
    let child_scratch = if scaling.bounded {
        children.child_scratch
    } else {
        children
            .fft
            .iter()
            .map(ComplexCore::scratch_len)
            .max()
            .unwrap_or(0)
    };
    let q = scioxide_core::quantity!("nufft_preparation_bytes");
    let scratch_len = checked_add(budget.fixed_scratch, child_scratch, q)?;
    let workspace_bytes = checked_mul(
        scratch_len,
        size_of::<Complex<S>>(),
        scioxide_core::quantity!("nufft_workspace_bytes"),
    )?;
    admit_bytes(
        workspace_bytes,
        spec.limits.workspace_bytes,
        scioxide_core::quantity!("workspace_bytes"),
    )?;
    let plan_bytes = checked_add(budget.own_bytes, children.child_bytes, q)?;
    let preparation_bytes =
        children
            .preparation_bytes
            .max(checked_add(plan_bytes, budget.temporary_bytes, q)?);
    admit_bytes(
        preparation_bytes,
        spec.limits.preparation_bytes,
        scioxide_core::quantity!("preparation_bytes"),
    )?;
    context.execution().check()?;
    Ok(NufftPlan {
        plan_bytes,
        geometry,
        spec,
        input_len: budget.input_len,
        output_len: budget.output_len,
        shape: children.shape,
        grid_shape: children.grid_shape,
        grid_strides,
        fft: children.fft,
        modes,
        base_len: selection.sizing.base_len,
        kernel,
        scale: scaling.scale,
        scratch_len,
        lane_len: selection.sizing.lane_len,
        tile_len: selection.sizing.tile_len,
        headroom: scaling.headroom,
        bounded: scaling.bounded,
        diagnostics: NufftDiagnostics {
            approximation_error: selection.approximation,
            roundoff_error: selection.roundoff,
            kernel_support: selection.width,
            grid_points: selection.sizing.grid_len,
            stencil_visits: selection.sizing.stencil_visits,
            preparation_bytes,
            workspace_bytes,
        },
    })
}

// Product phases preserve the low product through FMA before argument reduction.
// a * b = hi + lo exactly (lo from the FMA residual), and
// exp(i s (hi + lo)) = exp(i s hi) * exp(i s lo); for large type-3 centre
// products this keeps the phase error at the rounding of sin/cos rather than
// ulp(a * b), which could be many radians.
fn product_phase(a: f64, b: f64, sign: f64) -> Result<Complex<f64>, SciOxideError> {
    let hi = a * b;
    if !hi.is_finite() {
        return Err(crate::validation::arithmetic_overflow(
            "phase product exceeds supported finite range",
        ));
    }
    let lo = a.mul_add(b, -hi);
    let (sh, ch) = (sign * hi).sin_cos();
    let (sl, cl) = (sign * lo).sin_cos();
    Ok(Complex::new(ch, sh) * Complex::new(cl, sl))
}
// Reduce an arbitrary type-1/2 coordinate to (-pi, pi] via sin/cos, whose
// library range reduction is accurate for large arguments, instead of a
// floating `x % TAU` whose error grows with |x|.
fn periodic_angle(x: f64) -> f64 {
    if x.abs() <= PI {
        return x;
    }
    let (s, c) = x.sin_cos();
    s.atan2(c)
}

// Binary64 high/low split of 1/(2*pi). Retaining the low product until
// after integer-cell subtraction avoids an O(grid_extent * epsilon) phase
// error for ordinary coordinates, without changing the input geometry.
fn grid_position(theta: f64, theta_low: f64, grid: usize) -> (f64, f64) {
    const INV_TAU: f64 = 0.15915494309189535;
    const INV_TAU_LOW: f64 = -9.839338337591243e-18;
    let hi = theta * INV_TAU;
    let lo = theta.mul_add(INV_TAU, -hi) + theta * INV_TAU_LOW + theta_low * INV_TAU;
    (hi * grid as f64, lo * grid as f64)
}

/// Grid position (high, low) and periodic extent of one point coordinate,
/// multiplying its type-3 phase and deconvolution into `factor`.
#[inline]
fn point_position<const THREE: bool, const TARGET: bool>(
    x: f64,
    axis: &Axis,
    kernel: &Kernel,
    sign: f64,
    factor: &mut Complex<f64>,
) -> Result<(f64, f64, usize), SciOxideError> {
    if !THREE {
        let (hi, lo) = grid_position(periodic_angle(x), 0.0, axis.grid);
        return Ok((hi, lo, axis.grid));
    }
    if TARGET {
        let delta = x - axis.frequency_center;
        let theta = if axis.rescale == 0.0 {
            0.0
        } else {
            delta / axis.rescale
        };
        *factor *= product_phase(axis.source_center, delta, sign)?;
        if axis.grid != 1 {
            *factor /= kernel.fourier(theta);
        }
        let low = if axis.rescale == 0.0 {
            0.0
        } else {
            (-theta).mul_add(axis.rescale, delta) / axis.rescale
        };
        let (hi, lo) = grid_position(theta, low, axis.grid);
        return Ok((hi, lo, axis.grid));
    }
    *factor *= product_phase(x, axis.frequency_center, sign)?;
    let product = (x - axis.source_center) * axis.rescale;
    let low = (x - axis.source_center).mul_add(axis.rescale, -product);
    let center = (axis.base / 2) as f64;
    let hi = product + center;
    let z = hi - product;
    Ok((hi, low + (product - (hi - z)) + (center - z), axis.base))
}

// Grid construction bounds these finite values well inside the i64 range.
// Truncation plus a sign correction avoids a libm floor call on baseline CPUs.
fn cell_floor(value: f64) -> f64 {
    let integer = value as i64;
    (integer - i64::from(value < integer as f64)) as f64
}

#[inline]
fn wrap_start(start: i64, extent: usize) -> usize {
    let extent = extent as i64;
    // Periodic positions lie within half a grid and type-3 sources within
    // their padded base box. Their stencil starts need at most one wrap.
    if (-extent..extent).contains(&start) {
        // Coordinates straddle zero, so a branch on the sign would be
        // unpredictable. The arithmetic shift supplies either extent or zero.
        (start + ((start >> 63) & extent)) as usize
    } else {
        start.rem_euclid(extent) as usize
    }
}

fn make_stencil<S: RealScalar, const THREE: bool, const TARGET: bool>(
    points: &NdView<'_, '_, S>,
    axes: &[Axis],
    strides: &[usize],
    kernel: &Kernel,
    sign: f64,
) -> Result<Stencil<S>, SciOxideError> {
    let width = kernel.width;
    let count = points.layout().shape().extents()[0];
    let entries = checked_mul(
        count,
        axes.len(),
        scioxide_core::quantity!("nufft_point_axes"),
    )?;
    let mut positions = try_with_capacity(entries, scioxide_core::quantity!("nufft_point_axes"))?;
    let mut factors = try_with_capacity(
        if THREE { count } else { 0 },
        scioxide_core::quantity!("nufft_point_factors"),
    )?;
    let mut order = try_with_capacity(count, scioxide_core::quantity!("nufft_point_order"))?;
    for j in 0..count {
        let mut factor = Complex::new(1.0, 0.0);
        for (d, axis) in axes.iter().enumerate() {
            let x = coordinate(points, j, d)?;
            let (position, position_low, extent) =
                point_position::<THREE, TARGET>(x, axis, kernel, sign, &mut factor)?;
            if axis.grid == 1 {
                positions.push(PointAxis {
                    start: 0,
                    fraction: 0.0,
                });
                continue;
            }
            let cell = cell_floor(position);
            let local = (position - cell) + position_low - width as f64 * 0.5;
            let offset = cell_floor(local) + 1.0;
            let start = cell as i64 + offset as i64;
            let fraction = local - offset + 1.0;
            positions.push(PointAxis {
                start: wrap_start(start, extent),
                fraction,
            });
        }
        if THREE {
            factors.push(Complex::new(scalar(factor.re)?, scalar(factor.im)?));
        }
        order.push(j);
    }
    let mut stencil_axes =
        try_with_capacity(axes.len(), scioxide_core::quantity!("nufft_stencil_axes"))?;
    for (a, &stride) in axes.iter().zip(strides) {
        stencil_axes.push(StencilAxis {
            extent: if THREE && !TARGET { a.base } else { a.grid },
            stride,
            support: if a.grid == 1 { 1 } else { width },
        });
    }
    let bins = spread_order(&positions, &stencil_axes, &mut order)?;
    Ok(Stencil {
        positions,
        factors,
        order,
        axes: stencil_axes,
        bins,
    })
}

/// Counting-sort `order` by spreading bin and return the dense 3-D tiles.
///
/// Dense 3-D bins also define bounded spreading tiles. Sparse and other
/// dimensional geometries keep linear-grid counting bins.
fn spread_order(
    positions: &[PointAxis],
    stencil_axes: &[StencilAxis],
    order: &mut [usize],
) -> Result<Vec<SpreadBin>, SciOxideError> {
    let (count, rank) = (order.len(), stencil_axes.len());
    let grid = stencil_axes.first().map_or(1, |a| a.extent * a.stride);
    let tile_extents = if let [a, b, c] = stencil_axes {
        [a.extent, b.extent, c.extent]
    } else {
        [1; 3]
    };
    let tile_count = spread_bins(tile_extents, count);
    let bins = if tile_count > 0 {
        tile_count
    } else {
        grid.min(4096).min(count.max(1))
    };
    let cells_per_bin = grid.div_ceil(bins);
    let bin = |j: usize| {
        let point = &positions[j * rank..][..rank];
        if tile_count > 0 {
            point.iter().zip(tile_extents).fold(0, |key, (p, n)| {
                key * n.div_ceil(SPREAD_BIN) + p.start / SPREAD_BIN
            })
        } else {
            point
                .iter()
                .zip(stencil_axes)
                .map(|(p, a)| p.start * a.stride)
                .sum::<usize>()
                / cells_per_bin
        }
    };
    let mut next = try_filled(bins, 0_usize, scioxide_core::quantity!("nufft_sort_bins"))?;
    for j in 0..count {
        next[bin(j)] += 1;
    }
    let mut tiles = try_with_capacity(tile_count, scioxide_core::quantity!("nufft_spread_bins"))?;
    let mut prefix = 0;
    for (index, size) in next.iter_mut().enumerate() {
        let n = *size;
        *size = prefix;
        if tile_count > 0 && n > 0 {
            let mut code = index;
            let mut origin = [0; 3];
            for d in (0..3).rev() {
                let radix = tile_extents[d].div_ceil(SPREAD_BIN);
                origin[d] = (code % radix) * SPREAD_BIN;
                code /= radix;
            }
            tiles.push(SpreadBin {
                start: prefix,
                end: prefix + n,
                origin,
            });
        }
        prefix += n;
    }
    for j in 0..count {
        let next = &mut next[bin(j)];
        order[*next] = j;
        *next += 1;
    }
    Ok(tiles)
}

/// Live axis metadata and stored child allocations before preparing a child.
fn live_preparation_bytes<S: RealScalar>(
    rank: usize,
    child_bytes: usize,
) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("nufft_preparation_bytes");
    checked_add(
        size_of::<NufftPlan<S>>(),
        checked_add(
            child_bytes,
            checked_mul(
                rank,
                size_of::<Axis>() + 2 * size_of::<usize>() + size_of::<ComplexCore<S>>(),
                q,
            )?,
            q,
        )?,
        q,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_coordinate_geometry_does_not_walk_zero_length_rows() {
        let shape = [usize::MAX, 0];
        let layout = scioxide_core::StridedLayout::new_broadcast(&shape, &[0, 0]).unwrap();
        let points = scioxide_core::NdView::<f64>::new(&[], layout).unwrap();
        assert_eq!(super::geometry(&points).unwrap(), (usize::MAX, 0));
    }

    #[test]
    fn stencil_cell_boundaries_and_wrapping() {
        for value in [
            -((1_u64 << 52) as f64),
            -32.5,
            -32.0,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            0.5,
            1.0,
            31.999999999999996,
            32.0,
            (1_u64 << 52) as f64,
        ] {
            assert_eq!(super::cell_floor(value), value.floor());
        }
        for extent in [1, 4, 17, 64, 1 << 30] {
            for start in [
                -3 * extent,
                -extent - 1,
                -extent,
                -1,
                0,
                extent - 1,
                extent,
                3 * extent,
            ] {
                assert_eq!(
                    super::wrap_start(start, extent as usize),
                    start.rem_euclid(extent) as usize
                );
            }
        }
    }

    #[test]
    fn kernel_blocks_match_bessel_weights_at_boundaries_and_tails() {
        use scioxide_core::{
            Complex, PortableProcessor,
            backend::{KernelBackend, KernelDispatch},
        };
        let dispatch =
            KernelDispatch::for_backend(KernelBackend::Scalar, &PortableProcessor).unwrap();
        for width in 4..20 {
            for accuracy in [1e-2, 1e-3, 1e-6, 1e-12] {
                // A narrow support is never admitted at a tight accuracy:
                // the series is qualified together with the alias envelope.
                if 3.0 * (-core::f64::consts::PI * (width - 1) as f64 * 0.5_f64.sqrt()).exp()
                    > accuracy
                {
                    continue;
                }
                let kernel = super::Kernel::new(width, 2.0, accuracy, dispatch).unwrap();
                let mut weights = alloc::vec![Complex::new(0.0, 17.0); width];
                for fraction in [
                    0.0,
                    0.125,
                    0.5,
                    0.875,
                    f64::from_bits(1.0_f64.to_bits() - 1),
                    1.0,
                ] {
                    kernel.weights(fraction, &mut weights).unwrap();
                    for (offset, weight) in weights.iter().enumerate() {
                        let z = (width as f64 - 2.0 - 2.0 * offset as f64 + 2.0 * fraction)
                            / width as f64;
                        let expected = super::i0(kernel.beta * (1.0 - z * z).max(0.0).sqrt())
                            .unwrap()
                            * kernel.inverse_i0;
                        assert!(
                            (weight.re - expected).abs() <= accuracy + 64.0 * f64::EPSILON,
                            "width {width}, accuracy {accuracy}, fraction {fraction}, offset {offset}: {} != {expected}",
                            weight.re
                        );
                        assert_eq!(weight.im, 17.0);
                    }
                }
            }
        }
    }

    #[test]
    fn kernel_normalization_distinguishes_overflow_from_finite_nonconvergence() {
        use scioxide_core::{NumericalIssue, SciOxideErrorView};
        for (x, expected) in [
            (1000.0, NumericalIssue::Overflow),
            (700.0, NumericalIssue::LossOfSignificance),
        ] {
            assert!(matches!(crate::nufft::i0(x).unwrap_err().view(),
                SciOxideErrorView::NumericalInstability { issue, .. } if issue == expected));
        }
    }
    use super::{Span, Spans, frame, select, validate};
    use crate::NufftTransform;
    use crate::{FftDirection, FftPlanningContext, Normalization, NufftLimits, NufftPlanSpec};
    use scioxide_core::{NdView, SciOxideErrorKind, StridedLayout};

    fn spec(accuracy: f64) -> NufftPlanSpec<f64> {
        NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, accuracy)
    }

    #[test]
    fn type3_spans_keep_axis_order_around_constant_axes() {
        // Two points sharing coordinates 1 and 3 make those axes constant phases.
        let points = [0.0, 0.5, 1.0, 0.5, -2.0, 1.0, 0.5, -1.0, 0.5, 3.0];
        let frequencies = [1.0, 2.0, -3.0, 4.0, 5.0, -1.0, 0.0, 2.0, 1.0, -5.0];
        let view = |x| NdView::new(x, StridedLayout::new(&[2, 5], &[5, 1]).unwrap()).unwrap();
        let transform = NufftTransform::Type3 {
            points: view(&points),
            frequencies: view(&frequencies),
        };
        let spec = spec(1e-6);
        let request = validate(&transform, &spec, FftPlanningContext::default()).unwrap();
        let spans = Spans::scan(&request).unwrap();
        let spanning = spans.iter().map(|s| matches!(s, Span::Box(_)));
        assert!(spanning.eq([true, false, true, false, true]));
        for (d, span) in spans.iter().enumerate() {
            let scanned = frame(&request, d).unwrap().span;
            assert_eq!(
                span.extents(9, 2.0).unwrap(),
                scanned.extents(9, 2.0).unwrap()
            );
        }
    }

    #[test]
    fn width_search_widens_until_the_rounding_floor_fits() {
        let points = [-0.75, 0.125, 0.5, 1.25];
        let transform = NufftTransform::Type1 {
            points: NdView::new(&points, StridedLayout::new(&[4, 1], &[1, 1]).unwrap()).unwrap(),
            modes: &[64],
        };
        let width = |accuracy, limits| {
            let spec = spec(accuracy);
            let request = validate(&transform, &spec, FftPlanningContext::default())?;
            select(&request, limits).map(|selection| selection.width)
        };
        let unlimited = NufftLimits::default();
        // Aliasing alone selects 17 for both requests; the floor needs 18 at 5e-15.
        assert_eq!(width(7e-15, unlimited).unwrap(), 17);
        assert_eq!(width(5e-15, unlimited).unwrap(), 18);
        assert_eq!(
            width(4e-15, unlimited).unwrap_err().kind(),
            SciOxideErrorKind::NumericalInstability
        );
        let narrow = NufftLimits {
            kernel_support: 17,
            ..unlimited
        };
        assert_eq!(width(7e-15, narrow).unwrap(), 17);
        assert_eq!(
            width(5e-15, narrow).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    }
}
