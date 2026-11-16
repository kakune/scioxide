use super::*;

fn desired<S: RealScalar>(b: &FirBand<S>, f: S) -> S {
    b.desired_lower + (b.desired_upper - b.desired_lower) * ((f - b.lower) / (b.upper - b.lower))
}
/// Designs a minimax (equiripple) linear-phase FIR filter of `n` taps by the
/// Parks–McClellan algorithm.
///
/// The taps minimize the largest weighted error of the zero-phase amplitude
/// `A` over the bands,
///
/// ```text
/// max_b max_{f ∈ [l_b, u_b]} W_b |A(f) − D_b(f)|,
/// ```
///
/// where band `b` covers `[l_b, u_b]` in multiples of Nyquist, `D_b` is its
/// linear desired amplitude and `W_b` its weight (see [`FirBand`]). The gaps
/// between bands are unconstrained. All four linear-phase types are supported
/// (see [`FirSymmetry`]).
///
/// The Remez exchange works on a grid of spacing at most
/// `1/((m + 1) · grid_density)` over the bands, where `m` is the number of
/// independent coefficients. Each iteration computes the levelled error `δ` on
/// a reference set of `m + 1` frequencies, interpolates the amplitude
/// `D ∓ δ/W` there as a polynomial in `x = cos(πf)` in barycentric form, and
/// then replaces the reference by the alternating extrema of the weighted
/// error on the grid that are at least `|δ|` in size. An iteration costs
/// `O(m² + g m)` for `g` grid points, and storage is `O(g + m)`. The initial
/// reference is placed at the equally spaced
/// quantiles of the equilibrium measure of the bands in `x`, the asymptotic
/// distribution of minimax alternation sets.
///
/// The exchange stops with [`FirTermination::Converged`] when the spread of
/// the alternating errors meets [`RemezOptions::tolerance`]. The reported
/// [`FirDesign::weighted_ripple`] is measured on the grid and estimates the
/// error on the continuous bands but does not bound it. This form uses
/// [`RemezOptions::default`] and runs in a fresh root scope.
/// [`design_fir_remez_with_options`] takes options and reports an unconverged
/// exchange in its result, and [`design_fir_remez_scoped`] runs inside the
/// caller's solve.
///
/// # Errors
///
/// - `InvalidInput` if `n` is zero, if `n = 1` with odd symmetry, or if the
///   bands are invalid (see [`FirBand`]).
/// - `InvalidInput` if the desired amplitude is nonzero where the symmetry
///   forces a zero, if a band has no grid point apart from a forced zero, if
///   bands are not separated on the grid, or if the grid has fewer than `m + 1`
///   points.
/// - `NumericalInstability` if the exchange breaks down or the ratio of band
///   weights underflows.
/// - `ConvergenceFailure` (`IterationLimit`) if the exchange reaches its
///   iteration limit, and (`Stagnation`) if it stalls.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{FirBand, FirSymmetry, design_fir_remez};
/// // Low-pass filter with an eight times heavier weight on the stop band.
/// let bands = [
///     FirBand::new(0.0_f64, 0.3, 1.0, 1.0),
///     FirBand::new(0.4, 1.0, 0.0, 0.0).with_weight(8.0),
/// ];
/// let design = design_fir_remez(31, FirSymmetry::Even, &bands)?;
/// assert_eq!(design.coefficients().len(), 31);
/// // The weighted error is at most the ripple at the grid points 0 and 1.
/// let ripple = design.weighted_ripple().unwrap_or(f64::NAN);
/// assert!((design.response(0.0)?.re - 1.0).abs() <= ripple * (1.0 + 1e-9));
/// assert!(8.0 * design.response(1.0)?.norm() <= ripple * (1.0 + 1e-9));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - T. W. Parks and J. H. McClellan, "Chebyshev approximation for
///   nonrecursive digital filters with linear phase," *IEEE Trans. Circuit
///   Theory* 19(2), 189–194, 1972.
/// - J. H. McClellan, T. W. Parks and L. R. Rabiner, "A computer program for
///   designing optimum FIR linear phase digital filters," *IEEE Trans. Audio
///   Electroacoust.* 21(6), 506–526, 1973.
pub fn design_fir_remez<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    bands: &[FirBand<S>],
) -> Result<FirDesign<S>, SciOxideError> {
    design_fir_remez_with_options(n, symmetry, bands, RemezOptions::default())
        .and_then(require_success)
}
/// Designs a minimax FIR filter as [`design_fir_remez`] does, with the
/// iteration limit, grid density and tolerance of `options`.
///
/// An exchange that reaches its iteration limit or stalls is not an error: it
/// returns [`FirTermination::IterationLimit`] or
/// [`FirTermination::ExchangeStalled`] with the last finite approximation, and
/// [`FirDesign::is_success`] returns `false`.
///
/// # Errors
///
/// - The errors of [`design_fir_remez`], except that an exchange that does not
///   converge is reported in the result.
/// - `InvalidInput` if `options` has zero iterations or a grid density below
///   16.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{
///     FirBand, FirSymmetry, FirTermination, RemezOptions, design_fir_remez_with_options,
/// };
/// let bands = [
///     FirBand::new(0.0_f64, 0.3, 1.0, 1.0),
///     FirBand::new(0.4, 1.0, 0.0, 0.0),
/// ];
/// // One exchange iteration does not reach the minimax solution.
/// let options = RemezOptions::new().with_max_iterations(1);
/// let design = design_fir_remez_with_options(31, FirSymmetry::Even, &bands, options)?;
/// assert_eq!(design.termination(), FirTermination::IterationLimit);
/// assert!(design.weighted_ripple().is_some_and(f64::is_finite));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn design_fir_remez_with_options<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    bands: &[FirBand<S>],
    options: RemezOptions<S>,
) -> Result<FirDesign<S>, SciOxideError> {
    design_fir_remez_scoped(
        n,
        symmetry,
        bands,
        options,
        ExecutionContext::sequential_native(),
    )
}
/// Designs a minimax FIR filter as [`design_fir_remez_with_options`] does, as
/// one epoch of `scope`.
///
/// Each exchange iteration counts toward the limits of `scope` and observes
/// its cancellation and deadline, as described in
/// [scoped designs](crate#scoped-designs).
///
/// # Errors
///
/// - The errors of [`design_fir_remez_with_options`].
/// - `ConvergenceFailure` if the iterations exhaust the limits of `scope`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
pub fn design_fir_remez_scoped<'scope, 'capability, S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    bands: &[FirBand<S>],
    options: RemezOptions<S>,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<FirDesign<S>, SciOxideError> {
    crate::iir::run_design(
        scope,
        scioxide_core::algorithm_id!("signal", "remez", 2),
        |context| fir_remez(n, symmetry, bands, options, context),
    )
}
/// Returns a converged design and maps an unconverged exchange to the
/// `ConvergenceFailure` of [`design_fir_remez`].
fn require_success<S: RealScalar>(design: FirDesign<S>) -> Result<FirDesign<S>, SciOxideError> {
    let (reason, detail) = match design.termination() {
        FirTermination::Completed | FirTermination::Converged => return Ok(design),
        FirTermination::IterationLimit => (
            scioxide_core::ConvergenceReason::IterationLimit,
            "Remez exchange reached its iteration limit",
        ),
        FirTermination::ExchangeStalled => (
            scioxide_core::ConvergenceReason::Stagnation,
            "Remez exchange stalled",
        ),
    };
    Err(SciOxideError::convergence_failure(
        reason,
        scioxide_core::ConvergenceDiagnostics::new(),
        detail,
    ))
}
fn fir_remez<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    bands: &[FirBand<S>],
    options: RemezOptions<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<FirDesign<S>, SciOxideError> {
    let execution = context.execution();
    execution.check()?;
    let m = validate(n, symmetry, bands)?;
    if options.max_iterations == 0 {
        return Err(invalid(
            scioxide_core::argument!("options.max_iterations"),
            "at least one exchange iteration is required",
        ));
    }
    if options.grid_density < 16 {
        return Err(invalid(
            scioxide_core::argument!("options.grid_density"),
            "grid density must be at least 16",
        ));
    }
    let grid_scale = checked_mul(
        m + 1,
        options.grid_density,
        scioxide_core::quantity!("remez_grid"),
    )?;
    let capacity = checked_add(
        grid_scale,
        checked_mul(bands.len(), 2, scioxide_core::quantity!("remez_grid"))?,
        scioxide_core::quantity!("remez_grid_capacity"),
    )?;
    let mut frequencies = scioxide_core::backend::try_with_capacity(
        capacity,
        scioxide_core::quantity!("remez_grid"),
    )?;
    let mut targets = scioxide_core::backend::try_with_capacity(
        capacity,
        scioxide_core::quantity!("remez_targets"),
    )?;
    let mut weights = scioxide_core::backend::try_with_capacity(
        capacity,
        scioxide_core::quantity!("remez_weights"),
    )?;
    let mut coordinates = scioxide_core::backend::try_with_capacity(
        capacity,
        scioxide_core::quantity!("remez_coordinates"),
    )?;
    let mut edges = scioxide_core::backend::try_with_capacity(
        bands.len(),
        scioxide_core::quantity!("remez_edges"),
    )?;
    let weight_scale = bands.iter().fold(S::ZERO, |a, b| a.max(b.weight));
    for b in bands {
        let start = frequencies.len();
        // At least two endpoints resolve every representable narrow band.
        let subdivisions =
            (((b.upper - b.lower).into_f64() * grid_scale as f64).ceil() as usize).max(1);
        for i in 0..=subdivisions {
            let f = if i == subdivisions {
                b.upper
            } else {
                b.lower + (b.upper - b.lower) * scalar::<S>(i)? / scalar::<S>(subdivisions)?
            };
            if forced_zero(n, symmetry, f) {
                continue;
            }
            if frequencies.last().is_some_and(|&previous| previous >= f) {
                return Err(invalid(
                    scioxide_core::argument!("bands"),
                    "bands must be separated on a representable grid",
                ));
            }
            let factor = remez_factor(n, symmetry, f)?;
            let weight = (b.weight / weight_scale) * factor;
            if weight == S::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "transformed band weights span an unrepresentable dynamic range",
                ));
            }
            frequencies.push(f);
            coordinates.push(S::try_from_f64_lossy(
                (core::f64::consts::PI * f.into_f64()).cos(),
            )?);
            targets.push(finite(desired(b, f) / factor)?);
            weights.push(finite(weight)?);
        }
        if start == frequencies.len() {
            return Err(invalid(
                scioxide_core::argument!("bands"),
                "band has no unconstrained representable grid points",
            ));
        }
        edges.push((start, frequencies.len()));
    }
    let g = frequencies.len();
    if g < m + 1 {
        return Err(invalid(
            scioxide_core::argument!("bands"),
            "constrained grid has fewer points than the design needs",
        ));
    }
    let mut selected = try_filled(m + 1, 0usize, scioxide_core::quantity!("remez_extrema"))?;
    equilibrium_reference(bands, &frequencies, &edges, &mut selected, execution)?;
    let mut errors = try_filled(g, S::ZERO, scioxide_core::quantity!("remez_errors"))?;
    let mut candidates: Vec<usize> =
        scioxide_core::backend::try_with_capacity(g, scioxide_core::quantity!("remez_candidates"))?;
    let mut result = from_amplitudes(n, symmetry, &[])?;
    result.extrema = try_filled(
        m + 1,
        S::ZERO,
        scioxide_core::quantity!("remez_report_extrema"),
    )?;
    let mut nodes = try_filled(m + 1, S::ZERO, scioxide_core::quantity!("remez_nodes"))?;
    let mut barycentric = try_filled(
        m + 1,
        S::ZERO,
        scioxide_core::quantity!("remez_barycentric"),
    )?;
    let mut values = try_filled(m + 1, S::ZERO, scioxide_core::quantity!("remez_values"))?;
    let mut scratch = try_filled(
        m + 1,
        scioxide_core::ScaledValue::ZERO,
        scioxide_core::quantity!("remez_products"),
    )?;
    for iteration in 0..options.max_iterations {
        execution.check()?;
        context.record_iteration()?.require()?;
        for (node, &index) in nodes.iter_mut().zip(&selected) {
            *node = coordinates[index];
        }
        scioxide_poly::backend::polynomial_basis::barycentric_weights_into(
            &nodes,
            &mut barycentric,
            &mut scratch,
        )?;
        let sums =
            scioxide_core::compensated_reduce(barycentric.iter().zip(&selected).enumerate().map(
                |(j, (&w, &i))| {
                    let sign = if j.is_multiple_of(2) { S::ONE } else { -S::ONE };
                    [w * targets[i], sign * w / weights[i]]
                },
            ));
        let (numerator, denominator) = match sums.and_then(|[n, d]| Ok((n.total()?, d.total()?))) {
            Ok(sums) => sums,
            Err(_) => (
                crate::util::sum_products(
                    barycentric
                        .iter()
                        .copied()
                        .zip(selected.iter().map(|&i| targets[i])),
                )?,
                crate::util::sum_products(barycentric.iter().enumerate().map(|(j, &w)| {
                    (
                        w,
                        if j.is_multiple_of(2) {
                            S::ONE / weights[selected[j]]
                        } else {
                            -S::ONE / weights[selected[j]]
                        },
                    )
                }))?,
            ),
        };
        if denominator == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::LossOfSignificance,
                "vanished exchange ripple denominator",
            ));
        }
        let ripple = finite(numerator / denominator)?;
        for (j, (value, &index)) in values.iter_mut().zip(&selected).enumerate() {
            let sign = if j.is_multiple_of(2) { S::ONE } else { -S::ONE };
            *value = finite(targets[index] - sign * ripple / weights[index])?;
        }
        let mut maximum = S::ZERO;
        for (block, (points, output)) in coordinates.chunks(4).zip(errors.chunks_mut(4)).enumerate()
        {
            if block.is_multiple_of(16) {
                execution.check()?;
            }
            remez_evaluate_block(&nodes, &barycentric, &values, points, output)?;
            for ((value, &target), &weight) in output
                .iter_mut()
                .zip(&targets[block * 4..])
                .zip(&weights[block * 4..])
            {
                *value = finite((*value - target) * weight)?;
                maximum = maximum.max(value.abs());
            }
        }
        select_extrema(&errors, &edges, ripple.abs(), &selected, &mut candidates);
        execution.check()?;
        let current = &mut result;
        current.iterations = iteration + 1;
        current.ripple = Some(finite(maximum * weight_scale)?);
        current
            .extrema
            .iter_mut()
            .zip(&selected)
            .for_each(|(f, &i)| *f = frequencies[i]);
        let threshold = options.tolerance.threshold(maximum)?;
        let minimum = candidates
            .iter()
            .fold(maximum, |a, &i| a.min(errors[i].abs()));
        if maximum <= threshold || (candidates.len() == m + 1 && maximum - minimum <= threshold) {
            result.termination = FirTermination::Converged;
            break;
        }
        if candidates.len() < m + 1 {
            current.termination = FirTermination::ExchangeStalled;
            break;
        }
        selected.copy_from_slice(&candidates);
        current.termination = FirTermination::IterationLimit;
    }
    remez_reconstruct(
        n,
        symmetry,
        &nodes,
        &barycentric,
        &values,
        &mut result.taps,
        execution,
    )?;
    execution.check()?;
    Ok(result)
}

/// Midpoint cells per band or gap in the equilibrium-measure quadrature.
const MEASURE_CELLS: usize = 256;

// The initial reference takes the global quantiles `i/m` of the equilibrium
// measure of the constrained set `E` in `x = cos(pi f)`, snapped to the grid.
// Alternation sets of weighted polynomial minimax approximation on `E`
// distribute asymptotically by this measure, so the first levelled error is
// comparable to the minimax ripple. A reference uniform in frequency misplaces
// O(m) nodes between unequal bands; its levelled error is then exponentially
// smaller than the ripple, and rounding noise of the error sweep chooses the
// next reference.
//
// For `K` bands with `x`-edges `e`, the density is `|r(x)|/(pi*sqrt|prod(x-e)|)`,
// where `r` has degree `K-1` and the gap conditions `integral r/sqrt|prod| = 0`
// put one zero in every gap. Each band or gap substitutes
// `x = mid + half*cos(theta)`, which absorbs its own edge singularities, and
// applies a midpoint rule in `theta`; logarithmic weights keep many-band
// products in range. The placement only selects grid indices, so it uses
// binary64 for every scalar width.
fn equilibrium_reference<S: RealScalar>(
    bands: &[FirBand<S>],
    frequencies: &[S],
    edges: &[(usize, usize)],
    selected: &mut [usize],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let k = bands.len();
    let mut intervals =
        scioxide_core::backend::try_with_capacity(k, scioxide_core::quantity!("remez_intervals"))?;
    intervals.extend(bands.iter().map(|b| {
        (
            (core::f64::consts::PI * b.upper.into_f64()).cos(),
            (core::f64::consts::PI * b.lower.into_f64()).cos(),
        )
    }));
    let mut cells = try_filled(
        MEASURE_CELLS,
        0.0_f64,
        scioxide_core::quantity!("remez_measure"),
    )?;
    // Chebyshev coefficients of `r`, with unit leading coefficient.
    let mut polynomial = try_filled(k, 0.0_f64, scioxide_core::quantity!("remez_gap_polynomial"))?;
    polynomial[k - 1] = 1.0;
    if k > 1 {
        let mut matrix = Matrix::<f64>::zeros(k - 1, k - 1)?;
        let mut rhs = Vector::<f64>::zeros(k - 1)?;
        let mut basis = try_filled(k, 0.0_f64, scioxide_core::quantity!("remez_gap_basis"))?;
        let mut row = try_filled(k, 0.0_f64, scioxide_core::quantity!("remez_gap_row"))?;
        for gap in 0..k - 1 {
            execution.check()?;
            // Gap between bands `gap` and `gap + 1`; each row has its own positive scale.
            let interval = (intervals[gap + 1].1, intervals[gap].0);
            let peak = log_weights(&intervals, interval, [2 * gap, 2 * gap + 3], &mut cells);
            row.fill(0.0);
            for (i, &weight) in cells.iter().enumerate() {
                let w = (weight - peak).exp();
                chebyshev_basis(measure_node(interval, i), &mut basis);
                for (sum, &t) in row.iter_mut().zip(&basis) {
                    *sum += w * t;
                }
            }
            for (p, &value) in row[..k - 1].iter().enumerate() {
                matrix.view_mut().set(gap, p, value)?;
            }
            rhs.as_mut_slice()[gap] = -row[k - 1];
        }
        let mut solve = DesignSolve::<f64>::new(k - 1, k - 1)?;
        let _ = solve.solve(&matrix, &rhs, execution)?;
        polynomial[..k - 1].copy_from_slice(solve.solution.as_slice());
    }
    // Normalized band masses, accumulated from logarithmic band scales.
    let mut masses = try_filled(k, 0.0_f64, scioxide_core::quantity!("remez_band_masses"))?;
    for (band, mass) in masses.iter_mut().enumerate() {
        execution.check()?;
        let peak = band_density(&intervals, band, &polynomial, &mut cells);
        *mass = peak + (cells.iter().sum::<f64>() / MEASURE_CELLS as f64).ln();
    }
    let peak = masses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if !peak.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::LossOfSignificance,
            "equilibrium measure of the bands vanished",
        ));
    }
    masses
        .iter_mut()
        .for_each(|mass| *mass = (*mass - peak).exp());
    let total = masses.iter().sum::<f64>();
    masses.iter_mut().for_each(|mass| *mass /= total);
    let m = selected.len() - 1;
    let g = frequencies.len();
    let (mut band, mut start, mut prepared) = (0, 0.0_f64, false);
    let mut previous = 0;
    for (i, slot) in selected.iter_mut().enumerate() {
        let quantile = i as f64 / m as f64;
        while band + 1 < k && quantile > start + masses[band] {
            start += masses[band];
            band += 1;
            prepared = false;
        }
        if !prepared {
            execution.check()?;
            let _ = band_density(&intervals, band, &polynomial, &mut cells);
            let mut sum = 0.0;
            for cell in &mut cells {
                sum += *cell;
                *cell = sum;
            }
            if sum > 0.0 {
                cells.iter_mut().for_each(|cell| *cell /= sum);
            }
            prepared = true;
        }
        // Invert the piecewise-linear cumulative measure of the band in `theta`.
        let local = if masses[band] > 0.0 {
            ((quantile - start) / masses[band]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let cell = cells.partition_point(|&c| c < local).min(MEASURE_CELLS - 1);
        let below = if cell == 0 { 0.0 } else { cells[cell - 1] };
        let fraction = if cells[cell] > below {
            (local - below) / (cells[cell] - below)
        } else {
            0.5
        };
        let theta = (cell as f64 + fraction) * core::f64::consts::PI / MEASURE_CELLS as f64;
        let (lower, upper) = intervals[band];
        let x = ((lower + upper) / 2.0 + (upper - lower) / 2.0 * theta.cos()).clamp(-1.0, 1.0);
        let target = x.acos() / core::f64::consts::PI;
        let (first, end) = edges[band];
        let next = first + frequencies[first..end].partition_point(|f| f.into_f64() < target);
        let chosen = if next == end
            || (next > first
                && target - frequencies[next - 1].into_f64()
                    < frequencies[next].into_f64() - target)
        {
            next - 1
        } else {
            next
        };
        *slot = chosen.clamp(if i == 0 { 0 } else { previous + 1 }, g - (m + 1 - i));
        previous = *slot;
    }
    Ok(())
}

// Midpoint node `i` of `[lower, upper]` under `x = mid + half*cos(theta)`;
// increasing `theta` is increasing frequency.
fn measure_node((lower, upper): (f64, f64), i: usize) -> f64 {
    let theta = (i as f64 + 0.5) * core::f64::consts::PI / MEASURE_CELLS as f64;
    (lower + upper) / 2.0 + (upper - lower) / 2.0 * theta.cos()
}

// Store `-ln(prod |x - e|)/2` over every band edge except `own` at the midpoint
// nodes of `interval`, and return the largest value.
fn log_weights(
    intervals: &[(f64, f64)],
    interval: (f64, f64),
    own: [usize; 2],
    cells: &mut [f64],
) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    for (i, cell) in cells.iter_mut().enumerate() {
        let x = measure_node(interval, i);
        let mut sum = 0.0;
        for (edge, e) in intervals.iter().flat_map(|&(a, b)| [a, b]).enumerate() {
            if !own.contains(&edge) {
                sum += (x - e).abs().max(f64::MIN_POSITIVE).ln();
            }
        }
        *cell = -0.5 * sum;
        peak = peak.max(*cell);
    }
    peak
}

// Store the band's equilibrium density relative to its logarithmic peak, which is returned.
fn band_density(
    intervals: &[(f64, f64)],
    band: usize,
    polynomial: &[f64],
    cells: &mut [f64],
) -> f64 {
    let _ = log_weights(intervals, intervals[band], [2 * band, 2 * band + 1], cells);
    let mut peak = f64::NEG_INFINITY;
    for (i, cell) in cells.iter_mut().enumerate() {
        *cell += chebyshev_series(polynomial, measure_node(intervals[band], i))
            .abs()
            .ln();
        peak = peak.max(*cell);
    }
    if peak.is_finite() {
        cells
            .iter_mut()
            .for_each(|cell| *cell = (*cell - peak).exp());
    } else {
        cells.fill(0.0);
    }
    peak
}

// Chebyshev polynomials `T_0..T_{len-1}` at `x`.
fn chebyshev_basis(x: f64, basis: &mut [f64]) {
    let (mut previous, mut current) = (1.0, x);
    for (p, value) in basis.iter_mut().enumerate() {
        *value = match p {
            0 => 1.0,
            1 => x,
            _ => {
                (previous, current) = (current, 2.0 * x * current - previous);
                current
            }
        };
    }
}

// Clenshaw evaluation of a Chebyshev series.
fn chebyshev_series(coefficients: &[f64], x: f64) -> f64 {
    let (mut b1, mut b2) = (0.0, 0.0);
    for &c in coefficients.iter().skip(1).rev() {
        let b0 = 2.0 * x * b1 - b2 + c;
        b2 = b1;
        b1 = b0;
    }
    coefficients[0] + x * b1 - b2
}

// A(f) = factor(f) P(cos(pi f)); P has degree m-1 for Types I–IV.
fn remez_factor<S: RealScalar>(n: usize, symmetry: FirSymmetry, f: S) -> Result<S, SciOxideError> {
    let phase = core::f64::consts::PI * f.into_f64();
    let factor = match (symmetry, n.is_multiple_of(2)) {
        (FirSymmetry::Even, false) => return Ok(S::ONE),
        (FirSymmetry::Even, true) => (phase * 0.5).cos(),
        (FirSymmetry::Odd, false) => phase.sin(),
        (FirSymmetry::Odd, true) => (phase * 0.5).sin(),
    };
    S::try_from_f64_lossy(factor)
}

// The exchange owns validated real nodes. Use a short native quotient in the
// grid sweep; the shared anchored evaluator handles exceptional range/cancellation.
fn remez_evaluate_block<S: RealScalar>(
    nodes: &[S],
    weights: &[S],
    values: &[S],
    points: &[S],
    output: &mut [S],
) -> Result<(), SciOxideError> {
    if let Ok(points) = <&[S; 4]>::try_from(points) {
        let mut numerators = [S::ZERO; 4];
        let mut denominators = [S::ZERO; 4];
        let mut masses = [S::ZERO; 4];
        for ((&node, &weight), &value) in nodes.iter().zip(weights).zip(values) {
            for lane in 0..4 {
                let term = weight / (points[lane] - node);
                numerators[lane] += term * value;
                denominators[lane] += term;
                masses[lane] += term.abs();
            }
        }
        for (lane, output) in output.iter_mut().enumerate() {
            let value = numerators[lane] / denominators[lane];
            *output = if value.is_finite() && denominators[lane].abs() > S::EPSILON * masses[lane] {
                value
            } else {
                remez_evaluate(nodes, weights, values, points[lane])?
            };
        }
    } else {
        for (&point, output) in points.iter().zip(output) {
            *output = remez_evaluate(nodes, weights, values, point)?;
        }
    }
    Ok(())
}

fn remez_evaluate<S: RealScalar>(
    nodes: &[S],
    weights: &[S],
    values: &[S],
    x: S,
) -> Result<S, SciOxideError> {
    let mut numerator = S::ZERO;
    let mut denominator = S::ZERO;
    let mut mass = S::ZERO;
    for ((&node, &weight), &value) in nodes.iter().zip(weights).zip(values) {
        let gap = x - node;
        if gap == S::ZERO {
            return Ok(value);
        }
        let term = weight / gap;
        numerator += term * value;
        denominator += term;
        mass += term.abs();
    }
    let value = numerator / denominator;
    if value.is_finite() && denominator.abs() > S::EPSILON * mass {
        return Ok(value);
    }
    let right = nodes.partition_point(|&node| node > x).min(nodes.len() - 1);
    let reference = if right > 0 && (x - nodes[right - 1]).abs() < (x - nodes[right]).abs() {
        right - 1
    } else {
        right
    };
    Ok(
        scioxide_poly::backend::barycentric::evaluate::<_, _, false, true, false>(
            nodes,
            weights,
            |j| values[j],
            x,
            reference,
            S::ZERO,
        )?
        .value,
    )
}

fn remez_reconstruct<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    nodes: &[S],
    weights: &[S],
    values: &[S],
    taps: &mut [S],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    use scioxide_fft::{
        DctType, FftDirection, Normalization, RealTransformPlan, RealTransformPlanSpec,
    };
    let m = nodes.len() - 1;
    let count = scalar::<S>(m)?;
    let mut samples = try_filled(m, S::ZERO, scioxide_core::quantity!("remez_samples"))?;
    let mut coefficients = try_filled(m, S::ZERO, scioxide_core::quantity!("remez_coefficients"))?;
    for (i, value) in samples.iter_mut().enumerate() {
        execution.check()?;
        let x = S::try_from_f64_lossy((core::f64::consts::PI * (i as f64 + 0.5) / m as f64).cos())?;
        *value = remez_evaluate(nodes, weights, values, x)?;
    }
    let plan = RealTransformPlan::build(
        RealTransformPlanSpec::new(
            m,
            DctType::II.into(),
            FftDirection::Forward,
            Normalization::Unscaled,
        ),
        execution,
    )?;
    let mut workspace = plan.workspace(())?;
    plan.execute_into(&samples, &mut coefficients, &mut workspace, execution)?;
    for (j, value) in coefficients.iter_mut().enumerate() {
        *value = finite(
            *value
                * if j == 0 {
                    S::ONE / count
                } else {
                    S::TWO / count
                },
        )?;
    }
    // Multiplication by the symmetry factor is a neighbour sum/difference in
    // the cosine/sine basis; retain the original independent-amplitude convention.
    samples.fill(S::ZERO);
    match (symmetry, n.is_multiple_of(2)) {
        (FirSymmetry::Even, false) => samples.copy_from_slice(&coefficients),
        (FirSymmetry::Even, true) => {
            samples[0] = coefficients[0];
            for (k, &c) in coefficients.iter().enumerate().skip(1) {
                samples[k - 1] += c * S::HALF;
                samples[k] += c * S::HALF;
            }
        }
        (FirSymmetry::Odd, false) => {
            samples[0] = coefficients[0];
            for (k, &c) in coefficients.iter().enumerate().skip(1) {
                samples[k] += c * S::HALF;
                if k > 1 {
                    samples[k - 2] -= c * S::HALF;
                }
            }
        }
        (FirSymmetry::Odd, true) => {
            samples[0] = coefficients[0];
            for (k, &c) in coefficients.iter().enumerate().skip(1) {
                samples[k - 1] -= c * S::HALF;
                samples[k] += c * S::HALF;
            }
        }
    }
    fill_taps(taps, symmetry, &samples)
}

// Pure signed-extremum selection; caller reserves candidate storage once.
// As in the Parks–McClellan search, only extrema at least as large as the
// levelled error `floor` are admitted; current reference nodes, whose exact
// error is the levelled error, stay admissible under rounding. Every node lies
// in its own alternating lobe, so at least `reference.len()` alternating
// candidates remain, and discarded small lobes cannot displace them.
fn select_extrema<S: RealScalar>(
    errors: &[S],
    edges: &[(usize, usize)],
    floor: S,
    reference: &[usize],
    candidates: &mut Vec<usize>,
) {
    let count = reference.len();
    candidates.clear();
    let mut node = 0;
    for &(start, end) in edges {
        for i in start..end {
            while node < count && reference[node] < i {
                node += 1;
            }
            let admissible = errors[i].abs() >= floor || reference.get(node) == Some(&i);
            if admissible
                && ((errors[i] >= S::ZERO
                    && (i == start || errors[i] >= errors[i - 1])
                    && (i + 1 == end || errors[i] >= errors[i + 1]))
                    || (errors[i] < S::ZERO
                        && (i == start || errors[i] <= errors[i - 1])
                        && (i + 1 == end || errors[i] <= errors[i + 1])))
            {
                if let Some(&last) = candidates.last()
                    && (errors[last] >= S::ZERO) == (errors[i] >= S::ZERO)
                {
                    if errors[i].abs() > errors[last].abs()
                        && let Some(v) = candidates.last_mut()
                    {
                        *v = i;
                    }
                    continue;
                }
                candidates.push(i);
            }
        }
    }
    // Remove the weakest endpoint or adjacent pair while preserving alternation.
    while candidates.len() > count {
        let excess = candidates.len() - (count);
        let last = candidates.len() - 1;
        if excess == 1 {
            if errors[candidates[0]].abs() <= errors[candidates[last]].abs() {
                candidates.remove(0);
            } else {
                candidates.pop();
            }
            continue;
        }
        let mut weakest = (errors[candidates[0]].abs(), 0, 1);
        if errors[candidates[last]].abs() < weakest.0 {
            weakest = (errors[candidates[last]].abs(), last, 1);
        }
        for i in 0..last {
            let size = errors[candidates[i]]
                .abs()
                .max(errors[candidates[i + 1]].abs());
            if size < weakest.0 {
                weakest = (size, i, 2);
            }
        }
        candidates.drain(weakest.1..weakest.1 + weakest.2);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::string::String;
    use scioxide_core::{ConvergenceReason, SciOxideErrorView};

    fn bands() -> [FirBand<f64>; 2] {
        [
            FirBand::new(0.0, 0.3, 1.0, 1.0),
            FirBand::new(0.4, 1.0, 0.0, 0.0),
        ]
    }

    /// Reason and detail of the `ConvergenceFailure` that `design` maps to.
    fn failure(design: FirDesign<f64>) -> Option<(ConvergenceReason, String)> {
        match require_success(design).unwrap_err().view() {
            SciOxideErrorView::ConvergenceFailure { reason, detail, .. } => {
                Some((reason, detail.into()))
            }
            _ => None,
        }
    }

    #[test]
    fn unconverged_exchanges_are_convergence_failures() {
        let options = RemezOptions::new().with_max_iterations(1);
        let limited =
            design_fir_remez_with_options(31, FirSymmetry::Even, &bands(), options).unwrap();
        assert_eq!(limited.termination(), FirTermination::IterationLimit);
        let stalled = FirDesign {
            termination: FirTermination::ExchangeStalled,
            ..limited.clone()
        };
        assert_eq!(
            failure(limited),
            Some((
                ConvergenceReason::IterationLimit,
                "Remez exchange reached its iteration limit".into()
            ))
        );
        assert_eq!(
            failure(stalled),
            Some((
                ConvergenceReason::Stagnation,
                "Remez exchange stalled".into()
            ))
        );
    }

    #[test]
    fn default_design_succeeds_through_the_one_shot() {
        let options = RemezOptions::default();
        let configured =
            design_fir_remez_with_options(31, FirSymmetry::Even, &bands(), options).unwrap();
        let design = design_fir_remez(31, FirSymmetry::Even, &bands()).unwrap();
        assert_eq!(design.termination(), FirTermination::Converged);
        assert_eq!(design.coefficients(), configured.coefficients());
        let accepted = require_success(configured).unwrap();
        assert_eq!(accepted.coefficients(), design.coefficients());
    }
}
