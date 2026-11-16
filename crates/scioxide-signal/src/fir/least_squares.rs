use super::*;

// Independent coordinates are center-outward cosine/sine amplitudes, not taps.
fn order<S: RealScalar>(n: usize, symmetry: FirSymmetry, j: usize) -> Result<S, SciOxideError> {
    let offset = if n.is_multiple_of(2) {
        S::HALF
    } else if symmetry == FirSymmetry::Odd {
        S::ONE
    } else {
        S::ZERO
    };
    Ok(scalar::<S>(j)? + offset)
}
// Integral of linear desired amplitude times one sine/cosine basis.
fn linear_integral<S: RealScalar>(b: &FirBand<S>, k: S, odd: bool) -> Result<S, SciOxideError> {
    let two = S::TWO;
    let h = (b.upper - b.lower) / two;
    let x = S::PI * k * h;
    let phase = S::PI * k * (b.upper + b.lower) / two;
    // (sin(x)-x*cos(x))/x², with its removable singularity evaluated by series.
    let g = if x.abs() < S::try_from_f64_lossy(0.1)? {
        let x2 = x * x;
        x * (S::ONE / scalar::<S>(3)? - x2 / scalar::<S>(30)? + x2 * x2 / scalar::<S>(840)?
            - x2 * x2 * x2 / scalar::<S>(45360)?)
    } else {
        (x.sin() - x * x.cos()) / (x * x)
    };
    let base = if odd {
        sine_integral(k, b.lower, b.upper)?
    } else {
        cosine_integral(k, b.lower, b.upper)?
    };
    let slope_term = (b.desired_upper - b.desired_lower) * h * g;
    finite(
        (b.desired_lower / two + b.desired_upper / two) * base
            + slope_term * if odd { phase.cos() } else { -phase.sin() },
    )
}
/// Designs a linear-phase FIR filter of `n` taps by weighted least squares.
///
/// The taps minimize the weighted integral squared error of the zero-phase
/// amplitude `A` over the bands,
///
/// ```text
/// E = Σ_b W_b ∫_{l_b}^{u_b} (A(f) − D_b(f))² df,
/// ```
///
/// where band `b` covers `[l_b, u_b]` in multiples of Nyquist, `D_b` is its
/// linear desired amplitude and `W_b` its weight (see [`FirBand`]). The gaps
/// between bands are unconstrained. With `m` independent coefficients
/// (`⌈n/2⌉` for even symmetry, `⌊n/2⌋` for odd), the design forms the `m × m`
/// normal equations from closed-form integrals in `O(m² B)` operations for `B`
/// bands and solves them by Householder QR with column pivoting in `O(m³)`.
/// The design is direct and reports [`FirTermination::Completed`] and
/// [`FirDesign::condition_estimate`], the estimated one-norm condition number
/// of the normal equations.
///
/// # Errors
///
/// - `InvalidInput` if `n` is zero, if `n = 1` with odd symmetry, or if the
///   bands are invalid (see [`FirBand`]).
/// - `InvalidInput` if the desired amplitude is nonzero where the symmetry
///   forces a zero (see [`FirSymmetry`]).
/// - `NumericalInstability` (`LossOfSignificance`) if the normal equations are
///   numerically rank-deficient or the solve fails its check.
/// - `NumericalInstability` (`Underflow`) if the ratio of the smallest to the
///   largest band weight underflows.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{FirBand, FirSymmetry, design_fir_least_squares};
/// // Low-pass filter: pass band [0, 0.3], stop band [0.5, 1].
/// let bands = [FirBand::new(0.0_f64, 0.3, 1.0, 1.0), FirBand::new(0.5, 1.0, 0.0, 0.0)];
/// let design = design_fir_least_squares(31, FirSymmetry::Even, &bands)?;
/// assert!((design.response(0.0)?.norm() - 1.0).abs() < 1e-2);
/// assert!(design.response(0.8)?.norm() < 1e-2);
/// assert!(design.condition_estimate().is_some_and(|condition| condition >= 1.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// T. W. Parks and C. S. Burrus, *Digital Filter Design*, Wiley, 1987.
pub fn design_fir_least_squares<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    bands: &[FirBand<S>],
) -> Result<FirDesign<S>, SciOxideError> {
    let m = validate(n, symmetry, bands)?;
    let execution = ExecutionContext::sequential_native();
    let mut matrix = Matrix::zeros(m, m)?;
    let mut rhs = try_filled(
        m,
        S::ZERO,
        scioxide_core::quantity!("fir_least_squares_rhs"),
    )?;
    let weight_scale = bands.iter().fold(S::ZERO, |a, b| a.max(b.weight));
    let two = S::TWO;
    for (j, rhs_j) in rhs.iter_mut().enumerate() {
        let kj = order::<S>(n, symmetry, j)?;
        let mut r = scioxide_core::RangeAccumulator::new();
        for b in bands {
            if b.weight / weight_scale == S::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "band weights span an unrepresentable dynamic range",
                ));
            }
            r.add_product(
                b.weight / weight_scale,
                linear_integral(b, kj, symmetry == FirSymmetry::Odd)?,
            )?;
        }
        *rhs_j = r.finish()?;
        for k in 0..m {
            let kk = order::<S>(n, symmetry, k)?;
            let mut sum = scioxide_core::RangeAccumulator::new();
            for b in bands {
                let difference = cosine_integral(kj - kk, b.lower, b.upper)?;
                let addition = cosine_integral(kj + kk, b.lower, b.upper)?;
                let v = if symmetry == FirSymmetry::Even {
                    (difference + addition) / two
                } else {
                    (difference - addition) / two
                };
                sum.add_product(b.weight / weight_scale, v)?;
            }
            matrix.view_mut().set(j, k, sum.finish()?)?;
        }
    }
    let mut solve = DesignSolve::new(m, m)?;
    let condition = solve.solve(&matrix, &Vector::from_vec(rhs), execution)?;
    let mut result = from_amplitudes(n, symmetry, solve.solution.as_slice())?;
    result.condition = Some(condition);
    Ok(result)
}
