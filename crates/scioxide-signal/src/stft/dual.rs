//! Finite-support painless Gabor duals, projected residue class by residue class.
use super::*;
use crate::util::sum_products;
use scioxide_core::AlgebraicScalar;

/// Overlap diagnostics of a window and hop, as [`check_overlap`] computes them.
///
/// For hop `h`, residue `r = 0, …, h − 1` collects the window coefficients
/// `w[r + q·h]`, `q = 0, 1, …`. The diagnostics do not depend on the scale of
/// the window and carry no error bound.
#[derive(Clone, Copy, Debug)]
pub struct OverlapDiagnostics<S: RealScalar> {
    cola: bool,
    nola: bool,
    energy_ratio: S,
}
impl<S: RealScalar> OverlapDiagnostics<S> {
    /// Returns whether the window satisfies the constant overlap-add (COLA)
    /// condition: the residue sums `Σ_q w[r + q·h]` are equal within the
    /// tolerance.
    pub fn cola(self) -> bool {
        self.cola
    }
    /// Returns whether the window satisfies the nonzero overlap-add (NOLA)
    /// condition, with [`OverlapDiagnostics::energy_ratio`] above the
    /// tolerance. Synthesis requires NOLA.
    pub fn nola(self) -> bool {
        self.nola
    }
    /// Returns the smallest residue energy divided by the largest,
    /// `min_r E_r / max_r E_r` with `E_r = Σ_q |w[r + q·h]|²`.
    ///
    /// The ratio is zero when a residue has zero energy. A small ratio means
    /// that synthesis amplifies errors in the coefficients.
    pub fn energy_ratio(self) -> S {
        self.energy_ratio
    }
}
/// Whether [`closest_stft_dual`] may rescale the analysis window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualScale {
    /// Keeps the analysis window, so [`ClosestStftDual::window_scale`] is one.
    Fixed,
    /// Rescales the analysis window by the factor that brings the dual
    /// closest to the desired window.
    Optimized,
}

fn validate<T: NumericScalar>(window: &[T], hop: usize) -> Result<(), SciOxideError> {
    if window.is_empty() {
        return Err(empty(
            scioxide_core::argument!("window"),
            "window must be nonempty",
        ));
    }
    if hop == 0 {
        return Err(invalid(
            scioxide_core::argument!("hop"),
            "hop must be positive",
        ));
    }
    finite(window)
}

/// Checks the COLA and NOLA conditions of `window` with hop `hop`.
///
/// With `h = hop`, the window `w` divided by its largest component magnitude,
/// and `(atol, rtol)` from `tolerance`, each residue `r = 0, …, h − 1` has the
/// sum and energy
///
/// ```text
/// S_r = Σ_q w[r + q·h],    E_r = Σ_q |w[r + q·h]|².
/// ```
///
/// COLA holds when `max_r |S_r − S_0| ≤ atol + rtol · max_r |S_r|`, and NOLA
/// holds when `min_r E_r / max_r E_r > atol + rtol`. A hop longer than the
/// window, or an all-zero window, fails both. The cost is O(m) for a window of
/// length `m`.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `window` is empty.
/// - `InvalidInput` (`OutOfRange`) if `hop` is zero.
/// - `NonFinite` if a window coefficient is NaN or infinite.
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_signal::check_overlap;
///
/// let hann = [0.0_f64, 0.5, 1.0, 0.5]; // periodic Hann of length 4
/// let half = check_overlap(&hann, 2, Tolerance::from_sqrt_epsilon())?;
/// assert!(half.cola() && half.nola());
/// assert!((half.energy_ratio() - 0.5).abs() < 1e-12);
/// assert!(!check_overlap(&hann, 5, Tolerance::from_sqrt_epsilon())?.nola());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn check_overlap<T: NumericScalar>(
    window: &[T],
    hop: usize,
    tolerance: Tolerance<T::Real>,
) -> Result<OverlapDiagnostics<T::Real>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    validate(window, hop)?;
    let scale = crate::util::scale(window);
    if scale == T::Real::ZERO || hop > window.len() {
        return Ok(OverlapDiagnostics {
            cola: false,
            nola: false,
            energy_ratio: T::Real::ZERO,
        });
    }
    let mut minimum = T::Real::INFINITY;
    let mut maximum = T::Real::ZERO;
    let mut first = zero::<T::Real>();
    let mut deviation = T::Real::ZERO;
    let mut sum_scale = T::Real::ZERO;
    for r in 0..hop {
        let terms = || {
            window
                .iter()
                .skip(r)
                .step_by(hop)
                .map(|&w| complex(w).unscale(scale))
        };
        let energy = sum_products(terms().map(|w| (w.conj(), w)))?.re;
        let sum = sum_products(terms().map(|w| (w, Complex::new(T::Real::ONE, T::Real::ZERO))))?;
        if r == 0 {
            first = sum;
        }
        deviation = deviation.max((sum - first).norm());
        sum_scale = sum_scale.max(sum.norm());
        minimum = minimum.min(energy);
        maximum = maximum.max(energy);
    }
    let ratio = minimum / maximum;
    Ok(OverlapDiagnostics {
        cola: tolerance.is_residual_ok(deviation, sum_scale)?,
        nola: ratio > tolerance.threshold(T::Real::ONE)?,
        energy_ratio: ratio,
    })
}

/// Computes the minimum-norm dual window of `window` for hop `hop`.
///
/// With `h = hop`, a dual `d` of the window `w` satisfies
/// `Σ_q conj(w[r + q·h]) d[r + q·h] = 1` for each residue `r = 0, …, h − 1`.
/// Among these, the dual with the smallest Euclidean norm is
///
/// ```text
/// d[j] = w[j] / E_r,   E_r = Σ_q |w[r + q·h]|²,   r = j mod h.
/// ```
///
/// Synthesis with this dual gives the least-squares estimate of a signal from
/// modified STFT coefficients. The dual of `d` is `w` again. The cost is O(m)
/// for a window of length `m`.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `window` is empty.
/// - `InvalidInput` (`OutOfRange`) if `hop` is zero.
/// - `InvalidInput` (`Inconsistent`) if the window and hop violate NOLA, as
///   [`check_overlap`] decides with `tolerance`.
/// - `NonFinite` if a window coefficient is NaN or infinite.
/// - `NumericalInstability` if a dual coefficient overflows, or the dual fails
///   its reconstruction check at `tolerance` (`LossOfSignificance`).
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_signal::canonical_stft_dual;
///
/// let hann = [0.0_f64, 0.5, 1.0, 0.5];
/// let dual = canonical_stft_dual(&hann, 2, Tolerance::from_sqrt_epsilon())?;
/// // Residue energies are 1 and 1/2, so the dual is the window divided by them.
/// assert!(dual.iter().zip([0.0, 1.0, 1.0, 1.0]).all(|(d, e)| (d - e).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time
///   Fourier transform", *IEEE Trans. Acoust., Speech, Signal Process.* 32(2),
///   236–243, 1984.
/// - I. Daubechies, A. Grossmann and Y. Meyer, "Painless nonorthogonal
///   expansions", *J. Math. Phys.* 27(5), 1271–1283, 1986.
pub fn canonical_stft_dual<T: NumericScalar>(
    window: &[T],
    hop: usize,
    tolerance: Tolerance<T::Real>,
) -> Result<Vec<T>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    if !check_overlap(window, hop, tolerance)?.nola {
        return Err(inconsistent(
            scioxide_core::argument!("window"),
            "window and hop violate the nonzero overlap-add condition",
        ));
    }
    let scale = crate::util::scale(window);
    let mut dual = try_filled(window.len(), T::ZERO, scioxide_core::quantity!("stft_dual"))?;
    for r in 0..hop {
        let energy = sum_products(window.iter().skip(r).step_by(hop).map(|&w| {
            let z = complex(w).unscale(scale);
            (z.conj(), z)
        }))?
        .re;
        for (slot, &w) in dual.iter_mut().zip(window).skip(r).step_by(hop) {
            let z = complex(w).unscale(scale).unscale(energy).unscale(scale);
            *slot = T::from_components(z.re, z.im);
        }
    }
    finite(&dual)?;
    validate_dual(window, &dual, hop, tolerance, computed_dual_violation)?;
    Ok(dual)
}

/// A computed dual that fails its own reconstruction check lost precision.
fn computed_dual_violation() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::LossOfSignificance,
        "computed dual fails the reconstruction constraint",
    )
}

/// Check the dual reconstruction constraint; `violation` builds the error
/// for a dual of the right length that does not reconstruct the window: an
/// inconsistent caller dual or a computed dual that lost precision.
pub(super) fn validate_dual<T: NumericScalar>(
    window: &[T],
    dual: &[T],
    hop: usize,
    tolerance: Tolerance<T::Real>,
    violation: impl Fn() -> SciOxideError,
) -> Result<(), SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    exact_len(window.len(), dual.len())?;
    finite(dual)?;
    for r in 0..hop {
        let value = sum_products(
            window
                .iter()
                .zip(dual)
                .skip(r)
                .step_by(hop)
                .map(|(&w, &d)| (complex(w).conj(), complex(d))),
        )?;
        if !tolerance.is_residual_ok(
            (value - Complex::new(T::Real::ONE, T::Real::ZERO)).norm(),
            T::Real::ONE,
        )? {
            return Err(violation());
        }
    }
    Ok(())
}

/// Dual window closest to a desired window, as [`closest_stft_dual`] computes
/// it.
///
/// The dual reconstructs with the analysis window `window · window_scale`.
#[derive(Clone, Debug)]
pub struct ClosestStftDual<T: NumericScalar> {
    dual: Vec<T>,
    window_scale: T,
}
impl<T: NumericScalar> ClosestStftDual<T> {
    /// Returns the dual (synthesis) window.
    pub fn dual(&self) -> &[T] {
        &self.dual
    }
    /// Returns the factor by which to multiply the analysis window before
    /// using [`ClosestStftDual::dual`]. It is one with [`DualScale::Fixed`].
    pub fn window_scale(&self) -> T {
        self.window_scale
    }
}

/// Computes the dual window of `window` for hop `hop` that is closest to
/// `desired` in the Euclidean norm.
///
/// With [`DualScale::Fixed`], the result is the orthogonal projection of
/// `desired` onto the duals of `window`, defined at [`canonical_stft_dual`].
/// With [`DualScale::Optimized`], the analysis window is also multiplied by
/// the factor [`ClosestStftDual::window_scale`], chosen so that the dual is as
/// close to `desired` as possible over all such factors. Both are closed-form
/// projections that cost O(m) for a window of length `m`.
///
/// # Errors
///
/// - The errors of [`canonical_stft_dual`].
/// - `DimensionMismatch` if `desired` and `window` differ in length.
/// - `NonFinite` if a coefficient of `desired` is NaN or infinite.
/// - `InvalidInput` (`OutOfRange`) if, with [`DualScale::Optimized`],
///   `desired` has a negligible component along the minimum-norm dual.
/// - `NumericalInstability` if the scaled window overflows, or the dual fails
///   its reconstruction check at `tolerance` (`LossOfSignificance`).
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_signal::{DualScale, closest_stft_dual};
///
/// let window = [1.0_f64; 4];
/// let tolerance = Tolerance::from_sqrt_epsilon();
/// // A constant 0.5 is already a dual of this window, so it is kept.
/// let result = closest_stft_dual(&window, 2, &[0.5; 4], DualScale::Fixed, tolerance)?;
/// assert_eq!(result.window_scale(), 1.0);
/// assert!(result.dual().iter().all(|d| (d - 0.5).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn closest_stft_dual<T: NumericScalar>(
    window: &[T],
    hop: usize,
    desired: &[T],
    scale: DualScale,
    tolerance: Tolerance<T::Real>,
) -> Result<ClosestStftDual<T>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    exact_len(window.len(), desired.len())?;
    finite(desired)?;
    let canonical = canonical_stft_dual(window, hop, tolerance)?;
    let one = Complex::new(T::Real::ONE, T::Real::ZERO);
    let beta = if scale == DualScale::Optimized {
        let numerator = sum_products(
            canonical
                .iter()
                .zip(desired)
                .map(|(&a, &b)| (complex(a).conj(), complex(b))),
        )?;
        let denominator = sum_products(canonical.iter().map(|&a| (complex(a).conj(), complex(a))))?;
        let value = numerator / denominator;
        if value.norm() <= tolerance.threshold(T::Real::ONE)? {
            return Err(invalid(
                scioxide_core::argument!("desired"),
                "desired dual has a negligible canonical component",
            ));
        }
        value
    } else {
        one
    };
    let factor = one / beta.conj();
    let mut dual = copy(desired)?;
    for r in 0..hop {
        let constraint = sum_products(
            window
                .iter()
                .zip(desired)
                .skip(r)
                .step_by(hop)
                .map(|(&w, &d)| (complex(w).conj(), complex(d))),
        )?;
        for (slot, &d) in dual.iter_mut().zip(&canonical).skip(r).step_by(hop) {
            let z =
                sum_products([(complex(*slot), one), (complex(d), beta - constraint)].into_iter())?;
            *slot = T::from_components(z.re, z.im);
        }
    }
    let window_scale = T::from_components(factor.re, factor.im);
    let mut scaled = copy(window)?;
    for w in &mut scaled {
        *w = *w * window_scale;
    }
    finite(&scaled)?;
    validate_dual(&scaled, &dual, hop, tolerance, computed_dual_violation)?;
    Ok(ClosestStftDual { dual, window_scale })
}

/// Computes the tight window of `window` for hop `hop`: a window that is its
/// own dual.
///
/// With `h = hop`, each residue class of the window `w` is divided by the
/// square root of its energy,
///
/// ```text
/// g[j] = w[j] / √E_r,   E_r = Σ_q |w[r + q·h]|²,   r = j mod h,
/// ```
///
/// so every residue of `g` has unit energy and the same window serves
/// analysis and synthesis. The cost is O(m) for a window of length `m`.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `window` is empty.
/// - `InvalidInput` (`OutOfRange`) if `hop` is zero.
/// - `InvalidInput` (`Inconsistent`) if the window and hop violate NOLA, as
///   [`check_overlap`] decides with `tolerance`.
/// - `NonFinite` if a window coefficient is NaN or infinite.
/// - `NumericalInstability` (`LossOfSignificance`) if the result fails its
///   reconstruction check at `tolerance`.
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_signal::tight_stft_window;
///
/// let hann = [0.0_f64, 0.5, 1.0, 0.5];
/// let tight = tight_stft_window(&hann, 2, Tolerance::from_sqrt_epsilon())?;
/// for residue in 0..2 {
///     let energy: f64 = tight.iter().skip(residue).step_by(2).map(|w| w * w).sum();
///     assert!((energy - 1.0).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn tight_stft_window<T: NumericScalar>(
    window: &[T],
    hop: usize,
    tolerance: Tolerance<T::Real>,
) -> Result<Vec<T>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    if !check_overlap(window, hop, tolerance)?.nola {
        return Err(inconsistent(
            scioxide_core::argument!("window"),
            "window and hop violate the nonzero overlap-add condition",
        ));
    }
    let scale = crate::util::scale(window);
    let mut result = copy(window)?;
    for r in 0..hop {
        let energy = sum_products(window.iter().skip(r).step_by(hop).map(|&w| {
            let z = complex(w).unscale(scale);
            (z.conj(), z)
        }))?
        .re
        .sqrt();
        for v in result.iter_mut().skip(r).step_by(hop) {
            let z = complex(*v).unscale(scale).unscale(energy);
            *v = T::from_components(z.re, z.im);
        }
    }
    validate_dual(&result, &result, hop, tolerance, computed_dual_violation)?;
    Ok(result)
}
