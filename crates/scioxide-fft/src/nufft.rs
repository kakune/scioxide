//! Nonuniform fast Fourier transforms with a Kaiser–Bessel kernel.
use crate::complex::ComplexCore;
use crate::validation::invalid;
use crate::validation::{nonfinite, numerical};
use crate::{FftDirection, FftWorkspaceRequirements, Normalization};
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    Argument, Complex, ExecutionContext, ExecutionPolicy, NdView, Quantity, RealScalar,
    SciOxideError,
};

mod execution;
mod preparation;

// Invalid inputs name their argument; resource limits name their quantity.

use crate::memory::admit_bytes;
fn scalar<S: RealScalar>(x: f64) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(x)
}
fn product(xs: &[usize], quantity: Quantity) -> Result<usize, SciOxideError> {
    xs.iter().try_fold(1, |a, &b| checked_mul(a, b, quantity))
}
fn integer(x: f64, quantity: Quantity) -> Result<usize, SciOxideError> {
    if !x.is_finite() || x < 0.0 || x >= (1_u64 << 52) as f64 {
        return Err(SciOxideError::size_overflow(quantity));
    }
    usize::try_from(x as u64).map_err(|_| SciOxideError::size_overflow(quantity))
}

/// Limits on the storage and work of a [`NufftPlan`].
///
/// All fields default to `usize::MAX`. Byte counts exclude allocator overhead
/// and the input views. Preparation checks every limit before it allocates and
/// returns `ResourceLimit` if one is exceeded; it never enlarges a limit.
#[derive(Debug, Clone, Copy)]
pub struct NufftLimits {
    /// Maximum bytes held at once during preparation, including the FFT plans.
    pub preparation_bytes: usize,
    /// Maximum bytes of an execution workspace, including staged output.
    pub workspace_bytes: usize,
    /// Maximum number of points in either oversampled grid.
    pub grid_points: usize,
    /// Maximum number of grid-point visits by spreading and interpolation in
    /// one application or adjoint application.
    pub stencil_visits: usize,
    /// Maximum kernel width `w`, in grid points per axis.
    pub kernel_support: usize,
}

impl Default for NufftLimits {
    fn default() -> Self {
        Self {
            preparation_bytes: usize::MAX,
            workspace_bytes: usize::MAX,
            grid_points: usize::MAX,
            stencil_visits: usize::MAX,
            kernel_support: usize::MAX,
        }
    }
}

/// Selects the NUFFT type and borrows its geometry.
///
/// Coordinates are the rows of a view of shape `[count, d]`. The types and the
/// mode order are defined on [`NufftPlan`].
#[derive(Debug)]
pub enum NufftTransform<'a, 'layout, S: RealScalar> {
    /// Type 1: from values at nonuniform points to a grid of integer modes.
    Type1 {
        /// Nonuniform coordinate rows.
        points: NdView<'a, 'layout, S>,
        /// Positive mode count on each coordinate axis.
        modes: &'a [usize],
    },
    /// Type 2: from a grid of integer modes to values at nonuniform points.
    Type2 {
        /// Nonuniform coordinate rows.
        points: NdView<'a, 'layout, S>,
        /// Positive mode count on each coordinate axis.
        modes: &'a [usize],
    },
    /// Type 3: from values at nonuniform points to arbitrary angular frequencies.
    Type3 {
        /// Nonuniform coordinate rows.
        points: NdView<'a, 'layout, S>,
        /// Angular frequency rows.
        frequencies: NdView<'a, 'layout, S>,
    },
}

/// Specifies a NUFFT: sign, normalization, accuracy, oversampling, limits and
/// execution options.
///
/// The defaults are oversampling ratio 2, no limits, sequential execution and
/// automatic backend selection.
#[derive(Debug, Clone, Copy)]
pub struct NufftPlanSpec<S: RealScalar> {
    accuracy: S,
    /// Storage and work limits.
    limits: NufftLimits,
    /// Minimum oversampling ratio, at least 2 for every transform type.
    oversampling: f64,
    /// Exponent sign: forward is negative, inverse is positive.
    direction: FftDirection,
    /// Explicit output scaling.
    normalization: Normalization,
    /// Execution policy of the FFTs; spreading runs in a fixed spatial order.
    execution_policy: ExecutionPolicy,
    backend_strategy: crate::FftBackendStrategy,
}
impl<S: RealScalar> NufftPlanSpec<S> {
    /// Sets the backend strategy of the FFTs, checked during preparation.
    pub const fn with_backend_strategy(mut self, strategy: crate::FftBackendStrategy) -> Self {
        self.backend_strategy = strategy;
        self
    }
    /// Returns the backend strategy.
    pub const fn backend_strategy(&self) -> crate::FftBackendStrategy {
        self.backend_strategy
    }

    /// Creates a spec with the sign of the exponent, the normalization and the
    /// requested accuracy relative to the input 1-norm.
    ///
    /// Preparation checks that `accuracy` lies in `(0, 1)`.
    pub fn new(direction: FftDirection, normalization: Normalization, accuracy: S) -> Self {
        Self {
            accuracy,
            limits: NufftLimits::default(),
            oversampling: 2.0,
            direction,
            normalization,
            execution_policy: ExecutionPolicy::Sequential,
            backend_strategy: crate::FftBackendStrategy::Automatic,
        }
    }
    /// Sets the storage and work limits.
    pub const fn with_limits(mut self, value: NufftLimits) -> Self {
        self.limits = value;
        self
    }
    /// Returns the storage and work limits.
    pub const fn limits(&self) -> NufftLimits {
        self.limits
    }
    /// Sets the minimum oversampling ratio `σ_o` of the grid.
    ///
    /// Preparation checks that it is finite and at least 2. Larger ratios
    /// allow a narrower kernel at the cost of a larger grid.
    pub const fn with_oversampling(mut self, value: f64) -> Self {
        self.oversampling = value;
        self
    }
    /// Returns the minimum oversampling ratio.
    pub const fn oversampling(&self) -> f64 {
        self.oversampling
    }
    /// Returns the direction, which sets the sign of the exponent.
    pub const fn direction(&self) -> FftDirection {
        self.direction
    }
    /// Returns the normalization.
    pub const fn normalization(&self) -> Normalization {
        self.normalization
    }
    /// Sets the execution policy of the FFTs.
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.execution_policy = value;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution_policy
    }
    /// Returns the requested maximum output error relative to the input 1-norm.
    pub const fn accuracy(&self) -> S {
        self.accuracy
    }
}

/// Reports the error estimates and sizes that preparation of a [`NufftPlan`]
/// chose.
///
/// The error estimates are defined in the [accuracy section](NufftPlan#accuracy)
/// of [`NufftPlan`]; they are not bounds. The requested values are read
/// through [`NufftPlan::spec`].
#[derive(Debug, Clone, Copy)]
pub struct NufftDiagnostics {
    /// Estimated aliasing error `E_a`.
    pub approximation_error: f64,
    /// Estimated rounding error `E_r`.
    pub roundoff_error: f64,
    /// Kernel width `w` in grid points per axis.
    pub kernel_support: usize,
    /// Number of points of the oversampled grid.
    pub grid_points: usize,
    /// Grid-point visits by spreading and interpolation per application.
    pub stencil_visits: usize,
    /// Peak bytes held during preparation. An injected FFT provider reports its
    /// own share through [`FftProviderPlan`](crate::FftProviderPlan).
    pub preparation_bytes: usize,
    /// Workspace bytes for one application or adjoint application.
    pub workspace_bytes: usize,
}

// Kernel coefficients are local power series on each unit grid interval.
// I0 is entire in its squared argument, so these intervals remain smooth at
// the support endpoints. Coefficients are prepared once in binary64.
const KERNEL_DEGREE: usize = 16;
#[derive(Debug, Clone)]
struct Kernel {
    dispatch: scioxide_core::backend::KernelDispatch,
    width: usize,
    beta: f64,
    inverse_i0: f64,
    degree: usize,
    coefficients: Vec<[f64; 4]>,
}
fn i0(x: f64) -> Result<f64, SciOxideError> {
    let q = x * x * 0.25;
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..256 {
        term *= q / (k * k) as f64;
        sum += term;
        if term <= f64::EPSILON * sum && sum.is_finite() {
            return Ok(sum);
        }
    }
    if !sum.is_finite() {
        return Err(crate::validation::arithmetic_overflow(
            "Kaiser–Bessel normalization exceeds finite range",
        ));
    }
    Err(numerical(
        scioxide_core::NumericalIssue::LossOfSignificance,
        "Kaiser–Bessel normalization series did not converge",
    ))
}
impl Kernel {
    fn degree(accuracy: f64) -> usize {
        ((-accuracy.log10()).ceil() as usize + 5).min(KERNEL_DEGREE)
    }
    fn coefficient_blocks(width: usize, accuracy: f64) -> usize {
        width.div_ceil(4) * Self::degree(accuracy).next_multiple_of(4)
    }
    fn new(
        width: usize,
        sigma: f64,
        accuracy: f64,
        dispatch: scioxide_core::backend::KernelDispatch,
    ) -> Result<Self, SciOxideError> {
        let beta = PI * ((width as f64 * (1.0 - 0.5 / sigma)).powi(2) - 0.8).sqrt();
        let inverse_i0 = 1.0 / i0(beta)?;
        let degree = Self::degree(accuracy);
        let mut coefficients = scioxide_core::backend::try_filled(
            Self::coefficient_blocks(width, accuracy),
            [0.0; 4],
            scioxide_core::quantity!("nufft_kernel"),
        )?;
        for offset in 0..width {
            // Expand I0(beta*sqrt(1-z²)) directly as a polynomial in
            // u=fraction-1/2. This avoids converting a noisy Chebyshev tail.
            let z = (width as f64 - 1.0 - 2.0 * offset as f64) / width as f64;
            let q = beta * beta * 0.25;
            let (a, b, c) = (
                q * (1.0 - z * z),
                -4.0 * q * z / width as f64,
                -4.0 * q / (width * width) as f64,
            );
            let mut term = [0.0; KERNEL_DEGREE];
            term[0] = inverse_i0;
            let mut coefficients_row = term;
            for k in 1..256 {
                let denominator = (k * k) as f64;
                for j in (0..KERNEL_DEGREE).rev() {
                    term[j] = (a * term[j]
                        + if j > 0 { b * term[j - 1] } else { 0.0 }
                        + if j > 1 { c * term[j - 2] } else { 0.0 })
                        / denominator;
                    coefficients_row[j] += term[j];
                }
                if k > width && term.iter().all(|t| t.abs() < f64::EPSILON * f64::EPSILON) {
                    break;
                }
            }
            if term.iter().chain(&coefficients_row).any(|v| !v.is_finite()) {
                return Err(crate::validation::arithmetic_overflow(
                    "Kaiser–Bessel coefficients exceed finite range",
                ));
            }
            if term.iter().any(|t| t.abs() >= f64::EPSILON * f64::EPSILON) {
                return Err(numerical(
                    scioxide_core::NumericalIssue::LossOfSignificance,
                    "Kaiser–Bessel coefficient series did not converge",
                ));
            }
            for (power, value) in coefficients_row[..degree].iter().enumerate() {
                coefficients[(offset / 4) * degree.next_multiple_of(4) + power][offset % 4] =
                    *value;
            }
        }
        Ok(Self {
            dispatch,
            width,
            beta,
            inverse_i0,
            degree: degree.next_multiple_of(4),
            coefficients,
        })
    }
    fn weights<S: RealScalar>(
        &self,
        fraction: f64,
        output: &mut [Complex<S>],
    ) -> Result<(), SciOxideError> {
        let x = fraction - 0.5;
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        // Four neighbouring offsets share coefficient vectors. Estrin groups
        // keep the arithmetic chains short; padding to four powers lets both
        // precisions vectorize without evaluating the unused high groups.
        for (out, block) in output
            .chunks_mut(4)
            .zip(self.coefficients.chunks_exact(self.degree))
        {
            let mut groups = [[0.0; 4]; 4];
            for (group, coefficients) in groups.iter_mut().zip(block.chunks_exact(4)) {
                for (lane, value) in group.iter_mut().enumerate() {
                    *value = (coefficients[0][lane] + coefficients[1][lane] * x)
                        + (coefficients[2][lane] + coefficients[3][lane] * x) * x2;
                }
            }
            for (lane, out) in out.iter_mut().enumerate() {
                let low = groups[0][lane] + groups[1][lane] * x4;
                let value = if self.degree > 8 {
                    low + (groups[2][lane] + groups[3][lane] * x4) * x8
                } else {
                    low
                };
                out.re = scalar(value)?;
            }
        }
        Ok(())
    }
    // Fourier transform of I0(beta sqrt(1-(2x/w)^2))/I0(beta).
    fn fourier(&self, theta: f64) -> f64 {
        let q = self.beta * self.beta - (theta * self.width as f64 * 0.5).powi(2);
        let sinhc = if q.abs() < 1e-8 {
            1.0 + q / 6.0 + q * q / 120.0
        } else if q > 0.0 {
            let r = q.sqrt();
            r.sinh() / r
        } else {
            let r = (-q).sqrt();
            r.sin() / r
        };
        self.width as f64 * self.inverse_i0 * sinhc
    }
}

#[derive(Debug, Clone, Copy)]
struct PointAxis {
    start: usize,
    fraction: f64,
}
#[derive(Debug, Clone, Copy)]
struct StencilAxis {
    extent: usize,
    stride: usize,
    support: usize,
}
const SPREAD_BIN: usize = 8;
#[derive(Debug, Clone, Copy)]
struct SpreadBin {
    start: usize,
    end: usize,
    origin: [usize; 3],
}
fn spread_bins(extents: [usize; 3], count: usize) -> usize {
    if count < 4096 || extents.iter().any(|&n| n < SPREAD_BIN) {
        return 0;
    }
    // Caller has already checked the product of the full grid extents.
    let bins = extents
        .iter()
        .map(|&n| n.div_ceil(SPREAD_BIN))
        .product::<usize>();
    if bins <= count / 16 { bins } else { 0 }
}
fn spread_chunk<S: RealScalar>() -> usize {
    if S::EPSILON > S::try_from_f64_lossy(1e-10).unwrap_or(S::EPSILON) {
        4
    } else {
        64
    }
}

#[derive(Debug, Clone)]
struct Stencil<S: RealScalar> {
    positions: Vec<PointAxis>,
    factors: Vec<Complex<S>>,
    order: Vec<usize>,
    axes: Vec<StencilAxis>,
    bins: Vec<SpreadBin>,
}
impl<S: RealScalar> Stencil<S> {
    fn factor(&self, point: usize) -> Complex<S> {
        if self.factors.is_empty() {
            Complex::new(S::ONE, S::ZERO)
        } else {
            self.factors[point]
        }
    }

    fn weights(
        &self,
        point: usize,
        kernel: &Kernel,
        weights: &mut [Complex<S>],
    ) -> Result<(), SciOxideError> {
        for (axis, (&position, weights)) in self.axes.iter().zip(
            self.positions[point * self.axes.len()..]
                .iter()
                .zip(weights.chunks_exact_mut(kernel.width)),
        ) {
            if axis.support == 1 {
                weights[0].re = S::ONE;
            } else {
                kernel.weights(position.fraction, weights)?;
            }
        }
        Ok(())
    }
    #[inline]
    fn for_each_row(
        &self,
        point: usize,
        width: usize,
        weights: &[Complex<S>],
        mut visit: impl FnMut(usize, S, &[Complex<S>]),
    ) {
        let positions = &self.positions[point * self.axes.len()..][..self.axes.len()];
        #[inline(always)]
        fn axis<S: RealScalar>(
            a: StencilAxis,
            p: PointAxis,
            weights: &[Complex<S>],
            mut visit: impl FnMut(usize, S),
        ) {
            let mut i = p.start;
            for w in &weights[..a.support] {
                visit(i * a.stride, w.re);
                i += 1;
                if i == a.extent {
                    i = 0;
                }
            }
        }
        #[inline(always)]
        fn row<S: RealScalar>(
            a: StencilAxis,
            p: PointAxis,
            weights: &[Complex<S>],
            index: usize,
            weight: S,
            visit: &mut impl FnMut(usize, S, &[Complex<S>]),
        ) {
            debug_assert_eq!(a.stride, 1);
            let head = a.support.min(a.extent - p.start);
            visit(index + p.start, weight, &weights[..head]);
            if head < a.support {
                visit(index, weight, &weights[head..a.support]);
            }
        }
        match self.axes.as_slice() {
            [] => visit(0, S::ONE, &[Complex::new(S::ONE, S::ZERO)]),
            [a] => row(*a, positions[0], weights, 0, S::ONE, &mut visit),
            [a, b] => axis(*a, positions[0], weights, |i, x| {
                row(*b, positions[1], &weights[width..], i, x, &mut visit)
            }),
            [a, b, c] => axis(*a, positions[0], weights, |i, x| {
                axis(*b, positions[1], &weights[width..], |j, y| {
                    row(
                        *c,
                        positions[2],
                        &weights[2 * width..],
                        i + j,
                        x * y,
                        &mut visit,
                    )
                });
            }),
            _ => {
                fn walk<S: RealScalar>(
                    axes: &[StencilAxis],
                    positions: &[PointAxis],
                    width: usize,
                    weights: &[Complex<S>],
                    index: usize,
                    weight: S,
                    visit: &mut impl FnMut(usize, S, &[Complex<S>]),
                ) {
                    if let Some((&a, rest)) = axes.split_first() {
                        if rest.is_empty() {
                            row(a, positions[0], weights, index, weight, visit);
                        } else {
                            axis(a, positions[0], weights, |i, w| {
                                walk(
                                    rest,
                                    &positions[1..],
                                    width,
                                    &weights[width..],
                                    index + i,
                                    weight * w,
                                    visit,
                                )
                            });
                        }
                    }
                }
                walk(&self.axes, positions, width, weights, 0, S::ONE, &mut visit);
            }
        }
    }
}
#[derive(Debug, Clone)]
struct ModeAxis<S: RealScalar> {
    indices: Vec<usize>,
    weights: Vec<S>,
}
fn modes_each<S: RealScalar>(axes: &[ModeAxis<S>], mut visit: impl FnMut(usize, usize, S)) {
    fn walk<S: RealScalar>(
        axes: &[ModeAxis<S>],
        index: usize,
        weight: S,
        logical: &mut usize,
        visit: &mut impl FnMut(usize, usize, S),
    ) {
        if let Some((axis, rest)) = axes.split_first() {
            for (&i, &w) in axis.indices.iter().zip(&axis.weights) {
                walk(rest, index + i, weight * w, logical, visit);
            }
        } else {
            visit(*logical, index, weight);
            *logical += 1;
        }
    }
    walk(axes, 0, S::ONE, &mut 0, &mut visit);
}

/// Computes nonuniform fast Fourier transforms (NUFFT) of types 1, 2 and 3 in
/// any dimension `d`.
///
/// Let `x_j ∈ ℝᵈ`, `j = 0, …, M − 1`, be the nonuniform points, `σ = −1` for
/// [`FftDirection::Forward`] and `σ = +1` for [`FftDirection::Inverse`]. The
/// three types compute
///
/// ```text
/// type 1:  f_k = s Σⱼ c_j exp(σ i k·x_j),   k ∈ K,
/// type 2:  c_j = s Σ_k f_k exp(σ i k·x_j),   k ∈ K,
/// type 3:  f_k = s Σⱼ c_j exp(σ i ω_k·x_j),
/// ```
///
/// where `ω_k ∈ ℝᵈ` are arbitrary target frequencies and `K` is the grid of
/// integer modes with `N_a` modes along axis `a`. Along each axis the modes
/// are `−⌊N_a/2⌋, …, ⌈N_a/2⌉ − 1` in increasing order, and the grid is stored
/// in row-major order (last axis fastest). Types 1 and 2 are 2π-periodic in
/// each coordinate. Points and frequencies are the rows of rank-two views of
/// shape `[count, d]`. Type 2 is a Fourier sum, not the inverse of type 1.
///
/// The scale `s` follows the [`Normalization`], applied to the number of
/// summed terms: the number of points `M` for types 1 and 3 and the number of modes
/// `N_1 ⋯ N_d` for type 2. [`Normalization::Forward`] and
/// [`Normalization::Inverse`] divide by that count in the matching direction
/// only, [`Normalization::Orthonormal`] divides by its square root, and
/// [`Normalization::Unscaled`] does not scale. Empty sums are zero. The
/// adjoint, applied through [`AdjointOperator`](scioxide_core::AdjointOperator),
/// is the conjugate transpose with the same scale.
///
/// # Method
///
/// The plan spreads the input onto an oversampled uniform grid with a
/// Kaiser–Bessel kernel of width `w`, transforms the grid with FFTs, and
/// corrects by the kernel's Fourier transform (Barnett, Magland and
/// af Klinteberg, 2019). Type 2 corrects first and interpolates from the grid
/// instead of spreading onto it. Type 3 spreads the rescaled points onto a
/// grid, transforms it, and interpolates at the rescaled frequencies. Along an axis
/// with `N_a > 1` the grid has the smallest power of two not below
/// `max(⌈σ_o N_a⌉, w + 2)` points, where `σ_o ≥ 2` is the oversampling ratio.
/// The kernel shape parameter is
/// `β = π √((w (1 − 1/(2σ_o)))² − 0.8)` (Beatty, Nishimura and Pauly, 2005).
/// Preparation chooses the smallest `w ≥ 4` whose combined error estimate
/// below meets the requested accuracy.
///
/// Spreading and interpolation visit `wᵈ` grid points per nonuniform point;
/// type 3 does both, for the points and for the frequencies.
/// [`NufftDiagnostics::stencil_visits`] reports the total. The plan
/// stores `O(M d + Σ_a N_a + d w)` values besides its FFT tables. The
/// workspace is dominated by two copies of the oversampled grid.
///
/// # Accuracy
///
/// The requested accuracy `ε_r` bounds the maximum output error relative to
/// the 1-norm of the input. Preparation estimates an aliasing error `E_a` and
/// a rounding error `E_r`:
///
/// ```text
/// E_a = (1 + a)^r − 1,   a = 3 exp(−π (w − 1) √(1 − 1/σ_o)),
/// E_r = ε (4 + C + log₂ max(2, G)) + ε₆₄ (2P + 8d),
/// ```
///
/// where `r = d` for types 1 and 2 and `r = 2d` for type 3, `ε` is the machine
/// epsilon of `S`, `ε₆₄` that of `f64`, and `G` the number of points of the
/// oversampled grid. `P` is the sum of `N_a` over the axes on which some
/// coordinate has `|x| > π`; it is zero when all coordinates lie in `[−π, π]`.
/// `C` is 64 for `f64` and 4 for `f32` when three-dimensional spreading groups
/// points into local tiles, and zero otherwise. The rate in `a` is that of the
/// Kaiser–Bessel kernel; the prefactor 3 is empirical, checked against direct
/// sums. Both are estimates, not bounds; cancellation can make the observed
/// error much smaller. A request below `E_r` fails with
/// `NumericalInstability`. [`Self::diagnostics`] reports both estimates.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext, NdView, StridedLayout};
/// use scioxide_fft::{FftDirection, Normalization, NufftPlan, NufftPlanSpec, NufftTransform};
///
/// // A unit source at x = 0 has the coefficient 1 at every mode.
/// let points = NdView::new(&[0.0_f64], StridedLayout::new(&[1, 1], &[1, 1])?)?;
/// let spec = NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-8);
/// let modes: &[usize] = &[3];
/// let plan = NufftPlan::new(NufftTransform::Type1 { points, modes }, spec)?;
/// let (input, mut result) = ([Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0); 3]);
/// let cx = ExecutionContext::sequential_native();
/// plan.execute_into(&input[..], &mut result[..], &mut plan.workspace(())?, cx)?;
/// for z in result {
///     assert!((z - Complex::new(1.0, 0.0)).norm() < 1e-8);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. H. Barnett, J. Magland and L. af Klinteberg, "A parallel nonuniform
///   fast Fourier transform library based on an 'exponential of semicircle'
///   kernel", *SIAM J. Sci. Comput.* 41(5), C479–C504, 2019, equations 5–7.
///   <https://arxiv.org/abs/1808.06736>
/// - P. J. Beatty, D. G. Nishimura and J. M. Pauly, "Rapid gridding
///   reconstruction with a minimal oversampling ratio", *IEEE Trans. Med.
///   Imaging* 24(6), 799–808, 2005. <https://doi.org/10.1109/TMI.2005.848376>
#[derive(Debug, Clone)]
pub struct NufftPlan<S: RealScalar> {
    plan_bytes: usize,
    geometry: PreparedGeometry<S>,
    spec: NufftPlanSpec<S>,
    input_len: usize,
    output_len: usize,
    shape: Vec<usize>,
    grid_shape: Vec<usize>,
    grid_strides: Vec<usize>,
    fft: Vec<ComplexCore<S>>,
    // Centered logical modes -> periodic FFT slots and diagonal deconvolution.
    modes: Vec<ModeAxis<S>>,
    base_len: usize,
    kernel: Kernel,
    scale: S,
    scratch_len: usize,
    lane_len: usize,
    tile_len: usize,
    headroom: i32,
    bounded: bool,
    diagnostics: NufftDiagnostics,
}

#[derive(Debug, Clone)]
enum PreparedGeometry<S: RealScalar> {
    One(Stencil<S>),
    Two(Stencil<S>),
    Three {
        source: Stencil<S>,
        target: Stencil<S>,
    },
}

impl<S: RealScalar> NufftPlan<S> {
    /// Returns the bytes the plan occupies, including its FFT plans.
    pub const fn plan_bytes(&self) -> usize {
        self.plan_bytes
    }

    /// Returns the specification the plan was prepared from.
    pub const fn spec(&self) -> &NufftPlanSpec<S> {
        &self.spec
    }

    /// Prepares a plan for the given transform with the default processor
    /// capabilities.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if the accuracy, the oversampling ratio, a coordinate or a
    ///   frequency is NaN or infinite.
    /// - `DimensionMismatch` if a view is not of rank two, or the number of mode
    ///   counts or the frequency dimension differs from the point dimension.
    /// - `InvalidInput` (`OutOfRange`) if the accuracy is not in `(0, 1)` or the
    ///   oversampling ratio is below 2.
    /// - `InvalidInput` (`Empty`) if a mode count is zero.
    /// - `NumericalInstability` if the requested accuracy is below the rounding
    ///   estimate `E_r`.
    /// - `ResourceLimit` if a limit in [`NufftLimits`] is exceeded.
    /// - `SizeOverflow` or `AllocationFailure` if storage cannot be sized or
    ///   reserved.
    pub fn new(
        transform: NufftTransform<'_, '_, S>,
        spec: NufftPlanSpec<S>,
    ) -> Result<Self, SciOxideError> {
        Self::build(transform, spec, crate::FftPlanningContext::default())
    }
    /// Prepares a plan with the processor capabilities, FFT provider and
    /// execution context of `context`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], plus:
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the spec requires a
    ///   backend that `context` does not verify.
    /// - `ExecutionStopped` on cancellation or deadline.
    /// - Errors from the FFT provider.
    pub fn build<'a>(
        transform: NufftTransform<'_, '_, S>,
        spec: NufftPlanSpec<S>,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        preparation::prepare(transform, spec, context.into())
    }
    /// Returns the backends of the per-axis grid FFTs, in axis order.
    pub fn resolved_backends(&self) -> impl Iterator<Item = crate::FftBackend> + '_ {
        self.fft.iter().map(ComplexCore::resolved_backend)
    }
    /// Returns the length of one input vector.
    pub const fn input_len(&self) -> usize {
        self.input_len
    }
    /// Returns the length of one output vector.
    pub const fn output_len(&self) -> usize {
        self.output_len
    }
    /// Returns the dimension `d`.
    pub fn dimension(&self) -> usize {
        self.shape.len()
    }
    /// Returns the error estimates and sizes chosen during preparation.
    pub const fn diagnostics(&self) -> NufftDiagnostics {
        self.diagnostics
    }
    /// Returns the number of points of the oversampled grid along each axis.
    pub fn grid_shape(&self) -> &[usize] {
        &self.grid_shape
    }

    /// Returns the workspace size for [`Self::execute_into`] (request `()`) or
    /// for [`Self::execute_axis_into`] (an [`AxisWorkspaceRequest`](crate::AxisWorkspaceRequest)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` or `InvalidInput` if an axis request does not fit
    ///   the plan.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds `limits.workspace_bytes`.
    pub fn workspace_requirements<'r>(
        &self,
        request: impl Into<crate::AxisWorkspaceRequest<'r>>,
    ) -> Result<FftWorkspaceRequirements, SciOxideError> {
        crate::axis::requirements(self, request.into())
    }

    /// Allocates a workspace for the given request.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<'r>(
        &self,
        request: impl Into<crate::AxisWorkspaceRequest<'r>>,
    ) -> Result<crate::FftWorkspace<S>, SciOxideError> {
        crate::workspace::allocate(&self.workspace_requirements(request)?)
    }
}
