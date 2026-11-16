//! Bounded triangular matrix-function kernels on borrowed workspace slots.
//!
//! Fifteen `n x n` complex slots live in one scalar slice: 0 holds the input,
//! 1 the result, 2..=8 the shared dense polynomial and recurrence scratch,
//! 9..=12 LU scratch, and 13/14 the reordering basis and saved numerator.
use super::prepared::{SchurFunction, Trigonometric};
use super::{AnalyticMatrixFunction, FunctionControl, MatrixFunctionOptions};
use scioxide_core::AlgebraicScalar;
use scioxide_core::ScaledValue;
use scioxide_core::{Argument, Complex, DomainReason, NumericScalar, RealScalar, SciOxideError};

/// Dense kernel slots plus the reordering basis and tangent numerator.
pub(crate) const SLOTS: usize = super::dense::SLOTS + 2;
const BASIS: usize = 13;
const NUMERATOR: usize = 14;
const STAGING: usize = 8;

/// Entire scalar functions with cyclic derivatives; no exponential differences.
#[derive(Clone, Copy)]
enum Entire {
    Sin,
    Cos,
    Sinh,
    Cosh,
}
impl Entire {
    /// Returns the Taylor coefficient `f⁽ᵒʳᵈᵉʳ⁾(center) / order!`.
    fn coefficient<R: RealScalar>(
        self,
        center: Complex<R>,
        order: usize,
    ) -> Result<Complex<R>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let mut value = match self {
            Self::Sin => match order % 4 {
                0 => center.sin(),
                1 => center.cos(),
                2 => -center.sin(),
                _ => -center.cos(),
            },
            Self::Cos => match order % 4 {
                0 => center.cos(),
                1 => -center.sin(),
                2 => -center.cos(),
                _ => center.sin(),
            },
            Self::Sinh if order.is_multiple_of(2) => center.sinh(),
            Self::Cosh if order % 2 == 1 => center.sinh(),
            _ => center.cosh(),
        };
        for k in 1..=order {
            value = crate::rotation::divide_real(value, R::try_from_usize_exact(k)?);
        }
        Ok(value)
    }

    /// Returns whether the remainder after `terms` Taylor terms about `center`
    /// is at most `threshold` in the one-norm.
    ///
    /// Write the block as `T_b = cI + D + N` with `D` diagonal and `N` strictly
    /// upper triangular, `μ = maxᵢ |dᵢ|`, `ν = ‖N‖₁` and `m` the block order
    /// (see [`BlockShape`]). By the divided-difference form of `f(T_b)`
    /// (Higham 2008, Thm. 4.11) and the Hermite–Genocchi formula, the
    /// remainder `g = f − Σ_{j<s} cⱼ (z − c)ʲ` satisfies
    ///
    /// ```text
    /// ‖g(T_b)‖₁ ≤ Σ_{k<m} νᵏ max_{|z−c|≤μ} |g⁽ᵏ⁾(z)| / k!
    ///          ≤ e^(γ+μ) Σ_{k<m} (νᵏ / k!) μ^((s−k)₊) / ((s−k)₊)!,
    /// ```
    ///
    /// because every derivative of `f` at `c` is bounded by `e^γ`, with
    /// `γ = |Im c|` for sin and cos and `γ = |Re c|` for sinh and cosh. Unlike
    /// a bound in `‖T_b − cI‖₁ ≤ μ + ν`, the coupling enters only to powers
    /// below `m`, as in Davies and Higham (2003, §2). The sum is formed in
    /// log space, so a bound beyond the floating-point range only means that
    /// the series has not converged yet. `log_factorial` is `ln s!`.
    fn remainder_within<R: RealScalar>(
        self,
        center: Complex<R>,
        shape: &BlockShape<R>,
        terms: usize,
        log_factorial: R,
        threshold: R,
    ) -> Result<bool, SciOxideError> {
        let BlockShape {
            spread,
            coupling,
            order,
        } = *shape;
        let growth = match self {
            Self::Sin | Self::Cos => center.im.abs(),
            Self::Sinh | Self::Cosh => center.re.abs(),
        };
        let (log_spread, log_coupling) = (spread.ln(), coupling.ln());
        // Log-sum-exp of the terms; `None` while the sum is zero.
        let mut log_sum: Option<R> = None;
        let mut log_k_factorial = R::ZERO;
        let mut log_q_factorial = log_factorial;
        for k in 0..order {
            if k > 0 {
                let index = R::try_from_usize_exact(k)?;
                log_k_factorial += index.ln();
                if k <= terms {
                    log_q_factorial -= R::try_from_usize_exact(terms - k + 1)?.ln();
                }
            }
            let q = terms.saturating_sub(k);
            if (k > 0 && coupling == R::ZERO) || (q > 0 && spread == R::ZERO) {
                continue;
            }
            let mut term = -log_k_factorial;
            if k > 0 {
                term += R::try_from_usize_exact(k)? * log_coupling;
            }
            if q > 0 {
                term += R::try_from_usize_exact(q)? * log_spread - log_q_factorial;
            }
            log_sum = Some(log_sum.map_or(term, |sum| {
                let (high, low) = if sum >= term {
                    (sum, term)
                } else {
                    (term, sum)
                };
                high + (low - high).exp().ln_1p()
            }));
        }
        Ok(match log_sum {
            None => true,
            Some(sum) => threshold > R::ZERO && growth + spread + sum <= threshold.ln(),
        })
    }
}

/// Distances of a Taylor block `T_b = cI + D + N` from its centre `c`.
#[derive(Clone, Copy)]
struct BlockShape<R: RealScalar> {
    /// `μ = maxᵢ |dᵢ|`, the largest distance of an eigenvalue from `c`.
    spread: R,
    /// `ν = ‖N‖₁`, the one-norm of the strictly upper triangular part.
    coupling: R,
    /// The block order `m`.
    order: usize,
}

/// Function summed on the Taylor blocks of the Schur–Parlett method.
#[derive(Clone, Copy)]
enum Series<'f, R: RealScalar> {
    /// A callback, with its own remainder bound in terms of `‖T_b − cI‖₁`.
    Callback(&'f dyn AnalyticMatrixFunction<R>),
    /// A built-in entire function, with [`Entire::remainder_within`].
    Entire(Entire),
}

pub(crate) struct Kernel<'w, R: RealScalar>
where
    Complex<R>: NumericScalar<Real = R>,
{
    data: &'w mut [Complex<R>],
    stride: usize,
    n: usize,
    rhs: &'w mut [ScaledValue<scioxide_core::Complex<R>>],
    blocks: &'w mut [usize],
    packed: &'w mut [Complex<R>],
}
impl<'w, R: RealScalar> Kernel<'w, R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Borrow validated slots for order `n`: `data` holds at least `SLOTS * n * n`
    /// values, `rhs` at least `n * n`, `blocks` at least `4 * n + 1`, and
    /// `packed` the shared dense product/solve packing requirement.
    pub(crate) fn new(
        data: &'w mut [Complex<R>],
        rhs: &'w mut [ScaledValue<scioxide_core::Complex<R>>],
        blocks: &'w mut [usize],
        packed: &'w mut [Complex<R>],
        n: usize,
    ) -> Self {
        Self {
            data,
            stride: n * n,
            n,
            rhs,
            blocks,
            packed,
        }
    }
    pub(crate) fn get(&self, s: usize, i: usize, j: usize) -> Complex<R> {
        self.data[s * self.stride + i * self.n + j]
    }
    pub(crate) fn set(&mut self, s: usize, i: usize, j: usize, v: Complex<R>) {
        self.data[s * self.stride + i * self.n + j] = v;
    }
    pub(crate) fn clear(&mut self, s: usize) {
        self.data[s * self.stride..(s + 1) * self.stride].fill(Complex::ZERO);
    }
    fn copy(&mut self, a: usize, b: usize) {
        self.data
            .copy_within(a * self.stride..(a + 1) * self.stride, b * self.stride);
    }
    fn identity(&mut self, s: usize) {
        self.clear(s);
        for i in 0..self.n {
            self.set(s, i, i, Complex::ONE);
        }
    }
    fn scale(&mut self, s: usize, v: Complex<R>) -> Result<(), SciOxideError> {
        for i in 0..self.n {
            for j in 0..self.n {
                self.set(
                    s,
                    i,
                    j,
                    crate::checked::finite_candidate_with_detail(self.get(s, i, j) * v, || {
                        "matrix function is not representable".into()
                    })?,
                );
            }
        }
        Ok(())
    }
    fn product(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        super::dense::Dense::new(self.data, self.blocks, self.packed, self.rhs, self.n)?
            .product(a, b, c, control)
    }
    // One shared quasi-triangular solve on validated workspace values.
    fn solve(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        super::dense::Dense::new(self.data, self.blocks, self.packed, self.rhs, self.n)?
            .quasi_solve(a, b, c, control)
    }
    fn exp(&mut self, control: &mut FunctionControl<'_, '_, '_, '_>) -> Result<(), SciOxideError> {
        super::dense::Dense::new(self.data, self.blocks, self.packed, self.rhs, self.n)?
            .exp(control)
    }
    fn log(
        &mut self,
        options: &MatrixFunctionOptions<R>,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        super::dense::Dense::new(self.data, self.blocks, self.packed, self.rhs, self.n)?
            .log(options, control)
    }
    fn sign(&mut self, control: &mut FunctionControl<'_, '_, '_, '_>) -> Result<(), SciOxideError> {
        // Parlett recurrence on the triangular Schur factor. From `F T = T F`,
        // `f_ij (t_ii - t_jj) = t_ij (s_i - s_j) + sum_k (f_ik t_kj - t_ik f_kj)`,
        // usable when the diagonal signs differ. When they agree, `t_ii - t_jj`
        // may be tiny, so `F^2 = I` is used instead:
        // `f_ij (s_i + s_j) = -sum_k f_ik f_kj` with `s_i + s_j = +-2`.
        self.clear(1);
        for i in 0..self.n {
            self.set(
                1,
                i,
                i,
                Complex::from_real(if self.get(0, i, i).re > R::ZERO {
                    R::ONE
                } else {
                    -R::ONE
                }),
            );
        }
        for d in 1..self.n {
            control.step()?;
            for i in 0..self.n - d {
                let j = i + d;
                let si = self.get(1, i, i);
                let sj = self.get(1, j, j);
                let mut sum = ScaledValue::ZERO;
                let divisor;
                if si == sj {
                    for k in i + 1..j {
                        sum = sum.sub(wide(self.get(1, i, k))?.mul(wide(self.get(1, k, j))?)?)?;
                    }
                    divisor = wide(si + sj)?;
                } else {
                    sum = wide(self.get(0, i, j))?.mul(wide(si - sj)?)?;
                    for k in i + 1..j {
                        sum = sum
                            .add(wide(self.get(1, i, k))?.mul(wide(self.get(0, k, j))?)?)?
                            .sub(wide(self.get(0, i, k))?.mul(wide(self.get(1, k, j))?)?)?;
                    }
                    divisor = wide(self.get(0, i, i))?.sub(wide(self.get(0, j, j))?)?;
                }
                self.set(1, i, j, sum.div(divisor)?.round()?);
            }
        }
        Ok(())
    }
    pub(crate) fn evaluate(
        &mut self,
        function: SchurFunction<'_, R>,
        options: &MatrixFunctionOptions<R>,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        match function {
            SchurFunction::Analytic(f) => {
                let (count, reordered) = self.cluster(control)?;
                self.custom(Series::Callback(f), count, options, control)?;
                self.restore_basis(reordered, control)
            }
            SchurFunction::Sign => {
                control.step()?;
                self.sign(control)
            }
            // Integer powers use binary exponentiation (inverting first for a
            // negative exponent) and are exact up to product rounding; other
            // real powers use `exp(p log T)` on the principal branch.
            SchurFunction::Power(p) => {
                control.step()?;
                let mut p = p.get();
                if p.floor() == p {
                    // An integer exponent beyond 2⁶² exceeds the precision of
                    // every supported scalar and is even: halve it exactly to
                    // `m` and square `T^m` once per halving.
                    let limit = R::try_from_f64_lossy(4_611_686_018_427_387_904.0)?;
                    let mut squarings = 0usize;
                    while p.abs() >= limit {
                        p *= R::HALF;
                        squarings += 1;
                    }
                    let power = p.try_to_i64_exact()?;
                    let mut exponent = power.unsigned_abs();
                    self.copy(0, 2);
                    self.identity(1);
                    if power < 0 {
                        self.solve(2, 1, 2, control)?;
                    }
                    while exponent > 0 {
                        if exponent % 2 == 1 {
                            self.product(1, 2, 1, control)?;
                        }
                        exponent /= 2;
                        if exponent > 0 {
                            self.product(2, 2, 2, control)?;
                        }
                    }
                    for _ in 0..squarings {
                        self.product(1, 1, 1, control)?;
                    }
                    return Ok(());
                }
                self.log(options, control)?;
                self.scale(1, Complex::from_real(p))?;
                self.copy(1, 0);
                self.exp(control)
            }
            SchurFunction::Trigonometric(f) => {
                control.step()?;
                let (count, reordered) = self.cluster(control)?;
                // Built-ins target scalar precision independently of the caller's
                // tolerance for callback tails and real-output acceptance.
                let options = options.with_function_settings(
                    options
                        .settings
                        .with_tolerance(scioxide_core::Tolerance::new(R::ZERO, R::EPSILON)?),
                );
                let numerator = match f {
                    Trigonometric::Sin | Trigonometric::Tan => Entire::Sin,
                    Trigonometric::Cos => Entire::Cos,
                    Trigonometric::Sinh | Trigonometric::Tanh => Entire::Sinh,
                    Trigonometric::Cosh => Entire::Cosh,
                };
                self.custom(Series::Entire(numerator), count, &options, control)?;
                if matches!(f, Trigonometric::Tan | Trigonometric::Tanh) {
                    self.copy(1, NUMERATOR);
                    let denominator = if matches!(f, Trigonometric::Tan) {
                        Entire::Cos
                    } else {
                        Entire::Cosh
                    };
                    self.custom(Series::Entire(denominator), count, &options, control)?;
                    // Solve before undoing the ordering: both factors are triangular.
                    self.solve(1, NUMERATOR, 1, control)?;
                }
                self.restore_basis(reordered, control)
            }
        }
    }

    /// Davies--Higham connected components at absolute separation delta = 0.1.
    /// Stable grouping keeps unrelated eigenvalues out of each Taylor block.
    fn cluster(
        &mut self,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(usize, bool), SciOxideError> {
        let delta = R::try_from_f64_lossy(0.1)?;
        for i in 0..self.n {
            self.blocks[i] = i;
        }
        for i in 0..self.n {
            control.step()?;
            for j in i + 1..self.n {
                if (self.get(0, i, i) - self.get(0, j, j)).norm() <= delta {
                    let old = self.blocks[j];
                    let new = self.blocks[i];
                    if old == new {
                        continue;
                    }
                    for label in &mut self.blocks[..self.n] {
                        if *label == old {
                            *label = new;
                        }
                    }
                }
            }
        }
        let mut count = 0;
        let mut start = 0;
        let mut reordered = false;
        let mut exponent = 0;
        self.blocks[self.n] = 0;
        while start < self.n {
            let label = self.blocks[start];
            let mut end = start + 1;
            for source in start + 1..self.n {
                if self.blocks[source] != label {
                    continue;
                }
                for position in (end..source).rev() {
                    control.step()?;
                    if !reordered {
                        self.identity(BASIS);
                        let maximum = self.data[..self.stride].iter().fold(R::ZERO, |m, &v| {
                            m.max(crate::kernel::standard::magnitude(v))
                        });
                        exponent = crate::kernel::standard::exponent(maximum)?;
                        for value in &mut self.data[..self.stride] {
                            *value = crate::kernel::standard::scale(*value, -exponent);
                        }
                        reordered = true;
                    }
                    let (data, basis) =
                        crate::workspace::split_mut(self.data, BASIS * self.stride)?;
                    let mut t = crate::workspace::storage::ScratchMatrix::new(
                        &mut data[..self.stride],
                        self.n,
                        self.n,
                    )?;
                    let mut q = crate::workspace::storage::ScratchMatrix::new(
                        &mut basis[..self.stride],
                        self.n,
                        self.n,
                    )?;
                    crate::eigen::swap_schur_blocks(&mut t, &mut q, position, 1, 1)?;
                    self.blocks.swap(position, position + 1);
                }
                end += 1;
            }
            count += 1;
            self.blocks[self.n + count] = end;
            start = end;
        }
        if reordered {
            for value in &mut self.data[..self.stride] {
                *value = crate::kernel::standard::restore(*value, exponent)?;
            }
        }
        self.blocks.copy_within(self.n..self.n + count + 1, 0);
        Ok((count, reordered))
    }

    fn restore_basis(
        &mut self,
        reordered: bool,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        if !reordered {
            return Ok(());
        }
        use crate::kernel::GemmOp;
        use crate::{MatrixView, MatrixViewMut};
        let (work, basis) = crate::workspace::split_mut(self.data, BASIS * self.stride)?;
        let (values, stage) = crate::workspace::split_mut(work, STAGING * self.stride)?;
        let result = &mut values[self.stride..2 * self.stride];
        let stage = &mut stage[..self.stride];
        let basis = MatrixView::from_slice(self.n, self.n, &basis[..self.stride])?;
        super::workspace_product(
            basis,
            MatrixView::from_slice(self.n, self.n, result)?,
            MatrixViewMut::from_slice(self.n, self.n, stage)?,
            GemmOp::None,
            GemmOp::None,
            self.packed,
            control,
        )?;
        super::workspace_product(
            MatrixView::from_slice(self.n, self.n, stage)?,
            basis,
            MatrixViewMut::from_slice(self.n, self.n, result)?,
            GemmOp::None,
            GemmOp::Adjoint,
            self.packed,
            control,
        )
    }

    fn custom(
        &mut self,
        f: Series<'_, R>,
        count: usize,
        options: &MatrixFunctionOptions<R>,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        self.clear(1);
        for block in 0..count {
            self.taylor_block(
                f,
                self.blocks[block],
                self.blocks[block + 1],
                options,
                control,
            )?;
        }
        for distance in 1..count {
            for row_block in 0..count - distance {
                let col_block = row_block + distance;
                let r = self.blocks[row_block];
                let re = self.blocks[row_block + 1];
                let c = self.blocks[col_block];
                let ce = self.blocks[col_block + 1];
                let width = ce - c;
                for i in r..re {
                    for j in c..ce {
                        let mut sum = ScaledValue::ZERO;
                        for k in r..re {
                            sum =
                                sum.add(wide(self.get(1, i, k))?.mul(wide(self.get(0, k, j))?)?)?;
                        }
                        for k in c..ce {
                            sum =
                                sum.sub(wide(self.get(0, i, k))?.mul(wide(self.get(1, k, j))?)?)?;
                        }
                        for k in re..c {
                            sum = sum
                                .add(wide(self.get(1, i, k))?.mul(wide(self.get(0, k, j))?)?)?
                                .sub(wide(self.get(0, i, k))?.mul(wide(self.get(1, k, j))?)?)?;
                        }
                        self.rhs[(i - r) * width + j - c] = sum;
                    }
                }
                let view =
                    crate::MatrixView::from_slice(self.n, self.n, &self.data[..self.n * self.n])?;
                if !crate::kernel::sylvester::pair(
                    view.block(r, r, re - r, re - r)?,
                    view.block(c, c, width, width)?,
                    -Complex::ONE,
                    &mut self.rhs[..(re - r) * width],
                    false,
                    || control.step(),
                )? {
                    return Err(domain(
                        scioxide_core::argument!("function"),
                        "custom function has inseparable Schur blocks",
                    ));
                }
                for i in r..re {
                    for j in c..ce {
                        self.set(1, i, j, self.rhs[(i - r) * width + j - c].round()?);
                    }
                }
            }
        }
        Ok(())
    }
    fn taylor_block(
        &mut self,
        f: Series<'_, R>,
        start: usize,
        end: usize,
        options: &MatrixFunctionOptions<R>,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        // `f(T_block) = sum_k f^(k)(c)/k! (T - cI)^k` around the mean diagonal
        // entry c. If all diagonal entries equal c, `T - cI` is strictly upper
        // triangular (nilpotent) and the series is exact after `block size`
        // terms; otherwise the series stops when a bound on its remainder is
        // below tolerance relative to the current result. A callback must
        // also contain the block in its disk of analyticity.
        let first = self.get(0, start, start);
        let nilpotent = (start..end).all(|i| self.get(0, i, i) == first);
        let center = if nilpotent {
            first
        } else {
            let mut sum = ScaledValue::ZERO;
            for i in start..end {
                sum = sum.add(wide(self.get(0, i, i))?)?;
            }
            sum.div(wide(Complex::from_real(R::try_from_usize_exact(
                end - start,
            )?))?)?
            .round()?
        };
        // One-norms of `T - cI` and of its strictly upper part, and the
        // largest distance of an eigenvalue from the centre.
        let (mut norm, mut coupling, mut spread) = (R::ZERO, R::ZERO, R::ZERO);
        for j in start..end {
            let (mut sum, mut upper) = (R::ZERO, R::ZERO);
            for i in start..end {
                let value = self.get(0, i, j) - if i == j { center } else { Complex::ZERO };
                self.set(2, i, j, value);
                sum += value.norm();
                if i < j {
                    upper += value.norm();
                } else if i == j {
                    spread = spread.max(value.norm());
                }
                self.set(3, i, j, if i == j { Complex::ONE } else { Complex::ZERO });
            }
            norm = norm.max(sum);
            coupling = coupling.max(upper);
        }
        if let (false, Series::Callback(f)) = (nilpotent, f) {
            let radius = f.radius(center);
            if !radius.is_finite() || radius <= norm {
                return Err(domain(
                    scioxide_core::argument!("function"),
                    "clustered Schur block exceeds the callback analytic disk",
                ));
            }
        }
        let mut log_factorial = R::ZERO;
        for k in 0..=options.settings.series_limit.get() {
            control.step()?;
            let coefficient = match f {
                Series::Callback(f) => f.coefficient(center, k)?,
                Series::Entire(f) => f.coefficient(center, k)?,
            };
            let coefficient = crate::checked::finite_candidate_with_detail(coefficient, || {
                "matrix function is not representable".into()
            })?;
            for i in start..end {
                for j in i..end {
                    let value = wide(self.get(1, i, j))?
                        .add(wide(self.get(3, i, j))?.mul(wide(coefficient)?)?)?
                        .round()?;
                    self.set(1, i, j, value);
                }
            }
            if nilpotent && k + 1 >= end - start {
                return Ok(());
            }
            log_factorial += R::try_from_usize_exact(k + 1)?.ln();
            if !nilpotent {
                let output_norm = (start..end)
                    .map(|j| (start..end).fold(R::ZERO, |sum, i| sum + self.get(1, i, j).norm()))
                    .fold(R::ZERO, R::max);
                let threshold = options.settings.tolerance.threshold(output_norm)?;
                let converged = match f {
                    Series::Callback(f) => {
                        let tail = f.tail_bound(center, k, norm)?;
                        if !tail.is_finite() || tail < R::ZERO {
                            return Err(domain(
                                scioxide_core::argument!("function"),
                                "invalid analytic Taylor tail estimate",
                            ));
                        }
                        tail <= threshold
                    }
                    Series::Entire(f) => f.remainder_within(
                        center,
                        &BlockShape {
                            spread,
                            coupling,
                            order: end - start,
                        },
                        k + 1,
                        log_factorial,
                        threshold,
                    )?,
                };
                if converged {
                    return Ok(());
                }
            }
            let (inputs, stage) = crate::workspace::split_mut(self.data, STAGING * self.stride)?;
            let shifted = crate::MatrixView::from_slice(
                self.n,
                self.n,
                &inputs[2 * self.stride..3 * self.stride],
            )?
            .block(start, start, end - start, end - start)?;
            let power = crate::MatrixView::from_slice(
                self.n,
                self.n,
                &inputs[3 * self.stride..4 * self.stride],
            )?
            .block(start, start, end - start, end - start)?;
            let mut stage =
                crate::MatrixViewMut::from_slice(self.n, self.n, &mut stage[..self.stride])?;
            super::workspace_product(
                power,
                shifted,
                stage.block_mut(start, start, end - start, end - start)?,
                crate::kernel::GemmOp::None,
                crate::kernel::GemmOp::None,
                self.packed,
                control,
            )?;
            for i in start..end {
                for j in i..end {
                    self.set(3, i, j, self.get(STAGING, i, j));
                }
            }
        }
        Err(super::limit(
            "clustered analytic Taylor series exhausted",
            control.counts(),
        ))
    }
}
fn wide<R: RealScalar>(
    z: Complex<R>,
) -> Result<ScaledValue<scioxide_core::Complex<R>>, SciOxideError> {
    ScaledValue::from_value(z)
}

pub(crate) fn domain(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::domain_error(argument, DomainReason::OutsideDomain, detail)
}
