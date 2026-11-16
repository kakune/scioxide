//! Dense Padé exponential and blocked (quasi-)triangular principal functions.
//!
//! Ordinary workspace arithmetic uses the shared Standard binary frame. Products
//! and solves write only finite results; no checked arithmetic runs in a
//! cubic element loop. Algorithms: Al-Mohy--Higham (2009, exp; 2012, log),
//! Higham (1987, real square root), Deadman--Higham--Ralha (2013, blocking).
use super::{FunctionControl, MatrixFunctionOptions};
use crate::kernel::GemmConfig;
use crate::kernel::frames::{extended, scalar};
use crate::kernel::small::Singularity;
use crate::kernel::standard;
use crate::kernel::sylvester;
use crate::workspace::Extended;
use crate::{
    DenseWorkspaceRequest, FiniteMatrixView, LinalgExecutionOptions, LinalgRequirements,
    LinalgSlices, LinalgWorkspace, MatrixView, MatrixViewMut,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Complex, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};

pub(super) const SLOTS: usize = 13;
const STAGE: usize = 8;
const TILE: usize = 32;

pub(super) fn packing<S: NumericScalar>(n: usize) -> Result<LinalgRequirements, SciOxideError> {
    let options = LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
    let product = DenseWorkspaceRequest::product(n, n, n, &options)
        .with_worker_bound(core::num::NonZeroUsize::MIN)
        .with_provider_packing();
    let solve = DenseWorkspaceRequest::triangular_solve(n, n, n, &options)
        .with_worker_bound(core::num::NonZeroUsize::MIN)
        .with_provider_packing();
    Ok(crate::kernel::product_requirements::<S>(&product)?
        .max(&crate::kernel::product_requirements::<S>(&solve)?))
}

pub(super) struct Dense<'a, S: NumericScalar> {
    data: &'a mut [S],
    rhs: &'a mut [Extended<S>],
    indices: &'a mut [usize],
    gemm: LinalgWorkspace<'a, S>,
    n: usize,
}
impl<'a, S: NumericScalar> Dense<'a, S> {
    pub(super) fn new(
        data: &'a mut [S],
        indices: &'a mut [usize],
        packed: &'a mut [S],
        rhs: &'a mut [Extended<S>],
        n: usize,
    ) -> Result<Self, SciOxideError> {
        let requirements = packing::<S>(n)?;
        let gemm = LinalgWorkspace::from_slices(
            LinalgSlices {
                scalars: packed,
                ..LinalgSlices::default()
            },
            &requirements,
        )?;
        Ok(Self {
            data,
            rhs,
            indices,
            gemm,
            n,
        })
    }
    fn at(&self, slot: usize, i: usize, j: usize) -> S {
        self.data[slot * self.n * self.n + i * self.n + j]
    }
    fn put(&mut self, slot: usize, i: usize, j: usize, value: S) {
        self.data[slot * self.n * self.n + i * self.n + j] = value;
    }
    fn copy(&mut self, a: usize, b: usize) {
        let square = self.n * self.n;
        self.data
            .copy_within(a * square..(a + 1) * square, b * square);
    }
    fn clear(&mut self, slot: usize) {
        let square = self.n * self.n;
        self.data[slot * square..(slot + 1) * square].fill(S::ZERO);
    }
    fn identity(&mut self, slot: usize) {
        self.clear(slot);
        for i in 0..self.n {
            self.put(slot, i, i, S::ONE);
        }
    }
    fn check(&self, slot: usize) -> Result<(), SciOxideError> {
        let square = self.n * self.n;
        for &value in &self.data[slot * square..(slot + 1) * square] {
            crate::checked::finite_candidate(value)?;
        }
        Ok(())
    }
    fn norm(&self, slot: usize) -> S::Real {
        (0..self.n)
            .map(|j| (0..self.n).fold(S::Real::ZERO, |sum, i| sum + self.at(slot, i, j).abs()))
            .fold(S::Real::ZERO, S::Real::max)
    }
    fn scale(&mut self, slot: usize, exponent: i32) -> Result<(), SciOxideError> {
        let square = self.n * self.n;
        for value in &mut self.data[slot * square..(slot + 1) * square] {
            *value = standard::scale(*value, exponent);
        }
        self.check(slot)
    }
    fn combine(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        alpha: S::Real,
        beta: S::Real,
    ) -> Result<(), SciOxideError> {
        let maximum = |slot| {
            (0..self.n * self.n).fold(S::Real::ZERO, |m, i| {
                m.max(standard::magnitude(self.data[slot * self.n * self.n + i]))
            })
        };
        let frame = standard::Affine::new(
            S::from_real(alpha),
            maximum(a),
            S::from_real(beta),
            maximum(b),
        )?;
        if S::IS_REAL && alpha == S::Real::ONE && beta != S::Real::ZERO && frame.is_unscaled() {
            // fma(1, x, beta*y) is exactly x + beta*y for finite real
            // operands. Hoist the range decision and avoid a scalar FMA call
            // per entry; the rounded beta*y and addition order are unchanged.
            let beta = S::from_real(beta);
            let square = self.n * self.n;
            for i in 0..square {
                self.data[c * square + i] =
                    self.data[a * square + i] + beta * self.data[b * square + i];
            }
        } else {
            for i in 0..self.n * self.n {
                self.data[c * self.n * self.n + i] = frame.apply(
                    self.data[a * self.n * self.n + i],
                    self.data[b * self.n * self.n + i],
                );
            }
        }
        self.check(c)
    }
    pub(super) fn product(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        control.step()?;
        let square = self.n * self.n;
        let (inputs, rest) = crate::workspace::split_mut(self.data, STAGE * square)?;
        let left = FiniteMatrixView::from_workspace(MatrixView::from_slice(
            self.n,
            self.n,
            &inputs[a * square..(a + 1) * square],
        )?);
        let right = FiniteMatrixView::from_workspace(MatrixView::from_slice(
            self.n,
            self.n,
            &inputs[b * square..(b + 1) * square],
        )?);
        crate::kernel::gemm_workspace_with_context(
            crate::kernel::WorkspaceGemmProblem {
                left,
                right,
                output: MatrixViewMut::from_slice(self.n, self.n, &mut rest[..square])?,
                config: GemmConfig::ordinary(),
            },
            ExecutionPolicy::Sequential,
            &mut self.gemm,
            control.kernels(),
        )?;
        self.copy(STAGE, c);
        Ok(())
    }
    fn solve(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        control.step()?;
        let square = self.n * self.n;
        let (inputs, scratch) = crate::workspace::split_mut(self.data, 9 * square)?;
        let (inputs, candidate) = crate::workspace::split_mut(inputs, STAGE * square)?;
        let factor = crate::decompose::factor_kernel(
            MatrixView::from_slice(self.n, self.n, &inputs[a * square..(a + 1) * square])?,
            scratch,
            self.indices,
            &mut self.gemm,
            ExecutionPolicy::Sequential,
            control.kernels(),
        )?;
        factor.solve_matrix(
            MatrixView::from_slice(self.n, self.n, &inputs[b * square..(b + 1) * square])?,
            candidate,
            &mut self.gemm,
            control.kernels(),
        )?;
        self.copy(STAGE, c);
        Ok(())
    }
    /// Al-Mohy--Higham Algorithm 6.1, with exact small power norms rather than
    /// estimates. The ell safeguard uses scaled positive matrix-vector products.
    pub(super) fn exp(
        &mut self,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        if self.n == 0 {
            return Ok(());
        }
        let maximum = self.data[..self.n * self.n]
            .iter()
            .fold(S::Real::ZERO, |m, &v| m.max(standard::magnitude(v)));
        // Bound tenth powers before probing their norms. Every entry and partial
        // sum of `Aᵏ`, `k ≤ 10`, is at most `‖A‖₁ᵏ ≤ (n max|aᵢⱼ|)ᵏ`, so the
        // largest entry may use a tenth of the exponent range less `⌈log₂ n⌉`.
        // Ordinary matrices stay in their original frame; extreme matrices use
        // exact binary scaling.
        let order = self.n.max(1);
        let log_order = (order.ilog2() + u32::from(!order.is_power_of_two())) as i32;
        let safe = (standard::exponent(S::Real::MAX)? - 8) / 10 - log_order;
        let initial = (standard::exponent(maximum)? - safe).max(0);
        self.scale(0, -initial)?;
        self.product(0, 0, 2, control)?;
        self.product(2, 2, 3, control)?;
        self.product(3, 2, 4, control)?;
        let d4 = self.norm(3).sqrt().sqrt();
        let d6 = self.norm(4).powf(real::<S>(1.0 / 6.0)?);
        let mut degree = 13;
        let mut scaling = 0i32;
        if d4.max(d6) <= real::<S>(1.495585217958292e-2)? && self.ell(3, control)? == 0 {
            degree = 3;
        } else if d4.max(d6) <= real::<S>(2.539_398_330_063_23e-1)? && self.ell(5, control)? == 0 {
            degree = 5;
        } else {
            self.product(3, 3, 5, control)?;
            let d8 = self.norm(5).sqrt().sqrt().sqrt();
            if d6.max(d8) <= real::<S>(9.504178996162932e-1)? && self.ell(7, control)? == 0 {
                degree = 7;
            } else if d6.max(d8) <= real::<S>(2.097847961257068)? && self.ell(9, control)? == 0 {
                degree = 9;
            } else {
                self.product(3, 4, 1, control)?;
                let d10 = self.norm(1).powf(real::<S>(0.1)?);
                let eta = d6.max(d8).min(d8.max(d10));
                let theta = real::<S>(4.25)?;
                if eta > theta {
                    scaling = ((eta / theta).ln() / real::<S>(core::f64::consts::LN_2)?)
                        .ceil()
                        .try_to_i64_exact()? as i32;
                }
                self.scale(0, -scaling)?;
                let extra = self.ell(13, control)?;
                self.scale(0, -extra)?;
                scaling += extra;
                self.scale(2, -2 * scaling)?;
                self.scale(3, -4 * scaling)?;
                self.scale(4, -6 * scaling)?;
            }
        }
        // Normalized coefficients keep f32 intermediates small. The f64
        // backward-error thresholds are conservative for f32 as well.
        let mut coefficients = [S::Real::ZERO; 14];
        coefficients[0] = S::Real::ONE;
        for k in 1..=degree {
            coefficients[k] = coefficients[k - 1]
                * crate::checked::rounded_count::<S::Real>(degree - k + 1)?
                / (crate::checked::rounded_count::<S::Real>(k)?
                    * crate::checked::rounded_count::<S::Real>(2 * degree - k + 1)?);
        }
        if degree == 13 {
            self.polynomial(
                5,
                &[
                    (4, coefficients[13]),
                    (3, coefficients[11]),
                    (2, coefficients[9]),
                ],
                S::Real::ZERO,
            )?;
            self.product(4, 5, 5, control)?;
            self.add_polynomial(
                5,
                &[
                    (4, coefficients[7]),
                    (3, coefficients[5]),
                    (2, coefficients[3]),
                ],
                coefficients[1],
            )?;
            self.product(0, 5, 5, control)?;
            self.polynomial(
                1,
                &[
                    (4, coefficients[12]),
                    (3, coefficients[10]),
                    (2, coefficients[8]),
                ],
                S::Real::ZERO,
            )?;
            self.product(4, 1, 1, control)?;
            self.add_polynomial(
                1,
                &[
                    (4, coefficients[6]),
                    (3, coefficients[4]),
                    (2, coefficients[2]),
                ],
                coefficients[0],
            )?;
        } else {
            // A2, A4, A6 and A8 are already available. Accumulate the two
            // polynomials without destroying any power needed by the other.
            self.identity(1);
            self.identity(6);
            for i in 0..self.n {
                self.put(6, i, i, S::from_real(coefficients[1]));
            }
            for k in 1..=degree / 2 {
                let power = k + 1;
                self.combine(1, power, 1, S::Real::ONE, coefficients[2 * k])?;
                self.combine(6, power, 6, S::Real::ONE, coefficients[2 * k + 1])?;
            }
            self.product(0, 6, 5, control)?;
        }
        self.combine(1, 5, 2, S::Real::ONE, S::Real::ONE)?;
        self.combine(1, 5, 3, S::Real::ONE, -S::Real::ONE)?;
        self.solve(3, 2, 1, control)?;
        for _ in 0..initial + scaling {
            self.product(1, 1, 1, control)?;
        }
        Ok(())
    }
    fn polynomial(
        &mut self,
        slot: usize,
        terms: &[(usize, S::Real)],
        constant: S::Real,
    ) -> Result<(), SciOxideError> {
        self.clear(slot);
        self.add_polynomial(slot, terms, constant)
    }
    fn add_polynomial(
        &mut self,
        slot: usize,
        terms: &[(usize, S::Real)],
        constant: S::Real,
    ) -> Result<(), SciOxideError> {
        for &(source, weight) in terms {
            self.combine(slot, source, slot, S::Real::ONE, weight)?;
        }
        for i in 0..self.n {
            self.put(slot, i, i, self.at(slot, i, i) + S::from_real(constant));
        }
        self.check(slot)
    }
    fn ell(
        &mut self,
        degree: usize,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<i32, SciOxideError> {
        let reciprocal = match degree {
            3 => 100800.0,
            5 => 10059033600.0,
            7 => 4487938430976000.0,
            9 => 5914384781877411840000.0,
            _ => 1.1325077560602111e35,
        };
        let norm = self.norm(0);
        if norm == S::Real::ZERO {
            return Ok(0);
        }
        let exponent = standard::exponent(norm)?;
        // Normalize and transpose |A| once. Repeated positive matvecs then
        // use contiguous ordinary arithmetic, with only vector-level rescaling.
        // The matrix and two vectors reuse the later LU scratch.
        let absolute = 9 * self.n * self.n;
        for j in 0..self.n {
            for i in 0..self.n {
                self.data[absolute + j * self.n + i] =
                    S::from_real(standard::scale(self.at(0, i, j), -exponent).abs());
            }
        }
        let start = 10 * self.n * self.n;
        self.data[start..start + self.n].fill(S::ONE);
        let mut accumulated = 0i32;
        for _ in 0..2 * degree + 1 {
            control.step()?;
            let mut maximum = S::Real::ZERO;
            for j in 0..self.n {
                let sum = (0..self.n).fold(S::Real::ZERO, |v, i| {
                    self.data[absolute + j * self.n + i].real_component()
                        * self.data[start + i].real_component()
                        + v
                });
                self.data[start + self.n + j] = S::from_real(sum);
                maximum = maximum.max(sum);
            }
            if maximum == S::Real::ZERO {
                return Ok(0);
            }
            let shift = standard::exponent(maximum)?;
            for j in 0..self.n {
                self.data[start + j] = standard::scale(self.data[start + self.n + j], -shift);
            }
            accumulated += exponent + shift;
        }
        let maximum = self.data[start..start + self.n]
            .iter()
            .fold(S::Real::ZERO, |m, &v| m.max(v.real_component()));
        let ln2 = real::<S>(core::f64::consts::LN_2)?;
        let log = maximum.ln() / ln2 + S::Real::try_from_i64_exact(i64::from(accumulated))?
            - norm.ln() / ln2
            - real::<S>(reciprocal)?.ln() / ln2
            - (S::Real::EPSILON * S::Real::HALF).ln() / ln2;
        let value = (log / crate::checked::rounded_count::<S::Real>(2 * degree)?)
            .ceil()
            .max(S::Real::ZERO);
        Ok(value.try_to_i64_exact()? as i32)
    }
    /// Blocked Schur square root. Panels never split a real conjugate pair.
    pub(super) fn sqrt(
        &mut self,
        input: usize,
        output: usize,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        self.copy(input, output);
        let maximum = (0..self.n * self.n).fold(S::Real::ZERO, |m, i| {
            m.max(standard::magnitude(self.data[output * self.n * self.n + i]))
        });
        let exponent = standard::exponent(maximum)?;
        let mut frame = if exponent.abs() > crate::kernel::frames::range_frame_bound::<S::Real>()? {
            exponent.div_euclid(2) * 2
        } else {
            0
        };
        // Do not flush a small Schur entry merely because another block is
        // large. The unit frame preserves graded spectra; local solves and
        // Standard updates still normalize their own operand ranges.
        if self.data[output * self.n * self.n..(output + 1) * self.n * self.n]
            .iter()
            .any(|&value| {
                (value.real_component() != S::Real::ZERO
                    && value.real_component().scalbn(-frame) == S::Real::ZERO)
                    || (value.imaginary_component() != S::Real::ZERO
                        && value.imaginary_component().scalbn(-frame) == S::Real::ZERO)
            })
        {
            frame = 0;
        }
        self.scale(output, -frame)?;
        let mut count = 0;
        let mut end = 0;
        self.indices[0] = 0;
        while end < self.n {
            let mut next = (end + TILE).min(self.n);
            if next < self.n && self.at(output, next, next - 1) != S::ZERO {
                next += 1;
            }
            count += 1;
            self.indices[count] = next;
            end = next;
        }
        for tile in 0..count {
            control.step()?;
            let start = self.indices[tile];
            let end = self.indices[tile + 1];
            let mut i = start;
            while i < end {
                let begin = i;
                if i + 1 < end && self.at(output, i + 1, i) != S::ZERO {
                    let root = pair_root(
                        self.at(output, i, i),
                        self.at(output, i, i + 1),
                        self.at(output, i + 1, i),
                    )?;
                    self.put(output, i, i, root[0]);
                    self.put(output, i, i + 1, root[1]);
                    self.put(output, i + 1, i, root[2]);
                    self.put(output, i + 1, i + 1, root[3]);
                    i += 2;
                } else {
                    self.put(output, i, i, scalar_sqrt(self.at(output, i, i))?);
                    i += 1;
                }
                if begin > start {
                    self.sylvester(output, start, begin, begin, i)?;
                }
            }
        }
        for distance in 1..count {
            for tile in 0..count - distance {
                control.step()?;
                let r = self.indices[tile];
                let re = self.indices[tile + 1];
                let c = self.indices[tile + distance];
                let ce = self.indices[tile + distance + 1];
                if re < c {
                    let square = self.n * self.n;
                    let (values, stage) = crate::workspace::split_mut(self.data, STAGE * square)?;
                    let u = MatrixView::from_slice(
                        self.n,
                        self.n,
                        &values[output * square..(output + 1) * square],
                    )?;
                    let mut rhs_matrix =
                        MatrixViewMut::from_slice(self.n, self.n, &mut stage[..square])?;
                    let rhs = rhs_matrix.block_mut(r, c, re - r, ce - c)?;
                    crate::kernel::gemm_workspace_with_context(
                        crate::kernel::WorkspaceGemmProblem {
                            left: FiniteMatrixView::from_workspace(u.block(
                                r,
                                re,
                                re - r,
                                c - re,
                            )?),
                            right: FiniteMatrixView::from_workspace(u.block(
                                re,
                                c,
                                c - re,
                                ce - c,
                            )?),
                            output: rhs,
                            config: GemmConfig::ordinary(),
                        },
                        ExecutionPolicy::Sequential,
                        &mut self.gemm,
                        control.kernels(),
                    )?;
                    for i in r..re {
                        for j in c..ce {
                            self.put(output, i, j, self.at(output, i, j) - self.at(STAGE, i, j));
                        }
                    }
                }
                self.sylvester(output, r, re, c, ce)?;
            }
        }
        self.scale(output, frame / 2)
    }
    /// Solve one root panel through the shared oriented pair kernel.
    fn sylvester(
        &mut self,
        slot: usize,
        r: usize,
        re: usize,
        c: usize,
        ce: usize,
    ) -> Result<(), SciOxideError> {
        let (rows, cols) = (re - r, ce - c);
        for i in 0..rows {
            for j in 0..cols {
                self.rhs[i * cols + j] = extended(self.at(slot, r + i, c + j))?;
            }
        }
        let square = self.n * self.n;
        let (inputs, stage) = crate::workspace::split_mut(self.data, STAGE * square)?;
        let root =
            MatrixView::from_slice(self.n, self.n, &inputs[slot * square..(slot + 1) * square])?;
        if !sylvester::pair_oriented(
            sylvester::Recurrence {
                a: root.block(r, r, rows, rows)?,
                b: root.block(c, c, cols, cols)?,
                sign: S::ONE,
                left_adjoint: false,
                right_adjoint: false,
                discrete: false,
            },
            &mut self.rhs[..rows * cols],
            &mut stage[..rows * cols],
            Singularity::Consistent,
            || Ok(()),
        )? {
            return Err(super::kernel::domain(
                scioxide_core::argument!("self"),
                "square root has a singular block recurrence",
            ));
        }
        for i in 0..rows {
            for j in 0..cols {
                self.put(slot, r + i, c + j, scalar(self.rhs[i * cols + j])?);
            }
        }
        Ok(())
    }
    /// Inverse scaling and squaring with Gauss--Legendre evaluation of the
    /// [m/m] Padé approximant to log(1+x). Eight nodes at ||x|| <= 1/4
    /// bound the rational truncation below f64 unit roundoff; four serve f32.
    pub(super) fn log(
        &mut self,
        options: &MatrixFunctionOptions<S::Real>,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        self.copy(0, 2);
        let mut roots = 0;
        loop {
            self.copy(2, 3);
            for i in 0..self.n {
                self.put(3, i, i, self.at(3, i, i) - S::ONE);
            }
            if self.norm(3) <= real::<S>(0.25)? {
                break;
            }
            control.step()?;
            self.sqrt(2, 2, control)?;
            roots += 1;
        }
        // Stable Briggs correction for scalar and real 2x2 diagonal blocks,
        // and exact root divided differences on adjacent scalar blocks.
        let mut i = 0;
        while i < self.n {
            if i + 1 < self.n && self.at(0, i + 1, i) != S::ZERO {
                let mut original = [
                    self.at(0, i, i),
                    self.at(0, i, i + 1),
                    self.at(0, i + 1, i),
                    self.at(0, i + 1, i + 1),
                ];
                let mut delta = original;
                delta[0] = delta[0] - S::ONE;
                delta[3] = delta[3] - S::ONE;
                for _ in 0..roots {
                    original = pair_root(original[0], original[1], original[2])?;
                    for col in 0..2 {
                        let mut a = [S::ZERO; 16];
                        let mut b = [S::ZERO; 4];
                        a[0] = original[0] + S::ONE;
                        a[1] = original[1];
                        a[4] = original[2];
                        a[5] = original[3] + S::ONE;
                        b[0] = delta[col];
                        b[1] = delta[2 + col];
                        solve_small(&a, &mut b, 2)?;
                        delta[col] = b[0];
                        delta[2 + col] = b[1];
                    }
                }
                self.put(3, i, i, delta[0]);
                self.put(3, i, i + 1, delta[1]);
                self.put(3, i + 1, i, delta[2]);
                self.put(3, i + 1, i + 1, delta[3]);
                i += 2;
            } else {
                let mut z = self.at(0, i, i);
                let mut delta = z - S::ONE;
                for _ in 0..roots {
                    z = scalar_sqrt(z)?;
                    delta = delta / (z + S::ONE);
                }
                self.put(3, i, i, delta);
                if i + 1 < self.n
                    && self.at(0, i + 1, i) == S::ZERO
                    && (i + 2 == self.n || self.at(0, i + 2, i + 1) == S::ZERO)
                {
                    let mut a = self.at(0, i, i);
                    let mut b = self.at(0, i + 1, i + 1);
                    let mut value = self.at(0, i, i + 1);
                    for _ in 0..roots {
                        a = scalar_sqrt(a)?;
                        b = scalar_sqrt(b)?;
                        value = value / (a + b);
                    }
                    self.put(3, i, i + 1, value);
                }
                i += 1;
            }
        }
        let degree = if S::Real::MANTISSA_DIGITS <= 24 { 4 } else { 8 };
        if options.series_limit().get() < degree {
            return Err(super::limit(
                "logarithm approximation terms exhausted",
                control.counts(),
            ));
        }
        self.clear(1);
        for node in 0..degree {
            let (point, weight) = legendre::<S>(degree, node)?;
            self.copy(3, 4);
            for i in 0..self.n * self.n {
                self.data[4 * self.n * self.n + i] =
                    self.data[4 * self.n * self.n + i] * S::from_real(point);
            }
            for i in 0..self.n {
                self.put(4, i, i, self.at(4, i, i) + S::ONE);
            }
            self.quasi_solve(4, 3, 5, control)?;
            self.combine(1, 5, 1, S::Real::ONE, weight)?;
        }
        self.scale(1, roots)?;
        let mut i = 0;
        while i < self.n {
            if i + 1 < self.n && self.at(0, i + 1, i) != S::ZERO {
                let a = self.at(0, i, i).real_component();
                let b = self.at(0, i, i + 1).real_component();
                let c = self.at(0, i + 1, i).real_component();
                let omega = b.abs().sqrt() * c.abs().sqrt();
                let z = Complex::new(a, omega);
                let logarithm = z.ln();
                self.put(1, i, i, S::from_real(logarithm.re));
                self.put(1, i + 1, i + 1, S::from_real(logarithm.re));
                self.put(1, i, i + 1, S::from_real((b / omega) * logarithm.im));
                self.put(1, i + 1, i, S::from_real((c / omega) * logarithm.im));
                i += 2;
            } else {
                self.put(1, i, i, scalar_log(self.at(0, i, i)));
                if i + 1 < self.n
                    && self.at(0, i + 1, i) == S::ZERO
                    && (i + 2 == self.n || self.at(0, i + 2, i + 1) == S::ZERO)
                {
                    let value = log_divided(
                        self.at(0, i, i),
                        self.at(0, i + 1, i + 1),
                        self.at(0, i, i + 1),
                    )?;
                    self.put(1, i, i + 1, value);
                }
                i += 1;
            }
        }
        self.check(1)
    }
    pub(super) fn quasi_solve(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        control: &mut FunctionControl<'_, '_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        control.step()?;
        self.copy(b, STAGE);
        let square = self.n * self.n;
        let (inputs, stage) = crate::workspace::split_mut(self.data, STAGE * square)?;
        let triangular = FiniteMatrixView::from_workspace(MatrixView::from_slice(
            self.n,
            self.n,
            &inputs[a * square..(a + 1) * square],
        )?);
        crate::kernel::level3::schur_trsm_bounded(
            triangular,
            MatrixViewMut::from_slice(self.n, self.n, &mut stage[..square])?,
            &mut self.gemm,
            control.kernels(),
        )?;
        self.copy(STAGE, c);
        self.check(c)
    }
}
fn real<S: NumericScalar>(value: f64) -> Result<S::Real, SciOxideError> {
    S::Real::try_from_f64_lossy(value)
}
/// Principal square root. A zero imaginary part is taken as `+0`, so a
/// negative real value `x` maps to `i √|x|` whatever the sign of its zero
/// imaginary part (IEEE `csqrt(x − 0i) = −i √|x|`).
fn scalar_sqrt<S: NumericScalar>(value: S) -> Result<S, SciOxideError> {
    let mut z = crate::eigen::into_complex(value);
    if z.im == S::Real::ZERO {
        z.im = S::Real::ZERO;
    }
    let z = z.sqrt();
    crate::checked::finite_candidate(S::from_components(z.re, z.im))
}
fn scalar_log<S: NumericScalar>(value: S) -> S {
    if S::IS_REAL {
        S::from_real(value.real_component().ln())
    } else {
        let z = crate::eigen::into_complex(value).ln();
        S::from_components(z.re, z.im)
    }
}
fn pair_root<S: NumericScalar>(a: S, b: S, c: S) -> Result<[S; 4], SciOxideError> {
    let omega = b.real_component().abs().sqrt() * c.real_component().abs().sqrt();
    let root = Complex::new(a.real_component(), omega).sqrt();
    let divisor = root.re * S::Real::TWO;
    if divisor == S::Real::ZERO {
        return Err(super::kernel::domain(
            scioxide_core::argument!("self"),
            "square root has a singular block recurrence",
        ));
    }
    let result = [
        S::from_real(root.re),
        b / S::from_real(divisor),
        c / S::from_real(divisor),
        S::from_real(root.re),
    ];
    for &value in &result {
        crate::checked::finite_candidate(value)?;
    }
    Ok(result)
}
// Adapt ordinary local equations to the single complete-pivoting kernel.
fn solve_small<S: NumericScalar>(
    a: &[S; 16],
    b: &mut [S; 4],
    n: usize,
) -> Result<(), SciOxideError> {
    let mut coefficients = [scioxide_core::ScaledValue::ZERO; 16];
    let mut rhs = [scioxide_core::ScaledValue::ZERO; 4];
    for i in 0..n {
        for j in 0..n {
            coefficients[i * 4 + j] = extended(a[i * 4 + j])?;
        }
    }
    for (out, &value) in rhs[..n].iter_mut().zip(&b[..n]) {
        *out = extended(value)?;
    }
    if !crate::kernel::small::small(&mut coefficients, &mut rhs, n, Singularity::Consistent)? {
        return Err(super::kernel::domain(
            scioxide_core::argument!("self"),
            "square root has a singular block recurrence",
        ));
    }
    for (out, &value) in b[..n].iter_mut().zip(&rhs[..n]) {
        *out = scalar(value)?;
    }
    Ok(())
}
fn legendre<S: NumericScalar>(
    degree: usize,
    node: usize,
) -> Result<(S::Real, S::Real), SciOxideError> {
    let n = crate::checked::rounded_count::<S::Real>(degree)?;
    let mut z = (real::<S>(core::f64::consts::PI)?
        * (crate::checked::rounded_count::<S::Real>(node)? + real::<S>(0.75)?)
        / (n + S::Real::HALF))
        .cos();
    let mut derivative = S::Real::ONE;
    for _ in 0..16 {
        let mut p = S::Real::ONE;
        let mut previous = S::Real::ZERO;
        for k in 1..=degree {
            let kr = crate::checked::rounded_count::<S::Real>(k)?;
            let next =
                ((kr * S::Real::TWO - S::Real::ONE) * z * p - (kr - S::Real::ONE) * previous) / kr;
            previous = p;
            p = next;
        }
        derivative = n * (z * p - previous) / (z * z - S::Real::ONE);
        let step = p / derivative;
        z -= step;
        if step.abs() <= S::Real::EPSILON * real::<S>(4.)? {
            break;
        }
    }
    Ok((
        (z + S::Real::ONE) * S::Real::HALF,
        S::Real::ONE / ((S::Real::ONE - z * z) * derivative * derivative),
    ))
}
fn log_divided<S: NumericScalar>(a: S, b: S, value: S) -> Result<S, SciOxideError> {
    if a == b {
        return Ok(value / a);
    }
    if S::IS_REAL && (b - a).abs() < a.abs() * S::Real::HALF {
        let ratio = (b.real_component() - a.real_component()) / a.real_component();
        return Ok(value * S::from_real(ratio.ln_1p()) / (b - a));
    }
    let difference = scalar_log(b) - scalar_log(a);
    let sum = a + b;
    if !S::IS_REAL
        && sum.has_finite_components()
        && (b - a).abs() <= sum.abs() * S::Real::HALF
        && difference.imaginary_component().abs()
            <= S::Real::try_from_f64_lossy(core::f64::consts::PI)?
    {
        // Higham's atanh divided difference, without subtracting nearby logs.
        // The logarithm difference selects the branch before using this local
        // expansion: conjugates across the negative cut must retain 2*pi*i.
        let ratio = (b - a) / sum;
        let square = ratio * ratio;
        let mut power = S::ONE;
        let mut series = S::ONE;
        for k in 1..=32 {
            power = power * square;
            let term = power / S::from_real(crate::checked::rounded_count::<S::Real>(2 * k + 1)?);
            series = series + term;
            if term.abs() <= S::Real::EPSILON * series.abs() {
                break;
            }
        }
        return Ok((value / sum) * (S::from_real(S::Real::TWO) * series));
    }
    Ok(value * difference / (b - a))
}
