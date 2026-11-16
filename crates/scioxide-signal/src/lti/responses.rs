//! Prepared frequency and time responses of state-space models.

use super::*;
use scioxide_core::{Capacity, backend::try_reserve};
use scioxide_linalg::backend::hessenberg;

/// Workspace size that a [`FrequencyResponsePlan`] needs for a sweep of a given length.
///
/// For `K` frequencies, `n` states, `m` inputs and `p` outputs the workspace
/// holds `n² + n m + K p m` complex values.
#[derive(Clone, Copy, Debug)]
pub struct FrequencyResponseRequirements {
    factor: usize,
    rhs: usize,
    output: usize,
    total: usize,
}

impl FrequencyResponseRequirements {
    /// Returns the output length `K p m`, in `[frequency][output][input]` order.
    pub const fn output_len(&self) -> usize {
        self.output
    }
}

/// Workspace of a [`FrequencyResponsePlan`].
///
/// [`FrequencyResponsePlan::workspace`] allocates one for a sweep length, and
/// [`Self::try_reserve`] grows it for a longer sweep. Execution never grows it.
#[derive(Clone, Debug)]
pub struct FrequencyResponseWorkspace<S: RealScalar> {
    values: Vec<Complex<S>>,
}

impl<S: RealScalar> Default for FrequencyResponseWorkspace<S> {
    fn default() -> Self {
        Self { values: Vec::new() }
    }
}

impl<S: RealScalar> FrequencyResponseWorkspace<S> {
    /// Grows the workspace to hold `request`; it never shrinks.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn try_reserve(
        &mut self,
        request: &FrequencyResponseRequirements,
    ) -> Result<(), SciOxideError> {
        if self.values.len() < request.total {
            let additional = request.total - self.values.len();
            try_reserve(
                &mut self.values,
                additional,
                scioxide_core::quantity!("lti_frequency_workspace"),
            )?;
            self.values
                .resize(request.total, Complex::new(S::ZERO, S::ZERO));
        }
        Ok(())
    }

    /// Reports whether the workspace can hold `request`, without growing it.
    ///
    /// # Errors
    ///
    /// Never returns an error. The `Result` follows the workspace protocol in
    /// [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn check(
        &self,
        request: &FrequencyResponseRequirements,
    ) -> Result<Capacity, SciOxideError> {
        Ok(if self.values.len() >= request.total {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
}

/// Prepared frequency response `H(λ) = C (λI − A)⁻¹ B + D` of a state-space model.
///
/// Continuous models use `λ = iω`, with `ω` in radians per time unit. Discrete
/// models use `λ = exp(iπf)`, with `f` in multiples of the Nyquist frequency,
/// so the sample period does not enter (see the [crate conventions](crate#conventions)).
///
/// For `n` states and `m` inputs, preparation reduces `A` to upper Hessenberg
/// form `A = Q H Qᴴ` with Householder reflections in O(n³) operations and
/// stores `H`, `Qᴴ B` and `C Q`. Each frequency then solves
/// `(λI − H) X = Qᴴ B` by Gaussian elimination with row interchanges in
/// O(n² m) operations and forms
/// `C Q X + D`. It does not form the characteristic polynomial, whose
/// coefficients are ill-conditioned for high-order models. Accuracy degrades
/// as `λ` approaches a pole, where `λI − A` is ill-conditioned.
///
/// Plans and workspaces follow the rules in
/// [Plans and workspaces](scioxide_core#plans-and-workspaces). For one sweep,
/// [`StateSpace::frequency_response`] is simpler.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::Matrix;
/// use scioxide_signal::{FrequencyResponsePlan, LtiTime, StateSpace};
/// let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
/// let b = Matrix::from_vec(1, 1, vec![1.0])?;
/// let d = Matrix::zeros(1, 1)?;
/// let model = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), d.view())?;
/// let plan = FrequencyResponsePlan::new(&model)?;
/// let mut workspace = plan.workspace(2)?;
/// let mut output = vec![scioxide_core::Complex::new(0.0, 0.0); 2];
/// let cx = ExecutionContext::sequential_native();
/// plan.frequency_response_into(&[0.0, 1.0], &mut output, &mut workspace, cx)?;
/// // H(s) = 1 / (s + 1): unit DC gain and H(i) = (1 - i) / 2.
/// assert!((output[0] - scioxide_core::Complex::new(1.0, 0.0)).norm() < 1e-14);
/// assert!((output[1] - scioxide_core::Complex::new(0.5, -0.5)).norm() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. J. Laub, "Efficient multivariable frequency response computations",
///   *IEEE Trans. Automat. Control* 26(2), 1981, 407–408.
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
///   Hopkins, 2013, §7.4.3.
#[derive(Clone, Debug)]
pub struct FrequencyResponsePlan<T: NumericScalar> {
    time: LtiTime<T::Real>,
    h: Matrix<Complex<T::Real>>,
    b: Matrix<Complex<T::Real>>,
    c: Matrix<Complex<T::Real>>,
    d: Matrix<Complex<T::Real>>,
}

impl<T> FrequencyResponsePlan<T>
where
    T: NumericScalar,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Prepares the plan for `model` with sequential execution.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if an entry of `Qᴴ B` or `C Q` overflows.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(model: &StateSpace<T>) -> Result<Self, SciOxideError> {
        Self::build(model, ExecutionContext::sequential_native())
    }

    /// Prepares the plan for `model` under the execution context `cx`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `ExecutionStopped` if `cx` is cancelled or its deadline passes.
    pub fn build(model: &StateSpace<T>, cx: ExecutionContext<'_>) -> Result<Self, SciOxideError> {
        cx.check()?;
        let (h, q) = hessenberg::reduce(model.a.view(), cx)?;
        let (n, m, p) = (model.states(), model.inputs(), model.outputs());
        let mut b = Matrix::zeros(n, m)?;
        let mut c = Matrix::zeros(p, n)?;
        for row in 0..n {
            cx.check()?;
            for col in 0..m {
                b.as_mut_slice()[row * m + col] =
                    crate::util::working_sum_products((0..n).map(|k| {
                        (
                            q.as_slice()[k * n + row].conj(),
                            model.b.as_slice()[k * m + col],
                        )
                    }))?;
            }
        }
        for row in 0..p {
            cx.check()?;
            for col in 0..n {
                c.as_mut_slice()[row * n + col] = crate::util::working_sum_products(
                    (0..n).map(|k| (model.c.as_slice()[row * n + k], q.as_slice()[k * n + col])),
                )?;
            }
        }
        cx.check()?;
        Ok(Self {
            time: model.time,
            h: complex_matrix(h.view())?,
            b: complex_matrix(b.view())?,
            c: complex_matrix(c.view())?,
            d: complex_matrix(model.d.view())?,
        })
    }

    /// Returns the workspace size for a sweep of `frequencies` points.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        frequencies: usize,
    ) -> Result<FrequencyResponseRequirements, SciOxideError> {
        let factor = checked_len(
            &[self.h.rows(), self.h.rows()],
            scioxide_core::quantity!("lti_resolvent"),
        )?;
        let rhs = self.b.as_slice().len();
        let output = checked_len(
            &[frequencies, self.c.rows(), self.b.cols()],
            scioxide_core::quantity!("lti_frequency_output"),
        )?;
        let total = factor
            .checked_add(rhs)
            .and_then(|v| v.checked_add(output))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("lti_frequency_workspace"))
            })?;
        Ok(FrequencyResponseRequirements {
            factor,
            rhs,
            output,
            total,
        })
    }

    /// Allocates a workspace for a sweep of `frequencies` points.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn workspace(
        &self,
        frequencies: usize,
    ) -> Result<FrequencyResponseWorkspace<T::Real>, SciOxideError> {
        let mut ws = FrequencyResponseWorkspace::default();
        ws.try_reserve(&self.workspace_requirements(frequencies)?)?;
        Ok(ws)
    }

    /// Computes the frequency response at `frequencies` and writes it to `output`.
    ///
    /// `output` has length `K p m` for `K` frequencies, `p` outputs and `m`
    /// inputs, and holds `p × m` row-major blocks in `[frequency][output][input]` order. A model without
    /// inputs or outputs costs nothing per frequency.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` does not have length `K p m` or the
    ///   workspace is too small.
    /// - `NonFinite` if a frequency is NaN or infinite.
    /// - `Singular` if `λI − A` is exactly singular at a frequency.
    /// - `NumericalInstability` if `π f` or a response value overflows.
    /// - `SizeOverflow` if `K p m` overflows `usize`.
    /// - `ExecutionStopped` if `cx` is cancelled or its deadline passes.
    pub fn frequency_response_into(
        &self,
        frequencies: &[T::Real],
        output: &mut [Complex<T::Real>],
        ws: &mut FrequencyResponseWorkspace<T::Real>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let req = self.workspace_requirements(frequencies.len())?;
        if output.len() != req.output {
            return Err(dimension(req.output, output.len()));
        }
        if ws.check(&req)? != Capacity::Sufficient {
            return Err(dimension(req.total, ws.values.len()));
        }
        crate::util::check_real_slice_finite(frequencies)?;
        cx.check()?;
        let (factor, rest) = ws.values.split_at_mut(req.factor);
        let (rhs, rest) = rest.split_at_mut(req.rhs);
        let candidate = &mut rest[..req.output];
        let (n, m, p) = (self.h.rows(), self.b.cols(), self.c.rows());
        let h = hessenberg::HessenbergView::new(self.h.view())?;
        // Empty channel axes are zero work, including at otherwise singular shifts.
        if m != 0 && p != 0 {
            for (sample, &frequency) in frequencies.iter().enumerate() {
                cx.check()?;
                let shift = match self.time {
                    LtiTime::Continuous => Complex::new(T::Real::ZERO, frequency),
                    LtiTime::Discrete { .. } => {
                        let angle = (frequency % T::Real::TWO) * T::Real::PI;
                        Complex::from_polar(T::Real::ONE, angle)
                    }
                };
                rhs.copy_from_slice(self.b.as_slice());
                hessenberg::solve_shifted(h, shift, m, rhs, factor)?;
                for row in 0..p {
                    for col in 0..m {
                        let value = crate::util::working_sum_products(
                            (0..n).map(|k| (self.c.as_slice()[row * n + k], rhs[k * m + col])),
                        )?;
                        candidate[sample * p * m + row * m + col] =
                            finite(value + self.d.as_slice()[row * m + col])?;
                    }
                }
            }
        }
        cx.check()?;
        output.copy_from_slice(candidate);
        Ok(())
    }
}

/// Workspace of a [`TimeResponsePlan`], allocated by [`TimeResponsePlan::workspace`].
///
/// Its size is fixed by the plan, and one workspace serves every execution
/// method of that plan.
#[derive(Clone, Debug)]
pub struct TimeResponseWorkspace<T: NumericScalar> {
    values: Vec<T>,
}

/// Prepared impulse, step and sampled-input responses of a state-space model on a time grid.
///
/// The grid `times` must be finite and nonempty, start at zero and be uniform
/// with step `h = times[1]`. Each sample may differ from the rounded product
/// `k h` by at most `64 ε (|times[k]| + h)`, where `ε` is the machine epsilon,
/// so grids computed as `k h` or `k T / (K − 1)` pass, where `K = times.len()`.
/// For a discrete model with sample period `T`, `h` must lie within
/// `64 ε (1 + T)` of `T`.
///
/// For a continuous model, preparation computes the exact zero-order-hold
/// transition for step `h` once (see [`DiscretizationMethod::ZeroOrderHold`]).
/// Each execution then steps `x[k+1] = A_d x[k] + B_d u[k]` in
/// O(K (n + p)(n + m)) operations per response channel, for `n` states, `m`
/// inputs and `p` outputs. Accuracy depends on the
/// conditioning of the system and on rounding accumulated over the steps.
///
/// Every execution method writes the outputs and the state history together.
/// [`StateSpace::impulse_response`], [`StateSpace::step_response`] and
/// [`StateSpace::simulate_zoh`] are the owned forms.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::Matrix;
/// use scioxide_signal::{LtiTime, StateSpace, TimeResponsePlan};
/// let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
/// let b = Matrix::from_vec(1, 1, vec![1.0])?;
/// let d = Matrix::zeros(1, 1)?;
/// let model = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), d.view())?;
/// let plan = TimeResponsePlan::new(&model, &[0.0, 0.1, 0.2])?;
/// let (mut ws, cx) = (plan.workspace()?, ExecutionContext::sequential_native());
/// let mut output = vec![0.0; plan.output_len()];
/// let mut states = vec![0.0; plan.states_len()];
/// plan.step_response_into(&mut output, &mut states, &mut ws, cx)?;
/// // A unit input from the zero state reproduces the step response.
/// let mut simulated = vec![0.0; plan.simulation_output_len()];
/// let mut history = vec![0.0; plan.simulation_states_len()];
/// plan.simulate_into(&[1.0; 3], &[0.0], &mut simulated, &mut history, &mut ws, cx)?;
/// assert!((simulated[2] - output[2]).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct TimeResponsePlan<T: NumericScalar> {
    model: StateSpace<T>,
    initial_impulse: Matrix<T>,
    continuous: bool,
    times: Vec<T::Real>,
    output_len: usize,
    states_len: usize,
    simulation_input_len: usize,
    simulation_output_len: usize,
    simulation_states_len: usize,
    workspace_size: usize,
}

impl<T> TimeResponsePlan<T>
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Prepares the plan for `model` on the grid `times` with sequential execution.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `times` is empty.
    /// - `InvalidInput` (`OutOfRange`) if `times` does not start at zero, if
    ///   `times[1]` is not positive, or if the grid is not uniform within the
    ///   tolerance above.
    /// - `InvalidInput` (`Inconsistent`) if the grid step of a discrete model
    ///   differs from its sample period.
    /// - `NonFinite` if a time is NaN or infinite.
    /// - `NumericalInstability` if the matrix exponential overflows.
    /// - `SizeOverflow` if a buffer length overflows `usize`.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(model: &StateSpace<T>, times: &[T::Real]) -> Result<Self, SciOxideError> {
        Self::build(model, times, ExecutionContext::sequential_native())
    }

    /// Prepares the plan for `model` on the grid `times` under the execution context `cx`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `ExecutionStopped` if `cx` is cancelled or its deadline passes.
    pub fn build(
        model: &StateSpace<T>,
        times: &[T::Real],
        cx: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        validate_times(times, model.time)?;
        cx.check()?;
        let continuous = matches!(model.time, LtiTime::Continuous);
        let output_len = checked_len(
            &[times.len(), model.outputs(), model.inputs()],
            scioxide_core::quantity!("lti_response"),
        )?;
        let states_len = checked_len(
            &[times.len(), model.states(), model.inputs()],
            scioxide_core::quantity!("lti_response_states"),
        )?;
        let work = checked_len(
            &[2, model.states(), model.inputs()],
            scioxide_core::quantity!("lti_response_scratch"),
        )?;
        let simulation_input_len = checked_len(
            &[times.len(), model.inputs()],
            scioxide_core::quantity!("lti_input"),
        )?;
        let simulation_output_len = checked_len(
            &[times.len(), model.outputs()],
            scioxide_core::quantity!("lti_output"),
        )?;
        let simulation_states_len = checked_len(
            &[times.len(), model.states()],
            scioxide_core::quantity!("lti_states"),
        )?;
        let total = |parts: [usize; 3]| {
            parts
                .into_iter()
                .try_fold(0_usize, usize::checked_add)
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("lti_time_workspace"))
                })
        };
        let simulation_work = checked_len(
            &[2, model.states()],
            scioxide_core::quantity!("lti_simulation_scratch"),
        )?;
        let workspace_size = total([output_len, states_len, work])?.max(total([
            simulation_output_len,
            simulation_states_len,
            simulation_work,
        ])?);
        let initial_impulse = materialize(model.b.view())?;
        let prepared = if continuous && times.len() > 1 && model.states() != 0 {
            model.discretize_zoh(times[1] - times[0], cx)?
        } else {
            model.view().to_owned()?
        };
        cx.check()?;
        Ok(Self {
            model: prepared,
            initial_impulse,
            continuous,
            times: copy_slice(times, scioxide_core::quantity!("lti_times"))?,
            output_len,
            states_len,
            simulation_input_len,
            simulation_output_len,
            simulation_states_len,
            workspace_size,
        })
    }

    /// Returns the sample times.
    pub fn times(&self) -> &[T::Real] {
        &self.times
    }
    /// Returns the output length `K p m` of the impulse and step responses, in
    /// `[sample][output][input]` order.
    pub const fn output_len(&self) -> usize {
        self.output_len
    }
    /// Returns the state-history length `K n m` of the impulse and step
    /// responses, in `[sample][state][input]` order.
    pub const fn states_len(&self) -> usize {
        self.states_len
    }
    /// Returns the input length `K m` of [`Self::simulate_into`], in `[sample][input]` order.
    pub const fn simulation_input_len(&self) -> usize {
        self.simulation_input_len
    }
    /// Returns the output length `K p` of [`Self::simulate_into`], in `[sample][output]` order.
    pub const fn simulation_output_len(&self) -> usize {
        self.simulation_output_len
    }
    /// Returns the state-history length `K n` of [`Self::simulate_into`], in
    /// `[sample][state]` order.
    pub const fn simulation_states_len(&self) -> usize {
        self.simulation_states_len
    }
    /// Allocates a workspace for every execution method of this plan.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn workspace(&self) -> Result<TimeResponseWorkspace<T>, SciOxideError> {
        Ok(TimeResponseWorkspace {
            values: try_zeros(
                self.workspace_size,
                scioxide_core::quantity!("lti_time_workspace"),
            )?,
        })
    }

    /// Simulates the response to sampled inputs held constant between samples.
    ///
    /// `inputs` holds `u[k]` in `[sample][input]` order and `initial_state`
    /// holds `x(0)`. The outputs go to `output` in `[sample][output]` order and
    /// the states to `states` in `[sample][state]` order. The first row of
    /// `states` is `initial_state`. Continuous models step the exact
    /// zero-order-hold transition and discrete models step their own matrices.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `inputs`, `initial_state`, `output` or `states`
    ///   does not have the length the plan reports, or the workspace is too small.
    /// - `NonFinite` if `inputs` or `initial_state` contains NaN or infinity.
    /// - `NumericalInstability` if a state or output value overflows.
    /// - `ExecutionStopped` if `cx` is cancelled or its deadline passes.
    pub fn simulate_into(
        &self,
        inputs: &[T],
        initial_state: &[T],
        output: &mut [T],
        states: &mut [T],
        ws: &mut TimeResponseWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (n, m, p) = (
            self.model.states(),
            self.model.inputs(),
            self.model.outputs(),
        );
        if inputs.len() != self.simulation_input_len {
            return Err(dimension(self.simulation_input_len, inputs.len()));
        }
        if initial_state.len() != n {
            return Err(dimension(n, initial_state.len()));
        }
        if output.len() != self.simulation_output_len {
            return Err(dimension(self.simulation_output_len, output.len()));
        }
        if states.len() != self.simulation_states_len {
            return Err(dimension(self.simulation_states_len, states.len()));
        }
        if ws.values.len() < self.workspace_size {
            return Err(dimension(self.workspace_size, ws.values.len()));
        }
        crate::util::inputs(inputs)?;
        crate::util::inputs(initial_state)?;
        cx.check()?;
        let (candidate, rest) = ws.values.split_at_mut(self.simulation_output_len);
        let (history, rest) = rest.split_at_mut(self.simulation_states_len);
        let (mut state, rest) = rest.split_at_mut(n);
        let mut next = &mut rest[..n];
        state.copy_from_slice(initial_state);
        let (a, b, c, d) = (
            self.model.a.as_slice(),
            self.model.b.as_slice(),
            self.model.c.as_slice(),
            self.model.d.as_slice(),
        );
        for sample in 0..self.times.len() {
            cx.check()?;
            history[sample * n..(sample + 1) * n].copy_from_slice(state);
            let input = &inputs[sample * m..(sample + 1) * m];
            for (row, value) in candidate[sample * p..(sample + 1) * p]
                .iter_mut()
                .enumerate()
            {
                let mut sum = T::ZERO;
                for (index, &state_value) in state.iter().enumerate() {
                    sum = sum + c[row * n + index] * state_value;
                }
                for (col, &input_value) in input.iter().enumerate() {
                    sum = sum + d[row * m + col] * input_value;
                }
                *value = finite(sum)?;
            }
            if sample + 1 < self.times.len() {
                for (row, next_value) in next.iter_mut().enumerate() {
                    let mut sum = T::ZERO;
                    for (col, &state_value) in state.iter().enumerate() {
                        sum = sum + a[row * n + col] * state_value;
                    }
                    for (col, &input_value) in input.iter().enumerate() {
                        sum = sum + b[row * m + col] * input_value;
                    }
                    *next_value = finite(sum)?;
                }
                core::mem::swap(&mut state, &mut next);
            }
        }
        cx.check()?;
        output.copy_from_slice(candidate);
        states.copy_from_slice(history);
        Ok(())
    }

    /// Computes the response to a unit impulse on each input and writes the outputs and states.
    ///
    /// Channel `j` is the response to an impulse on input `j` alone. For a
    /// continuous model the outputs are `C exp(A t) B`, the regular part of the
    /// impulse response. The term `D δ(t)`, with `δ` the Dirac delta, is not
    /// sampled. For a discrete
    /// model the outputs are `y[0] = D` and `y[k] = C Aᵏ⁻¹ B` for `k ≥ 1`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` or `states` does not have the length
    ///   the plan reports, or the workspace is too small.
    /// - `NumericalInstability` if a state or output value overflows.
    /// - `ExecutionStopped` if `cx` is cancelled or its deadline passes.
    pub fn impulse_response_into(
        &self,
        output: &mut [T],
        states: &mut [T],
        ws: &mut TimeResponseWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.execute(false, output, states, ws, cx)
    }

    /// Computes the response to a unit step on each input and writes the outputs and states.
    ///
    /// Channel `j` is the response to a step on input `j` alone, starting from
    /// rest. The outputs include `D` at every sample.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` or `states` does not have the length
    ///   the plan reports, or the workspace is too small.
    /// - `NumericalInstability` if a state or output value overflows.
    /// - `ExecutionStopped` if `cx` is cancelled or its deadline passes.
    pub fn step_response_into(
        &self,
        output: &mut [T],
        states: &mut [T],
        ws: &mut TimeResponseWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.execute(true, output, states, ws, cx)
    }

    fn execute(
        &self,
        step: bool,
        output: &mut [T],
        states: &mut [T],
        ws: &mut TimeResponseWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if output.len() != self.output_len {
            return Err(dimension(self.output_len, output.len()));
        }
        if states.len() != self.states_len {
            return Err(dimension(self.states_len, states.len()));
        }
        if ws.values.len() < self.workspace_size {
            return Err(dimension(self.workspace_size, ws.values.len()));
        }
        cx.check()?;
        let (candidate, rest) = ws.values.split_at_mut(self.output_len);
        let (history, rest) = rest.split_at_mut(self.states_len);
        let (n, m, p) = (
            self.model.states(),
            self.model.inputs(),
            self.model.outputs(),
        );
        let (mut state, rest) = rest.split_at_mut(n * m);
        let mut next = &mut rest[..n * m];
        state.fill(T::ZERO);
        if !step && self.continuous {
            state.copy_from_slice(self.initial_impulse.as_slice());
        }
        for sample in 0..self.times.len() {
            cx.check()?;
            history[sample * n * m..(sample + 1) * n * m].copy_from_slice(state);
            for row in 0..p {
                for col in 0..m {
                    let value = crate::util::working_sum_products(
                        (0..n).map(|k| (self.model.c.as_slice()[row * n + k], state[k * m + col])),
                    )?;
                    let direct = if step || !self.continuous && sample == 0 {
                        self.model.d.as_slice()[row * m + col]
                    } else {
                        T::ZERO
                    };
                    candidate[sample * p * m + row * m + col] = finite(value + direct)?;
                }
            }
            if sample + 1 < self.times.len() {
                for row in 0..n {
                    for col in 0..m {
                        let value =
                            crate::util::working_sum_products((0..n).map(|k| {
                                (self.model.a.as_slice()[row * n + k], state[k * m + col])
                            }))?;
                        let drive = if step || !self.continuous && sample == 0 {
                            self.model.b.as_slice()[row * m + col]
                        } else {
                            T::ZERO
                        };
                        next[row * m + col] = finite(value + drive)?;
                    }
                }
                core::mem::swap(&mut state, &mut next);
            }
        }
        cx.check()?;
        output.copy_from_slice(candidate);
        states.copy_from_slice(history);
        Ok(())
    }
}
