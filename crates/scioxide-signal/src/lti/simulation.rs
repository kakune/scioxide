use super::*;

impl<T> StateSpace<T>
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Simulates the response to sampled inputs held constant between samples.
    ///
    /// `inputs` holds `u[k]` in `[sample][input]` order, `times.len() × m` values,
    /// and `initial_state` holds `x(0)`. The returned [`TimeResponse`] has one
    /// channel. A continuous model steps the exact zero-order-hold transition
    /// for the grid step, so the samples are exact up to rounding for an input
    /// that is constant on each interval. A discrete model steps its own
    /// matrices. [`TimeResponsePlan`] states the conditions on `times`.
    ///
    /// This owned form prepares a [`TimeResponsePlan`] and runs
    /// [`TimeResponsePlan::simulate_into`] with sequential execution. Reuse a
    /// plan for repeated simulations on one grid.
    ///
    /// # Errors
    ///
    /// - The errors of [`TimeResponsePlan::new`] and
    ///   [`TimeResponsePlan::simulate_into`].
    /// - `AllocationFailure` if the result cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// use scioxide_signal::{LtiTime, StateSpace};
    /// let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
    /// let b = Matrix::from_vec(1, 1, vec![1.0])?;
    /// let d = Matrix::zeros(1, 1)?;
    /// let model = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), d.view())?;
    /// // Free decay x' = −x from x(0) = 2 gives y(1) = 2 exp(−1).
    /// let response = model.simulate_zoh(&[0.0, 0.5, 1.0], &[0.0; 3], &[2.0])?;
    /// assert!((response.values()[2] - 2.0 * (-1.0_f64).exp()).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn simulate_zoh(
        &self,
        times: &[T::Real],
        inputs: &[T],
        initial_state: &[T],
    ) -> Result<TimeResponse<T>, SciOxideError> {
        let plan = TimeResponsePlan::new(self, times)?;
        let mut workspace = plan.workspace()?;
        let mut values = try_zeros(
            plan.simulation_output_len(),
            scioxide_core::quantity!("lti_output"),
        )?;
        let mut states = try_zeros(
            plan.simulation_states_len(),
            scioxide_core::quantity!("lti_states"),
        )?;
        plan.simulate_into(
            inputs,
            initial_state,
            &mut values,
            &mut states,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(TimeResponse {
            times: copy_slice(times, scioxide_core::quantity!("lti_times"))?,
            values,
            states,
            outputs: self.outputs(),
            channels: 1,
            state_count: self.states(),
        })
    }

    /// Computes the response to a unit impulse on each input, one channel per input.
    ///
    /// For a continuous model the samples are `C exp(A t) B`, the regular part
    /// of the impulse response. The term `D δ(t)`, with `δ` the Dirac delta, is
    /// not sampled. For a discrete model the samples are `y[0] = D` and
    /// `y[k] = C Aᵏ⁻¹ B` for `k ≥ 1`. [`TimeResponsePlan`] states the conditions
    /// on `times`.
    ///
    /// # Errors
    ///
    /// - The errors of [`TimeResponsePlan::new`].
    /// - `NumericalInstability` if a state or output value overflows.
    /// - `AllocationFailure` if the result cannot be stored.
    pub fn impulse_response(&self, times: &[T::Real]) -> Result<TimeResponse<T>, SciOxideError> {
        self.standard_response(times, false)
    }

    /// Computes the response to a unit step on each input, one channel per input.
    ///
    /// The state starts at zero and the output includes `D` at every sample. A
    /// continuous model steps the exact zero-order-hold transition, so the
    /// samples are exact up to rounding. [`TimeResponsePlan`] states the
    /// conditions on `times`.
    ///
    /// # Errors
    ///
    /// - The errors of [`TimeResponsePlan::new`].
    /// - `NumericalInstability` if a state or output value overflows.
    /// - `AllocationFailure` if the result cannot be stored.
    pub fn step_response(&self, times: &[T::Real]) -> Result<TimeResponse<T>, SciOxideError> {
        self.standard_response(times, true)
    }

    fn standard_response(
        &self,
        times: &[T::Real],
        step: bool,
    ) -> Result<TimeResponse<T>, SciOxideError> {
        let plan = TimeResponsePlan::new(self, times)?;
        let mut workspace = plan.workspace()?;
        let mut values = try_zeros(plan.output_len(), scioxide_core::quantity!("lti_response"))?;
        let mut states = try_zeros(
            plan.states_len(),
            scioxide_core::quantity!("lti_response_states"),
        )?;
        if step {
            plan.step_response_into(
                &mut values,
                &mut states,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )?;
        } else {
            plan.impulse_response_into(
                &mut values,
                &mut states,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )?;
        }
        Ok(TimeResponse {
            times: copy_slice(times, scioxide_core::quantity!("lti_times"))?,
            values,
            states,
            outputs: self.outputs(),
            channels: self.inputs(),
            state_count: self.states(),
        })
    }
}
