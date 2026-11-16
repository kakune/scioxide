//! Dense continuous- and discrete-time linear time-invariant systems.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::AlgebraicScalar;

use scioxide_core::{
    Argument, Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_linalg::{
    ConditionEstimate, GeneralEigenOptions, LinalgWorkspace, LuDecomposition, Matrix,
    MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan, MatrixView, RankPolicy, RankReport,
};

use crate::util::{
    empty, exact_len, inconsistent, invalid, lti_finite as finite, zeroed as try_zeros,
};
use crate::{IirDomain, PoleZeroGain, TransferFunction};
use scioxide_core::backend::{checked_product as checked_len, try_copy_slice as copy_slice};

mod analysis;
mod conversions;
mod discretize;
mod placement;
mod responses;
mod simulation;
#[cfg(test)]
mod tests;
pub use placement::{PolePlacementOptions, PolePlacementReport, PolePlacementTermination};
pub use responses::{
    FrequencyResponsePlan, FrequencyResponseRequirements, FrequencyResponseWorkspace,
    TimeResponsePlan, TimeResponseWorkspace,
};

/// Time domain of a state-space model: continuous, or discrete with a sample period.
///
/// The output equation is `y = C x + D u` in both domains. Frequency arguments
/// follow the [crate conventions](crate#conventions): radians per time unit in
/// continuous time and multiples of the Nyquist frequency in discrete time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LtiTime<S: RealScalar> {
    /// Continuous-time dynamics `x' = A x + B u`, where `x'` is the time derivative of `x`.
    Continuous,
    /// Discrete-time dynamics `x[k+1] = A x[k] + B u[k]`.
    Discrete {
        /// Time between adjacent samples. It must be finite and positive.
        sample_period: S,
    },
}

impl<S: RealScalar> LtiTime<S> {
    /// Validates the sample period; failures name `argument`.
    fn validate(self, argument: Argument) -> Result<(), SciOxideError> {
        match self {
            Self::Continuous => Ok(()),
            Self::Discrete { sample_period } => {
                crate::util::check_real_slice_finite(&[sample_period])?;
                if sample_period > S::ZERO {
                    Ok(())
                } else {
                    Err(invalid(argument, "sample period must be positive"))
                }
            }
        }
    }

    /// Returns the sample period, or `None` for continuous time.
    pub const fn sample_period(self) -> Option<S> {
        match self {
            Self::Continuous => None,
            Self::Discrete { sample_period } => Some(sample_period),
        }
    }
}

/// Method that [`StateSpace::discretize`] uses to convert a continuous model to discrete time.
///
/// For sample period `h`, both methods give a model `x[k+1] = A_d x[k] + B_d u[k]`,
/// `y[k] = C_d x[k] + D_d u[k]`.
///
/// # References
///
/// G. F. Franklin, J. D. Powell and M. L. Workman, *Digital Control of Dynamic
/// Systems*, 3rd ed., Addison-Wesley, 1998.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiscretizationMethod<S: RealScalar> {
    /// Zero-order hold: the exact discrete model for an input held constant over
    /// each sample interval.
    ///
    /// `A_d = exp(A h)` and `B_d = (∫₀^h exp(A s) ds) B` are read from the matrix
    /// exponential of the block matrix `[A B; 0 0] h` of order `n + m`, where `n`
    /// is the number of states and `m` the number of inputs. `C` and `D` are
    /// unchanged.
    ZeroOrderHold,
    /// Generalized bilinear transform with weight `α` (`alpha`) in `[0, 1]`.
    ///
    /// ```text
    /// A_d = (I − α h A)⁻¹ (I + (1 − α) h A)
    /// B_d = (I − α h A)⁻¹ h B
    /// C_d = C (I − α h A)⁻¹
    /// D_d = D + α C B_d
    /// ```
    ///
    /// `α = 0` is forward Euler, `α = 1/2` is the bilinear (Tustin) transform and
    /// `α = 1` is backward Euler. One LU factorization of `I − α h A` gives all
    /// four matrices.
    GeneralizedBilinear {
        /// Weight `α` of the implicit side of the transform, in `[0, 1]`.
        alpha: S,
    },
}

/// Borrowed view of the matrices and time domain of a [`StateSpace`] model.
///
/// [`StateSpace::view`] creates it without copying, and [`Self::to_owned`]
/// copies it back into a model.
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_signal::{LtiTime, StateSpace};
/// let a = Matrix::from_vec(2, 2, vec![0.0_f64, 1.0, -2.0, -3.0])?;
/// let b = Matrix::from_vec(2, 1, vec![0.0, 1.0])?;
/// let c = Matrix::from_vec(1, 2, vec![1.0, 0.0])?;
/// let model = StateSpace::new(
///     LtiTime::Continuous,
///     a.view(),
///     b.view(),
///     c.view(),
///     Matrix::zeros(1, 1)?.view(),
/// )?;
/// let view = model.view();
/// assert_eq!((view.states(), view.inputs(), view.outputs()), (2, 1, 1));
/// assert_eq!(view.to_owned()?.a().as_slice(), model.a().as_slice());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StateSpaceView<'a, T: NumericScalar> {
    time: LtiTime<T::Real>,
    a: MatrixView<'a, T>,
    b: MatrixView<'a, T>,
    c: MatrixView<'a, T>,
    d: MatrixView<'a, T>,
}

impl<'a, T: NumericScalar> StateSpaceView<'a, T> {
    /// Returns the time domain.
    pub const fn time(self) -> LtiTime<T::Real> {
        self.time
    }
    /// Returns the state matrix `A`, of shape `states × states`.
    pub const fn a(self) -> MatrixView<'a, T> {
        self.a
    }
    /// Returns the input matrix `B`, of shape `states × inputs`.
    pub const fn b(self) -> MatrixView<'a, T> {
        self.b
    }
    /// Returns the output matrix `C`, of shape `outputs × states`.
    pub const fn c(self) -> MatrixView<'a, T> {
        self.c
    }
    /// Returns the feedthrough matrix `D`, of shape `outputs × inputs`.
    pub const fn d(self) -> MatrixView<'a, T> {
        self.d
    }
    /// Returns the number of states `n`.
    pub const fn states(self) -> usize {
        self.a.rows()
    }
    /// Returns the number of inputs `m`.
    pub const fn inputs(self) -> usize {
        self.b.cols()
    }
    /// Returns the number of outputs `p`.
    pub const fn outputs(self) -> usize {
        self.c.rows()
    }

    /// Copies the view into an owned [`StateSpace`].
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the copies cannot be stored. The view comes from
    ///   a valid model, so no other error of [`StateSpace::new`] occurs.
    pub fn to_owned(self) -> Result<StateSpace<T>, SciOxideError> {
        StateSpace::new(self.time, self.a, self.b, self.c, self.d)
    }
}

/// Dense state-space model of a linear time-invariant system.
///
/// ```text
/// continuous:  x'(t)  = A x(t) + B u(t),    y(t) = C x(t) + D u(t)
/// discrete:    x[k+1] = A x[k] + B u[k],    y[k] = C x[k] + D u[k]
/// ```
///
/// `x` is the state vector of length `n`, `u` the input of length `m` and `y`
/// the output of length `p`. `A` is `n × n`, `B` is `n × m`, `C` is `p × n` and
/// `D` is `p × m`. All entries are finite. Matrices are stored row-major and may
/// be real or complex. The [`LtiTime`] value selects the domain.
///
/// The model computes frequency and time responses, poles, controllability and
/// observability, discretization, pole placement, and conversion of one
/// input–output channel to a transfer function or pole-zero form.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_linalg::Matrix;
/// use scioxide_signal::{LtiTime, StateSpace};
/// let a = Matrix::from_vec(1, 1, vec![0.5_f64])?;
/// let b = Matrix::from_vec(1, 1, vec![1.0])?;
/// let d = Matrix::zeros(1, 1)?;
/// let time = LtiTime::Discrete { sample_period: 0.1 };
/// let model = StateSpace::new(time, a.view(), b.view(), b.view(), d.view())?;
/// assert_eq!(model.states(), 1);
/// assert_eq!(model.time().sample_period(), Some(0.1));
/// // C needs one column per state.
/// let wide = Matrix::from_vec(1, 2, vec![1.0, 1.0])?;
/// let mismatch = StateSpace::new(time, a.view(), b.view(), wide.view(), d.view());
/// let kind = mismatch.unwrap_err().kind();
/// assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct StateSpace<T: NumericScalar> {
    time: LtiTime<T::Real>,
    a: Matrix<T>,
    b: Matrix<T>,
    c: Matrix<T>,
    d: Matrix<T>,
}

impl<T: NumericScalar> StateSpace<T> {
    /// Creates a model from copies of `A`, `B`, `C` and `D` after checking them.
    ///
    /// Any of `n`, `m` and `p` may be zero.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` is not square or the shapes of `b`, `c` and
    ///   `d` do not match `n × m`, `p × n` and `p × m`.
    /// - `NonFinite` if a matrix entry or the sample period is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if a discrete sample period is not positive.
    /// - `AllocationFailure` if the copies cannot be stored.
    pub fn new(
        time: LtiTime<T::Real>,
        a: MatrixView<'_, T>,
        b: MatrixView<'_, T>,
        c: MatrixView<'_, T>,
        d: MatrixView<'_, T>,
    ) -> Result<Self, SciOxideError> {
        validate_model(time, a, b, c, d)?;
        Ok(Self {
            time,
            a: materialize(a)?,
            b: materialize(b)?,
            c: materialize(c)?,
            d: materialize(d)?,
        })
    }

    /// Returns a view of the model matrices without copying them.
    pub fn view(&self) -> StateSpaceView<'_, T> {
        StateSpaceView {
            time: self.time,
            a: self.a.view(),
            b: self.b.view(),
            c: self.c.view(),
            d: self.d.view(),
        }
    }
    /// Returns the time domain.
    pub const fn time(&self) -> LtiTime<T::Real> {
        self.time
    }
    /// Returns the state matrix `A`, of shape `n × n`.
    pub const fn a(&self) -> &Matrix<T> {
        &self.a
    }
    /// Returns the input matrix `B`, of shape `n × m`.
    pub const fn b(&self) -> &Matrix<T> {
        &self.b
    }
    /// Returns the output matrix `C`, of shape `p × n`.
    pub const fn c(&self) -> &Matrix<T> {
        &self.c
    }
    /// Returns the feedthrough matrix `D`, of shape `p × m`.
    pub const fn d(&self) -> &Matrix<T> {
        &self.d
    }
    /// Returns the number of states `n`.
    pub const fn states(&self) -> usize {
        self.a.rows()
    }
    /// Returns the number of inputs `m`.
    pub const fn inputs(&self) -> usize {
        self.b.cols()
    }
    /// Returns the number of outputs `p`.
    pub const fn outputs(&self) -> usize {
        self.c.rows()
    }
}

/// Frequency response `H` of a state-space model on a frequency grid, from
/// [`StateSpace::frequency_response`].
///
/// For each frequency, `H` is a `p × m` complex matrix, where `p` is the number
/// of outputs and `m` the number of inputs.
#[derive(Clone, Debug)]
pub struct FrequencyResponse<S: RealScalar> {
    frequencies: Vec<S>,
    values: Vec<Complex<S>>,
    outputs: usize,
    inputs: usize,
}

impl<S: RealScalar> FrequencyResponse<S> {
    /// Returns the frequencies, in the units of the [crate conventions](crate#conventions):
    /// radians per time unit for continuous models and multiples of Nyquist for
    /// discrete models.
    pub fn frequencies(&self) -> &[S] {
        &self.frequencies
    }
    /// Returns the response values in `[frequency][output][input]` order: entry
    /// `(k, i, j)` is at index `(k p + i) m + j`.
    pub fn values(&self) -> &[Complex<S>] {
        &self.values
    }
    /// Returns the number of outputs `p`.
    pub const fn outputs(&self) -> usize {
        self.outputs
    }
    /// Returns the number of inputs `m`.
    pub const fn inputs(&self) -> usize {
        self.inputs
    }
}

/// Sampled output and state trajectories of a state-space model.
///
/// [`StateSpace::impulse_response`] and [`StateSpace::step_response`] return one
/// response channel per input, each driven by that input alone.
/// [`StateSpace::simulate_zoh`] returns a single channel.
#[derive(Clone, Debug)]
pub struct TimeResponse<T: NumericScalar> {
    times: Vec<T::Real>,
    values: Vec<T>,
    states: Vec<T>,
    outputs: usize,
    channels: usize,
    state_count: usize,
}

impl<T: NumericScalar> TimeResponse<T> {
    /// Returns the sample times.
    pub fn times(&self) -> &[T::Real] {
        &self.times
    }
    /// Returns the outputs in `[sample][output][channel]` order.
    pub fn values(&self) -> &[T] {
        &self.values
    }
    /// Returns the states in `[sample][state][channel]` order.
    pub fn states(&self) -> &[T] {
        &self.states
    }
    /// Returns the number of outputs.
    pub const fn outputs(&self) -> usize {
        self.outputs
    }
    /// Returns the number of response channels.
    pub const fn channels(&self) -> usize {
        self.channels
    }
    /// Returns the number of states.
    pub const fn state_count(&self) -> usize {
        self.state_count
    }
}

fn validate_model<T: NumericScalar>(
    time: LtiTime<T::Real>,
    a: MatrixView<'_, T>,
    b: MatrixView<'_, T>,
    c: MatrixView<'_, T>,
    d: MatrixView<'_, T>,
) -> Result<(), SciOxideError> {
    time.validate(scioxide_core::argument!("time"))?;
    let n = a.rows();
    if a.cols() != n {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(n, n),
            scioxide_core::Dimensions::two(a.rows(), a.cols()),
        ));
    }
    if b.rows() != n {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(n, b.cols()),
            scioxide_core::Dimensions::two(b.rows(), b.cols()),
        ));
    }
    if c.cols() != n {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(c.rows(), n),
            scioxide_core::Dimensions::two(c.rows(), c.cols()),
        ));
    }
    if d.rows() != c.rows() || d.cols() != b.cols() {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(c.rows(), b.cols()),
            scioxide_core::Dimensions::two(d.rows(), d.cols()),
        ));
    }
    let _ = a.try_finite()?;
    let _ = b.try_finite()?;
    let _ = c.try_finite()?;
    let _ = d.try_finite()?;
    Ok(())
}

fn materialize<T: NumericScalar>(view: MatrixView<'_, T>) -> Result<Matrix<T>, SciOxideError> {
    let mut matrix = Matrix::zeros(view.rows(), view.cols())?;
    for row in 0..view.rows() {
        for col in 0..view.cols() {
            matrix.as_mut_slice()[row * view.cols() + col] = view.get(row, col)?;
        }
    }
    Ok(matrix)
}

fn complex_matrix<T: NumericScalar>(
    view: MatrixView<'_, T>,
) -> Result<Matrix<Complex<T::Real>>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let mut matrix = Matrix::zeros(view.rows(), view.cols())?;
    for row in 0..view.rows() {
        for col in 0..view.cols() {
            matrix.as_mut_slice()[row * view.cols() + col] = {
                let value = view.get(row, col)?;
                Complex::new(value.real_component(), value.imaginary_component())
            };
        }
    }
    Ok(matrix)
}

fn controllability_matrix<T: NumericScalar>(
    a: &Matrix<T>,
    b: &Matrix<T>,
) -> Result<Matrix<T>, SciOxideError> {
    let n = a.rows();
    let m = b.cols();
    let mut output = Matrix::zeros(
        n,
        n.checked_mul(m).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("controllability_matrix"))
        })?,
    )?;
    let mut block = b.clone();
    for power in 0..n {
        for row in 0..n {
            let destination = row * n * m + power * m;
            output.as_mut_slice()[destination..destination + m]
                .copy_from_slice(&block.as_slice()[row * m..(row + 1) * m]);
        }
        if power + 1 < n {
            block = a.view().mul_mat(&block)?;
        }
    }
    Ok(output)
}

/// Checks the `times` grid.
///
/// The step is `h = times[1]`, the step the responses use. Every sample must
/// satisfy `|times[k] - k h| <= 64 epsilon (|times[k]| + h)`, where `k h` is
/// the rounded product. Grids computed as `k * h` or `k * T / (n - 1)` carry
/// rounding of a few units of `epsilon |times[k]|`, so the tolerance grows with
/// `|times[k]|` and long grids pass.
fn validate_times<S: RealScalar>(times: &[S], time: LtiTime<S>) -> Result<(), SciOxideError> {
    let argument = scioxide_core::argument!("times");
    crate::util::check_real_slice_finite(times)?;
    if times.is_empty() {
        return Err(empty(argument, "time grid must be nonempty"));
    }
    if times[0] != S::ZERO {
        return Err(invalid(argument, "time grid must start at zero"));
    }
    if times.len() == 1 {
        return Ok(());
    }
    let step = times[1] - times[0];
    if step <= S::ZERO {
        return Err(invalid(argument, "time grid must be strictly increasing"));
    }
    let tolerance = S::EPSILON * S::try_from_f64_lossy(64.0)?;
    for (k, &sample) in times.iter().enumerate().skip(2) {
        let expected = crate::util::scalar_from_usize_rounded::<S>(k)? * step;
        if (sample - expected).abs() > tolerance * (sample.abs() + step) {
            return Err(invalid(argument, "time grid must be uniformly spaced"));
        }
    }
    if let LtiTime::Discrete { sample_period } = time
        && (step - sample_period).abs() > tolerance * (S::ONE + sample_period.abs())
    {
        return Err(inconsistent(
            argument,
            "time-grid spacing must equal the discrete sample period",
        ));
    }
    Ok(())
}

fn dimension(expected: usize, actual: usize) -> SciOxideError {
    SciOxideError::dimension_mismatch(
        scioxide_core::Dimensions::one(expected),
        scioxide_core::Dimensions::one(actual),
    )
}

fn numerical(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::LossOfSignificance, detail)
}
