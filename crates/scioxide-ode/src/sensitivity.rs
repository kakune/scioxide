//! Forward sensitivities: the ODE augmented with its variational equations.
use crate::{
    OdeSystem,
    safety::{finite, invalid},
};
use alloc::vec::Vec;
use core::cell::RefCell;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_reserve};
use scioxide_core::{
    Capacity, ErrorTolerance, NormSelection, NumericScalar, RealScalar, SciOxideError, SolveContext,
};
use scioxide_linalg::{MatrixView, MatrixViewMut};

/// Defines a parametrized ODE `y′ = f(t, y, p)` and its directional derivatives for
/// [`ForwardSensitivitySystem`].
///
/// Parameters are real; states may be real or complex. Directional derivatives are
/// taken along real paths, so a right-hand side that is not holomorphic can supply
/// real-linear products. Every callback writes every output element, uses at most
/// [`Self::workspace_len`] scratch scalars, and runs nested work in `cx`.
pub trait OdeSensitivitySystem<S: RealScalar, T: NumericScalar<Real = S> = S> {
    /// Returns the number of states `n`.
    fn dimension(&self) -> usize;
    /// Returns the number of real parameters `m`.
    fn parameter_dimension(&self) -> usize;
    /// Returns the number of scratch scalars one callback needs.
    ///
    /// # Errors
    ///
    /// Errors you return are returned unchanged by the plan methods that query this.
    fn workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    /// Evaluates `f(time, state, parameters)` into `output`.
    ///
    /// # Errors
    ///
    /// Errors you return end the integration and are returned unchanged.
    fn rhs(
        &self,
        time: S,
        state: &[T],
        parameters: &[S],
        output: &mut [T],
        scratch: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Evaluates the directional derivative
    /// `(∂f/∂y) state_direction + (∂f/∂p) parameter_direction` into `output`.
    ///
    /// The product can come from a dense or sparse Jacobian, a matrix-free operator or
    /// forward-mode automatic differentiation.
    ///
    /// # Errors
    ///
    /// Errors you return end the integration and are returned unchanged.
    #[allow(clippy::too_many_arguments)]
    fn tangent_rhs(
        &self,
        time: S,
        state: &[T],
        parameters: &[S],
        state_direction: &[T],
        parameter_direction: &[S],
        output: &mut [T],
        scratch: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}
/// Describes the storage that [`ForwardSensitivitySystem::bind`] needs, returned by
/// [`ForwardSensitivityRequirements::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardSensitivityRequirements {
    directions: usize,
    scratch: usize,
}

impl ForwardSensitivityRequirements {
    /// Returns the number of stored parameter-direction values, `m k`.
    pub const fn packed_directions(&self) -> usize {
        self.directions
    }
    /// Returns the scratch length, [`OdeSensitivitySystem::workspace_len`].
    pub const fn provider_scratch(&self) -> usize {
        self.scratch
    }
}

/// Holds the parameter directions and callback scratch of a
/// [`ForwardSensitivitySystem`].
///
/// The bound system borrows the workspace for its lifetime. A workspace can serve
/// successive bindings and keeps the largest size it has reached.
pub struct ForwardSensitivityWorkspace<S, T = S> {
    directions: Vec<S>,
    scratch: Vec<T>,
}

impl<S: RealScalar, T: NumericScalar<Real = S>> Default for ForwardSensitivityWorkspace<S, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, T> core::fmt::Debug for ForwardSensitivityWorkspace<S, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ForwardSensitivityWorkspace")
            .field("packed_directions", &self.directions.len())
            .field("provider_scratch", &self.scratch.len())
            .finish()
    }
}

impl<S: RealScalar, T: NumericScalar<Real = S>> ForwardSensitivityWorkspace<S, T> {
    /// Creates an empty workspace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            directions: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// Does not fail. The `Result` follows the shared
    /// [workspace protocol](scioxide_core#plans-and-workspaces).
    pub fn check(
        &self,
        requirements: &ForwardSensitivityRequirements,
    ) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.directions.len() >= requirements.directions
                && self.scratch.len() >= requirements.scratch
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Grows the workspace to meet `requirements`.
    ///
    /// After a failure the workspace may keep part of the growth.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` if memory cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &ForwardSensitivityRequirements,
    ) -> Result<(), SciOxideError> {
        if self.directions.len() < requirements.directions {
            let additional = requirements.directions - self.directions.len();
            try_reserve(
                &mut self.directions,
                additional,
                scioxide_core::quantity!("parameter_directions"),
            )?;
            self.directions.resize(requirements.directions, S::ZERO);
        }
        if self.scratch.len() < requirements.scratch {
            let additional = requirements.scratch - self.scratch.len();
            try_reserve(
                &mut self.scratch,
                additional,
                scioxide_core::quantity!("sensitivity_scratch"),
            )?;
            self.scratch.resize(requirements.scratch, T::ZERO);
        }
        Ok(())
    }
}

/// Borrows or stores call-local execution parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SensitivityLayout {
    states: usize,
    parameters: usize,
    directions: usize,
    augmented: usize,
}
impl SensitivityLayout {
    /// Creates a plan for `states` states, `parameters` real parameters and
    /// `directions` sensitivity directions.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `states` or `directions` is zero.
    /// - `SizeOverflow` if `n (k + 1)` or `m k` overflows `usize`.
    fn new(states: usize, parameters: usize, directions: usize) -> Result<Self, SciOxideError> {
        if states == 0 || directions == 0 {
            return Err(invalid(
                if states == 0 {
                    scioxide_core::argument!("provider")
                } else {
                    scioxide_core::argument!("parameter_directions")
                },
                "sensitivity states and directions must be positive",
            ));
        }
        checked_mul(
            parameters,
            directions,
            scioxide_core::quantity!("parameter_directions"),
        )?;
        Ok(Self {
            states,
            parameters,
            directions,
            augmented: checked_mul(
                states,
                checked_add(
                    directions,
                    1,
                    scioxide_core::quantity!("sensitivity_blocks"),
                )?,
                scioxide_core::quantity!("sensitivity_dimension"),
            )?,
        })
    }
}

/// Evaluates the right-hand side `[f, (∂f/∂y) s₁ + (∂f/∂p) d₁, …]` of the
/// sensitivity equations, as returned by [`ForwardSensitivitySystem::bind`].
///
/// Here `n` is the state dimension, `m` is the number of real parameters `p`,
/// and `k` is the number of parameter directions `dⱼ` in `Rᵐ`. Each `sⱼ` is the
/// state derivative in direction `dⱼ`. One evaluation calls `f` once and the
/// directional-derivative callback `k` times, writing `N = n (k + 1)` scalars.
/// Accuracy follows the chosen ODE method and augmented error tolerance.
///
/// It is an ordinary [`OdeSystem`] of dimension `N`, and every ODE method
/// integrates it. The sensitivities hold at fixed time and are not corrected for
/// event times or resets; at a terminal event, convert them with
/// [`ForwardSensitivitySystem::event_sensitivities_into`]. Each directional derivative
/// counts as one linear-operator evaluation of the solve.
///
/// The system has no analytic Jacobian, so implicit methods (BDF, Radau IIA, backward
/// Euler and the other stiff methods) build one by forward differences. Each dense
/// Jacobian costs about `N` evaluations of the augmented right-hand side and stores
/// `N²` entries; the Krylov stage backend uses directional differences instead.
pub struct ForwardSensitivitySystem<'a, S: RealScalar, T: NumericScalar<Real = S> = S> {
    layout: SensitivityLayout,
    provider: &'a dyn OdeSensitivitySystem<S, T>,
    parameters: &'a [S],
    directions: &'a [S],
    scratch: RefCell<&'a mut [T]>,
}

impl<S: RealScalar, T: NumericScalar<Real = S>> OdeSystem<S, T>
    for ForwardSensitivitySystem<'_, S, T>
{
    fn dimension(&self) -> usize {
        self.layout.augmented
    }
    /// Evaluates `f` and the `k` directional derivatives.
    ///
    /// The integrator counts the call as one right-hand-side evaluation, and each
    /// directional derivative counts as one `LinearOperator` evaluation in `cx`.
    fn rhs(
        &self,
        time: S,
        state: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if state.len() != self.dimension() || output.len() != self.dimension() {
            return Err(invalid(
                if state.len() != self.dimension() {
                    scioxide_core::argument!("state")
                } else {
                    scioxide_core::argument!("output")
                },
                "augmented sensitivity dimension mismatch",
            ));
        }
        finite(state, scioxide_core::NonFiniteOrigin::Intermediate)?;
        let mut scratch = self.scratch.borrow_mut();
        let n = self.layout.states;
        let p = self.layout.parameters;
        output.fill(T::from_real(S::NAN));
        self.provider.rhs(
            time,
            &state[..n],
            self.parameters,
            &mut output[..n],
            &mut scratch,
            cx,
        )?;
        finite(&output[..n], scioxide_core::NonFiniteOrigin::Callback)?;
        for direction in 0..self.layout.directions {
            let start = (direction + 1) * n;
            crate::termination::admit(cx, scioxide_core::EvaluationRole::LinearOperator)?;
            self.provider.tangent_rhs(
                time,
                &state[..n],
                self.parameters,
                &state[start..start + n],
                &self.directions[direction * p..(direction + 1) * p],
                &mut output[start..start + n],
                &mut scratch,
                cx,
            )?;
            finite(
                &output[start..start + n],
                scioxide_core::NonFiniteOrigin::Callback,
            )?;
        }
        Ok(())
    }
}
/// Says whether [`ForwardSensitivitySystem::event_sensitivities_into`] could compute the
/// derivatives of a terminal event.
///
/// When they are unavailable, the integration itself is still valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EventSensitivityStatus {
    /// The event is isolated and transverse, and the derivatives were written.
    Available,
    /// The event slope `|dg/dt|` is not above the threshold `minimum_slope`.
    NonTransverse,
    /// The caller passed [`EventIsolation::Ambiguous`] in
    /// [`EventSensitivityInput::isolation`], for example because the root is at the
    /// initial time or several roots coincide. This is checked before transversality.
    Ambiguous,
}

/// Says whether the root of a terminal event is isolated from other roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventIsolation {
    /// The root is the only one nearby.
    Isolated,
    /// The root is not known to be isolated.
    Ambiguous,
}
/// Holds the input of [`ForwardSensitivitySystem::event_sensitivities_into`] for an
/// event function `g(t, y, p)` at its root `t*`.
///
/// For direction `j`, let `sⱼ` be the fixed-time state sensitivity and `dⱼ` the
/// parameter direction. The fields are evaluated at `t*`.
pub struct EventSensitivityInput<'a, S: RealScalar, T: NumericScalar<Real = S> = S> {
    /// The `[n, k]` matrix whose column `j` is `sⱼ(t*)`.
    pub fixed_time: MatrixView<'a, T>,
    /// The state derivative `f(t*, y(t*), p)`.
    pub state_slope: &'a [T],
    /// The fixed-time event derivatives `(∂g/∂y) sⱼ + (∂g/∂p) dⱼ`, one per direction.
    pub numerators: &'a [S],
    /// The total time derivative `dg/dt = ∂g/∂t + (∂g/∂y) f` of the event function.
    pub event_slope: S,
    /// The nonnegative threshold: the event is transverse if
    /// `|event_slope| > minimum_slope`.
    pub minimum_slope: S,
    /// Whether the root is isolated.
    pub isolation: EventIsolation,
}

impl ForwardSensitivityRequirements {
    /// Computes storage for the provider and number of parameter directions.
    /// # Errors
    /// Rejects empty state/direction shapes, overflow and provider scratch errors.
    pub fn new<S: RealScalar, T: NumericScalar<Real = S>>(
        provider: &dyn OdeSensitivitySystem<S, T>,
        directions: usize,
    ) -> Result<Self, SciOxideError> {
        let layout = SensitivityLayout::new(
            provider.dimension(),
            provider.parameter_dimension(),
            directions,
        )?;
        Ok(Self {
            directions: layout.parameters * layout.directions,
            scratch: provider.workspace_len()?,
        })
    }
}

impl<'a, S: RealScalar, T: NumericScalar<Real = S>> ForwardSensitivitySystem<'a, S, T> {
    /// Binds `provider` at `parameters` with one parameter direction per matrix row.
    ///
    /// Derives `n` and `m` from the provider and `k` from the matrix rows. Copies
    /// the `k × m` direction entries into the borrowed workspace in `O(k m)` work,
    /// growing its direction and provider scratch storage only when necessary.
    ///
    /// # Errors
    /// Returns `InvalidInput` naming `provider` for zero states, `parameters` for
    /// a wrong length, or `parameter_directions` for zero rows or wrong columns.
    /// Also returns `NonFinite`, size overflow, allocation and provider scratch errors.
    ///
    /// # References
    /// Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I*, §I.14.
    pub fn bind(
        provider: &'a dyn OdeSensitivitySystem<S, T>,
        parameters: &'a [S],
        parameter_directions: MatrixView<'_, S>,
        workspace: &'a mut ForwardSensitivityWorkspace<S, T>,
    ) -> Result<ForwardSensitivitySystem<'a, S, T>, SciOxideError> {
        let layout = SensitivityLayout::new(
            provider.dimension(),
            provider.parameter_dimension(),
            parameter_directions.rows(),
        )?;
        if parameters.len() != layout.parameters {
            return Err(invalid(
                scioxide_core::argument!("parameters"),
                "parameter length must match the provider",
            ));
        }
        if parameter_directions.cols() != layout.parameters {
            return Err(invalid(
                scioxide_core::argument!("parameter_directions"),
                "direction columns must match the parameter count",
            ));
        }
        finite(parameters, scioxide_core::NonFiniteOrigin::Input)?;
        let requirements = ForwardSensitivityRequirements::new(provider, layout.directions)?;
        if workspace.check(&requirements)? == Capacity::NeedsGrowth {
            workspace.try_reserve(&requirements)?;
        }
        let ForwardSensitivityWorkspace {
            directions,
            scratch,
        } = workspace;
        let directions = &mut directions[..requirements.directions];
        for row in 0..layout.directions {
            for col in 0..layout.parameters {
                let value = parameter_directions.get(row, col)?;
                finite(&[value], scioxide_core::NonFiniteOrigin::Input)?;
                directions[row * layout.parameters + col] = value;
            }
        }
        Ok(ForwardSensitivitySystem {
            layout,
            provider,
            parameters,
            directions,
            scratch: RefCell::new(&mut scratch[..requirements.scratch]),
        })
    }
    /// Returns the dimension `n (k + 1)` of the augmented system.
    pub const fn augmented_dimension(&self) -> usize {
        self.layout.augmented
    }

    /// Writes the augmented initial state `[y₀, s⁰₁, …, s⁰_k]` to `output`, from the
    /// initial state `state` and the `[n, k]` matrix `tangents` whose column `j` is
    /// `s⁰ⱼ`.
    ///
    /// Zero columns give pure parameter sensitivities; a unit column `eᵢ` gives the
    /// sensitivity to the initial value `y₀ᵢ`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `state`, `tangents` or `output` does not have the shape
    ///   `[n]`, `[n, k]` or `[n (k + 1)]`.
    /// - `NonFinite` (origin `Input`) if `state` or `tangents` contains NaN or
    ///   infinity.
    pub fn initialize_into(
        &self,
        state: &[T],
        tangents: MatrixView<'_, T>,
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        if state.len() != self.layout.states
            || tangents.rows() != self.layout.states
            || tangents.cols() != self.layout.directions
            || output.len() != self.layout.augmented
        {
            return Err(invalid(
                if state.len() != self.layout.states {
                    scioxide_core::argument!("state")
                } else if output.len() != self.layout.augmented {
                    scioxide_core::argument!("output")
                } else {
                    scioxide_core::argument!("tangents")
                },
                "sensitivity initial shapes must be [n], [n,k], and [n*(k+1)]",
            ));
        }
        finite(state, scioxide_core::NonFiniteOrigin::Input)?;
        for row in 0..self.layout.states {
            for col in 0..self.layout.directions {
                finite(
                    &[tangents.get(row, col)?],
                    scioxide_core::NonFiniteOrigin::Input,
                )?;
            }
        }
        output[..self.layout.states].copy_from_slice(state);
        for col in 0..self.layout.directions {
            for row in 0..self.layout.states {
                output[(col + 1) * self.layout.states + row] = tangents.get(row, col)?;
            }
        }
        Ok(())
    }

    /// Splits the augmented state `augmented` into the state `y`, written to `state`,
    /// and the sensitivities `s₁, …, s_k`, written as the columns of the `[n, k]` view
    /// `tangents`.
    ///
    /// Elements of a strided view outside the `[n, k]` matrix are not written.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `augmented`, `state` or `tangents` does not have the shape
    ///   `[n (k + 1)]`, `[n]` or `[n, k]`.
    /// - `NonFinite` (origin `Input`) if `augmented` contains NaN or infinity.
    pub fn extract_into(
        &self,
        augmented: &[T],
        state: &mut [T],
        mut tangents: MatrixViewMut<'_, T>,
    ) -> Result<(), SciOxideError> {
        if augmented.len() != self.layout.augmented
            || state.len() != self.layout.states
            || tangents.rows() != self.layout.states
            || tangents.cols() != self.layout.directions
        {
            return Err(invalid(
                if augmented.len() != self.layout.augmented {
                    scioxide_core::argument!("augmented")
                } else if state.len() != self.layout.states {
                    scioxide_core::argument!("state")
                } else {
                    scioxide_core::argument!("tangents")
                },
                "sensitivity output shapes must match the plan",
            ));
        }
        finite(augmented, scioxide_core::NonFiniteOrigin::Input)?;
        state.copy_from_slice(&augmented[..self.layout.states]);
        for col in 0..self.layout.directions {
            for row in 0..self.layout.states {
                tangents.set(row, col, augmented[(col + 1) * self.layout.states + row])?;
            }
        }
        Ok(())
    }

    /// Returns an error tolerance for the augmented system, with absolute tolerance
    /// `primal_absolute` on `y`, `sensitivity_absolute` on every sensitivity, and the
    /// common relative tolerance `relative`.
    ///
    /// Pass it to [`OdeOptions::with_error_tolerance`](crate::OdeOptions::with_error_tolerance).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if a tolerance is negative.
    /// - `NonFinite` if a tolerance is NaN or infinite.
    /// - `AllocationFailure` if the `n (k + 1)` absolute tolerances cannot be stored.
    pub fn error_tolerance(
        &self,
        primal_absolute: S,
        sensitivity_absolute: S,
        relative: S,
        norm: NormSelection,
    ) -> Result<ErrorTolerance<S>, SciOxideError> {
        let mut absolute = try_filled(
            self.layout.augmented,
            sensitivity_absolute,
            scioxide_core::quantity!("sensitivity_tolerances"),
        )?;
        absolute[..self.layout.states].fill(primal_absolute);
        ErrorTolerance::try_vector(&absolute, relative, norm)
    }
    /// Converts fixed-time sensitivities at an event into derivatives of the event time
    /// and of the state at the event.
    ///
    /// For each direction `j` it writes, with the notation of [`EventSensitivityInput`],
    ///
    /// ```text
    /// τⱼ = −numeratorⱼ / event_slope,      σⱼ = sⱼ + f τⱼ,
    /// ```
    ///
    /// `τⱼ` to `time_output[j]` and `σⱼ` to column `j` of `state_output`. This is the
    /// derivative of the implicit equation `g(t*, y(t*), p) = 0`. It costs `O(n k)`
    /// operations. The result is sensitive to a small event slope: its condition
    /// number grows like `1/|event_slope|`, and the threshold only excludes slopes at
    /// or below `minimum_slope`. State derivatives retain the unrounded time
    /// derivative through the correction, even when `τⱼ` rounds to zero.
    ///
    /// If the event is not isolated or not transverse, the method returns
    /// [`EventSensitivityStatus::Ambiguous`] or
    /// [`EventSensitivityStatus::NonTransverse`] and writes nothing.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a shape does not match the plan, or if `minimum_slope` is
    ///   negative.
    /// - `NonFinite` (origin `Input`) if an input value is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a result overflows.
    ///
    /// # References
    ///
    /// E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
    /// Equations I*, 2nd ed., Springer, 1993, chapter II.
    pub fn event_sensitivities_into(
        &self,
        input: EventSensitivityInput<'_, S, T>,
        time_output: &mut [S],
        mut state_output: MatrixViewMut<'_, T>,
    ) -> Result<EventSensitivityStatus, SciOxideError> {
        let EventSensitivityInput {
            fixed_time,
            state_slope: slope,
            numerators,
            event_slope,
            minimum_slope,
            isolation,
        } = input;
        let isolated = isolation == EventIsolation::Isolated;

        let n = self.layout.states;
        let k = self.layout.directions;
        if fixed_time.rows() != n
            || fixed_time.cols() != k
            || slope.len() != n
            || numerators.len() != k
            || time_output.len() != k
            || state_output.rows() != n
            || state_output.cols() != k
        {
            return Err(invalid(
                if fixed_time.rows() != n || fixed_time.cols() != k {
                    scioxide_core::argument!("fixed_time")
                } else if slope.len() != n {
                    scioxide_core::argument!("state_slope")
                } else if numerators.len() != k {
                    scioxide_core::argument!("numerators")
                } else if time_output.len() != k {
                    scioxide_core::argument!("time_output")
                } else {
                    scioxide_core::argument!("state_output")
                },
                "event sensitivity shapes must match the plan's state and direction counts",
            ));
        }
        finite(slope, scioxide_core::NonFiniteOrigin::Input)?;
        finite(numerators, scioxide_core::NonFiniteOrigin::Input)?;
        finite(
            &[event_slope, minimum_slope],
            scioxide_core::NonFiniteOrigin::Input,
        )?;
        if minimum_slope < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("minimum_slope"),
                "transversality threshold must be nonnegative",
            ));
        }
        for row in 0..n {
            for col in 0..k {
                finite(
                    &[fixed_time.get(row, col)?],
                    scioxide_core::NonFiniteOrigin::Input,
                )?;
            }
        }
        if !isolated {
            return Ok(EventSensitivityStatus::Ambiguous);
        }
        if event_slope.abs() <= minimum_slope {
            return Ok(EventSensitivityStatus::NonTransverse);
        }
        // Validate every calculation before publishing either view, without scratch.
        for publish in [false, true] {
            for col in 0..k {
                let dt = scioxide_core::checked_float_quotient(-numerators[col], event_slope)?;
                for (row, &f) in slope.iter().enumerate() {
                    let tangent = fixed_time.get(row, col)?;
                    let component = |tangent, slope| {
                        let mut sum = scioxide_core::RangeAccumulator::new();
                        sum.add_product(tangent, event_slope)?;
                        sum.subtract_product(slope, numerators[col])?;
                        sum.finish_quotient(event_slope)
                    };
                    let value = T::from_components(
                        component(tangent.real_component(), f.real_component())?,
                        if T::IS_REAL {
                            S::ZERO
                        } else {
                            component(tangent.imaginary_component(), f.imaginary_component())?
                        },
                    );
                    if publish {
                        state_output.set(row, col, value)?;
                    }
                }
                if publish {
                    time_output[col] = dt;
                }
            }
        }
        Ok(EventSensitivityStatus::Available)
    }
}
