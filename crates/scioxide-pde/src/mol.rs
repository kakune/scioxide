//! The method-of-lines adapter: diagonal mass, field scaling, second-order
//! dynamics, field reconstruction and integrator-aware explicit step limits.
use crate::validation::finite_slice;
use crate::validation::invalid;
use crate::validation::shape;

use alloc::vec::Vec;
use core::cell::{Cell, RefCell, RefMut};
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{AlgebraicScalar, Scaled};
use scioxide_core::{
    ErrorTolerance, EvaluationRole, ExecutionContext, NumericScalar, RealScalar, SciOxideError,
    SolveContext, StructuralPattern, StructuralPatternView,
};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::{OdeLinearStability, OdeSystem};

/// Safety factor applied to the integrator's stability segment.
const SAFETY: f64 = 0.9;

/// Stability restriction for explicit time integration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CflPolicy<S: RealScalar> {
    /// Derive the limit from the mass-scaled Jacobian and the integrator's
    /// declared stability segment; unconditionally stable integrators get none.
    #[default]
    Automatic,
    /// Use only the ODE error controller and caller ODE step bounds.
    Disabled,
    /// Apply this positive finite maximum step magnitude instead of estimating it.
    MaximumStep(S),
}
impl<S: RealScalar> CflPolicy<S> {
    /// Validate; `argument` names the policy source.
    pub(crate) fn validate(self, argument: scioxide_core::Argument) -> Result<(), SciOxideError> {
        if let Self::MaximumStep(h) = self
            && (!h.is_finite() || h <= S::ZERO)
        {
            return Err(invalid(
                argument,
                "CFL maximum step must be positive and finite",
            ));
        }
        Ok(())
    }
}

/// Order of the semidiscrete dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MethodOfLinesOrder {
    /// `M y' = f(t, y)`.
    #[default]
    First,
    /// Undamped `M u'' = f(t, u)`, integrated as `[u, v]` with `u' = v`.
    Second,
}

/// Physical fields reconstructed from the free unknowns of a provider.
///
/// Providers that eliminate constrained unknowns (such as
/// [`crate::SemidiscreteSystem`]) restore them here; the identity applies otherwise.
pub trait MethodOfLinesFields<S: RealScalar, T: NumericScalar<Real = S> = S> {
    /// Full physical field length.
    fn field_len(&self) -> usize;
    /// Write the full physical field at `time` from free physical values.
    /// # Errors
    /// Rejects shapes and propagates constraint-data failures.
    fn reconstruct_into(&self, time: S, free: &[T], full: &mut [T]) -> Result<(), SciOxideError>;
}

/// Native component scaling; replay the original factors after range loss.
#[inline]
fn scale<T: NumericScalar>(
    value: T,
    numerator: T::Real,
    mass: T::Real,
    field: T::Real,
) -> Result<T, SciOxideError> {
    let part = |mut v: T::Real| {
        if numerator != T::Real::ONE {
            let next = v * numerator;
            if !next.is_finite() || (v != T::Real::ZERO && !next.is_normal()) {
                return None;
            }
            v = next;
        }
        for divisor in [mass, field] {
            if divisor != T::Real::ONE {
                let next = v / divisor;
                if !next.is_finite() || (v != T::Real::ZERO && !next.is_normal()) {
                    return None;
                }
                v = next;
            }
        }
        v.is_finite().then_some(v)
    };
    if let Some(real) = part(value.real_component()) {
        if T::IS_REAL {
            return Ok(T::from_real(real));
        }
        if let Some(imaginary) = part(value.imaginary_component()) {
            return Ok(T::from_components(real, imaginary));
        }
    }
    scale_fallback(value, numerator, mass, field)
}

#[cold]
fn scale_fallback<T: NumericScalar>(
    value: T,
    numerator: T::Real,
    mass: T::Real,
    field: T::Real,
) -> Result<T, SciOxideError> {
    let part = |v| {
        Scaled::from_value(v)?
            .mul(Scaled::from_value(numerator)?)?
            .div(Scaled::from_value(mass)?)?
            .div(Scaled::from_value(field)?)?
            .round()
    };
    Ok(T::from_components(
        part(value.real_component())?,
        part(value.imaginary_component())?,
    ))
}

/// Fixed normalization for `M y' = f(t,y)` (or `M u'' = f(t,u)`) with physical
/// state `y = D z`.
///
/// `M` and `D` are positive real diagonal matrices, in provider-component order.
/// The provider describes the physical residual and its derivatives using
/// `OdeSystem`; it must declare identity ODE mass, since this adapter owns mass
/// normalization. Provider step limits are not forwarded, because they are
/// expressed for identity mass. Under [`CflPolicy::Automatic`] the adapter bounds
/// the spectrum of the normalized Jacobian `M⁻¹D⁻¹JD` by its Gershgorin row sums
/// `ρ` and limits steps to `0.9·real/ρ` (first order) or `0.9·imaginary/√ρ`
/// (second order) using the integrator's [`OdeLinearStability`]. These bounds
/// assume a real nonpositive spectrum for first-order dynamics (symmetric
/// diffusion with diagonal mass) and a purely imaginary one for second-order
/// dynamics. All four core real/complex scalar domains are supported.
/// Prepared normalizers and native component operations use ordinary rounding;
/// range loss replays the original mass and field factors in extended range.
/// Prepared RHS and Jacobian products allocate nothing and commit only on success.
///
/// ```
/// use scioxide_pde::{CflPolicy, MethodOfLinesOrder, MethodOfLinesPlan};
/// use scioxide_ode::{DormandPrince5, OdeMethod};
/// let plan = MethodOfLinesPlan::new(2, Some(&[0.5, 0.25]), None)?
///     .with_order(MethodOfLinesOrder::Second)
///     .with_step_limit(CflPolicy::MaximumStep(0.1), DormandPrince5::<f64>::new().linear_stability())?;
/// assert_eq!(plan.state_dimension(), 4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MethodOfLinesPlan<S: RealScalar> {
    order: MethodOfLinesOrder,
    mass: Vec<S>,
    scales: Vec<S>,
    normalizers: Vec<S>,
    identity_scales: bool,
    cfl: CflPolicy<S>,
    stability: OdeLinearStability<S>,
    pattern: Option<StructuralPattern>,
}
impl<S: RealScalar> MethodOfLinesPlan<S> {
    /// Prepare first-order dynamics with positive finite diagonal mass and field
    /// scales; `None` means identity. Explicit step limiting starts disabled. A
    /// zero dimension describes a fully constrained system with no dynamics.
    /// # Errors
    /// Rejects wrong lengths, nonpositive/nonfinite entries, checked-size
    /// overflow and allocation failure, before numerical callbacks.
    pub fn new(
        dimension: usize,
        mass: Option<&[S]>,
        scales: Option<&[S]>,
    ) -> Result<Self, SciOxideError> {
        let prepare = |values: Option<&[S]>,
                       argument: scioxide_core::Argument,
                       quantity: scioxide_core::Quantity|
         -> Result<Vec<S>, SciOxideError> {
            if let Some(values) = values {
                shape(values.len(), dimension)?;
                if values.iter().any(|v| !v.is_finite() || *v <= S::ZERO) {
                    return Err(invalid(
                        argument,
                        "mass and field scales must be positive and finite",
                    ));
                }
            }
            let mut out = try_filled(dimension, S::ONE, quantity)?;
            if let Some(values) = values {
                out.copy_from_slice(values);
            }
            Ok(out)
        };
        let mass = prepare(
            mass,
            scioxide_core::argument!("mass"),
            scioxide_core::quantity!("mass"),
        )?;
        let scales = prepare(
            scales,
            scioxide_core::argument!("scales"),
            scioxide_core::quantity!("scales"),
        )?;
        let mut normalizers =
            try_with_capacity(dimension, scioxide_core::quantity!("normalizers"))?;
        normalizers.extend(mass.iter().zip(&scales).map(|(&m, &d)| {
            let inverse = S::ONE / m;
            let factor = inverse / d;
            // Zero selects replay from the original mass and field factors.
            if inverse.is_normal() && factor.is_normal() {
                factor
            } else {
                S::ZERO
            }
        }));
        let identity_scales = scales.iter().all(|&d| d == S::ONE);
        Ok(Self {
            order: MethodOfLinesOrder::First,
            mass,
            scales,
            normalizers,
            identity_scales,
            cfl: CflPolicy::Disabled,
            stability: OdeLinearStability::Undeclared,
            pattern: None,
        })
    }
    /// Select first- or second-order dynamics.
    #[must_use]
    pub fn with_order(mut self, order: MethodOfLinesOrder) -> Self {
        self.order = order;
        self
    }
    /// Select explicit step limiting for an integrator of declared stability.
    /// # Errors
    /// Rejects a nonpositive or nonfinite maximum step.
    pub fn with_step_limit(
        mut self,
        cfl: CflPolicy<S>,
        stability: OdeLinearStability<S>,
    ) -> Result<Self, SciOxideError> {
        cfl.validate(scioxide_core::argument!("cfl"))?;
        self.cfl = cfl;
        self.stability = stability;
        Ok(self)
    }
    /// Declare the provider's sparse Jacobian pattern, used by automatic step
    /// limits and by the doubled second-order pattern.
    /// # Errors
    /// Rejects a pattern whose shape differs from the provider dimension.
    pub fn with_jacobian_pattern(
        mut self,
        pattern: StructuralPattern,
    ) -> Result<Self, SciOxideError> {
        let view = pattern.view();
        shape(view.rows(), self.dimension())?;
        shape(view.cols(), self.dimension())?;
        self.pattern = Some(pattern);
        Ok(self)
    }
    /// Number of provider (physical field) components.
    pub fn dimension(&self) -> usize {
        self.mass.len()
    }
    /// ODE state dimension: the provider dimension, doubled for second order.
    pub fn state_dimension(&self) -> usize {
        match self.order {
            MethodOfLinesOrder::First => self.dimension(),
            MethodOfLinesOrder::Second => 2 * self.dimension(),
        }
    }
    /// Dynamics order.
    pub const fn order(&self) -> MethodOfLinesOrder {
        self.order
    }
    /// Physical diagonal mass in provider order.
    pub fn mass(&self) -> &[S] {
        &self.mass
    }
    /// Coordinate scales in `y = D z`.
    pub fn field_scales(&self) -> &[S] {
        &self.scales
    }
    /// Allocate reusable state and Jacobian scratch. No numerical callback is invoked.
    /// # Errors
    /// Propagates allocation and checked-size failures.
    pub fn workspace<T: NumericScalar<Real = S>>(
        &self,
    ) -> Result<MethodOfLinesWorkspace<T>, SciOxideError> {
        let n = self.dimension();
        let vector = |quantity| try_filled(n, T::ZERO, quantity);
        Ok(MethodOfLinesWorkspace {
            field: Vec::new(),
            parameters: Vec::new(),
            dense: None,
            state: vector(scioxide_core::quantity!("workspace.state"))?,
            direction: vector(scioxide_core::quantity!("workspace.direction"))?,
            output: vector(scioxide_core::quantity!("workspace.output"))?,
            jacobian: try_filled(
                self.pattern
                    .as_ref()
                    .map_or(0, |pattern| pattern.view().col_indices().len()),
                T::ZERO,
                scioxide_core::quantity!("workspace.jacobian"),
            )?,
        })
    }

    pub(crate) fn workspace_bytes<T: NumericScalar<Real = S>>(
        &self,
        parameters: usize,
    ) -> Result<usize, SciOxideError> {
        const BYTES: scioxide_core::Quantity = scioxide_core::quantity!("mol_workspace_bytes");
        let vectors = checked_mul(self.dimension(), 3, BYTES)?;
        let entries = self
            .pattern
            .as_ref()
            .map_or(0, |p| p.view().col_indices().len());
        checked_add(
            checked_mul(
                checked_add(vectors, entries, BYTES)?,
                core::mem::size_of::<T>(),
                BYTES,
            )?,
            checked_mul(parameters, core::mem::size_of::<S>(), BYTES)?,
            BYTES,
        )
    }
    /// Transform physical absolute tolerances once; relative tolerance and norm are preserved.
    /// Vector tolerances list the full ODE state (`[u, v]` for second order).
    /// # Errors
    /// Rejects a vector length mismatch or unrepresentable normalized tolerance.
    pub fn error_tolerance(
        &self,
        physical: &ErrorTolerance<S>,
    ) -> Result<ErrorTolerance<S>, SciOxideError> {
        let total = self.state_dimension();
        let mut absolute =
            try_with_capacity(total, scioxide_core::quantity!("absolute_tolerances"))?;
        for i in 0..total {
            let a = physical.absolute().component(i, total)?;
            let normalized = scale(a, S::ONE, S::ONE, self.scales[i % self.dimension()])?;
            if a > S::ZERO && normalized == S::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("physical"),
                    "normalized tolerance underflows",
                ));
            }
            absolute.push(normalized);
        }
        ErrorTolerance::try_vector(&absolute, physical.relative(), physical.norm())
    }
    /// Encode physical state into ODE coordinates. Input and output cannot alias.
    /// # Errors
    /// Shape/nonfinite/range errors leave output unchanged.
    pub fn encode_into<T: NumericScalar<Real = S>>(
        &self,
        physical: &[T],
        normalized: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.convert(physical, normalized, false)
    }
    /// Decode ODE coordinates into physical state. Input and output cannot alias.
    /// # Errors
    /// Shape/nonfinite/range errors leave output unchanged.
    pub fn decode_into<T: NumericScalar<Real = S>>(
        &self,
        normalized: &[T],
        physical: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.convert(normalized, physical, true)
    }
    /// Shared encode/decode kernel over a full or one-block state.
    fn convert<T: NumericScalar<Real = S>>(
        &self,
        input: &[T],
        output: &mut [T],
        decode: bool,
    ) -> Result<(), SciOxideError> {
        if input.len() != self.dimension() {
            shape(input.len(), self.state_dimension())?;
        }
        shape(output.len(), input.len())?;
        finite_slice(input, scioxide_core::NonFiniteOrigin::Input)?;
        if self.identity_scales {
            output.copy_from_slice(input);
            return Ok(());
        }
        let n = self.dimension();
        let native = input.iter().enumerate().all(|(i, value)| {
            let factor = self.scales[i % n];
            let bound = if decode {
                if factor <= S::ONE {
                    return true;
                }
                (S::MAX * S::HALF) / factor
            } else {
                if factor >= S::ONE {
                    return true;
                }
                (S::MAX * S::HALF) * factor
            };
            value
                .real_component()
                .abs()
                .max(value.imaginary_component().abs())
                <= bound
        });
        if native {
            for (i, out) in output.iter_mut().enumerate() {
                let factor = self.scales[i % n];
                let value = input[i];
                *out = if decode {
                    T::from_components(
                        value.real_component() * factor,
                        value.imaginary_component() * factor,
                    )
                } else {
                    T::from_components(
                        value.real_component() / factor,
                        value.imaginary_component() / factor,
                    )
                };
            }
            return Ok(());
        }
        let value = |i: usize| {
            if decode {
                scale(input[i], self.scales[i % n], S::ONE, S::ONE)
            } else {
                scale(input[i], S::ONE, S::ONE, self.scales[i % n])
            }
        };
        for i in 0..input.len() {
            value(i)?;
        }
        for (i, out) in output.iter_mut().enumerate() {
            *out = value(i)?;
        }
        Ok(())
    }
    fn validate_limit(&self) -> Result<(), SciOxideError> {
        if self.cfl != CflPolicy::Automatic {
            return Ok(());
        }
        let OdeLinearStability::Axes { real, imaginary } = self.stability else {
            return Err(invalid(
                scioxide_core::argument!("stability"),
                "the integrator declares no stability segment; select a disabled or maximum step policy",
            ));
        };
        let bound = match self.order {
            MethodOfLinesOrder::First => real,
            MethodOfLinesOrder::Second => imaginary,
        };
        match bound {
            None => Ok(()),
            Some(bound) if bound <= S::ZERO || !bound.is_finite() => Err(invalid(
                scioxide_core::argument!("stability"),
                "the integrator is unstable on the axis carrying these dynamics",
            )),
            Some(_) if self.pattern.is_none() => Err(invalid(
                scioxide_core::argument!("self"),
                "automatic step limits need a declared Jacobian pattern",
            )),
            Some(_) => Ok(()),
        }
    }
    /// Bind a complete physical RHS pullback with parameter-independent diagonal
    /// mass and coordinate scales. Parameter-dependent mass uses a residual DAE.
    /// Objective and initial cotangents must use the same normalized coordinates.
    /// # Errors
    /// Returns ordinary binding failures, rejects a nonreplayable provider and
    /// second-order dynamics.
    pub fn bind_adjoint<'a, T: NumericScalar<Real = S>>(
        &'a self,
        provider: &'a dyn scioxide_ode::OdeAdjointSystem<T>,
        workspace: &'a mut MethodOfLinesWorkspace<T>,
    ) -> Result<MethodOfLinesSystem<'a, S, T>, SciOxideError> {
        if !provider.is_replayable() {
            return Err(invalid(
                scioxide_core::argument!("provider"),
                "adjoint provider must be replayable",
            ));
        }
        if self.order != MethodOfLinesOrder::First {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "adjoint binding supports first-order dynamics",
            ));
        }
        workspace.reserve_parameters(provider.parameters().len())?;
        let mut system = self.bind(provider, workspace)?;
        system.adjoint = Some(provider);
        Ok(system)
    }
    /// Bind a borrowed physical residual provider and exclusive prepared workspace.
    /// # Errors
    /// Rejects dimension mismatches, competing provider-owned mass matrices and
    /// automatic step limits that the integrator or pattern cannot support.
    pub fn bind<'a, T: NumericScalar<Real = S>>(
        &'a self,
        provider: &'a dyn OdeSystem<S, T>,
        workspace: &'a mut MethodOfLinesWorkspace<T>,
    ) -> Result<MethodOfLinesSystem<'a, S, T>, SciOxideError> {
        shape(provider.dimension(), self.dimension())?;
        shape(workspace.state.len(), self.dimension())?;
        shape(
            workspace.jacobian.len(),
            self.pattern
                .as_ref()
                .map_or(0, |pattern| pattern.view().col_indices().len()),
        )?;
        if provider.mass_matrix_revision().is_some() {
            return Err(invalid(
                scioxide_core::argument!("provider"),
                "mass is supplied exclusively by the method-of-lines plan",
            ));
        }
        self.validate_limit()?;
        let doubled = match (&self.pattern, self.order) {
            (Some(pattern), MethodOfLinesOrder::Second) => Some(doubled_pattern(pattern.view())?),
            _ => None,
        };
        Ok(MethodOfLinesSystem {
            plan: self,
            provider,
            fields: None,
            workspace: RefCell::new(workspace),
            stage_preconditioner: None,
            adjoint: None,
            doubled,
            spectrum: Cell::new(None),
        })
    }
    /// Bind a provider together with its full-field reconstruction.
    /// # Errors
    /// Same as [`Self::bind`], plus a field-length mismatch.
    pub fn bind_with_fields<'a, T: NumericScalar<Real = S>>(
        &'a self,
        provider: &'a dyn OdeSystem<S, T>,
        fields: &'a dyn MethodOfLinesFields<S, T>,
        workspace: &'a mut MethodOfLinesWorkspace<T>,
    ) -> Result<MethodOfLinesSystem<'a, S, T>, SciOxideError> {
        workspace.reserve_field(fields.field_len())?;
        let mut system = self.bind(provider, workspace)?;
        system.fields = Some(fields);
        Ok(system)
    }
}

/// Second-order pattern `[[0, I], [P, 0]]` for a provider pattern `P`.
fn doubled_pattern(pattern: StructuralPatternView<'_>) -> Result<StructuralPattern, SciOxideError> {
    let n = pattern.rows();
    let total = checked_mul(n, 2, scioxide_core::quantity!("second_order_rows"))?;
    let mut rows = try_with_capacity(
        checked_add(total, 1, scioxide_core::quantity!("second_order_rows"))?,
        scioxide_core::quantity!("second_order_rows"),
    )?;
    let mut cols = try_with_capacity(
        checked_add(
            n,
            pattern.col_indices().len(),
            scioxide_core::quantity!("second_order_columns"),
        )?,
        scioxide_core::quantity!("second_order_columns"),
    )?;
    rows.push(0);
    for i in 0..n {
        cols.push(n + i);
        rows.push(cols.len());
    }
    for i in 0..n {
        cols.extend_from_slice(
            &pattern.col_indices()[pattern.row_ptr()[i]..pattern.row_ptr()[i + 1]],
        );
        rows.push(cols.len());
    }
    StructuralPattern::new(total, total, rows, cols)
}

/// Caller-owned normalization scratch, reusable across independent bindings.
/// Default storage is O(n + nnz); dense Jacobian staging is explicitly prepared.
pub struct MethodOfLinesWorkspace<T: NumericScalar> {
    field: Vec<T>,
    parameters: Vec<T::Real>,
    dense: Option<Vec<T>>,
    state: Vec<T>,
    direction: Vec<T>,
    output: Vec<T>,
    jacobian: Vec<T>,
}

impl<T: NumericScalar> MethodOfLinesWorkspace<T> {
    pub(crate) fn reserve_field(&mut self, count: usize) -> Result<(), SciOxideError> {
        if self.field.len() < count {
            let extra = count - self.field.len();
            scioxide_core::backend::try_reserve(
                &mut self.field,
                extra,
                scioxide_core::quantity!("field_staging"),
            )?;
        }
        self.field.resize(count, T::ZERO);
        Ok(())
    }
    pub(crate) fn reserve_parameters(&mut self, count: usize) -> Result<(), SciOxideError> {
        if self.parameters.len() < count {
            let extra = count - self.parameters.len();
            scioxide_core::backend::try_reserve(
                &mut self.parameters,
                extra,
                scioxide_core::quantity!("parameter_staging"),
            )?;
        }
        self.parameters.resize(count, T::Real::ZERO);
        Ok(())
    }
    /// Reserve transactional dense Jacobian staging before binding a system.
    /// Default workspaces use linear vector storage plus the declared sparse pattern;
    /// dense derivatives return `None` until this option is prepared. Second-order
    /// adapters continue to expose sparse derivatives and products only.
    /// # Errors
    /// Returns checked-size overflow or allocation failure.
    /// ```
    /// use scioxide_pde::MethodOfLinesPlan;
    /// let plan = MethodOfLinesPlan::<f64>::new(3, None, None)?;
    /// let workspace = plan.workspace::<f64>()?.with_dense_jacobian()?;
    /// # let _ = workspace;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn with_dense_jacobian(mut self) -> Result<Self, SciOxideError> {
        let n = self.state.len();
        self.dense = Some(try_filled(
            checked_mul(n, n, scioxide_core::quantity!("dense_jacobian"))?,
            T::ZERO,
            scioxide_core::quantity!("dense_jacobian"),
        )?);
        Ok(self)
    }
    /// Whether this scratch has exactly the shape [`MethodOfLinesPlan::workspace`] prepares.
    pub(crate) fn fits<S: RealScalar>(&self, plan: &MethodOfLinesPlan<S>) -> bool {
        let n = plan.dimension();
        let entries = plan
            .pattern
            .as_ref()
            .map_or(0, |pattern| pattern.view().col_indices().len());
        self.state.len() == n
            && self.direction.len() == n
            && self.output.len() == n
            && self.jacobian.len() == entries
    }
}
/// Caller-prepared inverse action for `I - shift * J_z` in normalized coordinates.
///
/// Arguments are physical time, normalized state `z`, and stage shift. The
/// returned provider's dimensions/scratch must fit the ODE backend reservation.
/// Preparation and revision are explicit caller responsibilities; the callback
/// must not allocate numerical storage during a prepared solve. Return `None`
/// to explicitly select an unpreconditioned stage.
pub trait MethodOfLinesStagePreconditioner<S: RealScalar, T: NumericScalar<Real = S> = S> {
    /// Borrow a prepared inverse for the normalized effective stage equation.
    /// # Errors
    /// Propagates explicit preparation, revision or provider failures.
    fn preconditioner(
        &self,
        time: S,
        state: &[T],
        shift: S,
    ) -> Result<Option<&dyn scioxide_core::Preconditioner<T>>, SciOxideError>;
}
/// The method-of-lines adapter: a borrowed normalized ODE system.
///
/// The ODE driver owns all acceptance and stage solves. Dense and sparse
/// derivatives are staged and normalized before publication; products use caller operator scratch and
/// O(n) prepared vectors without column materialization. Second-order dynamics
/// expose sparse Jacobians and products (no dense Jacobian). First-order dense
/// derivatives require [`MethodOfLinesWorkspace::with_dense_jacobian`]; sparse
/// derivatives require the plan's declared pattern. Unprepared derivatives return
/// `None` without invoking the provider. Physical-space
/// preconditioners are deliberately not forwarded to the changed effective system.
pub struct MethodOfLinesSystem<'a, S: RealScalar, T: NumericScalar<Real = S> = S> {
    plan: &'a MethodOfLinesPlan<S>,
    provider: &'a dyn OdeSystem<S, T>,
    fields: Option<&'a dyn MethodOfLinesFields<S, T>>,
    workspace: RefCell<&'a mut MethodOfLinesWorkspace<T>>,
    adjoint: Option<&'a dyn scioxide_ode::OdeAdjointSystem<T>>,
    stage_preconditioner: Option<&'a dyn MethodOfLinesStagePreconditioner<S, T>>,
    doubled: Option<StructuralPattern>,
    spectrum: Cell<Option<Scaled<S>>>,
}
impl<'a, S: RealScalar, T: NumericScalar<Real = S>> MethodOfLinesSystem<'a, S, T> {
    /// Inject a preconditioner for the normalized, shifted ODE stage equation.
    ///
    /// This deliberately does not forward the physical provider's inverse.
    /// Use field blocks in normalized free-coordinate order. The ODE solver
    /// checks reserved scratch before applying the returned preconditioner.
    #[must_use]
    pub fn with_stage_preconditioner(
        mut self,
        provider: &'a dyn MethodOfLinesStagePreconditioner<S, T>,
    ) -> Self {
        self.stage_preconditioner = Some(provider);
        self
    }
    /// Adapter configuration.
    pub const fn plan(&self) -> &'a MethodOfLinesPlan<S> {
        self.plan
    }
    /// Sparse pattern of the normalized ODE Jacobian, when declared.
    pub fn jacobian_pattern(&self) -> Option<StructuralPatternView<'_>> {
        match self.plan.order {
            MethodOfLinesOrder::First => self.plan.pattern.as_ref().map(StructuralPattern::view),
            MethodOfLinesOrder::Second => self.doubled.as_ref().map(StructuralPattern::view),
        }
    }
    /// Full physical field length (velocity is not part of field output).
    pub fn field_len(&self) -> usize {
        self.fields
            .map_or(self.plan.dimension(), |fields| fields.field_len())
    }
    /// Reconstruct the full physical field (displacement for second order) at
    /// `time` from normalized ODE state. Output is unchanged on error.
    /// # Errors
    /// Rejects malformed state/output, nonfinite values, or failed constraint data.
    pub fn reconstruct_into(
        &self,
        time: S,
        state: &[T],
        out: &mut [T],
    ) -> Result<(), SciOxideError> {
        shape(state.len(), self.dimension())?;
        shape(out.len(), self.field_len())?;
        if !time.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "time must be finite",
            ));
        }
        let mut w = self.work()?;
        let w = &mut **w;
        let n = self.plan.dimension();
        self.decode(&state[..n], &mut w.state)?;
        match self.fields {
            Some(fields) => {
                shape(w.field.len(), out.len())?;
                crate::validation::poison(&mut w.field);
                fields.reconstruct_into(time, &w.state, &mut w.field)?;
                finite_slice(&w.field, scioxide_core::NonFiniteOrigin::Callback)?;
                out.copy_from_slice(&w.field);
                Ok(())
            }
            None => {
                out.copy_from_slice(&w.state);
                Ok(())
            }
        }
    }
    fn work(&self) -> Result<RefMut<'_, &'a mut MethodOfLinesWorkspace<T>>, SciOxideError> {
        self.workspace.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "workspace is already borrowed",
            )
        })
    }
    fn validate_request(
        &self,
        time: S,
        state: &[T],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::validation::finite(time, scioxide_core::NonFiniteOrigin::Input)?;
        shape(state.len(), self.dimension())?;
        finite_slice(state, scioxide_core::NonFiniteOrigin::Input)?;
        shape(self.provider.dimension(), self.plan.dimension())?;
        if self.provider.mass_matrix_revision().is_some() {
            return Err(invalid(
                scioxide_core::argument!("provider"),
                "provider mass changed after binding",
            ));
        }
        execution.check()
    }
    fn normalize(&self, values: &mut [T]) -> Result<(), SciOxideError> {
        for (i, v) in values.iter_mut().enumerate() {
            if !v.has_finite_components() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Callback,
                    "provider output must be finite",
                ));
            }
            let multiplier = self.plan.normalizers[i];
            let real = v.real_component() * multiplier;
            let imaginary = v.imaginary_component() * multiplier;
            let safe = |original: S, result: S| {
                result.is_finite() && (original == S::ZERO || result.is_normal())
            };
            *v = if multiplier.is_normal()
                && safe(v.real_component(), real)
                && (T::IS_REAL || safe(v.imaginary_component(), imaginary))
            {
                T::from_components(real, imaginary)
            } else {
                scale_fallback(*v, S::ONE, self.plan.mass[i], self.plan.scales[i])?
            };
        }
        Ok(())
    }
    fn decode(&self, normalized: &[T], physical: &mut [T]) -> Result<(), SciOxideError> {
        shape(physical.len(), normalized.len())?;
        finite_slice(normalized, scioxide_core::NonFiniteOrigin::Input)?;
        if self.plan.identity_scales {
            physical.copy_from_slice(normalized);
        } else {
            for ((out, &value), &factor) in
                physical.iter_mut().zip(normalized).zip(&self.plan.scales)
            {
                *out = scale(value, factor, S::ONE, S::ONE)?;
            }
        }
        Ok(())
    }
    fn second(&self) -> bool {
        self.plan.order == MethodOfLinesOrder::Second
    }
    /// Displacement block of a (possibly doubled) normalized state.
    fn displacement<'s>(&self, state: &'s [T]) -> Result<&'s [T], SciOxideError> {
        shape(state.len(), self.dimension())?;
        Ok(&state[..self.plan.dimension()])
    }
    fn dense(
        &self,
        state: &[T],
        out: &mut MatrixViewMut<'_, T>,
        execution: ExecutionContext<'_>,
        invoke: impl FnOnce(&[T], &mut MatrixViewMut<'_, T>) -> Result<Option<()>, SciOxideError>,
    ) -> Result<Option<()>, SciOxideError> {
        shape(out.rows(), self.dimension())?;
        shape(out.cols(), self.dimension())?;
        if self.second() {
            return Ok(None);
        }
        let mut w = self.work()?;
        let w = &mut **w;
        if w.dense.is_none() {
            return Ok(None);
        }
        self.decode(state, &mut w.state)?;
        let Some(staged) = w.dense.as_mut() else {
            return Ok(None);
        };
        staged.fill(T::from_real(S::NAN));
        let mut matrix = MatrixViewMut::from_slice(self.dimension(), self.dimension(), staged)?;
        let available = invoke(&w.state, &mut matrix)?;
        if available.is_some() {
            for i in 0..self.dimension() {
                for j in 0..self.dimension() {
                    let v = matrix.get(i, j)?;
                    finite_slice(&[v], scioxide_core::NonFiniteOrigin::Callback)?;
                    matrix.set(
                        i,
                        j,
                        scale(
                            v,
                            self.plan.scales[j],
                            self.plan.mass[i],
                            self.plan.scales[i],
                        )?,
                    )?;
                }
            }
            execution.check()?;
            for i in 0..self.dimension() {
                for j in 0..self.dimension() {
                    out.set(i, j, matrix.get(i, j)?)?;
                }
            }
        }
        Ok(available)
    }
    fn sparse(
        &self,
        y: &[T],
        p: StructuralPatternView<'_>,
        out: &mut [T],
        execution: ExecutionContext<'_>,
        invoke: impl FnOnce(
            &[T],
            StructuralPatternView<'_>,
            &mut [T],
        ) -> Result<Option<()>, SciOxideError>,
    ) -> Result<Option<()>, SciOxideError> {
        shape(p.rows(), self.dimension())?;
        shape(p.cols(), self.dimension())?;
        shape(out.len(), p.col_indices().len())?;
        let n = self.plan.dimension();
        let mut w = self.work()?;
        let w = &mut **w;
        self.decode(self.displacement(y)?, &mut w.state)?;
        if !self.second() {
            let Some(own) = &self.plan.pattern else {
                return Ok(None);
            };
            let own = own.view();
            if p.row_ptr() != own.row_ptr() || p.col_indices() != own.col_indices() {
                return Ok(None);
            }
            w.jacobian.fill(T::from_real(S::NAN));
            let available = invoke(&w.state, p, &mut w.jacobian)?;
            if available.is_some() {
                finite_slice(&w.jacobian, scioxide_core::NonFiniteOrigin::Callback)?;
                normalize_rows(self.plan, p, &mut w.jacobian)?;
                execution.check()?;
                out.copy_from_slice(&w.jacobian);
            }
            return Ok(available);
        }
        let (Some(doubled), Some(provider)) = (&self.doubled, &self.plan.pattern) else {
            return Ok(None);
        };
        let doubled = doubled.view();
        if p.row_ptr() != doubled.row_ptr() || p.col_indices() != doubled.col_indices() {
            return Err(invalid(
                scioxide_core::argument!("p"),
                "second-order Jacobian pattern mismatch",
            ));
        }
        let provider = provider.view();
        w.jacobian.fill(T::from_real(S::NAN));
        if invoke(&w.state, provider, &mut w.jacobian)?.is_none() {
            return Ok(None);
        }
        finite_slice(&w.jacobian, scioxide_core::NonFiniteOrigin::Callback)?;
        normalize_rows(self.plan, provider, &mut w.jacobian)?;
        execution.check()?;
        out[..n].fill(T::ONE);
        out[n..].copy_from_slice(&w.jacobian);
        Ok(Some(()))
    }
    /// Gershgorin bound of the normalized provider Jacobian.
    fn spectral_bound(
        &self,
        t: S,
        y: &[T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Scaled<S>, SciOxideError> {
        if let Some(bound) = self.spectrum.get()
            && self.provider.jacobian_is_constant()
        {
            return Ok(bound);
        }
        context.execution().check()?;
        if !context
            .record_provider(EvaluationRole::Jacobian)?
            .is_granted()
        {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::EvaluationLimit,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                "step-limit Jacobian budget exhausted",
            ));
        }
        let pattern = self
            .plan
            .pattern
            .as_ref()
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("self"),
                    "automatic step limits need a declared Jacobian pattern",
                )
            })?
            .view();
        let mut w = self.work()?;
        let w = &mut **w;
        self.decode(self.displacement(y)?, &mut w.state)?;
        crate::validation::poison(&mut w.jacobian);
        self.provider
            .sparse_jacobian(t, &w.state, pattern, &mut w.jacobian, context)?
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("provider"),
                    "automatic step limits need the provider's sparse Jacobian",
                )
            })?;
        finite_slice(&w.jacobian, scioxide_core::NonFiniteOrigin::Callback)?;
        let mut bound = Scaled::from_value(S::ZERO)?;
        for i in 0..pattern.rows() {
            let mut row = Scaled::from_value(S::ZERO)?;
            for k in pattern.row_ptr()[i]..pattern.row_ptr()[i + 1] {
                let j = pattern.col_indices()[k];
                let magnitude = if T::IS_REAL {
                    Scaled::from_value(w.jacobian[k].abs())?
                } else {
                    scioxide_core::ScaledValue::from_value(w.jacobian[k])?.magnitude()?
                };
                row = row.add(magnitude.mul(Scaled::from_value(self.plan.scales[j])?)?)?;
            }
            let row = row
                .div(Scaled::from_value(self.plan.mass[i])?)?
                .div(Scaled::from_value(self.plan.scales[i])?)?;
            if bound.abs_le(row) {
                bound = row;
            }
        }
        context.execution().check()?;
        self.spectrum.set(Some(bound));
        Ok(bound)
    }
}

/// Scale sparse provider entries to normalized coordinates.
fn normalize_rows<S: RealScalar, T: NumericScalar<Real = S>>(
    plan: &MethodOfLinesPlan<S>,
    p: StructuralPatternView<'_>,
    out: &mut [T],
) -> Result<(), SciOxideError> {
    for i in 0..p.rows() {
        for (value, &column) in out[p.row_ptr()[i]..p.row_ptr()[i + 1]]
            .iter_mut()
            .zip(&p.col_indices()[p.row_ptr()[i]..p.row_ptr()[i + 1]])
        {
            *value = scale(*value, plan.scales[column], plan.mass[i], plan.scales[i])?;
        }
    }
    Ok(())
}

impl<S: RealScalar, T: NumericScalar<Real = S>> OdeSystem<S, T> for MethodOfLinesSystem<'_, S, T> {
    fn stage_preconditioner(
        &self,
        time: S,
        state: &[T],
        shift: S,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<&dyn scioxide_core::Preconditioner<T>>, SciOxideError> {
        shape(state.len(), self.dimension())?;
        finite_slice(state, scioxide_core::NonFiniteOrigin::Callback)?;
        if !time.is_finite() || !shift.is_finite() {
            return Err(invalid(
                if time.is_finite() {
                    scioxide_core::argument!("shift")
                } else {
                    scioxide_core::argument!("time")
                },
                "stage time and shift must be finite",
            ));
        }
        let Some(provider) = self.stage_preconditioner else {
            return Ok(None);
        };
        let preconditioner = provider.preconditioner(time, state, shift)?;
        if let Some(p) = preconditioner {
            shape(p.dimension(), self.dimension())?;
        }
        Ok(preconditioner)
    }
    fn dimension(&self) -> usize {
        self.plan.state_dimension()
    }
    fn jacobian_is_constant(&self) -> bool {
        self.provider.jacobian_is_constant()
    }
    fn rhs(
        &self,
        t: S,
        y: &[T],
        out: &mut [T],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.validate_request(t, y, c.execution())?;
        self.evaluate(y, out, c.execution(), |state, output| {
            self.provider.rhs(t, state, output, c)
        })
    }
    fn admits(&self, t: S, y: &[T], c: &mut SolveContext<'_, '_>) -> Result<bool, SciOxideError> {
        self.validate_request(t, y, c.execution())?;
        let mut w = self.work()?;
        self.decode(self.displacement(y)?, &mut w.state)?;
        self.provider.admits(t, &w.state, c)
    }
    fn step_limit(
        &self,
        t: S,
        y: &[T],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        self.validate_request(t, y, c.execution())?;
        match self.plan.cfl {
            CflPolicy::Disabled => return Ok(None),
            CflPolicy::MaximumStep(h) => return Ok(Some(h)),
            CflPolicy::Automatic => {}
        }
        let OdeLinearStability::Axes { real, imaginary } = self.plan.stability else {
            return Ok(None);
        };
        let Some(segment) = (if self.second() { imaginary } else { real }) else {
            return Ok(None);
        };
        let rho = self.spectral_bound(t, y, c)?;
        if rho.is_zero() {
            return Ok(None);
        }
        let limit = Scaled::from_value(S::from_f64_ieee(SAFETY))?
            .mul(Scaled::from_value(segment)?)?
            .div(if self.second() { rho.sqrt()? } else { rho })?;
        if !limit.abs_le(Scaled::from_value(S::MAX)?) {
            return Ok(None);
        }
        let h = limit.round()?;
        if h == S::ZERO {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::StepSizeLimit,
                scioxide_core::ConvergenceDiagnostics::new(),
                "no representable stable step",
            ));
        }
        Ok(Some(h))
    }
    fn jacobian(
        &self,
        t: S,
        y: &[T],
        out: &mut MatrixViewMut<'_, T>,
        c: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.validate_request(t, y, c.execution())?;
        self.dense(y, out, c.execution(), |state, out| {
            self.provider.jacobian(t, state, out, c)
        })
    }
    fn automatic_jacobian(
        &self,
        t: S,
        y: &[T],
        out: &mut MatrixViewMut<'_, T>,
        c: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.validate_request(t, y, c.execution())?;
        self.dense(y, out, c.execution(), |state, out| {
            self.provider.automatic_jacobian(t, state, out, c)
        })
    }
    fn sparse_jacobian(
        &self,
        t: S,
        y: &[T],
        p: StructuralPatternView<'_>,
        out: &mut [T],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.validate_request(t, y, c.execution())?;
        self.sparse(y, p, out, c.execution(), |state, p, out| {
            self.provider.sparse_jacobian(t, state, p, out, c)
        })
    }
    fn jacobian_product(
        &self,
        t: S,
        y: &[T],
        x: &[T],
        out: &mut [T],
        scratch: &mut [T],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.validate_request(t, y, c.execution())?;
        self.product(y, x, out, c.execution(), |state, direction, output| {
            self.provider
                .jacobian_product(t, state, direction, output, scratch, c)
        })
    }
    fn time_derivative(
        &self,
        t: S,
        y: &[T],
        out: &mut [T],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.validate_request(t, y, c.execution())?;
        shape(out.len(), self.dimension())?;
        let n = self.plan.dimension();
        let mut w = self.work()?;
        let w = &mut **w;
        self.decode(self.displacement(y)?, &mut w.state)?;
        w.output.fill(T::from_components(S::NAN, S::NAN));
        let available = self
            .provider
            .time_derivative(t, &w.state, &mut w.output, c)?;
        if available.is_some() {
            self.normalize(&mut w.output)?;
            c.execution().check()?;
            let (top, bottom) = out.split_at_mut(if self.second() { n } else { 0 });
            top.fill(T::ZERO);
            bottom.copy_from_slice(&w.output);
        }
        Ok(available)
    }
}

impl<S: RealScalar, T: NumericScalar<Real = S>> MethodOfLinesSystem<'_, S, T> {
    /// `out = [v, M⁻¹D⁻¹ f(D u)]` (second order) or `M⁻¹D⁻¹ f(D z)` (first order).
    fn evaluate(
        &self,
        y: &[T],
        out: &mut [T],
        execution: ExecutionContext<'_>,
        invoke: impl FnOnce(&[T], &mut [T]) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        shape(out.len(), self.dimension())?;
        let n = self.plan.dimension();
        let mut w = self.work()?;
        let w = &mut **w;
        self.decode(self.displacement(y)?, &mut w.state)?;
        w.output.fill(T::from_components(S::NAN, S::NAN));
        invoke(&w.state, &mut w.output)?;
        self.normalize(&mut w.output)?;
        execution.check()?;
        if self.second() {
            finite_slice(&y[n..], scioxide_core::NonFiniteOrigin::Callback)?;
            out[..n].copy_from_slice(&y[n..]);
            out[n..].copy_from_slice(&w.output);
        } else {
            out.copy_from_slice(&w.output);
        }
        Ok(())
    }
    fn product(
        &self,
        y: &[T],
        x: &[T],
        out: &mut [T],
        execution: ExecutionContext<'_>,
        invoke: impl FnOnce(&[T], &[T], &mut [T]) -> Result<Option<()>, SciOxideError>,
    ) -> Result<Option<()>, SciOxideError> {
        shape(out.len(), self.dimension())?;
        shape(x.len(), self.dimension())?;
        finite_slice(x, scioxide_core::NonFiniteOrigin::Input)?;
        let n = self.plan.dimension();
        let mut w = self.work()?;
        let w = &mut **w;
        self.decode(self.displacement(y)?, &mut w.state)?;
        self.decode(&x[..n], &mut w.direction)?;
        w.output.fill(T::from_components(S::NAN, S::NAN));
        if invoke(&w.state, &w.direction, &mut w.output)?.is_none() {
            return Ok(None);
        }
        self.normalize(&mut w.output)?;
        execution.check()?;
        if self.second() {
            finite_slice(&x[n..], scioxide_core::NonFiniteOrigin::Input)?;
            out[..n].copy_from_slice(&x[n..]);
            out[n..].copy_from_slice(&w.output);
        } else {
            out.copy_from_slice(&w.output);
        }
        Ok(Some(()))
    }
}

impl<S: RealScalar, T: NumericScalar<Real = S>> scioxide_ode::OdeAdjointSystem<T>
    for MethodOfLinesSystem<'_, S, T>
{
    fn parameters(&self) -> &[S] {
        self.adjoint.map_or(&[], |p| p.parameters())
    }
    fn is_replayable(&self) -> bool {
        self.adjoint.is_some_and(|p| p.is_replayable())
    }
    fn pullback_workspace_len(&self) -> usize {
        self.adjoint.map_or(0, |p| p.pullback_workspace_len())
    }
    fn pullback(
        &self,
        time: S,
        state: &[T],
        seed: &[T],
        state_output: &mut [T],
        parameter_output: &mut [S],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let provider = self.adjoint.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("self"),
                "use bind_adjoint to supply a physical pullback",
            )
        })?;
        shape(seed.len(), self.plan.dimension())?;
        shape(state_output.len(), self.plan.dimension())?;
        shape(parameter_output.len(), provider.parameters().len())?;
        if scratch.len() < provider.pullback_workspace_len() {
            return Err(invalid(
                scioxide_core::argument!("scratch"),
                "insufficient physical pullback scratch",
            ));
        }
        finite_slice(seed, scioxide_core::NonFiniteOrigin::Input)?;
        crate::validation::finite(time, scioxide_core::NonFiniteOrigin::Input)?;
        context.execution().check()?;
        let mut work = self.work()?;
        let work = &mut **work;
        self.decode(state, &mut work.state)?;
        for (i, (out, &v)) in work.direction.iter_mut().zip(seed).enumerate() {
            *out = scale(v, S::ONE, self.plan.mass[i], self.plan.scales[i])?;
        }
        work.output.fill(T::from_real(S::NAN));
        shape(work.parameters.len(), parameter_output.len())?;
        work.parameters.fill(S::NAN);
        provider.pullback(
            time,
            &work.state,
            &work.direction,
            &mut work.output,
            &mut work.parameters,
            scratch,
            context,
        )?;
        finite_slice(&work.output, scioxide_core::NonFiniteOrigin::Callback)?;
        finite_slice(&work.parameters, scioxide_core::NonFiniteOrigin::Callback)?;
        for (i, out) in work.output.iter_mut().enumerate() {
            *out = scale(*out, self.plan.scales[i], S::ONE, S::ONE)?;
        }
        context.execution().check()?;
        state_output.copy_from_slice(&work.output);
        parameter_output.copy_from_slice(&work.parameters);
        Ok(())
    }
}
