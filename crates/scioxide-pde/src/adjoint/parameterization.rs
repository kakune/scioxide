//! Parameter derivatives of prepared linear spatial problems.
//!
//! A caller describes how the data of a prepared plan depends on real parameters
//! `p` through derivative *data* in the plan's own vocabulary: a signed derivative
//! coefficient field, a derivative source and derivative boundary data and Robin
//! coefficients. The plan then assembles the exact parameter derivative of its own
//! eliminated residual `R(u, p, t) = A(p) u − b(p, t)` with the same assembly code
//! that prepared it; callers assemble no matrices.

use alloc::vec::Vec;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{Dimensions, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError};
use scioxide_sparse::CsrMatrix;
use scioxide_spatial::MeshToken;

use crate::boundary::{
    BoundaryConditions, BoundaryData, BoundaryKind, DirichletConflictPolicy, FieldData,
};
use crate::discretization::{ConditionData, PreparedLoads, SpatialDiscretization};
use crate::validation::{finite, finite_slice, invalid};

/// Derivative of one boundary condition's data and Robin coefficient with respect
/// to one parameter.
///
/// ```
/// use scioxide_pde::{BoundaryData, PdeConditionTangent};
/// let tangent = PdeConditionTangent::<f64, f64, 1>::new(BoundaryData::constant(1.0))
///     .with_robin_coefficient(0.5);
/// assert_eq!(tangent.robin_coefficient(), 0.5);
/// ```
#[derive(Clone, Debug)]
pub struct PdeConditionTangent<R, V, const D: usize> {
    data: BoundaryData<R, V, D>,
    robin: V,
}

impl<R: RealScalar, V: NumericScalar<Real = R> + Send + Sync + 'static, const D: usize>
    PdeConditionTangent<R, V, D>
{
    /// Derivative data `∂g/∂p` with a parameter-independent Robin coefficient.
    #[must_use]
    pub fn new(data: BoundaryData<R, V, D>) -> Self {
        Self {
            data,
            robin: V::ZERO,
        }
    }

    /// A condition whose data and coefficient do not depend on the parameter.
    #[must_use]
    pub fn zero() -> Self {
        Self::new(BoundaryData::zero())
    }

    /// Set the Robin coefficient derivative `∂α/∂p`; only Robin conditions accept
    /// a nonzero value.
    #[must_use]
    pub const fn with_robin_coefficient(mut self, alpha: V) -> Self {
        self.robin = alpha;
        self
    }

    /// Derivative data.
    pub const fn data(&self) -> &BoundaryData<R, V, D> {
        &self.data
    }

    /// Robin coefficient derivative.
    pub const fn robin_coefficient(&self) -> V {
        self.robin
    }
}

/// Derivative of a prepared plan's data with respect to one real parameter.
///
/// `C` is the plan family's coefficient protocol (for example
/// `dyn StructuredFdmCoefficients<R, V, D>`); a supplied provider returns the
/// coefficient *derivatives* at the sampling points, which may have any sign.
/// Absent parts are parameter independent. Boundary tangents list one entry per
/// bound condition, in the original order. Mesh coordinates are never
/// differentiated: geometry, quadrature and the boundary partition stay fixed.
///
/// ```
/// use scioxide_pde::{FieldData, PdeParameterTangent, StructuredFdmCoefficients};
/// let source = PdeParameterTangent::<f64, f64, 1, dyn StructuredFdmCoefficients<f64, f64, 1>>::new()
///     .with_source(FieldData::constant(1.0));
/// assert!(source.coefficients().is_none());
/// ```
pub struct PdeParameterTangent<'a, R, V, const D: usize, C: ?Sized> {
    coefficients: Option<&'a C>,
    source: Option<FieldData<R, V, D>>,
    boundaries: Option<Vec<PdeConditionTangent<R, V, D>>>,
}

impl<R, V, const D: usize, C: ?Sized> Default for PdeParameterTangent<'_, R, V, D, C> {
    fn default() -> Self {
        Self {
            coefficients: None,
            source: None,
            boundaries: None,
        }
    }
}

impl<'a, R, V, const D: usize, C: ?Sized> PdeParameterTangent<'a, R, V, D, C> {
    /// A parameter on which nothing depends.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Coefficient derivatives sampled by the plan's own assembly.
    #[must_use]
    pub fn with_coefficients(mut self, coefficients: &'a C) -> Self {
        self.coefficients = Some(coefficients);
        self
    }

    /// Source derivative `∂f/∂p`.
    #[must_use]
    pub fn with_source(mut self, source: FieldData<R, V, D>) -> Self {
        self.source = Some(source);
        self
    }

    /// Boundary derivatives, one per bound condition in its original order.
    #[must_use]
    pub fn with_boundaries(mut self, boundaries: Vec<PdeConditionTangent<R, V, D>>) -> Self {
        self.boundaries = Some(boundaries);
        self
    }

    /// Coefficient derivatives, if any.
    pub const fn coefficients(&self) -> Option<&'a C> {
        self.coefficients
    }

    /// Source derivative, if any.
    pub const fn source(&self) -> Option<&FieldData<R, V, D>> {
        self.source.as_ref()
    }

    /// Boundary derivatives, if any.
    pub fn boundaries(&self) -> Option<&[PdeConditionTangent<R, V, D>]> {
        self.boundaries.as_deref()
    }
}

/// One assembled parameter direction.
#[derive(Clone, Debug)]
struct Direction<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    /// `∂A/∂p` on free rows, every column kept; `None` when coefficient independent.
    operator: Option<CsrMatrix<V>>,
    /// Tangent of the load weights at the base data (finite volumes only).
    weights: Option<PreparedLoads<R, V, D>>,
    /// Whether source or boundary data depend on the parameter.
    data_dependent: bool,
    source: FieldData<R, V, D>,
    data: Vec<BoundaryData<R, V, D>>,
}

/// Owned, exact parameter derivatives of one prepared linear plan.
///
/// For each parameter `j`, [`Self::residual_tangent_into`] evaluates
/// `∂R/∂p_j` of the plan's eliminated residual `R(u, p, t) = A(p) u − b(p, t)`,
/// whose constrained (Dirichlet) rows read `u − g_D`:
///
/// `∂R_j = (∂A/∂p_j) ū − b(∂f/∂p_j, ∂g/∂p_j, t) − W_j(g, t)`,
///
/// where `ū` holds the free values of the state and the base Dirichlet values
/// `g_D(t)`, the load `b(·)` uses the plan's own weights, lifting and Dirichlet
/// rows, and `W_j` is the tangent of finite-volume boundary-data weights. Linear
/// families (finite differences and elements) assemble `∂A/∂p_j` by re-assembly
/// with the derivative coefficients; finite volumes propagate the derivative
/// through their nonlinear face laws in forward mode. Each parameter with a
/// coefficient derivative or a nonzero Robin-coefficient derivative stores one
/// sparse matrix `∂A/∂p_j` with the operator's pattern.
///
/// Built by [`DifferentiableDiscretization::parameterize`]; `Send + Sync + Clone`.
#[derive(Clone, Debug)]
pub struct PdeParameterization<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    pub(super) preparation: alloc::sync::Arc<()>,
    scratch: usize,
    token: MeshToken,
    values: Vec<R>,
    unknowns: usize,
    loads: PreparedLoads<R, V, D>,
    data: Vec<BoundaryData<R, V, D>>,
    policy: DirichletConflictPolicy,
    zero_source: FieldData<R, V, D>,
    directions: Vec<Direction<R, V, D>>,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> PdeParameterization<R, V, D> {
    /// Real parameter values at which the plan was prepared.
    pub fn values(&self) -> &[R] {
        &self.values
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Exact mesh revision of the parameterized plan.
    pub const fn mesh_token(&self) -> &MeshToken {
        &self.token
    }

    /// Algebraic unknowns of the parameterized plan, including constrained rows.
    pub const fn unknowns(&self) -> usize {
        self.unknowns
    }

    /// Scalar scratch required by [`Self::residual_tangent_into`].
    pub const fn scratch_len(&self) -> usize {
        self.scratch
    }

    pub(crate) fn check_plan(
        &self,
        plan: &impl DifferentiableDiscretization<R, V, D>,
    ) -> Result<(), SciOxideError> {
        if !alloc::sync::Arc::ptr_eq(&self.preparation, plan.preparation()) {
            return Err(invalid(
                scioxide_core::argument!("parameterization"),
                "parameterization belongs to another prepared operator",
            ));
        }
        Ok(())
    }

    /// Evaluate `∂R/∂p_parameter` at `state` and `time` into `out`.
    ///
    /// Constrained entries of `state` are ignored: the derivative is taken at the
    /// free values with the plan's own Dirichlet values. Failures preserve `out`.
    /// # Errors
    /// Rejects an invalid parameter, wrong lengths, non-finite state or time, and
    /// propagates callback and non-finite failures of the derivative data.
    pub fn residual_tangent_into(
        &self,
        parameter: usize,
        state: &[V],
        time: R,
        out: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError> {
        let direction = self.directions.get(parameter).ok_or_else(|| {
            SciOxideError::out_of_bounds(
                scioxide_core::quantity!("parameters"),
                parameter,
                self.directions.len(),
            )
        })?;
        let n = self.unknowns;
        for (actual, expected) in [(state.len(), n), (out.len(), n)] {
            if actual != expected {
                return Err(SciOxideError::dimension_mismatch(
                    Dimensions::one(expected),
                    Dimensions::one(actual),
                ));
            }
        }
        if scratch.len() < self.scratch_len() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.scratch_len()),
                Dimensions::one(scratch.len()),
            ));
        }
        finite_slice(state, NonFiniteOrigin::Input)?;
        finite(time, NonFiniteOrigin::Input)?;
        let (staged, scratch) = scratch[..self.scratch_len()].split_at_mut(n);
        self.evaluate(direction, state, time, staged, scratch)?;
        out.copy_from_slice(staged);
        Ok(())
    }

    fn evaluate(
        &self,
        direction: &Direction<R, V, D>,
        state: &[V],
        time: R,
        out: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError> {
        let base = ConditionData {
            data: &self.data,
            policy: self.policy,
        };
        out.fill(V::ZERO);
        if let Some(operator) = &direction.operator {
            scratch.copy_from_slice(state);
            self.loads.dirichlet_into(&base, time, scratch)?;
            let (pointers, columns, values) = (
                operator.row_ptr(),
                operator.col_indices(),
                operator.values(),
            );
            for (row, value) in out.iter_mut().enumerate() {
                *value = scioxide_core::backend::standard::sum_products(
                    (pointers[row]..pointers[row + 1]).map(|k| (values[k], scratch[columns[k]])),
                    scioxide_core::Accuracy::Standard,
                )?;
            }
        }
        if direction.data_dependent {
            let tangent = ConditionData {
                data: &direction.data,
                policy: self.policy,
            };
            self.loads
                .load_into(&tangent, &direction.source, time, scratch)?;
            subtract(out, scratch)?;
        }
        if let Some(weights) = &direction.weights {
            weights.load_into(&base, &self.zero_source, time, scratch)?;
            subtract(out, scratch)?;
        }
        Ok(())
    }
}

fn subtract<V: NumericScalar>(out: &mut [V], values: &[V]) -> Result<(), SciOxideError> {
    for (out, &value) in out.iter_mut().zip(values) {
        *out = scioxide_core::backend::standard::sum_products(
            [(*out, V::ONE), (-value, V::ONE)].into_iter(),
            scioxide_core::Accuracy::Standard,
        )?;
    }
    Ok(())
}

mod sealed {
    pub trait Sealed {
        fn preparation(&self) -> &alloc::sync::Arc<()>;
    }
}

/// A prepared linear plan whose data derivatives it can assemble itself.
///
/// Implemented by [`crate::StructuredFdmPlan`], [`crate::StructuredFvmPlan`],
/// [`crate::SimplexFvmPlan`] and [`crate::SimplexFemPlan`].
///
/// ```
/// use scioxide_core::SciOxideError;
/// use scioxide_pde::{
///     BoundaryCondition, BoundaryData, DifferentiableDiscretization, FieldData,
///     PdeParameterTangent, SpatialLimits, StructuredBoundaries, StructuredFdmCoefficients,
///     StructuredFdmPlan,
/// };
/// use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh};
/// struct Scaled(f64);
/// impl StructuredFdmCoefficients<f64, f64, 1> for Scaled {
///     fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
///         Ok(self.0)
///     }
/// }
/// let mesh = SegmentMesh::new(Grid1d::new(
///     [CoordinateAxis::uniform(0.0, 1.0, 5)?], [AxisTopology::Bounded])?)?;
/// let side = |s| BoundarySet::axis_side(&mesh, 0, s);
/// let boundaries = StructuredBoundaries::new(&mesh, 1, vec![
///     BoundaryCondition::dirichlet(side(AxisSide::Start)?, BoundaryData::zero()),
///     BoundaryCondition::dirichlet(side(AxisSide::End)?, BoundaryData::zero()),
/// ])?;
/// let k = 2.0;
/// let plan = StructuredFdmPlan::with_options(mesh, &Scaled(k), &boundaries,
///     SpatialLimits::new(10, 100, 100_000))?;
/// // R(u; k) = A(k) u − b with A linear in k, so ∂R/∂k · u = A(1) u.
/// let unit = Scaled(1.0);
/// let parameterization = plan.parameterize(&[k], &[PdeParameterTangent::new()
///     .with_coefficients(&unit as &dyn StructuredFdmCoefficients<f64, f64, 1>)])?;
/// let u = [0.0, 1.0, 1.0, 1.0, 0.0];
/// let mut tangent = [0.0; 5];
/// let mut scratch = vec![0.0; parameterization.scratch_len()];
/// parameterization.residual_tangent_into(0, &u, 0.0, &mut tangent, &mut scratch)?;
/// assert!((tangent[1] - 4.0).abs() < 1e-12 && tangent[2].abs() < 1e-12);
/// # Ok::<(), SciOxideError>(())
/// ```
pub trait DifferentiableDiscretization<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>:
    SpatialDiscretization<R, V, D> + sealed::Sealed
{
    /// The family's coefficient protocol, as a trait object.
    type Coefficients: ?Sized;

    /// Assemble exact derivatives for `values.len()` real parameters, one tangent each.
    /// # Errors
    /// Rejects mismatched counts, boundary tangents not listing one entry per
    /// condition, Robin derivatives on non-Robin conditions (or complex ones for
    /// finite volumes), non-finite values, and propagates callback and assembly
    /// failures.
    fn parameterize(
        &self,
        values: &[R],
        tangents: &[PdeParameterTangent<'_, R, V, D, Self::Coefficients>],
    ) -> Result<PdeParameterization<R, V, D>, SciOxideError>;
}

/// Assembled tangent of one direction: `∂A/∂p` and optional weight tangents.
pub(crate) type AssembledTangent<R, V, const D: usize> =
    (CsrMatrix<V>, Option<PreparedLoads<R, V, D>>);

/// The one validation and assembly path shared by every family.
pub(crate) fn parameterize<R, V, const D: usize, C: ?Sized, Region>(
    plan: &impl SpatialDiscretization<R, V, D>,
    boundaries: &BoundaryConditions<R, V, D, Region>,
    loads: &PreparedLoads<R, V, D>,
    values: &[R],
    tangents: &[PdeParameterTangent<'_, R, V, D, C>],
    mut assemble: impl FnMut(Option<&C>, &[V]) -> Result<AssembledTangent<R, V, D>, SciOxideError>,
) -> Result<PdeParameterization<R, V, D>, SciOxideError>
where
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
{
    const TANGENTS: scioxide_core::Argument = scioxide_core::argument!("tangents");
    if values.len() != tangents.len() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(values.len()),
            Dimensions::one(tangents.len()),
        ));
    }
    finite_slice(values, NonFiniteOrigin::Input)?;
    let conditions = boundaries.conditions();
    let mut directions = try_with_capacity(tangents.len(), scioxide_core::quantity!("parameters"))?;
    let mut robin = try_with_capacity(conditions.len(), scioxide_core::quantity!("robin"))?;
    for tangent in tangents {
        robin.clear();
        let mut data = try_with_capacity(conditions.len(), scioxide_core::quantity!("data"))?;
        match tangent.boundaries() {
            Some(list) => {
                if list.len() != conditions.len() {
                    return Err(invalid(
                        TANGENTS,
                        "boundary tangents must list one entry per condition",
                    ));
                }
                for (condition, entry) in conditions.iter().zip(list) {
                    let alpha = finite(entry.robin, NonFiniteOrigin::Input)?;
                    if alpha != V::ZERO && !matches!(condition.kind(), BoundaryKind::Robin { .. }) {
                        return Err(invalid(
                            TANGENTS,
                            "a Robin coefficient derivative on a non-Robin condition",
                        ));
                    }
                    robin.push(alpha);
                    data.push(entry.data.clone());
                }
            }
            None => {
                robin.resize(conditions.len(), V::ZERO);
                data.resize_with(conditions.len(), BoundaryData::zero);
            }
        }
        let operator = if tangent.coefficients().is_some() || robin.iter().any(|&a| a != V::ZERO) {
            let (operator, weights) = assemble(tangent.coefficients(), &robin)?;
            (Some(operator), weights)
        } else {
            (None, None)
        };
        directions.push(Direction {
            operator: operator.0,
            weights: operator.1,
            data_dependent: tangent.source().is_some() || tangent.boundaries().is_some(),
            source: tangent.source().cloned().unwrap_or_else(FieldData::zero),
            data,
        });
    }
    let mut owned = try_with_capacity(values.len(), scioxide_core::quantity!("parameters"))?;
    owned.extend_from_slice(values);
    let mut data = try_with_capacity(conditions.len(), scioxide_core::quantity!("data"))?;
    data.extend(conditions.iter().map(|condition| condition.data().clone()));
    Ok(PdeParameterization {
        preparation: alloc::sync::Arc::clone(&loads.preparation),
        scratch: scioxide_core::backend::checked_mul(
            plan.unknowns(),
            2,
            scioxide_core::quantity!("parameter_scratch"),
        )?,
        token: plan.mesh_token().clone(),
        values: owned,
        unknowns: plan.unknowns(),
        loads: loads.clone(),
        data,
        policy: boundaries.conflict_policy(),
        zero_source: FieldData::zero(),
        directions,
    })
}

pub(crate) use sealed::Sealed;
