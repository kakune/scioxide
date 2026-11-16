//! Cell-centered conservative spatial finite volumes in one to three dimensions.

use alloc::vec::Vec;
use core::mem::size_of;

use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    Dimensions, ExecutionContext, LinearOperator, NonFiniteOrigin, NumericScalar, RealScalar,
    SciOxideError, Tolerance,
};
use scioxide_sparse::CsrMatrix;
use scioxide_spatial::{
    AxisTopology, Cell, CellField, CellMesh, FaceField, FieldView, MeshToken, SimplexMesh,
    UnstructuredFaceField,
};

use super::geometry::{
    FaceGeometry, Geometry, difference, dot, inverse_normal_matrix, simplex_geometry,
    structured_geometry,
};
use super::lane::{Lane, Tangent};
use crate::boundary::{
    BoundaryConditions, BoundaryKind, BoundaryPoint, FieldData, FieldPoint, StructuredBoundaries,
    TaggedBoundaries,
};
use crate::discretization::{
    AssembledOperator, Elimination, OperatorBuilder, PreparedLoads, SpatialDiscretization,
    SpatialLimits, entry,
};
use crate::validation::{finite, invalid};

/// Pointwise coefficients for `-div(K grad u) + c u = f`.
///
/// `K` is real, symmetric and positive definite. The reaction may be complex.
/// Both coefficients are sampled once per cell center when a plan is prepared.
/// Rounding-level asymmetry is projected to the symmetric part. Interior face
/// fluxes eliminate a shared face value using both cell-owned half-cell laws.
/// Preparation rejects non-positive half-cell conductances; positive jumps do
/// not imply an M-matrix for general skew geometry or anisotropic tensors.
/// Wave problems supply the squared speed as diffusion.
pub trait FvmCoefficients<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    /// Full diffusion tensor at a cell center, where the cell owns its material.
    /// # Errors
    /// Callback errors propagate unchanged.
    fn diffusion(&self, point: [R; D]) -> Result<[[R; D]; D], SciOxideError>;
    /// Reaction coefficient at a cell centroid.
    /// # Errors
    /// Callback errors propagate unchanged.
    fn reaction(&self, point: [R; D]) -> Result<V, SciOxideError> {
        let _ = point;
        Ok(V::ZERO)
    }
}

/// Face law `alpha·u + beta·(K grad u · n) = g` whose value is a data site.
#[derive(Clone, Copy)]
struct BoundaryLaw<R> {
    alpha: R,
    beta: R,
    site: usize,
}

struct Gradient<L, const D: usize> {
    terms: Vec<(usize, [L; D])>,
    constant: Vec<(usize, [L; D])>,
}

#[derive(Clone)]
struct FaceFlux<L> {
    left: Option<usize>,
    right: Option<usize>,
    terms: Vec<(usize, L)>,
    constant: Vec<(usize, L)>,
}

/// Prepared operator, face laws and loads shared by structured and simplex FVM.
#[derive(Clone)]
struct FvmCore<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    geometry: Geometry<R, D>,
    tensors: Vec<[[R; D]; D]>,
    laws: Vec<Option<BoundaryLaw<R>>>,
    faces: Vec<FaceFlux<R>>,
    sites: Vec<BoundaryPoint<R, D>>,
    site_conditions: Vec<usize>,
    operator: AssembledOperator<R, V>,
    loads: PreparedLoads<R, V, D>,
}

fn tensor_normal<L: Lane, const D: usize>(tensor: [[L; D]; D], normal: [L::Real; D]) -> [L; D] {
    core::array::from_fn(|column| {
        (0..D).fold(L::ZERO, |sum, row| {
            sum + L::constant(normal[row]) * tensor[row][column]
        })
    })
}

/// Project rounding-level asymmetry and admit the cell-owned SPD tensor.
fn validate_tensor<R: RealScalar, const D: usize>(
    tensor: &mut [[R; D]; D],
) -> Result<(), SciOxideError> {
    let mut factor = [[R::ZERO; D]; D];
    crate::validation::positive_tensor(tensor.as_flattened_mut(), factor.as_flattened_mut(), D)
}

/// Minimal-correction half-cell flux: K n = a d + transverse.
fn half_face<L: Lane, const D: usize>(
    kn: [L; D],
    d: [L::Real; D],
) -> Result<(L, [L; D]), SciOxideError> {
    let length_sq = dot(d, d);
    let a = dot(kn, d.map(L::constant)) / L::constant(length_sq);
    if length_sq.is_finite() && length_sq > L::Real::ZERO && !a.all_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "assembled operator entry is unrepresentable",
        ));
    }
    if !length_sq.is_finite() || length_sq <= L::Real::ZERO || a.primal() <= L::Real::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "FVM half-cell transmissibility is non-positive",
        ));
    }
    Ok((a, core::array::from_fn(|i| kn[i] - a * L::constant(d[i]))))
}

fn preflight<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    cells: usize,
    faces: usize,
    limits: SpatialLimits,
) -> Result<usize, SciOxideError> {
    let max_terms = checked_add(
        checked_mul(8, D, scioxide_core::quantity!("fvm_stencil_bound"))?,
        2,
        scioxide_core::quantity!("fvm_stencil_bound"),
    )?;
    let entry_bound = checked_add(
        checked_mul(
            checked_mul(faces, 2, scioxide_core::quantity!("fvm_entry_bound"))?,
            max_terms,
            scioxide_core::quantity!("fvm_entry_bound"),
        )?,
        cells,
        scioxide_core::quantity!("fvm_entry_bound"),
    )?;
    let cell_bytes = checked_mul(
        cells,
        2 * size_of::<[R; D]>()
            + size_of::<[[R; D]; D]>()
            + 2 * size_of::<R>()
            + size_of::<V>()
            + size_of::<Vec<usize>>()
            + 2 * D * size_of::<usize>()
            + size_of::<Gradient<R, D>>()
            + (8 * D + 4) * size_of::<(usize, [R; D])>(),
        scioxide_core::quantity!("fvm_storage_bound"),
    )?;
    let face_bytes = checked_mul(
        faces,
        size_of::<FaceGeometry<R, D>>()
            + size_of::<FaceFlux<R>>()
            + 2 * max_terms * size_of::<(usize, R)>()
            + size_of::<BoundaryPoint<R, D>>()
            + size_of::<Option<BoundaryLaw<R>>>(),
        scioxide_core::quantity!("fvm_storage_bound"),
    )?;
    let triplet_bytes = checked_mul(
        entry_bound,
        2 * (2 * size_of::<usize>() + size_of::<V>()) + size_of::<u32>(),
        scioxide_core::quantity!("fvm_storage_bound"),
    )?;
    let storage_bound = checked_add(
        checked_add(
            checked_add(
                cell_bytes,
                face_bytes,
                scioxide_core::quantity!("fvm_storage_bound"),
            )?,
            triplet_bytes,
            scioxide_core::quantity!("fvm_storage_bound"),
        )?,
        4096,
        scioxide_core::quantity!("fvm_temporary_reconstruction_bound"),
    )?;
    limits.admit(cells, entry_bound, storage_bound)?;
    Ok(entry_bound)
}

fn gradient<L: Lane, const D: usize>(
    cell: usize,
    geometry: &Geometry<L::Real, D>,
    boundaries: &[Option<BoundaryLaw<L>>],
    tensors: &[[[L; D]; D]],
) -> Result<Gradient<L, D>, SciOxideError> {
    struct Constraint<L, const D: usize> {
        row: [L; D],
        other: Option<usize>,
        self_weight: L,
        site: Option<usize>,
    }
    let mut constraints = try_with_capacity(
        2 * D + 2,
        scioxide_core::quantity!("fvm_gradient_constraints"),
    )?;
    for &face_id in &geometry.cell_faces[cell] {
        let face = &geometry.faces[face_id];
        let other = if face.left == Some(cell) {
            face.right
        } else {
            face.left
        };
        if let Some(other) = other {
            constraints.push(Constraint {
                row: difference(geometry.centers[other], geometry.centers[cell]).map(L::constant),
                other: Some(other),
                self_weight: -L::ONE,
                site: None,
            });
        } else {
            let law = boundaries[face_id].ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("boundaries"),
                    "missing face condition",
                )
            })?;
            let mut normal = face.normal;
            if face.right == Some(cell) {
                for value in &mut normal {
                    *value = -*value;
                }
            }
            let displacement = difference(face.point, geometry.centers[cell]);
            let kn = tensor_normal(tensors[cell], normal);
            let row: [L; D] = core::array::from_fn(|i| {
                law.alpha * L::constant(displacement[i]) + law.beta * kn[i]
            });
            if row.iter().all(|value| value.primal() == L::Real::ZERO) {
                return Err(invalid(
                    scioxide_core::argument!("boundaries"),
                    "Robin coefficient makes the boundary gradient constraint degenerate",
                ));
            }
            constraints.push(Constraint {
                row,
                other: None,
                self_weight: -law.alpha,
                site: Some(law.site),
            });
        }
    }
    let mut rows = try_with_capacity(
        constraints.len(),
        scioxide_core::quantity!("fvm_gradient_rows"),
    )?;
    rows.extend(constraints.iter().map(|constraint| constraint.row));
    let inverse = inverse_normal_matrix(&rows)?;
    let mut terms = try_with_capacity(
        2 * constraints.len(),
        scioxide_core::quantity!("fvm_gradient_terms"),
    )?;
    let mut constant = try_with_capacity(
        constraints.len(),
        scioxide_core::quantity!("fvm_gradient_constants"),
    )?;
    for constraint in &constraints {
        let length_sq = dot(constraint.row, constraint.row);
        let weights: [L; D] = core::array::from_fn(|axis| {
            (0..D).fold(L::ZERO, |sum, j| {
                sum + inverse[axis][j] * constraint.row[j] / length_sq
            })
        });
        if let Some(other) = constraint.other {
            terms.push((other, weights));
        }
        terms.push((
            cell,
            core::array::from_fn(|axis| weights[axis] * constraint.self_weight),
        ));
        if let Some(site) = constraint.site {
            constant.push((site, weights));
        }
    }
    if terms
        .iter()
        .chain(&constant)
        .any(|(_, weights)| weights.iter().any(|value| !value.all_finite()))
    {
        return Err(invalid(
            scioxide_core::argument!("mesh"),
            "gradient reconstruction is unrepresentable",
        ));
    }
    Ok(Gradient { terms, constant })
}

fn face_flux<L: Lane, const D: usize>(
    face: &FaceGeometry<L::Real, D>,
    geometry: &Geometry<L::Real, D>,
    gradients: &[Gradient<L, D>],
    boundary: Option<BoundaryLaw<L>>,
    tensors: &[[[L; D]; D]],
) -> Result<FaceFlux<L>, SciOxideError> {
    let mut terms = try_with_capacity(8 * D + 2, scioxide_core::quantity!("fvm_face_stencil"))?;
    let mut constant = try_with_capacity(4 * D + 2, scioxide_core::quantity!("fvm_face_data"))?;
    let area = L::constant(face.area);
    if let (Some(left), Some(right)) = (face.left, face.right) {
        let (a, transverse_left) = half_face(
            tensor_normal(tensors[left], face.normal),
            difference(face.point, geometry.centers[left]),
        )?;
        let (b, transverse_right) = half_face(
            tensor_normal(tensors[right], face.normal),
            difference(geometry.centers[right], face.point),
        )?;
        // Eliminate the common face value by continuity of the two half-fluxes.
        // Normalize before adding conductances to avoid overflow in a+b.
        let scale = a.larger(b);
        let total = a / scale + b / scale;
        let weights = [(b / scale) / total, (a / scale) / total];
        for ((&cell, transverse), weight) in [left, right]
            .iter()
            .zip([transverse_left, transverse_right])
            .zip(weights)
        {
            let gradient = &gradients[cell];
            for &(neighbor, coefficients) in &gradient.terms {
                terms.push((neighbor, -area * weight * dot(transverse, coefficients)));
            }
            for &(site, coefficients) in &gradient.constant {
                constant.push((site, -area * weight * dot(transverse, coefficients)));
            }
        }
        let jump = (a.smaller(b) / (L::ONE + a.smaller(b) / scale)).scaled(face.area)?;
        if jump.primal() <= L::Real::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "FVM face transmissibility lost positivity",
            ));
        }
        terms.push((left, jump));
        terms.push((right, -jump));
    } else {
        let cell = face
            .left
            .or(face.right)
            .ok_or_else(|| invalid(scioxide_core::argument!("mesh"), "face has no cell"))?;
        let law = boundary.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("boundaries"),
                "missing boundary condition",
            )
        })?;
        let orientation = if face.left == Some(cell) {
            L::ONE
        } else {
            -L::ONE
        };
        let kn = tensor_normal(tensors[cell], face.normal);
        let outward_kn = core::array::from_fn(|i| orientation * kn[i]);
        let displacement = difference(face.point, geometry.centers[cell]);
        half_face(outward_kn, displacement)?;
        if law.alpha.exactly_zero() {
            // Prescribed conormal flux is independent of every cell value.
            constant.push((law.site, -area * orientation / law.beta));
        } else {
            let row: [L; D] = core::array::from_fn(|axis| {
                law.alpha * L::constant(displacement[axis]) + law.beta * outward_kn[axis]
            });
            let length_sq = dot(row, row);
            if !length_sq.all_finite() || length_sq.primal() <= L::Real::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "FVM boundary face elimination is singular",
                ));
            }
            let correction = dot(kn, row) / length_sq;
            let gradient = &gradients[cell];
            for &(neighbor, weights) in &gradient.terms {
                terms.push((
                    neighbor,
                    -area * (dot(kn, weights) - correction * dot(row, weights)),
                ));
            }
            terms.push((cell, area * correction * law.alpha));
            constant.push((law.site, -area * correction));
            for &(site, weights) in &gradient.constant {
                constant.push((
                    site,
                    -area * (dot(kn, weights) - correction * dot(row, weights)),
                ));
            }
        }
    }
    if terms
        .iter()
        .chain(&constant)
        .any(|(_, coefficient)| !coefficient.all_finite())
    {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "assembled operator entry is unrepresentable",
        ));
    }
    Ok(FaceFlux {
        left: face.left,
        right: face.right,
        terms,
        constant,
    })
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> FvmCore<R, V, D> {
    fn new<Region>(
        geometry: Geometry<R, D>,
        coefficients: &dyn FvmCoefficients<R, V, D>,
        boundaries: &BoundaryConditions<R, V, D, Region>,
        boundary_face_id: impl Fn(usize) -> Result<usize, SciOxideError>,
        entry_bound: usize,
    ) -> Result<Self, SciOxideError> {
        if boundaries.components() != 1 {
            return Err(invalid(
                scioxide_core::argument!("boundaries"),
                "cell-centred finite volumes are scalar",
            ));
        }
        boundaries.require_real_robin()?;
        let count = geometry.centers.len();
        let mut sites = try_with_capacity(0, scioxide_core::quantity!("fvm_boundary_sites"))?;
        let mut site_conditions =
            try_with_capacity(0, scioxide_core::quantity!("fvm_boundary_sites"))?;
        let mut laws = try_with_capacity(
            geometry.faces.len(),
            scioxide_core::quantity!("fvm_boundary_laws"),
        )?;
        let mut anchored = false;
        for (index, face) in geometry.faces.iter().enumerate() {
            if face.left.is_some() && face.right.is_some() {
                laws.push(None);
                continue;
            }
            let id = boundary_face_id(index)?;
            let condition = boundaries.slot(id, 0)?;
            let (alpha, beta) = match boundaries.kind_at(id, 0)? {
                BoundaryKind::Dirichlet => (R::ONE, R::ZERO),
                BoundaryKind::Natural => (R::ZERO, R::ONE),
                BoundaryKind::Robin { alpha } => (alpha.real_component(), R::ONE),
            };
            anchored |= alpha != R::ZERO;
            let outward = if face.left.is_some() { R::ONE } else { -R::ONE };
            sites.push(BoundaryPoint {
                point: face.point,
                normal: core::array::from_fn(|axis| outward * face.normal[axis]),
                component: 0,
            });
            site_conditions.push(condition);
            laws.push(Some(BoundaryLaw {
                alpha,
                beta,
                site: sites.len() - 1,
            }));
        }
        let mut tensors = try_with_capacity(count, scioxide_core::quantity!("fvm_diffusion"))?;
        for &point in &geometry.centers {
            let mut tensor = coefficients.diffusion(point)?;
            validate_tensor(&mut tensor)?;
            tensors.push(tensor);
        }
        let mut gradients = try_with_capacity(count, scioxide_core::quantity!("fvm_gradients"))?;
        for cell in 0..count {
            gradients.push(gradient(cell, &geometry, &laws, &tensors)?);
        }
        let constrained = try_filled(count, false, scioxide_core::quantity!("fvm_constraints"))?;
        let mut builder = OperatorBuilder::new(constrained, entry_bound)?;
        let mut reactive = false;
        for (cell, &point) in geometry.centers.iter().enumerate() {
            let value = finite(coefficients.reaction(point)?, NonFiniteOrigin::Callback)?;
            reactive |= value != V::ZERO;
            builder.push(
                cell,
                cell,
                entry(value * V::from_real(geometry.volumes[cell]))?,
            )?;
        }
        let mut faces = try_with_capacity(
            geometry.faces.len(),
            scioxide_core::quantity!("fvm_face_laws"),
        )?;
        let mut boundary_entries: Vec<(usize, usize, R)> =
            try_with_capacity(0, scioxide_core::quantity!("fvm_boundary_loads"))?;
        for (id, face) in geometry.faces.iter().enumerate() {
            let flux = face_flux(face, &geometry, &gradients, laws[id], &tensors)?;
            for &(column, coefficient) in &flux.terms {
                let value = V::from_real(coefficient);
                if let Some(left) = flux.left {
                    builder.push(left, column, value)?;
                }
                if let Some(right) = flux.right {
                    builder.push(right, column, -value)?;
                }
            }
            for &(site, coefficient) in &flux.constant {
                for (cell, sign) in [(flux.left, -R::ONE), (flux.right, R::ONE)] {
                    if let Some(cell) = cell {
                        boundary_entries.try_reserve(1).map_err(|_| {
                            SciOxideError::allocation_failure(
                                scioxide_core::quantity!("fvm_boundary_loads"),
                                boundary_entries.len() + 1,
                            )
                        })?;
                        boundary_entries.push((site, cell, sign * coefficient));
                    }
                }
            }
            faces.push(flux);
        }
        boundary_entries.sort_by_key(|&(site, cell, _)| (site, cell));
        let mut loads = PreparedLoads::new(count)?;
        let mut cursor = 0;
        for (site, &point) in sites.iter().enumerate() {
            // Data sites are pushed in site order, so site indices agree.
            let start = cursor;
            while cursor < boundary_entries.len() && boundary_entries[cursor].0 == site {
                cursor += 1;
            }
            loads.push_data(
                point,
                site_conditions[site],
                boundary_entries[start..cursor]
                    .iter()
                    .map(|&(_, cell, coefficient)| (cell, coefficient)),
            )?;
        }
        for (cell, &point) in geometry.centers.iter().enumerate() {
            loads.push_source(
                FieldPoint {
                    point,
                    component: 0,
                },
                [(cell, geometry.volumes[cell])],
            )?;
        }
        let (matrix, _, constrained) = builder.finish()?;
        let mut mass = try_with_capacity(count, scioxide_core::quantity!("fvm_mass"))?;
        mass.extend_from_slice(&geometry.volumes);
        Ok(Self {
            geometry,
            tensors,
            laws,
            faces,
            sites,
            site_conditions,
            operator: AssembledOperator {
                bounds: crate::kernels::RowBounds::new(matrix.values(), matrix.row_ptr()),
                compact_columns: crate::kernels::compact_columns(&matrix)?,
                matrix,
                constrained,
                mass,
                components: 1,
                constant_modes: !anchored && !reactive,
                entry_bound,
            },
            loads: loads.finish(None),
        })
    }

    fn cells(&self) -> usize {
        self.geometry.volumes.len()
    }

    /// Exact tangent of the operator and of the boundary-data load weights for a
    /// signed diffusion-tensor/reaction derivative field and real Robin coefficient
    /// derivatives (one per condition), through forward-mode face laws at the
    /// prepared tensors.
    fn tangent(
        &self,
        coefficients: Option<&dyn FvmCoefficients<R, V, D>>,
        robin: &[R],
    ) -> Result<(CsrMatrix<V>, PreparedLoads<R, V, D>), SciOxideError> {
        let count = self.cells();
        let mut tensors = try_with_capacity(count, scioxide_core::quantity!("fvm_diffusion"))?;
        for (&base, &point) in self.tensors.iter().zip(&self.geometry.centers) {
            let mut derivative = match coefficients {
                Some(coefficients) => coefficients.diffusion(point)?,
                None => [[R::ZERO; D]; D],
            };
            crate::validation::hermitian_part(derivative.as_flattened_mut(), D)?;
            tensors.push(core::array::from_fn(|i| {
                core::array::from_fn(|j| Tangent {
                    value: base[i][j],
                    derivative: derivative[i][j],
                })
            }));
        }
        let mut laws = try_with_capacity(self.laws.len(), scioxide_core::quantity!("fvm_laws"))?;
        for law in &self.laws {
            laws.push(match *law {
                Some(law) => Some(BoundaryLaw {
                    alpha: Tangent {
                        value: law.alpha,
                        derivative: crate::spatial::finite_callback(
                            *robin.get(self.site_conditions[law.site]).ok_or_else(|| {
                                invalid(
                                    scioxide_core::argument!("robin"),
                                    "one Robin derivative per condition",
                                )
                            })?,
                        )?,
                    },
                    beta: Tangent::constant(law.beta),
                    site: law.site,
                }),
                None => None,
            });
        }
        let mut gradients = try_with_capacity(count, scioxide_core::quantity!("fvm_gradients"))?;
        for cell in 0..count {
            gradients.push(gradient(cell, &self.geometry, &laws, &tensors)?);
        }
        let constrained = try_filled(count, false, scioxide_core::quantity!("fvm_constraints"))?;
        let mut builder = OperatorBuilder::with_elimination(
            constrained,
            self.operator.entry_bound,
            Elimination::Tangent,
        )?;
        if let Some(coefficients) = coefficients {
            for (cell, &point) in self.geometry.centers.iter().enumerate() {
                let value = finite(coefficients.reaction(point)?, NonFiniteOrigin::Callback)?;
                builder.push(
                    cell,
                    cell,
                    entry(value * V::from_real(self.geometry.volumes[cell]))?,
                )?;
            }
        }
        let mut entries: Vec<(usize, usize, R)> =
            try_with_capacity(0, scioxide_core::quantity!("fvm_boundary_loads"))?;
        for (id, face) in self.geometry.faces.iter().enumerate() {
            let flux = face_flux(face, &self.geometry, &gradients, laws[id], &tensors)?;
            for &(column, coefficient) in &flux.terms {
                let value = V::from_real(coefficient.derivative);
                if let Some(left) = flux.left {
                    builder.push(left, column, value)?;
                }
                if let Some(right) = flux.right {
                    builder.push(right, column, -value)?;
                }
            }
            for &(site, coefficient) in &flux.constant {
                for (cell, sign) in [(flux.left, -R::ONE), (flux.right, R::ONE)] {
                    if let Some(cell) = cell {
                        entries.try_reserve(1).map_err(|_| {
                            SciOxideError::allocation_failure(
                                scioxide_core::quantity!("fvm_boundary_loads"),
                                entries.len() + 1,
                            )
                        })?;
                        entries.push((site, cell, sign * coefficient.derivative));
                    }
                }
            }
        }
        entries.sort_by_key(|&(site, cell, _)| (site, cell));
        let mut loads = PreparedLoads::new(count)?;
        let mut cursor = 0;
        for (site, &point) in self.sites.iter().enumerate() {
            let start = cursor;
            while cursor < entries.len() && entries[cursor].0 == site {
                cursor += 1;
            }
            loads.push_data(
                point,
                self.site_conditions[site],
                entries[start..cursor]
                    .iter()
                    .map(|&(_, cell, coefficient)| (cell, coefficient)),
            )?;
        }
        Ok((builder.finish()?.0, loads.finish(None)))
    }

    fn fluxes<Region>(
        &self,
        boundaries: &BoundaryConditions<R, V, D, Region>,
        values: &[V],
        time: R,
        output: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError> {
        for (actual, expected) in [
            (values.len(), self.cells()),
            (output.len(), self.faces.len()),
            (scratch.len(), self.faces.len()),
        ] {
            if actual != expected {
                return Err(SciOxideError::dimension_mismatch(
                    Dimensions::one(expected),
                    Dimensions::one(actual),
                ));
            }
        }
        if values.iter().any(|value| !value.has_finite_components()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "cell values contain a non-finite component",
            ));
        }
        for (id, face) in self.faces.iter().enumerate() {
            let mut flux = V::ZERO;
            for &(cell, coefficient) in &face.terms {
                flux = finite(
                    flux + V::from_real(coefficient) * values[cell],
                    NonFiniteOrigin::Intermediate,
                )?;
            }
            for &(site, coefficient) in &face.constant {
                let g = boundaries.evaluate(self.site_conditions[site], self.sites[site], time)?;
                flux = finite(
                    flux + V::from_real(coefficient) * g,
                    NonFiniteOrigin::Intermediate,
                )?;
            }
            scratch[id] = flux;
        }
        output.copy_from_slice(scratch);
        Ok(())
    }
}

/// Conservative spatial FVM plan on bounded line, quadrilateral or hexahedral cells.
///
/// Unknowns are cell values in axis-zero-fast order and `D` is one to three.
/// Preparation samples coefficients once and keeps boundary data symbolic: loads
/// and face fluxes evaluate it at the requested time. Natural data is the outward
/// conormal flux `K∇u·n` and Robin data `K∇u·n + αu = g` with real `α`; Dirichlet
/// values enter through the boundary face law, so no unknown is constrained.
/// Application uses prepared rows and ordinary rounding with range-safe replay;
/// errors leave output unchanged and prepared applications allocate nothing.
///
/// ```
/// use scioxide_pde::{BoundaryData, BoundaryKind, ConstantDiffusion, EllipticProblem,
///     FieldData, StructuredBoundaries, StructuredFdmPlan, StructuredFvmPlan};
/// let grid = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [9])?;
/// let mesh = grid.mesh().clone();
/// let boundaries = StructuredBoundaries::uniform(
///     &mesh, 1, BoundaryKind::Dirichlet, BoundaryData::zero())?;
/// let fvm = StructuredFvmPlan::new(mesh, &ConstantDiffusion::new(1.0), &boundaries)?;
/// let u = EllipticProblem::new(&fvm, &FieldData::zero()).solve()?;
/// assert!(u.iter().all(|x: &f64| x.abs() < 1e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct StructuredFvmPlan<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    mesh: CellMesh<R, D>,
    boundaries: StructuredBoundaries<R, V, D>,
    core: FvmCore<R, V, D>,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StructuredFvmPlan<R, V, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        mesh: CellMesh<R, D>,
        coefficients: &dyn FvmCoefficients<R, V, D>,
        boundaries: &StructuredBoundaries<R, V, D>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(mesh, coefficients, boundaries, SpatialLimits::default())
    }

    /// Prepare a spatial operator under explicit resource ceilings.
    /// # Errors
    /// Rejects periodic topology, foreign or multi-component boundaries, complex
    /// Robin coefficients, invalid geometry or coefficients, rank-deficient
    /// reconstruction, size limits and allocation failures.
    pub fn with_options(
        mesh: CellMesh<R, D>,
        coefficients: &dyn FvmCoefficients<R, V, D>,
        boundaries: &StructuredBoundaries<R, V, D>,
        limits: SpatialLimits,
    ) -> Result<Self, SciOxideError> {
        boundaries
            .mesh_token()
            .ensure_compatible(mesh.mesh_token())?;
        for axis in 0..D {
            if mesh.grid().axis_topology(axis)? != AxisTopology::Bounded {
                return Err(invalid(
                    scioxide_core::argument!("mesh"),
                    "periodic structured finite volumes are unsupported",
                ));
            }
        }
        let entry_bound = preflight::<R, V, D>(
            mesh.topology_sizes().cells(),
            mesh.topology_sizes().faces(),
            limits,
        )?;
        let geometry = structured_geometry(&mesh)?;
        let core = FvmCore::new(geometry, coefficients, boundaries, Ok, entry_bound)?;
        Ok(Self {
            mesh,
            boundaries: boundaries.clone(),
            core,
        })
    }

    /// Return one owned face field for structured normal axis `A` at `time`, under
    /// sequential native execution. The field's sign follows the increasing
    /// coordinate direction.
    /// # Errors
    /// Propagates identity, shape, numerical and allocation failures.
    pub fn face_fluxes<const A: usize>(
        &self,
        values: FieldView<'_, V, Cell, D>,
        time: R,
    ) -> Result<FaceField<V, D, A>, SciOxideError> {
        if A >= D {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("fvm_face_axis"),
                A,
                D,
            ));
        }
        let mut output = try_filled(
            self.core.faces.len(),
            V::ZERO,
            scioxide_core::quantity!("fvm_face_output"),
        )?;
        self.face_fluxes_into(
            values,
            time,
            &mut output,
            &mut self.workspace()?,
            ExecutionContext::sequential_native(),
        )?;
        let mut offset = 0;
        for axis in 0..A {
            offset = checked_add(
                offset,
                self.mesh.topology_sizes().faces_normal_to(axis)?,
                scioxide_core::quantity!("fvm_face_offset"),
            )?;
        }
        let length = self.mesh.topology_sizes().faces_normal_to(A)?;
        let mut axis_values =
            try_with_capacity(length, scioxide_core::quantity!("fvm_face_field"))?;
        axis_values.extend_from_slice(&output[offset..offset + length]);
        FaceField::new(&self.mesh, axis_values)
    }
}

/// Conservative spatial FVM plan on intervals, triangles or tetrahedra.
/// `D` is one to three and face fluxes follow the mesh's canonical face IDs.
#[derive(Clone)]
pub struct SimplexFvmPlan<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    mesh: SimplexMesh<R, D>,
    boundaries: TaggedBoundaries<R, V, D>,
    core: FvmCore<R, V, D>,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> SimplexFvmPlan<R, V, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        mesh: SimplexMesh<R, D>,
        coefficients: &dyn FvmCoefficients<R, V, D>,
        boundaries: &TaggedBoundaries<R, V, D>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(mesh, coefficients, boundaries, SpatialLimits::default())
    }

    /// Prepare a spatial operator under explicit resource ceilings.
    /// # Errors
    /// Propagates boundary, tensor, geometry, resource and allocation errors.
    pub fn with_options(
        mesh: SimplexMesh<R, D>,
        coefficients: &dyn FvmCoefficients<R, V, D>,
        boundaries: &TaggedBoundaries<R, V, D>,
        limits: SpatialLimits,
    ) -> Result<Self, SciOxideError> {
        boundaries
            .mesh_token()
            .ensure_compatible(mesh.mesh_token())?;
        let entry_bound = preflight::<R, V, D>(mesh.cell_count(), mesh.face_count(), limits)?;
        let geometry = simplex_geometry(&mesh)?;
        let core = FvmCore::new(geometry, coefficients, boundaries, Ok, entry_bound)?;
        Ok(Self {
            mesh,
            boundaries: boundaries.clone(),
            core,
        })
    }

    /// Reconstruct all oriented canonical simplex-face fluxes at `time` under
    /// sequential native execution.
    /// # Errors
    /// Propagates identity, shape, numerical and allocation failures.
    pub fn face_fluxes(
        &self,
        values: FieldView<'_, V, Cell, D>,
        time: R,
    ) -> Result<UnstructuredFaceField<V, D>, SciOxideError> {
        let mut output = try_filled(
            self.core.faces.len(),
            V::ZERO,
            scioxide_core::quantity!("fvm_face_output"),
        )?;
        self.face_fluxes_into(
            values,
            time,
            &mut output,
            &mut self.workspace()?,
            ExecutionContext::sequential_native(),
        )?;
        UnstructuredFaceField::new(&self.mesh, output)
    }
}

/// Plan-fixed staging of transactional FVM face-flux reconstruction.
#[derive(Debug, Clone)]
pub struct FvmFluxWorkspace<V> {
    staged: Vec<V>,
}

macro_rules! common_plan {
    ($plan:ident, $mesh:ident) => {
        impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> $plan<R, V, D> {
            /// Exact source mesh and revision.
            pub const fn mesh(&self) -> &$mesh<R, D> {
                &self.mesh
            }

            /// Conservative upper bound on emitted COO triplets.
            pub const fn entry_bound(&self) -> usize {
                self.core.operator.entry_bound
            }

            /// Cell measures in mesh cell order.
            pub fn volumes(&self) -> &[R] {
                &self.core.geometry.volumes
            }

            /// Check an integrated right-hand side against the known constant mode.
            /// This algebraic check is not a uniqueness or accuracy certificate.
            /// # Errors
            /// Rejects incompatible loads, unavailable mode and invalid values.
            pub fn check_constant_mode_compatibility(
                &self,
                rhs: &[V],
                tolerance: Tolerance<R>,
            ) -> Result<(), SciOxideError> {
                if rhs.len() != self.core.cells() {
                    return Err(SciOxideError::dimension_mismatch(
                        Dimensions::one(self.core.cells()),
                        Dimensions::one(rhs.len()),
                    ));
                }
                self.core
                    .operator
                    .check_constant_mode_compatibility(rhs, tolerance)
            }

            /// Apply to a borrowed cell field and return an owned cell field.
            /// # Errors
            /// Propagates identity, shape, numerical and allocation failures.
            pub fn apply_field(
                &self,
                values: FieldView<'_, V, Cell, D>,
            ) -> Result<CellField<V, D>, SciOxideError> {
                self.mesh
                    .mesh_token()
                    .ensure_compatible(values.mesh_token())?;
                let mut output = crate::discretization::zeros(self.core.cells())?;
                let mut workspace = crate::discretization::zeros(self.core.cells())?;
                self.core
                    .operator
                    .apply(values.values(), &mut output, &mut workspace, None)?;
                CellField::new(&self.mesh, output)
            }

            /// Face-flux staging sized for this plan; it never grows.
            /// # Errors
            /// Returns an allocation failure.
            pub fn workspace(&self) -> Result<FvmFluxWorkspace<V>, SciOxideError> {
                Ok(FvmFluxWorkspace {
                    staged: crate::discretization::zeros(self.core.faces.len())?,
                })
            }

            /// Reconstruct integrated physical fluxes at `time` into canonical face order.
            /// Structured faces are axis-major and oriented toward increasing
            /// coordinates; simplex faces follow stored outward orientation from
            /// their first incident cell. Output is unchanged on error.
            /// # Errors
            /// Rejects foreign fields, invalid shapes, an undersized workspace,
            /// non-finite fluxes and cancellation or deadline stops.
            pub fn face_fluxes_into(
                &self,
                values: FieldView<'_, V, Cell, D>,
                time: R,
                output: &mut [V],
                workspace: &mut FvmFluxWorkspace<V>,
                execution: ExecutionContext<'_>,
            ) -> Result<(), SciOxideError> {
                self.mesh
                    .mesh_token()
                    .ensure_compatible(values.mesh_token())?;
                execution.check()?;
                self.core.fluxes(
                    &self.boundaries,
                    values.values(),
                    time,
                    output,
                    &mut workspace.staged,
                )
            }
        }

        impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearOperator<V>
            for $plan<R, V, D>
        {
            fn rows(&self) -> usize {
                self.core.cells()
            }
            fn cols(&self) -> usize {
                self.core.cells()
            }
            fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(self.core.cells())
            }
            fn apply_into(
                &self,
                input: &[V],
                output: &mut [V],
                workspace: &mut [V],
            ) -> Result<(), SciOxideError> {
                self.core.operator.apply(input, output, workspace, None)
            }
            fn apply_into_scoped(
                &self,
                input: &[V],
                output: &mut [V],
                workspace: &mut [V],
                context: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
            ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
                // Deterministic sequential rows with the caller's cancellation and deadline.
                self.core
                    .operator
                    .apply(input, output, workspace, Some(context.execution()))
                    .map(scioxide_core::Invocation::Evaluated)
            }
        }

        impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>
            SpatialDiscretization<R, V, D> for $plan<R, V, D>
        {
            type Field = CellField<V, D>;

            fn mesh_token(&self) -> &MeshToken {
                self.mesh.mesh_token()
            }
            fn components(&self) -> usize {
                1
            }
            fn dof_point(&self, unknown: usize) -> Result<FieldPoint<R, D>, SciOxideError> {
                let point = self
                    .core
                    .geometry
                    .centers
                    .get(unknown)
                    .copied()
                    .ok_or_else(|| {
                        SciOxideError::out_of_bounds(
                            scioxide_core::quantity!("fvm_cell"),
                            unknown,
                            self.core.cells(),
                        )
                    })?;
                Ok(FieldPoint {
                    point,
                    component: 0,
                })
            }
            fn constrained(&self) -> &[bool] {
                &self.core.operator.constrained
            }
            fn lumped_mass(&self) -> &[R] {
                &self.core.operator.mass
            }
            fn has_constant_modes(&self) -> bool {
                self.core.operator.constant_modes
            }
            fn assemble_csr(&self) -> Result<CsrMatrix<V>, SciOxideError> {
                Ok(self.core.operator.matrix.clone())
            }
            fn load_time_dependence(&self) -> crate::TimeDependence {
                self.boundaries.time_dependence()
            }
            fn load_into(
                &self,
                source: &FieldData<R, V, D>,
                time: R,
                out: &mut [V],
                scratch: &mut [V],
            ) -> Result<(), SciOxideError> {
                self.core
                    .loads
                    .load_transaction(&self.boundaries, source, time, out, scratch)
            }
            fn field(&self, values: &[V]) -> Result<Self::Field, SciOxideError> {
                let mut owned =
                    try_with_capacity(values.len(), scioxide_core::quantity!("fvm_cell_field"))?;
                owned.extend_from_slice(values);
                CellField::new(&self.mesh, owned)
            }
        }
        impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> crate::adjoint::Sealed
            for $plan<R, V, D>
        {
            fn preparation(&self) -> &alloc::sync::Arc<()> {
                &self.core.loads.preparation
            }
        }

        impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>
            crate::DifferentiableDiscretization<R, V, D> for $plan<R, V, D>
        {
            type Coefficients = dyn FvmCoefficients<R, V, D>;

            fn parameterize(
                &self,
                values: &[R],
                tangents: &[crate::PdeParameterTangent<'_, R, V, D, Self::Coefficients>],
            ) -> Result<crate::PdeParameterization<R, V, D>, SciOxideError> {
                crate::adjoint::parameterize(
                    self,
                    &self.boundaries,
                    &self.core.loads,
                    values,
                    tangents,
                    |coefficients, robin| {
                        let mut real =
                            try_with_capacity(robin.len(), scioxide_core::quantity!("robin"))?;
                        for &alpha in robin {
                            if alpha.imaginary_component() != R::ZERO {
                                return Err(invalid(
                                    scioxide_core::argument!("tangents"),
                                    "finite volumes require real Robin derivatives",
                                ));
                            }
                            real.push(alpha.real_component());
                        }
                        let (operator, weights) = self.core.tangent(coefficients, &real)?;
                        Ok((operator, Some(weights)))
                    },
                )
            }
        }
    };
}

common_plan!(StructuredFvmPlan, CellMesh);
common_plan!(SimplexFvmPlan, SimplexMesh);

#[cfg(test)]
#[path = "multidimensional_tests.rs"]
mod tests;
