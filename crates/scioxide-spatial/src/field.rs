//! Values attached to mesh vertices, cells, faces or edges.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{Dimensions, RealScalar, SciOxideError};

use crate::AdaptiveMesh;
use crate::{CellMesh, Extent, MeshToken, SimplexMesh};

/// Field location: mesh vertices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[must_use]
pub struct Vertex;

/// Field location: mesh cells.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[must_use]
pub struct Cell;

/// Field location: faces of a [`CellMesh`] normal to axis `A`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[must_use]
pub struct Face<const A: usize>;

/// Field location: edges of a [`CellMesh`] parallel to axis `A`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[must_use]
pub struct Edge<const A: usize>;

// Sealing keeps the location/mesh pairs closed. Field length validation
// trusts `entity_count` and `mesh_token`, so an external implementation
// could otherwise attach a field to a mesh whose entities it does not
// describe.
mod private {
    pub trait Sealed {}
    pub trait MeshSealed {}
}

impl private::Sealed for Vertex {}
impl private::Sealed for Cell {}
impl<const A: usize> private::Sealed for Face<A> {}
impl<const A: usize> private::Sealed for Edge<A> {}

/// A field location on a [`CellMesh`]: [`Vertex`], [`Cell`], [`Face`] or
/// [`Edge`].
///
/// The trait is sealed. Code can be generic over a location but cannot add
/// new ones.
pub trait FieldLocation<const D: usize>: private::Sealed + Copy + 'static {
    /// Returns the number of entities at this location along each axis.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the axis of a face or edge location is not less
    ///   than `D`.
    fn entity_extent<S: RealScalar>(mesh: &CellMesh<S, D>) -> Result<Extent<D>, SciOxideError>;
}

// Every location implements the same trait-owned operation.

impl<const D: usize> FieldLocation<D> for Vertex {
    fn entity_extent<S: RealScalar>(mesh: &CellMesh<S, D>) -> Result<Extent<D>, SciOxideError> {
        Ok(mesh.vertex_extent())
    }
}

impl<const D: usize> FieldLocation<D> for Cell {
    fn entity_extent<S: RealScalar>(mesh: &CellMesh<S, D>) -> Result<Extent<D>, SciOxideError> {
        Ok(mesh.cell_extent())
    }
}

impl<const D: usize, const A: usize> FieldLocation<D> for Face<A> {
    fn entity_extent<S: RealScalar>(mesh: &CellMesh<S, D>) -> Result<Extent<D>, SciOxideError> {
        validate_location_axis(A, D, scioxide_core::quantity!("face_field_normal_axis"))?;
        Extent::new(core::array::from_fn(|axis| {
            if axis == A {
                mesh.vertex_extent().as_array()[axis]
            } else {
                mesh.cell_extent().as_array()[axis]
            }
        }))
    }
}

impl<const D: usize, const A: usize> FieldLocation<D> for Edge<A> {
    fn entity_extent<S: RealScalar>(mesh: &CellMesh<S, D>) -> Result<Extent<D>, SciOxideError> {
        validate_location_axis(A, D, scioxide_core::quantity!("edge_field_direction"))?;
        Extent::new(core::array::from_fn(|axis| {
            if axis == A {
                mesh.cell_extent().as_array()[axis]
            } else {
                mesh.vertex_extent().as_array()[axis]
            }
        }))
    }
}

/// Field location: all faces of a [`SimplexMesh`] or [`AdaptiveMesh`].
///
/// On an adaptive mesh these are whole cell faces; a coarse face next to
/// finer cells is one face.
///
/// ```
/// use scioxide_spatial::{
///     AdaptationLimits, AdaptiveMesh, Point, TriangleMesh, UnstructuredEdgeField,
///     UnstructuredFaceField,
/// };
/// let simplex = TriangleMesh::new(
///     vec![
///         Point::new([0.0_f64, 0.0])?,
///         Point::new([1.0, 0.0])?,
///         Point::new([0.0, 1.0])?,
///     ],
///     vec![[0, 1, 2]],
/// )?;
/// let adaptive = AdaptiveMesh::from_simplex(&simplex, AdaptationLimits::new())?;
/// let simplex_faces = UnstructuredFaceField::<u32, 2>::from_fn(&simplex, |_| 0)?;
/// let adaptive_faces = UnstructuredFaceField::<u32, 2>::from_fn(&adaptive, |_| 0)?;
/// let edges = UnstructuredEdgeField::<u32, 2>::from_fn(&adaptive, |_| 0)?;
/// assert_eq!(simplex_faces.len(), adaptive_faces.len());
/// assert_eq!(edges.len(), adaptive.edge_count());
/// assert!(!simplex_faces.is_compatible_with(&adaptive));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// ```compile_fail
/// use scioxide_spatial::{FaceField, TriangleMesh};
/// fn wrong_axis(mesh: &TriangleMesh<f64>) {
///     let _ = FaceField::<f64, 2, 0>::new(mesh, vec![]);
/// }
/// ```
///
/// ```compile_fail
/// use scioxide_spatial::{UnstructuredFaceField, VertexField};
/// fn needs_vertices(_: &VertexField<f64, 2>) {}
/// fn wrong_location(faces: &UnstructuredFaceField<f64, 2>) {
///     needs_vertices(faces);
/// }
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct UnstructuredFace;
/// Field location: all edges of a [`SimplexMesh`] or [`AdaptiveMesh`],
/// including hanging edges of an adaptive mesh.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct UnstructuredEdge;

/// Meshes that support fields at location `L`.
///
/// The trait is sealed. Combinations without a meaning, such as axis-normal
/// faces on a simplex mesh, do not implement it, so fields for them do not
/// compile.
pub trait MeshEntities<L, const D: usize>: private::MeshSealed {
    /// Returns the token of this mesh revision.
    fn mesh_token(&self) -> &MeshToken;
    /// Returns the number of entities at location `L`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the axis of a face or edge location is not less
    ///   than `D`.
    /// - `SizeOverflow` if the count exceeds `usize`.
    fn entity_count(&self) -> Result<usize, SciOxideError>;
}
impl<S: RealScalar, const D: usize> private::MeshSealed for CellMesh<S, D> {}
impl<S: RealScalar, const D: usize> private::MeshSealed for SimplexMesh<S, D> {}
impl<S: RealScalar, const D: usize> private::MeshSealed for AdaptiveMesh<S, D> {}
impl<S: RealScalar, L: FieldLocation<D>, const D: usize> MeshEntities<L, D> for CellMesh<S, D> {
    fn mesh_token(&self) -> &MeshToken {
        self.mesh_token()
    }
    fn entity_count(&self) -> Result<usize, SciOxideError> {
        L::entity_extent(self)?.element_count()
    }
}
macro_rules! mesh_location {
    ($mesh:ident, $location:ty, $count:ident) => {
        impl<S: RealScalar, const D: usize> MeshEntities<$location, D> for $mesh<S, D> {
            fn mesh_token(&self) -> &MeshToken {
                self.mesh_token()
            }
            fn entity_count(&self) -> Result<usize, SciOxideError> {
                Ok(self.$count())
            }
        }
    };
}
mesh_location!(SimplexMesh, Vertex, vertex_count);
mesh_location!(AdaptiveMesh, Vertex, vertex_count);
mesh_location!(SimplexMesh, Cell, cell_count);
mesh_location!(AdaptiveMesh, Cell, cell_count);
mesh_location!(SimplexMesh, UnstructuredFace, face_count);
mesh_location!(AdaptiveMesh, UnstructuredFace, face_count);
mesh_location!(SimplexMesh, UnstructuredEdge, edge_count);
mesh_location!(AdaptiveMesh, UnstructuredEdge, edge_count);

/// Field on all faces of a simplex or adaptive mesh.
pub type UnstructuredFaceField<T, const D: usize> = Field<T, UnstructuredFace, D>;
/// Field on all edges of a simplex or adaptive mesh.
pub type UnstructuredEdgeField<T, const D: usize> = Field<T, UnstructuredEdge, D>;

/// Immutable values, one per mesh entity at location `L`, for one mesh
/// revision.
///
/// The location is part of the type, so a function that takes
/// `Field<T, Cell, D>` cannot receive vertex or face values. The value type
/// `T` is independent of the coordinate type of the mesh. Values are stored in
/// mesh entity order, axis 0 fastest on a [`CellMesh`]. The field stores the
/// mesh token, and operations that combine fields check it. Cloning shares the
/// values in O(1).
///
/// ```compile_fail
/// use scioxide_spatial::{Cell, Field, FieldLocation, Vertex};
///
/// fn combine<T, L: FieldLocation<2>>(_: &Field<T, L, 2>, _: &Field<T, L, 2>) {}
/// fn rejected(vertex: &Field<f64, Vertex, 2>, cell: &Field<f64, Cell, 2>) {
///     combine(vertex, cell);
/// }
/// ```
///
/// ```
/// use scioxide_spatial::{AxisTopology, Cell, CellField, CellMesh, CoordinateAxis, FieldView, Grid1d};
/// let mesh = CellMesh::new(Grid1d::new(
///     [CoordinateAxis::uniform(0.0_f64, 1.0, 3)?],
///     [AxisTopology::Bounded],
/// )?)?;
/// let field = CellField::<i32, 1>::from_fn(&mesh, |cell| cell as i32)?;
/// let doubled = field.try_map(|value| 2 * value)?;
/// let view = FieldView::<_, Cell, 1>::new(&mesh, doubled.values())?;
/// assert_eq!(view.values(), &[0, 2]);
/// assert!(CellField::<i32, 1>::new(&mesh, vec![0]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Field<T, L, const D: usize> {
    values: Arc<Vec<T>>,
    token: MeshToken,
    location: PhantomData<L>,
}

impl<T, L, const D: usize> Field<T, L, D> {
    /// Creates a field from one value per entity of `mesh` at location `L`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `values.len()` is not the number of entities.
    pub fn new<M: MeshEntities<L, D>>(mesh: &M, values: Vec<T>) -> Result<Self, SciOxideError> {
        let expected = Self::required_len(mesh)?;
        if values.len() != expected {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(expected),
                Dimensions::one(values.len()),
            ));
        }
        Ok(Self {
            values: Arc::new(values),
            token: mesh.mesh_token().clone(),
            location: PhantomData,
        })
    }

    /// Returns the number of values a field at location `L` on `mesh` needs.
    ///
    /// # Errors
    ///
    /// As [`MeshEntities::entity_count`].
    pub fn required_len<M: MeshEntities<L, D>>(mesh: &M) -> Result<usize, SciOxideError> {
        mesh.entity_count()
    }

    /// Creates a field by calling `value(i)` for each entity index `i`.
    ///
    /// # Errors
    ///
    /// As [`MeshEntities::entity_count`], and `AllocationFailure` if the values
    /// cannot be stored.
    pub fn from_fn<M: MeshEntities<L, D>, F: FnMut(usize) -> T>(
        mesh: &M,
        mut value: F,
    ) -> Result<Self, SciOxideError> {
        let len = Self::required_len(mesh)?;
        let mut values = try_with_capacity(len, scioxide_core::quantity!("mesh_field_values"))?;
        values.extend((0..len).map(&mut value));
        Self::new(mesh, values)
    }

    /// Returns the values in mesh entity order.
    pub fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    /// Returns the number of values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the field has no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the token of the mesh the field was built for.
    pub const fn mesh_token(&self) -> &MeshToken {
        &self.token
    }

    /// Returns a borrowed view of the field.
    pub fn view(&self) -> FieldView<'_, T, L, D> {
        FieldView {
            values: self.values(),
            token: &self.token,
            location: PhantomData,
        }
    }

    /// Returns whether the field was built for this revision of `mesh`.
    pub fn is_compatible_with<M: MeshEntities<L, D>>(&self, mesh: &M) -> bool {
        self.token.is_compatible_with(mesh.mesh_token())
    }

    /// Returns a field with `transform` applied to each value, on the same
    /// mesh and location.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the new values cannot be stored.
    pub fn try_map<U, F: FnMut(&T) -> U>(
        &self,
        mut transform: F,
    ) -> Result<Field<U, L, D>, SciOxideError> {
        let mut values = try_with_capacity(
            self.len(),
            scioxide_core::quantity!("mapped_mesh_field_values"),
        )?;
        values.extend(self.values().iter().map(&mut transform));
        Ok(Field {
            values: Arc::new(values),
            token: self.token.clone(),
            location: PhantomData,
        })
    }

    /// Returns a field whose values are `combine(a, b)` for corresponding
    /// values `a` of `self` and `b` of `other`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the two fields belong to different mesh revisions.
    /// - `AllocationFailure` if the new values cannot be stored.
    pub fn try_zip_map<U, V, F: FnMut(&T, &U) -> V>(
        &self,
        other: &Field<U, L, D>,
        mut combine: F,
    ) -> Result<Field<V, L, D>, SciOxideError> {
        // Equal tokens for the same location imply equal entity counts, so
        // the zip below cannot silently truncate either field.
        self.token.ensure_compatible(&other.token)?;
        let mut values = try_with_capacity(
            self.len(),
            scioxide_core::quantity!("combined_mesh_field_values"),
        )?;
        values.extend(
            self.values()
                .iter()
                .zip(other.values())
                .map(|(left, right)| combine(left, right)),
        );
        Ok(Field {
            values: Arc::new(values),
            token: self.token.clone(),
            location: PhantomData,
        })
    }
}

/// Borrowed values for one mesh revision at location `L`: the borrowed form of
/// [`Field`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct FieldView<'a, T, L, const D: usize> {
    values: &'a [T],
    token: &'a MeshToken,
    location: PhantomData<L>,
}

impl<'a, T, L, const D: usize> FieldView<'a, T, L, D> {
    /// Creates a view of one value per entity of `mesh` at location `L`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `values.len()` is not the number of entities.
    pub fn new<M: MeshEntities<L, D>>(mesh: &'a M, values: &'a [T]) -> Result<Self, SciOxideError> {
        let expected = Field::<T, L, D>::required_len(mesh)?;
        if values.len() != expected {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(expected),
                Dimensions::one(values.len()),
            ));
        }
        Ok(Self {
            values,
            token: mesh.mesh_token(),
            location: PhantomData,
        })
    }
}

impl<'a, T, L, const D: usize> FieldView<'a, T, L, D> {
    /// Copies the values into a [`Field`] for the same mesh and location.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the values cannot be stored.
    pub fn to_owned(self) -> Result<Field<T, L, D>, SciOxideError>
    where
        T: Clone,
    {
        let mut values = scioxide_core::backend::try_with_capacity(
            self.values.len(),
            scioxide_core::quantity!("owned_field_values"),
        )?;
        values.extend_from_slice(self.values);
        Ok(Field {
            values: Arc::new(values),
            token: self.token.clone(),
            location: PhantomData,
        })
    }

    /// Returns the values in mesh entity order.
    pub const fn values(self) -> &'a [T] {
        self.values
    }

    /// Returns the number of values.
    pub const fn len(self) -> usize {
        self.values.len()
    }

    /// Returns whether the view has no values.
    pub const fn is_empty(self) -> bool {
        self.values.is_empty()
    }

    /// Returns the token of the mesh the view was built for.
    pub const fn mesh_token(self) -> &'a MeshToken {
        self.token
    }
}

/// Field on mesh vertices.
pub type VertexField<T, const D: usize> = Field<T, Vertex, D>;
/// Field on mesh cells.
pub type CellField<T, const D: usize> = Field<T, Cell, D>;
/// Field on the faces of a [`CellMesh`] normal to axis `A`.
pub type FaceField<T, const D: usize, const A: usize> = Field<T, Face<A>, D>;
/// Field on the edges of a [`CellMesh`] parallel to axis `A`.
pub type EdgeField<T, const D: usize, const A: usize> = Field<T, Edge<A>, D>;

fn validate_location_axis(
    axis: usize,
    dimension: usize,
    collection: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    if axis < dimension {
        Ok(())
    } else {
        Err(SciOxideError::out_of_bounds(collection, axis, dimension))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{AxisTopology, CoordinateAxis, Grid2d, QuadMesh};
    use alloc::vec;

    fn build_mesh() -> QuadMesh<f32> {
        CellMesh::new(
            Grid2d::new(
                [
                    CoordinateAxis::uniform(0.0, 2.0, 3).unwrap(),
                    CoordinateAxis::uniform(5.0, 8.0, 4).unwrap(),
                ],
                [AxisTopology::Bounded; 2],
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn location_queries_have_distinct_checked_counts() {
        let mesh = build_mesh();
        assert_eq!(VertexField::<f64, 2>::required_len(&mesh).unwrap(), 12);
        assert_eq!(CellField::<f64, 2>::required_len(&mesh).unwrap(), 6);
        assert_eq!(FaceField::<f64, 2, 0>::required_len(&mesh).unwrap(), 9);
        assert_eq!(FaceField::<f64, 2, 1>::required_len(&mesh).unwrap(), 8);
        assert_eq!(EdgeField::<f64, 2, 0>::required_len(&mesh).unwrap(), 8);
        assert_eq!(EdgeField::<f64, 2, 1>::required_len(&mesh).unwrap(), 9);
        assert!(
            matches ! (& (FaceField :: < f64 , 2 , 2 > :: required_len (& mesh)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: OutOfBounds { .. }))
        );
    }

    #[test]
    fn field_values_are_independent_of_coordinate_scalar() {
        let mesh = build_mesh();
        let field = VertexField::<f64, 2>::from_fn(&mesh, |index| index as f64).unwrap();
        assert_eq!(
            field.values(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
        assert_eq!(field.view().mesh_token(), mesh.mesh_token());
    }

    #[test]
    fn construction_rejects_wrong_entity_count() {
        let mesh = build_mesh();
        assert!(
            matches ! (& (CellField :: < f64 , 2 > :: new (& mesh , vec ! [0.0 ; 12])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: DimensionMismatch { .. }))
        );
    }

    #[test]
    fn same_location_composition_checks_exact_mesh_token() {
        let mesh = build_mesh();
        let left = CellField::<f64, 2>::from_fn(&mesh, |index| index as f64).unwrap();
        let right = CellField::<f64, 2>::from_fn(&mesh, |_| 2.0).unwrap();
        let sum = left.try_zip_map(&right, |a, b| a + b).unwrap();
        assert_eq!(sum.values(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let independent = build_mesh();
        let unrelated = CellField::<f64, 2>::from_fn(&independent, |_| 1.0).unwrap();
        assert!(
            matches ! (& (left . try_zip_map (& unrelated , | a , b | a + b)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );

        let refined = CellMesh::new(mesh.grid().refined([2, 1]).unwrap()).unwrap();
        assert!(!left.is_compatible_with(&refined));
    }
}
