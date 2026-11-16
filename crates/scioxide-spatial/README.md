# scioxide-spatial

Computational geometry for Rust.
The crate builds structured and simplicial meshes with typed fields, refines them locally and partitions them.
It decides orientation and in-sphere tests exactly, computes convex hulls, Delaunay triangulations, Voronoi diagrams and halfspace intersections, and locates points in them.
It also represents and interpolates 3-D rotations and rigid motions, computes distance matrices, and answers nearest-neighbor queries with KD trees, for `f32` and `f64` coordinates.

## Main entry points

- Points and index spaces: `Point`, `Extent`, `Index`, `PointSet`.
- Structured meshes: `CoordinateAxis`, `StructuredGrid`, `CellMesh`, `BoundarySet`.
- Simplex meshes: `IntervalMesh`, `TriangleMesh`, `TetrahedronMesh`, `SimplexBoundaryTags`.
- Fields on meshes: `Field`, `FieldView`, `VertexField`, `CellField`.
- Exact predicates: `orient2d`, `orient3d`, `incircle`, `insphere`, `SimplexPredicateWorkspace`.
- Rotations and rigid motions: `Rotation`, `RigidTransform`, `Slerp`, `RotationSpline`, `estimate_rigid_transform`.
- Distances: `distance`, `pairwise`, `condensed`, `PreparedDistance`, `directed_hausdorff`.
- Nearest-neighbor search: `KdTree`.
- Tessellations: `ConvexHull`, `Delaunay`, `Voronoi`, `SphericalVoronoi`, `HalfspaceIntersection`.
- Adaptive meshes: `AdaptiveMesh`, `TopologyMap`, `MeshPartition`, `RepartitionMap`.

## Feature flags

- `std` (default): threads and clocks from `scioxide-core`.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-spatial) for a quick start, a guide to choosing a method, and the error and accuracy rules.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
