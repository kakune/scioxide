use super::*;

fn check_primal_and_dual<S: RealScalar>(values: &[f64], d: usize) {
    let values: Vec<S> = values
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let dual = with_points(&values, d, |p| Voronoi::with_options(p, options())).unwrap();
    let geometry = dual.delaunay();
    let sites = geometry.points().len() / d;
    for site in 0..sites {
        let expected: Vec<_> = (0..geometry.simplex_count())
            .filter(|&cell| geometry.simplex(cell).unwrap().contains(&site))
            .collect();
        assert_eq!(
            geometry
                .incident_simplices(site)
                .unwrap()
                .collect::<Vec<_>>(),
            expected
        );
        let vertices: Vec<_> = (0..dual.vertex_count())
            .filter(|vertex| {
                expected
                    .iter()
                    .any(|&cell| dual.simplex_vertices()[cell] == *vertex)
            })
            .collect();
        assert_eq!(
            dual.region_vertices(site).unwrap().collect::<Vec<_>>(),
            vertices
        );
        assert_eq!(
            dual.region_is_unbounded(site).unwrap(),
            dual.rays().any(|ray| ray.sites().contains(&site))
        );
    }
    for invalid in [sites, usize::MAX] {
        assert!(geometry.incident_simplices(invalid).is_err());
        assert!(dual.region_vertices(invalid).is_err());
        assert!(dual.region_is_unbounded(invalid).is_err());
    }

    // Every nonempty simplex face, including sites and full-dimensional
    // interiors. Dyadic sites and repeated midpoints keep these queries exact
    // in both scalar widths, even for triangular/tetrahedral supporting faces.
    let mut queries = Vec::new();
    for cell in 0..geometry.simplex_count() {
        let vertices = geometry.simplex(cell).unwrap();
        for mask in 1..1 << (d + 1) {
            let mut selected = vertices
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (mask & (1 << i) != 0).then_some(v));
            let mut query = geometry.cloud.point(selected.next().unwrap()).to_vec();
            for vertex in selected {
                for (x, &y) in query.iter_mut().zip(geometry.cloud.point(vertex)) {
                    *x = (*x + y) / S::try_from_f64_lossy(2.).unwrap();
                }
            }
            queries.push(query);
        }
    }
    queries.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    queries.dedup();
    let requirements = test_requirements(d, options()).unwrap();
    let mut workspace = test_workspace(requirements).unwrap();
    let expected: Vec<_> = queries
        .iter()
        .map(|query| exhaustive_location(geometry, query).0)
        .collect();
    for (query, &expected) in queries
        .iter()
        .zip(&expected)
        .chain(queries.iter().zip(&expected).rev())
    {
        assert_eq!(
            geometry
                .find_simplex_scoped(
                    query,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap(),
            expected
        );
    }
}

#[test]
fn indexed_queries_match_exhaustive_definitions_in_dimensions_two_through_four() {
    for d in 2..=4 {
        let random: Vec<_> = random_coordinates(12 * d, 1118 + d as u64)
            .iter()
            .map(|x| (x * 32.).floor() / 32.)
            .collect();
        // Two adjacent unit cubes; exhaust every resulting simplex face,
        // including shared cospherical cell boundaries in 4D.
        let grid: Vec<_> = (0..3 * (1 << (d - 1)))
            .flat_map(|i| {
                (0..d).map(move |axis| {
                    if axis == 0 {
                        (i % 3) as f64
                    } else {
                        (((i / 3) >> (axis - 1)) & 1) as f64
                    }
                })
            })
            .collect();
        // All cube vertices are exactly cospherical (cocircular in 2D).
        // A duplicate input also exercises original-to-canonical mapping.
        let mut cospherical = cube::<f64>(d);
        cospherical.extend_from_within(..d);
        for sites in [&random, &grid, &cospherical] {
            check_primal_and_dual::<f64>(sites, d);
            check_primal_and_dual::<f32>(sites, d);
        }
    }
}

#[test]
fn local_boundary_resolution_needs_only_containing_cell_predicates() {
    let storage = Storage::new(usize::MAX, 0).unwrap();

    for d in 2..=4 {
        let mut sites = cube::<f64>(d);
        sites.extend(random_coordinates(16 * d, 64 + d as u64));
        let geometry = with_points(&sites, d, |p| Delaunay::with_options(p, options())).unwrap();
        let mut workspace = test_workspace(test_requirements(d, options()).unwrap()).unwrap();
        let mut limited = options();
        limited.limits.max_predicate_terms = (d + 1) * (1..=d + 1).product::<usize>();
        for site in 0..geometry.cloud.len() {
            let star: Vec<_> = geometry.incident_simplices(site).unwrap().collect();
            for &seed in &star {
                let mut work = Work::new(
                    &mut workspace,
                    d,
                    limited,
                    None,
                    scioxide_core::argument!("points"),
                    &storage,
                )
                .unwrap();
                assert_eq!(
                    geometry
                        .walk(geometry.cloud.point(site), seed, 1, &mut work)
                        .unwrap(),
                    Some(SimplexLocation {
                        simplex: star[0],
                        boundary: true
                    })
                );
            }
        }
    }
}

// Independent exact normal-rank oracle for the dyadic halfspace fixtures.
// Two distinct vertices share an edge iff their common active normals have
// rank d-1. Use integer minors, not the production dual facet adjacency.
fn shared_edge(result: &HalfspaceIntersection<f64>, rows: &[f64], a: usize, b: usize) -> bool {
    let d = result.dimension();
    let common: Vec<_> = result
        .active_halfspaces(a)
        .unwrap()
        .iter()
        .copied()
        .filter(|plane| result.active_halfspaces(b).unwrap().contains(plane))
        .collect();
    for mask in 0_usize..1 << common.len() {
        if mask.count_ones() as usize != d - 1 {
            continue;
        }
        for omitted in 0..d {
            let matrix: Vec<_> = common
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, &plane)| {
                    (0..d)
                        .filter(|&axis| axis != omitted)
                        .map(|axis| (rows[plane * (d + 1) + axis] * 32.) as i128)
                        .collect()
                })
                .collect();
            if integer_determinant(matrix) != 0 {
                return true;
            }
        }
    }
    false
}

#[test]
fn halfspace_neighbors_match_exhaustive_shared_faces() {
    for d in 2..=4 {
        let mut cube = Vec::new();
        for axis in 0..d {
            for sign in [-1., 1.] {
                let mut row = vec![0.; d + 1];
                row[axis] = sign;
                row[d] = -1.;
                cube.extend(row);
            }
        }
        let mut random = cube.clone();
        for normal in random_coordinates(12 * d, 1118).chunks_exact(d) {
            random.extend(normal.iter().map(|x| (x * 64.).floor() / 32. - 1.));
            random.push(-1.);
        }
        // Duplicate active planes remain original indices, but must not add
        // neighbors or turn a shared vertex into an edge.
        for rows in [&mut cube, &mut random] {
            rows.extend_from_within(..d + 1);
            let result = with_points(rows, d + 1, |p| {
                HalfspaceIntersection::with_options(p, &vec![0.; d], options())
            })
            .unwrap();
            let count = result.vertices().len() / d;
            for vertex in 0..count {
                let expected: Vec<_> = (0..count)
                    .filter(|&other| other != vertex && shared_edge(&result, rows, vertex, other))
                    .collect();
                assert_eq!(
                    result.neighbors(vertex).unwrap().collect::<Vec<_>>(),
                    expected
                );
            }
            assert!(result.neighbors(count).is_err());
            assert!(result.neighbors(usize::MAX).is_err());
        }
    }
    let unbounded = with_points(&[1., 0., -1., 0., 1., -1.], 3, |p| {
        HalfspaceIntersection::with_options(p, &[0.; 2], options())
    })
    .unwrap();
    assert_eq!(unbounded.kind(), IntersectionKind::Unbounded);
    assert!(unbounded.neighbors(0).is_err());
}

#[test]
fn spherical_neighbors_match_exhaustive_region_arcs() {
    let random: Vec<_> = random_coordinates(72, 1118)
        .iter()
        .map(|x| 2. * x - 1.)
        .collect();
    let cube: Vec<_> = cube::<f64>(3).iter().map(|x| 2. * x - 1.).collect();
    for points in [random, cube] {
        let mut sites: Vec<_> = points
            .chunks_exact(3)
            .flat_map(|p| {
                let norm = p.iter().map(|x| x * x).sum::<f64>().sqrt();
                p.iter().map(move |x| x / norm)
            })
            .collect();
        sites.extend_from_within(..3);
        let sphere = with_points(&sites, 3, |p| {
            SphericalVoronoi::with_options(
                p,
                [0.; 3],
                1.,
                scioxide_core::Tolerance::new(1e-12, 1e-12).unwrap(),
                options(),
            )
        })
        .unwrap();
        let count = sphere.areas().len();
        for site in 0..count {
            let region = sphere.region(site).unwrap();
            let expected: Vec<_> = (0..count)
                .filter(|&other| {
                    if other == site {
                        return false;
                    }
                    let neighbor = sphere.region(other).unwrap();
                    region
                        .iter()
                        .zip(region.iter().cycle().skip(1))
                        .any(|(&a, &b)| {
                            neighbor
                                .iter()
                                .zip(neighbor.iter().cycle().skip(1))
                                .any(|(&c, &d)| (a == c && b == d) || (a == d && b == c))
                        })
                })
                .collect();
            assert_eq!(
                sphere.neighbors(site).unwrap().collect::<Vec<_>>(),
                expected
            );
        }
        assert!(sphere.neighbors(count).is_err());
        assert!(sphere.neighbors(usize::MAX).is_err());
    }
}

// Independent fraction-free elimination oracle. The production predicates use
// significand-product expansion, so this shares neither arithmetic nor control flow.
#[allow(clippy::needless_range_loop)]
fn integer_determinant(mut a: Vec<Vec<i128>>) -> i128 {
    let n = a.len();
    let mut divisor = 1;
    let mut parity = 1;
    for k in 0..n - 1 {
        let Some(pivot) = (k..n).find(|&i| a[i][k] != 0) else {
            return 0;
        };
        if pivot != k {
            a.swap(pivot, k);
            parity = -parity;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) / divisor;
            }
        }
        divisor = a[k][k];
        for row in a.iter_mut().skip(k + 1) {
            row[k] = 0;
        }
    }
    parity * a[n - 1][n - 1]
}
