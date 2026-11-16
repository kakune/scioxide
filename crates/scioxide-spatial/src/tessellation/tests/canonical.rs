use super::*;

fn lifted_grid(k: usize) -> Vec<f64> {
    (0..k)
        .flat_map(|i| (0..k).flat_map(move |j| [i as f64, j as f64, (i * i + j * j) as f64]))
        .collect()
}

fn compare_reference<S: RealScalar>(values: &[f64], d: usize) {
    let values: Vec<S> = values
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    with_points(&values, d, |points| {
        let storage = Storage::new(usize::MAX, 0)?;
        let o = options();
        let cloud = Cloud::new(points, o, &storage)?;
        let mut workspace = test_workspace(test_requirements(d, o)?)?;
        let mut work = Work::new(
            &mut workspace,
            d,
            o,
            None,
            scioxide_core::argument!("points"),
            &storage,
        )?;
        let raw = engine::hull(&cloud, false, &mut work)?;
        let (expected, expected_groups) = super::canonical_reference::hull(&cloud, raw, &mut work)?;
        let raw = engine::hull(&cloud, false, &mut work)?;
        let (actual, actual_groups) = super::super::canonical::hull(&cloud, raw, &mut work)?;
        assert_eq!(actual_groups, expected_groups);
        assert_eq!(actual.neighbors, expected.neighbors);
        assert_eq!(actual.facets.len(), expected.facets.len());
        for (a, b) in actual.facets.iter().zip(expected.facets.iter()) {
            assert_eq!(a.vertices, b.vertices);
            assert_eq!(a.inside, b.inside);
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn canonical_pulling_matches_exhaustive_topology() {
    for d in 2..=5 {
        let cube = cube::<f64>(d);
        let random = random_coordinates(d * (d + 8), 1922 + d as u64);
        for values in [&cube, &random] {
            compare_reference::<f32>(values, d);
            compare_reference::<f64>(values, d);
        }
    }
    for k in [3, 5, 8] {
        compare_reference::<f32>(&lifted_grid(k), 3);
        compare_reference::<f64>(&lifted_grid(k), 3);
    }
    // Non-extreme sites on faces and edges, plus an interior site.
    let mut boundary = cube::<f64>(3);
    boundary.extend([0.5, 0., 0., 0.5, 0.5, 0., 0.5, 0.5, 0.5]);
    compare_reference::<f64>(&boundary, 3);
    // Many non-simplicial side facets share this high-degree apex.
    let mut pyramid: Vec<_> = lifted_grid(5)
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 0.])
        .collect();
    pyramid.extend([10., 10., 10., 1.]);
    compare_reference::<f64>(&pyramid, 4);
}

#[test]
fn canonical_work_grows_near_linearly_on_exact_lifted_grids() {
    let mut previous = 0;
    for k in [8, 16, 32, 64] {
        let values = lifted_grid(k);
        let work = with_points(&values, 3, |points| {
            ConvexHull::new(points).map(|h| h.provenance().canonical_work)
        })
        .unwrap();
        assert!(work > 0);
        if previous != 0 {
            assert!(work < 6 * previous, "k={k}: {previous} -> {work}");
        }
        previous = work;
    }
}

#[test]
fn high_degree_apex_does_not_restore_global_support_scans() {
    let mut previous = 0;
    for k in [8, 16, 32] {
        let mut values: Vec<_> = lifted_grid(k)
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 0.])
            .collect();
        values.extend([0., 0., 0., 1.]);
        let work = with_points(&values, 4, |points| {
            ConvexHull::new(points).map(|h| h.provenance().canonical_work)
        })
        .unwrap();
        if previous != 0 {
            assert!(work < 6 * previous, "k={k}: {previous} -> {work}");
        }
        previous = work;
    }
}
