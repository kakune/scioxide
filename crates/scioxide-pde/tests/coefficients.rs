//! Compare isotropic adapters with independently written coefficient providers.
#![allow(clippy::unwrap_used)]
use scioxide_core::{SciOxideError, SciOxideErrorView};
use scioxide_pde::*;
use scioxide_spatial::{Point, TriangleMesh};

struct Coefficients {
    reaction: f64,
    constant: bool,
}
impl<const D: usize> StructuredFdmCoefficients<f64, f64, D> for Coefficients {
    fn diffusion(&self, _: usize, _: usize, _: [f64; D]) -> Result<f64, SciOxideError> {
        Ok(2.)
    }
    fn reaction(&self, row: usize, column: usize, _: [f64; D]) -> Result<f64, SciOxideError> {
        Ok(if row == column { self.reaction } else { 0. })
    }
}
impl<const D: usize> FvmCoefficients<f64, f64, D> for Coefficients {
    fn diffusion(&self, _: [f64; D]) -> Result<[[f64; D]; D], SciOxideError> {
        let mut tensor = [[0.; D]; D];
        for (i, row) in tensor.iter_mut().enumerate() {
            row[i] = 2.;
        }
        Ok(tensor)
    }
    fn reaction(&self, _: [f64; D]) -> Result<f64, SciOxideError> {
        Ok(self.reaction)
    }
}
impl<const D: usize> SimplexFemCoefficients<f64, f64, D> for Coefficients {
    fn is_spatially_constant(&self) -> bool {
        self.constant
    }
    fn evaluate(
        &self,
        _: [f64; D],
        diffusion: &mut [f64],
        reaction: &mut [f64],
    ) -> Result<(), SciOxideError> {
        diffusion.fill(0.);
        reaction.fill(0.);
        let n = diffusion.len().isqrt();
        for i in 0..n {
            diffusion[i * n + i] = 2.;
        }
        let n = reaction.len().isqrt();
        for i in 0..n {
            reaction[i * n + i] = self.reaction;
        }
        Ok(())
    }
}
fn matrix<P: SpatialDiscretization<f64, f64, D>, const D: usize>(plan: &P) -> Vec<f64> {
    plan.assemble_csr().unwrap().values().to_vec()
}

#[test]
fn finite_differences_and_structured_volumes_match_providers() {
    fn check<const D: usize>() {
        let box_plan = StructuredFdmPlan::dirichlet_laplacian([0.; D], [1.; D], [4; D]).unwrap();
        let mesh = box_plan.mesh();
        let boundaries =
            StructuredBoundaries::uniform(mesh, 1, BoundaryKind::Natural, BoundaryData::zero())
                .unwrap();
        let reference = Coefficients {
            reaction: 0.3,
            constant: true,
        };
        let constant = ConstantDiffusion::new(2.).with_reaction(0.3);
        assert_eq!(
            matrix(&StructuredFdmPlan::new(mesh.clone(), &reference, &boundaries).unwrap()),
            matrix(&StructuredFdmPlan::new(mesh.clone(), &constant, &boundaries).unwrap())
        );
        assert_eq!(
            matrix(&StructuredFvmPlan::new(mesh.clone(), &reference, &boundaries).unwrap()),
            matrix(&StructuredFvmPlan::new(mesh.clone(), &constant, &boundaries).unwrap())
        );
        let reference = Coefficients {
            reaction: 0.,
            constant: false,
        };
        let variable = IsotropicDiffusion::new(|_: [f64; D]| Ok(2.));
        assert_eq!(
            matrix(&StructuredFdmPlan::new(mesh.clone(), &reference, &boundaries).unwrap()),
            matrix(&StructuredFdmPlan::new(mesh.clone(), &variable, &boundaries).unwrap())
        );
        assert_eq!(
            matrix(&StructuredFvmPlan::new(mesh.clone(), &reference, &boundaries).unwrap()),
            matrix(&StructuredFvmPlan::new(mesh.clone(), &variable, &boundaries).unwrap())
        );
        assert!(matches!(
            StructuredFdmPlan::new(mesh.clone(), &ConstantDiffusion::new(-1.), &boundaries)
                .unwrap_err()
                .view(),
            SciOxideErrorView::DomainError { .. }
        ));
        let failing = IsotropicDiffusion::new(|_: [f64; D]| {
            Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "coefficient failure",
            ))
        });
        assert!(matches!(
            StructuredFdmPlan::new(mesh.clone(), &failing, &boundaries)
                .unwrap_err()
                .view(),
            SciOxideErrorView::NonFinite { .. }
        ));
    }
    check::<1>();
    check::<2>();
    check::<3>();
}

#[test]
fn simplex_volumes_and_multicomponent_elements_match_providers() {
    let mesh = TriangleMesh::new(
        vec![
            Point::new([0., 0.]).unwrap(),
            Point::new([1., 0.]).unwrap(),
            Point::new([0., 1.]).unwrap(),
            Point::new([1., 1.]).unwrap(),
        ],
        vec![[0, 1, 2], [1, 3, 2]],
    )
    .unwrap();
    let boundaries =
        TaggedBoundaries::uniform(&mesh, 1, BoundaryKind::Natural, BoundaryData::zero()).unwrap();
    let constant = ConstantDiffusion::new(2.).with_reaction(0.3);
    let reference = Coefficients {
        reaction: 0.3,
        constant: true,
    };
    assert_eq!(
        matrix(&SimplexFvmPlan::new(mesh.clone(), &reference, &boundaries).unwrap()),
        matrix(&SimplexFvmPlan::new(mesh.clone(), &constant, &boundaries).unwrap())
    );
    let variable = IsotropicDiffusion::new(|_: [f64; 2]| Ok(2.));
    let varying = Coefficients {
        reaction: 0.,
        constant: false,
    };
    assert_eq!(
        matrix(&SimplexFvmPlan::new(mesh.clone(), &varying, &boundaries).unwrap()),
        matrix(&SimplexFvmPlan::new(mesh.clone(), &variable, &boundaries).unwrap())
    );
    for components in [1, 2] {
        let boundaries = TaggedBoundaries::uniform(
            &mesh,
            components,
            BoundaryKind::Natural,
            BoundaryData::zero(),
        )
        .unwrap();
        for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
            let build = |c: &dyn SimplexFemCoefficients<f64, f64, 2>| {
                SimplexFemPlan::new(
                    mesh.clone(),
                    SimplexFemGeometry::Affine,
                    order,
                    c,
                    &boundaries,
                )
                .unwrap()
            };
            assert_eq!(matrix(&build(&reference)), matrix(&build(&constant)));
            assert_eq!(matrix(&build(&varying)), matrix(&build(&variable)));
        }
    }
}

#[test]
fn uniform_boundaries_match_explicit_faces_and_tags() {
    use scioxide_spatial::{
        AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh,
        SimplexBoundaryTags,
    };
    let grid = StructuredFdmPlan::dirichlet_laplacian([0.], [1.], [9]).unwrap();
    for kind in [
        BoundaryKind::Dirichlet,
        BoundaryKind::Natural,
        BoundaryKind::Robin { alpha: 0.5 },
    ] {
        let mesh = grid.mesh();
        let data = BoundaryData::constant(0.2);
        let uniform = StructuredBoundaries::uniform(mesh, 1, kind, data.clone()).unwrap();
        let make = |region| match kind {
            BoundaryKind::Dirichlet => BoundaryCondition::dirichlet(region, data.clone()),
            BoundaryKind::Natural => BoundaryCondition::natural(region, data.clone()),
            BoundaryKind::Robin { alpha } => BoundaryCondition::robin(region, alpha, data.clone()),
        };
        let explicit = StructuredBoundaries::new(
            mesh,
            1,
            vec![
                make(BoundarySet::axis_side(mesh, 0, AxisSide::Start).unwrap()),
                make(BoundarySet::axis_side(mesh, 0, AxisSide::End).unwrap()),
            ],
        )
        .unwrap();
        let a = StructuredFdmPlan::new(
            mesh.clone(),
            &ConstantDiffusion::new(1.).with_reaction(0.5),
            &uniform,
        )
        .unwrap();
        let b = StructuredFdmPlan::new(
            mesh.clone(),
            &ConstantDiffusion::new(1.).with_reaction(0.5),
            &explicit,
        )
        .unwrap();
        assert_eq!(matrix(&a), matrix(&b));
        assert_eq!(
            a.load(&FieldData::constant(1.), 0.).unwrap(),
            b.load(&FieldData::constant(1.), 0.).unwrap()
        );
    }
    let periodic = SegmentMesh::new(
        Grid1d::new(
            [CoordinateAxis::uniform(0., 1., 9).unwrap()],
            [AxisTopology::Periodic],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        StructuredBoundaries::<f64, f64, 1>::uniform(
            &periodic,
            1,
            BoundaryKind::Dirichlet,
            BoundaryData::zero()
        )
        .unwrap()
        .conditions()
        .is_empty()
    );
    assert!(
        StructuredBoundaries::<f64, f64, 1>::uniform(
            grid.mesh(),
            0,
            BoundaryKind::Dirichlet,
            BoundaryData::zero()
        )
        .is_err()
    );
    assert!(
        StructuredBoundaries::<f64, f64, 1>::uniform(
            grid.mesh(),
            1,
            BoundaryKind::Robin { alpha: f64::NAN },
            BoundaryData::zero()
        )
        .is_err()
    );
    let mesh = TriangleMesh::new(
        vec![
            Point::new([0., 0.]).unwrap(),
            Point::new([1., 0.]).unwrap(),
            Point::new([0., 1.]).unwrap(),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|f| (f, 0)).collect()).unwrap();
    for kind in [
        BoundaryKind::Dirichlet,
        BoundaryKind::Natural,
        BoundaryKind::Robin { alpha: 0.5 },
    ] {
        let data = BoundaryData::constant(0.2);
        let uniform = TaggedBoundaries::uniform(&mesh, 1, kind, data.clone()).unwrap();
        let condition = match kind {
            BoundaryKind::Dirichlet => BoundaryCondition::dirichlet(0, data),
            BoundaryKind::Natural => BoundaryCondition::natural(0, data),
            BoundaryKind::Robin { alpha } => BoundaryCondition::robin(0, alpha, data),
        };
        let explicit = TaggedBoundaries::new(&mesh, &tags, 1, vec![condition]).unwrap();
        let build = |b| {
            SimplexFemPlan::new(
                mesh.clone(),
                SimplexFemGeometry::Affine,
                SimplexFemOrder::P2,
                &ConstantDiffusion::new(1.).with_reaction(0.5),
                b,
            )
            .unwrap()
        };
        let a = build(&uniform);
        let b = build(&explicit);
        assert_eq!(matrix(&a), matrix(&b));
        assert_eq!(
            a.load(&FieldData::constant(1.), 0.).unwrap(),
            b.load(&FieldData::constant(1.), 0.).unwrap()
        );
        let source = FieldData::constant(1.);
        let problem = EllipticProblem::new(&a, &source);
        assert_eq!(
            problem.solve().unwrap(),
            problem
                .solve_with_options(&EllipticOptions::default())
                .unwrap()
                .values()
        );
        let a = SimplexFvmPlan::new(
            mesh.clone(),
            &ConstantDiffusion::new(1.).with_reaction(0.5),
            &uniform,
        )
        .unwrap();
        let problem = EllipticProblem::new(&a, &source);
        assert_eq!(
            problem.solve().unwrap(),
            problem
                .solve_with_options(&EllipticOptions::default())
                .unwrap()
                .values()
        );
    }
    assert!(
        TaggedBoundaries::<f64, f64, 2>::uniform(
            &mesh,
            0,
            BoundaryKind::Dirichlet,
            BoundaryData::zero()
        )
        .is_err()
    );
    assert!(
        TaggedBoundaries::<f64, f64, 2>::uniform(
            &mesh,
            1,
            BoundaryKind::Robin { alpha: f64::NAN },
            BoundaryData::zero()
        )
        .is_err()
    );
}
