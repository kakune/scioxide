use scioxide_core::{LinearOperator, SciOxideError};
use scioxide_pde::{
    BoundaryData, BoundaryKind, FieldData, FieldPoint, SemidiscreteSystem, SpatialDiscretization,
    StructuredBoundaries, StructuredFdmCoefficients, StructuredFdmPlan,
};

struct AxisDiffusion(f64);
impl StructuredFdmCoefficients<f64, f64, 2> for AxisDiffusion {
    fn diffusion(&self, axis: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
        Ok(if axis == 0 { self.0 } else { 0.0 })
    }
}

#[test]
fn fdm_conductances_preserve_representable_extreme_scales() -> Result<(), SciOxideError> {
    for (end, diffusion) in [(8.0, f64::MAX * 0.5), (1.0, f64::from_bits(1))] {
        let base = StructuredFdmPlan::dirichlet_laplacian([0.0; 2], [end; 2], [3; 2])?;
        let boundaries = StructuredBoundaries::uniform(
            base.mesh(),
            1,
            BoundaryKind::Natural,
            BoundaryData::zero(),
        )?;
        let plan =
            StructuredFdmPlan::new(base.mesh().clone(), &AxisDiffusion(diffusion), &boundaries)?;
        let matrix = plan.assemble_csr()?;
        // At the central vertex, transverse area and longitudinal spacing are equal.
        assert_eq!(matrix.get(4, 3)?, -diffusion);
        assert_eq!(matrix.get(4, 4)?, 2.0 * diffusion);
    }
    Ok(())
}

#[test]
fn boundary_transport_does_not_declare_a_constant_nullspace() -> Result<(), SciOxideError> {
    struct BoundaryTransport;
    impl StructuredFdmCoefficients<f64, f64, 1> for BoundaryTransport {
        fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
        fn advection(&self, _: usize, _: usize, x: [f64; 1]) -> Result<f64, SciOxideError> {
            Ok(if x[0] == 0.0 {
                -1.0
            } else if x[0] == 1.0 {
                1.0
            } else {
                0.0
            })
        }
    }
    let base = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let boundaries =
        StructuredBoundaries::uniform(base.mesh(), 1, BoundaryKind::Natural, BoundaryData::zero())?;
    let plan = StructuredFdmPlan::new(base.mesh().clone(), &BoundaryTransport, &boundaries)?;
    assert!(!plan.has_constant_modes());
    let solution = scioxide_pde::EllipticProblem::new(&plan, &FieldData::constant(1.0)).solve()?;
    assert!(solution.iter().all(|v| v.is_finite()));
    Ok(())
}

#[test]
fn supplied_nullspace_honors_sparse_factor_limits() -> Result<(), SciOxideError> {
    let base = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let boundaries =
        StructuredBoundaries::uniform(base.mesh(), 1, BoundaryKind::Natural, BoundaryData::zero())?;
    let plan = StructuredFdmPlan::new(
        base.mesh().clone(),
        &scioxide_pde::ConstantDiffusion::new(1.0),
        &boundaries,
    )?;
    let basis = [1.0 / 5.0_f64.sqrt(); 5];
    let basis = scioxide_core::BlockVectorView::new(&basis, 5, 1)?;
    let source = FieldData::zero();
    let problem = scioxide_pde::EllipticProblem::new(&plan, &source).with_nullspace(basis, basis);
    let options =
        scioxide_pde::EllipticOptions::new().with_gauge(scioxide_pde::EllipticGauge::Orthogonal);
    assert!(problem.solve_with_options(&options)?.report().is_success());
    assert!(
        problem
            .solve_with_options(
                &options.with_direct_limits(scioxide_sparse::SparseLuLimits::new(0, 0))
            )
            .is_err()
    );
    Ok(())
}

struct Provider {
    plan: StructuredFdmPlan<f64, f64, 1>,
    mask: Vec<bool>,
    mass: Vec<f64>,
    wrong_matrix: bool,
    components: usize,
}
impl LinearOperator<f64> for Provider {
    fn rows(&self) -> usize {
        self.plan.rows()
    }
    fn cols(&self) -> usize {
        self.plan.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.plan.apply_workspace_len()
    }
    fn apply_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.plan.apply_into(x, out, scratch)
    }
}
impl SpatialDiscretization<f64, f64, 1> for Provider {
    type Field = Vec<f64>;
    fn mesh_token(&self) -> &scioxide_spatial::MeshToken {
        self.plan.mesh_token()
    }
    fn components(&self) -> usize {
        self.components
    }
    fn dof_point(&self, i: usize) -> Result<FieldPoint<f64, 1>, SciOxideError> {
        self.plan.dof_point(i)
    }
    fn constrained(&self) -> &[bool] {
        &self.mask
    }
    fn lumped_mass(&self) -> &[f64] {
        &self.mass
    }
    fn has_constant_modes(&self) -> bool {
        false
    }
    fn assemble_csr(&self) -> Result<scioxide_sparse::CsrMatrix<f64>, SciOxideError> {
        if self.wrong_matrix {
            scioxide_sparse::CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![1.0])
        } else {
            self.plan.assemble_csr()
        }
    }
    fn load_into(
        &self,
        source: &FieldData<f64, f64, 1>,
        time: f64,
        out: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.plan.load_into(source, time, out, scratch)
    }
    fn field(&self, values: &[f64]) -> Result<Self::Field, SciOxideError> {
        Ok(values.to_vec())
    }
}

#[test]
fn semidiscrete_rejects_malformed_provider_metadata_before_indexing() -> Result<(), SciOxideError> {
    let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [5])?;
    let source = FieldData::zero();
    for (mask, mass, wrong_matrix) in [
        (vec![false; 6], vec![1.0; 5], false),
        (vec![false; 4], vec![1.0; 5], false),
        (vec![false; 5], vec![1.0; 4], false),
        (vec![false; 5], vec![0.0; 5], false),
        (vec![false; 5], vec![f64::NAN; 5], false),
        (vec![false; 5], vec![1.0; 5], true),
    ] {
        let provider = Provider {
            plan: plan.clone(),
            mask,
            mass,
            wrong_matrix,
            components: 1,
        };
        assert!(SemidiscreteSystem::new(&provider, &source).is_err());
        assert!(
            scioxide_pde::EllipticProblem::new(&provider, &source)
                .solve()
                .is_err()
        );
    }
    let provider = Provider {
        plan,
        mask: vec![false; 5],
        mass: vec![1.0; 5],
        wrong_matrix: false,
        components: 0,
    };
    assert!(SemidiscreteSystem::new(&provider, &source).is_err());
    assert!(
        scioxide_pde::EllipticProblem::new(&provider, &source)
            .solve()
            .is_err()
    );
    Ok(())
}
