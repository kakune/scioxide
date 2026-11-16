//! Malformed results retain callback classification and transactional publication.
use super::*;
use scioxide_linalg::{CopyPolicy, MatrixViewMut};

#[derive(Clone, Copy)]
enum Fault {
    NonFiniteAsum,
    NegativeNorm,
    BadIndex,
    BadRotationFlag,
    NonFiniteRotation,
    HermitianDiagonal(BlasOperation),
}
struct FaultProvider {
    fault: Fault,
    calls: AtomicUsize,
}
impl FaultProvider {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            calls: AtomicUsize::new(0),
        }
    }
    fn context(&self) -> LinalgExecutionContext<'_, '_> {
        LinalgExecutionContext::sequential()
            .with_blas(BlasDispatch::new(self, BlasFallbackPolicy::Error))
    }
    fn invoke<S: NumericScalar>(&self, call: BlasCall<'_, S>) -> Result<(), CapabilityError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let expected = match self.fault {
            Fault::NonFiniteAsum => BlasOperation::Asum,
            Fault::NegativeNorm => BlasOperation::Norm2,
            Fault::BadIndex => BlasOperation::Iamax,
            Fault::BadRotationFlag | Fault::NonFiniteRotation => BlasOperation::Rotmg,
            Fault::HermitianDiagonal(op) => op,
        };
        assert_eq!(call.operation(), expected);
        match (self.fault, call) {
            (Fault::NonFiniteAsum, BlasCall::Asum { result, .. }) => *result = S::Real::NAN,
            (Fault::NegativeNorm, BlasCall::Norm2 { result, .. }) => *result = -S::Real::ONE,
            (Fault::BadIndex, BlasCall::Iamax { result, .. }) => *result = usize::MAX,
            (Fault::BadRotationFlag, BlasCall::Rotmg { parameters, .. }) => {
                parameters[0] = S::Real::TWO + S::Real::ONE
            }
            (Fault::NonFiniteRotation, BlasCall::Rotmg { parameters, .. }) => {
                parameters[0] = -S::Real::ONE;
                parameters[1] = S::Real::NAN;
            }
            (
                Fault::HermitianDiagonal(_),
                BlasCall::Herk { mut c, .. } | BlasCall::Her2k { mut c, .. },
            ) => matrix_set(&mut c, 0, 0, S::from_components(S::Real::ONE, S::Real::ONE)),
            (Fault::HermitianDiagonal(_), BlasCall::Her { mut a, .. }) => {
                matrix_set(&mut a, 0, 0, S::from_components(S::Real::ONE, S::Real::ONE))
            }
            (Fault::HermitianDiagonal(_), BlasCall::Hpr { mut a, .. }) => {
                a.values_mut()[0] = S::from_components(S::Real::ONE, S::Real::ONE)
            }
            _ => return Err(CapabilityError::new("unexpected fault injection")),
        }
        Ok(())
    }
}
impl BlasProvider for FaultProvider {
    fn name(&self) -> &'static str {
        "malformed-result"
    }
    fn capabilities(&self) -> BlasCapabilities {
        CblasProvider::new().capabilities
    }
    fn execute_f32(&self, call: BlasCall<'_, f32>, _: NonZeroUsize) -> Result<(), CapabilityError> {
        self.invoke(call)
    }
    fn execute_f64(&self, call: BlasCall<'_, f64>, _: NonZeroUsize) -> Result<(), CapabilityError> {
        self.invoke(call)
    }
    fn execute_complex32(
        &self,
        call: BlasCall<'_, Complex32>,
        _: NonZeroUsize,
    ) -> Result<(), CapabilityError> {
        self.invoke(call)
    }
    fn execute_complex64(
        &self,
        call: BlasCall<'_, Complex64>,
        _: NonZeroUsize,
    ) -> Result<(), CapabilityError> {
        self.invoke(call)
    }
}

#[test]
fn malformed_reductions_and_rotmg_have_the_required_error_origins() {
    for fault in [
        Fault::NonFiniteAsum,
        Fault::NegativeNorm,
        Fault::BadIndex,
        Fault::BadRotationFlag,
        Fault::NonFiniteRotation,
    ] {
        let provider = FaultProvider::new(fault);
        let x = VectorView::from_slice(&[1.0_f64, 2.0]);
        let error = match fault {
            Fault::NonFiniteAsum => x
                .asum_scoped(
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    provider.context(),
                )
                .unwrap_err(),
            Fault::NegativeNorm => x
                .norm2_scoped(
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    provider.context(),
                )
                .unwrap_err(),
            Fault::BadIndex => x
                .iamax_scoped(
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    provider.context(),
                )
                .unwrap_err(),
            _ => scioxide_linalg::modified_givens_rotation_scoped(
                1.0_f64,
                1.0,
                3.0,
                4.0,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                provider.context(),
            )
            .unwrap_err(),
        };
        if matches!(fault, Fault::NonFiniteAsum | Fault::NonFiniteRotation) {
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Callback,
                    ..
                }
            ));
        } else {
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::CapabilityFailure
            );
        }
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn non_real_provider_diagonals_fail_before_dense_publication() {
    let one = Complex64::new(1.0, 0.0);
    let a = Matrix::from_vec(2, 2, vec![one; 4]).unwrap();
    let values = [one, one];
    let x = VectorView::from_slice(&values);
    for op in [
        BlasOperation::Her,
        BlasOperation::Herk,
        BlasOperation::Her2k,
    ] {
        let provider = FaultProvider::new(Fault::HermitianDiagonal(op));
        let mut output = [one; 4];
        let mut workspace = LinalgWorkspace::new();
        let options = LinalgExecutionOptions::new();
        let out = MatrixViewMut::from_slice(2, 2, &mut output).unwrap();
        let error = match op {
            BlasOperation::Her => MatrixViewMut::hermitian_rank_one_update_in_place(
                Triangle::Lower,
                one,
                x,
                &options,
                out,
                &mut workspace,
                provider.context(),
            ),
            BlasOperation::Herk => MatrixViewMut::hermitian_rank_k_in_place(
                a.view(),
                Triangle::Lower,
                StructuredOperation::Normal,
                one,
                one,
                &options,
                out,
                &mut workspace,
                provider.context(),
            ),
            _ => MatrixViewMut::hermitian_rank_two_k_in_place(
                a.view(),
                a.view(),
                Triangle::Lower,
                StructuredOperation::Normal,
                one,
                one,
                &options,
                out,
                &mut workspace,
                provider.context(),
            ),
        }
        .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::CapabilityFailure
        );
        assert_eq!(output, [one; 4]);
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn preserve_copy_is_structural_even_with_an_attached_provider() {
    let provider = FaultProvider::new(Fault::NonFiniteAsum);
    let input = [f64::NAN, f64::INFINITY];
    let mut output = [0.0; 2];
    VectorView::from_slice(&input)
        .copy_into(
            CopyPolicy::Preserve,
            &LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut output),
            &mut LinalgWorkspace::new(),
            provider.context(),
        )
        .unwrap();
    assert!(output[0].is_nan());
    assert_eq!(output[1], f64::INFINITY);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn capabilities_are_specific_to_scalar_and_routine() {
    let caps = BlasCapabilities::empty(BlasIntegerWidth::Lp64)
        .with_operation(ScalarKind::F64, BlasOperation::Gemm, BlasUlp::new(1))
        .unwrap()
        .with_operation(ScalarKind::Complex64, BlasOperation::Gemm, BlasUlp::new(7))
        .unwrap();
    assert_eq!(
        caps.operation_ulp(ScalarKind::F64, BlasOperation::Gemm)
            .unwrap()
            .maximum(),
        1
    );
    assert_eq!(
        caps.operation_ulp(ScalarKind::Complex64, BlasOperation::Gemm)
            .unwrap()
            .maximum(),
        7
    );
    assert!(
        caps.operation_ulp(ScalarKind::F32, BlasOperation::Gemm)
            .is_none()
    );
    assert_eq!(
        caps.with_operation(ScalarKind::F64, BlasOperation::Herk, BlasUlp::new(0))
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn rank_update_checks_the_second_vector_increment_before_invoking() {
    let provider = CblasProvider::new();
    for fallback in [BlasFallbackPolicy::Reference, BlasFallbackPolicy::Error] {
        let mut output = [0.0_f64];
        let result = MatrixViewMut::rank_one_update_in_place(
            1.0,
            VectorView::from_slice(&[2.0]),
            VectorView::from_slice_with_stride(&[3.0], 1, 0).unwrap(),
            &LinalgExecutionOptions::new(),
            MatrixViewMut::from_slice(1, 1, &mut output).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential().with_blas(BlasDispatch::new(&provider, fallback)),
        );
        if fallback == BlasFallbackPolicy::Reference {
            result.unwrap();
            assert_eq!(output, [6.0]);
        } else {
            assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::CapabilityUnavailable
            );
            assert_eq!(output, [0.0]);
        }
        assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn packed_non_real_provider_diagonals_fail_before_publication() {
    let one = Complex64::new(1.0, 0.0);
    let values = [one, one];
    for triangle in [Triangle::Lower, Triangle::Upper] {
        let provider = FaultProvider::new(Fault::HermitianDiagonal(BlasOperation::Hpr));
        let mut output = [one; 3];
        let view = scioxide_linalg::PackedMatrixViewMut::new(
            2,
            triangle,
            scioxide_linalg::PackedKind::Hermitian,
            &mut output,
        )
        .unwrap();
        let error = scioxide_linalg::PackedMatrixViewMut::hermitian_rank_one_update_in_place(
            one,
            VectorView::from_slice(&values),
            &LinalgExecutionOptions::new(),
            view,
            &mut LinalgWorkspace::new(),
            provider.context(),
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::CapabilityFailure
        );
        assert_eq!(output, [one; 3]);
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn provider_swap_still_requires_staging_for_both_outputs() {
    let provider = CblasProvider::new();
    let mut x = [1.0_f64, 2.0];
    let mut y = [3.0_f64, 4.0];
    let mut short = [0.0; 3];
    let error = VectorViewMut::swap_in_place(
        &LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut x),
        VectorViewMut::from_slice(&mut y),
        &mut LinalgWorkspace::from_scalar_slice(&mut short),
        provider.context(),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(x, [1.0, 2.0]);
    assert_eq!(y, [3.0, 4.0]);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
}
