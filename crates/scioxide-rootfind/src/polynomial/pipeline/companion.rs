//! Candidates from companion and colleague matrix eigenvalues.
use super::*;
pub(crate) fn companion<S>(
    p: Prepared<'_, S>,
    w: &mut PolynomialRootWorkspace<S>,
    options: &PolynomialRootOptions<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), PolynomialFailure>
where
    S: RealScalar,
    Complex<S>: NumericScalar<Real = S>,
{
    let n = p.degree;
    let options = GeneralEigenOptions::default()
        .with_deflation_tolerance(options.backward_error_tolerance.tolerance())
        .with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(options.policy)
                .with_iteration_budget(Budget::try_from(checked_mul(
                    options.max_iterations.get(),
                    n,
                    scioxide_core::quantity!("eigen_iterations"),
                )?)?),
        );
    admit(session, scioxide_core::EvaluationRole::Factorization)?;
    if p.real {
        for (real, coefficient) in w.real_coefficients[..=n]
            .iter_mut()
            .zip(&w.coefficients[..=n])
        {
            *real = coefficient.re;
        }
        w.real_matrix.try_resize_flat(n, n)?;
        scioxide_poly::polynomial_root_matrix_into(
            &w.real_coefficients[..=n],
            p.basis,
            &mut w.real_matrix,
            session.execution()?,
        )?;
        GeneralEigenvaluesBuffer::compute_into(
            &w.real_matrix,
            &options,
            &mut w.eigen_result,
            &mut w.eigen_workspace,
            session.execution()?,
        )?;
    } else {
        w.matrix.try_resize_flat(n, n)?;
        scioxide_poly::polynomial_root_matrix_into(
            &w.coefficients[..=n],
            p.basis,
            &mut w.matrix,
            session.execution()?,
        )?;
        GeneralEigenvaluesBuffer::compute_into(
            &w.matrix,
            &options,
            &mut w.eigen_result,
            &mut w.eigen_workspace,
            session.execution()?,
        )?;
    }
    w.current[..n].copy_from_slice(w.eigen_result.eigenvalues());
    Ok(())
}
