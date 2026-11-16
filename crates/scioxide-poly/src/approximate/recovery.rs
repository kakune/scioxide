//! Recovery of a GCD of fixed degree by Gauss–Newton refinement.
//!
//! A right singular vector of the subresultant matrix gives the starting
//! cofactors, followed by a joint least-squares fit of the GCD and both
//! cofactors (Zeng, "The numerical greatest common divisor of univariate
//! polynomials", 2011, §7 and §9–10). The degree comes from the rank decision
//! of the caller and is not changed here.
use super::*;
use crate::PolynomialBasis;
use crate::util::try_collect;
use alloc::vec::Vec;
use scioxide_core::backend::checked_add;
use scioxide_core::{ExecutionContext, NumericalIssue, ScaledValue};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, RankPolicy, ThinSvdDecomposition, Vector,
};

pub(super) fn recover_gcd<C: NumericScalar>(
    left: &MonomialPolynomial<C>,
    right: &MonomialPolynomial<C>,
    degree: usize,
) -> Result<MonomialPolynomial<C>, SciOxideError> {
    if degree == 0 {
        return ApproxPolynomial::monomial_with_options(
            try_filled(1, C::ONE, scioxide_core::quantity!("unit_approximate_gcd"))?,
            left.policy(),
        );
    }
    let p = equilibrate(left.basis().coefficients())?;
    let q = equilibrate(right.basis().coefficients())?;
    let u_count = p.len() - degree;
    let v_count = q.len() - degree;
    let rows = checked_add(
        p.len(),
        q.len() - degree - 1,
        scioxide_core::quantity!("gcd_subresultant_rows"),
    )?;
    let cols = checked_add(
        u_count,
        v_count,
        scioxide_core::quantity!("gcd_subresultant_columns"),
    )?;
    let mut subresultant = Matrix::zeros(rows, cols)?;
    convolution_columns(&mut subresultant, 0, 0, &p, v_count, C::ONE);
    convolution_columns(&mut subresultant, 0, v_count, &q, u_count, -C::ONE);
    let factor = ThinSvdDecomposition::compute(&subresultant)?;
    let cofactors = factor.vt_matrix().as_slice();
    let last = &cofactors[(cols - 1) * cols..];
    let mut u = try_collect(
        last[v_count..].iter().map(|&c| Ok(c.conj())),
        scioxide_core::quantity!("gcd_left_cofactor"),
    )?;
    let mut v = try_collect(
        last[..v_count].iter().map(|&c| Ok(c.conj())),
        scioxide_core::quantity!("gcd_right_cofactor"),
    )?;

    let fit_rows = checked_add(p.len(), q.len(), scioxide_core::quantity!("gcd_fit_rows"))?;
    let mut fit = Matrix::zeros(fit_rows, degree + 1)?;
    convolution_columns(&mut fit, 0, 0, &u, degree + 1, C::ONE);
    convolution_columns(&mut fit, p.len(), 0, &v, degree + 1, C::ONE);
    let rhs = try_collect(
        p.iter().chain(&q).copied().map(Ok),
        scioxide_core::quantity!("gcd_fit_rhs"),
    )?;
    let fitted = ThinSvdDecomposition::compute(&fit)?.solve(
        rhs.as_slice(),
        RankPolicy::DimensionScaledEpsilon,
        C::Real::ZERO,
    )?;
    let mut g = try_copy_slice(fitted.as_slice(), scioxide_core::quantity!("gcd_factor"))?;
    let leading = g[degree];
    if leading == C::ZERO {
        return Err(breakdown("rank-selected GCD has zero leading coefficient"));
    }
    for coefficient in &mut g {
        *coefficient = round_complex(scaled_complex(*coefficient)?.div(scaled_complex(leading)?)?)?;
    }
    g[degree] = C::ONE;
    for coefficient in u.iter_mut().chain(&mut v) {
        *coefficient = round_complex(scaled_complex(*coefficient)?.mul(scaled_complex(leading)?)?)?;
    }
    refine(&p, &q, &mut g, &mut u, &mut v)?;
    ApproxPolynomial::monomial_with_options(g, left.policy())
}

fn equilibrate<C: NumericScalar>(values: &[C]) -> Result<Vec<C>, SciOxideError> {
    let exponent = coefficient_exponent(values)?;
    try_collect(
        values
            .iter()
            .map(|&c| round_complex(scaled_complex(c)?.shifted(-exponent)?)),
        scioxide_core::quantity!("gcd_equilibrated_operand"),
    )
}

// Each column is one shifted coefficient vector. All dimensions are constructed
// from the checked operand/target degrees before this bounded kernel is called.
fn convolution_columns<C: NumericScalar>(
    matrix: &mut Matrix<C>,
    row: usize,
    column: usize,
    values: &[C],
    count: usize,
    sign: C,
) {
    let stride = matrix.cols();
    for shift in 0..count {
        for (i, &value) in values.iter().enumerate() {
            matrix.as_mut_slice()[(row + shift + i) * stride + column + shift] = sign * value;
        }
    }
}

fn residual<C: NumericScalar>(
    p: &[C],
    q: &[C],
    g: &[C],
    u: &[C],
    v: &[C],
    output: &mut [C],
) -> Result<Scaled<C::Real>, SciOxideError> {
    let mut squared = Scaled::ZERO;
    let (left_output, right_output) = output.split_at_mut(p.len());
    for ((operand, cofactor), destination) in [(p, u), (q, v)]
        .into_iter()
        .zip([left_output, right_output])
    {
        for (k, entry) in destination.iter_mut().enumerate() {
            let mut value = ScaledValue::from_value(-operand[k])?;
            for (j, &coefficient) in g.iter().enumerate().take(k + 1) {
                if let Some(&other) = cofactor.get(k - j) {
                    value = value.add(
                        ScaledValue::from_value(coefficient)?
                            .mul(ScaledValue::from_value(other)?)?,
                    )?;
                }
            }
            *entry = value.round()?;
            let magnitude = value.magnitude()?;
            squared = squared.add(magnitude.mul(magnitude)?)?;
        }
    }
    Ok(squared)
}

fn refine<C: NumericScalar>(
    p: &[C],
    q: &[C],
    g: &mut [C],
    u: &mut [C],
    v: &mut [C],
) -> Result<(), SciOxideError> {
    let degree = g.len() - 1;
    let rows = p.len() + q.len();
    let cols = degree + u.len() + v.len();
    let mut jacobian = Matrix::zeros(rows, cols)?;
    let mut errors = try_filled(rows, C::ZERO, scioxide_core::quantity!("gcd_fit_residual"))?;
    let mut candidate_errors = try_filled(
        rows,
        C::ZERO,
        scioxide_core::quantity!("gcd_trial_residual"),
    )?;
    let mut candidate = try_filled(
        cols + 1,
        C::ZERO,
        scioxide_core::quantity!("gcd_trial_factors"),
    )?;
    let mut step = Vector::zeros(cols)?;
    let mut factor = ThinSvdDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    let options = LinalgExecutionOptions::new();
    let threshold = C::Real::EPSILON * C::Real::try_from_usize_exact(rows + cols)?;
    let mut error = residual(p, q, g, u, v, &mut errors)?;
    // A finite bound prevents malformed or highly ill-conditioned data from
    // turning an owned algebra operation into an unbounded nonlinear solve.
    for _ in 0..32 {
        if error.is_zero() {
            break;
        }
        jacobian.as_mut_slice().fill(C::ZERO);
        convolution_columns(&mut jacobian, 0, 0, u, degree, C::ONE);
        convolution_columns(&mut jacobian, p.len(), 0, v, degree, C::ONE);
        convolution_columns(&mut jacobian, 0, degree, g, u.len(), C::ONE);
        convolution_columns(&mut jacobian, p.len(), degree + u.len(), g, v.len(), C::ONE);
        ThinSvdDecomposition::compute_into(
            &jacobian,
            &options,
            &mut factor,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        factor.solve_into(
            errors.as_slice(),
            RankPolicy::DimensionScaledEpsilon,
            C::Real::ZERO,
            &options,
            step.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        let mut fraction = C::Real::ONE;
        let mut accepted = false;
        for _ in 0..16 {
            let (trial_g, rest) = candidate.split_at_mut(g.len());
            let (trial_u, trial_v) = rest.split_at_mut(u.len());
            for ((target, &value), &delta) in trial_g[..degree]
                .iter_mut()
                .chain(trial_u.iter_mut())
                .chain(trial_v.iter_mut())
                .zip(g[..degree].iter().chain(u.iter()).chain(v.iter()))
                .zip(step.as_slice())
            {
                *target =
                    round_complex(scaled_complex(value)?.sub(
                        scaled_complex(delta)?.mul(scaled_complex(C::from_real(fraction))?)?,
                    )?)?;
            }
            trial_g[degree] = C::ONE;
            let trial_error = residual(p, q, trial_g, trial_u, trial_v, &mut candidate_errors)?;
            if !error.abs_le(trial_error) {
                g.copy_from_slice(trial_g);
                u.copy_from_slice(trial_u);
                v.copy_from_slice(trial_v);
                errors.copy_from_slice(&candidate_errors);
                error = trial_error;
                accepted = true;
                break;
            }
            fraction *= C::Real::HALF;
        }
        if !accepted {
            break;
        }
        let scale = g
            .iter()
            .chain(u.iter())
            .chain(v.iter())
            .map(|c| c.abs())
            .fold(C::Real::ONE, C::Real::max);
        let change = step
            .as_slice()
            .iter()
            .map(|c| c.abs())
            .fold(C::Real::ZERO, C::Real::max)
            * fraction;
        if change <= threshold * scale {
            break;
        }
    }
    Ok(())
}

fn breakdown(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(NumericalIssue::Breakdown, detail)
}
