//! Structured operations: solves, factorizations, normalizations, implicit
//! solutions, checkpointed loops and complex functions.
#![allow(clippy::unwrap_used, clippy::panic, clippy::needless_range_loop)]

use scioxide_ad::{
    CheckpointSchedule, ComplexElementary, Dual, Jet, Lanes, LoopBody, MultiDual, NonFinitePolicy,
    Real, Residual, Tape, TapeOptions, Var, checkpointed_loop, cholesky_into, dot, gemm_into,
    gemv_into, grad, hessian, hvp, implicit_solution_into, jvp, least_squares_into,
    log_softmax_into, logsumexp, qr_into, softmax_into, solve_into, value_and_grad,
};
use scioxide_core::{
    Complex, ExecutionContext, RangeAccumulator, RealScalar, ResidualTolerance, SciOxideError,
    SciOxideErrorView, checked_float_product,
};
use std::sync::Arc;
use support::bits;

mod support;

const N: usize = 3;

/// A diagonally dominant matrix and right-hand side depending on `t`.
fn system<R: Real>(t: R) -> ([R; N * N], [R; 2 * N]) {
    let a = core::array::from_fn(|k| {
        let (i, j) = (k / N, k % N);
        let base = if i == j {
            6.0 + i as f64
        } else {
            0.5 + 0.1 * (i + 2 * j) as f64
        };
        t * (0.3 + 0.05 * k as f64) + base
    });
    let b = core::array::from_fn(|k| (t * t) * (0.2 * k as f64) - t * 0.4 + (1.0 + k as f64));
    (a, b)
}

/// Gaussian elimination without pivoting, evaluated in `R`.
fn eliminate<R: Real>(mut a: [R; N * N], mut b: [R; 2 * N]) -> [R; 2 * N] {
    for k in 0..N {
        for i in k + 1..N {
            let l = a[i * N + k] / a[k * N + k];
            for j in k..N {
                let update = l * a[k * N + j];
                a[i * N + j] -= update;
            }
            for c in 0..2 {
                let update = l * b[k * 2 + c];
                b[i * 2 + c] -= update;
            }
        }
    }
    for k in (0..N).rev() {
        for c in 0..2 {
            let mut v = b[k * 2 + c];
            for j in k + 1..N {
                v -= a[k * N + j] * b[j * 2 + c];
            }
            b[k * 2 + c] = v / a[k * N + k];
        }
    }
    b
}

fn solved<R: Real>(t: R) -> [R; 2 * N] {
    let (a, b) = system(t);
    let mut x = [R::zero(); 2 * N];
    solve_into(&a, N, &b, 2, &mut x).unwrap();
    x
}

fn eliminated<R: Real>(t: R) -> [R; 2 * N] {
    let (a, b) = system(t);
    eliminate(a, b)
}

/// `Σₖ (k + 1)·xₖ` of the solution for the matrix and right-hand sides stored
/// one after the other in `entries`, by `solve_into` or by elimination.
fn weighted_solution<R: Real>(entries: &[R], by_elimination: bool) -> Result<R, SciOxideError> {
    let a = core::array::from_fn(|k| entries[k]);
    let b = core::array::from_fn(|k| entries[N * N + k]);
    let x = if by_elimination {
        eliminate(a, b)
    } else {
        let mut x = [R::zero(); 2 * N];
        solve_into(&a, N, &b, 2, &mut x)?;
        x
    };
    Ok((0..2 * N).map(|k| x[k] * (k + 1) as f64).sum())
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance * b.abs().max(1.0), "{a} != {b}");
}

#[test]
fn solve_derivatives_match_elimination_in_every_number_type() {
    let t = 0.7;
    let dual = Dual::variable(t);
    for (a, b) in solved(dual).iter().zip(&eliminated(dual)) {
        close(a.value(), b.value(), 1e-14);
        close(a.derivative(), b.derivative(), 1e-13);
    }
    let multi = MultiDual::<f64, 2>::new(t, [1.0, -2.0]);
    for (a, b) in solved(multi).iter().zip(&eliminated(multi)) {
        close(a.value(), b.value(), 1e-14);
        for k in 0..2 {
            close(a.gradient()[k], b.gradient()[k], 1e-13);
        }
    }
    let jet = Jet::<f64, 4>::variable(t);
    for (a, b) in solved(jet).iter().zip(&eliminated(jet)) {
        for k in 0..4 {
            close(a.coefficients()[k], b.coefficients()[k], 1e-12);
        }
    }
    // On a tape the solve is one call statement and the elimination one statement
    // per operation: `Var` gradients and `Var<Dual>` second derivatives agree.
    for output in 0..2 * N {
        let g = grad(|x| Ok(solved(x[0])[output]), &[t]).unwrap();
        let reference = grad(|x| Ok(eliminated(x[0])[output]), &[t]).unwrap();
        close(g.as_slice()[0], reference.as_slice()[0], 1e-13);
        let h = hessian(|x| Ok(solved(x[0])[output]), &[t]).unwrap();
        let reference = hessian(|x| Ok(eliminated(x[0])[output]), &[t]).unwrap();
        close(h.as_slice()[0], reference.as_slice()[0], 1e-12);
    }
    // The same with every matrix and right-hand side entry independent.
    let (a, b) = system(t);
    let point: Vec<f64> = a.iter().chain(&b).copied().collect();
    let direction: Vec<f64> = (0..point.len())
        .map(|k| 0.1 * (k % 5) as f64 - 0.2)
        .collect();
    let g = grad(|x| weighted_solution(x, false), &point).unwrap();
    let reference = grad(|x| weighted_solution(x, true), &point).unwrap();
    for (a, b) in g.as_slice().iter().zip(reference.as_slice()) {
        close(*a, *b, 1e-13);
    }
    let h = hvp(|x| weighted_solution(x, false), &point, &direction).unwrap();
    let reference = hvp(|x| weighted_solution(x, true), &point, &direction).unwrap();
    for (a, b) in h.as_slice().iter().zip(reference.as_slice()) {
        close(*a, *b, 1e-12);
    }
    // Reverse sweeps through the call also serve linearizations.
    let cx = ExecutionContext::sequential();
    let jacobian = |by_elimination: bool| {
        let mut tape = Tape::<f64>::new(TapeOptions::new());
        let mut linear = tape
            .linearize(
                |x, y| {
                    y.extend(if by_elimination {
                        eliminated(x[0])
                    } else {
                        solved(x[0])
                    });
                    Ok(())
                },
                &[t],
                cx,
            )
            .unwrap();
        let mut jacobian = [0.0; 2 * N];
        linear.jacobian_into(&mut jacobian, cx).unwrap();
        jacobian
    };
    for (a, b) in jacobian(false).iter().zip(&jacobian(true)) {
        close(*a, *b, 1e-13);
    }
}

#[test]
fn solves_interchange_and_scale_rows() {
    // A = [[0, s], [ℓ, 0]] with s = 2⁻⁵⁰⁰ and ℓ = 2⁵⁰⁰: x₀ + x₁ = b₁/ℓ + b₀/s.
    fn total<R: Real>(b: &[R]) -> Result<R, SciOxideError> {
        let a = [0.0, 2.0_f64.powi(-500), 2.0_f64.powi(500), 0.0].map(R::constant);
        let mut x = [R::zero(); 2];
        solve_into(&a, 2, b, 1, &mut x)?;
        Ok(x[0] + x[1])
    }
    let (small, large) = (2.0_f64.powi(-500), 2.0_f64.powi(500));
    let g = grad(|b| total(b), &[small, large]).unwrap();
    close(g.as_slice()[0] / large, 1.0, 1e-13);
    close(g.as_slice()[1] / small, 1.0, 1e-13);
    let tangent = jvp(
        |b, y| {
            y.push(total(b)?);
            Ok(())
        },
        &[small, large],
        &[small, large],
    )
    .unwrap();
    close(tangent.as_slice()[0], 2.0, 1e-13);
    // An empty system has an empty solution.
    solve_into::<f64, f64>(&[], 0, &[], 1, &mut []).unwrap();
}

/// The outputs of `kind` for a row-major `rows × cols` matrix followed by a
/// `rows × columns` right-hand side in `x`: the solution, the Cholesky factor, or
/// `R` followed by `Q`.
fn outputs<S: RealScalar, R: Real<S>>(
    kind: &str,
    rows: usize,
    cols: usize,
    x: &[R],
) -> Result<Vec<R>, SciOxideError> {
    let (a, b) = x.split_at(rows * cols);
    let columns = b.len() / rows;
    let len = match kind {
        "solve" | "least_squares" => cols * columns,
        "cholesky" => rows * rows,
        _ => cols * cols + rows * cols,
    };
    let mut out = vec![R::zero(); len];
    match kind {
        "solve" => solve_into(a, rows, b, columns, &mut out)?,
        "least_squares" => least_squares_into(a, rows, cols, b, columns, &mut out)?,
        "cholesky" => cholesky_into(a, rows, &mut out)?,
        _ => {
            let (r, q) = out.split_at_mut(cols * cols);
            qr_into(a, rows, cols, q, r)?;
        }
    }
    Ok(out)
}

fn scalar_factors<S: RealScalar>() {
    let s = |x: f64| S::try_from_f64_exact(x).unwrap();
    // The 1 × 1 output at A = [4], B = [8]: b/a for the solves, √a for Cholesky and
    // R = |a| for QR. Every operation is exact on these dyadic values.
    for (kind, g, h) in [
        ("solve", -0.5, 0.25),
        ("least_squares", -0.5, 0.25),
        ("cholesky", 0.25, -0.03125),
        ("qr", 1.0, 0.0),
    ] {
        let point = [s(4.0), s(8.0)];
        let point = if matches!(kind, "cholesky" | "qr") {
            &point[..1]
        } else {
            &point[..]
        };
        let gradient = grad(|x| Ok(outputs(kind, 1, 1, x)?[0]), point).unwrap();
        let hessian = hessian(|x| Ok(outputs(kind, 1, 1, x)?[0]), point).unwrap();
        assert_eq!(gradient.as_slice()[0], s(g), "{kind}");
        assert_eq!(hessian.as_slice()[0], s(h), "{kind}");
    }
    // Both entries active at (a, b) = (2, 3): ∇(b/a) = (−b/a², 1/a) and
    // H·(½, −¼) = (2b/a³·½ + ¼/a², −½/a²).
    for kind in ["solve", "least_squares"] {
        let point = [s(2.0), s(3.0)];
        let (value, gradient) = value_and_grad(|x| Ok(outputs(kind, 1, 1, x)?[0]), &point).unwrap();
        let product = hvp(
            |x| Ok(outputs(kind, 1, 1, x)?[0]),
            &point,
            &[s(0.5), s(-0.25)],
        )
        .unwrap();
        assert_eq!(value, s(1.5), "{kind}");
        assert_eq!(gradient.as_slice(), [s(-0.75), s(0.5)], "{kind}");
        assert_eq!(product.as_slice(), [s(0.4375), s(-0.125)], "{kind}");
    }
}

#[test]
fn scalar_factors_have_exact_derivatives_in_both_precisions() {
    scalar_factors::<f32>();
    scalar_factors::<f64>();
}

#[test]
fn a_singular_matrix_is_reported() {
    let a = [1.0, 2.0, 2.0, 4.0];
    let mut x = [0.0; 2];
    let error = solve_into(&a, 2, &[1.0, 1.0], 1, &mut x).unwrap_err();
    assert!(matches!(error.view(), SciOxideErrorView::Singular { .. }));
    assert_eq!(x, [0.0; 2]);
    assert!(solve_into(&a, 2, &[1.0], 1, &mut x).is_err());
}

#[test]
fn products_differentiate_their_entries() {
    let g = grad(|x| dot(&x[..2], &x[2..]), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(g.as_slice(), [3.0, 4.0, 1.0, 2.0]);
    let g = grad(
        |x| {
            let mut y = [Var::constant(0.0); 2];
            gemv_into(&x[..4], 2, 2, &x[4..], &mut y)?;
            Ok(y[0] + y[1])
        },
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    assert_eq!(g.as_slice(), [5.0, 6.0, 5.0, 6.0, 4.0, 6.0]);
    let mut c = [0.0; 4];
    gemm_into(
        &[1.0, 2.0, 3.0, 4.0],
        2,
        2,
        &[5.0, 6.0, 7.0, 8.0],
        2,
        &mut c,
    )
    .unwrap();
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn factorizations_have_closed_form_derivatives() {
    // L(t) of [[4 + t, 2], [2, 5]]: l₀₀ = √(4 + t), l₁₀ = 2/l₀₀, l₁₁ = √(5 − l₁₀²).
    let t = 0.5;
    let l = |x: Dual<f64>| {
        let a = [
            x + 4.0,
            Dual::constant(2.0),
            Dual::constant(2.0),
            Dual::constant(5.0),
        ];
        let mut out = [Dual::constant(0.0); 4];
        cholesky_into(&a, 2, &mut out).unwrap();
        out
    };
    let factor = l(Dual::variable(t));
    let l00 = (4.0_f64 + t).sqrt();
    close(factor[0].derivative(), 0.5 / l00, 1e-15);
    close(
        factor[2].derivative(),
        -2.0 / (l00 * l00) * (0.5 / l00),
        1e-14,
    );
    let l10 = 2.0 / l00;
    close(
        factor[3].derivative(),
        -(l10 * (-2.0 / (l00 * l00) * (0.5 / l00))) / (5.0 - l10 * l10).sqrt(),
        1e-14,
    );
    assert!(cholesky_into(&[1.0, 2.0, 2.0, 1.0], 2, &mut [0.0; 4]).is_err());
    // At diag(2, 1) the tangent of l₀₀ = √a₀₀ along a₀₀ is 1/(2√2); a negative
    // pivot is rejected and leaves the output unchanged.
    let tangent = jvp(
        |x, y| {
            y.resize(4, Dual::constant(0.0));
            cholesky_into(x, 2, y)
        },
        &[2.0, 0.0, 0.0, 1.0],
        &[1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    close(tangent.as_slice()[0], 0.5 / 2.0_f64.sqrt(), 1e-15);
    let mut out = [7.0; 4];
    assert!(cholesky_into(&[-1.0, 0.0, 0.0, 1.0], 2, &mut out).is_err());
    assert_eq!(out, [7.0; 4]);

    // A QR without columns does no work proportional to the rows.
    qr_into::<f64, f64>(&[], usize::MAX, 0, &mut [], &mut []).unwrap();
    // Q·R reproduces A and the derivative of R along a column scaling.
    let a = |x: Dual<f64>| {
        [
            x * 3.0,
            Dual::constant(1.0),
            x * 4.0,
            Dual::constant(2.0),
            Dual::constant(1.0),
            Dual::constant(1.0),
        ]
    };
    let (mut q, mut r) = ([Dual::constant(0.0); 6], [Dual::constant(0.0); 4]);
    qr_into(&a(Dual::variable(1.0)), 3, 2, &mut q, &mut r).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            let product: f64 = (0..2)
                .map(|k| q[i * 2 + k].value() * r[k * 2 + j].value())
                .sum();
            close(product, a(Dual::variable(1.0))[i * 2 + j].value(), 1e-14);
        }
    }
    // R₀₀ = ‖first column‖ = √(25t² + 1), so dR₀₀/dt = 25t/R₀₀ at t = 1.
    close(r[0].derivative(), 25.0 / 26.0_f64.sqrt(), 1e-14);
    assert!(r[2].value() == 0.0 && r[3].value() > 0.0);
    // The first column of Q is (3t, 4t, 1)/R₀₀, so dq/dt = (3, 4, −25)/26^(3/2) at
    // t = 1; the second column against central differences of the values.
    for (i, slope) in [3.0, 4.0, -25.0].into_iter().enumerate() {
        close(q[i * 2].derivative(), slope / 26.0_f64.powf(1.5), 1e-14);
    }
    let column = |t: f64| {
        let (mut q, mut r) = ([0.0; 6], [0.0; 4]);
        qr_into(&a(Dual::constant(t)).map(Dual::value), 3, 2, &mut q, &mut r).unwrap();
        q
    };
    let (plus, minus) = (column(1.0 + 1e-5), column(1.0 - 1e-5));
    for i in 0..3 {
        let difference = (plus[i * 2 + 1] - minus[i * 2 + 1]) / 2e-5;
        close(q[i * 2 + 1].derivative(), difference, 1e-9);
    }

    // Least squares of a line fit: the slope of (0, 1), (1, 3 + s), (2, 5) is 2 + s/2·0 …
    let fit = |s: Dual<f64>| {
        let a = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0].map(Dual::constant);
        let b = [Dual::constant(1.0), s + 3.0, Dual::constant(5.0)];
        let mut x = [Dual::constant(0.0); 2];
        least_squares_into(&a, 3, 2, &b, 1, &mut x).unwrap();
        x
    };
    let x = fit(Dual::variable(0.0));
    // x = (AᵀA)⁻¹Aᵀb: ∂x/∂b₁ = row 1 of the pseudo-inverse = (1/3, 0).
    close(x[0].derivative(), 1.0 / 3.0, 1e-14);
    close(x[1].derivative(), 0.0, 1e-14);
}

/// Checks the gradient and `H·d` of `Σ outputs(kind)` against central differences
/// of the value and the gradient with step `10⁻⁵`.
fn central_differences(kind: &str, rows: usize, cols: usize, point: &[f64], d: &[f64]) {
    let evaluate = |x: &[f64]| {
        value_and_grad(|x| Ok(outputs(kind, rows, cols, x)?.into_iter().sum()), x).unwrap()
    };
    let shifted =
        |step: f64| -> Vec<f64> { point.iter().zip(d).map(|(x, d)| x + step * d).collect() };
    let ((_, g), (vp, gp), (vm, gm)) = (
        evaluate(point),
        evaluate(&shifted(1e-5)),
        evaluate(&shifted(-1e-5)),
    );
    let slope: f64 = g.as_slice().iter().zip(d).map(|(g, d)| g * d).sum();
    close((vp - vm) / 2e-5, slope, 1e-8);
    let h = hvp(
        |x| Ok(outputs(kind, rows, cols, x)?.into_iter().sum()),
        point,
        d,
    )
    .unwrap();
    for (h, (p, m)) in h
        .as_slice()
        .iter()
        .zip(gp.as_slice().iter().zip(gm.as_slice()))
    {
        close(*h, (p - m) / 2e-5, 1e-8);
    }
}

#[test]
fn factorizations_differentiate_on_tapes() {
    // Σ L of [[a, ·], [c, d]] is √a + c/√a + √(d − c²/a); its gradient at
    // (a, c, d) = (4, 2, 5) is (3/16, 1/4, 1/4) and the upper triangle is not read.
    let g = grad(
        |x| Ok(outputs("cholesky", 2, 2, x)?.into_iter().sum()),
        &[4.0, 2.0, 2.0, 5.0],
    )
    .unwrap();
    assert_eq!(g.as_slice(), [0.1875, 0.0, 0.25, 0.25]);
    // Σ L, Σ Q + Σ R and the sums of square and tall solutions with two right-hand
    // sides, to second order.
    let cycle =
        |len: usize| -> Vec<f64> { (0..len).map(|i| ((i % 5) as f64 - 2.0) * 0.2).collect() };
    for (kind, rows, cols, point, direction) in [
        (
            "cholesky",
            2,
            2,
            vec![4.0, 1.0, 1.0, 3.0],
            vec![0.3, -0.2, -0.2, 0.4],
        ),
        (
            "qr",
            3,
            2,
            vec![2.0, 0.5, -0.3, 1.0, 0.7, -0.4],
            vec![0.1, -0.2, 0.3, 0.1, -0.1, 0.2],
        ),
        (
            "solve",
            2,
            2,
            vec![3.0, 1.0, -1.0, 2.0, 1.0, 3.0, 2.0, -1.0],
            cycle(8),
        ),
        (
            "least_squares",
            3,
            2,
            vec![2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 3.0, 2.0, -1.0, 0.5, 2.0],
            cycle(12),
        ),
    ] {
        central_differences(kind, rows, cols, &point, &direction);
    }
}

#[test]
fn least_squares_match_dyadic_values_and_report_overflow() {
    // A stacks I₁₆ four times and B = (i mod 23)/8 − 1 is inconsistent: x is the
    // mean of the four blocks, and the derivatives below are dyadic. Householder
    // QR in the number type rounds them.
    let (m, n, p) = (64, 16, 3);
    let mut point: Vec<f64> = (0..m * n)
        .map(|i| if (i / n) % n == i % n { 1.0 } else { 0.0 })
        .collect();
    point.extend((0..m * p).map(|i| (i % 23) as f64 / 8.0 - 1.0));
    let direction: Vec<f64> = (0..point.len())
        .map(|i| (i % 13) as f64 / 32.0 - 0.125)
        .collect();
    let (value, g) = value_and_grad(
        |x| Ok(outputs("least_squares", m, n, x)?.into_iter().sum()),
        &point,
    )
    .unwrap();
    let h = hvp(
        |x| Ok(outputs("least_squares", m, n, x)?.into_iter().sum()),
        &point,
        &direction,
    )
    .unwrap();
    let (g, h) = (g.as_slice(), h.as_slice());
    let last = point.len() - 1;
    for (actual, expected) in [value, g[0], g[1], g[m * n], h[0], h[1], h[m * n], h[last]]
        .into_iter()
        .zip([
            16.125,
            0.09375,
            -0.1875,
            0.25,
            0.160888671875,
            0.62396240234375,
            -0.330078125,
            -0.201171875,
        ])
    {
        close(actual, expected, 1e-14);
    }
    // A = 1 (4 × 1) and b = s·(1, −1, −1, −1): x = −s/2, ∂x/∂b = ¼ and
    // ∂x/∂A = (r − x)/4 = (s/2, 0, 0, 0) through the residual r = s·(3, −1, −1, −1)/2,
    // which overflows for s near the largest finite value.
    let cx = ExecutionContext::sequential();
    for s in [1.0, 0.875 * f64::MAX] {
        let mut point = vec![1.0; 4];
        point.extend([s, -s, -s, -s]);
        let passive = grad(
            |b| {
                let mut x = vec![Var::constant(1.0); 4];
                x.extend_from_slice(b);
                Ok(outputs("least_squares", 4, 1, &x)?[0])
            },
            &point[4..],
        )
        .unwrap();
        for g in passive.as_slice() {
            assert!((g - 0.25).abs() <= 8.0 * f64::EPSILON, "{g}");
        }
        let options = TapeOptions::new().with_non_finite_policy(NonFinitePolicy::Error);
        let mut tape = Tape::<f64>::new(options);
        let mut gradient = [19.0; 8];
        let result = tape.gradient_into(
            |x| Ok(outputs("least_squares", 4, 1, x)?[0]),
            &point,
            &mut gradient,
            cx,
        );
        if s > 1.0 {
            let error = result.unwrap_err();
            assert!(matches!(error.view(), SciOxideErrorView::NonFinite { .. }));
            assert_eq!(gradient, [19.0; 8]);
        } else {
            close(result.unwrap(), -0.5, 1e-15);
            for (g, e) in gradient
                .iter()
                .zip([0.5, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
            {
                assert!((g - e).abs() <= 8.0 * f64::EPSILON, "{g} != {e}");
            }
        }
    }
}

/// `s ← y(s)` for the 1 × 1 output `y` of `kind` on `A = [s]` (`[s, 0]ᵀ` for the
/// rectangular kinds) and `B = [1]` (`[1, 0]ᵀ`).
struct Factored(&'static str);

impl LoopBody for Factored {
    fn step<R: Real>(&self, _: usize, state: &mut [R], _: &[R]) -> Result<(), SciOxideError> {
        let (zero, one) = (R::zero(), R::one());
        state[0] = if matches!(self.0, "least_squares" | "qr") {
            outputs(self.0, 2, 1, &[state[0], zero, one, zero])?[0]
        } else {
            outputs(self.0, 1, 1, &[state[0], one])?[0]
        };
        Ok(())
    }
}

/// Three steps of `body`, plain or checkpointed with `schedule`.
fn three_steps<R: Real>(
    body: &Arc<Factored>,
    schedule: Option<&CheckpointSchedule>,
    s: R,
) -> Result<R, SciOxideError> {
    let mut state = [s];
    match schedule {
        Some(schedule) => checkpointed_loop(body, 3, &mut state, &[], schedule)?,
        None => {
            for index in 0..3 {
                body.step(index, &mut state, &[])?;
            }
        }
    }
    Ok(state[0])
}

#[test]
fn factors_compose_in_checkpointed_loops() {
    // From s = 2, three steps of s ↦ 1/s end at 1/s, of s ↦ √s at s^(1/8) and of
    // s ↦ |s| at s: value, first and second derivative.
    let root = |k: f64| 2.0_f64.powf(0.125 - k);
    let schedule = CheckpointSchedule::binomial(3, 1).unwrap();
    for (kind, expected) in [
        ("solve", [0.5, -0.25, 0.25]),
        ("least_squares", [0.5, -0.25, 0.25]),
        (
            "cholesky",
            [root(0.0), 0.125 * root(1.0), -0.109375 * root(2.0)],
        ),
        ("qr", [2.0, 1.0, 0.0]),
    ] {
        let body = Arc::new(Factored(kind));
        for schedule in [None, Some(&schedule)] {
            let (value, g) =
                value_and_grad(|x| three_steps(&body, schedule, x[0]), &[2.0]).unwrap();
            let h = hessian(|x| three_steps(&body, schedule, x[0]), &[2.0]).unwrap();
            close(value, expected[0], 1e-13);
            close(g.as_slice()[0], expected[1], 1e-12);
            close(h.as_slice()[0], expected[2], 1e-11);
        }
    }
}

#[test]
fn normalizations_have_softmax_derivatives() {
    let x = [0.3, -1.2, 2.0];
    let g = grad(|v| logsumexp(v), &x).unwrap();
    let m = 2.0_f64;
    let s: f64 = x.iter().map(|v| (v - m).exp()).sum();
    for (g, v) in g.as_slice().iter().zip(&x) {
        close(*g, (v - m).exp() / s, 1e-15);
    }
    assert_eq!(logsumexp::<f64, f64>(&[]).unwrap(), f64::NEG_INFINITY);
    assert_eq!(
        logsumexp(&[f64::NEG_INFINITY; 2]).unwrap(),
        f64::NEG_INFINITY
    );
    // ∂pᵢ/∂xⱼ = pᵢ(δᵢⱼ − pⱼ).
    let p: Vec<f64> = x.iter().map(|v| (v - m).exp() / s).collect();
    for i in 0..3 {
        let g = grad(
            |v| {
                let mut out = [Var::constant(0.0); 3];
                softmax_into(v, &mut out)?;
                Ok(out[i])
            },
            &x,
        )
        .unwrap();
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            close(g.as_slice()[j], p[i] * (delta - p[j]), 1e-14);
        }
        let g = grad(
            |v| {
                let mut out = [Var::constant(0.0); 3];
                log_softmax_into(v, &mut out)?;
                Ok(out[i])
            },
            &x,
        )
        .unwrap();
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            close(g.as_slice()[j], delta - p[j], 1e-14);
        }
    }
    let mut out = [1.0; 2];
    softmax_into(&[0.0, f64::NEG_INFINITY], &mut out).unwrap();
    assert_eq!(out, [1.0, 0.0]);
    let error = softmax_into(&[f64::NEG_INFINITY; 2], &mut out).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DomainError { .. }
    ));
}

const NORMALIZATIONS: [&str; 3] = ["softmax", "log_softmax", "logsumexp"];

/// Pushes the normalization `kind` of `x` onto `y`.
fn normalized<S: RealScalar, R: Real<S>>(
    kind: &str,
    x: &[R],
    y: &mut Vec<R>,
) -> Result<(), SciOxideError> {
    if kind == "logsumexp" {
        y.push(logsumexp(x)?);
        return Ok(());
    }
    y.resize(x.len(), R::zero());
    if kind == "softmax" {
        softmax_into(x, y)
    } else {
        log_softmax_into(x, y)
    }
}

/// `Σᵢ wᵢ·yᵢ` of the normalization `kind`, or log-sum-exp itself.
fn weighted<S: RealScalar, R: Real<S>>(
    kind: &str,
    x: &[R],
    weights: &[S],
) -> Result<R, SciOxideError> {
    let mut y = Vec::new();
    normalized(kind, x, &mut y)?;
    Ok(if kind == "logsumexp" {
        y[0]
    } else {
        y.iter().zip(weights).map(|(&y, &w)| y * w).sum()
    })
}

/// The cotangent `x̄ᵢ` of `kind` at the probabilities `p` for the output cotangents
/// `ȳ`, from correctly rounded sums that exclude entry `i`: `pᵢ·ȳ₀` for
/// log-sum-exp, `ȳᵢ·Σⱼ≠ᵢ pⱼ − pᵢ·Σⱼ≠ᵢ ȳⱼ` for log-softmax and
/// `pᵢ·(ȳᵢ·Σⱼ≠ᵢ pⱼ − Σⱼ≠ᵢ pⱼȳⱼ)` for softmax.
fn excluded_cotangent<S: RealScalar>(kind: &str, p: &[S], bars: &[S], i: usize) -> S {
    if kind == "logsumexp" {
        return checked_float_product(p[i], bars[0]).unwrap();
    }
    let mut sums = [(); 3].map(|()| RangeAccumulator::new());
    for j in (0..p.len()).filter(|&j| j != i) {
        sums[0].add(p[j]).unwrap();
        sums[1].add(bars[j]).unwrap();
        sums[2].add_product(p[j], bars[j]).unwrap();
    }
    let [weights, others, weighted] = sums.map(|sum| sum.finish().unwrap());
    let mut result = RangeAccumulator::new();
    result.add_product(bars[i], weights).unwrap();
    if kind == "log_softmax" {
        result.subtract_product(p[i], others).unwrap();
        return result.finish().unwrap();
    }
    result.add(-weighted).unwrap();
    checked_float_product(p[i], result.finish().unwrap()).unwrap()
}

fn normalization_products<S: RealScalar>() {
    let s = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let f = |x: S| x.into_f64();
    let cx = ExecutionContext::sequential();
    for (n, dominant) in [(1, false), (1, true), (4096, false), (4096, true)] {
        let mut logits = vec![s(0.0); n];
        if dominant {
            logits[0] = s(80.0);
        }
        let mut p = vec![s(0.0); n];
        softmax_into(&logits, &mut p).unwrap();
        for kind in NORMALIZATIONS {
            let outputs = if kind == "logsumexp" { 1 } else { n };
            let bars: Vec<S> = (0..outputs)
                .map(|i| s((i % 19) as f64 / 8.0 - 1.0))
                .collect();
            let mut tape = Tape::<S>::new(TapeOptions::new());
            let mut linear = tape
                .linearize(|x, y| normalized(kind, x, y), &logits, cx)
                .unwrap();
            let mut gradient = vec![s(0.0); n];
            linear.vjp_into(&bars, &mut gradient, cx).unwrap();
            // Shift invariance: the tangent along (¼, …, ¼) is ¼ for log-sum-exp and 0
            // otherwise; scaling by ¼ is exact, so both terms of a tangent round alike.
            let mut tangent = vec![s(1.0); outputs];
            linear
                .jvp_into(&vec![s(0.25); n], &mut tangent, cx)
                .unwrap();
            let shift = s(if kind == "logsumexp" { 0.25 } else { 0.0 });
            assert!(tangent.iter().all(|&t| bits(t) == bits(shift)), "{kind}");
            for i in [0, n / 2, n - 1] {
                let expected = excluded_cotangent(kind, &p, &bars, i);
                if kind == "softmax" && dominant {
                    // The tape forms pᵢ·(ȳᵢ − Σⱼ pⱼȳⱼ). With p₀ rounded to 1 the bracket
                    // cancels to 0 at i = 0 and loses the excluded sum
                    // Σⱼ≠₀ pⱼ·(ȳ₀ − ȳⱼ) ≈ −8.3·10⁻³², so this case holds to the rounding
                    // bound 4ε·pᵢ·(|ȳᵢ| + Σⱼ pⱼ|ȳⱼ|) of that formula, not bit for bit.
                    let mass: f64 = p.iter().zip(&bars).map(|(&p, &b)| f(p) * f(b).abs()).sum();
                    let bound = 4.0 * f(S::EPSILON) * f(p[i]) * (f(bars[i]).abs() + mass);
                    assert!((f(gradient[i]) - f(expected)).abs() <= bound, "{i}");
                } else {
                    let context = format!("{kind}, n = {n}, dominant = {dominant}, i = {i}");
                    assert_eq!(bits(gradient[i]), bits(expected), "{context}");
                }
            }
        }
    }
}

#[test]
fn normalization_products_match_excluded_sums_in_both_precisions() {
    normalization_products::<f32>();
    normalization_products::<f64>();
}

/// Asserts `|actual − expected| ≤ 128·ε·|expected|` in the precision of `S`.
fn near<S: RealScalar>(actual: S, expected: f64) {
    let bound = 128.0 * S::EPSILON.into_f64() * expected.abs();
    assert!(
        (actual.into_f64() - expected).abs() <= bound,
        "{actual:?} != {expected}"
    );
}

fn normalization_curvature<S: RealScalar>() {
    let s = |x: f64| S::try_from_f64_lossy(x).unwrap();
    // Three logits at zero: the gradient and H·e₀ of y₀, or of log-sum-exp.
    let e0 = [s(1.0), s(0.0), s(0.0)];
    let (third, ninth, twenty_seventh) = (1.0 / 3.0, 1.0 / 9.0, 1.0 / 27.0);
    for (kind, gradient, curvature) in [
        (
            "softmax",
            [2.0 * ninth, -ninth, -ninth],
            [2.0 * twenty_seventh, -twenty_seventh, -twenty_seventh],
        ),
        (
            "log_softmax",
            [2.0 * third, -third, -third],
            [-2.0 * ninth, ninth, ninth],
        ),
        ("logsumexp", [third; 3], [2.0 * ninth, -ninth, -ninth]),
    ] {
        let g = grad(|x| weighted(kind, x, &e0), &[s(0.0); 3]).unwrap();
        let h = hvp(|x| weighted(kind, x, &e0), &[s(0.0); 3], &e0).unwrap();
        for k in 0..3 {
            near(g.as_slice()[k], gradient[k]);
            near(h.as_slice()[k], curvature[k]);
        }
    }
    // Log-sum-exp of (c, c) is c + ln 2 with gradient (½, ½) and H·e₀ = (¼, −¼),
    // for a moderate and a large shift.
    for c in [2.0, 1000.0] {
        let point = [s(c); 2];
        let (value, g) = value_and_grad(|x| logsumexp(x), &point).unwrap();
        let h = hvp(|x| logsumexp(x), &point, &[s(1.0), s(0.0)]).unwrap();
        near(value, c + core::f64::consts::LN_2);
        near(g.as_slice()[0], 0.5);
        near(g.as_slice()[1], 0.5);
        near(h.as_slice()[0], 0.25);
        near(h.as_slice()[1], -0.25);
    }
    // 4096 logits at zero, p = 1/n: H·d of Σ wᵢpᵢ, Σ wᵢ ln pᵢ and log-sum-exp is
    // (d̃ᵢw̃ᵢ − cov(w, d))/n, −d̃ᵢ·w̄ and d̃ᵢ/n, where x̃ = x − x̄.
    let n = 4096;
    let weights: Vec<S> = (0..n).map(|i| s((i % 19) as f64 / 8.0 - 1.0)).collect();
    let direction: Vec<S> = (0..n).map(|i| s((i % 7) as f64 / 8.0 - 0.25)).collect();
    let mean = |v: &[S]| v.iter().map(|x| x.into_f64()).sum::<f64>() / n as f64;
    let (mean_w, mean_d) = (mean(&weights), mean(&direction));
    let covariance = weights
        .iter()
        .zip(&direction)
        .map(|(w, d)| (w.into_f64() - mean_w) * (d.into_f64() - mean_d))
        .sum::<f64>()
        / n as f64;
    for kind in NORMALIZATIONS {
        let h = hvp(
            |x| weighted(kind, x, &weights),
            &vec![s(0.0); n],
            &direction,
        )
        .unwrap();
        for ((h, w), d) in h.as_slice().iter().zip(&weights).zip(&direction) {
            let centered = d.into_f64() - mean_d;
            let expected = match kind {
                "softmax" => (centered * (w.into_f64() - mean_w) - covariance) / n as f64,
                "log_softmax" => -centered * mean_w,
                _ => centered / n as f64,
            };
            let tolerance = 128.0 * S::EPSILON.into_f64() * expected.abs().max(1.0 / n as f64);
            assert!((h.into_f64() - expected).abs() <= tolerance, "{kind}");
        }
    }
}

#[test]
fn normalization_curvature_matches_closed_forms_in_both_precisions() {
    normalization_curvature::<f32>();
    normalization_curvature::<f64>();
}

/// `y³ + p₀·y − p₁ = 0`.
struct Cubic;

impl Residual for Cubic {
    fn residual<R: Real<f64, Mask = bool>>(
        &self,
        y: &[R],
        p: &[R],
        out: &mut [R],
    ) -> Result<(), SciOxideError> {
        out[0] = y[0] * y[0] * y[0] + p[0] * y[0] - p[1];
        Ok(())
    }
}

#[test]
fn implicit_solutions_match_closed_forms_to_second_order() {
    // At p = (2, 3) the root is y = 1; F_y = 3y² + p₀ = 5, F_p = (y, −1).
    let tolerance = ResidualTolerance::new(1e-12, 0.0).unwrap();
    let g = grad(
        |p| {
            let mut y = [p[0]];
            implicit_solution_into(&Cubic, &[1.0], p, tolerance, &mut y)?;
            Ok(y[0])
        },
        &[2.0, 3.0],
    )
    .unwrap();
    close(g.as_slice()[0], -1.0 / 5.0, 1e-15);
    close(g.as_slice()[1], 1.0 / 5.0, 1e-15);
    // Second order: differentiate y' = −F_p/F_y once more, here ∂²y/∂p₁² = −6y·y₁²/F_y.
    let h = hessian(
        |p| {
            let mut y = [p[0]];
            implicit_solution_into(&Cubic, &[1.0], p, tolerance, &mut y)?;
            Ok(y[0])
        },
        &[2.0, 3.0],
    )
    .unwrap();
    let y1 = 1.0 / 5.0;
    close(h.as_slice()[3], -6.0 * y1 * y1 / 5.0, 1e-13);
    // A point that misses the tolerance is rejected.
    let mut y = [0.0];
    let error = implicit_solution_into(&Cubic, &[1.1], &[2.0, 3.0], tolerance, &mut y).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
}

/// `s ← (s + p₀)·sin(s)·p₁ + s`.
struct Map;

impl LoopBody for Map {
    fn step<R: Real>(&self, index: usize, state: &mut [R], p: &[R]) -> Result<(), SciOxideError> {
        let s = state[0];
        state[0] = (s + p[0]) * s.sin() * p[1] * (1.0 / (1.0 + index as f64)) + s;
        state[1] = state[1] * 0.9 + s * p[0];
        Ok(())
    }
}

#[test]
fn checkpointed_loops_match_the_plain_loop() {
    let steps = 13;
    let body = Arc::new(Map);
    let point = [0.4, -0.2, 0.7, 1.1];
    let plain = grad(
        |x| {
            let mut state = [x[0], x[1]];
            for index in 0..steps {
                body.step(index, &mut state, &x[2..])?;
            }
            Ok(state[0] + state[1] * 2.0)
        },
        &point,
    )
    .unwrap();
    let direction = [0.3, -0.7, 1.1, 0.5];
    let plain_hv = hvp(
        |x| {
            let mut state = [x[0], x[1]];
            for index in 0..steps {
                body.step(index, &mut state, &x[2..])?;
            }
            Ok(state[0] + state[1] * 2.0)
        },
        &point,
        &direction,
    )
    .unwrap();
    // Binomial schedules, and explicit ones with one, two and three boundaries: the
    // states at the later boundaries are stored before the reversal.
    let mut schedules: Vec<_> = [1, 2, 5]
        .map(|slots| CheckpointSchedule::binomial(steps, slots).unwrap())
        .into();
    for boundaries in [&[0][..], &[0, 5], &[0, 4, 9], &[0, 1, 12]] {
        schedules.push(CheckpointSchedule::explicit(steps, boundaries).unwrap());
    }
    for schedule in &schedules {
        let checkpointed = grad(
            |x| {
                let mut state = [x[0], x[1]];
                checkpointed_loop(&body, steps, &mut state, &x[2..], schedule)?;
                Ok(state[0] + state[1] * 2.0)
            },
            &point,
        )
        .unwrap();
        for (a, b) in checkpointed.as_slice().iter().zip(plain.as_slice()) {
            close(*a, *b, 1e-12);
        }
        let checkpointed = hvp(
            |x| {
                let mut state = [x[0], x[1]];
                checkpointed_loop(&body, steps, &mut state, &x[2..], schedule)?;
                Ok(state[0] + state[1] * 2.0)
            },
            &point,
            &direction,
        )
        .unwrap();
        for (a, b) in checkpointed.as_slice().iter().zip(plain_hv.as_slice()) {
            close(*a, *b, 1e-11);
        }
    }
    // Tangent products run the loop on duals.
    let schedule = CheckpointSchedule::binomial(steps, 2).unwrap();
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let cx = ExecutionContext::sequential();
    let mut linear = tape
        .linearize(
            |x, y| {
                let mut state = [x[0], x[1]];
                checkpointed_loop(&body, steps, &mut state, &x[2..], &schedule)?;
                y.push(state[0] + state[1] * 2.0);
                Ok(())
            },
            &point,
            cx,
        )
        .unwrap();
    let mut jv = [0.0];
    linear.jvp_into(&[1.0, 0.0, 0.0, 0.0], &mut jv, cx).unwrap();
    close(jv[0], plain.as_slice()[0], 1e-12);
    assert!(linear.jacobian_operator().is_err());
    // The schedule must cover the loop.
    let mut state = [0.0; 2];
    assert!(checkpointed_loop(&body, steps + 1, &mut state, &[0.0; 2], &schedule).is_err());
}

/// `s ← p·s`: after `n` steps from `s₀`, `s = pⁿ·s₀`, `∂s/∂s₀ = pⁿ` and
/// `∂s/∂p = n·pⁿ⁻¹·s₀`.
struct Scale;

impl LoopBody for Scale {
    fn step<R: Real>(&self, _: usize, state: &mut [R], p: &[R]) -> Result<(), SciOxideError> {
        state[0] *= p[0];
        Ok(())
    }
}

#[test]
fn explicit_schedules_match_the_closed_form() {
    let (steps, s0, p) = (6, 1.5_f64, 1.2_f64);
    let expected = [p.powi(6), 6.0 * p.powi(5) * s0];
    let schedules = [
        CheckpointSchedule::explicit(steps, &[0]).unwrap(),
        CheckpointSchedule::explicit(steps, &[0, 3]).unwrap(),
        CheckpointSchedule::explicit(steps, &[0, 2, 4]).unwrap(),
        CheckpointSchedule::explicit(steps, &[0, 1, 2, 5]).unwrap(),
        CheckpointSchedule::binomial(steps, 2).unwrap(),
    ];
    for schedule in &schedules {
        let (value, g) = value_and_grad(
            |x| {
                let mut state = [x[0]];
                checkpointed_loop(&Arc::new(Scale), steps, &mut state, &x[1..], schedule)?;
                Ok(state[0])
            },
            &[s0, p],
        )
        .unwrap();
        close(value, p.powi(6) * s0, 1e-14);
        close(g.as_slice()[0], expected[0], 1e-14);
        close(g.as_slice()[1], expected[1], 1e-13);
    }
}

#[test]
fn the_complex_inverse_counts_no_kinks() {
    // Re(1/z) = x/(x² + y²): ∂ₓ = (y² − x²)/(x² + y²)², ∂ᵧ = −2xy/(x² + y²)². The
    // overflow scale max(|x|, |y|) is not differentiated, so a real z, an imaginary
    // z and |x| = |y| are no kinks, and the strict policy accepts them.
    let mut tape =
        Tape::<f64>::new(TapeOptions::new().with_kink_policy(scioxide_ad::KinkPolicy::Error));
    let cx = ExecutionContext::sequential();
    for (x, y) in [
        (2.0, 0.0),
        (-3.0, 0.0),
        (0.0, 0.5),
        (1.0, 1.0),
        (-1.0, 1.0),
        (0.8, -0.3),
    ] {
        let mut g = [0.0; 2];
        let value = tape
            .gradient_into(
                |p| Ok(ComplexElementary::inv(Complex::new(p[0], p[1])).re),
                &[x, y],
                &mut g,
                cx,
            )
            .unwrap();
        let r2: f64 = x * x + y * y;
        close(value, x / r2, 1e-15);
        close(g[0], (y * y - x * x) / (r2 * r2), 1e-15);
        close(g[1], -2.0 * x * y / (r2 * r2), 1e-15);
        assert_eq!(tape.report().kinks(), 0, "({x}, {y})");
    }
}

/// A 4 × 4 system that partial pivoting reorders at more than one step, with two
/// right-hand sides, depending on `t`.
fn pivoting<R: Real>(t: R) -> ([R; 16], [R; 8]) {
    let base = [
        1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0, 10.0, 2.0, 2.0, 9.0, 1.0, 3.0,
    ];
    let a = core::array::from_fn(|k| t * (0.1 + 0.02 * k as f64) + base[k]);
    let b = core::array::from_fn(|k| t * t * (0.05 * k as f64) + (1.0 + k as f64) * 0.5);
    (a, b)
}

/// Gaussian elimination with partial pivoting on base values, evaluated in `R`.
fn eliminate_pivoted<R: Real>(a: &[R], b: &[R], n: usize) -> Vec<R> {
    use scioxide_ad::BaseReal;
    let cols = b.len() / n;
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    let magnitude = |x: R| x.base().lane(0).unwrap().abs();
    for k in 0..n {
        let p = (k..n).fold(k, |p, i| {
            if magnitude(a[i * n + k]) > magnitude(a[p * n + k]) {
                i
            } else {
                p
            }
        });
        for j in 0..n {
            a.swap(k * n + j, p * n + j);
        }
        for c in 0..cols {
            b.swap(k * cols + c, p * cols + c);
        }
        for i in k + 1..n {
            let l = a[i * n + k] / a[k * n + k];
            for j in k..n {
                let update = l * a[k * n + j];
                a[i * n + j] -= update;
            }
            for c in 0..cols {
                let update = l * b[k * cols + c];
                b[i * cols + c] -= update;
            }
        }
    }
    for k in (0..n).rev() {
        for c in 0..cols {
            let mut v = b[k * cols + c];
            for j in k + 1..n {
                v -= a[k * n + j] * b[j * cols + c];
            }
            b[k * cols + c] = v / a[k * n + k];
        }
    }
    b
}

fn pivot_solution<R: Real>(t: R, by_elimination: bool) -> Result<R, SciOxideError> {
    let (a, b) = pivoting(t);
    let x = if by_elimination {
        eliminate_pivoted(&a, &b, 4)
    } else {
        let mut x = vec![R::zero(); 8];
        solve_into(&a, 4, &b, 2, &mut x)?;
        x
    };
    Ok(x.iter()
        .enumerate()
        .map(|(k, &v)| v * (k as f64 + 1.0))
        .sum())
}

#[test]
fn solves_with_row_interchanges_at_later_steps_are_exact() {
    // Pivoting swaps rows at steps 0 and 1: x = (5, −14, 8).
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
    let mut x = [0.0; 3];
    solve_into(&a, 3, &[1.0, -2.0, 3.0], 1, &mut x).unwrap();
    for (x, e) in x.iter().zip([5.0, -14.0, 8.0]) {
        close(*x, e, 1e-14);
    }
    // Every number type agrees with elimination with partial pivoting.
    let t = 0.4;
    close(
        pivot_solution(t, false).unwrap(),
        pivot_solution(t, true).unwrap(),
        1e-13,
    );
    let d = |by| pivot_solution(Dual::variable(t), by).unwrap().derivative();
    close(d(false), d(true), 1e-12);
    let jet = |by| {
        pivot_solution(Jet::<f64, 3>::variable(t), by)
            .unwrap()
            .coefficients()
    };
    for (a, b) in jet(false).iter().zip(jet(true)) {
        close(*a, b, 1e-11);
    }
    let g = |by: bool| grad(|x| pivot_solution(x[0], by), &[t]).unwrap().as_slice()[0];
    close(g(false), g(true), 1e-12);
    let h = |by: bool| {
        hessian(|x| pivot_solution(x[0], by), &[t])
            .unwrap()
            .as_slice()[0]
    };
    close(h(false), h(true), 1e-11);
    // Forward over reverse: the derivative part of a `Dual<Var>` solve.
    let nested = |by: bool| {
        grad(
            |x| pivot_solution(Dual::new(x[0], Var::constant(1.0)), by).map(|v| v.derivative()),
            &[t],
        )
        .unwrap()
        .as_slice()[0]
    };
    close(nested(false), nested(true), 1e-11);
    // Lanes: lane 0 at t, lane 1 at a point where the pivots differ; each lane has
    // the bits of the scalar solve.
    let lanes = Lanes::<f64, 2>::from_fn(|k| [t, -9.0][k]);
    let both = pivot_solution(lanes, false).unwrap();
    for (lane, t) in [t, -9.0].into_iter().enumerate() {
        assert_eq!(
            both.lane(lane).unwrap().to_bits(),
            pivot_solution(t, false).unwrap().to_bits()
        );
    }
}

/// `F(y, p) = A·y − p` for the 3 × 3 matrix of the pivoting test.
struct Linear3;

impl Residual for Linear3 {
    fn residual<R: Real<f64, Mask = bool>>(
        &self,
        y: &[R],
        p: &[R],
        out: &mut [R],
    ) -> Result<(), SciOxideError> {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        for i in 0..3 {
            out[i] = (0..3).map(|j| y[j] * a[i * 3 + j]).sum::<R>() - p[i];
        }
        Ok(())
    }
}

#[test]
fn implicit_derivatives_pivot_and_check_their_residual_without_overflow() {
    // y = A⁻¹p, so ∂y₀/∂p is the first row of A⁻¹ = (−2/3, −4/3, 1).
    let tolerance = ResidualTolerance::new(1e-12, 1e-12).unwrap();
    let p = [1.0, -2.0, 3.0];
    let g = grad(
        |p| {
            let mut y = [Var::constant(0.0); 3];
            implicit_solution_into(&Linear3, &[5.0, -14.0, 8.0], p, tolerance, &mut y)?;
            Ok(y[0])
        },
        &p,
    )
    .unwrap();
    for (g, e) in g.as_slice().iter().zip([-2.0 / 3.0, -4.0 / 3.0, 1.0]) {
        close(*g, e, 1e-14);
    }
    // y = p = 1e200 has an exact residual; the norms do not overflow.
    struct Identity;
    impl Residual for Identity {
        fn residual<R: Real<f64, Mask = bool>>(
            &self,
            y: &[R],
            p: &[R],
            out: &mut [R],
        ) -> Result<(), SciOxideError> {
            out[0] = y[0] - p[0];
            Ok(())
        }
    }
    let mut y = [0.0];
    implicit_solution_into(&Identity, &[1e200], &[1e200], tolerance, &mut y).unwrap();
    assert_eq!(y, [1e200]);
    let mut y = [Dual::constant(0.0)];
    implicit_solution_into(
        &Identity,
        &[1e160],
        &[Dual::variable(1e160)],
        tolerance,
        &mut y,
    )
    .unwrap();
    assert_eq!((y[0].value(), y[0].derivative()), (1e160, 1.0));
    // A NaN residual or solution misses the tolerance.
    for (solution, parameter) in [(f64::NAN, 1.0), (1.0, f64::NAN), (1e300, -1e300)] {
        let mut y = [7.0];
        let error = implicit_solution_into(&Identity, &[solution], &[parameter], tolerance, &mut y)
            .unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::InvalidInput { .. }),
            "{error}"
        );
        assert_eq!(y, [7.0]);
    }
    // A Jet gets c₀ and c₁; `combine` carries one derivative order, so c₂ is NaN.
    let mut y = [Jet::<f64, 3>::constant(0.0)];
    implicit_solution_into(&Identity, &[2.0], &[Jet::variable(2.0)], tolerance, &mut y).unwrap();
    let c = y[0].coefficients();
    assert_eq!((c[0], c[1]), (2.0, 1.0));
    assert!(c[2].is_nan());
}

#[test]
fn least_squares_checks_shapes_before_allocating_and_rejects_nan_pivots() {
    let mut empty: [f64; 0] = [];
    let error = least_squares_into::<f64, f64>(&[], usize::MAX, 2, &[], 0, &mut empty).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::SizeOverflow { .. }),
        "{error}"
    );
    let error =
        least_squares_into::<f64, f64>(&[1.0], 1 << 31, 1 << 31, &[], 0, &mut empty).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. }),
        "{error}"
    );
    let error =
        least_squares_into::<f64, f64>(&[1.0, 2.0], 1, 2, &[1.0], 1, &mut [0.0; 2]).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. }),
        "{error}"
    );
    let mut x = [9.0; 2];
    let error =
        least_squares_into(&[f64::NAN, 0.0, 0.0, 1.0], 2, 2, &[1.0, 1.0], 1, &mut x).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::Singular { .. }),
        "{error}"
    );
    assert_eq!(x, [9.0; 2]);
}

#[test]
fn normalizations_propagate_nan_and_reject_empty_input() {
    let nan = f64::NAN;
    for values in [
        &[nan][..],
        &[nan, f64::NEG_INFINITY],
        &[f64::INFINITY, nan],
        &[1.0, nan],
    ] {
        assert!(logsumexp(values).unwrap().is_nan(), "{values:?}");
        let mut p = vec![0.0; values.len()];
        softmax_into(values, &mut p).unwrap();
        assert!(p.iter().all(|v| v.is_nan()), "{values:?}");
        log_softmax_into(values, &mut p).unwrap();
        assert!(p.iter().all(|v| v.is_nan()), "{values:?}");
    }
    assert_eq!(logsumexp(&[f64::INFINITY, 1.0]).unwrap(), f64::INFINITY);
    assert_eq!(logsumexp::<f64, f64>(&[]).unwrap(), f64::NEG_INFINITY);
    // A NaN lane leaves the other lane exact.
    let x = [
        Lanes::<f64, 2>::from_fn(|k| [nan, 0.0][k]),
        Lanes::splat(0.0),
    ];
    let mut p = [Lanes::splat(0.0); 2];
    softmax_into(&x, &mut p).unwrap();
    assert!(p[0].lane(0).unwrap().is_nan());
    assert_eq!((p[0].lane(1), p[1].lane(1)), (Some(0.5), Some(0.5)));
    for result in [
        softmax_into::<f64, f64>(&[], &mut []),
        log_softmax_into::<f64, f64>(&[], &mut []),
    ] {
        let error = result.unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::DomainError { .. }),
            "{error}"
        );
    }
}

#[test]
fn empty_contractions_are_positive_zero() {
    let empty: [f64; 0] = [];
    assert_eq!(dot(&empty, &empty).unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        dot(&[-0.0_f64], &[1.0]).unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    let mut c = [7.0; 4];
    gemm_into(&empty, 2, 0, &empty, 2, &mut c).unwrap();
    assert!(c.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
    let mut y = [7.0; 2];
    gemv_into(&empty, 2, 0, &empty, &mut y).unwrap();
    assert!(y.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
    let g = grad(|x| Ok(dot(&x[..0], &x[..0])? + x[0]), &[1.0]).unwrap();
    assert_eq!(g.as_slice(), [1.0]);
    let empty_dual: [Dual<f64>; 0] = [];
    assert_eq!(
        dot(&empty_dual, &empty_dual).unwrap().value().to_bits(),
        0.0_f64.to_bits()
    );
}

/// `s ← |s| + p`: a kink at `s = 0`.
struct Folded;

impl LoopBody for Folded {
    fn step<R: Real>(&self, _: usize, state: &mut [R], p: &[R]) -> Result<(), SciOxideError> {
        state[0] = state[0].abs() + p[0];
        Ok(())
    }
}

#[test]
fn checkpointed_loop_bodies_run_on_values_while_recorded() {
    // The body's kinks are not counted, as documented; the derivatives follow the
    // midpoint rule of the body's own pullback.
    let schedule = CheckpointSchedule::binomial(3, 1).unwrap();
    let mut tape =
        Tape::<f64>::new(TapeOptions::new().with_kink_policy(scioxide_ad::KinkPolicy::Error));
    let mut g = [0.0; 2];
    let value = tape
        .gradient_into(
            |x| {
                let mut s = [x[0]];
                checkpointed_loop(&Arc::new(Folded), 3, &mut s, &x[1..], &schedule)?;
                Ok(s[0])
            },
            &[0.0, 1.0],
            &mut g,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(value, 3.0);
    assert_eq!(g, [0.0, 3.0]);
    assert_eq!(tape.report().kinks(), 0);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn implicit_size_overflow_precedes_callback() {
    struct Never;
    impl Residual for Never {
        fn residual<R: Real<Mask = bool>>(
            &self,
            _: &[R],
            _: &[R],
            _: &mut [R],
        ) -> Result<(), SciOxideError> {
            panic!("an overflowing derivative matrix must fail before the residual")
        }
    }
    let solution = vec![0.0; 65_536];
    let mut out = vec![Dual::new(19.0, 23.0); solution.len()];
    let error = implicit_solution_into(
        &Never,
        &solution,
        &[Dual::variable(0.0)],
        ResidualTolerance::new(1e-12, 0.0).unwrap(),
        &mut out,
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::SizeOverflow { .. }
    ));
    assert!(out.iter().all(|v| *v == Dual::new(19.0, 23.0)));
}

#[test]
fn complex_functions_differentiate_their_parts() {
    // d/dt exp(t + 2t·i) at t = 0.3 is (1 + 2i)·exp(0.3 + 0.6i).
    let z = Complex::new(Dual::variable(0.3), Dual::new(0.6, 2.0));
    let w = z.exp();
    let expected = Complex::new(1.0, 2.0) * Complex::new(0.3_f64, 0.6).exp();
    close(w.re.derivative(), expected.re, 1e-15);
    close(w.im.derivative(), expected.im, 1e-15);
    for value in [
        Complex::new(0.8, -0.3),
        Complex::new(-1.5, 0.4),
        Complex::new(-2.0, -0.7),
    ] {
        let lifted = Complex::new(value.re, value.im);
        let check = |a: Complex<f64>, b: Complex<f64>| {
            close(a.re, b.re, 1e-15);
            close(a.im, b.im, 1e-15);
        };
        check(ComplexElementary::inv(lifted), value.inv());
        check(ComplexElementary::ln(lifted), value.ln());
        check(ComplexElementary::sqrt(lifted), value.sqrt());
        check(ComplexElementary::sin(lifted), value.sin());
        check(ComplexElementary::cos(lifted), value.cos());
    }
}
