//! Independent manufactured surfaces for the nonuniform bicubic contract.
#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

mod support;

use support::{assert_close, assert_error};

use scioxide_interp::{BicubicInterpolator, ExtrapolationPolicy, Interpolant};
use scioxide_interp::{Differentiable, TwiceDifferentiable};

macro_rules! scalar_cases {
    ($module:ident, $scalar:ty, $tol:expr) => {
        mod $module {
            use super::*;
            type S = $scalar;
            const TOL: S = $tol;
            const HALF_TOL: f64 = ($tol as f64) * 0.5;
            const REJECT: ExtrapolationPolicy<S> = ExtrapolationPolicy::Reject;

            fn grid(x: &[S], y: &[S], f: impl Fn(S, S) -> S) -> BicubicInterpolator<S> {
                let z = y
                    .iter()
                    .flat_map(|&b| {
                        x.iter().map({
                            let f = &f;
                            move |&a| f(a, b)
                        })
                    })
                    .collect();
                BicubicInterpolator::new(x.to_vec(), y.to_vec(), z).unwrap()
            }

            #[test]
            fn tensor_quadratics_at_nodes_boundaries_and_interiors() {
                let x = [-2.0, -1.5, 0.0, 0.25, 2.0];
                let y = [-3.0, -0.5, 0.0, 1.5];
                for (px, py) in [(2, 0), (0, 2), (1, 1), (2, 1), (1, 2), (2, 2)] {
                    let f = |a: S, b: S| a.powi(px) * b.powi(py);
                    let interp = grid(&x, &y, f);
                    for a in [-2.0, -1.75, -1.5, -0.75, 0.0, 0.125, 0.25, 1.0, 2.0] {
                        for b in [-3.0, -2.0, -0.5, -0.25, 0.0, 0.75, 1.5] {
                            assert_close(
                                interp.evaluate([a, b], REJECT).unwrap(),
                                f(a, b),
                                HALF_TOL,
                                HALF_TOL,
                            );
                            let [dx, dy] = interp.gradient([a, b], REJECT).unwrap();
                            let h = interp.hessian([a, b], REJECT).unwrap();
                            assert_close(
                                h[0][0],
                                if px < 2 {
                                    0.0
                                } else {
                                    (px * (px - 1)) as S * a.powi(px - 2) * b.powi(py)
                                },
                                HALF_TOL,
                                HALF_TOL,
                            );
                            assert_close(
                                h[1][1],
                                if py < 2 {
                                    0.0
                                } else {
                                    (py * (py - 1)) as S * a.powi(px) * b.powi(py - 2)
                                },
                                HALF_TOL,
                                HALF_TOL,
                            );
                            let mixed = if px == 0 || py == 0 {
                                0.0
                            } else {
                                (px * py) as S * a.powi(px - 1) * b.powi(py - 1)
                            };
                            assert_close(h[0][1], mixed, HALF_TOL, (TOL as f64) * 0.5);
                            assert_close(h[1][0], mixed, HALF_TOL, (TOL as f64) * 0.5);

                            assert_close(
                                dx,
                                if px == 0 {
                                    0.0
                                } else {
                                    px as S * a.powi(px - 1) * b.powi(py)
                                },
                                HALF_TOL,
                                HALF_TOL,
                            );
                            assert_close(
                                dy,
                                if py == 0 {
                                    0.0
                                } else {
                                    py as S * a.powi(px) * b.powi(py - 1)
                                },
                                HALF_TOL,
                                HALF_TOL,
                            );
                        }
                    }
                }
            }

            #[test]
            fn two_node_axes_use_linear_fallback() {
                for x in [vec![-2.0, 2.0], vec![-2.0, -0.5, 0.0, 2.0]] {
                    for y in [vec![-1.0, 3.0], vec![-1.0, 0.0, 0.25, 3.0]] {
                        let interp = grid(&x, &y, |a, b| 2.0 + a - 3.0 * b + 2.0 * a * b);
                        assert_close(
                            interp.evaluate([0.5, 0.5], REJECT).unwrap(),
                            2.0 + 0.5 - 1.5 + 0.5,
                            HALF_TOL,
                            HALF_TOL,
                        );
                        let [dx, dy] = interp.gradient([0.5, 0.5], REJECT).unwrap();
                        assert_close(dx, 2.0, HALF_TOL, (TOL as f64) * 0.5);
                        assert_close(dy, -2.0, HALF_TOL, (TOL as f64) * 0.5);
                    }
                }
            }

            #[test]
            fn coordinate_scale_does_not_limit_preparation() {
                // Each axis is tiny enough that a physical mixed derivative
                // overflows, although all normalized constraints are ordinary.
                let scale = S::MIN_POSITIVE * 16.0;
                let x = [0.0, scale, 3.0 * scale];
                let interp = grid(&x, &x, |a, b| (a / scale) * (b / scale));
                assert_close(
                    interp.evaluate([0.5 * scale, 2.0 * scale], REJECT).unwrap(),
                    1.0,
                    HALF_TOL,
                    HALF_TOL,
                );
                let [dx, dy] = interp.gradient([0.5 * scale, 2.0 * scale], REJECT).unwrap();
                assert_close(dx * scale, 2.0, HALF_TOL, (TOL as f64) * 0.5);
                assert_close(dy * scale, 0.5, HALF_TOL, (TOL as f64) * 0.5);
                let huge = S::MAX / 4.0;
                let x = [-3.0 * huge, -huge, 0.0, huge, 3.0 * huge];
                let interp = grid(&x, &x, |a, b| (a / huge) * (b / huge));
                assert_close(
                    interp.evaluate([0.5 * huge, 0.5 * huge], REJECT).unwrap(),
                    0.25,
                    HALF_TOL,
                    HALF_TOL,
                );
                let [dx, dy] = interp.gradient([0.5 * huge, 0.5 * huge], REJECT).unwrap();
                assert_close(dx * huge, 0.5, HALF_TOL, (TOL as f64) * 0.5);
                assert_close(dy * huge, 0.5, HALF_TOL, (TOL as f64) * 0.5);
            }

            #[test]
            fn transpose_and_independent_cross_derivatives() {
                let x = [-2.0, -1.5, 0.0, 0.25, 2.0];
                let y = [-3.0, -0.5, 0.0, 1.5];
                let f = |a: S, b: S| a.sin() * b.cos() + a * b * b;
                let interp = grid(&x, &y, f);
                let transposed = grid(&y, &x, |b, a| f(a, b));
                for (a, b) in [(-1.75, -2.0), (-0.75, -0.25), (0.125, 0.75), (1.0, 0.5)] {
                    assert_close(
                        interp.evaluate([a, b], REJECT).unwrap(),
                        transposed.evaluate([b, a], REJECT).unwrap(),
                        HALF_TOL,
                        HALF_TOL,
                    );
                    let g = interp.gradient([a, b], REJECT).unwrap();
                    let gt = transposed.gradient([b, a], REJECT).unwrap();
                    assert_close(g[0], gt[1], HALF_TOL, (TOL as f64) * 0.5);
                    assert_close(g[1], gt[0], HALF_TOL, (TOL as f64) * 0.5);
                    let h = interp.hessian([a, b], REJECT).unwrap();
                    let ht = transposed.hessian([b, a], REJECT).unwrap();
                    assert_close(h[0][0], ht[1][1], HALF_TOL, (TOL as f64) * 0.5);
                    assert_close(h[1][1], ht[0][0], HALF_TOL, (TOL as f64) * 0.5);
                    assert_close(h[0][1], ht[0][1], HALF_TOL, (TOL as f64) * 0.5);
                    // Central differences of both gradient components are independent
                    // of the Hessian evaluation. All probes stay inside their cell.
                    let step = S::EPSILON.cbrt() * 0.25;
                    let dxy = (interp.gradient([a, b + step], REJECT).unwrap()[0]
                        - interp.gradient([a, b - step], REJECT).unwrap()[0])
                        / (2.0 * step);
                    let dyx = (interp.gradient([a + step, b], REJECT).unwrap()[1]
                        - interp.gradient([a - step, b], REJECT).unwrap()[1])
                        / (2.0 * step);
                    let tolerance = 8.0 * S::EPSILON.powf(2.0 / 3.0) * h[0][1].abs().max(1.0);
                    assert!((dxy - h[0][1]).abs() <= tolerance);
                    assert!((dyx - h[0][1]).abs() <= tolerance);
                }
            }

            #[test]
            fn cubic_refinement_and_cell_derivative_consistency() {
                // Quadratic derivative estimates need not reproduce cubic data.
                // On smoothly refined nonuniform grids the value error is O(h³).
                let mut previous = S::INFINITY;
                for n in [8, 16, 32] {
                    let x = (0..=n)
                        .map(|i| (i as S / n as S).powi(2))
                        .collect::<Vec<_>>();
                    let interp = grid(&x, &x, |a, b| a * a * a + 2.0 * b * b * b + a * a * b);
                    let mut error: S = 0.0;
                    for i in 0..n {
                        let a = (x[i] + x[i + 1]) * 0.5;
                        let b = a;
                        error = error.max(
                            (interp.evaluate([a, b], REJECT).unwrap()
                                - (a * a * a + 2.0 * b * b * b + a * a * b))
                                .abs(),
                        );
                    }
                    assert!(
                        error < previous * 0.3,
                        "n={n}, error={error:e}, previous={previous:e}"
                    );
                    previous = error;
                }
            }

            #[test]
            fn hessian_extreme_width_products_and_subnormal_axes() {
                // Width products can underflow/overflow while a second derivative
                // remains representable. Opposing axis scales also cancel exactly.
                for scale in [S::MIN_POSITIVE.sqrt() * 0.25, S::MAX.sqrt() * 2.0] {
                    let amplitude = if scale < 1.0 { S::MIN_POSITIVE } else { 1.0 };
                    let nodes = [0.0, scale, 2.0 * scale];
                    let interp = grid(&nodes, &nodes, |a, b| amplitude * (a / scale) * (b / scale));
                    let h = interp.hessian([0.5 * scale, 0.5 * scale], REJECT).unwrap();
                    let expected = (amplitude / scale) / scale;
                    assert!((h[0][1] / expected - 1.0).abs() <= TOL);
                    assert_close(h[0][0], 0.0, HALF_TOL, (TOL as f64) * 0.5);
                    assert_close(h[1][1], 0.0, HALF_TOL, (TOL as f64) * 0.5);
                }
                let large = S::MAX / 4.0;
                let small = 1.0 / large;
                let interp = grid(
                    &[0.0, large, 2.0 * large],
                    &[0.0, small, 2.0 * small],
                    |a, b| (a / large) * (b / small),
                );
                assert_close(
                    interp.hessian([0.5 * large, 0.5 * small], REJECT).unwrap()[0][1],
                    1.0,
                    HALF_TOL,
                    HALF_TOL,
                );
                let tiny = S::from_bits(4);
                let interp = grid(&[0.0, tiny, 2.0 * tiny], &[0.0, 1.0, 2.0], |a, b| a * b);
                assert_close(
                    interp.evaluate([tiny, 0.5], REJECT).unwrap() / tiny,
                    0.5,
                    HALF_TOL,
                    HALF_TOL,
                );
                let h = interp.hessian([tiny, 0.5], REJECT).unwrap();
                assert_close(h[0][1], 1.0, HALF_TOL, (TOL as f64) * 0.5);
            }

            #[test]
            fn translated_coordinates_and_one_sided_hessian_at_knots() {
                let shift = 1024.0;
                let scale = 0.125;
                let x = [
                    shift - 2.0 * scale,
                    shift,
                    shift + scale,
                    shift + 3.0 * scale,
                ];
                let interp = grid(&x, &x, |a, b| {
                    ((a - shift) / scale).powi(2) + ((b - shift) / scale).powi(2)
                });
                let q = shift + 0.5 * scale;
                assert_close(
                    interp.evaluate([q, q], REJECT).unwrap(),
                    0.5,
                    HALF_TOL,
                    HALF_TOL,
                );
                let g = interp.gradient([q, q], REJECT).unwrap();
                assert_close(g[0] * scale, 1.0, HALF_TOL, (TOL as f64) * 0.5);
                assert_close(g[1] * scale, 1.0, HALF_TOL, (TOL as f64) * 0.5);
                assert_close(
                    interp.hessian([q, q], REJECT).unwrap()[0][0] * scale * scale,
                    2.0,
                    HALF_TOL,
                    HALF_TOL,
                );
                let cubic = grid(&[0.0, 1.0, 3.0, 4.0], &[0.0, 1.0, 2.0], |a, b| {
                    a * a * a + b
                });
                let step = 2.0 * S::EPSILON;
                let g = cubic.gradient([1.0, 0.5], REJECT).unwrap();
                for a in [1.0 - step, 1.0 + step] {
                    assert_close(
                        cubic.evaluate([a, 0.5], REJECT).unwrap(),
                        1.5,
                        HALF_TOL,
                        HALF_TOL,
                    );
                    assert_close(
                        cubic.gradient([a, 0.5], REJECT).unwrap()[0],
                        g[0],
                        HALF_TOL,
                        HALF_TOL,
                    );
                }
                assert_close(
                    cubic.hessian([1.0 - step, 0.5], REJECT).unwrap()[0][0],
                    8.0,
                    HALF_TOL,
                    HALF_TOL,
                );
                assert_close(
                    cubic.hessian([1.0, 0.5], REJECT).unwrap()[0][0],
                    0.0,
                    HALF_TOL,
                    HALF_TOL,
                );
                assert_close(
                    cubic.hessian([4.0, 0.5], REJECT).unwrap()[0][0],
                    16.0,
                    HALF_TOL,
                    HALF_TOL,
                );
                let affine = grid(&[0.0, 1.0, 3.0], &[0.0, 1.0, 2.0], |a, b| a + 2.0 * b);
                let continued = affine
                    .hessian([-2.0, 4.0], ExtrapolationPolicy::MethodDefined)
                    .unwrap();
                for entry in continued.into_iter().flatten() {
                    assert_close(entry, 0.0, HALF_TOL, (TOL as f64) * 0.5);
                }
            }

            #[test]
            fn unrepresentable_spacing_ratios_keep_finite_constraints() {
                let x = [-S::MAX / 4.0, 0.0, S::MIN_POSITIVE];
                let y = [0.0, 1.0, 2.0];
                let constant = grid(&x, &y, |_, _| 2.0);
                assert_close(
                    constant.evaluate([-1.0, 0.5], REJECT).unwrap(),
                    2.0,
                    HALF_TOL,
                    HALF_TOL,
                );
                assert_close(
                    constant.hessian([-1.0, 0.5], REJECT).unwrap()[0][0],
                    0.0,
                    HALF_TOL,
                    HALF_TOL,
                );
                // Unit gradient despite a width ratio beyond the scalar range.
                let affine = grid(&x, &y, |a, _| a);
                assert_close(
                    affine.gradient([0.0, 0.5], REJECT).unwrap()[0],
                    1.0,
                    HALF_TOL,
                    HALF_TOL,
                );
            }

            #[test]
            fn finite_input_and_domain_errors() {
                let interp = grid(&[0.0, 1.0, 3.0], &[0.0, 2.0, 3.0], |a, b| a + b);
                assert_error(
                    &interp.hessian([-1.0, 0.0], REJECT),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
                assert_error(
                    &interp.hessian([S::INFINITY, 0.0], REJECT),
                    scioxide_core::SciOxideErrorKind::NonFinite,
                );
                let tiny = S::MIN_POSITIVE;
                let extreme = grid(
                    &[0.0, tiny, 2.0 * tiny],
                    &[0.0, tiny, 2.0 * tiny],
                    |a, b| (a / tiny) * (b / tiny),
                );
                assert_error(
                    &extreme.hessian([tiny, tiny], REJECT),
                    scioxide_core::SciOxideErrorKind::NonFinite,
                );
                assert_error(
                    &interp.gradient([-1.0, 0.0], REJECT),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
                assert_error(
                    &interp.evaluate([S::NAN, 0.0], REJECT),
                    scioxide_core::SciOxideErrorKind::NonFinite,
                );
            }
        }
    };
}
scalar_cases!(f32_cases, f32, 3e-5);
scalar_cases!(f64_cases, f64, 3e-12);
