//! Affine values of the polynomial and rational families: per-component
//! equality, transactional preflight, workspace sizing and allocation-free reuse.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, Interval, SciOxideErrorKind, Tolerance,
};
use scioxide_interp::{
    BarycentricInterpolator, BarycentricRational, BulirschStoerInterpolator, Differentiable,
    ExtrapolationPolicy, FloaterHormannInterpolator, HermiteInterpolator, Interpolant,
    NevilleInterpolator, NewtonInterpolator, PadeApproximant, QueryWorkspace, RationalDomain,
};
use scioxide_linalg::RankPolicy;
use scioxide_poly::PolynomialApproximationPolicy;

#[test]
fn newton_insertion_matches_reconstruction_for_affine_values() {
    use scioxide_core::{AffineValue, AffineValueMut, RealScalar, SciOxideError};

    fn compare<S: RealScalar, V: AffineValue<Scale = S> + Sync>(
        values: [V; 3],
        rebuilt_values: Vec<V>,
    ) {
        let [first, second, incoming] = values;
        let source = NewtonInterpolator::new(vec![S::ZERO, S::ONE], vec![first, second]).unwrap();
        let inserted = source.insert_point(S::TWO, incoming).unwrap();
        let rebuilt =
            NewtonInterpolator::new(vec![S::ZERO, S::ONE, S::TWO], rebuilt_values).unwrap();
        assert_eq!(inserted.coefficients(), rebuilt.coefficients());
        for x in [S::ZERO, S::HALF, S::ONE, S::TWO] {
            let a = inserted.evaluate([x], ExtrapolationPolicy::Reject).unwrap();
            let b = rebuilt.evaluate([x], ExtrapolationPolicy::Reject).unwrap();
            for i in 0..a.dimension().unwrap() {
                let (a, b) = (a.component(i).unwrap(), b.component(i).unwrap());
                assert!((a - b).abs() <= S::EPSILON * S::TWO * b.abs().max(S::ONE));
            }
        }
        assert_eq!(source.nodes(), &[S::ZERO, S::ONE]);
    }

    // No Clone capability: insertion must consume the input template directly.
    struct Value(f64);
    impl AffineValue for Value {
        type Scale = f64;
        type Owned = Self;
        fn dimension(&self) -> Result<usize, SciOxideError> {
            Ok(1)
        }
        fn component(&self, index: usize) -> Result<f64, SciOxideError> {
            self.0.component(index)
        }
        fn try_to_owned(&self) -> Result<Self, SciOxideError> {
            Ok(Self(self.0))
        }
    }
    impl AffineValueMut for Value {
        fn for_each_component_mut(&mut self, visit: impl FnMut(&mut f64)) {
            self.0.for_each_component_mut(visit);
        }
    }
    compare(
        [Value(1.0), Value(2.0), Value(5.0)],
        vec![Value(1.0), Value(2.0), Value(5.0)],
    );
    macro_rules! precisions {
        ($s:ty) => {{
            let values = [1.0 as $s, 2.0, 5.0];
            compare(values, values.to_vec());
            let values = values.map(|v| Complex::new(v, 2.0 * v));
            compare(values, values.to_vec());
            let values = values.map(|v| [v.re, v.im]);
            compare(values, values.to_vec());
            compare(values.map(Vec::from), values.map(Vec::from).to_vec());
            compare([[] as [$s; 0]; 3], vec![[]; 3]);
        }};
    }
    precisions!(f32);
    precisions!(f64);
}

#[test]
fn newton_insertion_validates_before_publishing_a_new_model() {
    let model =
        NewtonInterpolator::new(vec![0.0_f64, 1.0], vec![vec![1.0, 2.0], vec![2.0, 4.0]]).unwrap();
    for (x, y, kind) in [
        (f64::NAN, vec![1.0, 2.0], SciOxideErrorKind::NonFinite),
        (f64::INFINITY, vec![1.0, 2.0], SciOxideErrorKind::NonFinite),
        (1.0, vec![1.0, 2.0], SciOxideErrorKind::InvalidInput),
        (-1.0, vec![1.0, 2.0], SciOxideErrorKind::InvalidInput),
        (2.0, vec![1.0], SciOxideErrorKind::DimensionMismatch),
        (2.0, vec![1.0, f64::NAN], SciOxideErrorKind::NonFinite),
    ] {
        assert_eq!(model.insert_point(x, y).unwrap_err().kind(), kind);
        assert_eq!(model.nodes(), &[0.0, 1.0]);
        assert_eq!(
            model.evaluate([0.5], ExtrapolationPolicy::Reject).unwrap(),
            vec![1.5, 3.0]
        );
    }
}

macro_rules! cases {
    ($module:ident, $scalar:ty) => {
        mod $module {
            use super::*;
            type S = $scalar;
            const CONTINUE: ExtrapolationPolicy<S> = ExtrapolationPolicy::MethodDefined;
            const REJECT: ExtrapolationPolicy<S> = ExtrapolationPolicy::Reject;
            const QUERIES: [S; 6] = [-1.0, -0.73, 0.1, 0.55, 1.3, 1.6];

            fn nodes() -> Vec<S> {
                vec![-1.0, -0.4, 0.1, 0.7, 1.3]
            }
            /// Three components with very different magnitudes.
            fn columns(f: [fn(S) -> S; 3]) -> [Vec<S>; 3] {
                f.map(|f| nodes().into_iter().map(f).collect())
            }
            fn polynomial_columns() -> [Vec<S>; 3] {
                columns([|x| x * x - x, |x| 1e3 * x * x * x, |x| x.exp()])
            }
            fn slope_columns() -> [Vec<S>; 3] {
                columns([|x| 2.0 * x - 1.0, |x| 3e3 * x * x, |x| x.exp()])
            }
            fn rational_columns() -> [Vec<S>; 3] {
                columns([|x| 1.0 / (2.0 + x), |x| (1.0 + x) / (3.0 + x), |_| 2.0])
            }
            fn vectors(columns: &[Vec<S>; 3]) -> Vec<Vec<S>> {
                (0..columns[0].len())
                    .map(|i| columns.iter().map(|column| column[i]).collect())
                    .collect()
            }
            fn complex_values(columns: &[Vec<S>; 3]) -> Vec<Complex<S>> {
                columns[0]
                    .iter()
                    .zip(&columns[1])
                    .map(|(&re, &im)| Complex::new(re, im))
                    .collect()
            }

            /// Vector and complex models equal scalar models component by component.
            fn equal_values<A, C, R>(vector: &A, complex: &C, scalar: &[R; 3])
            where
                A: Interpolant<S, 1, Value = Vec<S>>,
                C: Interpolant<S, 1, Value = Complex<S>>,
                R: Interpolant<S, 1, Value = S>,
            {
                for query in QUERIES {
                    let expected = scalar
                        .each_ref()
                        .map(|model| model.evaluate([query], CONTINUE).unwrap().to_bits());
                    let value = vector.evaluate([query], CONTINUE).unwrap();
                    assert_eq!(value.map_bits(), expected, "vector at {query}");
                    let value = complex.evaluate([query], CONTINUE).unwrap();
                    assert_eq!(
                        [value.re.to_bits(), value.im.to_bits()],
                        [expected[0], expected[1]],
                        "complex at {query}"
                    );
                }
            }

            fn equal_gradients<A, C, R>(vector: &A, complex: &C, scalar: &[R; 3])
            where
                A: Differentiable<S, 1, Value = Vec<S>>,
                C: Differentiable<S, 1, Value = Complex<S>>,
                R: Differentiable<S, 1, Value = S>,
            {
                for query in QUERIES {
                    for policy in [CONTINUE, ExtrapolationPolicy::Clamp] {
                        let expected = scalar
                            .each_ref()
                            .map(|model| model.gradient([query], policy).unwrap()[0].to_bits());
                        let [slope] = vector.gradient([query], policy).unwrap();
                        assert_eq!(slope.map_bits(), expected, "vector slope at {query}");
                        let [slope] = complex.gradient([query], policy).unwrap();
                        assert_eq!(
                            [slope.re.to_bits(), slope.im.to_bits()],
                            [expected[0], expected[1]]
                        );
                    }
                }
            }

            trait Bits {
                fn map_bits(&self) -> [<S as BitsOf>::Bits; 3];
            }
            impl Bits for Vec<S> {
                fn map_bits(&self) -> [<S as BitsOf>::Bits; 3] {
                    [self[0].to_bits(), self[1].to_bits(), self[2].to_bits()]
                }
            }

            /// Wrong output shapes, non-finite and rejected queries leave output
            /// unchanged; reserved queries allocate nothing.
            fn transactional<M>(model: &M)
            where
                M: Interpolant<S, 1, Value = Vec<S>, Workspace = QueryWorkspace<S>>,
            {
                let mut workspace = model.workspace().unwrap();
                let mut out = [7.0 as S; 3];
                let mut short = [7.0 as S; 2];
                let kind =
                    |result: Result<(), scioxide_core::SciOxideError>| result.unwrap_err().kind();
                assert_eq!(
                    kind(model.evaluate_into([0.3], REJECT, &mut short[..], &mut workspace)),
                    SciOxideErrorKind::DimensionMismatch
                );
                assert_eq!(
                    kind(model.evaluate_into([S::NAN], REJECT, &mut out[..], &mut workspace)),
                    SciOxideErrorKind::NonFinite
                );
                assert_eq!(
                    kind(model.evaluate_into([5.0], REJECT, &mut out[..], &mut workspace)),
                    SciOxideErrorKind::DomainError
                );
                assert_eq!((out, short), ([7.0; 3], [7.0; 2]));
                {
                    model
                        .evaluate_into([0.3], REJECT, &mut out[..], &mut workspace)
                        .unwrap();
                };
                assert_eq!(out.to_vec(), model.evaluate([0.3], REJECT).unwrap());
            }

            /// A workspace prepared for a three-component model cannot serve a wide one.
            fn undersized<M>(wide: &M, mut small: QueryWorkspace<S>)
            where
                M: Interpolant<S, 1, Workspace = QueryWorkspace<S>>,
            {
                let mut out = vec![7.0 as S; wide.value_dimension()];
                assert_eq!(
                    wide.workspace_capacity(&small),
                    scioxide_core::Capacity::NeedsGrowth
                );
                let error = wide
                    .evaluate_into([0.3], REJECT, &mut out[..], &mut small)
                    .unwrap_err();
                assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
                assert!(out.iter().all(|&value| value == 7.0));
            }

            fn wide_values() -> Vec<Vec<S>> {
                nodes()
                    .iter()
                    .map(|&x| (0_u8..20).map(|k| x + S::from(k)).collect())
                    .collect()
            }

            #[test]
            fn barycentric_components_match_scalar_interpolants() {
                let columns = polynomial_columns();
                let vector = BarycentricInterpolator::new(nodes(), vectors(&columns)).unwrap();
                let complex =
                    BarycentricInterpolator::new(nodes(), complex_values(&columns)).unwrap();
                let scalar = columns
                    .clone()
                    .map(|column| BarycentricInterpolator::new(nodes(), column).unwrap());
                equal_values(&vector, &complex, &scalar);
                equal_gradients(&vector, &complex, &scalar);
                transactional(&vector);
                let wide = BarycentricInterpolator::new(nodes(), wide_values()).unwrap();
                undersized(&wide, vector.workspace().unwrap());

                let (lower, upper) = (-1.0, 1.3);
                equal_values(
                    &BarycentricInterpolator::from_chebyshev_lobatto(
                        vectors(&columns),
                        lower,
                        upper,
                    )
                    .unwrap(),
                    &BarycentricInterpolator::from_chebyshev_lobatto(
                        complex_values(&columns),
                        lower,
                        upper,
                    )
                    .unwrap(),
                    &columns.clone().map(|column| {
                        BarycentricInterpolator::from_chebyshev_lobatto(column, lower, upper)
                            .unwrap()
                    }),
                );
                // The prepared antiderivative also integrates component by component.
                let prepared = vector.prepare_antiderivative().unwrap();
                let integral = prepared.integral(-0.5, 1.0, REJECT).unwrap();
                for (k, model) in scalar.iter().enumerate() {
                    let expected = model
                        .prepare_antiderivative()
                        .unwrap()
                        .integral(-0.5, 1.0, REJECT)
                        .unwrap();
                    assert_eq!(integral[k].to_bits(), expected.to_bits());
                }
            }

            #[test]
            fn newton_components_match_scalar_interpolants() {
                let columns = polynomial_columns();
                let vector = NewtonInterpolator::new(nodes(), vectors(&columns)).unwrap();
                let complex = NewtonInterpolator::new(nodes(), complex_values(&columns)).unwrap();
                let scalar =
                    columns.map(|column| NewtonInterpolator::new(nodes(), column).unwrap());
                equal_values(&vector, &complex, &scalar);
                equal_gradients(&vector, &complex, &scalar);
                transactional(&vector);
                let wide = NewtonInterpolator::new(nodes(), wide_values()).unwrap();
                undersized(&wide, vector.workspace().unwrap());
            }

            #[test]
            fn hermite_components_match_scalar_interpolants() {
                let columns = polynomial_columns();
                let slopes = slope_columns();
                let vector =
                    HermiteInterpolator::new(nodes(), vectors(&columns), vectors(&slopes)).unwrap();
                let complex = HermiteInterpolator::new(
                    nodes(),
                    complex_values(&columns),
                    complex_values(&slopes),
                )
                .unwrap();
                let scalar = [0, 1, 2].map(|k| {
                    HermiteInterpolator::new(nodes(), columns[k].clone(), slopes[k].clone())
                        .unwrap()
                });
                equal_values(&vector, &complex, &scalar);
                equal_gradients(&vector, &complex, &scalar);
                transactional(&vector);
            }

            #[test]
            fn neville_components_match_scalar_interpolants() {
                let columns = polynomial_columns();
                let vector = NevilleInterpolator::new(nodes(), vectors(&columns)).unwrap();
                let complex = NevilleInterpolator::new(nodes(), complex_values(&columns)).unwrap();
                let scalar =
                    columns.map(|column| NevilleInterpolator::new(nodes(), column).unwrap());
                equal_values(&vector, &complex, &scalar);
                transactional(&vector);
                let wide = NevilleInterpolator::new(nodes(), wide_values()).unwrap();
                undersized(&wide, vector.workspace().unwrap());
            }

            #[test]
            fn bulirsch_stoer_components_match_scalar_interpolants() {
                let columns = rational_columns();
                let vector = BulirschStoerInterpolator::new(nodes(), vectors(&columns)).unwrap();
                let complex =
                    BulirschStoerInterpolator::new(nodes(), complex_values(&columns)).unwrap();
                let scalar =
                    columns.map(|column| BulirschStoerInterpolator::new(nodes(), column).unwrap());
                equal_values(&vector, &complex, &scalar);
                transactional(&vector);
                let wide = BulirschStoerInterpolator::new(nodes(), wide_values()).unwrap();
                undersized(&wide, vector.workspace().unwrap());
            }

            #[test]
            fn floater_hormann_components_match_scalar_interpolants() {
                let columns = polynomial_columns();
                let vector =
                    FloaterHormannInterpolator::new(nodes(), vectors(&columns), 2).unwrap();
                let complex =
                    FloaterHormannInterpolator::new(nodes(), complex_values(&columns), 2).unwrap();
                let scalar = columns
                    .map(|column| FloaterHormannInterpolator::new(nodes(), column, 2).unwrap());
                equal_values(&vector, &complex, &scalar);
                transactional(&vector);
            }

            #[test]
            fn rational_plans_evaluate_real_and_complex_fields_without_allocation() {
                let domain = RationalDomain::Real(Interval::new(-1.0, 1.0).unwrap());
                // r(z) = 2 + z with denominator weights (1, -1).
                let real = BarycentricRational::<S>::new(
                    vec![-1.0, 1.0],
                    vec![1.0, -3.0],
                    vec![1.0, -1.0],
                    domain,
                )
                .unwrap();
                let one = Complex::new(1.0 as S, 0.0);
                let field = BarycentricRational::new(
                    vec![-one, one],
                    vec![Complex::new(1.0, 2.0), Complex::new(-3.0, -2.0)],
                    vec![one, -one],
                    domain,
                )
                .unwrap();
                let policy = PolynomialApproximationPolicy::new(
                    Tolerance::new(1e-5, 1e-5).unwrap(),
                    RankPolicy::DimensionScaledEpsilon,
                );
                let pade =
                    PadeApproximant::<S>::with_options(&[1.0, 1.0, 1.0], 1, 1, 0.0, domain, policy)
                        .unwrap();
                let (mut value, mut complex_value) = (7.0 as S, Complex::new(7.0 as S, 7.0));
                let (mut short, mut wide) = ([7.0 as S; 2], [7.0 as S; 3]);
                {
                    real.evaluate_into([0.5], REJECT, &mut value, &mut ())
                        .unwrap();
                    assert!((value - 2.5).abs() < 1e-6);
                    field
                        .evaluate_into([0.5], REJECT, &mut complex_value, &mut ())
                        .unwrap();
                    assert!((complex_value - Complex::new(2.5, 2.0)).norm() < 1e-5);
                    field
                        .evaluate_into([0.5, 0.0], REJECT, &mut short, &mut ())
                        .unwrap();
                    assert_eq!(short, [complex_value.re, complex_value.im]);
                    pade.evaluate_into([0.2], REJECT, &mut value, &mut ())
                        .unwrap();
                    assert!((value - 1.25).abs() < 1e-5);
                    // Rejections leave output unchanged.
                    assert!(
                        real.evaluate_into([0.5], REJECT, &mut wide[..], &mut ())
                            .is_err()
                    );
                    assert!(
                        real.evaluate_into([S::NAN], REJECT, &mut value, &mut ())
                            .is_err()
                    );
                    assert!(
                        field
                            .evaluate_into([0.0, 0.5], REJECT, &mut short, &mut ())
                            .is_err()
                    );
                    assert!(
                        pade.evaluate_into([2.0], REJECT, &mut value, &mut ())
                            .is_err()
                    );
                };
                assert!((value - 1.25).abs() < 1e-5);
                assert_eq!(wide, [7.0; 3]);
            }
        }
    };
}

/// Exact bit patterns of the portable widths.
trait BitsOf {
    type Bits: PartialEq + core::fmt::Debug;
}
impl BitsOf for f32 {
    type Bits = u32;
}
impl BitsOf for f64 {
    type Bits = u64;
}

cases!(binary32, f32);
cases!(binary64, f64);

#[test]
fn newton_vector_batches_are_strided_transactional_and_executor_independent() {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_interp::{BatchEvaluator, BatchRequest};
    let model = NewtonInterpolator::new(
        vec![0., 1., 2.],
        vec![vec![0., 1.], vec![1., 3.], vec![4., 5.]],
    )
    .unwrap();
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let mut workspace = plan.workspace(BatchRequest::values(2)).unwrap();
    let query_data = [0.5, 88., 1.5];
    let queries = NdView::new(&query_data, StridedLayout::new(&[2, 1], &[2, 1]).unwrap()).unwrap();
    let mut data = [99.; 7];
    {
        let mut out =
            NdViewMut::new(&mut data, StridedLayout::new(&[2, 2], &[1, 4]).unwrap()).unwrap();
        plan.evaluate_into(
            &queries,
            ExtrapolationPolicy::Reject,
            &mut out,
            None,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    }
    assert_eq!(data, [0.25, 2.25, 99., 99., 2., 4., 99.]);
    // A late failing query commits nothing.
    let before = data;
    let late = NdView::new(&[0.5, 3.], StridedLayout::new(&[2, 1], &[1, 1]).unwrap()).unwrap();
    let mut out = NdViewMut::new(&mut data, StridedLayout::new(&[2, 2], &[1, 4]).unwrap()).unwrap();
    assert!(
        plan.evaluate_into(
            &late,
            ExtrapolationPolicy::Reject,
            &mut out,
            None,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert_eq!(data, before);

    #[cfg(feature = "std")]
    {
        use core::num::NonZeroUsize;
        use scioxide_core::ExecutionPolicy;
        let workers = NonZeroUsize::new(3).unwrap();
        let pool = scioxide_core::ThreadExecutor::new(workers);
        let parallel = BatchEvaluator::new(&model, ExecutionPolicy::Parallel(workers));
        let points = [0.25, 0.5, 1.5];
        let q = NdView::new(&points, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
        let mut results = [[0.; 6]; 2];
        for (data, context) in results
            .iter_mut()
            .zip([ExecutionContext::sequential(), ExecutionContext::new(&pool)])
        {
            let mut w = parallel.workspace(BatchRequest::values(3)).unwrap();
            let mut out =
                NdViewMut::new(data, StridedLayout::new(&[3, 2], &[2, 1]).unwrap()).unwrap();
            parallel
                .evaluate_into(
                    &q,
                    ExtrapolationPolicy::Reject,
                    &mut out,
                    None,
                    &mut w,
                    context,
                )
                .unwrap();
        }
        assert_eq!(results[0], results[1]);
    }
}

/// Distinct nodes whose normalized coordinates round together cannot support
/// an antiderivative; this is a numerical failure, not invalid input.
#[test]
fn antiderivative_rejects_collapsed_normalized_nodes_as_instability() {
    let interpolator =
        BarycentricInterpolator::new(vec![0.0_f64, 1e-17, 1.0], vec![0.0, 1.0, 2.0]).unwrap();
    let error = interpolator.prepare_antiderivative().unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::LossOfSignificance,
                ..
            }
        ),
        "{error:?}"
    );
}

/// The owned two-output queries return the bits of their `_into` forms, and
/// the same error for a point outside the domain.
#[test]
fn owned_two_output_queries_equal_their_into_forms() {
    use scioxide_core::{AffineValue, RealScalar, SciOxideError};

    fn bits<V: AffineValue>(value: &V) -> Vec<u64> {
        (0..value.dimension().unwrap())
            .map(|i| {
                let component = value.component(i).unwrap();
                component.try_to_f64_exact().unwrap().to_bits()
            })
            .collect()
    }

    fn same<V: AffineValue>(
        owned: Result<(V, V), SciOxideError>,
        into: Result<(V, V), SciOxideError>,
    ) {
        let pair_bits = |pair: (V, V)| (bits(&pair.0), bits(&pair.1));
        assert_eq!(owned.map(pair_bits), into.map(pair_bits));
    }

    fn compare<S: RealScalar, V: AffineValue<Scale = S> + Clone + Sync>(y: Vec<V>) {
        let x: Vec<S> = [-1.0, -0.4, 0.1, 0.7, 1.3]
            .map(|v| S::try_from_f64_lossy(v).unwrap())
            .to_vec();
        let neville = NevilleInterpolator::new(x.clone(), y.clone()).unwrap();
        let barycentric = BarycentricInterpolator::new(x, y).unwrap();
        let policies = [
            ExtrapolationPolicy::Reject,
            ExtrapolationPolicy::Clamp,
            ExtrapolationPolicy::MethodDefined,
        ];
        for t in [-1.0, -0.73, 0.1, 0.55, 1.6] {
            let t = S::try_from_f64_lossy(t).unwrap();
            for policy in policies {
                let into = || {
                    let mut workspace = neville.workspace()?;
                    let (mut value, mut error) = (neville.zero_value()?, neville.zero_value()?);
                    neville.evaluate_with_error_into(
                        [t],
                        policy,
                        &mut value,
                        &mut error,
                        &mut workspace,
                    )?;
                    Ok((value, error))
                };
                same(neville.evaluate_with_error([t], policy), into());
                let into = || {
                    let mut workspace = barycentric.workspace()?;
                    let mut value = barycentric.zero_value()?;
                    let mut slope = barycentric.zero_value()?;
                    barycentric.evaluate_with_derivative_into(
                        [t],
                        policy,
                        &mut value,
                        &mut slope,
                        &mut workspace,
                    )?;
                    Ok((value, slope))
                };
                same(barycentric.evaluate_with_derivative([t], policy), into());
            }
        }
    }

    fn families<S: RealScalar>() {
        let real: Vec<S> = [0.5, -1.25, 2.0, 0.75, -0.5]
            .map(|v| S::try_from_f64_lossy(v).unwrap())
            .to_vec();
        let complex = real
            .iter()
            .map(|&v| Complex::new(v, S::ONE - v))
            .collect::<Vec<_>>();
        let arrays = real.iter().map(|&v| [v, v * v, -v]).collect::<Vec<_>>();
        let vectors = arrays.iter().map(|a| a.to_vec()).collect::<Vec<_>>();
        compare(real);
        compare(complex);
        compare(arrays);
        compare(vectors);
    }

    families::<f32>();
    families::<f64>();
}
