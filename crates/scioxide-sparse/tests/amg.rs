//! Independent transfer, Galerkin, manufactured residual and failure tests.
#![allow(clippy::unwrap_used)]
mod amg_support;
use amg_support::*;
use scioxide_core::{
    BlockVectorView, Complex, ExecutionContext, LinearOperator, NumericScalar, Preconditioner,
};
use scioxide_sparse::*;
fn domains<S: NumericScalar>() {
    let a = grid::<S>(6, 2, 1.);
    let h = hierarchy(&a);
    let again = hierarchy(&a);
    assert!(h.diagnostics().levels() >= 2);
    for level in 0..h.diagnostics().levels() {
        let a = h.matrix(level).unwrap();
        assert_eq!(a, again.matrix(level).unwrap());
        if let Some(t) = h.tentative(level) {
            let bc = h.modes(level + 1).unwrap();
            let b = h.modes(level).unwrap();
            let mut reconstructed = vec![S::ZERO; t.rows()];
            t.apply_into(bc, &mut reconstructed, &mut vec![S::ZERO; t.rows()])
                .unwrap();
            close(&reconstructed, b, 128.);
            let p = h.prolongation(level).unwrap();
            let ac = h.matrix(level + 1).unwrap();
            // Independently sum P^H A P, without the production sparse-product kernel.
            let mut dense_p = vec![S::ZERO; p.rows() * p.cols()];
            for (i, j, v) in p.iter() {
                dense_p[i * p.cols() + j] = v;
            }
            let mut dense_ac = vec![S::ZERO; ac.rows() * ac.cols()];
            for (i, j, v) in ac.iter() {
                dense_ac[i * ac.cols() + j] = v;
            }
            for i in 0..ac.rows() {
                for j in 0..ac.cols() {
                    let expected = a.iter().fold(S::ZERO, |s, (r, c, v)| {
                        s + dense_p[r * p.cols() + i].conj() * v * dense_p[c * p.cols() + j]
                    });
                    close(&[dense_ac[i * ac.cols() + j]], &[expected], 1024.);
                }
            }
        }
    }
    for kind in [AmgCycleKind::V, AmgCycleKind::W] {
        let p = AmgPreconditioner::build(
            hierarchy(&a),
            AmgCycleOptions::new().with_kind(kind),
            coarse_limits(),
            SparseSolveOptions::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        let n = a.rows();
        let exact: Vec<S> = (0..n).map(|i| scalar(1. + (i % 7) as f64 / 7.)).collect();
        let mut rhs = vec![S::ZERO; n];
        let mut aw = vec![S::ZERO; n];
        a.apply_into(&exact, &mut rhs, &mut aw).unwrap();
        let mut x = vec![S::ZERO; n];
        let mut residual = rhs.clone();
        let mut correction = x.clone();
        let mut work = vec![S::ZERO; p.precondition_workspace_len().unwrap()];
        for _ in 0..45 {
            p.precondition_into(&residual, &mut correction, &mut work)
                .unwrap();
            for (x, &d) in x.iter_mut().zip(&correction) {
                *x = *x + d;
            }
            a.apply_into(&x, &mut residual, &mut aw).unwrap();
            for (r, &b) in residual.iter_mut().zip(&rhs) {
                *r = b - *r;
            }
        }
        close(&x, &exact, 4096.);
    }
}
#[test]
fn all_scalar_domains() {
    domains::<f32>();
    domains::<f64>();
    domains::<Complex<f32>>();
    domains::<Complex<f64>>();
}
#[test]
fn complex_phases_and_multiple_modes() {
    let mut a = grid::<Complex<f64>>(24, 1, 1.);
    // Unitary diagonal similarity creates genuinely complex Hermitian off-diagonals.
    let phase = |i: usize| {
        if i.is_multiple_of(2) {
            Complex::new(1., 0.)
        } else {
            Complex::new(0., 1.)
        }
    };
    let cols = a.col_indices().to_vec();
    for i in 0..a.rows() {
        for (pos, &col) in cols
            .iter()
            .enumerate()
            .take(a.row_ptr()[i + 1])
            .skip(a.row_ptr()[i])
        {
            a.set_value(pos, phase(i) * a.values()[pos] * phase(col).conj())
                .unwrap();
        }
    }
    let b: Vec<_> = (0..24).map(phase).collect();
    let h = AmgHierarchy::build(
        a.view(),
        BlockVectorView::new(&b, a.rows(), 1).unwrap(),
        AmgSetupOptions::new().with_coarse_size(2),
        limits(),
        SparseSolveOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let t = h.tentative(0).unwrap();
    let mut out = vec![Complex::new(0., 0.); 24];
    t.apply_into(
        h.modes(1).unwrap(),
        &mut out,
        &mut vec![Complex::new(0., 0.); 24],
    )
    .unwrap();
    close(&out, &b, 64.);
    let a = grid::<f64>(30, 1, 1.);
    let modes: Vec<f64> = (0..60)
        .map(|i| if i < 30 { 1. } else { (i - 30) as f64 / 30. })
        .collect();
    let h = AmgHierarchy::build(
        a.view(),
        BlockVectorView::new(&modes, a.rows(), 2).unwrap(),
        AmgSetupOptions::new().with_coarse_size(4),
        limits(),
        SparseSolveOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    for level in 0..h.diagnostics().levels() - 1 {
        let t = h.tentative(level).unwrap();
        for k in 0..2 {
            let mut out = vec![0.; t.rows()];
            t.apply_into(
                &h.modes(level + 1).unwrap()[k * t.cols()..(k + 1) * t.cols()],
                &mut out,
                &mut vec![0.; t.rows()],
            )
            .unwrap();
            close(
                &out,
                &h.modes(level).unwrap()[k * t.rows()..(k + 1) * t.rows()],
                1024.,
            );
        }
    }
}
#[test]
fn exact_nullspace_and_anisotropy_limits() {
    let mut a = grid::<f64>(32, 1, 1.);
    a.set_value(0, 1.).unwrap();
    let last = a.nnz() - 1;
    a.set_value(last, 1.).unwrap();
    let h = hierarchy(&a); // Setup permits PSD; a default Cholesky solve is not claimed.
    for level in 0..h.diagnostics().levels() - 1 {
        let p = h.prolongation(level).unwrap();
        let mut out = vec![0.; p.rows()];
        p.apply_into(
            h.modes(level + 1).unwrap(),
            &mut out,
            &mut vec![0.; p.rows()],
        )
        .unwrap();
        close(&out, h.modes(level).unwrap(), 256.);
    }
    let a = grid::<f64>(12, 2, 0.0001);
    let h = hierarchy(&a);
    assert!(h.diagnostics().levels() > 1);
    assert!(h.diagnostics().operator_entries() < 10 * a.nnz());
}

#[test]
fn poisson_coarsening_and_complexity() {
    // Finite-grid acceptance bounds at the default strength and coarse size four.
    for (side, dimensions, minimum_ratio, maximum_complexity) in
        [(96, 1, 3., 2.), (16, 2, 5., 3.), (8, 3, 6., 4.)]
    {
        let a = grid::<f64>(side, dimensions, 1.);
        let h = hierarchy(&a);
        let d = h.diagnostics();
        let ratio = d.coarsening_ratio().unwrap();
        let complexity = d.operator_complexity().unwrap();
        eprintln!("{dimensions}D side={side}: {d:?}");
        assert!(ratio >= minimum_ratio, "{dimensions}D ratio {ratio}");
        assert!(
            complexity <= maximum_complexity,
            "{dimensions}D complexity {complexity}"
        );
        close(
            &[ratio],
            &[a.rows() as f64 / h.matrix(1).unwrap().rows() as f64],
            4.,
        );
        let entries: usize = (0..d.levels()).map(|l| h.matrix(l).unwrap().nnz()).sum();
        assert_eq!(d.operator_entries(), entries);
        close(&[complexity], &[entries as f64 / a.nnz() as f64], 4.);
    }
    // Endpoint aggregates make the finite-chain ratio n / ceil(n / 3).
    for n in [95_usize, 96, 97] {
        let a = grid::<f64>(n, 1, 1.);
        let h = hierarchy(&a);
        assert_eq!(h.matrix(1).unwrap().rows(), n.div_ceil(3));
        assert_eq!(h.aggregates(0), hierarchy(&a).aggregates(0));
    }
}

#[test]
fn frozen_attachments_and_isolated_nodes() {
    // Seeds 0 and 2 form {0,1} and {2,6}. Node 3 attaches through 1;
    // node 3 chooses neighbour 1 over 6, and node 4 must attach through 6
    // rather than the newly attached node 3.
    let mut builder = SparseBuilder::new(7, 7);
    for i in 0..7 {
        builder.push(i, i, 4.).unwrap();
    }
    for (i, j) in [(0, 1), (2, 6), (1, 3), (3, 4), (4, 6), (3, 6)] {
        builder.push(i, j, -1.).unwrap();
        builder.push(j, i, -1.).unwrap();
    }
    // Explicit zeros must not turn isolated node 5 into a strong neighbour.
    builder.push(0, 5, 0.).unwrap();
    builder.push(5, 0, 0.).unwrap();
    let a = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let h = hierarchy(&a);
    assert_eq!(h.aggregates(0).unwrap(), &[0, 0, 1, 0, 1, 2, 1]);
    let t = h.tentative(0).unwrap();
    let mut reconstructed = vec![0.; 7];
    t.apply_into(h.modes(1).unwrap(), &mut reconstructed, &mut [0.; 7])
        .unwrap();
    close(&reconstructed, &[1.; 7], 32.);
}

#[test]
fn strength_boundary_and_uncoarsened_diagnostics() {
    let mut a = grid::<f64>(6, 1, 1.);
    for i in 0..a.rows() {
        let diagonal = a.row_ptr()[i]
            + a.col_indices()[a.row_ptr()[i]..a.row_ptr()[i + 1]]
                .binary_search(&i)
                .unwrap();
        a.set_value(diagonal, 4.).unwrap();
    }
    let build = |strength| {
        AmgHierarchy::build(
            a.view(),
            BlockVectorView::new(&[1.; 6], a.rows(), 1).unwrap(),
            AmgSetupOptions::new()
                .with_coarse_size(1)
                .with_strength(strength),
            limits(),
            SparseSolveOptions::new(),
            ExecutionContext::sequential(),
        )
        .unwrap()
    };
    assert_eq!(build(0.25).aggregates(0).unwrap(), &[0, 0, 1, 1, 1, 1]);
    assert_eq!(build(0.).aggregates(0), build(0.25).aggregates(0));
    let h = build(0.25 + f64::EPSILON);
    assert_eq!(
        h.diagnostics().termination(),
        AmgSetupTermination::Stagnation
    );
    assert_eq!(h.diagnostics().coarsening_ratio(), None);
    assert_eq!(h.diagnostics().operator_complexity(), Some(1.));
    for n in [0, 1] {
        let a = grid::<f64>(n, 1, 1.);
        let d = hierarchy(&a).diagnostics();
        assert_eq!(d.coarsening_ratio(), None);
        assert_eq!(
            d.operator_complexity(),
            if n == 0 { None } else { Some(1.) }
        );
    }
}
#[test]
fn validation_resources_and_transactions() {
    let a = grid::<f64>(24, 1, 1.);
    let build = |b: &[f64], k, l| {
        BlockVectorView::new(b, 24, k).and_then(|modes| {
            AmgHierarchy::build(
                a.view(),
                modes,
                AmgSetupOptions::new(),
                l,
                SparseSolveOptions::new(),
                ExecutionContext::sequential(),
            )
        })
    };
    assert!(build(&[], 1, limits()).is_err());
    assert!(build(&[f64::NAN; 24], 1, limits()).is_err());
    assert!(build(&[0.; 24], 1, limits()).is_err());
    assert!(build(&[1.; 24], usize::MAX, limits()).is_err());
    assert!(build(&[], 0, limits()).is_err());
    assert!(build(&[1.; 24], 1, limits().with_max_setup_bytes(1)).is_err());
    assert!(build(&[1.; 24], 1, limits().with_max_entries(1)).is_err());
    assert!(build(&[1.; 24], 1, limits().with_max_levels(0)).is_err());
    let h = build(&[1.; 24], 1, limits()).unwrap();
    let bytes = h.diagnostics().setup_bytes_bound();
    assert!(build(&[1.; 24], 1, limits().with_max_setup_bytes(bytes)).is_ok());
    assert!(build(&[1.; 24], 1, limits().with_max_setup_bytes(bytes - 1)).is_err());
    let p = preconditioner(&a, AmgCycleKind::V);
    let mut work = vec![0.; p.precondition_workspace_len().unwrap()];
    let mut output = vec![77.; 24];
    assert!(
        p.precondition_into(&[1.; 24], &mut output, &mut [])
            .is_err()
    );
    assert!(
        p.precondition_into(&[f64::NAN; 24], &mut output, &mut work)
            .is_err()
    );
    assert_eq!(output, vec![77.; 24]);
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert!(
        p.apply_into(
            &[1.; 24],
            SparseSolveOptions::new(),
            &mut output,
            &mut SparseSolveWorkspace::borrowed(&mut work),
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(output, vec![77.; 24]);
}
