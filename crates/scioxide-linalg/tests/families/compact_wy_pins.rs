//! Blocked compact-WY Householder fingerprints and decomposition invariants.
//!
//! Each case spans at least two 32-reflector groups, so it runs the shared
//! blocked driver: QR/LQ (Q, Qᴴ, explicit Q and solves), bidiagonal SVD
//! vectors (left offset 0, right offset 1), Hessenberg Schur vectors, QZ
//! triangular packing. Portable sequential dispatch fixes the linear-algebra
//! kernels, but `std` elementary functions may differ across platforms and Rust
//! versions. SVD and Schur/QZ therefore check reconstruction and orthogonality
//! with both scalar backends; their bit fingerprints additionally pin the owned
//! elementary implementation under `no_std`. QR/LQ retain their algebraic pins.
//! Fingerprints are arithmetic regression checks, not accuracy oracles.

use scioxide_core::{Complex, ExecutionContext, ExecutionPolicy, NumericScalar, SciOxideError};
use scioxide_linalg::{
    DirectSolveOptions, GeneralizedSchurDecomposition, HermitianEigenDecomposition,
    HermitianEigenOptions, HermitianInput, HouseholderDecomposition, HouseholderOrientation,
    HouseholderPivoting, LinalgExecutionOptions, LinalgWorkspace, Matrix, RankPolicy,
    SchurDecomposition, SchurOptions, StructuredOperation, SvdDecomposition, ThinSvdDecomposition,
};

type C64 = Complex<f64>;

/// Deterministic pseudo-random entry in `[-0.5, 0.5)` for each component.
trait Entry: NumericScalar<Real = f64> {
    fn entry(row: usize, col: usize, seed: u64) -> Self;
}

impl Entry for f64 {
    fn entry(row: usize, col: usize, seed: u64) -> Self {
        unit(row, col, seed)
    }
}

impl Entry for C64 {
    fn entry(row: usize, col: usize, seed: u64) -> Self {
        Complex::new(unit(row, col, seed), unit(row, col, seed + 0x100))
    }
}

/// SplitMix64 of the entry position, mapped to 53-bit uniform values.
fn unit(row: usize, col: usize, seed: u64) -> f64 {
    let mut state = (row as u64) << 32 ^ col as u64 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^= state >> 31;
    (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
}

fn general<S: Entry>(rows: usize, cols: usize, seed: u64) -> Result<Matrix<S>, SciOxideError> {
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|index| S::entry(index / cols, index % cols, seed))
            .collect(),
    )
}

fn hermitian<S: Entry>(order: usize, seed: u64) -> Result<Matrix<S>, SciOxideError> {
    Matrix::from_vec(
        order,
        order,
        (0..order * order)
            .map(|index| {
                let (row, col) = (index / order, index % order);
                match row.cmp(&col) {
                    core::cmp::Ordering::Less => S::entry(row, col, seed),
                    core::cmp::Ordering::Greater => S::entry(col, row, seed).conj(),
                    core::cmp::Ordering::Equal => {
                        S::from_real(S::entry(row, row, seed).real_component())
                    }
                }
            })
            .collect(),
    )
}

/// FNV-1a over the bit patterns of every component, in order.
fn fingerprint<S: NumericScalar<Real = f64>>(parts: &[&[S]]) -> u64 {
    parts
        .iter()
        .flat_map(|part| part.iter())
        .flat_map(|value| [value.real_component(), value.imaginary_component()])
        .flat_map(|component| component.to_bits().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn sequential() -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential)
}

fn householder<S: Entry>(orientation: HouseholderOrientation) -> Result<u64, SciOxideError> {
    let (rows, cols) = match orientation {
        HouseholderOrientation::Qr => (70, 45),
        HouseholderOrientation::Lq => (45, 70),
    };
    let options = sequential();
    let mut workspace = LinalgWorkspace::new();
    let mut factor = HouseholderDecomposition::empty();
    HouseholderDecomposition::factor_into(
        &general::<S>(rows, cols, 1)?,
        orientation,
        HouseholderPivoting::None,
        &options,
        &mut factor,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    let order = rows.max(cols);
    let mut q = Matrix::zeros(order, order)?;
    factor.q_matrix_into(
        &options,
        q.view_mut(),
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    let rhs = general::<S>(order, 9, 2)?;
    let mut applied = Matrix::zeros(order, 9)?;
    let mut adjoint = Matrix::zeros(order, 9)?;
    for (operation, output) in [
        (StructuredOperation::Normal, &mut applied),
        (StructuredOperation::Adjoint, &mut adjoint),
    ] {
        factor.apply_q_into(
            &rhs,
            operation,
            &options,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )?;
    }
    let mut parts = vec![q.as_slice(), applied.as_slice(), adjoint.as_slice()];
    let mut solution = Matrix::zeros(cols, 3)?;
    if orientation == HouseholderOrientation::Qr {
        let _report = factor.solve_matrix_into(
            &general::<S>(rows, 3, 3)?,
            &DirectSolveOptions::new(RankPolicy::Exact).with_execution_options(options),
            solution.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )?;
        parts.push(solution.as_slice());
    }
    Ok(fingerprint(&parts))
}

fn svd<S: Entry + super::support::decomposition::InvariantScalar>(
    rows: usize,
    cols: usize,
) -> Result<u64, SciOxideError> {
    let options = sequential();
    let matrix = general::<S>(rows, cols, 4)?;
    let mut workspace = LinalgWorkspace::new();
    let mut thin = ThinSvdDecomposition::empty();
    ThinSvdDecomposition::compute_into(
        &matrix,
        &options,
        &mut thin,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    let mut full = SvdDecomposition::empty();
    SvdDecomposition::compute_into(
        &matrix,
        &options,
        &mut full,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    super::support::decomposition::assert_thin_svd_invariants(
        &matrix,
        &thin,
        "blocked compact-WY thin SVD",
    );
    super::support::decomposition::assert_full_svd_invariants(
        &matrix,
        &full,
        "blocked compact-WY full SVD",
    );
    Ok(fingerprint(&[
        thin.u_matrix().as_slice(),
        thin.vt_matrix().as_slice(),
        full.u_matrix().as_slice(),
        full.vt_matrix().as_slice(),
    ]))
}

fn schur<S: Entry>(order: usize) -> Result<u64, SciOxideError> {
    let matrix = general::<S>(order, order, 5)?;
    let mut output = SchurDecomposition::empty();
    SchurDecomposition::decompose_into(
        &matrix,
        &SchurOptions::new().with_execution_options(sequential()),
        &mut output,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential(),
    )?;
    assert_unitary_equivalence(&matrix, output.q(), output.t(), output.q(), "Schur")?;
    Ok(fingerprint(&[output.t().as_slice(), output.q().as_slice()]))
}

fn qz<S: Entry>(order: usize) -> Result<u64, SciOxideError> {
    let a = general::<S>(order, order, 6)?;
    let b = general::<S>(order, order, 7)?;
    let mut output = GeneralizedSchurDecomposition::empty();
    GeneralizedSchurDecomposition::decompose_into(
        &a,
        &b,
        &SchurOptions::new().with_execution_options(sequential()),
        &mut output,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential(),
    )?;
    assert_unitary_equivalence(&a, output.q(), output.s(), output.z(), "QZ A")?;
    assert_unitary_equivalence(&b, output.q(), output.t(), output.z(), "QZ B")?;
    Ok(fingerprint(&[
        output.s().as_slice(),
        output.t().as_slice(),
        output.q().as_slice(),
        output.z().as_slice(),
    ]))
}

fn assert_unitary_equivalence<S: Entry>(
    matrix: &Matrix<S>,
    left: &Matrix<S>,
    factor: &Matrix<S>,
    right: &Matrix<S>,
    label: &str,
) -> Result<(), SciOxideError> {
    use super::support::metrics;

    let right_adjoint = right.view().adjoint().try_to_matrix()?;
    let reconstructed = left
        .view()
        .mul_mat(factor)?
        .view()
        .mul_mat(&right_adjoint)?;
    let bound = metrics::invariant_bound::<f64>(matrix.rows());
    let residual = metrics::relative_matrix_difference(&reconstructed, matrix);
    assert!(
        residual <= bound,
        "{label}: reconstruction {residual} > {bound}"
    );
    for (side, vectors) in [("left", left), ("right", right)] {
        let residual = metrics::column_orthogonality_residual(vectors);
        assert!(
            residual <= bound,
            "{label}: {side} orthogonality {residual} > {bound}"
        );
    }
    Ok(())
}

fn hermitian_vectors<S: Entry + super::support::decomposition::InvariantScalar>(
    order: usize,
) -> Result<(), SciOxideError> {
    let matrix = hermitian::<S>(order, 8)?;
    let mut output = HermitianEigenDecomposition::empty();
    HermitianEigenDecomposition::decompose_into(
        &matrix,
        HermitianInput::Exact,
        &HermitianEigenOptions::default().with_execution_options(sequential()),
        &mut output,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential(),
    )?;
    super::support::decomposition::assert_hermitian_invariants(
        &matrix,
        &output,
        "blocked compact-WY Hermitian vectors",
    );
    Ok(())
}

fn pin(label: &str, actual: u64, expected: u64) {
    assert_eq!(actual, expected, "{label}: fingerprint {actual:#018x}");
}

#[test]
fn qr_and_lq_application_bits_are_pinned() -> Result<(), SciOxideError> {
    pin(
        "f64 QR",
        householder::<f64>(HouseholderOrientation::Qr)?,
        0x20b4_0007_8d95_c4a8,
    );
    pin(
        "c64 QR",
        householder::<C64>(HouseholderOrientation::Qr)?,
        0x19aa_041e_134e_e618,
    );
    pin(
        "f64 LQ",
        householder::<f64>(HouseholderOrientation::Lq)?,
        0x926c_749c_1ba1_f9d6,
    );
    Ok(())
}

#[cfg(not(feature = "std"))]
#[test]
fn svd_vector_bits_are_pinned() -> Result<(), SciOxideError> {
    pin("f64 SVD tall", svd::<f64>(70, 45)?, 0x3916_32fb_b67e_6fb5);
    pin("f64 SVD wide", svd::<f64>(45, 70)?, 0xcd72_ce21_1c68_0ac5);
    pin("c64 SVD tall", svd::<C64>(70, 45)?, 0x7ff8_e4c0_c4cb_85f5);
    Ok(())
}

#[cfg(feature = "std")]
#[test]
fn svd_vectors_preserve_reconstruction_and_orthogonality() -> Result<(), SciOxideError> {
    svd::<f64>(70, 45)?;
    svd::<f64>(45, 70)?;
    svd::<C64>(70, 45)?;
    Ok(())
}

#[cfg(not(feature = "std"))]
#[test]
fn hessenberg_schur_and_qz_vector_bits_are_pinned() -> Result<(), SciOxideError> {
    pin("f64 Schur", schur::<f64>(161)?, 0xd016_724a_db23_917b);
    pin("f64 QZ", qz::<f64>(161)?, 0xe303_2f6b_3864_20d7);
    Ok(())
}

#[cfg(feature = "std")]
#[test]
fn hessenberg_schur_and_qz_vectors_preserve_reconstruction_and_orthogonality()
-> Result<(), SciOxideError> {
    schur::<f64>(161)?;
    qz::<f64>(161)?;
    Ok(())
}

#[test]
fn blocked_hermitian_vectors_preserve_eigenpairs_and_orthogonality() -> Result<(), SciOxideError> {
    // Eigenvector bases may change with the tridiagonal eigensolver and scalar
    // backend. Check the mathematical result; worker-count reproducibility is
    // covered separately in numerical_quality.
    hermitian_vectors::<f64>(100)?;
    hermitian_vectors::<C64>(100)?;
    Ok(())
}
