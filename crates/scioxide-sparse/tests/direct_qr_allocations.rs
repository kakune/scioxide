//! Fallible ownership/preparation and allocation-free prepared QR epochs.
#![allow(clippy::unwrap_used)]
#![forbid(unsafe_code)]

use scioxide_core::SciOxideErrorKind;
use scioxide_sparse::{CsrMatrix, SparseOrdering, SparseQrLimits, SparseQrSymbolic};

#[test]
fn reservation_admits_intermediate_retained_capacity_before_allocating() {
    use scioxide_core::Capacity;
    let tall = CsrMatrix::<f64>::from_raw(4000, 1, vec![0; 4001], vec![], vec![]).unwrap();
    let old = SparseQrSymbolic::analyze(
        tall.structural_pattern(),
        SparseOrdering::Natural,
        SparseQrLimits::new(0, 0, 1 << 20),
    )
    .unwrap();
    let mut workspace = old.workspace::<f64>(()).unwrap();
    let empty = CsrMatrix::<f64>::from_raw(0, 0, vec![0], vec![], vec![]).unwrap();
    let analyze = |bytes| {
        SparseQrSymbolic::analyze(
            empty.structural_pattern(),
            SparseOrdering::Natural,
            SparseQrLimits::new(1000, 0, bytes),
        )
        .unwrap()
    };
    let roomy = analyze(usize::MAX);
    let target = analyze(
        roomy
            .workspace_requirements::<f64>(())
            .unwrap()
            .total_bytes(),
    );
    let requirements = target.workspace_requirements::<f64>(()).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let error = workspace.try_reserve(&requirements).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(
        workspace
            .check(&old.workspace_requirements::<f64>(()).unwrap())
            .unwrap(),
        Capacity::Sufficient
    );
}
