use super::storage::{byte_sum, cca_bytes, factor_work_bytes, payload, pca_bytes};
use scioxide_core::{Complex, ExecutionContext, SciOxideError};

#[test]
fn byte_bounds_detect_overflow_without_allocating() {
    let context = ExecutionContext::sequential();
    for error in [
        pca_bytes::<f64>(usize::MAX, 2, context).err(),
        cca_bytes::<Complex<f64>>(2, usize::MAX, 1, context).err(),
        payload::<f64>(&[usize::MAX]).err(),
        byte_sum([usize::MAX, 1]).err(),
    ] {
        assert!(matches!(
            error.map(|e| e.kind()),
            Some(scioxide_core::SciOxideErrorKind::SizeOverflow)
        ));
    }
}

#[test]
fn cross_workspace_bound_covers_truncated_rank_geometries() -> Result<(), SciOxideError> {
    let context = ExecutionContext::sequential();
    for (nx, ny) in [(7, 9), (65, 66), (129, 65), (64, 132)] {
        let bound = factor_work_bytes::<Complex<f64>>(nx, ny, context)?;
        for rx in 1..=nx {
            for ry in 1..=ny {
                assert!(
                    factor_work_bytes::<Complex<f64>>(rx, ry, context)? <= bound,
                    "{rx}x{ry} exceeds {nx}x{ny}"
                );
            }
        }
    }
    Ok(())
}
