//! Options-selected accuracy and call-local operator capabilities.
#![allow(clippy::unwrap_used)]

use core::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{
    Budget, ExecutionContext, ProcessorCapabilities, ProcessorFeature, ResidualTolerance,
};
use scioxide_linalg::{LinalgExecutionContext, Matrix, Preconditioning, Vector};

struct Processor(AtomicUsize);
impl ProcessorCapabilities for Processor {
    fn supports(&self, _: ProcessorFeature) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        false
    }
}

#[test]
fn arnoldi_and_short_recurrences_reuse_dispatch_for_their_reductions() {
    use scioxide_linalg::{GmresSolver, IterativeWorkspace, LsmrSolver, LsqrSolver, MinresSolver};
    let n = 16;
    let matrix = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    (i / n + 1) as f64
                } else {
                    0.0
                }
            })
            .collect(),
    )
    .unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let rhs = Vector::from_vec(vec![1.0; n]);
    let tolerance = ResidualTolerance::new(0.0, 1e-10).unwrap();
    let limit = Budget::try_from(128).unwrap();
    for method in 0..4 {
        let processor = Processor(AtomicUsize::new(0));
        let context =
            LinalgExecutionContext::new(ExecutionContext::sequential().with_processor(&processor));
        let expected = processor.0.load(Ordering::Relaxed);
        let mut workspace = IterativeWorkspace::new();
        let mut output = Vector::zeros(n).unwrap();
        let report = match method {
            0 => GmresSolver::new(tolerance, limit).solve_into(
                &operator,
                &rhs,
                Preconditioning::None,
                output.view_mut(),
                &mut workspace,
                context,
            ),
            1 => MinresSolver::new(tolerance, limit).solve_into(
                &operator,
                &rhs,
                Preconditioning::None,
                output.view_mut(),
                &mut workspace,
                context,
            ),
            2 => LsqrSolver::new(tolerance, limit).solve_into(
                &operator,
                &rhs,
                output.view_mut(),
                &mut workspace,
                context,
            ),
            _ => LsmrSolver::new(tolerance, limit).solve_into(
                &operator,
                &rhs,
                output.view_mut(),
                &mut workspace,
                context,
            ),
        }
        .unwrap();
        assert!(report.is_success(), "method {method}: {report:?}");
        assert_eq!(
            processor.0.load(Ordering::Relaxed),
            expected,
            "method {method}"
        );
    }
}
