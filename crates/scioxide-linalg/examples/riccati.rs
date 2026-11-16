//! Stabilizing descriptor CARE/DARE with an explicit cross term.
use scioxide_core::SciOxideError;
use scioxide_linalg::{Matrix, RiccatiKind, RiccatiOptions, RiccatiPlan, RiccatiProblem};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::<f64>::from_vec(1, 1, vec![0.])?;
    let b = Matrix::<f64>::identity(1)?;
    let r = Matrix::<f64>::identity(1)?;
    let e = b.view().try_scale(2.)?;
    let s = b.view().try_scale(0.25)?;
    for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
        let q = b.view().try_scale(if kind == RiccatiKind::Continuous {
            5.0625
        } else {
            4.03125
        })?;
        let plan = RiccatiPlan::new(
            RiccatiProblem {
                a: a.view(),
                b: b.view(),
                q: q.view(),
                r: r.view(),
                e: Some(e.view()),
                s: Some(s.view()),
            },
            kind,
            RiccatiOptions::new(),
        )?;
        let result = plan.solve()?;
        let report = result.report();
        match (result.solution(), result.gain()) {
            (Some(x), Some(k)) => println!(
                "{kind:?}: X={:?}, K={:?}, {:?}, stability={:?}",
                x.as_slice(),
                k.as_slice(),
                report.termination(),
                report.stability_margin()
            ),
            _ => println!("{kind:?}: rejected with {:?}", report.termination()),
        }
    }
    Ok(())
}
