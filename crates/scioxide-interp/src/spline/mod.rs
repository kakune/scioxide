//! B-spline bases, tensor and rational B-splines, spline fits and prepared
//! antiderivatives.

mod antiderivative;
mod basis;
mod bspline;
mod cardinal;
mod collocation;
mod nurbs;
mod penalized;
pub(crate) mod preparation;

pub use antiderivative::PreparedSplineAntiderivative;
pub(crate) use antiderivative::primitive;
pub use basis::{BSplineBasis, KnotSide, SplineBasisWorkspace};
pub use bspline::BSpline;
pub use cardinal::cardinal_bspline;
pub use collocation::{
    TensorSplineInterpolationPlan, TensorSplineInterpolationRequirements,
    TensorSplineInterpolationWorkspace,
};
pub use nurbs::Nurbs;
pub use penalized::{
    PenalizedSplineFit, PenalizedSplinePlan, PenalizedSplineRequirements, PenalizedSplineWorkspace,
    SplineFitDiagnostics, SplineFitReport, SplinePenalty, difference_coefficients,
};
