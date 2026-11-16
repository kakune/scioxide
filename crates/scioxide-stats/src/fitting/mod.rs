//! Parametric distribution fitting with a solver you supply, probability plots
//! and power transformations.
use crate::error_helpers::{finite, invalid, number};
mod model;
mod problem;
mod solver;
mod transform;

pub use model::{
    DistributionFitModel, ExactFitDerivatives, FitModel, FitObservation, FitTruncation, LawFitModel,
};
pub use problem::{FitMethod, FitObjective, PreparedFit, fit_objectives_into};
pub use solver::{
    FitDerivativeCapabilities, FitDerivatives, FitIdentifiability, FitInformation, FitReport,
    FitRequest, FitResult, FitSolver, FitSolverCapabilities, FitWorkspace, fit, fit_information,
    fit_into, fit_scoped, fit_with_options,
};
pub use transform::{
    PowerFitMethod, PowerTransform, PreparedPowerFit, PreparedPpcc, ProbabilityPlot,
    ProbabilityPlotFit, power_transform_into, power_transform_llf, power_transform_normmax,
    power_transform_normmax_scoped, power_transform_normmax_with_options, ppcc_max,
    ppcc_max_scoped, ppcc_max_with_options, probplot, probplot_with_options,
};
