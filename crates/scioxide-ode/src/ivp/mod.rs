//! Initial-value methods: options, problems, workspaces, output and reports.

mod builtin;
mod driver;
mod events;
mod oneshot;
mod options;
pub use oneshot::{solve_ivp, solve_ivp_at};
pub(crate) mod plan;
mod problem;
mod report;
mod sink;
#[cfg(test)]
mod tests;
mod workspace;

pub use builtin::BuiltinOdeMethod;
pub use options::{OdeOptions, OdeOutputSchedule, OdeStepControl};

pub use problem::{OdeProblem, OdeState};
pub use report::{OdeProgress, OdeSolution, OdeSolveReport, OdeTermination};
pub use sink::{EventInfo, OdeEvent, OdeOutputSink, OdeTrajectory};
pub use workspace::{OdeRequirements, OdeWorkspace};

pub(crate) use driver::{
    Drive, Finish, Recorder, TrialDrive, TrialKernel, drive, drive_trials, validate_requested_times,
};
pub(crate) use plan::{progress, report, unsupported};
pub(crate) use workspace::IvpStorage;
