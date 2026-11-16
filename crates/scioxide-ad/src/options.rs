//! Policies shared by tapes and batches.

/// Selects how non-finite points, values and derivatives are treated at a
/// transform boundary.
///
/// Number types always follow IEEE 754; the policy applies where a transform reads
/// its point and writes its results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NonFinitePolicy {
    /// Writes what was computed, including NaN and infinity.
    #[default]
    Propagate,
    /// Rejects a non-finite point before the function runs (`NonFinite`, origin
    /// `Input`) and a non-finite value or derivative before anything is written
    /// (origin `Output`).
    Error,
}

/// Selects what a tape does when an active argument sits on a kink.
///
/// Every number type uses the midpoint of the one-sided derivatives at a kink (see
/// [kinks](crate#kinks)). A tape also counts one hit per lane in which an active
/// argument sits on one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KinkPolicy {
    /// Uses the midpoint derivative and reports the count.
    #[default]
    Midpoint,
    /// Turns a non-zero count into `DomainError` (argument `point`, reason
    /// `Undefined`) at the transform boundary.
    Error,
}

/// Selects the sweep direction of a dense Jacobian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JacobianMode {
    /// Reverse sweeps when there are fewer outputs than inputs, tangent sweeps
    /// otherwise.
    #[default]
    Auto,
    /// One tangent sweep per input.
    Forward,
    /// One reverse sweep per output.
    Reverse,
}

/// Selects how a batch computes gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DerivativeMode {
    /// Forward mode up to four active inputs, reverse mode beyond and under
    /// [`KinkPolicy::Error`].
    #[default]
    Auto,
    /// Forward mode with one tangent per active input.
    Forward,
    /// Reverse mode on a tape.
    Reverse,
}
