//! Error assertions shared by the signal integration tests.
use scioxide_core::{SciOxideError, SciOxideErrorKind};

/// Error of an expected failure; panics at the caller when `result` is `Ok`.
#[allow(dead_code)]
#[track_caller]
pub fn error<T>(result: Result<T, SciOxideError>) -> SciOxideError {
    #[allow(clippy::panic)]
    match result {
        Ok(_) => panic!("expected an error, got Ok"),
        Err(error) => error,
    }
}

/// Kind of an expected failure; panics at the caller when `result` is `Ok`.
#[allow(dead_code)]
#[track_caller]
pub fn error_kind<T>(result: Result<T, SciOxideError>) -> SciOxideErrorKind {
    error(result).kind()
}
