//! One error type for the whole crate, and the panic guard every thread boundary uses.

use std::any::Any;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// A readable failure message. Becomes a JavaScript `Error` at the N-API boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<Error> for napi::Error {
    fn from(error: Error) -> Self {
        napi::Error::from_reason(error.0)
    }
}

/// Shorthand for turning any displayable error into an [`Error`].
pub fn err<E: fmt::Display>(error: E) -> Error {
    Error::new(error.to_string())
}

/// Runs `f`, converting a panic into an [`Error`] so it can be reported instead of
/// silently killing a worker thread (which would leave a JavaScript caller waiting forever).
pub fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic| Err(panic_error(&panic)))
}

/// Describes a panic payload as an [`Error`].
pub fn panic_error(panic: &Box<dyn Any + Send>) -> Error {
    let detail = panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown cause".to_string());
    Error::new(format!("internal error: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_passes_results_through() {
        assert_eq!(guard(|| Ok(7)), Ok(7));
        assert_eq!(guard::<()>(|| Err(Error::new("nope"))), Err(Error::new("nope")));
    }

    #[test]
    fn guard_turns_a_panic_into_an_error() {
        let result = guard::<()>(|| panic!("boom"));
        assert_eq!(result, Err(Error::new("internal error: boom")));
    }
}
