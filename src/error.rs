//! One place that turns any displayable error into a JavaScript `Error`.

pub use napi::Result;

pub fn err<E: std::fmt::Display>(e: E) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}
