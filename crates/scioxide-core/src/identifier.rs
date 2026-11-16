//! Names of arguments, quantities and algorithm components, checked at compile time.

use crate::{InvalidInputReason, SciOxideError};
use core::fmt;

/// Syntax rules for identifiers. Validation never changes the spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IdentifierGrammar {
    /// `[A-Za-z0-9_][A-Za-z0-9_.-]*`, case-sensitive.
    AlgorithmComponent,
    /// Lowercase ASCII identifier segments separated by `.`.
    ArgumentPath,
}

/// Error for an identifier that breaks its syntax rules, with the byte position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentifierError {
    grammar: IdentifierGrammar,
    position: usize,
}
impl IdentifierError {
    /// Returns the zero-based position of the first invalid byte, or the input
    /// length if a segment is missing at the end.
    pub const fn position(self) -> usize {
        self.position
    }
    /// Converts the error to an `InvalidInput` [`SciOxideError`] with reason
    /// [`InvalidInputReason::Malformed`] and argument `identifier`.
    ///
    /// ```
    /// use scioxide_core::{Argument, SciOxideErrorView, InvalidInputReason};
    /// let error = Argument::new("not a path").unwrap_err().into_scioxide();
    /// assert!(matches!(
    ///     error.view(),
    ///     SciOxideErrorView::InvalidInput {
    ///         argument,
    ///         reason: InvalidInputReason::Malformed,
    ///         ..
    ///     } if argument == scioxide_core::argument!("identifier")
    /// ));
    /// ```
    pub const fn into_scioxide(self) -> SciOxideError {
        SciOxideError::invalid_input_static(
            crate::argument!("identifier"),
            InvalidInputReason::Malformed,
            "identifier does not satisfy its declared grammar",
        )
    }
}
impl fmt::Display for IdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid {:?} identifier at byte {}",
            self.grammar, self.position
        )
    }
}
impl core::error::Error for IdentifierError {}

/// A borrowed identity whose spelling satisfies its declared grammar.
///
/// Backs the public identifier types; there is no unchecked constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ValidatedIdentifier<'a> {
    text: &'a str,
    grammar: IdentifierGrammar,
}
impl<'a> ValidatedIdentifier<'a> {
    /// Checks the syntax of `text`; usable in const evaluation.
    ///
    /// # Errors
    ///
    /// Returns the position of the first invalid byte or missing segment.
    pub(crate) const fn new(
        text: &'a str,
        grammar: IdentifierGrammar,
    ) -> Result<Self, IdentifierError> {
        let bytes = text.as_bytes();
        let mut index = 0;
        let mut first = true;
        while index < bytes.len() {
            let byte = bytes[index];
            let lower = byte >= b'a' && byte <= b'z';
            let upper = byte >= b'A' && byte <= b'Z';
            let digit = byte >= b'0' && byte <= b'9';
            let letter = lower || (upper && !matches!(grammar, IdentifierGrammar::ArgumentPath));
            let valid = letter
                || byte == b'_'
                || (digit && (!first || matches!(grammar, IdentifierGrammar::AlgorithmComponent)))
                || (!first
                    && matches!(grammar, IdentifierGrammar::AlgorithmComponent)
                    && (byte == b'.' || byte == b'-'));
            if valid {
                first = false;
                index += 1;
            } else if !first && matches!(grammar, IdentifierGrammar::ArgumentPath) && byte == b'.' {
                first = true;
                index += 1;
            } else {
                return Err(IdentifierError {
                    grammar,
                    position: index,
                });
            }
        }
        if first {
            return Err(IdentifierError {
                grammar,
                position: index,
            });
        }
        Ok(Self { text, grammar })
    }
    /// Returns the checked spelling unchanged.
    pub(crate) const fn as_str(self) -> &'a str {
        self.text
    }
}
impl fmt::Display for ValidatedIdentifier<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text)
    }
}

macro_rules! identity {
    ($(#[$doc:meta])* $name:ident, $grammar:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(&'static str);
        impl $name {
            /// Checks the syntax of `text` and wraps it; usable in const evaluation.
            ///
            /// # Errors
            ///
            /// Returns an [`IdentifierError`] with the position of the first
            /// invalid byte or missing segment.
            pub const fn new(text: &'static str) -> Result<Self, IdentifierError> {
                match ValidatedIdentifier::new(text, IdentifierGrammar::$grammar) {
                    Ok(_) => Ok(Self(text)), Err(error) => Err(error),
                }
            }
            /// Returns the spelling.
            pub const fn as_str(self) -> &'static str { self.0 }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.0) }
        }
    };
}

identity! {
/// Name of an input parameter or of a field inside one, such as `options.max_steps`.
///
/// Errors use it to say which argument was rejected. Each segment starts with a
/// lowercase ASCII letter or underscore, followed by lowercase ASCII letters,
/// digits or underscores. Dots separate nonempty segments. Create one with the
/// [`argument!`](macro@crate::argument) macro, which checks the name at compile time.
///
/// ```
/// const INPUT: scioxide_core::Argument = scioxide_core::argument!("options.max_steps");
/// assert_eq!(INPUT.as_str(), "options.max_steps");
/// ```
    Argument, ArgumentPath
}
identity! {
/// Name of a size, capacity, counter or collection, such as `workspace.capacity`.
///
/// Errors use it to say which size failed. It has the same syntax as
/// [`Argument`] but is a separate type, because a resource is not an input
/// parameter: `workspace.capacity` names a resource, while `options.capacity`
/// could name a parameter. Create one with the [`quantity!`](macro@crate::quantity)
/// macro.
///
/// ```
/// const CAPACITY: scioxide_core::Quantity = scioxide_core::quantity!("workspace.capacity");
/// assert_eq!(CAPACITY.as_str(), "workspace.capacity");
/// ```
    Quantity, ArgumentPath
}
identity! {
/// Family or method name in an [`AlgorithmId`](crate::AlgorithmId).
///
/// Components match `[A-Za-z0-9_][A-Za-z0-9_.-]*`. A component checked once,
/// usually at compile time, can be combined into identities with
/// [`AlgorithmId::from_components`](crate::AlgorithmId::from_components)
/// without being checked again.
///
/// ```
/// use scioxide_core::{AlgorithmComponent, AlgorithmId};
/// const FAMILY: AlgorithmComponent = scioxide_core::algorithm_component!("linear");
/// const METHOD: AlgorithmComponent = scioxide_core::algorithm_component!("cg");
/// let identity = AlgorithmId::from_components(FAMILY, METHOD, 1);
/// assert_eq!(identity, scioxide_core::algorithm_id!("linear", "cg", 1));
/// ```
    AlgorithmComponent, AlgorithmComponent
}
impl ValidatedIdentifier<'static> {
    /// The validated spelling of an algorithm component.
    pub(crate) const fn component(component: AlgorithmComponent) -> Self {
        Self {
            text: component.0,
            grammar: IdentifierGrammar::AlgorithmComponent,
        }
    }
}

/// Creates an [`Argument`](crate::Argument) whose syntax is checked at compile time.
///
/// The check uses [`Argument::new`](crate::Argument::new); an invalid name fails to compile.
///
/// ```compile_fail
/// let _ = scioxide_core::argument!("Options.MaxSteps");
/// ```
#[macro_export]
macro_rules! argument {
    ($text:literal) => {
        const {
            match $crate::Argument::new($text) {
                Ok(identity) => identity,
                Err(_) => panic!("invalid argument literal"),
            }
        }
    };
}

/// Creates an [`AlgorithmComponent`](crate::AlgorithmComponent) checked at compile time.
///
/// Accepts a string literal or a `concat!` or `stringify!` expression.
///
/// ```compile_fail
/// let _ = scioxide_core::algorithm_component!("bad method");
/// ```
#[macro_export]
macro_rules! algorithm_component {
    ($text:expr) => {
        const {
            match $crate::AlgorithmComponent::new($text) {
                Ok(component) => component,
                Err(_) => panic!("invalid algorithm component literal"),
            }
        }
    };
}

/// Creates a [`Quantity`](crate::Quantity) whose syntax is checked at compile time.
///
/// The check uses [`Quantity::new`](crate::Quantity::new); an invalid name fails to compile.
///
/// ```compile_fail
/// let _ = scioxide_core::quantity!("workspace capacity");
/// ```
#[macro_export]
macro_rules! quantity {
    ($text:literal) => {
        const {
            match $crate::Quantity::new($text) {
                Ok(identity) => identity,
                Err(_) => panic!("invalid quantity literal"),
            }
        }
    };
}

/// Creates an [`AlgorithmId`](crate::AlgorithmId) checked at compile time.
///
/// The arguments are the family, the method and the version number.
///
/// ```
/// const ID: scioxide_core::AlgorithmId = scioxide_core::algorithm_id!("linear", "cg", 1);
/// assert_eq!(ID.to_string(), "linear/cg@1");
/// ```
/// ```compile_fail
/// let _ = scioxide_core::algorithm_id!("linear", "bad method", 1);
/// ```
#[macro_export]
macro_rules! algorithm_id {
    ($family:expr, $method:expr, $version:expr) => {
        const {
            match $crate::AlgorithmId::new($family, $method, $version) {
                Ok(identity) => identity,
                Err(_) => panic!("invalid algorithm identity literal"),
            }
        }
    };
}
