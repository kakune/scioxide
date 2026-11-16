//! Public const and literal boundaries from an external crate.
#![allow(clippy::unwrap_used)]
use renamed::{AlgorithmId, Argument, Quantity};
use scioxide_core as renamed;

const SOLVER: AlgorithmId = renamed::algorithm_id!("linear", "cg", 2);
const CHECKED: Result<AlgorithmId, renamed::IdentifierError> = AlgorithmId::new("linear", "cg", 2);

const ARGUMENT: Argument = renamed::argument!("options.max_steps");
const ARGUMENT_CHECKED: Result<Argument, renamed::IdentifierError> =
    Argument::new("options.max_steps");
const QUANTITY: Quantity = renamed::quantity!("workspace.capacity");
const QUANTITY_CHECKED: Result<Quantity, renamed::IdentifierError> =
    Quantity::new("workspace.capacity");

#[test]
fn const_and_literal_identities_are_equal() {
    assert_eq!(SOLVER, CHECKED.unwrap());
    assert_eq!(SOLVER.to_string(), "linear/cg@2");
    assert_ne!(SOLVER, renamed::algorithm_id!("linear", "cg", 1));

    assert_eq!(ARGUMENT, ARGUMENT_CHECKED.unwrap());
    assert_eq!(QUANTITY, QUANTITY_CHECKED.unwrap());
    assert_ne!(ARGUMENT.as_str(), QUANTITY.as_str());
}

#[test]
fn runtime_inputs_share_const_validation_without_panicking() {
    for value in ["", "bad name", "bad/name", "é", "line\nfeed"] {
        assert!(AlgorithmId::new("family", value, 0).is_err());
    }
    for text in ["", ".field", "field.", "field..name", "Field", "field-name"] {
        let argument = Argument::new(text).unwrap_err();
        let quantity = Quantity::new(text).unwrap_err();
        assert_eq!(argument, quantity);
        assert!(argument.position() <= text.len());
    }
}
