//! Reviewed public surface and independent source-inventory regressions.
#![allow(clippy::unwrap_used)]
#[path = "families/api_consumer.rs"]
mod consumer;
#[path = "support/public_surface.rs"]
mod public_surface;

#[test]
fn public_surface_matches_reviewed_snapshot() {
    let actual = public_surface::linalg().snapshot();
    let expected = include_str!("data/public_surface.txt");
    let actual: std::collections::BTreeSet<_> = actual.lines().collect();
    let expected: std::collections::BTreeSet<_> = expected.lines().collect();
    let added: Vec<_> = actual.difference(&expected).collect();
    let removed: Vec<_> = expected.difference(&actual).collect();
    assert!(
        added.is_empty() && removed.is_empty(),
        "public API changed; review and update tests/data/public_surface.txt\nadded: {added:#?}\nremoved: {removed:#?}"
    );
}

#[test]
fn snapshot_detects_additions_removals_and_signatures() {
    let original =
        public_surface::Surface::from_source("pub struct P; impl P { pub fn run(&self) {} }")
            .snapshot();
    for source in [
        "pub struct P; impl P { pub fn run(&self) {} pub fn added() {} }",
        "pub struct P; pub struct Added; impl P { pub fn run(&self) {} }",
        "pub struct P; pub static ADDED: usize = 0; impl P { pub fn run(&self) {} }",
        "pub struct P;",
        "pub struct P; impl P { pub fn run(&self, x: usize) {} }",
    ] {
        assert_ne!(
            original,
            public_surface::Surface::from_source(source).snapshot()
        );
    }
    assert_eq!(original, public_surface::Surface::from_source("/// docs\npub struct P; impl P { /// method docs\npub fn run(&self) { let _private = 1; } }").snapshot());
}

#[test]
fn public_fields_variants_and_associated_items_change_the_snapshot() {
    for (before, after) in [
        (
            "pub struct P { pub x: usize }",
            "pub struct P { pub x: usize, pub y: usize }",
        ),
        ("pub enum E { A }", "pub enum E { A, B }"),
        (
            "pub struct P;",
            "pub struct P; impl P { pub const C: usize = 0; }",
        ),
        (
            "pub trait T { fn run(&self); }",
            "pub trait T { fn run(&self); fn added(&self); }",
        ),
    ] {
        assert_ne!(
            public_surface::Surface::from_source(before).snapshot(),
            public_surface::Surface::from_source(after).snapshot()
        );
    }
}

#[test]
#[should_panic(expected = "unhandled item macro")]
fn unrecognized_declaration_macros_fail_closed() {
    public_surface::Surface::from_source("external_macro!(NewPublicType);");
}

#[test]
fn macro_members_aliases_and_feature_gates_are_visible() {
    let surface = public_surface::Surface::from_source(
        r#"
        macro_rules! plan { ($name:ident) => { pub struct $name; impl $name { pub fn execute(&self) {} } }; }
        plan!(Plan);
        #[cfg(feature = "blas")] pub use Plan as Alias;
    "#,
    );
    assert!(surface.names.contains("Plan::execute"));
    assert!(surface.names.contains("Alias::execute"));
    assert!(!surface.names.contains("Plan::removed"));
    assert!(surface.snapshot().contains("pub use Plan as Alias"));
    assert!(surface.snapshot().contains("blas"));
    assert_ne!(
        surface.snapshot(),
        public_surface::Surface::from_source("pub struct Plan;").snapshot()
    );
}
