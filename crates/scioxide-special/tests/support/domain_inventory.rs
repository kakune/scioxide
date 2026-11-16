//! Source-derived inventory of public real and complex scalar functions, independent of the sweep table.
//!
//! Follow Rust module declarations and root re-exports, including aliases and globs.
//! Local facade macros are inspected for public function declarations; their name
//! metavariables are bound at each invocation. Unsupported bindings fail closed.

use proc_macro2::{TokenStream, TokenTree};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::{GenericArgument, Item, PathArguments, ReturnType, Type, UseTree, Visibility};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ScalarDomains {
    real: bool,
    complex: bool,
}
type Exports = BTreeMap<String, ScalarDomains>;

fn scalar(ty: &Type, generics: &syn::Generics) -> ScalarDomains {
    let Type::Path(path) = ty else {
        return ScalarDomains::default();
    };
    let Some(segment) = path.path.segments.last() else {
        return ScalarDomains::default();
    };
    if segment.ident == "Result" {
        if let PathArguments::AngleBracketed(args) = &segment.arguments
            && let Some(GenericArgument::Type(ty)) = args.args.first()
        {
            return scalar(ty, generics);
        }
        return ScalarDomains::default();
    }
    if matches!(
        segment.ident.to_string().as_str(),
        "Complex" | "Complex32" | "Complex64"
    ) {
        return ScalarDomains {
            real: false,
            complex: true,
        };
    }
    if !matches!(segment.arguments, PathArguments::None) || path.path.segments.len() != 1 {
        return ScalarDomains::default();
    }
    if segment.ident == "f32" || segment.ident == "f64" {
        return ScalarDomains {
            real: true,
            complex: false,
        };
    }
    let bound = |names: &[&str]| {
        let matches = |b: &syn::TypeParamBound| {
            matches!(b, syn::TypeParamBound::Trait(b)
            if b.path.segments.last().is_some_and(|s| names.contains(&s.ident.to_string().as_str())))
        };
        generics.type_params().any(|p| p.ident == segment.ident && p.bounds.iter().any(matches))
            || generics.where_clause.as_ref().is_some_and(|clause| clause.predicates.iter().any(|p|
                matches!(p, syn::WherePredicate::Type(p) if matches!(&p.bounded_ty,Type::Path(t) if t.path.is_ident(&segment.ident)) && p.bounds.iter().any(matches))))
    };
    ScalarDomains {
        real: bound(&["RealScalar", "NumericScalar"]),
        complex: bound(&["NumericScalar"]) && !bound(&["RealScalar"]),
    }
}

fn domains(signature: &syn::Signature) -> ScalarDomains {
    match &signature.output {
        ReturnType::Type(_, ty) => scalar(ty, &signature.generics),
        ReturnType::Default => ScalarDomains::default(),
    }
}

fn segments(tokens: TokenStream) -> Vec<Vec<TokenTree>> {
    let mut result = vec![Vec::new()];
    for token in tokens {
        if matches!(&token, TokenTree::Punct(p) if p.as_char() == ',') {
            result.push(Vec::new());
        } else {
            result.last_mut().unwrap().push(token);
        }
    }
    result
}

fn substituted(tokens: TokenStream, bindings: &BTreeMap<String, TokenStream>) -> TokenStream {
    let mut result = TokenStream::new();
    let mut tokens = tokens.into_iter();
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(p) if p.as_char() == '$' => match tokens.next().unwrap() {
                TokenTree::Ident(name) => result.extend(
                    bindings
                        .get(&name.to_string())
                        .cloned()
                        .unwrap_or_else(|| name.to_string().parse().unwrap()),
                ),
                TokenTree::Group(group) => {
                    result.extend(substituted(group.stream(), bindings));
                    let separator = tokens.next().unwrap();
                    if !matches!(&separator, TokenTree::Punct(p) if matches!(p.as_char(), '*' | '+' | '?'))
                    {
                        assert!(
                            matches!(tokens.next(), Some(TokenTree::Punct(p)) if matches!(p.as_char(), '*' | '+' | '?'))
                        );
                    }
                }
                other => panic!("unsupported API macro token: {other}"),
            },
            TokenTree::Group(group) => result.extend([TokenTree::Group(proc_macro2::Group::new(
                group.delimiter(),
                substituted(group.stream(), bindings),
            ))]),
            token => result.extend([token]),
        }
    }
    result
}

/// Read the shared entry declaration's ordinary signature. Detailed and scoped
/// results are diagnostic containers, so they do not enter scalar domain sweeps.
fn entry_functions(call: &syn::ItemMacro) -> Exports {
    let tokens: Vec<_> = call.mac.tokens.clone().into_iter().collect();
    if matches!(tokens.first(), Some(TokenTree::Punct(p)) if p.as_char() == '@') {
        return Exports::new();
    }
    let mut index = 0;
    while matches!(tokens.get(index), Some(TokenTree::Punct(p)) if p.as_char() == '#') {
        assert!(matches!(tokens.get(index + 1), Some(TokenTree::Group(_))));
        index += 2;
    }
    if !matches!(tokens.get(index), Some(TokenTree::Ident(v)) if v == "pub") {
        return Exports::new();
    }
    index += 1;
    if matches!(tokens.get(index), Some(TokenTree::Group(g)) if g.delimiter() == proc_macro2::Delimiter::Parenthesis)
    {
        return Exports::new();
    }
    let TokenTree::Ident(name) = &tokens[index] else {
        panic!("entry name")
    };
    let TokenTree::Group(generics) = &tokens[index + 1] else {
        panic!("entry generics")
    };
    let TokenTree::Group(arguments) = &tokens[index + 2] else {
        panic!("entry arguments")
    };
    let end = tokens
        .iter()
        .enumerate()
        .skip(index + 3)
        .find(|(_, token)| matches!(token, TokenTree::Ident(n) if n == "where"))
        .map(|(i, _)| i)
        .unwrap();
    let output: TokenStream = tokens[index + 3..end].iter().cloned().collect();
    let TokenTree::Group(bounds) = &tokens[end + 1] else {
        panic!("entry bounds")
    };
    let function: syn::ItemFn = syn::parse_str(&format!(
        "pub fn {name}<{}>({}) {output} where {} {{}}",
        generics.stream(),
        arguments.stream(),
        bounds.stream()
    ))
    .unwrap();
    BTreeMap::from([(name.to_string(), domains(&function.sig))])
}

fn macro_functions(definition: &syn::ItemMacro, call: &syn::ItemMacro) -> Exports {
    let args = segments(call.mac.tokens.clone());
    let rules: Vec<_> = definition.mac.tokens.clone().into_iter().collect();
    let mut result = Exports::new();
    let mut saw_function = false;
    for rule in rules.windows(4) {
        let [
            TokenTree::Group(matcher),
            TokenTree::Punct(eq),
            TokenTree::Punct(gt),
            TokenTree::Group(body),
        ] = rule
        else {
            continue;
        };
        if eq.as_char() != '=' || gt.as_char() != '>' {
            continue;
        }
        let mut bindings = BTreeMap::new();
        for (index, parameter) in segments(matcher.stream()).iter().enumerate() {
            for binding in parameter.windows(4) {
                if let [
                    TokenTree::Punct(dollar),
                    TokenTree::Ident(name),
                    TokenTree::Punct(colon),
                    TokenTree::Ident(kind),
                ] = binding
                    && dollar.as_char() == '$'
                    && colon.as_char() == ':'
                    && let Some(argument) = args.get(index)
                {
                    let value = if kind == "ident" {
                        argument
                            .iter()
                            .rev()
                            .find(|t| matches!(t, TokenTree::Ident(_)))
                            .unwrap()
                            .clone()
                            .into()
                    } else {
                        argument.iter().cloned().collect()
                    };
                    bindings.insert(name.to_string(), value);
                }
            }
        }
        // Parse headers only: the facade body may contain repetitions or
        // expression metavariables irrelevant to visibility and return types.
        let tokens: Vec<_> = body.stream().into_iter().collect();
        // Type/impl templates publish associated methods, not root functions.
        saw_function |= tokens.iter().any(|token| {
            matches!(token,TokenTree::Ident(name)
            if matches!(name.to_string().as_str(),"struct"|"enum"|"impl"))
        });

        for (index, token) in tokens.iter().enumerate() {
            if !matches!(token, TokenTree::Ident(name) if name == "fn") {
                continue;
            }
            saw_function = true;
            let prefix = index.saturating_sub(4)..index;
            let Some(start) = prefix.rev().find(|&i| {
                matches!(&tokens[i], TokenTree::Ident(v) if v == "pub")
                    || matches!(&tokens[i], TokenTree::Punct(p) if p.as_char() == '$')
            }) else {
                continue;
            };
            if let (Some(TokenTree::Punct(dollar)), Some(TokenTree::Ident(name))) =
                (tokens.get(index + 1), tokens.get(index + 2))
                && dollar.as_char() == '$'
            {
                assert!(
                    bindings.contains_key(&name.to_string()),
                    "unbound public function name {name}"
                );
            }
            let end = tokens.iter().enumerate().skip(index + 1)
                .find(|(_, t)| matches!(t, TokenTree::Group(g) if g.delimiter() == proc_macro2::Delimiter::Brace))
                .map(|(i, _)| i).unwrap();
            let header: TokenStream = tokens[start..end]
                .iter()
                .cloned()
                .enumerate()
                .map(|(offset, token)| match token {
                    TokenTree::Group(g)
                        if start + offset > index
                            && g.delimiter() == proc_macro2::Delimiter::Parenthesis =>
                    {
                        TokenTree::Group(proc_macro2::Group::new(g.delimiter(), TokenStream::new()))
                    }
                    t => t,
                })
                .collect();
            let mut header = substituted(header, &bindings);
            header.extend("{}".parse::<TokenStream>().unwrap());
            let function: syn::ItemFn = syn::parse2(header.clone()).unwrap_or_else(|e| {
                panic!(
                    "unsupported public API macro {}: {header}: {e}",
                    call.mac.path.segments.last().unwrap().ident
                )
            });
            if matches!(function.vis, Visibility::Public(_)) {
                result.insert(function.sig.ident.to_string(), domains(&function.sig));
            }
        }
    }
    assert!(
        saw_function,
        "unsupported public API macro without function headers: {:?}",
        definition.ident
    );
    result
}

#[derive(Clone)]
struct Module {
    declarations: Exports,
    children: BTreeMap<String, PathBuf>,
    uses: Vec<UseTree>,
}

/// Resolve explicitly imported crate macros alongside local facade macros.
fn imported_macros(
    tree: &UseTree,
    prefix: &mut Vec<String>,
    macros: &mut BTreeMap<String, syn::ItemMacro>,
) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            imported_macros(&path.tree, prefix, macros);
            prefix.pop();
        }
        UseTree::Group(group) => {
            for tree in &group.items {
                imported_macros(tree, prefix, macros);
            }
        }
        UseTree::Name(name) if prefix.first().is_some_and(|p| p == "crate") => {
            let mut source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
            for part in &prefix[1..] {
                source.push(part);
            }
            let simple = source.with_extension("rs");
            let source = if simple.exists() {
                simple
            } else {
                source.join("mod.rs")
            };
            if source.exists() {
                let source = syn::parse_file(&std::fs::read_to_string(source).unwrap()).unwrap();
                for item in source.items {
                    if let Item::Macro(definition) = item
                        && definition.mac.path.is_ident("macro_rules")
                        && definition
                            .ident
                            .as_ref()
                            .is_some_and(|id| id == &name.ident)
                    {
                        macros.insert(name.ident.to_string(), definition);
                    }
                }
            }
        }
        _ => {}
    }
}

fn module(path: &Path) -> Module {
    thread_local! { static CACHE: std::cell::RefCell<BTreeMap<PathBuf, Module>> = const { std::cell::RefCell::new(BTreeMap::new()) }; }
    if let Some(module) = CACHE.with(|cache| cache.borrow().get(path).cloned()) {
        return module;
    }
    let file = syn::parse_file(&std::fs::read_to_string(path).unwrap()).unwrap();
    let directory = if matches!(
        path.file_name().unwrap().to_str().unwrap(),
        "lib.rs" | "mod.rs"
    ) {
        path.parent().unwrap().to_owned()
    } else {
        path.with_extension("")
    };
    let mut module = Module {
        declarations: Exports::new(),
        children: BTreeMap::new(),
        uses: Vec::new(),
    };
    let mut macros: BTreeMap<_, _> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Macro(m) if m.mac.path.is_ident("macro_rules") => {
                Some((m.ident.as_ref().unwrap().to_string(), m.clone()))
            }
            _ => None,
        })
        .collect();
    for item in &file.items {
        if let Item::Use(import) = item {
            imported_macros(&import.tree, &mut Vec::new(), &mut macros);
        }
    }
    for item in &file.items {
        if let Item::Mod(m) = item
            && m.attrs.iter().any(|a| a.path().is_ident("macro_use"))
        {
            let name = m.ident.to_string();
            let simple = directory.join(format!("{name}.rs"));
            let child = if simple.exists() {
                simple
            } else {
                directory.join(name).join("mod.rs")
            };
            let child = syn::parse_file(&std::fs::read_to_string(child).unwrap()).unwrap();
            for item in child.items {
                if let Item::Macro(m) = item
                    && m.mac.path.is_ident("macro_rules")
                {
                    macros.insert(m.ident.as_ref().unwrap().to_string(), m);
                }
            }
        }
    }
    if let Some(parent) = path.parent() {
        let parent_file = parent.with_extension("rs");
        if parent_file.exists() {
            let ancestor =
                syn::parse_file(&std::fs::read_to_string(&parent_file).unwrap()).unwrap();
            for item in ancestor.items {
                if let Item::Mod(m) = item
                    && m.attrs.iter().any(|a| a.path().is_ident("macro_use"))
                {
                    let source = parent.join(format!("{}.rs", m.ident));
                    let source =
                        syn::parse_file(&std::fs::read_to_string(source).unwrap()).unwrap();
                    for item in source.items {
                        if let Item::Macro(m) = item
                            && m.mac.path.is_ident("macro_rules")
                        {
                            macros
                                .entry(m.ident.as_ref().unwrap().to_string())
                                .or_insert(m);
                        }
                    }
                }
            }
        }
    }
    for item in &file.items {
        match item {
            Item::Fn(f) if matches!(f.vis, Visibility::Public(_)) => {
                module
                    .declarations
                    .insert(f.sig.ident.to_string(), domains(&f.sig));
            }
            Item::Mod(m) if m.content.is_none() => {
                let name = m.ident.to_string();
                let simple = directory.join(format!("{name}.rs"));
                module.children.insert(
                    name.clone(),
                    if simple.exists() {
                        simple
                    } else {
                        directory.join(name).join("mod.rs")
                    },
                );
            }
            Item::Use(u) if matches!(u.vis, Visibility::Public(_)) => {
                module.uses.push(u.tree.clone())
            }
            Item::Macro(m) if !m.mac.path.is_ident("macro_rules") => {
                let name = m.mac.path.segments.last().unwrap().ident.to_string();
                if m.mac.path.segments.len() == 3 && name == "entry" {
                    module.declarations.extend(entry_functions(m));
                    continue;
                }
                let definition = macros
                    .get(&name)
                    .unwrap_or_else(|| panic!("unhandled item macro {name} in {}", path.display()));
                module.declarations.extend(macro_functions(definition, m));
            }
            _ => {}
        }
    }
    CACHE.with(|cache| cache.borrow_mut().insert(path.to_owned(), module.clone()));
    module
}

fn resolve(path: &Path, tree: &UseTree, result: &mut Exports) {
    match tree {
        UseTree::Group(group) => {
            for item in &group.items {
                resolve(path, item, result);
            }
        }
        UseTree::Path(p) => {
            let module = module(path);
            let child = module
                .children
                .get(&p.ident.to_string())
                .unwrap_or_else(|| {
                    panic!("unknown re-export module {} in {}", p.ident, path.display())
                });
            resolve(child, &p.tree, result);
        }
        // Rust's named re-exports shadow glob imports regardless of order.
        UseTree::Glob(_) => {
            for (name, domains) in exports(path) {
                result.entry(name).or_insert(domains);
            }
        }
        UseTree::Name(name) => {
            if let Some(&real) = exports(path).get(&name.ident.to_string()) {
                result.insert(name.ident.to_string(), real);
            }
        }
        UseTree::Rename(name) => {
            if let Some(&real) = exports(path).get(&name.ident.to_string()) {
                result.insert(name.rename.to_string(), real);
            }
        }
    }
}

fn exports(path: &Path) -> Exports {
    let module = module(path);
    let mut result = module.declarations;
    for tree in &module.uses {
        resolve(path, tree, &mut result);
    }
    result
}

pub fn functions(complex: bool) -> BTreeSet<String> {
    // Non-scalar plain APIs only. Unknown shapes (including new type aliases)
    // fail closed instead of silently disappearing from either inventory.
    const OTHER: &[&str] = &[
        "checked_bernoulli",
        "checked_euler",
        "coulomb_normalization_sequence",
        "coulomb_normalization_sequence_into",
        "coulomb_sequence",
        "coulomb_sequence_into",
        "coulomb_wave",
        "coulomb_wave_sequence",
        "coulomb_wave_sequence_into",
        "kelvin",
    ];
    exports(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .into_iter()
        .filter_map(|(name, domains)| {
            assert!(
                domains.real
                    || domains.complex
                    || name.ends_with("_with_options")
                    || name.ends_with("_scoped")
                    || OTHER.contains(&name.as_str()),
                "unclassified public function {name}"
            );
            (if complex {
                domains.complex
            } else {
                domains.real
            })
            .then_some(name)
        })
        .collect()
}

#[test]
fn scalar_and_macro_inventory_distinguishes_values_from_reports_and_methods() {
    let file = syn::parse_file("pub fn value<S: RealScalar>(x: S) -> Result<S, E> {} pub fn report<S: RealScalar>(x: S) -> Result<Report<S>, E> {} impl P { pub fn method<S: RealScalar>() -> S {} } macro_rules! facade { ($name:ident) => { pub fn $name<T: NumericScalar>(x: T) -> Result<T, E> {} }; } facade!(new_function);").unwrap();
    let Item::Fn(value) = &file.items[0] else {
        panic!()
    };
    let Item::Fn(report) = &file.items[1] else {
        panic!()
    };
    assert_eq!(
        domains(&value.sig),
        ScalarDomains {
            real: true,
            complex: false
        }
    );
    assert_eq!(domains(&report.sig), ScalarDomains::default());
    let Item::Macro(definition) = &file.items[3] else {
        panic!()
    };
    let Item::Macro(call) = &file.items[4] else {
        panic!()
    };
    assert_eq!(
        macro_functions(definition, call),
        BTreeMap::from([(
            "new_function".into(),
            ScalarDomains {
                real: true,
                complex: true
            }
        )])
    );
    assert!(!BTreeSet::from(["value".to_owned()]).contains("new_function"));
}

#[test]
fn reexports_and_a_new_unregistered_function_are_detected() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("domain-inventory-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("lib.rs"),
        "mod family; pub use family::{value as renamed, generated, added}; pub use family::*;",
    )
    .unwrap();
    std::fs::write(root.join("family.rs"),"pub fn value<S: RealScalar>(x: S) -> Result<S,E> {} pub fn added<S>(x: S) -> S where S: NumericScalar {} pub fn complex<S: RealScalar>() -> Complex<S> {} macro_rules! facade { ($name:ident) => { pub fn $name<T: NumericScalar>() -> Result<T,E> {} }; } facade!(generated); impl P { pub fn method<S: RealScalar>() -> S {} }").unwrap();
    let public: BTreeSet<_> = exports(&root.join("lib.rs"))
        .into_iter()
        .filter_map(|(name, domains)| domains.real.then_some(name))
        .collect();
    assert_eq!(
        public,
        BTreeSet::from([
            "renamed".into(),
            "value".into(),
            "added".into(),
            "generated".into()
        ])
    );
    let table = BTreeSet::from(["renamed".into(), "value".into(), "generated".into()]);
    assert_eq!(
        public.difference(&table).cloned().collect::<Vec<_>>(),
        ["added"]
    );
    let complex: BTreeSet<_> = exports(&root.join("lib.rs"))
        .into_iter()
        .filter_map(|(name, domains)| domains.complex.then_some(name))
        .collect();
    assert_eq!(
        complex,
        BTreeSet::from(["added".into(), "generated".into(), "complex".into()])
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[should_panic(expected = "unsupported public API macro without function headers")]
fn forwarding_macros_fail_closed_instead_of_hiding_new_exports() {
    let file = syn::parse_file(
        "macro_rules! forward { ($name:ident) => { facade!($name); }; } forward!(future);",
    )
    .unwrap();
    let Item::Macro(definition) = &file.items[0] else {
        panic!()
    };
    let Item::Macro(call) = &file.items[1] else {
        panic!()
    };
    let _ = macro_functions(definition, call);
}

#[test]
fn real_bounds_and_named_reexports_override_complex_glob_candidates() {
    let signature: syn::ItemFn =
        syn::parse_str("pub fn value<R: RealScalar + NumericScalar<Real=R>>() -> R {}").unwrap();
    assert_eq!(
        domains(&signature.sig),
        ScalarDomains {
            real: true,
            complex: false
        }
    );
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("domain-shadow-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("named.rs"),
        "pub fn value<T: NumericScalar>() -> T {}",
    )
    .unwrap();
    std::fs::write(
        root.join("glob.rs"),
        "pub fn value<T: RealScalar>() -> T {}",
    )
    .unwrap();
    for (file, imports) in [
        ("first.rs", "pub use named::value; pub use glob::*;"),
        ("last.rs", "pub use glob::*; pub use named::value;"),
    ] {
        // Each source is a distinct root to avoid the parser cache.
        let directory = root.join(file.trim_end_matches(".rs"));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::copy(root.join("named.rs"), directory.join("named.rs")).unwrap();
        std::fs::copy(root.join("glob.rs"), directory.join("glob.rs")).unwrap();
        let source = directory.join("lib.rs");
        std::fs::write(&source, format!("mod named; mod glob; {imports}")).unwrap();
        assert_eq!(
            exports(&source)["value"],
            ScalarDomains {
                real: true,
                complex: true
            }
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn shared_entry_declarations_classify_scalar_domains_and_visibility() {
    for (signature, expected) in [
        (
            "pub value [S: RealScalar] (x: S) -> S where [];",
            Some((true, false)),
        ),
        (
            "pub value [T: NumericScalar] (x: T) -> T where [];",
            Some((true, true)),
        ),
        (
            "pub value [S: RealScalar] (x: Complex<S>) -> Complex<S> where [];",
            Some((false, true)),
        ),
        (
            "pub value [S: RealScalar] (x: S) -> [S; 4] where [];",
            Some((false, false)),
        ),
        (
            "pub(crate) value [S: RealScalar] (x: S) -> S where [];",
            None,
        ),
    ] {
        let call: syn::ItemMacro =
            syn::parse_str(&format!("crate::dispatch::entry! {{ {signature} }}")).unwrap();
        let functions = entry_functions(&call);
        assert_eq!(
            functions.get("value").map(|d| (d.real, d.complex)),
            expected
        );
    }
}

/// Rustdoc attached to every exported plain free function, including facade templates.
/// The root inventory determines coverage independently of documentation discovery.
#[allow(dead_code)]
pub fn plain_documentation() -> BTreeMap<String, String> {
    fn strings(tokens: TokenStream, output: &mut String) {
        for token in tokens {
            match token {
                TokenTree::Literal(literal) => {
                    if let Ok(value) = syn::parse_str::<syn::LitStr>(&literal.to_string()) {
                        output.push_str(&value.value());
                        output.push('\n');
                    }
                }
                TokenTree::Group(group) => strings(group.stream(), output),
                _ => {}
            }
        }
    }
    fn attributes(attrs: &[syn::Attribute]) -> String {
        let mut docs = String::new();
        for attr in attrs.iter().filter(|attr| attr.path().is_ident("doc")) {
            if let syn::Meta::NameValue(value) = &attr.meta {
                match &value.value {
                    syn::Expr::Lit(literal) => {
                        if let syn::Lit::Str(value) = &literal.lit {
                            docs.push_str(&value.value());
                            docs.push('\n');
                        }
                    }
                    syn::Expr::Macro(value) => strings(value.mac.tokens.clone(), &mut docs),
                    _ => panic!("unsupported documentation expression"),
                }
            }
        }
        docs
    }
    fn collect(path: &Path, docs: &mut BTreeMap<String, String>) {
        let source = syn::parse_file(&std::fs::read_to_string(path).unwrap()).unwrap();
        let declarations = module(path);
        for item in source.items {
            match item {
                Item::Fn(function) if matches!(function.vis, Visibility::Public(_)) => {
                    docs.insert(function.sig.ident.to_string(), attributes(&function.attrs));
                }
                Item::Macro(call) if !call.mac.path.is_ident("macro_rules") => {
                    let mut text = attributes(&call.attrs);
                    strings(call.mac.tokens.clone(), &mut text);
                    if call.mac.path.segments.last().unwrap().ident == "entry" {
                        for name in entry_functions(&call).keys() {
                            docs.insert(name.clone(), text.clone());
                        }
                    } else {
                        // Other facades supply operation-specific examples in their arguments.
                        // An unhandled template remains absent and fails the root census.
                        for name in declarations.declarations.keys() {
                            if text.contains(&format!("scioxide_special::{name}("))
                                || text.contains(&format!("scioxide_special::{name}::<"))
                            {
                                docs.insert(name.clone(), text.clone());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        for child in declarations
            .children
            .values()
            .filter(|child| child.is_file())
        {
            collect(child, docs);
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let mut docs = BTreeMap::new();
    collect(&root, &mut docs);
    exports(&root)
        .into_keys()
        .filter(|name| !name.ends_with("_with_options") && !name.ends_with("_scoped"))
        .map(|name| {
            let text = docs
                .remove(&name)
                .unwrap_or_else(|| panic!("missing rustdoc for {name}"));
            (name, text)
        })
        .collect()
}
