//! Stable-toolchain source inventory shared with the comparison catalog test.
//!
//! Public declarations are inventoried across the declared production modules.
//! `unreachable_pub` makes these the crate's exported surface. Local API macros
//! are instantiated at their call sites; unsupported matchers fail explicitly.

#![allow(dead_code, clippy::panic, clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use proc_macro2::{Group, TokenStream, TokenTree};
use quote::{ToTokens, quote};
use syn::{Attribute, ImplItem, Item, Type, Visibility};

type Bindings = BTreeMap<String, TokenStream>;

#[derive(Default)]
pub struct Surface {
    pub records: BTreeSet<String>,
    pub names: BTreeSet<String>,
    macros: BTreeMap<String, TokenStream>,
    implementations: Vec<(String, String, String)>,
    owners: BTreeMap<String, String>,
    exports: BTreeMap<String, String>,
    public_contexts: BTreeSet<String>,
    local_traits: BTreeSet<String>,
}

fn public(vis: &Visibility) -> bool {
    matches!(vis, Visibility::Public(_))
}

fn production(attrs: &[Attribute]) -> bool {
    !attrs.iter().any(|a| {
        a.path().is_ident("cfg")
            && (a.meta.to_token_stream().to_string() == "cfg (test)"
                || a.meta.to_token_stream().to_string().contains("all (test"))
    })
}

fn attributes(attrs: &[Attribute]) -> TokenStream {
    attrs
        .iter()
        .filter(|a| !a.path().is_ident("doc"))
        .flat_map(ToTokens::to_token_stream)
        .collect()
}

fn type_name(ty: &Type) -> String {
    match ty {
        Type::Path(p) => p.path.segments.last().unwrap().ident.to_string(),
        _ => ty.to_token_stream().to_string(),
    }
}

// Match the ordinary metavariables and optional groups used by the crate's
// declaration macros. Expression captures end at the next literal separator.
fn bind(pattern: TokenStream, input: TokenStream, bindings: &mut Bindings) -> bool {
    let pattern: Vec<_> = pattern.into_iter().collect();
    let input: Vec<_> = input.into_iter().collect();
    let (mut p, mut i) = (0, 0);
    while p < pattern.len() {
        if pattern[p].to_string() == "$" {
            if let Some(TokenTree::Group(group)) = pattern.get(p + 1) {
                assert_eq!(
                    pattern.get(p + 2).map(ToString::to_string).as_deref(),
                    Some("?"),
                    "unsupported repetition: {pattern:?}"
                );
                if i < input.len() {
                    return bind(
                        group.stream(),
                        input[i..].iter().cloned().collect(),
                        bindings,
                    );
                }
                p += 3;
                continue;
            }
            let name = pattern[p + 1].to_string();
            assert_eq!(pattern[p + 2].to_string(), ":");
            let kind = pattern[p + 3].to_string();
            if i == input.len() {
                return false;
            }
            let end = match kind.as_str() {
                "ident" | "literal" => i + 1,
                "expr" | "ty" | "path" => {
                    if let Some(separator) = pattern.get(p + 4) {
                        i + input[i..]
                            .iter()
                            .position(|t| t.to_string() == separator.to_string())
                            .unwrap_or(input.len() - i)
                    } else {
                        input.len()
                    }
                }
                _ => panic!("unsupported public macro capture: {kind}"),
            };
            bindings.insert(name, input[i..end].iter().cloned().collect());
            i = end;
            p += 4;
        } else {
            if input.get(i).map(ToString::to_string) != Some(pattern[p].to_string()) {
                return false;
            }
            p += 1;
            i += 1;
        }
    }
    i == input.len()
}

fn substitute(tokens: TokenStream, bindings: &Bindings) -> Option<TokenStream> {
    let mut output = TokenStream::new();
    let mut tokens = tokens.into_iter();
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(p) if p.as_char() == '$' => match tokens.next().unwrap() {
                TokenTree::Ident(name) => output.extend(bindings.get(&name.to_string())?.clone()),
                TokenTree::Group(group) => {
                    assert_eq!(tokens.next().unwrap().to_string(), "?");
                    if let Some(body) = substitute(group.stream(), bindings) {
                        output.extend(body);
                    }
                }
                token => panic!("unsupported substitution: {token}"),
            },
            TokenTree::Group(group) => output.extend([TokenTree::Group(Group::new(
                group.delimiter(),
                substitute(group.stream(), bindings)?,
            ))]),
            token => output.extend([token]),
        }
    }
    Some(output)
}

impl Surface {
    pub fn snapshot(&self) -> String {
        self.records.iter().cloned().collect::<Vec<_>>().join("\n") + "\n"
    }

    fn record(&mut self, name: &str, tokens: TokenStream, context: &str) {
        self.names.insert(name.to_owned());
        let record = format!("{context} {name}: {tokens}");
        self.owners
            .insert(record.clone(), name.split("::").next().unwrap().to_owned());
        self.records.insert(record);
        if self.public_contexts.contains(context) && !name.contains("::") {
            self.exports.insert(name.to_owned(), name.to_owned());
        }
    }

    fn expansion(&self, invocation: &syn::Macro) -> Option<TokenStream> {
        let name = invocation.path.segments.last().unwrap().ident.to_string();
        let definition = self.macros.get(&name)?;
        let tokens: Vec<_> = definition.clone().into_iter().collect();
        for rule in tokens.windows(4) {
            if let (
                TokenTree::Group(pattern),
                TokenTree::Punct(eq),
                TokenTree::Punct(gt),
                TokenTree::Group(body),
            ) = (&rule[0], &rule[1], &rule[2], &rule[3])
                && eq.as_char() == '='
                && gt.as_char() == '>'
            {
                let mut bindings = Bindings::new();
                if bind(pattern.stream(), invocation.tokens.clone(), &mut bindings) {
                    return Some(
                        substitute(body.stream(), &bindings)
                            .unwrap_or_else(|| panic!("unbound public macro variable in {name}")),
                    );
                }
            }
        }
        panic!("no supported rule for {name}!({})", invocation.tokens);
    }

    fn members(&mut self, items: &[ImplItem], owner: &str, context: &str) {
        for item in items {
            match item {
                ImplItem::Fn(f) if public(&f.vis) && production(&f.attrs) => {
                    let attrs = attributes(&f.attrs);
                    let sig = &f.sig;
                    self.record(
                        &format!("{owner}::{}", sig.ident),
                        quote!(#attrs #sig),
                        context,
                    );
                }
                ImplItem::Const(c) if public(&c.vis) => {
                    let attrs = attributes(&c.attrs);
                    let ty = &c.ty;
                    self.record(
                        &format!("{owner}::{}", c.ident),
                        quote!(#attrs const #ty),
                        context,
                    );
                }
                ImplItem::Type(t) if public(&t.vis) => self.record(
                    &format!("{owner}::{}", t.ident),
                    t.to_token_stream(),
                    context,
                ),
                ImplItem::Macro(m) => {
                    if let Some(expansion) = self.expansion(&m.mac) {
                        let wrapper: syn::ItemImpl =
                            syn::parse2(quote!(impl Placeholder { #expansion })).unwrap();
                        self.members(&wrapper.items, owner, context);
                    }
                }
                _ => {}
            }
        }
    }

    fn items(&mut self, items: &[Item], directory: &Path, context: &str) {
        for item in items {
            if let Item::Trait(t) = item {
                self.local_traits.insert(t.ident.to_string());
            }
            if let Item::Macro(m) = item
                && m.mac.path.is_ident("macro_rules")
            {
                if m.attrs.iter().any(|a| a.path().is_ident("macro_export")) {
                    let attrs = attributes(&m.attrs);
                    let definition = &m.mac;
                    self.record(
                        &m.ident.as_ref().unwrap().to_string(),
                        quote!(#attrs #definition),
                        "crate",
                    );
                }
                self.macros
                    .insert(m.ident.as_ref().unwrap().to_string(), m.mac.tokens.clone());
            }
        }
        for item in items {
            match item {
                Item::Mod(m) if production(&m.attrs) => {
                    let attrs = attributes(&m.attrs);
                    let child_context = format!("{context}::{name} {attrs}", name = m.ident);
                    if public(&m.vis) {
                        self.record(&m.ident.to_string(), quote!(pub mod), context);
                        if self.public_contexts.contains(context) {
                            self.public_contexts.insert(child_context.clone());
                        }
                    }
                    if let Some((_, items)) = &m.content {
                        self.items(items, &directory.join(m.ident.to_string()), &child_context);
                    } else {
                        let explicit = m.attrs.iter().find_map(|a| {
                            if a.path().is_ident("path") {
                                if let syn::Meta::NameValue(v) = &a.meta
                                    && let syn::Expr::Lit(lit) = &v.value
                                    && let syn::Lit::Str(path) = &lit.lit
                                {
                                    return Some(directory.join(path.value()));
                                }
                                None
                            } else {
                                None
                            }
                        });
                        let file = explicit.unwrap_or_else(|| {
                            let direct = directory.join(format!("{}.rs", m.ident));
                            if direct.is_file() {
                                direct
                            } else {
                                directory.join(m.ident.to_string()).join("mod.rs")
                            }
                        });
                        self.file(&file, &child_context);
                    }
                }
                Item::Struct(s) if public(&s.vis) && production(&s.attrs) => {
                    let attrs = attributes(&s.attrs);
                    let generics = &s.generics;
                    let constraints = &s.generics.where_clause;
                    self.record(
                        &s.ident.to_string(),
                        quote!(#attrs struct #generics #constraints),
                        context,
                    );
                    for (index, field) in
                        s.fields.iter().enumerate().filter(|(_, f)| public(&f.vis))
                    {
                        let attrs = attributes(&field.attrs);
                        let ty = &field.ty;
                        let name = field
                            .ident
                            .as_ref()
                            .map_or_else(|| index.to_string(), ToString::to_string);
                        self.record(&format!("{}::{name}", s.ident), quote!(#attrs #ty), context);
                    }
                }
                Item::Enum(e) if public(&e.vis) && production(&e.attrs) => {
                    let attrs = attributes(&e.attrs);
                    let generics = &e.generics;
                    let constraints = &e.generics.where_clause;
                    self.record(
                        &e.ident.to_string(),
                        quote!(#attrs enum #generics #constraints),
                        context,
                    );
                    for variant in &e.variants {
                        let mut variant = variant.clone();
                        variant.attrs.retain(|a| !a.path().is_ident("doc"));
                        for field in &mut variant.fields {
                            field.attrs.retain(|a| !a.path().is_ident("doc"));
                        }
                        self.record(
                            &format!("{}::{}", e.ident, variant.ident),
                            variant.to_token_stream(),
                            context,
                        );
                    }
                }
                Item::Fn(f) if public(&f.vis) && production(&f.attrs) => {
                    let attrs = attributes(&f.attrs);
                    let sig = &f.sig;
                    self.record(&sig.ident.to_string(), quote!(#attrs #sig), context);
                }
                Item::Trait(t) if public(&t.vis) && production(&t.attrs) => {
                    let mut t = t.clone();
                    t.attrs.retain(|a| !a.path().is_ident("doc"));
                    for member in &mut t.items {
                        if let syn::TraitItem::Fn(f) = member {
                            f.attrs.retain(|a| !a.path().is_ident("doc"));
                            f.default = None;
                        }
                    }
                    self.record(&t.ident.to_string(), t.to_token_stream(), context);
                }
                Item::Type(t) if public(&t.vis) && production(&t.attrs) => {
                    let mut t = t.clone();
                    t.attrs.retain(|a| !a.path().is_ident("doc"));
                    self.record(&t.ident.to_string(), t.to_token_stream(), context);
                }
                Item::Const(c) if public(&c.vis) && production(&c.attrs) => {
                    let attrs = attributes(&c.attrs);
                    let ty = &c.ty;
                    self.record(&c.ident.to_string(), quote!(#attrs const #ty), context);
                }
                Item::Static(s) if public(&s.vis) && production(&s.attrs) => {
                    let attrs = attributes(&s.attrs);
                    let ty = &s.ty;
                    let mutability = &s.mutability;
                    self.record(
                        &s.ident.to_string(),
                        quote!(#attrs static #mutability #ty),
                        context,
                    );
                }
                Item::Union(u) if public(&u.vis) && production(&u.attrs) => {
                    let mut u = u.clone();
                    u.attrs.retain(|a| !a.path().is_ident("doc"));
                    u.fields.named = u
                        .fields
                        .named
                        .into_iter()
                        .filter(|f| public(&f.vis))
                        .map(|mut f| {
                            f.attrs.retain(|a| !a.path().is_ident("doc"));
                            f
                        })
                        .collect();
                    self.record(&u.ident.to_string(), u.to_token_stream(), context);
                }
                Item::ExternCrate(e) if public(&e.vis) => {
                    self.record(&e.ident.to_string(), e.to_token_stream(), context)
                }
                Item::ForeignMod(f) => assert!(
                    !f.items.iter().any(|item| match item {
                        syn::ForeignItem::Fn(f) => public(&f.vis),
                        syn::ForeignItem::Static(s) => public(&s.vis),
                        syn::ForeignItem::Type(t) => public(&t.vis),
                        _ => false,
                    }),
                    "public foreign declarations need explicit inventory support"
                ),
                Item::Use(u) if public(&u.vis) && production(&u.attrs) => {
                    let attrs = attributes(&u.attrs);
                    let tree = &u.tree;
                    if self.public_contexts.contains(context) {
                        self.records.insert(format!(
                            "{context}: {attrs} pub use {tree_tokens}",
                            tree_tokens = tree.to_token_stream()
                        ));
                        fn exports(tree: &syn::UseTree, output: &mut BTreeMap<String, String>) {
                            match tree {
                                syn::UseTree::Name(n) => {
                                    output.insert(n.ident.to_string(), n.ident.to_string());
                                }
                                syn::UseTree::Rename(n) => {
                                    output.insert(n.rename.to_string(), n.ident.to_string());
                                }
                                syn::UseTree::Path(p) => exports(&p.tree, output),
                                syn::UseTree::Group(g) => {
                                    for item in &g.items {
                                        exports(item, output);
                                    }
                                }
                                syn::UseTree::Glob(_) => panic!("root exports must be explicit"),
                            }
                        }
                        exports(tree, &mut self.exports);
                    }
                }
                Item::Impl(i) if production(&i.attrs) => {
                    let owner = type_name(&i.self_ty);
                    let attrs = attributes(&i.attrs);
                    let generics = &i.generics;
                    let ty = &i.self_ty;
                    let constraints = &i.generics.where_clause;
                    let impl_context = format!(
                        "{context} {}",
                        quote!(#attrs impl #generics #ty #constraints)
                    );
                    if let Some((_, tr, _)) = &i.trait_ {
                        self.implementations.push((
                            owner,
                            tr.segments.last().unwrap().ident.to_string(),
                            format!(
                                "{impl_context}: {tr_tokens}",
                                tr_tokens = tr.to_token_stream()
                            ),
                        ));
                    } else {
                        self.members(&i.items, &owner, &impl_context);
                    }
                }
                Item::Macro(m) if !m.mac.path.is_ident("macro_rules") && production(&m.attrs) => {
                    if let Some(expansion) = self.expansion(&m.mac) {
                        let file: syn::File = syn::parse2(expansion).unwrap();
                        self.items(&file.items, directory, context);
                    } else {
                        panic!("unhandled item macro: {}", m.mac.path.to_token_stream());
                    }
                }
                _ => {}
            }
        }
    }

    fn file(&mut self, file: &Path, context: &str) {
        let parsed = syn::parse_file(&std::fs::read_to_string(file).unwrap()).unwrap();
        let directory = if matches!(
            file.file_name().unwrap().to_str(),
            Some("lib.rs" | "mod.rs")
        ) {
            file.parent().unwrap().to_path_buf()
        } else {
            file.with_extension("")
        };
        self.items(&parsed.items, &directory, context);
    }

    fn finish(&mut self) {
        let exported: BTreeSet<_> = self.exports.values().cloned().collect();
        self.records.retain(|record| {
            self.owners
                .get(record)
                .is_none_or(|owner| exported.contains(owner))
        });
        self.names
            .retain(|name| exported.contains(name.split("::").next().unwrap()));
        for (alias, original) in &self.exports {
            assert!(
                self.names.contains(original),
                "unresolved public export: {original}"
            );
            if alias != original {
                let methods: Vec<_> = self
                    .names
                    .iter()
                    .filter_map(|name| {
                        name.strip_prefix(&format!("{original}::"))
                            .map(|method| format!("{alias}::{method}"))
                    })
                    .collect();
                self.names.extend(methods);
                self.names.insert(alias.clone());
            }
        }
        for (owner, tr, record) in &self.implementations {
            if exported.contains(owner)
                && (!self.local_traits.contains(tr) || exported.contains(tr))
            {
                self.records.insert(record.clone());
            }
        }
    }

    pub fn from_source(source: &str) -> Self {
        let mut surface = Self::default();
        surface.public_contexts.insert("crate".to_owned());
        surface.items(
            &syn::parse_file(source).unwrap().items,
            Path::new("."),
            "crate",
        );
        surface.finish();
        surface
    }
}

pub fn linalg() -> Surface {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = root
        .ancestors()
        .find(|p| p.join("crates/scioxide-linalg/src/lib.rs").is_file())
        .unwrap();
    let mut surface = Surface::default();
    surface.public_contexts.insert("crate".to_owned());
    fn definitions(directory: &Path, surface: &mut Surface) {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                if path.file_name().unwrap() != "tests" {
                    definitions(&path, surface);
                }
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let file = syn::parse_file(&std::fs::read_to_string(path).unwrap()).unwrap();
                for item in file.items {
                    if let Item::Macro(m) = item
                        && m.mac.path.is_ident("macro_rules")
                    {
                        surface
                            .macros
                            .insert(m.ident.unwrap().to_string(), m.mac.tokens);
                    }
                }
            }
        }
    }
    definitions(&workspace.join("crates/scioxide-linalg/src"), &mut surface);
    surface.file(
        &workspace.join("crates/scioxide-linalg/src/lib.rs"),
        "crate",
    );
    surface.finish();
    surface
}
