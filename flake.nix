{
  description = "SciOxide development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rustToolchain = pkgs.rust-bin.stable."1.95.0".default.override {
          targets = [
            "i686-unknown-linux-gnu"
            "aarch64-unknown-linux-gnu"
            "aarch64-unknown-none"
            "wasm32-unknown-unknown"
          ];
          extensions = [
            "rust-src"
            "rust-analyzer"
            "rustfmt"
            "clippy"
          ];
        };

        nodejs = pkgs.nodejs_24;
        pnpm = pkgs.pnpm_11.overrideAttrs (_oldAttrs: {
          version = "11.2.2";
          src = pkgs.fetchurl {
            url = "https://registry.npmjs.org/pnpm/-/pnpm-11.2.2.tgz";
            hash = "sha256-mcS+gx7SMYKYlRQtlnk9vnWvxTeVkzrtg2bmjczh4bg=";
          };
        });
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain
            nodejs
            pkgs.pkg-config
            pkgs.stdenv.cc
          ];
        };
        devShells.msrv = pkgs.mkShell {
          packages = [
            (pkgs.rust-bin.stable."1.93.0".minimal.override {
              targets = [ "aarch64-unknown-none" ];
            })
            pkgs.pkg-config
            pkgs.stdenv.cc
          ];
        };
        devShells.ci = pkgs.mkShell {
          packages = [
            pkgs.actionlint
            pkgs.cargo-audit
            nodejs
            pnpm
          ];
        };
        # Handbook authoring under docs/handbook (Oxiquill needs Node 24).
        devShells.handbook = pkgs.mkShell {
          packages = [
            rustToolchain
            pkgs.git
            nodejs
            pnpm
            # Oxiquill uses wasm-opt from PATH for release optimization and its cache;
            # it downloads and verifies the pinned wasm-bindgen CLI itself.
            pkgs.binaryen
          ];
          OXIQUILL_NODE = nixpkgs.lib.getExe nodejs;
        };
        devShells.i686 = pkgs.mkShell {
          packages = [ rustToolchain pkgs.pkgsCross.gnu32.stdenv.cc ];
          CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_LINKER = "i686-unknown-linux-gnu-cc";
        };
      }
    );
}
