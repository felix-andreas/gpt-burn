{
  inputs = {
    # base
    systems.url = "github:nix-systems/default";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # extra
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
      # see: https://github.com/NixOS/nix/issues/5790
      inputs.flake-utils.inputs.systems.follows = "systems";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
      # see: https://github.com/NixOS/nix/issues/5790
      inputs.flake-utils.inputs.systems.follows = "systems";
    };
  };

  outputs =
    { self
      # base
    , systems
    , nixpkgs
      # extra
    , crane
    , devshell
    , rust-overlay
    } @ inputs:
    let
      l = inputs.nixpkgs.lib // builtins;
      fs = l.fileset;
      eachSystem = fn: l.genAttrs (import inputs.systems) fn;
      flake = (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              inputs.devshell.overlays.default
              (import inputs.rust-overlay)
            ];
          };
          nativeDeps = with pkgs;[
            pkg-config
            stdenv.cc.cc.lib # for burn dataset loader (sqlite)
            zlib # for burn dataset loader (sqlite)
            openssl.dev # for burn
            vulkan-headers # for burn
            vulkan-loader # for burn
            vulkan-tools # for burn
          ];
          rust-toolchain = pkgs.rust-bin.selectLatestNightlyWith
            (toolchain: toolchain.default.override {
              extensions = [ "rust-src" "rust-analyzer" ];
            });
          craneLib = (crane.mkLib pkgs).overrideToolchain rust-toolchain;
          rustFiles = fs.fileFilter (file: file.hasExt "rs") ./.;
          cargoFiles = fs.unions [
            (fs.fileFilter (file: file.name == "Cargo.toml" || file.name == "Cargo.lock") ./.)
          ];
          commonArgs = {
            pname = "crate";
            # we don't want to use our version tag here because it would force a rebuild for every commit
            version = "0.1";
            src = fs.toSource {
              root = ./.;
              fileset = cargoFiles;
            };
            nativeBuildInputs = with pkgs; (
              nativeDeps
              ++ l.optionals (pkgs.stdenv.isLinux) [ clang ]
            );
          };
          crateDepsOnly = craneLib.buildDepsOnly (commonArgs // {
            cargoCheckCommandcargo = "check --profile release --all-targets --all-features";
          });
          crateClippy = craneLib.cargoClippy (commonArgs // {
            cargoArtifacts = crateDepsOnly;
            cargoClippyExtraArgs = "--all-targets --all-features -- --deny warnings";
            src = fs.toSource {
              root = ./.;
              fileset = fs.unions ([
                cargoFiles
                rustFiles
              ]);
            };
          });
        in
        {
          devShell = pkgs.devshell.mkShell {
            motd = "";
            packages = with pkgs; [
              # Rust
              bacon
              cargo-expand
              cargo-sort
              evcxr
              rust-toolchain
              # Python
              (python311.withPackages (p: with p; [ black httpx ipykernel ipython isort matplotlib numpy pytorch tqdm ]))
            ]
            ++ nativeDeps;

            env = pkgs.lib.lists.optionals pkgs.stdenv.isLinux [
              {
                name = "LD_LIBRARY_PATH";
                prefix = "$DEVSHELL_DIR/lib";
              }
              {
                name = "PKG_CONFIG_PATH";
                prefix = "$DEVSHELL_DIR/lib/pkgconfig";
              }
            ];
          };
          check = crateClippy;
          package = craneLib.buildPackage (commonArgs // {
            pname = "gpt-burn";
            cargoArtifacts = crateClippy;
            src = fs.toSource {
              root = ./.;
              fileset = fs.unions ([
                cargoFiles
                rustFiles
              ]);
            };
          });
        });
    in
    {
      checks = eachSystem (system: { default = (flake system).check; });
      devShells = eachSystem (system: { default = (flake system).devShell; });
      packages = eachSystem (system: { default = (flake system).package; });
    };
}
