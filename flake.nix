{
  description = "Hippo basic development environment (CPU only)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          nim
          nim-atlas
          nimble
          tbb
        ];

        shellHook = ''
          echo "⚡ Basic Nim development environment ready (CPU only)"
        '';
      };
    };
} 