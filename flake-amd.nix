{
  description = "Hippo AMD/ROCm development environment";

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
          rocm-opencl-icd
          rocm-opencl-runtime
          hip
          rocm-device-libs
          rocm-runtime
          rocm-thunk
          nim
          nim-atlas
          nimble
          tbb
        ];

        shellHook = ''
          export ROCM_PATH=${pkgs.rocm-runtime}
          export HIP_PATH=${pkgs.hip}
          export LD_LIBRARY_PATH=${pkgs.rocm-runtime}/lib:${pkgs.hip}/lib:$LD_LIBRARY_PATH
          export EXTRA_CFLAGS="-I${pkgs.hip}/include -I${pkgs.rocm-runtime}/include"
          export EXTRA_LDFLAGS="-L${pkgs.rocm-runtime}/lib -L${pkgs.hip}/lib"
          echo "🔥 AMD/ROCm development environment ready"
        '';
      };
    };
} 