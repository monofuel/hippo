{
  description = "Hippo NVIDIA/CUDA development environment";

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
          cudatoolkit
          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cudart
          linuxPackages.nvidia_x11
          nim
          nim-atlas
          nimble
          tbb
        ];

        shellHook = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
          export EXTRA_CFLAGS="-I${pkgs.cudatoolkit}/include"
          export EXTRA_LDFLAGS="-L${pkgs.cudatoolkit}/lib -L${pkgs.cudaPackages.cuda_cudart}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          echo "🚀 NVIDIA/CUDA development environment ready"
        '';
      };
    };
} 