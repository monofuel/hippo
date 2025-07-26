{
  description = "Hippo development environments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system} = {
        # Default to basic CPU-only environment
        default = self.devShells.${system}.basic;

        # Unified development environment with NVIDIA/CUDA + AMD/ROCm + Basic support
        all = pkgs.mkShell {
          buildInputs = with pkgs; [
            # NVIDIA/CUDA packages
            cudatoolkit
            cudaPackages.cuda_nvcc
            cudaPackages.cuda_cudart
            linuxPackages.nvidia_x11
            
            # AMD/ROCm packages
            rocm-opencl-icd
            rocm-opencl-runtime
            hip
            rocm-device-libs
            rocm-runtime
            rocm-thunk
            
            # Basic Nim development
            nim
            nim-atlas
            nimble
            tbb
          ];

          shellHook = ''
            # NVIDIA/CUDA environment setup
            export CUDA_PATH=${pkgs.cudatoolkit}
            
            # AMD/ROCm environment setup
            export ROCM_PATH=${pkgs.rocm-runtime}
            export HIP_PATH=${pkgs.hip}
            
            # Combined library paths
            export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.rocm-runtime}/lib:${pkgs.hip}/lib:$LD_LIBRARY_PATH
            
            # Combined compiler flags
            export EXTRA_CFLAGS="-I${pkgs.cudatoolkit}/include -I${pkgs.hip}/include -I${pkgs.rocm-runtime}/include"
            export EXTRA_LDFLAGS="-L${pkgs.cudatoolkit}/lib -L${pkgs.cudaPackages.cuda_cudart}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib -L${pkgs.rocm-runtime}/lib -L${pkgs.hip}/lib"
            
            echo "🚀🔥⚡ Unified development environment ready (NVIDIA/CUDA + AMD/ROCm + Basic Nim)"
          '';
        };

        # NVIDIA/CUDA development only
        nvidia = pkgs.mkShell {
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

        # AMD/ROCm development only
        amd = pkgs.mkShell {
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

        # Basic CPU-only development
        basic = pkgs.mkShell {
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
    };
} 