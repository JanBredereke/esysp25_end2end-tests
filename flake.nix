{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default"; # can run on all systems
  };

  outputs = { self, nixpkgs, systems, ... }:
  let
    eachSystem = fn: nixpkgs.lib.genAttrs (import systems) (system: fn system (import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    }));
  in
  {
    devShells = eachSystem (system: pkgs: {
      default = pkgs.mkShell {
        packages = with pkgs; [
          cudatoolkit # fix "Failed to compile generated PTX with ptxas"
          (python3.withPackages (p: with p; [
            pip # to install dependencies
            jupyter # for git filter
            ipykernel # required by vscode?
          ]))
        ];

        # fix "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice"
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudatoolkit}";

        # hide tensorflow info logs
        TF_CPP_MIN_LOG_LEVEL = 1;

        # configure git to clean jupyter notebook files when staging them
        # redirect stderr to /dev/null to suppress irrelevant errors when using starship
        shellHook = ''
          git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR 2> /dev/null'
        '';
      };
    });
  };
}