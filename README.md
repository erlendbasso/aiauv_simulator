# AIAUV Simulator

1. Clone repo

    ```
    git clone https://github.com/erlendbasso/aiauv_simulator.git --recursive
    ```

2. Run simulator

    ```
    cargo run --release
    ```

## Vendored dependency: `ode_solvers`

`vendor/ode_solvers/` is a vendored copy of [`ode_solvers`](https://github.com/srenevey/ode-solvers)
`0.6.1` (Apache-2.0). The only change from upstream is bumping its `nalgebra`
dependency from `0.33` to `0.34` (and `simba` `0.9` → `0.10`); the source is
unmodified. This is required because `multibody_dynamics` `0.4` needs `nalgebra`
`0.34`, while no published `ode_solvers` release supports `0.34` yet. Once
upstream `ode_solvers` updates to `nalgebra` `0.34`, this vendored copy can be
dropped in favour of the crates.io release.