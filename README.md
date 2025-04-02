# phenotypic_heterogeneity_ibm
 
Code and data for the preprint "Identifiability of heterogeneous phenotype adaptation from low-cell-count experiments and a stochastic model" available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.08.19.608540v1).

## Getting started

First, download or clone this repository. The `Project.toml` and `Manifest.toml` contain the dependencies and a Julia environment that can be used to reproduce the results. To activate this environment and install all required dependencies, run the following code from the `Pkg` REPL (press `]` to enter the `Pkg` REPL)
```
(v1.11) pkg> activate .
(phenotypic_heterogeneity_ibm) pkg> instantiate
```

## Reproducing the results

Code used to reproduce all figures is in the `figures` folder. For example, to reproduce Figure 3 of the main text, run `Figures/fig3.jl`. The output will be an `svg` file in the main directory.