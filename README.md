# Stochastic representation of wavefunctions
This repository demonstrates a simple prototype implementation of an algorithm
that performs imaginary time propagation with a stochastic representation of
real-space many body quantum wavefunctions.

By default, it will attempt to find the ground state energy of two
noninteracting fermions in two spatial dimensions, allowing comparison with the
analytically known exact result. As of 2022, it should run in a couple of
minutes on a desktop machine with a consumer graphics card. However, the code
can easily be modified for application to any number of particles or potential.

## Instructions
First, clone the code from GitHub.
```bash
git clone git@github.com:HristianaAtanasova/stochastic_wavefunctions_public.git
```
Then, assuming Docker is installed, a convenient way to get it running is to
work within a tensorflow container. First, build the container:
```bash
docker build . -t stoc_wf
```
Then, run the demonstration:
```bash
docker run --gpus all -it --rm -v $PWD:/workspace -w /workspace stoc_wf python ./run.py
```

The parameters of the system can be adjusted in the `params.py` file, and
running `python plot_results.py` will show a convergence plot for energy.
