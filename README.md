# Stochastic representation of wavefunctions
This repository demonstrates an algorithm that performs imaginary time
propagation with a stochastic representation of real-space many body quantum
wavefunctions.

## Instructions
First, clone the code from GitHub.
```bash
git clone git@github.com:HristianaAtanasova/stochastic_wavefunctions_public.git
```
Then, assuming Docker is installed, a convenient way to get it running is to
work within a tensorflow container:
```bash
docker run --gpus all -it -v $PWD:/workspace -w /workspace --name naughty_feynman tensorflow/tensorflow:latest-gpu bash
```

The parameters of the system can be adjusted in the `params.py` file. To run
the imaginary time propagation call `run.py`.
