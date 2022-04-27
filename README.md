# REDcuFHE: FHE for Multi-GPUs
This library is an evolution of the [cuFHE](https://github.com/vernamlab/cuFHE) library, which implements the [TFHE](https://tfhe.github.io/tfhe/)
cryptosystem on a single GPU. Currently, REDcuFHE
supports leveled arithmetic operations over encrypted integers, encryptions of constant values, support for arbitrary numbers of GPUs,
and more robust I/O. 

## Prerequisites
REDcuFHE is only implemented for NVIDIA GPUs; NVIDIA Driver, NVIDIA CUDA
Toolkit and a GPU with **Compute Capability no less than 6.0** are required
(same restrictions as cuFHE). 
This library was tested with 8x T4 GPUs (Compute Capability 7.5) with
driver version 470.103.01 and CUDA Toolkit 11.4.

## Installation Instructions (Linux Only)
1. Clone the repo
2. Navigate to `redcufhe/` and run `make`
3. Move the generated shared library (`/bin/libredcufhe.so`) to a desired location and update the `LD_LIBRARY_PATH` environment variable if necessary.

## Examples
We provide two programs in the `examples/` directory that showcase both leveled
arithmetic operations and gate operations on an arbitrary number of GPUs. Both
programs also include an easily extensible dynamic scheduler (which runs on a
dedicated CPU thread). The executables for these two examples will be generated
during the `make` step and can be found in `bin/`. 