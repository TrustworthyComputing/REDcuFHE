# (RED)cuFHE: Evolution of FHE acceleration for Multi-GPUs
The (RED)cuFHE library is major overhaul of [cuFHE](https://github.com/vernamlab/cuFHE), and offers GPU acceleration of homomorphic operations accross multiple GPUs. The library implements the [TFHE](https://tfhe.github.io/tfhe/)
cryptosystem, with support for leveled arithmetic operations over encrypted integers, encryptions of constant values, support for an arbitrary numbers of GPUs with dynamic scheduling, as well as new and robust I/O. 

## Prerequisites
(RED)cuFHE currently supports NVIDIA GPUs; the NVIDIA Driver, NVIDIA CUDA
Toolkit and a GPU with **Compute Capability no less than 6.0** are required
(same restrictions its cuFHE predecessor). 
The library was tested with 8x T4 GPUs (Compute Capability 7.5) with
driver version 470.103.01 and CUDA Toolkit 11.4.

## Installation Instructions (Linux Only)
1. Clone the repo
2. Navigate to `redcufhe/` and run `make`
3. Move the generated shared library (`/bin/libredcufhe.so`) to a desired location and update the `LD_LIBRARY_PATH` environment variable if necessary.

## How to get started with (RED)cuFHE
We provide two programs in the `examples/` directory that showcase both leveled
arithmetic operations and gate operations on an arbitrary number of GPUs. Both
programs also include an easily extensible dynamic scheduler (which runs on a
dedicated CPU thread). The executables for these two examples will be generated
during the `make` step and can be found in `bin/`. 
