# (RED)cuFHE: Evolution of FHE acceleration for Multi-GPUs  <a href="https://github.com/TrustworthyComputing/REDcuFHE/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a> </h1>
The (RED)cuFHE library is major overhaul of [cuFHE](https://github.com/vernamlab/cuFHE), and offers scalable GPU acceleration of homomorphic operations across multiple GPUs. The library implements the [TFHE](https://tfhe.github.io/tfhe/)
cryptosystem, with support for leveled arithmetic operations over encrypted integers, encryptions of constant values, support for an arbitrary numbers of GPUs with dynamic scheduling, as well as new and robust I/O.

## Prerequisites
(RED)cuFHE currently supports NVIDIA GPUs; the NVIDIA Driver, NVIDIA CUDA
Toolkit and a GPU with **Compute Capability no less than 6.0** are required
(same restrictions its cuFHE predecessor).
The library was tested with 8x T4 GPUs (Compute Capability 7.5) with
driver version 470.103.01 and CUDA Toolkit 11.4.

## Installation Instructions (Linux Only)
1. Clone the repo:  `git clone https://github.com/TrustworthyComputing/REDcuFHE.git`
2. _(Optional)_ Update `REDcuFHE/lib/redcufhe_bootstrap_gpu.cu:33` with the maximum GPUs of the target system (default is 8)
3. Navigate to root directory: `cd REDcuFHE`
4. Build the library: `make`
5. Move the generated shared library to a desired location (e.g. `sudo cp
   /bin/libredcufhe.so /usr/local/lib/`) and update the `LD_LIBRARY_PATH`
   environment variable if necessary (e.g., `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`).

## (RED)cuFHE API

### Key Generation
```c++
SetSeed(); // seed the RNG
PriKey sk;
PubKey pk;
KeyGen(pk, sk); // generate public and private keys
```

### Enc/Dec Boolean Functions
```c++
Ctxt ct;
Ptxt pt; // Ptxt objects can encode binary values
pt = 1;
Encrypt(ct, pt, sk);
Decrypt(pt, ct, sk);
ConstantRed(ct_out, pt); // Trivial encryption using pk
```


### Encrypted Logic Gates
```c++
// stream_id is a handle to a CUDA stream
And(ct_out, ct_in[0], ct_in[1], stream_id);
Nand(ct_out, ct_in[0], ct_in[1], stream_id);
Or(ct_out, ct_in[0], ct_in[1], stream_id);
Nor(ct_out, ct_in[0], ct_in[1], stream_id);
Xor(ct_out, ct_in[0], ct_in[1], stream_id);
Xnor(ct_out, ct_in[0], ct_in[1], stream_id);
Not(ct_out, ct_in[0], stream_id);
Copy(ct_out, ct_in[0], stream_id);
```

### Enc/Dec Integer Functions
```c++
Ctxt ct;
int32_t int_pt = 127; // ptxt element
uint32_t pt_mod = 512; // all ptxt are modulo pt_mod
EncryptIntRed(ct, int_pt, pt_mod, sk);
DecryptIntRed(pt, ct, pt_mod, sk);
```

### Encrypted Integer Operations
```c++
AddRed(ct_out, ct_in[0], ct_in[1], stream_id);
SubRed(ct_out, ct_in[0], ct_in[1], stream_id);
MulConstRed(ct_out, ct_in[0], pt);
```

### I/O Operations
```c++
WritePriKeyToFile(sk, filename);
ReadPriKeyFromFile(sk, filename);
WritePubKeyToFile(pk, filename);
ReadPubKeyFromFile(pk, filename);
WriteCtxtToFileRed(ct[i], filename); // append ciphertext i to a file; supports multiple ciphertexts in the same file
ReadCtxtFromFileRed(ct, filestream); // read next ciphertext from an ifstream object; supports multiple ciphertexts in the same file
```

## Performance Comparison

|   Library  | Gate Cost | Speedup (relative to CPU) |
|:----------:|:---------:|:-------------------------:|
| (RED)cuFHE |  0.048 ms |            227X           |
|    cuFHE  |   0.37 ms  |            29X            |
|    TFHE*   |   10.9 ms   |             1X            |

All experiments performed on a g4dn.metal AWS instance.
*Uses SPQLIOS-FMA as the FFT engine.

## How to get started with (RED)cuFHE
We provide [two programs](examples/) that showcase both leveled
arithmetic operations and gate operations on an arbitrary number of GPUs. Both
programs also incorporate our easily extensible **Dynamic Scheduler** (which runs on a
dedicated CPU thread). Each available GPU is controlled by a CPU thread which
receives orders from the scheduler and issues orders to its partner GPU in turn.
The executables for these two examples will be generated
during the `make` step and can be found in `bin/`.

### [Example 1](examples/multigpu_gates_example.cu): Multi-GPU Bootstrapped Gates
This example illustrates scaling gate
evaluations to multiple GPUs and can achieve much higher throughput than
baseline cuFHE.
### [Example 2](examples/multigpu_arithmetic_example.cu): Multi-GPU Leveled Arithmetic Operations
This example highlights the new leveled features in the integer domain. Instead
of operating over encrypted bits, which is required in cuFHE, (RED)cuFHE allows
for operations over encrypted modular integers.

## Cite this work
This library was introduced in the REDsec paper, which presents a framework for
privacy-preserving neural network inference ([Cryptology ePrint
Archive](https://eprint.iacr.org/2021/1100.pdf)):
```
@misc{folkerts2021redsec,
    author       = {Lars Folkerts and Charles Gouert and Nektarios Georgios Tsoutsos},
    title        = {{REDsec: Running Encrypted Discretized Neural Networks in Seconds}},
    howpublished = {Cryptology ePrint Archive, Report 2021/1100},
    year         = {2021},
    note         = {\url{https://ia.cr/2021/1100}},
}
```

<p align="center">
    <img src="./logos/twc.png" height="20%" width="20%">
</p>
<h4 align="center">Trustworthy Computing Group</h4>
