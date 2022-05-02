/**
 * 
 * Copyright (c) 2022 TrustworthyComputing - Charles Gouert
 * 
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


/**
 * @file cufhe.h
 * @brief This is the user API of the cuFHE library.
 *        It hides most of the contents in the developer API and
 *        only provides essential data structures and functions.
 */

#pragma once

#include "redcufhe.h"

namespace redcufhe {

/**
 * Call before running gates on server.
 * 1. Generate necessary NTT data.
 * 2. Convert BootstrappingKey to NTT form.
 * 3. Copy KeySwitchingKey to GPU memory.
 */
void Initialize(const PubKey& pub_key);

/** Remove everything created in Initialize(). */
void CleanUp();

inline void Synchronize() { cudaDeviceSynchronize(); };

class Stream {
public:
  inline Stream() {}
  inline Stream(int id) { Assert(id == 0); st_ = 0; }
  inline ~Stream() {}
  inline void Create() { cudaStreamCreateWithFlags(&this->st_, cudaStreamNonBlocking); }
  inline void Destroy() { cudaStreamDestroy(this->st_); }
  inline cudaStream_t st() { return st_; };
private:
  cudaStream_t st_;
}; // class Stream

// Logic Gates
void And (Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st = 0);
void Or  (Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st = 0);
void Nand(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st = 0);
void Nor (Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st = 0);
void Xor (Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st = 0);
void Xnor(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st = 0);
void Not (Ctxt& out, const Ctxt& in, Stream st = 0);
void Copy(Ctxt& out, const Ctxt& in, Stream st = 0);

// Trivial Encryption
void ConstantRed(Ctxt& result, int32_t value);

// Arithmetic Operations
void AddRed(Ctxt& out, const Ctxt& in0, const Ctxt& in1, 
Stream st = 0);
void SubRed(Ctxt& out, const Ctxt& in0, const Ctxt& in1, 
Stream st = 0);
void MulConstRed(Ctxt& prod, const Ctxt& in, uint16_t constVal);

} // namespace redcufhe
