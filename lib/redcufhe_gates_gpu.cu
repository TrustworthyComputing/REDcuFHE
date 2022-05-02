/**
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

#include <include/redcufhe.h>
#include <include/redcufhe_gpu.cuh>
#include <include/redcufhe_bootstrap_gpu.cuh>

namespace redcufhe {

void Initialize(const PubKey& pub_key) {
  BootstrappingKeyToNTT(pub_key.bk_);
  KeySwitchingKeyToDevice(pub_key.ksk_);
}

void CleanUp() {
  DeleteBootstrappingKeyNTT();
  DeleteKeySwitchingKey();
}

inline void CtxtCopyH2D(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_device_->data(),
                  c.lwe_sample_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyHostToDevice,
                  st.st());
}

inline void CtxtCopyD2H(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_->data(),
                  c.lwe_sample_device_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyDeviceToHost,
                  st.st());
}

void Nand(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  NandBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Or(Ctxt& out,
        const Ctxt& in0,
        const Ctxt& in1,
        Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  OrBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void And(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  AndBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Nor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  NorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Xor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 4);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  XorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Xnor(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 4);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  XnorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void Not(Ctxt& out,
         const Ctxt& in,
         Stream st) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = -in.lwe_sample_->data()[i];
}

void Copy(Ctxt& out,
          const Ctxt& in,
          Stream st) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = in.lwe_sample_->data()[i];
}

void NoiselessTrivial(Ctxt& result, Torus mu){
  const int32_t n = result.lwe_sample_->n();

  for (int32_t i = 0; i < n; ++i) result.lwe_sample_->a()[i] = 0;
  result.lwe_sample_->b() = mu;
}

void ConstantRed(Ctxt& result, int32_t value) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  NoiselessTrivial(result, value ? mu : -mu);
}

__global__
void AddOp(Torus* out, Torus* in0, Torus* in1, uint32_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = in0[i] + in1[i];
  }
}

void AddRed(Ctxt& out, const Ctxt& in0, const Ctxt& in1, 
Stream st) {
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  int numBlocks = (in0.lwe_sample_->n() + 512 - 1)/512;
  AddOp<<<numBlocks,512,0,st.st()>>>(out.lwe_sample_device_->data(), in0.lwe_sample_device_->data(), in1.lwe_sample_device_->data(), in0.lwe_sample_device_->n());
  CtxtCopyD2H(out, st);
}

__global__
void SubOp(Torus* out, Torus* in0, Torus* in1, uint32_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = in0[i] - in1[i];
  }
}

void SubRed(Ctxt& out, const Ctxt& in0, const Ctxt& in1, 
Stream st) {
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  int numBlocks = (in0.lwe_sample_->n() + 512 - 1)/512;
  SubOp<<<numBlocks,512,0,st.st()>>>(out.lwe_sample_device_->data(), in0.lwe_sample_device_->data(), in1.lwe_sample_device_->data(), out.lwe_sample_device_->n());
  CtxtCopyD2H(out, st);
}

void MulConstRed(Ctxt& prod, const Ctxt& in, uint16_t constVal) {
  for (int i = 0; i < in.lwe_sample_->n(); i++) {
    prod.lwe_sample_->data()[i] = in.lwe_sample_->data()[i]*constVal;
  }
  prod.lwe_sample_->b() = in.lwe_sample_->b()*constVal;
}

} // namespace redcufhe
