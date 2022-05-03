/**
 * Copyright (c) 2022 TrustworthyComputing - Charles Gouert
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

//
// Multi-GPU Bootstrapped Gates example
//
// This example illustrates scaling gate
// evaluations to multiple GPUs and can
// achieve much higher throughput than
// baseline cuFHE.
//

#include <include/redcufhe_gpu.cuh>
#include <include/details/error_gpu.cuh>

using namespace redcufhe;

#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <utility>
#include <vector>
#include <math.h>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
using namespace std;
using namespace std::chrono;

PriKey pri_key;
uint32_t kNumTests;
PubKey bk;

// shared vector used to issue/receive commands
vector<vector<pair<int, int>>> requests;


void NandCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = 1 - in0.message_ * in1.message_;
}

void setup(uint32_t kNumSMs, Ctxt** inputs, Ptxt** pt, Stream** st, int idx) {
  cudaSetDevice(idx);

  // send bootstrapping key to GPU
  Initialize(bk);

  // create CUDA streams for the GPU
  st[idx] = new Stream[kNumSMs];
  for (int i = 0; i < kNumSMs; i++) {
    st[idx][i].Create();
  }
  Synchronize();

  // Allocate memory for ciphertexts and encrypt
  (*inputs) = new Ctxt[2 * kNumTests];
  for (int i = 0; i < 2 * kNumTests; i++) {
    Encrypt((*inputs)[i], pt[idx][i], pri_key);
  }
  Synchronize();
  return;
}

// Runs on a worker CPU thread controlling a GPU
void server(int shares, uint32_t kNumSMs, int idx, Ctxt** answers, Stream** st) {
  while(1) {
    for (int i = 0; i < shares; i++) {
      // check for assignment
      if (requests[idx][i].first != -1) {
        // terminate upon kill signal (-2)
        if (requests[idx][i].first == -2) {
          Synchronize();
          return;
        }
        // Perform gate computations
        Nand((*answers)[requests[idx][i].second], (*answers)[requests[idx][i].second], (*answers)[requests[idx][i].first], st[idx][i % kNumSMs]);
        // clear assignment
        requests[idx][i].first = -1;
        requests[idx][i].second = -1;
      }
    }
  }
}

int main() {
  srand(time(NULL));

  // get GPU stats (WARNING: assumes all GPUs have the same number of SMs)
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t kNumSMs = prop.multiProcessorCount;
  kNumTests = kNumSMs*kNumSMs*8;

  // get number of available GPUs
  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);

  // create 2D array of plaintext and streams
  Ptxt* pt[numGPUs];
  Stream* st[numGPUs];

  // generate keyset
  SetSeed();
  PriKeyGen(pri_key);
  PubKeyGen(bk, pri_key);

  for (int i = 0; i < numGPUs; i++) {
  // generate random ptxts (bits) for each GPU
    pt[i] = new Ptxt[2 * kNumTests];
    for (int j = 0; j < 2 * kNumTests; j++) {
      pt[i][j] = rand() % Ptxt::kPtxtSpace;
    }
  }

  // Initialize shared vector for thread communication
  int num_threads = numGPUs;
  requests.resize(num_threads);
  for (int i = 0; i < num_threads; i++) {
    requests[i].resize(kNumTests);
    for (int j = 0; j < kNumTests; j++) {
      // each element holds indices of data array
      requests[i][j] = make_pair(-1,-1);
    }
  }

  Ctxt* answers[numGPUs];
  omp_set_num_threads(numGPUs);

  // Initialize data on each available GPU
  #pragma omp parallel for shared(st, answers)
  for (int i = 0; i < numGPUs; i++) {
    setup(kNumSMs, &answers[i], pt, st, i);
  }

  // one worker thread for each GPU and a scheduler thread
  omp_set_num_threads(numGPUs+1);


  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  /////////////////////////////////////////
  //
  // (RED)cuFHE Dynamic Scheduler
  // Enables automatic allocation of FHE
  // workloads to multiple GPUs
  //
  /////////////////////////////////////////
  #pragma omp parallel for shared(answers, st, requests)
  for (int i = 0; i < (num_threads+1); i++) {
    if (i != 0) { // workers
      int thread_id = omp_get_thread_num() - 1;
      cudaSetDevice(thread_id);
      server(kNumTests, kNumSMs, thread_id, &answers[i-1], st);
      Synchronize();
    }
    else { // master thread
      int turn = 1; // indicates target worker
      for (int j = 0; j < (kNumTests*numGPUs); j++) {
        if ((j % kNumTests == 0) && (j > 0)) {
          turn++; // assign to next worker
          if (turn > num_threads) { // excludes scheduler
            turn = 1;
          }
        }
        // assign input 1 as index j of GPU array
        requests[turn-1][j % kNumTests].second = j % (kNumTests);
        // assign input 2 as index j+kNumTests
        requests[turn-1][j % kNumTests].first = ((j%kNumTests)+kNumTests) % (2*kNumTests);
      }
      // check to see if all threads are done
      bool end = false;
      while (end == false) {
        end = true;
        for (int j = 0; j < num_threads; j++) {
          for (int k = 0; k < kNumTests; k++) {
            if (requests[j][k].first != -1) {
              end = false;
              break;
            }
          }
        }
      }
      // terminate workers
      for (int j = 0; j < num_threads; j++) {
        for (int k = 0; k < kNumTests; k++) {
          requests[j][k].first = -2;
        }
      }
    }
  }

  cout << "Gate evals: " << kNumTests*numGPUs << endl;

  // Confirm results and check for errors
  int wrong_counter[numGPUs];
  omp_set_num_threads(numGPUs);
  #pragma omp parallel shared(wrong_counter)
  {
    Ptxt* recovered_pt = new Ptxt[kNumTests];
    int thread_num = omp_get_thread_num();
    cudaSetDevice(thread_num);
    for (int i = 0; i < kNumTests; i++) {
      NandCheck(pt[thread_num][i], pt[thread_num][i+kNumTests], pt[thread_num][i]);
      Decrypt(recovered_pt[i], answers[thread_num][i+kNumTests], pri_key);
    }
    wrong_counter[thread_num] = 0;
    for (int i = 0; i < kNumTests; i++) {
      if (pt[thread_num][i+kNumTests].message_ != recovered_pt[i].message_) {
        wrong_counter[thread_num]++;
      }
    }
    delete [] recovered_pt;
  }

  for (int i = 0; i < numGPUs; i++) {
    cout << "GPU #" << i << " errors: " << wrong_counter[i] << endl;
  }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "Time: " << time_span.count() << " seconds" << endl;

  for (int i = 0; i < numGPUs; i++) {
    delete [] pt[i];
  }
  // free GPU memory
  CleanUp();

  return 0;
}
