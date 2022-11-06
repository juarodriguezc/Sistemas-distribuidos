/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

// std::system includes

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>

int *pArgc = NULL;
char **pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device)
{
       CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

       if (CUDA_SUCCESS != error)
       {
              fprintf(
                  stderr,
                  "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
                  error, __FILE__, __LINE__);

              exit(EXIT_FAILURE);
       }
}

#endif /* CUDART_VERSION < 5000 */

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
       pArgc = &argc;
       pArgv = argv;

       int deviceCount = 0;
       cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

       // This function call returns 0 if there are no CUDA capable devices.
       if (deviceCount == 0)
       {
              printf("-1");
              exit(EXIT_SUCCESS);
       }

       int dev = 0, driverVersion = 0, runtimeVersion = 0;

       cudaSetDevice(dev);
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, dev);

       // Console log
       cudaDriverGetVersion(&driverVersion);
       cudaRuntimeGetVersion(&runtimeVersion);

       printf("%03d_%03d_%s", deviceProp.multiProcessorCount, _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor), deviceProp.name);
       // finish
       exit(EXIT_SUCCESS);
}
