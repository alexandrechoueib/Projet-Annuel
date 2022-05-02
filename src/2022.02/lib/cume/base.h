/*
 * base.h
 *
 *  Created on: Jun 19, 2018
 *      Author: richer
 */

#ifndef SRC_VERSION_2018_06_CUME_BASE_H_
#define SRC_VERSION_2018_06_CUME_BASE_H_

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// The base.h header file contains macro instructions and
// template functions defined to simplify the use of CUDA API
// Instead of using the concept of DeviceToHost and HostToDevice to
// transfer data from memories (GPU/CPU) we use the concept of push
// and pop :
// - push will transfer data from host to device
// - pop will transfer data from device to host
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <cuda.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <cassert>
using namespace std;
#include "essential/import.h"

using namespace eze;

namespace ez {

namespace cume {


#define cume_assert(expr) \
		if (!(expr)) { \
			notify("CUDA assertion failed !" << #expr); \
		}

// ------------------------------------------------------------------
// definition of a macro instruction that checks if a CUDA function
// was successull or not. If the execution of the function resulted
// in some error we display it and stop the program
// ------------------------------------------------------------------
#define cume_check(value) {	\
		cudaError_t err = value; \
		if (err != cudaSuccess) {	\
			cerr << endl; \
			cerr << "============================================\n"; \
			cerr << "Error: " << cudaGetErrorString(err) << " at line "; \
			cerr << __LINE__ << " in file " <<  __FILE__;	\
			cerr <<  endl; \
			exit(EXIT_FAILURE); \
		} \
}

// ------------------------------------------------------------------
// Same as cume_check but for kernel. This macro instruction is used
// after the execution of the kernel (see the macros KERNEL_EXECUTE_NR
// and KERNEL_EXECUTE_WR in cume_kernel.h)
// ------------------------------------------------------------------
#define cume_check_kernel() { \
		cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess)  { \
			cerr << endl; \
			cerr << "============================================\n"; \
			cerr << "Kernel Error: " << cudaGetErrorString(err) << " at line "; \
			cerr << __LINE__ << " in file " <<  __FILE__;	\
			cerr <<  endl; \
			exit(EXIT_FAILURE); \
		} \
}

#define cume_new_var(var, T) {\
		cudaError_t err = cudaMalloc((void **) &var, sizeof(T)); \
		if (err != cudaSuccess)  { \
			cerr << "============================================\n"; \
			cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "; \
			cerr << __LINE__ << " in file " <<  __FILE__;	\
		} \
}

#define cume_new_array(var, T, size) {\
		cudaError_t err = cudaMalloc((void **) &var, size * sizeof(T) ); \
		if (err != cudaSuccess)  { \
			size_t sz = size*sizeof(T); \
			notify(cudaGetErrorString(err) << "\nyou tried to allocate" << sz << " bytes"); \
		} \
}

#define cume_new_array_pinned(var, T, size) {\
		cudaError_t err = cudaHostAlloc( (void **)&var, size * sizeof(T), cudaHostAllocDefault); \
		if (err != cudaSuccess)  { \
			size_t sz = size*sizeof(T); \
			notify(cudaGetErrorString(err) << "\nyou tried to allocate pinned memory of " << sz << " bytes"); \
		} \
}

#define cume_new_array_zero_copy(var, T, size) {\
		int flags = cudaHostAllocMapped|cudaHostAllocWriteCombined; ;\
		cudaError_t err = cudaHostAlloc( (void **)&var, size * sizeof(T), flags); \
		if (err != cudaSuccess)  { \
			size_t sz = size*sizeof(T); \
			notify(cudaGetErrorString(err) << "\nyou tried to allocate zero-copy memory of " << sz << " bytes"); \
		} \
}

#define cume_new_array_zero(var, T, size) {\
		cudaError_t err = cudaMalloc((void **) &var, size * sizeof(T) ); \
		if (err != cudaSuccess)  { \
			size_t sz = size*sizeof(T); \
			notify(cudaGetErrorString(err) << "\nyou tried to allocate " << sz << " bytes"); \
		} \
		err = cudaMemset(var, 0, size * sizeof(T) ); \
		if (err != cudaSuccess)  { \
			notify(cudaGetErrorString(err) << "\nyou tried to initialize the memory at address " << hex << var); \
		} \
}

#define cume_free(var) {\
		cudaError_t err = cudaFree((void **) var); \
		if (err != cudaSuccess)  { \
			notify(cudaGetErrorString(err) << "\nyou tried to deallocate the memory at address " << hex << var); \
		} \
}

#define cume_free_host(var) {\
		cudaError_t err = cudaFreeHost((void **) var); \
		if (err != cudaSuccess)  { \
			notify(cudaGetErrorString(err) << "\nyou tried to free pinned memory at address " << hex << var); \
		} \
}


/**
 * function used fill the first count bytes of the memory area
 * pointed to by x with the constant byte value value.
 * @param x pointer in device memory to an array of type T
 * @param value value to set for each byte of specified memory
 * @param size number of bytes to set
 */
#define cume_memset(var, T, size, value) {\
		cudaError_t err = cudaMemset((void **) var, value, size * sizeof(T)); \
		if (err != cudaSuccess)  { \
			notify(cudaGetErrorString(err) << "\nyou tried to set the memory at address " << hex << var); \
		} \
}

/**
 * function that transfer data from host to device memory
 * @param gpu pointer in device memory to an array of type T
 * @param cpu pointer in host memory to an array of type T
 * @param size number of elements of type T to transfer
 */
#define cume_push(gpu, cpu, T, size) {\
		cudaError_t err = cudaMemcpy(gpu, cpu, size * sizeof(T), cudaMemcpyHostToDevice); \
		if (err != cudaSuccess)  { \
			notify(cudaGetErrorString(err) << "\nyou tried to copy host memory at address " << hex << cpu << " into device memory at address " << gpu); \
		} \
}

/**
 * a push for constant symbols
 */
#define cume_push_constant(gpu_cst, cpu, T, size) {\
		cudaError_t err = cudaMemcpyToSymbol(gpu_cst, cpu, size * sizeof(T)); \
		if (err != cudaSuccess)  { \
			notify(cudaGetErrorString(err) << "\nyou tried to copy host memory at address " << hex << cpu << " into device memory at address " << gpu_cst); \
		} \
}


/**
 * function that transfer data from device to host memory
 * @param cpu pointer in host memory to an array of type T
 * @param gpu pointer in device memory to an array of type T
 * @param size number of elements of type T to transfer
 */
#define cume_pull(cpu, gpu, T, size) {\
		cudaError_t err = cudaMemcpy(cpu, gpu, size * sizeof(T), cudaMemcpyDeviceToHost); \
		if (err != cudaSuccess)  { \
			notify(cudaGetErrorString(err) << "\nyou tried to copy host memory at address " << hex << cpu << " into device memory at address " << gpu); \
		} \
}

#define cume_copy(gpu1, gpu2, T, size) {\
		cudaError_t err = cudaMemcpy(gpu1, gpu2, size * sizeof(T), cudaMemcpyDeviceToDevice); \
		if (err != cudaSuccess)  { \
			notify(cudaGetErrorString(err) << "\nyou have tried to copy device memory at address " << hex << gpu1 << " into device memory at address " << gpu2); \
		} \
}

} // end of namespace cume

} // end of namespace ez


#endif /* SRC_VERSION_2018_06_CUME_BASE_H_ */
