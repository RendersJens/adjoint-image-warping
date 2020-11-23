#include <stdio.h>

#include <warpKernels.cu>
#include <warpAlgorithms.hu>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void backwardWarp2D(const float* f,
                    const float* u,
                    const float* v,
                    float* fWarped,
                    int degree,
                    int width,
                    int height){

    size_t size = width*height * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_fWarped;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_fWarped, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + 15)/16, (height + 15)/16);
    if(degree==1){
        linearBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
                                                                   d_u,
                                                                   d_v,
                                                                   d_fWarped,
                                                                   width,
                                                                   height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        float coeffs[] = {
            #include "cubic_2D_coefficients.inc"
        };
        float *d_coeffs;
        gpuErrchk(cudaMalloc(&d_coeffs, 16*16*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 16*16*sizeof(float), cudaMemcpyHostToDevice));
        cubicBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
                                                                  d_u,
                                                                  d_v,
                                                                  d_fWarped,
                                                                  width,
                                                                  height,
                                                                  d_coeffs);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_coeffs);
    }else{
        throw "Only degree 1 and 3 are implemented";
    }

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(fWarped, d_fWarped, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_fWarped);
}


void backwardWarp3D(const float* f,
                    const float* u,
                    const float* v,
                    const float* w,
                    float* fWarped,
                    int degree,
                    int width,
                    int height,
                    int depth){

    size_t size = width*height*depth * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_w, *d_fWarped;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_w, size));
    gpuErrchk(cudaMalloc(&d_fWarped, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire image
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((width + 7)/8, (height + 7)/8, (depth + 7)/8);
    if(degree==1){
        linearBackwardWarp3DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
                                                                   d_u,
                                                                   d_v,
                                                                   d_w,
                                                                   d_fWarped,
                                                                   width,
                                                                   height,
                                                                   depth);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        float coeffs[] = {
            #include "cubic_3D_coefficients.inc"
        };
        float *d_coeffs;
        gpuErrchk(cudaMalloc(&d_coeffs, 64*64*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 64*64*sizeof(float), cudaMemcpyHostToDevice));
        cubicBackwardWarp3DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
                                                                  d_u,
                                                                  d_v,
                                                                  d_w,
                                                                  d_fWarped,
                                                                  width,
                                                                  height,
                                                                  depth,
                                                                  d_coeffs);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_coeffs);
    }else{
        throw "Only degree 1 and 3 are implemented";
    }

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(fWarped, d_fWarped, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_fWarped);
}


void adjointBackwardWarp2D(const float* fWarped,
                           const float* u,
                           const float* v,
                           float* f,
                           int degree,
                           int width,
                           int height){

    /*
    GPU implementation of 2D adjoint backward image warping along the DVF (u,v)
    with rectangular multivariate spline interpolation
    */


    size_t size = width*height * sizeof(float);

    // allocate vectors in device memory
    float *d_fWarped, *d_u, *d_v, *d_f;
    gpuErrchk(cudaMalloc(&d_fWarped, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_f, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + 15)/16, (height + 15)/16);
    if(degree==1){
        adjointLinearBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_fWarped,
                                                                          d_u,
                                                                          d_v,
                                                                          d_f,
                                                                          width,
                                                                          height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        adjointCubicBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_fWarped,
                                                                         d_u,
                                                                         d_v,
                                                                         d_f,
                                                                         width,
                                                                         height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else{
        throw "Only degree 1 and 3 are implemented";
    }

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(f, d_f, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_fWarped);
}


void adjointBackwardWarp3D(const float* fWarped,
                           const float* u,
                           const float* v,
                           const float* w,
                           float* f,
                           int degree,
                           int width,
                           int height,
                           int depth){
    /*
    GPU implementation of 3D adjoint backward image warping along the DVF (u,v,w)
    with rectangular multivariate spline interpolation
    */

    size_t size = width*height*depth * sizeof(float);

    // allocate vectors in device memory
    float *d_fWarped, *d_u, *d_v, *d_w, *d_f;
    gpuErrchk(cudaMalloc(&d_fWarped, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_w, size));
    gpuErrchk(cudaMalloc(&d_f, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((width  + 8-1)/8,
                   (height + 8-1)/8,
                   (depth  + 8-1)/8);

    if(degree==1){
        adjointLinearBackwardWarp3DKernel<<<numBlocks, threadsPerBlock>>>(d_fWarped,
                                                                          d_u,
                                                                          d_v,
                                                                          d_w,
                                                                          d_f,
                                                                          width,
                                                                          height,
                                                                          depth);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        float coeffs[] = {
            #include "cubic_3D_coefficients.inc"
        };
        float *d_coeffs;
        gpuErrchk(cudaMalloc(&d_coeffs, 64*64*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 64*64*sizeof(float), cudaMemcpyHostToDevice));
        adjointCubicBackwardWarp3DKernel<<<numBlocks, threadsPerBlock>>>(d_fWarped,
                                                                         d_u,
                                                                         d_v,
                                                                         d_w,
                                                                         d_f,
                                                                         width,
                                                                         height,
                                                                         depth,
                                                                         d_coeffs);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_coeffs);
    }else{
        throw "Only degree 1 and 3 are implemented";
    }
    // copy the result back to the host
    gpuErrchk(cudaMemcpy(f, d_f, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_fWarped);
}