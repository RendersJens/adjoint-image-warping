#include <stdio.h>

#include <warpKernelsAffine.cu>
#include <warpAlgorithmsAffine.hu>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void affineBackwardWarp2D(const float* f,
                          const float* A,
                          const float* b,
                          float* fWarped,
                          int degree,
                          int width,
                          int height){

    size_t size = width*height * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_A, *d_b, *d_fWarped;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_A, 4 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_fWarped, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 4 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + 15)/16, (height + 15)/16);
    if(degree==1){
        affineLinearBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
                                                                         d_A,
                                                                         d_b,
                                                                         d_fWarped,
                                                                         width,
                                                                         height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        // affineCubicBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
        //                                                                 d_A,
        //                                                                 d_b,
        //                                                                 d_fWarped,
        //                                                                 width,
        //                                                                 height);
        // gpuErrchk(cudaPeekAtLastError());
        // gpuErrchk(cudaDeviceSynchronize());
        float coeffs[] = {
            #include "cubic_2D_coefficients.inc"
        };
        float *d_coeffs;
        gpuErrchk(cudaMalloc(&d_coeffs, 16*16*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 16*16*sizeof(float), cudaMemcpyHostToDevice));
        affineCubicBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
                                                                        d_A,
                                                                        d_b,
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
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_fWarped);
}


void affineBackwardWarp3D(const float* f,
                          const float* A,
                          const float* b,
                          float* fWarped,
                          int degree,
                          int width,
                          int height,
                          int depth){

    size_t size = width*height*depth * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_A, *d_b, *d_fWarped;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_A, 9 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_fWarped, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 3 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire image
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((width + 7)/8, (height + 7)/8, (depth + 7)/8);
    if(degree==1){
        affineLinearBackwardWarp3DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
                                                                         d_A,
                                                                         d_b,
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
        affineCubicBackwardWarp3DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
                                                                        d_A,
                                                                        d_b,
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
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_fWarped);
}


void adjointAffineBackwardWarp2D(const float* fWarped,
                                 const float* A,
                                 const float* b,
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
    float *d_fWarped, *d_A, *d_b, *d_f;
    gpuErrchk(cudaMalloc(&d_fWarped, size));
    gpuErrchk(cudaMalloc(&d_A, 4 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_f, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 4 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + 15)/16, (height + 15)/16);
    if(degree==1){
        adjointAffineLinearBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_fWarped,
                                                                                d_A,
                                                                                d_b,
                                                                                d_f,
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
        adjointAffineCubicBackwardWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_fWarped,
                                                                               d_A,
                                                                               d_b,
                                                                               d_f,
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
    gpuErrchk(cudaMemcpy(f, d_f, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_fWarped);
}


void adjointAffineBackwardWarp3D(const float* fWarped,
                                 const float* A,
                                 const float* b,
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
    float *d_fWarped, *d_A, *d_b, *d_f;
    gpuErrchk(cudaMalloc(&d_fWarped, size));
    gpuErrchk(cudaMalloc(&d_A, 9 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_f, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 3 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((width  + 8-1)/8,
                   (height + 8-1)/8,
                   (depth  + 8-1)/8);

    if(degree==1){
        adjointAffineLinearBackwardWarp3DKernel<<<numBlocks, threadsPerBlock>>>(d_fWarped,
                                                                                d_A,
                                                                                d_b,
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
        adjointAffineCubicBackwardWarp3DKernel<<<numBlocks, threadsPerBlock>>>(d_fWarped,
                                                                               d_A,
                                                                               d_b,
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
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_fWarped);
}