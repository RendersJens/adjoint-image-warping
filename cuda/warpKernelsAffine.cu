__global__ void affineLinearBackwardWarp2DKernel(const float* f,
                                                 const float* A,
                                                 const float* b,
                                                 float* fWarped,
                                                 int width,
                                                 int height){
    
    /*
    Kernel of GPU implementation of 2D backward image warping along the DVF (u,v)
    with linear interpolation (rectangular multivariate spline)
    */

   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width){

        // position at which to iterpolate
        float x = A[0]*j + A[1]*i + b[0];
        float y = A[2]*j + A[3]*i + b[1];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int Q[][2] = {{y1, x1},
                      {y1, x2},
                      {y2, x1},
                      {y2, x2}};

        // interpolation coefficients
        float coefficients[] = {(x2 - x)*(y2 - y),
                                (x - x1)*(y2 - y),
                                (x2 - x)*(y - y1),
                                (x - x1)*(y - y1)};

        for(int k = 0; k < 4; k++){
            if(0 <= Q[k][0] && Q[k][0] < height
            && 0 <= Q[k][1] && Q[k][1] < width){
                fWarped[i*width + j] += coefficients[k] * f[Q[k][0]*width + Q[k][1]];
            }
        }
    }
}


__global__ void adjointAffineLinearBackwardWarp2DKernel(const float* fWarped,
                                                        const float* A,
                                                        const float* b,
                                                        float* f,
                                                        int width,
                                                        int height){

    /*
    Kernel of GPU implementation of 2D adjoint backward image warping along the
    DVF (u,v) with linear interpolation (rectangular multivariate spline)
    */

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width){

        // position at which to iterpolate
        float x = A[0]*j + A[1]*i + b[0];
        float y = A[2]*j + A[3]*i + b[1];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int Q[][2] = {{y1, x1},
                      {y1, x2},
                      {y2, x1},
                      {y2, x2}};

        // interpolation coefficients
        float coefficients[] = {(x2 - x)*(y2 - y),
                                (x - x1)*(y2 - y),
                                (x2 - x)*(y - y1),
                                (x - x1)*(y - y1)};

        for(int k = 0; k < 4; k++){
            if(0 <= Q[k][0] && Q[k][0] < height
            && 0 <= Q[k][1] && Q[k][1] < width){
                atomicAdd(&f[Q[k][0]*width + Q[k][1]], coefficients[k] * fWarped[i*width + j]);
            }
        }
    }
}


__global__ void affineLinearBackwardWarp3DKernel(const float* f,
                                                 const float* A,
                                                 const float* b,
                                                 float* fWarped,
                                                 int width,
                                                 int height,
                                                 int depth){
    
    /*
    Kernel of GPU implementation of 3D backward image warping along the DVF (u,v)
    with linear interpolation (rectangular multivariate spline)
    */

    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width && h < depth){

        // position at which to iterpolate
        float x = A[0]*j + A[1]*i + A[2]*h + b[0];
        float y = A[3]*j + A[4]*i + A[5]*h + b[1];
        float z = A[6]*j + A[7]*i + A[8]*h + b[2];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int z2 = z1 + 1;
        int Q[][3] = {{x1, y1, z1},
                      {x2, y1, z1},
                      {x1, y2, z1},
                      {x2, y2, z1},
                      {x1, y1, z2},
                      {x2, y1, z2},
                      {x1, y2, z2},
                      {x2, y2, z2}};

        // interpolation coefficients
        float coefficients[] = {(x2 - x)*(y2 - y)*(z2 - z),
                                (x - x1)*(y2 - y)*(z2 - z),
                                (x2 - x)*(y - y1)*(z2 - z),
                                (x - x1)*(y - y1)*(z2 - z),
                                (x2 - x)*(y2 - y)*(z - z1),
                                (x - x1)*(y2 - y)*(z - z1),
                                (x2 - x)*(y - y1)*(z - z1),
                                (x - x1)*(y - y1)*(z - z1)};

        for(int k = 0; k < 8; k++){
            if(0 <= Q[k][0] && Q[k][0] < width
            && 0 <= Q[k][1] && Q[k][1] < height
            && 0 <= Q[k][2] && Q[k][2] < depth){
                fWarped[h*width*height + i*width + j] += coefficients[k] * f[Q[k][2]*width*height + Q[k][1]*width + Q[k][0]];
            }
        }
    }
}


__global__ void adjointAffineLinearBackwardWarp3DKernel(const float* fWarped,
                                                        const float* A,
                                                        const float* b,
                                                        float* f,
                                                        int width,
                                                        int height,
                                                        int depth){
    
    /*
    Kernel of GPU implementation of 3D adjoint backward image warping along the
    DVF (u,v) with linear interpolation (rectangular multivariate spline)
    */

    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width && h < depth){

        // position at which to iterpolate
        float x = A[0]*j + A[1]*i + A[2]*h + b[0];
        float y = A[3]*j + A[4]*i + A[5]*h + b[1];
        float z = A[6]*j + A[7]*i + A[8]*h + b[2];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int z2 = z1 + 1;
        int Q[][3] = {{x1, y1, z1},
                      {x2, y1, z1},
                      {x1, y2, z1},
                      {x2, y2, z1},
                      {x1, y1, z2},
                      {x2, y1, z2},
                      {x1, y2, z2},
                      {x2, y2, z2}};

        // interpolation coefficients
        float coefficients[] = {(x2 - x)*(y2 - y)*(z2 - z),
                                (x - x1)*(y2 - y)*(z2 - z),
                                (x2 - x)*(y - y1)*(z2 - z),
                                (x - x1)*(y - y1)*(z2 - z),
                                (x2 - x)*(y2 - y)*(z - z1),
                                (x - x1)*(y2 - y)*(z - z1),
                                (x2 - x)*(y - y1)*(z - z1),
                                (x - x1)*(y - y1)*(z - z1)};

        for(int k = 0; k < 8; k++){
            if(0 <= Q[k][0] && Q[k][0] < width
            && 0 <= Q[k][1] && Q[k][1] < height
            && 0 <= Q[k][2] && Q[k][2] < depth){
                atomicAdd(&f[Q[k][2]*width*height + Q[k][1]*width + Q[k][0]], coefficients[k] * fWarped[h*width*height + i*width + j]);
            }
        }
    }
}


__global__ void affineCubicBackwardWarp2DKernel(const float* f,
                                                const float* A,
                                                const float* b,
                                                float* fWarped,
                                                int width,
                                                int height,
                                                const float* coeffs){


    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width){

        // position at which to iterpolate
        float x = A[0]*j + A[1]*i + b[0];
        float y = A[2]*j + A[3]*i + b[1];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        // xi = x1 - 1 + i

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float monomials[] = {1, ymy1, ymy1_2, ymy1_3, xmx1, xmx1*ymy1, xmx1*ymy1_2, xmx1*ymy1_3, xmx1_2, xmx1_2*ymy1, xmx1_2*ymy1_2, xmx1_2*ymy1_3, xmx1_3, xmx1_3*ymy1, xmx1_3*ymy1_2, xmx1_3*ymy1_3};

        int k = 0;
        for(int ii = 0; ii < 4; ii++){
            for(int jj = 0; jj < 4; jj++){
                int Q0 = x1 + jj - 1;
                int Q1 = y1 + ii - 1;
                if(0 <= Q0 && Q0 < width
                && 0 <= Q1 && Q1 < height){
                    float coefficient = 0;
                    for(int n = 0; n < 16; n++){
                        coefficient += coeffs[k*16 + n] * monomials[n];
                    }
                    fWarped[i*width + j] += coefficient * f[Q1*width + Q0];
                }
                k++;
            }
        }
    }
}


__global__ void adjointAffineCubicBackwardWarp2DKernel(const float* fWarped,
                                                       const float* A,
                                                       const float* b,
                                                       float* f,
                                                       int width,
                                                       int height){

    /*
    Kernel of GPU implementation of adjoint backward image warping along the
    DVF (u,v) with cubic interpolation (rectangular multivariate catmull-rom spline)
    */

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width){

        // position at which to iterpolate
        float x = A[0]*j + A[1]*i + b[0];
        float y = A[2]*j + A[3]*i + b[1];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int x0 = x1 - 1;
        int y0 = y1 - 1;
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int x3 = x1 + 2;
        int y3 = y1 + 2;
        int Q[][2] = {{y0, x0},{y0, x1},{y0, x2},{y0, x3},
                      {y1, x0},{y1, x1},{y1, x2},{y1, x3},
                      {y2, x0},{y2, x1},{y2, x2},{y2, x3},
                      {y3, x0},{y3, x1},{y3, x2},{y3, x3}};

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float coefficients[] = { 0.25f*xmx1_3*ymy1_3 - 0.5f*xmx1_3*ymy1_2 - 0.5f*xmx1_2*ymy1_3 + 0.25f*xmx1_3*ymy1 + xmx1_2*ymy1_2 + 0.25f*xmx1*ymy1_3 - 0.5f*xmx1_2*ymy1 - 0.5f*xmx1*ymy1_2 + 0.25f*xmx1*ymy1,
                                -0.75f*xmx1_3*ymy1_3 + 1.5f*xmx1_3*ymy1_2 + 1.25f*xmx1_2*ymy1_3 - 0.75f*xmx1_3*ymy1 - 2.5f*xmx1_2*ymy1_2 + 1.25f*xmx1_2*ymy1 - 0.5f*ymy1_3 + ymy1_2 - 0.5f*ymy1,
                                 0.75f*xmx1_3*ymy1_3 - 1.5f*xmx1_3*ymy1_2 - xmx1_2*ymy1_3 + 0.75f*xmx1_3*ymy1 + 2*xmx1_2*ymy1_2 - 0.25f*xmx1*ymy1_3 - xmx1_2*ymy1 + 0.5f*xmx1*ymy1_2 - 0.25f*xmx1*ymy1,
                                -0.25f*xmx1_3*ymy1_3 + 0.5f*xmx1_3*ymy1_2 + 0.25f*xmx1_2*ymy1_3 - 0.25f*xmx1_3*ymy1 - 0.5f*xmx1_2*ymy1_2 + 0.25f*xmx1_2*ymy1,
                                -0.75f*xmx1_3*ymy1_3 + 1.25f*xmx1_3*ymy1_2 + 1.5f*xmx1_2*ymy1_3 - 2.5f*xmx1_2*ymy1_2 - 0.75f*xmx1*ymy1_3 - 0.5f*xmx1_3 + 1.25f*xmx1*ymy1_2 + xmx1_2 - 0.5f*xmx1,
                                 2.25f*xmx1_3*ymy1_3 - 3.75f*xmx1_3*ymy1_2 - 3.75f*xmx1_2*ymy1_3 + 6.25f*xmx1_2*ymy1_2 + 1.5f*xmx1_3 + 1.5f*ymy1_3 - 2.5f*xmx1_2 - 2.5f*ymy1_2 + 1,
                                -2.25f*xmx1_3*ymy1_3 + 3.75f*xmx1_3*ymy1_2 + 3*xmx1_2*ymy1_3 - 5*xmx1_2*ymy1_2 + 0.75f*xmx1*ymy1_3 - 1.5f*xmx1_3 - 1.25f*xmx1*ymy1_2 + 2*xmx1_2 + 0.5f*xmx1,
                                 0.75f*xmx1_3*ymy1_3 - 1.25f*xmx1_3*ymy1_2 - 0.75f*xmx1_2*ymy1_3 + 1.25f*xmx1_2*ymy1_2 + 0.5f*xmx1_3 - 0.5f*xmx1_2,
                                 0.75f*xmx1_3*ymy1_3 - xmx1_3*ymy1_2 - 1.5f*xmx1_2*ymy1_3 - 0.25f*xmx1_3*ymy1 + 2*xmx1_2*ymy1_2 + 0.75f*xmx1*ymy1_3 + 0.5f*xmx1_2*ymy1 - xmx1*ymy1_2 - 0.25f*xmx1*ymy1,
                                -2.25f*xmx1_3*ymy1_3 + 3*xmx1_3*ymy1_2 + 3.75f*xmx1_2*ymy1_3 + 0.75f*xmx1_3*ymy1 - 5*xmx1_2*ymy1_2 - 1.25f*xmx1_2*ymy1 - 1.5f*ymy1_3 + 2*ymy1_2 + 0.5f*ymy1,
                                 2.25f*xmx1_3*ymy1_3 - 3*xmx1_3*ymy1_2 - 3*xmx1_2*ymy1_3 - 0.75f*xmx1_3*ymy1 + 4*xmx1_2*ymy1_2 - 0.75f*xmx1*ymy1_3 + xmx1_2*ymy1 + xmx1*ymy1_2 + 0.25f*xmx1*ymy1,
                                -0.75f*xmx1_3*ymy1_3 + xmx1_3*ymy1_2 + 0.75f*xmx1_2*ymy1_3 + 0.25f*xmx1_3*ymy1 - xmx1_2*ymy1_2 - 0.25f*xmx1_2*ymy1,
                                -0.25f*xmx1_3*ymy1_3 + 0.25f*xmx1_3*ymy1_2 + 0.5f*xmx1_2*ymy1_3 - 0.5f*xmx1_2*ymy1_2 - 0.25f*xmx1*ymy1_3 + 0.25f*xmx1*ymy1_2,
                                 0.75f*xmx1_3*ymy1_3 - 0.75f*xmx1_3*ymy1_2 - 1.25f*xmx1_2*ymy1_3 + 1.25f*xmx1_2*ymy1_2 + 0.5f*ymy1_3 - 0.5f*ymy1_2,
                                -0.75f*xmx1_3*ymy1_3 + 0.75f*xmx1_3*ymy1_2 + xmx1_2*ymy1_3 - xmx1_2*ymy1_2 + 0.25f*xmx1*ymy1_3 - 0.25f*xmx1*ymy1_2,
                                 0.25f*xmx1_3*ymy1_3 - 0.25f*xmx1_3*ymy1_2 - 0.25f*xmx1_2*ymy1_3 + 0.25f*xmx1_2*ymy1_2};

        for(int k = 0; k < 16; k++){
            if(0 <= Q[k][0] && Q[k][0] < height
            && 0 <= Q[k][1] && Q[k][1] < width){
                atomicAdd(&f[Q[k][0]*width + Q[k][1]], coefficients[k] * fWarped[i*width + j]);
            }
        }
    }
}


__global__ void affineCubicBackwardWarp3DKernel(const float* f,
                                                const float* A,
                                                const float* b,
                                                float* fWarped,
                                                int width,
                                                int height,
                                                int depth,
                                                const float* coeffs){
    
    /*
    Kernel of GPU implementation of 3D backward image warping along the DVF (u,v)
    with cubic interpolation (rectangular multivariate spline)
    */

    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width && h < depth){

        // position at which to iterpolate
        float x = A[0]*j + A[1]*i + A[2]*h + b[0];
        float y = A[3]*j + A[4]*i + A[5]*h + b[1];
        float z = A[6]*j + A[7]*i + A[8]*h + b[2];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        // xi = x1 - 1 + i

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float zmz1 = z - z1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float zmz1_2 = zmz1 * zmz1;
        float zmz1_3 = zmz1 * zmz1_2;
        float monomials[] = {1, zmz1, zmz1_2, zmz1_3, ymy1, ymy1*zmz1, ymy1*zmz1_2, ymy1*zmz1_3, ymy1_2, ymy1_2*zmz1, ymy1_2*zmz1_2, ymy1_2*zmz1_3, ymy1_3, ymy1_3*zmz1, ymy1_3*zmz1_2, ymy1_3*zmz1_3, xmx1, xmx1*zmz1, xmx1*zmz1_2, xmx1*zmz1_3, xmx1*ymy1, xmx1*ymy1*zmz1, xmx1*ymy1*zmz1_2, xmx1*ymy1*zmz1_3, xmx1*ymy1_2, xmx1*ymy1_2*zmz1, xmx1*ymy1_2*zmz1_2, xmx1*ymy1_2*zmz1_3, xmx1*ymy1_3, xmx1*ymy1_3*zmz1, xmx1*ymy1_3*zmz1_2, xmx1*ymy1_3*zmz1_3, xmx1_2, xmx1_2*zmz1, xmx1_2*zmz1_2, xmx1_2*zmz1_3, xmx1_2*ymy1, xmx1_2*ymy1*zmz1, xmx1_2*ymy1*zmz1_2, xmx1_2*ymy1*zmz1_3, xmx1_2*ymy1_2, xmx1_2*ymy1_2*zmz1, xmx1_2*ymy1_2*zmz1_2, xmx1_2*ymy1_2*zmz1_3, xmx1_2*ymy1_3, xmx1_2*ymy1_3*zmz1, xmx1_2*ymy1_3*zmz1_2, xmx1_2*ymy1_3*zmz1_3, xmx1_3, xmx1_3*zmz1, xmx1_3*zmz1_2, xmx1_3*zmz1_3, xmx1_3*ymy1, xmx1_3*ymy1*zmz1, xmx1_3*ymy1*zmz1_2, xmx1_3*ymy1*zmz1_3, xmx1_3*ymy1_2, xmx1_3*ymy1_2*zmz1, xmx1_3*ymy1_2*zmz1_2, xmx1_3*ymy1_2*zmz1_3, xmx1_3*ymy1_3, xmx1_3*ymy1_3*zmz1, xmx1_3*ymy1_3*zmz1_2, xmx1_3*ymy1_3*zmz1_3};

        int k = 0;
        for(int hh = 0; hh < 4; hh++){
            for(int ii = 0; ii < 4; ii++){
                for(int jj = 0; jj < 4; jj++){
                    int Q0 = x1 + jj - 1;
                    int Q1 = y1 + ii - 1;
                    int Q2 = z1 + hh - 1;
                    if(0 <= Q0 && Q0 < width
                    && 0 <= Q1 && Q1 < height
                    && 0 <= Q2 && Q2 < depth){
                        float coefficient = 0;
                        for(int n = 0; n < 64; n++){
                            coefficient += coeffs[k*64 + n] * monomials[n];
                        }
                        fWarped[h*width*height + i*width + j] += coefficient * f[Q2*width*height + Q1*width + Q0];
                    }
                    k++;
                }
            }
        }
    }
}


__global__ void adjointAffineCubicBackwardWarp3DKernel(const float* fWarped,
                                                       const float* A,
                                                       const float* b,
                                                       float* f,
                                                       int width,
                                                       int height,
                                                       int depth,
                                                       const float* coeffs){
    
    /*
    Kernel of GPU implementation of adjoint 3D backward image warping along the
    DVF (u,v,w) with cubic interpolation (rectangular multivariate catmull-rom spline)
    */

    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width && h < depth){

        // position at which to iterpolate
        float x = A[0]*j + A[1]*i + A[2]*h + b[0];
        float y = A[3]*j + A[4]*i + A[5]*h + b[1];
        float z = A[6]*j + A[7]*i + A[8]*h + b[2];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        // xi = x1 - 1 + i

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float zmz1 = z - z1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float zmz1_2 = zmz1 * zmz1;
        float zmz1_3 = zmz1 * zmz1_2;
        float monomials[] = {1, zmz1, zmz1_2, zmz1_3, ymy1, ymy1*zmz1, ymy1*zmz1_2, ymy1*zmz1_3, ymy1_2, ymy1_2*zmz1, ymy1_2*zmz1_2, ymy1_2*zmz1_3, ymy1_3, ymy1_3*zmz1, ymy1_3*zmz1_2, ymy1_3*zmz1_3, xmx1, xmx1*zmz1, xmx1*zmz1_2, xmx1*zmz1_3, xmx1*ymy1, xmx1*ymy1*zmz1, xmx1*ymy1*zmz1_2, xmx1*ymy1*zmz1_3, xmx1*ymy1_2, xmx1*ymy1_2*zmz1, xmx1*ymy1_2*zmz1_2, xmx1*ymy1_2*zmz1_3, xmx1*ymy1_3, xmx1*ymy1_3*zmz1, xmx1*ymy1_3*zmz1_2, xmx1*ymy1_3*zmz1_3, xmx1_2, xmx1_2*zmz1, xmx1_2*zmz1_2, xmx1_2*zmz1_3, xmx1_2*ymy1, xmx1_2*ymy1*zmz1, xmx1_2*ymy1*zmz1_2, xmx1_2*ymy1*zmz1_3, xmx1_2*ymy1_2, xmx1_2*ymy1_2*zmz1, xmx1_2*ymy1_2*zmz1_2, xmx1_2*ymy1_2*zmz1_3, xmx1_2*ymy1_3, xmx1_2*ymy1_3*zmz1, xmx1_2*ymy1_3*zmz1_2, xmx1_2*ymy1_3*zmz1_3, xmx1_3, xmx1_3*zmz1, xmx1_3*zmz1_2, xmx1_3*zmz1_3, xmx1_3*ymy1, xmx1_3*ymy1*zmz1, xmx1_3*ymy1*zmz1_2, xmx1_3*ymy1*zmz1_3, xmx1_3*ymy1_2, xmx1_3*ymy1_2*zmz1, xmx1_3*ymy1_2*zmz1_2, xmx1_3*ymy1_2*zmz1_3, xmx1_3*ymy1_3, xmx1_3*ymy1_3*zmz1, xmx1_3*ymy1_3*zmz1_2, xmx1_3*ymy1_3*zmz1_3};

        int k = 0;
        for(int hh = 0; hh < 4; hh++){
            for(int ii = 0; ii < 4; ii++){
                for(int jj = 0; jj < 4; jj++){
                    int Q0 = x1 + jj - 1;
                    int Q1 = y1 + ii - 1;
                    int Q2 = z1 + hh - 1;
                    if(0 <= Q0 && Q0 < width
                    && 0 <= Q1 && Q1 < height
                    && 0 <= Q2 && Q2 < depth){
                        float coefficient = 0;
                        for(int n = 0; n < 64; n++){
                            coefficient += coeffs[k*64 + n] * monomials[n];
                        }
                        atomicAdd(&f[Q2*width*height + Q1*width + Q0], coefficient * fWarped[h*width*height + i*width + j]);
                    }
                    k++;
                }
            }
        }
    }
}