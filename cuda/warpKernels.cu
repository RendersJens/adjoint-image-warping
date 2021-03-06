__global__ void linearBackwardWarp2DKernel(const float* f,
                                           const float* u,
                                           const float* v,
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
        float x = j+u[i*width + j];
        float y = i+v[i*width + j];

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


__global__ void adjointLinearBackwardWarp2DKernel(const float* fWarped,
                                                  const float* u,
                                                  const float* v,
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
        float x = j+u[i*width + j];
        float y = i+v[i*width + j];

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


__global__ void linearBackwardWarp3DKernel(const float* f,
                                           const float* u,
                                           const float* v,
                                           const float* w,
                                           float* fWarped,
                                           int width,
                                           int height,
                                           int depth){
    
    /*
    Kernel of GPU implementation of 3D backward image warping along the DVF (u,v,w)
    with linear interpolation (rectangular multivariate spline)
    */

    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width && h < depth){

        // position at which to iterpolate
        float x = j+u[h*width*height + i*width + j];
        float y = i+v[h*width*height + i*width + j];
        float z = h+w[h*width*height + i*width + j];

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


__global__ void adjointLinearBackwardWarp3DKernel(const float* fWarped,
                                                  const float* u,
                                                  const float* v,
                                                  const float* w,
                                                  float* f,
                                                  int width,
                                                  int height,
                                                  int depth){
    
    /*
    Kernel of GPU implementation of 3D adjoint backward image warping along the
    DVF (u,v,w) with linear interpolation (rectangular multivariate spline)
    */

    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width && h < depth){

        // position at which to iterpolate
        float x = j+u[h*width*height + i*width + j];
        float y = i+v[h*width*height + i*width + j];
        float z = h+w[h*width*height + i*width + j];

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


__global__ void cubicBackwardWarp2DKernel(const float* f,
                                          const float* u,
                                          const float* v, 
                                          float* fWarped,
                                          int width,
                                          int height,
                                          const float* coeffs){
    /*
    Kernel of GPU implementation of 2D backward image warping along the DVF (u,v)
    with cubic interpolation. The cubic polynomials used are passed by the
    coeffs parameter. The are computed externally with sage.
    */

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width){

        // position at which to iterpolate
        float x = j+u[i*width + j];
        float y = i+v[i*width + j];

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


__global__ void adjointCubicBackwardWarp2DKernel(const float* fWarped,
                                                 const float* u,
                                                 const float* v,
                                                 float* f,
                                                 int width,
                                                 int height,
                                                 const float* coeffs){

    /*
    Kernel of GPU implementation of 2D adjoint backward image warping along the DVF (u,v)
    with cubic interpolation. The cubic polynomials used are passed by the
    coeffs parameter. The are computed externally with sage.
    */

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width){

        // position at which to iterpolate
        float x = j+u[i*width + j];
        float y = i+v[i*width + j];

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
                    atomicAdd(&f[Q1*width + Q0], coefficient * fWarped[i*width + j]);
                }
                k++;
            }
        }
    }
}


__global__ void cubicBackwardWarp3DKernel(const float* f,
                                          const float* u,
                                          const float* v,
                                          const float* w,
                                          float* fWarped,
                                          int width,
                                          int height,
                                          int depth,
                                          const float* coeffs){
    
    /*
    Kernel of GPU implementation of 3D backward image warping along the DVF (u,v,w)
    with cubic interpolation. The cubic polynomials used are passed by the
    coeffs parameter. The are computed externally with sage.
    */

    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width && h < depth){

        // position at which to iterpolate
        float x = j+u[h*width*height + i*width + j];
        float y = i+v[h*width*height + i*width + j];
        float z = h+w[h*width*height + i*width + j];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        // xi = fx + i - 1

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


__global__ void adjointCubicBackwardWarp3DKernel(const float* fWarped,
                                                 const float* u,
                                                 const float* v,
                                                 const float* w,
                                                 float* f,
                                                 int width,
                                                 int height,
                                                 int depth,
                                                 const float* coeffs){
    
    /*
    Kernel of GPU implementation of 3D adjoint backward image warping along the DVF (u,v,w)
    with cubic interpolation. The cubic polynomials used are passed by the
    coeffs parameter. The are computed externally with sage.
    */

    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < height && j < width && h < depth){

        // position at which to iterpolate
        float x = j+u[h*width*height + i*width + j];
        float y = i+v[h*width*height + i*width + j];
        float z = h+w[h*width*height + i*width + j];

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