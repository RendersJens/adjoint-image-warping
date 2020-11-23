import numpy as np
cimport numpy as np


# import the C++ versions of the warping functions
cdef extern from "cuda/warpAlgorithms.hu":
    void backwardWarp2D(const float* f,
                        const float* u,
                        const float* v,
                        float* fWarped,
                        int degree,
                        int width,
                        int height)

    void backwardWarp3D(const float* f,
                        const float* u,
                        const float* v,
                        const float* w,
                        float* fWarped,
                        int degree,
                        int width,
                        int height,
                        int depth)

    void adjointBackwardWarp2D(const float* fWarped,
                               const float* u,
                               const float* v,
                               float* f,
                               int degree,
                               int width,
                               int height)

    void adjointBackwardWarp3D(const float* fWarped,
                               const float* u,
                               const float* v,
                               const float* w,
                               float* f,
                               int degree,
                               int width,
                               int height,
                               int depth)


# python version of backwardWarp2D, this function accepts numpy arrays
def backward_warp_2D(np.ndarray[ndim=2, dtype=float, mode="c"] f,
                     np.ndarray[ndim=2, dtype=float, mode="c"] u,
                     np.ndarray[ndim=2, dtype=float, mode="c"] v,
                     np.ndarray[ndim=2, dtype=float, mode="c"] f_warped=None,
                     int degree=1):
    
    if f_warped is None:
        f_warped = np.zeros((f.shape[0], f.shape[1]), dtype=f.dtype)

    width = f.shape[1]
    height = f.shape[0]

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    backwardWarp2D(&f[0,0], &u[0,0], &v[0,0], &f_warped[0,0], degree, width, height)

    return f_warped


# python version of backwardWarp3D, this function accepts numpy arrays
def backward_warp_3D(np.ndarray[ndim=3, dtype=float, mode="c"] f,
                     np.ndarray[ndim=3, dtype=float, mode="c"] u,
                     np.ndarray[ndim=3, dtype=float, mode="c"] v,
                     np.ndarray[ndim=3, dtype=float, mode="c"] w,
                     np.ndarray[ndim=3, dtype=float, mode="c"] f_warped=None,
                     int degree=1):

    if f_warped is None:
        f_warped = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    width = f.shape[2]
    height = f.shape[1]
    depth = f.shape[0]

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    backwardWarp3D(&f[0,0,0],
                   &u[0,0,0],
                   &v[0,0,0],
                   &w[0,0,0],
                   &f_warped[0,0,0],
                   degree,
                   width,
                   height,
                   depth)

    return f_warped


# python version of adjointBackwardWarp2D, this function accepts numpy arrays
def adjoint_backward_warp_2D(np.ndarray[ndim=2, dtype=float, mode="c"] f_warped,
                             np.ndarray[ndim=2, dtype=float, mode="c"] u,
                             np.ndarray[ndim=2, dtype=float, mode="c"] v,
                             np.ndarray[ndim=2, dtype=float, mode="c"] f=None,
                             int degree=1):
    if f is None:
        f = np.zeros((f_warped.shape[0], f_warped.shape[1]), dtype=f_warped.dtype)

    width = f.shape[1]
    height = f.shape[0]

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    adjointBackwardWarp2D(&f_warped[0,0],
                          &u[0,0],
                          &v[0,0],
                          &f[0,0],
                          degree,
                          width,
                          height)

    return f


# python version of adjointBackwardWarp3D, this function accepts numpy arrays
def adjoint_backward_warp_3D(np.ndarray[ndim=3, dtype=float, mode="c"] f_warped,
                             np.ndarray[ndim=3, dtype=float, mode="c"] u,
                             np.ndarray[ndim=3, dtype=float, mode="c"] v,
                             np.ndarray[ndim=3, dtype=float, mode="c"] w,
                             np.ndarray[ndim=3, dtype=float, mode="c"] f=None,
                             int degree=1):

    if f is None:
        f = np.zeros((f_warped.shape[0], f_warped.shape[1], f_warped.shape[2]), dtype=f_warped.dtype)

    width = f.shape[2]
    height = f.shape[1]
    depth = f.shape[0]

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    adjointBackwardWarp3D(&f_warped[0,0,0],
                          &u[0,0,0],
                          &v[0,0,0],
                          &w[0,0,0],
                          &f[0,0,0],
                          degree,
                          width,
                          height,
                          depth)

    return f


# import the C++ versions of the affine warping functions
cdef extern from "cuda/warpAlgorithmsAffine.hu":
    void affineBackwardWarp2D(const float* f,
                              const float* A,
                              const float* b,
                              float* fWarped,
                              int degree,
                              int width,
                              int height)

    void affineBackwardWarp3D(const float* f,
                              const float* A,
                              const float* b,
                              float* fWarped,
                              int degree,
                              int width,
                              int height,
                              int depth)

    void adjointAffineBackwardWarp2D(const float* fWarped,
                                     const float* A,
                                     const float* b,
                                     float* f,
                                     int degree,
                                     int width,
                                     int height)

    void adjointAffineBackwardWarp3D(const float* fWarped,
                                     const float* A,
                                     const float* b,
                                     float* f,
                                     int degree,
                                     int width,
                                     int height,
                                     int depth)


# python version of backwardWarp2D, this function accepts numpy arrays
def affine_backward_warp_2D(np.ndarray[ndim=2, dtype=float, mode="c"] f,
                            np.ndarray[ndim=2, dtype=float, mode="c"] A,
                            np.ndarray[ndim=1, dtype=float, mode="c"] b,
                            np.ndarray[ndim=2, dtype=float, mode="c"] f_warped=None,
                            int degree=1,
                            str indexing = "xy"):
    
    if f_warped is None:
        f_warped = np.zeros((f.shape[0], f.shape[1]), dtype=f.dtype)

    if indexing != "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    width = f.shape[1]
    height = f.shape[0]

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    affineBackwardWarp2D(&f[0,0], &A[0,0], &b[0], &f_warped[0,0], degree, width, height)

    return f_warped


# python version of backwardWarp3D, this function accepts numpy arrays
def affine_backward_warp_3D(np.ndarray[ndim=3, dtype=float, mode="c"] f,
                            np.ndarray[ndim=2, dtype=float, mode="c"] A,
                            np.ndarray[ndim=1, dtype=float, mode="c"] b,
                            np.ndarray[ndim=3, dtype=float, mode="c"] f_warped=None,
                            int degree=1,
                            str indexing = "xy"):

    if f_warped is None:
        f_warped = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    if indexing != "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    width = f.shape[2]
    height = f.shape[1]
    depth = f.shape[0]

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    affineBackwardWarp3D(&f[0,0,0],
                         &A[0,0],
                         &b[0],
                         &f_warped[0,0,0],
                         degree,
                         width,
                         height,
                         depth)

    return f_warped


# python version of adjointBackwardWarp2D, this function accepts numpy arrays
def adjoint_affine_backward_warp_2D(np.ndarray[ndim=2, dtype=float, mode="c"] f_warped,
                                    np.ndarray[ndim=2, dtype=float, mode="c"] A,
                                    np.ndarray[ndim=1, dtype=float, mode="c"] b,
                                    np.ndarray[ndim=2, dtype=float, mode="c"] f=None,
                                    int degree=1,
                                    str indexing = "xy"):
    if f is None:
        f = np.zeros((f_warped.shape[0], f_warped.shape[1]), dtype=f_warped.dtype)

    if indexing != "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    width = f.shape[1]
    height = f.shape[0]

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    adjointAffineBackwardWarp2D(&f_warped[0,0],
                                &A[0,0],
                                &b[0],
                                &f[0,0],
                                degree,
                                width,
                                height)

    return f


# python version of adjointBackwardWarp3D, this function accepts numpy arrays
def adjoint_affine_backward_warp_3D(np.ndarray[ndim=3, dtype=float, mode="c"] f_warped,
                                    np.ndarray[ndim=2, dtype=float, mode="c"] A,
                                    np.ndarray[ndim=1, dtype=float, mode="c"] b,
                                    np.ndarray[ndim=3, dtype=float, mode="c"] f=None,
                                    int degree=1,
                                    str indexing = "xy"):

    if f is None:
        f = np.zeros((f_warped.shape[0], f_warped.shape[1], f_warped.shape[2]), dtype=f_warped.dtype)

    if indexing != "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    width = f.shape[2]
    height = f.shape[1]
    depth = f.shape[0]

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    adjointAffineBackwardWarp3D(&f_warped[0,0,0],
                                &A[0,0],
                                &b[0],
                                &f[0,0,0],
                                degree,
                                width,
                                height,
                                depth)

    return f