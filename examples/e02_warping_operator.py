# This example shows how the warping functions and adjoint warping
# functions can be used to create a linear operator to do image reconstruction.
# look at sample e03_reconstruction to see how the reconstruction is done

import numpy as np
from scipy.sparse.linalg import LinearOperator
from adjoint_image_warping import backward_warp_2D, adjoint_backward_warp_2D

# To define a scipy LinearOperator, we need to implement the
# _matvec and _rmatvec functions
class WarpingOperator(LinearOperator):

    def __init__(self, u, v, degree=3):
        self.dtype = np.dtype('float32')
        self.shape = (u.size, u.size)
        self.u = u
        self.v = v
        self.degree = degree

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.u.shape)

        # preform the warp
        x_warped = backward_warp_2D(x, self.u, self.v, degree=self.degree)

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.u.shape)

        # preform the adjoint warp
        x = adjoint_backward_warp_2D(x_warped, self.u, self.v, degree=self.degree)
        
        # return as flattened array
        return x.ravel()