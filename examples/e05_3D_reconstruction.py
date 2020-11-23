# This example shows how to do 3D image reconstruction
# by defining a 3D warping operator.
# THIS EXAMPLE REQUIRES A DECENT AMOUNT OF RAM AND GPU MEMORY

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import LinearOperator, lsqr
from adjoint_image_warping import backward_warp_3D, adjoint_backward_warp_3D

# To define a scipy LinearOperator, we need to implement the
# _matvec and _rmatvec functions
class WarpingOperator(LinearOperator):

    def __init__(self, u, v, w, degree=3):
        self.dtype = np.dtype('float32')
        self.shape = (u.size, u.size)
        self.u = u
        self.v = v
        self.w = w
        self.degree = degree

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.u.shape)

        # preform the warp
        x_warped = backward_warp_3D(x, self.u, self.v, self.w, degree=self.degree)

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.u.shape)

        # preform the adjoint warp
        x = adjoint_backward_warp_3D(x_warped, self.u, self.v, self.w, degree=self.degree)
        
        # return as flattened array
        return x.ravel()



# we start by preforming the same warp as in e04_3D_warping.py
import tomopy
im_size = 512
print("generating 3D images and DVFs")
shepp = tomopy.shepp3d(im_size).astype(np.float32)
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
w = 2*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)

# this time, we can preform the warp with more high level syntax,
# as if warping is a matrix vector multiplication (which it is).
A = WarpingOperator(u, v, w)
b = A @ shepp.ravel() # @ is matrix mult
warped_shepp = b.reshape(shepp.shape)

# we will now try to reconstruct shepp form the warped version "warped_shepp"
# this reconstruction problem can be fromulated as
# Ax = b
# where A is the warping operator, and b is the warped shepp
print("Solving")
x = lsqr(A, b, iter_lim=30, show=True)[0] # least squares solver of scipy
reconstruction = x.reshape(shepp.shape)

plt.figure()
plt.title("original")
plt.imshow(shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("warped")
plt.imshow(warped_shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("reconstructed")
plt.imshow(reconstruction[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.show()