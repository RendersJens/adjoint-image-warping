# This example shows the advantage of this implementation of adjoint warping
# versus a commonly used alternative: warping along de negative DVF

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import LinearOperator, lsqr
from warp_algorithms_gpu import backward_warp_2D, adjoint_backward_warp_2D


class WarpingOperator(LinearOperator):

    def __init__(self, u, v, adjoint_type="exact", degree=1):
        self.dtype = np.dtype('float32')
        self.shape = (u.size, u.size)
        self.u = u
        self.v = v
        self.adjoint_type = adjoint_type
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
        if self.adjoint_type == "exact":
            x = adjoint_backward_warp_2D(x_warped, self.u, self.v, degree=self.degree)
        elif self.adjoint_type == "negative":
            x = backward_warp_2D(x_warped, -self.u, -self.v, degree=self.degree)
        else:
            raise NotImplementedError("adjoint type should be 'exact' or 'negative'")
        
        # return as flattened array
        return x.ravel()



im_size = 128

# a circle
circle = np.zeros((im_size,)*2, dtype=np.float32)
x, y = np.meshgrid(*[np.arange(128)]*2, indexing="xy")
circle[(x-64)**2+(y-64)**2<=30**2] = 1

# a simple translation:
u = np.ones(circle.shape, dtype=np.float32)*10
v = np.ones(circle.shape, dtype=np.float32)*5

# but the background is still
u[np.logical_and((x+u-64)**2+(y+v-64)**2>30**2, (x-64)**2+(y-64)**2>30**2)] *= -1
v[u<0] *= -1


warped_circle = backward_warp_2D(circle, u, v, degree=1)

# we will now try to reconstruct circle form the warped version "warped_shepp"
# this reconstruction problem can be fromulated as
# Ax = b
# where A is the warping operator, and b is the warped shepp
b = warped_circle.ravel()
A1 = WarpingOperator(u, v, adjoint_type="negative")
A2 = WarpingOperator(u, v, adjoint_type="exact")

x1 = lsqr(A1, b, iter_lim=30)[0]
x2 = lsqr(A2, b, iter_lim=30)[0]
reconstruction1 = x1.reshape(circle.shape)
reconstruction2 = x2.reshape(circle.shape)

plt.figure()
plt.title("original")
plt.imshow(circle, cmap="gray")
plt.colorbar()
skip = (slice(None, None, 2), slice(None, None, 2))
x, y = np.mgrid[:u.shape[1],:u.shape[0]]
plt.quiver(x.T[skip],y.T[skip],u[skip],v[skip], color="r", angles='xy', scale_units='xy', scale=1)


plt.figure()
plt.title("warped")
plt.imshow(warped_circle, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("reconstructed negative")
plt.imshow(reconstruction1, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("reconstructed exact")
plt.imshow(reconstruction2, cmap="gray")
plt.colorbar()


plt.show()