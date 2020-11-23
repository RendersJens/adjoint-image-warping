# This example shows how to use the gpu warping functions
# the adjoints are not yet used.

import numpy as np
from matplotlib import pyplot as plt
from adjoint_image_warping import backward_warp_2D

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 512
shepp = tomopy.shepp2d(im_size)[0].astype(np.float32)

# an example DVF
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size).reshape((im_size, im_size))
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size).reshape((im_size, im_size))

# linear backward warp
linear_warped_shepp = backward_warp_2D(shepp, u, v)

# cubic backward warp
cubic_warped_shepp = backward_warp_2D(shepp, u, v, degree=3)

# a cubic warp can produce values outside of the original range
np.clip(cubic_warped_shepp, 0, 255, out=cubic_warped_shepp)

# plots
plt.figure()
plt.title("original")
plt.imshow(shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("linear warped")
plt.imshow(linear_warped_shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("cubic warped")
plt.imshow(cubic_warped_shepp, cmap="gray")
plt.colorbar()

plt.show()