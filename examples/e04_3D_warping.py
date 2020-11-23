# This example shows how to use the gpu warping functions in 3D

import numpy as np
from matplotlib import pyplot as plt
from adjoint_image_warping import backward_warp_3D

# We use a sample image from tomopy, but you can replace this with any 3D image
import tomopy
im_size = 256
shepp = tomopy.shepp3d(im_size).astype(np.float32)

# an example DVF
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
w = 2*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)

# linear backward warp
linear_warped_shepp = backward_warp_3D(shepp, u, v, w)

# cubic backward warp
cubic_warped_shepp = backward_warp_3D(shepp, u, v, w, degree=3)

# a cubic warp can produce values outside of the original range
np.clip(cubic_warped_shepp, 0, 255, out=cubic_warped_shepp)

# plots (we just show one slice of each 3D image)
plt.figure()
plt.title("original")
plt.imshow(shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("linear warped")
plt.imshow(linear_warped_shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("cubic warped")
plt.imshow(cubic_warped_shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.show()