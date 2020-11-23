# For affine warping, it would be wasteful to explicitly generate and
# stor the DVF. Instead, use the dedicated affine warping functions.

import numpy as np
from matplotlib import pyplot as plt
from adjoint_image_warping import affine_backward_warp_3D

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 256
shepp = tomopy.shepp3d(im_size).astype(np.float32)

# example affine transform
A = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6), 0],
              [np.sin(np.pi/6),  np.cos(np.pi/6), 0],
              [0,                0.5,             1]], dtype=np.float32)
b = np.array([1, 2, 3], dtype=np.float32)

# linear backward warp
linear_warped_shepp = affine_backward_warp_3D(shepp, A, b)

# cubic backward warp
cubic_warped_shepp = affine_backward_warp_3D(shepp, A, b, degree=3)

# plots
plt.figure()
plt.title("original")
plt.imshow(shepp[im_size//2, :, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("linear warped")
plt.imshow(linear_warped_shepp[im_size//2, :, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("cubic warped")
plt.imshow(cubic_warped_shepp[im_size//2, :, :], cmap="gray")
plt.colorbar()

plt.show()