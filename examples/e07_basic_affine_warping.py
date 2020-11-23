# For affine warping, it would be wasteful to explicitly generate and
# stor the DVF. Instead, use the dedicated affine warping functions.

import numpy as np
from matplotlib import pyplot as plt
from adjoint_image_warping import affine_backward_warp_2D
from skimage import transform

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 512
shepp = tomopy.shepp2d(im_size)[0].astype(np.float32)

# an example affine transform
tform = transform.AffineTransform(rotation=0.1).params.astype(np.float32)
A = tform[:2,:2].copy()
b = tform[2, :2]

print(A)
print(b)

# linear backward warp
linear_warped_shepp = affine_backward_warp_2D(shepp, A, b)

# cubic backward warp
cubic_warped_shepp = affine_backward_warp_2D(shepp, A, b, degree=3)

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