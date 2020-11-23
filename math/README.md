These sage notebooks show and perform the computation of the polynomials which are used in the cuda kernels. This is particularily important for the cubic case. This notebook will generate the files

 - `cubic_2D_coefficients.inc`
 - `cubic_3D_coefficients.inc`
 
which are imported by the respective cuda kernels. Just for reference, the same computations in the linear case are also included, but they can easily be computed by hand. I recommend looking at the cubic case first and using the linear case to compare.

Viewing the notebooks
=====================
The notebooks can be viewed right here on GitHub: https://github.com/RendersJens/adjoint-image-warping/tree/main/math

Running the notebooks
=====================
To run the notebooks you need to install sage from https://www.sagemath.org/.
Then run

`$sage -n jupyter`

from a terminal and select the notebooks from the UI.
