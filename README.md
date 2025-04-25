Self-proximity Loss
===================

This repository contains an implementation of the **self-proximity loss**, which appeared in 
[Voxel2Cortex with Correspondence](https://ieeexplore.ieee.org/document/10970096) and 
[Automated 3-D Extraction of Inner and Outer Surfaces of Cerebral Cortex from MRI](https://doi.org/10.1006/nimg.1999.0534).

Dependencies
------------
- NumPy
- SciPy
- PyTorch
- Matplotlib (for visualization)
- Pyvista (for visualization)

Usage
-----
The original self-proximity loss uses the formulation from [Automated 3-D Extraction of Inner and Outer Surfaces of Cerebral Cortex from MRI](https://doi.org/10.1006/nimg.1999.0534).
You may specify `kernel = 'MacDonald'` to reproduce this behavior.

However, in this implementation, an alternative kernel is provided, where the penalty is inspired by a repulsion force that is inversely proportional to the squared distance.
You may specify `kernel = 'repulsion'` to get this behavior.


Example
-------
There are also tools for pre-computing pairs of vertices that are not neighbors in a geodesic sense but are in proximity to each other in Euclidean sense.
In particular, we define two vertices to be neighbors in the geodesic sense if they are connected to the same vertex, i.e., at most two edges away from each other.
You may specify the value `r` in `find_r_pairs` to define the maximum distance between two vertices in Euclidean sense.
In the following figure, the red lines indicate pairs of vertices that are not neighbors in the geodesic sense but have a Euclidean distance less than `r`.

![Example](./surf.png)
