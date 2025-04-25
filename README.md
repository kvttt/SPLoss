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
To give a quick overview:
the original self-proximity loss uses the formulation from [Automated 3-D Extraction of Inner and Outer Surfaces of Cerebral Cortex from MRI](https://doi.org/10.1006/nimg.1999.0534).
You may specify `kernel = 'MacDonald'` to reproduce this behavior.
However, in this implementation, an alternative kernel is provided, where the penalty is inspired by a repulsion force that is inversely proportional to the squared distance.
You may specify `kernel = 'repulsion'` to get this behavior.

Now, I will go through every function in the code and explain how to use it.

1. **`load_mesh`**: Reads a mesh file as `PolyData` object and returns vertices and faces.
2. **`find_r_pairs`**: Uses Octree to find pairs of vertices that are at most `r` distance away from each other.
3. **`filter_neighbor_pairs`**: Uses adjacency matrix to keep pairs of vertices that are at least three edges away from each other.
4. **`build_adjacency`**: Builds an adjacency matrix from faces. Notice that the returned adjacency matrix is dense.
5. **`visualize`**: For visualization.

For self-proximity loss, you may create a `SelfProximityLoss` object. During initialization, you may specify reduction method, kernel, and the value of `delta` if you use the MacDonald kernel.

In the snippet below, I will illustrate how one may use the above functions during pre-processing and training.

```python
from sploss import load_mesh, find_r_pairs, filter_neighbor_pairs, build_adjacency, SelfProximityLoss
import torch

r = 0.2  # distance threshold
kernel = 'repulsion'  # kernel in self-proximity loss

v, f = load_mesh('./lh.pial.vtk')  # load vertices and faces
adj = build_adjacency(f, len(v))  # build adjacency matrix
pairs = find_r_pairs(v, r)  # find pairs of vertices that are at most r distance away from each other
pairs = filter_neighbor_pairs(pairs, adj)  # filter pairs of vertices that are at least three edges away from each other
print(f"pairs: {len(pairs)}")
spl = SelfProximityLoss(kernel=kernel)  # create a SelfProximityLoss object
vertices = torch.from_numpy(v).unsqueeze(0)  # convert vertices to PyTorch tensor of shape (1, N, 3)
spl_val = spl(vertices, pairs)  # compute self-proximity loss
print(spl_val)  # print self-proximity loss
```

Example
-------
This repository contains tools for pre-computing pairs of vertices that are not neighbors in a geodesic sense but are in proximity to each other in Euclidean sense.
In particular, we define two vertices to be neighbors in the geodesic sense if they are connected to the same vertex, i.e., at most two edges away from each other.
You may specify the value `r` in `find_r_pairs` to define the maximum distance between two vertices in Euclidean sense.
In the following figure, the red lines indicate pairs of vertices that are not neighbors in the geodesic sense but have a Euclidean distance less than `r`.

![Example](./surf.png)
