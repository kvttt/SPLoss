from typing import Tuple

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.spatial import KDTree
import torch
import torch.nn as nn


def load_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    mesh = pv.read(mesh_path)
    return mesh.points.astype(np.float32), mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def find_r_pairs(vertices: np.ndarray, r: float) -> np.ndarray:
    return KDTree(vertices).query_pairs(r, output_type='ndarray')


def filter_neighbor_pairs(pairs: np.ndarray, adj: np.ndarray) -> np.ndarray:
    return np.asarray([(i, j) for i, j in pairs if not np.any(adj[i] & adj[j])], dtype=np.int64)


def build_adjacency(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    edge_pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_undirected = np.vstack([edge_pairs, edge_pairs[:, ::-1]])
    row, col = edges_undirected.T
    data = np.ones(len(row), dtype=bool)
    adj = sp.coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
    return adj.toarray()


def visualize(
    vertices: np.ndarray, faces: np.ndarray, pairs: np.ndarray, n_show: int | None = None, 
    save_vtk_path: str | None = None, save_png_path: str | None = None, azimuth: float = 0.0,
) -> None | pv.pyvista_ndarray:
    n_show = len(pairs) if n_show is None else n_show

    # surface
    tri_flag = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    poly = pv.PolyData(vertices, np.hstack([tri_flag, faces]))

    # lines
    rng = np.random.default_rng(seed=0)
    sel = rng.choice(len(pairs), size=min(n_show, len(pairs)), replace=False)
    pts_l = vertices[pairs[sel, 0]]
    pts_r = vertices[pairs[sel, 1]]
    n = pts_l.shape[0]
    pts = np.zeros((2 * n, 3))
    pts[0::2] = pts_l
    pts[1::2] = pts_r
    n_lines = pts.shape[0] // 2
    connectivity = np.hstack([
        [2, 2 * i, 2 * i + 1] for i in range(n_lines)
    ]).astype(np.int64)
    lines_mesh = pv.PolyData(pts, lines=connectivity)
    if save_vtk_path is not None:
        lines_mesh.save(save_vtk_path)

    # plot
    if save_png_path is None:
        pl = pv.Plotter()
        pl.add_mesh(poly, opacity=1.0, color="lightslategray")
        pl.add_mesh(lines_mesh, color='red', line_width=1)
        pl.camera_position = 'yz'
        pl.camera.azimuth = azimuth
        pl.camera.zoom(1.5)
        pl.show()
    else:
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(poly, opacity=1.0, color="lightslategray")
        pl.add_mesh(lines_mesh, color='red', line_width=1)
        pl.camera_position = 'yz'
        pl.camera.azimuth = azimuth
        pl.camera.zoom(1.5)
        img = pl.screenshot(save_png_path, window_size=[3840, 2160])
        pl.close()
        return img


class SelfProximityLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', kernel: str = 'MacDonald', delta: float | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        if reduction not in ['none', 'sum', 'mean']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'.")
        self.reduction = reduction
        if kernel not in ['MacDonald', 'repulsion']:
            raise ValueError(f"Invalid kernel: {kernel}. Expected one of 'MacDonald', 'repulsion'.")
        self.kernel = kernel
        if kernel == 'MacDonald' and delta is None:
            raise ValueError("delta must be specified when using MacDonald kernel.")
        self.delta = delta
        self.eps = eps
    def forward(self, vertices: torch.Tensor, pairs: np.ndarray) -> torch.Tensor:
        v0 = vertices[:, pairs[:, 0], :]
        v1 = vertices[:, pairs[:, 1], :]
        d = torch.linalg.norm(v0 - v1, dim=-1)
        if self.kernel == 'MacDonald':
            p = torch.zeros_like(d)
            p[d < self.delta] = (self.delta - d[d < self.delta]) ** 2
        else:
            p = 1 / (d ** 2 + self.eps)

        if self.reduction == 'none':
            return p
        elif self.reduction == 'sum':
            return p.sum()
        else:
            if self.kernel == 'MacDonald':
                return p.sum() / (torch.gt(p, 0).sum() + self.eps)
            else:
                return p.mean()


if __name__ == "__main__":
    r = 0.2
    vertices, faces = load_mesh("./lh.pial.vtk")
    pairs = find_r_pairs(vertices, r)
    adj = build_adjacency(faces, len(vertices))
    pairs = filter_neighbor_pairs(pairs, adj)
    print(f"pairs: {len(pairs)}")
    sp_loss = SelfProximityLoss(reduction='mean', kernel='repulsion')
    vertices = torch.from_numpy(vertices).unsqueeze(0)
    sp_loss_val = sp_loss(vertices, pairs)
    print(f"sp_loss_val: {sp_loss_val}")
