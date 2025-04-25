from functools import partial
import os

import numpy as np

from sploss import load_mesh, build_adjacency, find_r_pairs, filter_neighbor_pairs, visualize


def process_mesh(mesh_path: str, out_path: str, r: float) -> None:
    subid = mesh_path.split("/")[-1].split("_")[0]
    vertices, faces = load_mesh(mesh_path)
    n_vertices = len(vertices)
    adj = build_adjacency(faces, n_vertices)
    pairs = find_r_pairs(vertices, r)
    pairs = filter_neighbor_pairs(pairs, adj)
    print(f"subid: {subid}, pairs: {len(pairs)}")
    out_mesh_path = os.path.join(out_path, f"{os.path.basename(mesh_path).replace('.vtk', '.npy')}")
    np.save(out_mesh_path, pairs)


def visualize_mesh(mesh_path: str, out_path: str, azimuth: float) -> None:
    subid = mesh_path.split("/")[-1].split("_")[0]
    vertices, faces = load_mesh(mesh_path)
    pair_path = os.path.join(out_path, f"{os.path.basename(mesh_path).replace('.vtk', '.npy')}")
    vtk_path = os.path.join(out_path, f"{os.path.basename(mesh_path)}")
    png_path = os.path.join(out_path, f"{os.path.basename(mesh_path).replace('.vtk', '.png')}")
    pairs = np.load(pair_path)
    print(f"subid: {subid}, pairs: {len(pairs)}")
    visualize(vertices, faces, pairs, save_vtk_path=vtk_path, save_png_path=png_path, azimuth=azimuth)


if __name__ == "__main__":
    r_lst = [0.4, 0.8, 1.2, 1.6, 2.0]
    out_path_lst = [f'./r_{str(r).replace(".", "p")}' for r in r_lst]
    mesh_path_lst = ["./lh.pial.vtk", "./rh.pial.vtk"]

    # process_mesh
    for r, out_path in zip(r_lst, out_path_lst):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        process_mesh_partial = partial(process_mesh, out_path=out_path, r=r)
        for mesh_path in mesh_path_lst:
            process_mesh_partial(mesh_path=mesh_path, out_path=out_path, r=r)
    
    # visualize_mesh
    for r, out_path in zip(r_lst, out_path_lst):
        visualize_mesh_partial = partial(visualize_mesh, out_path=out_path)
        visualize_mesh_partial(mesh_path_lst[0], azimuth=180.0)
        visualize_mesh_partial(mesh_path_lst[1], azimuth=0.0)

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(5, 2, figsize=(8, 16), layout='constrained')
    for i, r in enumerate(r_lst):
        for j, mesh_path in enumerate(mesh_path_lst):
            title = "Left hemisphere" if j == 0 else "Right hemisphere"
            png_path = os.path.join(out_path_lst[i], f"{os.path.basename(mesh_path).replace('.vtk', '.png')}")
            img = plt.imread(png_path)
            ax[i, j].imshow(img)
            ax[i, j].axis('off')
            ax[i, j].set_title(f"{title}, r={r}")
    f.savefig("surf.png", dpi=600)
    plt.show()
