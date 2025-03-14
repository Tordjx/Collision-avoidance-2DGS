import trimesh

import coacd
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--threshold", type=float, default=0.02, help="Concavity threshold for terminating the decomposition. default = 0.02")
argparser.add_argument("--preprocess_resolution", type=int, default=100, help = "Resolution for manifold preprocess. default = 100")
argparser.add_argument("--input_file", type=str, default="data/fuse_post.ply")
argparser.add_argument("--output_file", type=str, default="data/collision_mesh.obj")
args = argparser.parse_args()
mesh = trimesh.load(args.input_file, force="mesh")
mesh = coacd.Mesh(mesh.vertices, mesh.faces)
parts = coacd.run_coacd(
    mesh, threshold=args.threshold, preprocess_resolution=args.preprocess_resolution
)  # a list of convex hulls.
scene = trimesh.Scene()
for vertices, faces in parts:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    scene.add_geometry(mesh)


# Export the scene to a .obj file
scene.export(args.output_file)
