import io
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial import ConvexHull

from config import hand_pose, hand_shape, num_frames_hand
from utils_mano import generate_mano_submeshes


def mesh_to_obj_string(vertices, faces):
    """
    Converts mesh data to an OBJ format string, which is suitable for MuJoCo.

    Parameters:
    - vertices (np.ndarray, shape=(N, 3)): Vertex coordinates.
    - faces (np.ndarray, shape=(M, 3)): Face indices.

    Returns:
    - str: The mesh in OBJ format as a string.
    """
    with io.StringIO() as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # OBJ faces are 1-indexed
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
        return f.getvalue()


def orient_faces_outward(vertices, faces):
    """
    Ensures all face normals point outwards from the mesh's centroid.

    This function is crucial for ensuring correct rendering in engines that use
    back-face culling. It assumes the mesh is convex and its vertices are
    centered around the origin.

    Parameters:
    - vertices (np.ndarray, shape=(N, 3)): Vertex coordinates, assumed to be
      centered at the origin.
    - faces (np.ndarray, shape=(M, 3)): A list of faces, where each face is a
      list of three vertex indices.

    Returns:
    - np.ndarray, shape=(M, 3): The faces with corrected winding order.
    """
    oriented_faces = []
    for face_indices in faces:
        v0, v1, v2 = vertices[face_indices]
        # Compute the normal of the face
        normal = np.cross(v1 - v0, v2 - v0)
        # For a convex mesh centered at the origin, if the dot product of the
        # normal and a position vector on the face is negative, the normal
        # points towards the origin.
        if np.dot(normal, v0) < 0:
            # Flip the winding order to make the normal point outwards
            oriented_faces.append([face_indices[0], face_indices[2], face_indices[1]])
        else:
            oriented_faces.append(face_indices)
    return np.array(oriented_faces)


def generate_mujoco_xml(submeshes, colors):
    """
    Generates a MuJoCo XML string and mesh assets for simulating submeshes.

    This function creates an XML configuration for a MuJoCo simulation. It defines
    a world with gravity, a ground plane, and represents each submesh as a
    free-floating body. For simulation, each submesh is represented as its
    convex hull, which is computed explicitly. The bodies are positioned in space
    to form a complete hand shape based on the input submesh vertex locations.

    Parameters:
    - submeshes (list of dict): A list of submesh dictionaries. Each dict should
      contain 'vertices' (np.ndarray, shape=(N, 3)), 'faces' (np.ndarray,
      shape=(M, 3)), and 'joint_idx' (int).
    - colors (list of list of int): A list of RGB colors for the submeshes,
      indexed by joint index.

    Returns:
    - str: The MuJoCo XML model as a string.
    - dict: A dictionary of mesh assets, mapping filenames to OBJ data as bytes.
    """
    assets = {}
    asset_xml_parts = []
    body_xml_parts = []

    for i, submesh in enumerate(submeshes):
        # Compute the convex hull of the vertices to get the faces for the hull
        if submesh["vertices"].shape[0] >= 4:
            hull = ConvexHull(submesh["vertices"])
            hull_faces_unoriented = hull.simplices
        else:
            # Fallback for meshes with too few points for a 3D hull
            hull_faces_unoriented = submesh["faces"]

        # Center the vertices of the submesh for the asset definition
        center = np.mean(submesh["vertices"], axis=0)
        translated_vertices = submesh["vertices"] - center

        # Ensure face normals point outwards for correct rendering
        hull_faces = orient_faces_outward(translated_vertices, hull_faces_unoriented)

        obj_filename = f"submesh_{submesh['joint_idx']}.obj"
        obj_data = mesh_to_obj_string(translated_vertices, hull_faces)
        assets[obj_filename] = obj_data.encode()

        asset_xml_parts.append(f'<mesh name="mesh_{i}" file="{obj_filename}"/>')

        color = colors[submesh["joint_idx"]]
        color_str = f"{color[0] / 255:.3f} {color[1] / 255:.3f} {color[2] / 255:.3f} 1"

        # Position the body at the original center of the submesh
        pos_str = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"

        body_xml_parts.append(
            f"""
        <body name="body_{i}" pos="{pos_str}">
            <freejoint/>
            <geom type="mesh" mesh="mesh_{i}" rgba="{color_str}" mass="0.1" solimp="0.9 0.95 0.001" solref="0.02 1"/>
        </body>"""
        )

    asset_xml = "\n".join(asset_xml_parts)
    body_xml = "\n".join(body_xml_parts)

    xml = f"""
    <mujoco>
        <compiler autolimits="true"/>
        <option gravity="0 0 -9.81" timestep="0.002"/>

        <visual>
            <headlight active="0" ambient="0.3 0.3 0.3" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
            <map znear="0.01" zfar="50"/>
            <quality shadowsize="4096"/>
        </visual>

        <asset>
            {asset_xml}
        </asset>

        <worldbody>
            <light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
            <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1" name="ground"/>
            {body_xml}
        </worldbody>
    </mujoco>
    """
    return xml.strip(), assets


def main():
    """
    Main function to generate MANO submeshes and simulate them in MuJoCo.
    """
    # --- 1. Generate MANO mesh and submeshes ---

    # Use a neutral hand pose (flat hand) for the static mesh.
    # To make the hand lie flat with its back facing up (along the Z-axis in MuJoCo's world),
    # we apply a 90-degree rotation around the world's X-axis. This corresponds to
    # a `global_orient` of [pi/2, 0, 0] in axis-angle representation.
    hand_parms = {
        "global_orient": torch.tensor([[np.pi / 2, 0, 0]], dtype=torch.float32).repeat(
            num_frames_hand, 1
        ),
        "transl": torch.tensor([[0, 0, 0.5]], dtype=torch.float32).repeat(
            num_frames_hand, 1
        ),
        "hand_pose": torch.tensor(hand_pose, dtype=torch.float32),
        "betas": torch.tensor(hand_shape, dtype=torch.float32),
    }
    # Generate submeshes and other MANO data using the utility function
    submeshes, _, _, _, _ = generate_mano_submeshes(
        is_rhand=False,
        **hand_parms,
    )
    print(f"Generated {len(submeshes)} submeshes.")

    # Colors for 16 joints/submeshes
    colors = [
        [230, 97, 92],
        [88, 196, 157],
        [106, 137, 204],
        [255, 193, 84],
        [186, 123, 202],
        [95, 195, 228],
        [207, 106, 135],
        [139, 195, 74],
        [79, 134, 153],
        [255, 167, 38],
        [149, 117, 205],
        [38, 198, 218],
        [158, 158, 158],
        [121, 85, 72],
        [183, 28, 28],
        [56, 142, 60],
    ]

    # --- 2. Setup and run MuJoCo simulation ---

    xml, assets = generate_mujoco_xml(submeshes, colors)
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    data = mujoco.MjData(model)

    print("\nStarting MuJoCo simulation. Close the viewer to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        reset_interval_seconds = 3.0  # Reset simulation every 3 seconds

        while viewer.is_running():
            step_start = time.time()

            # If the simulation time exceeds the interval, reset it.
            if data.time >= reset_interval_seconds:
                mujoco.mj_resetData(model, data)

            mujoco.mj_step(model, data)

            # Sync the viewer to visualize the new state
            viewer.sync()

            # Rudimentary real-time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
