import io
import os

import numpy as np
import torch
import trimesh
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


def generate_mujoco_xml(submeshes, joint_positions, colors, mjcf_path, mesh_dir):
    """
    Generates a MuJoCo XML file and corresponding mesh assets, saving them to disk.

    This function creates an XML configuration for a MuJoCo simulation. It defines
    a world with gravity, a ground plane, and represents each submesh as a
    free-floating body. For each submesh, its convex hull is computed.
    Based on the convex hull and a given density, the mass, center of mass (position),
    and inertia tensor are calculated for each body. The bodies are positioned in space
    to form a complete hand shape based on the input submesh vertex locations.
    The generated XML file and OBJ mesh files are saved to specified paths.

    Parameters:
    - submeshes (list of dict): A list of submesh dictionaries. Each dict should
      contain 'vertices' (np.ndarray, shape=(N, 3)), 'faces' (np.ndarray,
      shape=(M, 3)), and 'joint_idx' (int).
    - joint_positions (np.ndarray, shape=(16, 3)): The positions of the 16
      MANO joints.
    - colors (list of list of int): A list of RGB colors for the submeshes,
      indexed by joint index.
    - mjcf_path (str): The path where the generated MJCF XML file will be saved.
    - mesh_dir (str): The directory where the generated OBJ mesh files will be saved.
    """
    # Create mesh directory if it doesn't exist
    os.makedirs(mesh_dir, exist_ok=True)
    mjcf_dir = os.path.dirname(mjcf_path)

    asset_xml_parts = []
    body_xml_parts = [""] * 16  # To be filled later
    actuator_xml_parts = []
    submesh_centers = np.zeros((16, 3))
    density = 980  # kg/m^3
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

    # Pre-process submeshes and create assets
    submeshes_by_joint = {submesh["joint_idx"]: submesh for submesh in submeshes}

    for i in range(16):  # Iterate through all possible joint indices
        submesh = submeshes_by_joint.get(i)
        if not submesh:
            continue

        # Compute the convex hull of the vertices to get the faces for the hull
        if submesh["vertices"].shape[0] >= 4:
            hull = ConvexHull(submesh["vertices"])
            hull_faces_unoriented = hull.simplices
        else:
            # Fallback for meshes with too few points for a 3D hull
            hull_faces_unoriented = submesh["faces"]

        # Center the vertices of the submesh for the asset definition
        center = np.mean(submesh["vertices"], axis=0)
        submesh_centers[i] = center
        translated_vertices = submesh["vertices"] - center

        # Ensure face normals point outwards for correct rendering
        hull_faces = orient_faces_outward(translated_vertices, hull_faces_unoriented)

        # Create a trimesh object for physical properties calculation
        hull_mesh = trimesh.Trimesh(vertices=translated_vertices, faces=hull_faces)
        if not hull_mesh.is_watertight:
            hull_mesh.fill_holes()

        # Calculate physical properties based on the convex hull
        mass = (
            hull_mesh.mass
        )  # mass is volume * density, trimesh assumes density=1 by default
        mass *= density
        inertia = (
            hull_mesh.moment_inertia * density
        )  # Inertia is calculated around the center of mass.
        cm = hull_mesh.center_mass

        # MuJoCo's `fullinertia` requires Ixx, Iyy, Izz, Ixy, Ixz, Iyz
        ixx, iyy, izz = inertia[0, 0], inertia[1, 1], inertia[2, 2]
        ixy, ixz, iyz = inertia[0, 1], inertia[0, 2], inertia[1, 2]
        inertia_str = f"{ixx:.6e} {iyy:.6e} {izz:.6e} {ixy:.6e} {ixz:.6e} {iyz:.6e}"

        obj_filename = f"submesh_{submesh['joint_idx']}.obj"
        obj_data = mesh_to_obj_string(translated_vertices, hull_faces)

        # Save the mesh data to an OBJ file
        obj_save_path = os.path.join(mesh_dir, obj_filename)
        with open(obj_save_path, "w") as f:
            f.write(obj_data)

        # Use a relative path for the mesh file in the XML
        mesh_xml_path = os.path.relpath(obj_save_path, mjcf_dir)
        asset_xml_parts.append(f'<mesh name="mesh_{i}" file="{mesh_xml_path}"/>')

        color = colors[submesh["joint_idx"]]
        color_str = f"{color[0] / 255:.3f} {color[1] / 255:.3f} {color[2] / 255:.3f} 1"
        cm_pos_str = f"{cm[0]:.3f} {cm[1]:.3f} {cm[2]:.3f}"

        # Determine body position and joint information
        parent_idx = parents[i]
        if parent_idx == -1:  # Root body (palm)
            pos_str = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"
            joint_xml = "<freejoint/>"
        else:
            parent_center = submesh_centers[parent_idx]
            relative_pos = center - parent_center
            pos_str = (
                f"{relative_pos[0]:.3f} {relative_pos[1]:.3f} {relative_pos[2]:.3f}"
            )

            # Joint position relative to the child's body frame
            joint_pos = joint_positions[i] - center
            joint_pos_str = f"{joint_pos[0]:.3f} {joint_pos[1]:.3f} {joint_pos[2]:.3f}"
            joint_xml = f'<joint name="joint_{i}" type="ball" pos="{joint_pos_str}" limited="false" stiffness="200" damping="5"/>'
            # Create 3 actuators for 3 DOF of ball joint
            actuator_xml_parts.append(
                f'<position name="motor_{i}_x" joint="joint_{i}" gear="1 0 0" ctrllimited="true" ctrlrange="-3.14 3.14" kp="100"/>'
            )
            actuator_xml_parts.append(
                f'<position name="motor_{i}_y" joint="joint_{i}" gear="0 1 0" ctrllimited="true" ctrlrange="-3.14 3.14" kp="100"/>'
            )
            actuator_xml_parts.append(
                f'<position name="motor_{i}_z" joint="joint_{i}" gear="0 0 1" ctrllimited="true" ctrlrange="-3.14 3.14" kp="100"/>'
            )

        body_xml_parts[i] = f"""<body name="body_{i}" pos="{pos_str}">
            {joint_xml}
            <inertial pos="{cm_pos_str}" mass="{mass:.6f}" fullinertia="{inertia_str}"/>
            <geom type="mesh" mesh="mesh_{i}" rgba="{color_str}" contype="1" conaffinity="2" material="hand_mat"/>"""

    # Assemble the nested body XML structure
    # Start with closing tags for all bodies
    body_xml_tree = [""] * 16
    for i in range(15, -1, -1):
        if body_xml_parts[i]:  # If the body exists
            body_xml_tree[i] += "</body>"
            parent_idx = parents[i]
            if parent_idx != -1:
                # Prepend this body's XML to its parent's closing tag
                body_xml_tree[parent_idx] = (
                    body_xml_parts[i] + body_xml_tree[i] + body_xml_tree[parent_idx]
                )

    # The final body XML is the content of the root body (joint 0)
    root_body_xml = body_xml_parts[0] + body_xml_tree[0]

    asset_xml = "\n".join(asset_xml_parts)
    actuator_xml = "\n".join(actuator_xml_parts)

    xml = f"""
    <mujoco>
        <compiler autolimits="true"/>
        <option gravity="0 0 -9.81" timestep="0.001"/>

        <default>
            <geom solimp="0.9 0.95 0.001" solref="0.02 1"/>
        </default>

        <visual>
            <headlight active="0" ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
            <map znear="0.01" zfar="50" shadowclip="0.5"/>
            <quality shadowsize="8192" offsamples="8"/>
            <global azimuth="120" elevation="-20"/>
        </visual>

        <asset>
            {asset_xml}
            
            <!-- Materials for better lighting -->
            <material name="ground_mat" reflectance="0.1" shininess="0.1" specular="0.2"/>
            <material name="hand_mat" reflectance="0.05" shininess="0.3" specular="0.4"/>
        </asset>

        <worldbody>
            <!-- Main key light -->
            <light name="sun" pos="2 2 4" dir="-0.3 -0.3 -1" diffuse="1.0 0.95 0.8" specular="0.3 0.3 0.3" castshadow="true"/>
            <!-- Fill light from opposite side -->
            <light name="fill" pos="-1.5 -1 3" dir="0.2 0.1 -1" diffuse="0.6 0.7 0.9" specular="0.1 0.1 0.1"/>
            <!-- Bottom fill for hand palm -->
            <light name="bottom" pos="0 0 -1" dir="0 0 1" diffuse="0.4 0.4 0.5" specular="0.05 0.05 0.05"/>
            <!-- Rim light for better definition -->
            <light name="rim" pos="0 3 2" dir="0 -1 -0.3" diffuse="0.3 0.3 0.4" specular="0.1 0.1 0.1"/>
            
            <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1" name="ground" contype="2" conaffinity="1" 
                  material="ground_mat"/>
            {root_body_xml}
        </worldbody>

        <actuator>
            {actuator_xml}
        </actuator>
    </mujoco>
    """
    with open(mjcf_path, "w") as f:
        f.write(xml.strip())

    print(f"MJCF model saved to {mjcf_path}")
    print(f"Mesh files saved in {mesh_dir}")


def main():
    """
    Main function to generate MANO submeshes and create MuJoCo MJCF files.

    This function generates MANO hand meshes using predefined hand pose and shape parameters,
    then creates the corresponding MuJoCo XML configuration file and mesh assets.
    """
    print("=== Generating MANO-MuJoCo MJCF Model ===")

    # --- 1. Generate MANO mesh and submeshes ---

    # Use a neutral hand pose (flat hand) for the static mesh.
    # To make the hand lie flat with its back facing up (along the Z-axis in MuJoCo's world),
    # we apply a 90-degree rotation around the world's X-axis. This corresponds to
    # a `global_orient` of [pi/2, 0, 0] in axis-angle representation.
    hand_parms = {
        "global_orient": torch.tensor([[np.pi / 2, 0, 0]], dtype=torch.float32).repeat(
            num_frames_hand, 1
        ),
        "transl": torch.tensor([[0, 0, 1.5]], dtype=torch.float32).repeat(
            num_frames_hand, 1
        ),
        "hand_pose": torch.tensor(hand_pose, dtype=torch.float32),
        "betas": torch.tensor(hand_shape, dtype=torch.float32),
    }
    # Generate submeshes and other MANO data using the utility function
    submeshes, _, _, hand_joints, _ = generate_mano_submeshes(
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

    # --- 2. Generate MuJoCo model ---
    # Define paths for saving the model and meshes
    mjcf_dir = "Models/mjcf"
    mesh_dir = os.path.join(mjcf_dir, "mesh")
    mjcf_path = os.path.join(mjcf_dir, "hand.xml")

    # Generate and save the MuJoCo model files
    generate_mujoco_xml(submeshes, hand_joints[0], colors, mjcf_path, mesh_dir)


if __name__ == "__main__":
    main()
