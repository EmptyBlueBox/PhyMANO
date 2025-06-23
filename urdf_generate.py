import io
import os

import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull

from config import hand_rest_pose, hand_shape, num_frames_hand
from utils_mano import generate_mano_submeshes


def mesh_to_obj_string(vertices, faces):
    """
    Converts mesh data to an OBJ format string, which is suitable for URDF.

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


def generate_urdf_xml(submeshes, joint_positions, colors, urdf_path, mesh_dir):
    """
    Generates a URDF XML file and corresponding mesh assets, saving them to disk.

    This function creates an XML configuration for a URDF model. It defines
    a robot with joints and links representing each submesh. For each submesh,
    its convex hull is computed. Based on the convex hull and a given density,
    the mass, center of mass (position), and inertia tensor are calculated for
    each link. The generated XML file and OBJ mesh files are saved to specified paths.

    Parameters:
    - submeshes (list of dict): A list of submesh dictionaries. Each dict should
      contain 'vertices' (np.ndarray, shape=(N, 3)), 'faces' (np.ndarray,
      shape=(M, 3)), and 'joint_idx' (int).
    - joint_positions (np.ndarray, shape=(16, 3)): The positions of the 16
      MANO joints.
    - colors (list of list of int): A list of RGB colors for the submeshes,
      indexed by joint index.
    - urdf_path (str): The path where the generated URDF XML file will be saved.
    - mesh_dir (str): The directory where the generated OBJ mesh files will be saved.
    """
    # Create mesh directory if it doesn't exist
    os.makedirs(mesh_dir, exist_ok=True)
    urdf_dir = os.path.dirname(urdf_path)

    link_xml_parts = []
    joint_xml_parts = []
    submesh_centers = np.zeros((16, 3))
    density = 980  # kg/m^3
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

    # Joint names for better understanding
    joint_names = [
        "wrist",  # 0 - palm/wrist (root)
        "thumb1",  # 1 - thumb MCP
        "thumb2",  # 2 - thumb PIP
        "thumb3",  # 3 - thumb DIP
        "index1",  # 4 - index MCP
        "index2",  # 5 - index PIP
        "index3",  # 6 - index DIP
        "middle1",  # 7 - middle MCP
        "middle2",  # 8 - middle PIP
        "middle3",  # 9 - middle DIP
        "ring1",  # 10 - ring MCP
        "ring2",  # 11 - ring PIP
        "ring3",  # 12 - ring DIP
        "pinky1",  # 13 - pinky MCP
        "pinky2",  # 14 - pinky PIP
        "pinky3",  # 15 - pinky DIP
    ]

    # Pre-process submeshes
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

        # URDF inertia tensor components
        ixx, iyy, izz = inertia[0, 0], inertia[1, 1], inertia[2, 2]
        ixy, ixz, iyz = inertia[0, 1], inertia[0, 2], inertia[1, 2]

        obj_filename = f"submesh_{submesh['joint_idx']}.obj"
        obj_data = mesh_to_obj_string(translated_vertices, hull_faces)

        # Save the mesh data to an OBJ file
        obj_save_path = os.path.join(mesh_dir, obj_filename)
        with open(obj_save_path, "w") as f:
            f.write(obj_data)

        # Use a relative path for the mesh file in the XML
        mesh_xml_path = os.path.relpath(obj_save_path, urdf_dir)

        color = colors[submesh["joint_idx"]]
        color_str = f"{color[0] / 255:.3f} {color[1] / 255:.3f} {color[2] / 255:.3f} 1"

        # Create link XML
        link_xml = f"""
    <link name="{joint_names[i]}">
        <inertial>
            <origin xyz="{cm[0]:.6f} {cm[1]:.6f} {cm[2]:.6f}" rpy="0 0 0"/>
            <mass value="{mass:.6f}"/>
            <inertia ixx="{ixx:.6e}" ixy="{ixy:.6e}" ixz="{ixz:.6e}" 
                     iyy="{iyy:.6e}" iyz="{iyz:.6e}" izz="{izz:.6e}"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_xml_path}"/>
            </geometry>
            <material name="material_{i}">
                <color rgba="{color_str}"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_xml_path}"/>
            </geometry>
        </collision>
    </link>"""

        link_xml_parts.append(link_xml)

        # Create joint XML (skip for root link)
        parent_idx = parents[i]
        if parent_idx != -1:  # Not root body
            parent_center = submesh_centers[parent_idx]
            relative_pos = center - parent_center

            # Joint position relative to parent's center
            joint_pos = joint_positions[i] - parent_center

            joint_xml = f"""
    <joint name="joint_{i}" type="continuous">
        <parent link="{joint_names[parent_idx]}"/>
        <child link="{joint_names[i]}"/>
        <origin xyz="{joint_pos[0]:.6f} {joint_pos[1]:.6f} {joint_pos[2]:.6f}" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit effort="10" velocity="10"/>
        <dynamics damping="0.1" friction="0.0"/>
    </joint>"""

            joint_xml_parts.append(joint_xml)

    # Combine all XML parts
    links_xml = "\n".join(link_xml_parts)
    joints_xml = "\n".join(joint_xml_parts)

    xml = f"""<?xml version="1.0"?>
<robot name="mano_hand">
    {links_xml}
    {joints_xml}
</robot>"""

    with open(urdf_path, "w") as f:
        f.write(xml)

    print(f"URDF model saved to {urdf_path}")
    print(f"Mesh files saved in {mesh_dir}")


def main():
    """
    Main function to generate MANO submeshes and create URDF files.

    This function generates MANO hand meshes using predefined hand pose and shape parameters,
    then creates the corresponding URDF XML configuration file and mesh assets.
    """
    print("=== Generating MANO-URDF Model ===")

    # --- 1. Generate MANO mesh and submeshes ---

    # Use a neutral hand pose (flat hand) for the static mesh.
    hand_parms = {
        "global_orient": torch.tensor([[0, 0, 0]], dtype=torch.float32).repeat(
            num_frames_hand, 1
        ),
        "transl": torch.tensor([[0, 0, 0.5]], dtype=torch.float32).repeat(
            num_frames_hand, 1
        ),
        "hand_pose": torch.tensor(hand_rest_pose, dtype=torch.float32),
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

    # --- 2. Generate URDF model ---
    # Define paths for saving the model and meshes
    urdf_dir = "Models/urdf"
    mesh_dir = os.path.join(urdf_dir, "mesh")
    urdf_path = os.path.join(urdf_dir, "hand.urdf")

    # Generate and save the URDF model files
    generate_urdf_xml(submeshes, hand_joints[0], colors, urdf_path, mesh_dir)


if __name__ == "__main__":
    main()
