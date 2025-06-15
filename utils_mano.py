import os
import pickle

import numpy as np
import smplx
import torch

MANO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "Models")


def create_submeshes(vertices, faces, weights):
    """
    Create submeshes by assigning each vertex to the joint with maximum weight.

    This function separates a mesh into submeshes based on vertex weights, typically
    used for skinning. Each vertex is assigned to the joint that influences it the most.

    Parameters:
    - vertices (np.ndarray, shape=(N, 3)): Array of vertex coordinates.
    - faces (np.ndarray, shape=(M, 3)): Array of face indices.
    - weights (np.ndarray, shape=(N, 16)): Weight matrix for N vertices and 16 joints.

    Returns:
    - list: A list of dictionaries, where each dictionary represents a submesh
            and contains 'vertices', 'faces', and 'joint_idx'.
    """
    # Find the joint with maximum weight for each vertex
    max_weight_joints = np.argmax(weights, axis=1)

    submeshes = []

    for joint_idx in range(16):
        # Get vertices belonging to this joint
        vertex_mask = max_weight_joints == joint_idx
        joint_vertex_indices = np.where(vertex_mask)[0]

        if len(joint_vertex_indices) == 0:
            continue

        # Create a mapping from old vertex indices to new vertex indices
        old_to_new_vertex_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(joint_vertex_indices)
        }

        # Extract vertices for this joint
        joint_vertices = vertices[joint_vertex_indices]

        # Find faces that have all vertices belonging to this joint
        valid_faces = []
        for face in faces:
            if all(vertex_idx in old_to_new_vertex_map for vertex_idx in face):
                # Remap face indices to new vertex indices
                new_face = [old_to_new_vertex_map[vertex_idx] for vertex_idx in face]
                valid_faces.append(new_face)

        if len(valid_faces) > 0:
            joint_faces = np.array(valid_faces)
            submesh = {
                "vertices": joint_vertices,
                "faces": joint_faces,
                "joint_idx": joint_idx,
            }
            submeshes.append(submesh)

    return submeshes


def generate_mano_submeshes(
    global_orient: torch.Tensor,
    hand_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor = None,
    is_rhand: bool = True,
    model_path: str = MANO_MODEL_PATH,
):
    """
    Generates MANO submeshes from hand parameters.

    Encapsulates MANO model loading, forward pass, and mesh segmentation into submeshes
    based on joint weights.

    Parameters:
    - global_orient (torch.Tensor, shape=(B, 3)): Global orientation for B frames.
    - hand_pose (torch.Tensor, shape=(B, 48)): Hand pose parameters for B frames.
    - betas (torch.Tensor, shape=(B, 10)): Shape parameters for B frames.
    - transl (torch.Tensor, shape=(B, 3), optional): Translation. Defaults to zero if None.
    - is_rhand (bool, optional): Specifies if the model is for the right hand. Defaults to True.
    - model_path (str, optional): Path to the directory containing MANO models.

    Returns:
    - tuple:
        - submeshes (list): A list of submesh dictionaries. The structure is based on the first frame.
          Each dict contains 'vertices', 'faces', 'joint_idx'.
        - hand_vertices (np.ndarray, shape=(B, 778, 3)): Vertex positions for all B frames.
        - hand_faces (np.ndarray, shape=(1552, 3)): Indices of the watertight mesh faces.
        - hand_joints (np.ndarray, shape=(B, 16, 3)): Joint positions for all B frames.
        - weights (np.ndarray, shape=(778, 16)): Skinning weights for the vertices.
    """
    batch_size = hand_pose.shape[0]
    if transl is None:
        transl = torch.zeros((batch_size, 3), dtype=torch.float32)

    smplx_model = smplx.create(
        model_path=model_path,
        model_type="mano",
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=batch_size,
    )

    hand_parms = {
        "global_orient": global_orient,
        "transl": transl,
        "hand_pose": hand_pose,
        "betas": betas,
    }

    smplx_output = smplx_model(**hand_parms)
    hand_vertices = smplx_output.vertices.detach().cpu().numpy()
    hand_joints = smplx_output.joints.detach().cpu().numpy()
    original_faces = smplx_model.faces

    # Create a watertight mesh by adding faces to the wrist area
    new_face_index = [
        121,
        214,
        215,
        279,
        239,
        234,
        92,
        38,
        122,
        118,
        117,
        119,
        120,
        108,
        79,
        78,
    ]
    more_face = []
    for i in range(2, len(new_face_index)):
        more_face.append([121, new_face_index[i - 1], new_face_index[i]])
    hand_faces = np.concatenate([original_faces, more_face], axis=0)

    # Load MANO weights for segmentation
    mano_pkl_filename = "MANO_RIGHT.pkl" if is_rhand else "MANO_LEFT.pkl"
    mano_pkl_path = os.path.join(model_path, "mano", mano_pkl_filename)
    with open(mano_pkl_path, "rb") as f:
        model_data = pickle.load(f, encoding="latin1")
    weights = model_data["weights"]

    # Create the submesh structure based on the vertex positions of the first frame
    submeshes = create_submeshes(hand_vertices[0], hand_faces, weights)

    return submeshes, hand_vertices, hand_faces, hand_joints, weights
