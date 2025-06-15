import os

import numpy as np
import rerun as rr
import torch

from config import hand_pose, hand_shape, num_frames_hand
from utils_mano import generate_mano_submeshes
from utils_mesh import compute_vertex_normals


def init_rerun(path: str, save: bool = True):
    """
    Initialize the rerun visualization.

    Parameters:
    - path (str): The path of the rerun session, e.g. `Result/RerunSession/<timestamp>/<frame_count>.rrd`.
    - save (bool): Whether to save the rerun session. Default is True.
        - If True, the rerun session will be saved in the `path`.
        - If False, the rerun session will not be saved, but the visualization will be directly shown in the browser.
    """

    folder_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    rr.init(file_name, spawn=not save)
    if save:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        rr.save(path)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time("stable_time", duration=0)


def main():
    # Initialize MANO model and calculate hand vertices and joints
    hand_parms = {
        "global_orient": torch.zeros((num_frames_hand, 3), dtype=torch.float32),
        "transl": torch.zeros((num_frames_hand, 3), dtype=torch.float32),
        "hand_pose": torch.tensor(hand_pose, dtype=torch.float32),
        "betas": torch.tensor(hand_shape, dtype=torch.float32),
    }

    (
        submesh_structure,
        hand_vertices,
        hand_faces,
        hand_joints,
        weights,
    ) = generate_mano_submeshes(is_rhand=False, **hand_parms)

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

    init_rerun("test_submesh_visualization.rrd", save=False)

    # Add color information to each submesh for visualization
    submeshes = []
    for submesh in submesh_structure:
        submesh["color"] = colors[submesh["joint_idx"]]
        submeshes.append(submesh)

    print(f"Created {len(submeshes)} submeshes")

    for i in range(num_frames_hand):
        # Visualize original hand
        rr.log(
            "original_hand",
            rr.Mesh3D(
                vertex_positions=hand_vertices[i],
                triangle_indices=hand_faces,
                vertex_normals=compute_vertex_normals(hand_vertices[i], hand_faces),
                vertex_colors=np.full(
                    hand_vertices[i].shape, [255, 255, 255], dtype=np.uint8
                ),
            ),
        )

        # Update and visualize submeshes with current frame vertices
        for submesh in submeshes:
            joint_idx = submesh["joint_idx"]
            # Get vertex indices for this joint from the original submesh structure
            max_weight_joints = np.argmax(weights, axis=1)
            vertex_mask = max_weight_joints == joint_idx
            joint_vertex_indices = np.where(vertex_mask)[0]

            # Update vertices with current frame data
            updated_vertices = hand_vertices[i][joint_vertex_indices]

            rr.log(
                f"submeshes/joint_{joint_idx:02d}",
                rr.Mesh3D(
                    vertex_positions=updated_vertices,
                    triangle_indices=submesh["faces"],
                    vertex_normals=compute_vertex_normals(
                        updated_vertices, submesh["faces"]
                    ),
                    vertex_colors=np.full(
                        updated_vertices.shape, submesh["color"], dtype=np.uint8
                    ),
                ),
            )


if __name__ == "__main__":
    main()
