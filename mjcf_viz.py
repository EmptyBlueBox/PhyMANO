import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from config import mujoco_config


def load_sequence_data(data_path):
    """
    Load sequence data from the NPZ file and convert rotations to quaternions.

    Parameters:
    - data_path (str): Path to the NPZ file containing hand pose sequence data.

    Returns:
    - dict: Dictionary containing loaded data with hand poses, orientations converted to quaternions, and metadata.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"Loading sequence data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    # Extract metadata
    metadata = data["metadata"].item()
    # print(f"Data metadata: {metadata}")

    # Extract sequence data
    hand_poses_axis_angle = data[
        "hand_poses"
    ]  # (N, 45) - MANO hand poses in axis-angle
    hand_orientations_axis_angle = data[
        "hand_orientations_axis_angle"
    ]  # (N, 3) - global orientation in axis-angle
    hand_translations = data["hand_translations"]  # (N, 3) - global translation

    print(f"Hand poses shape: {hand_poses_axis_angle.shape}")
    print(f"Hand orientations shape: {hand_orientations_axis_angle.shape}")
    print(f"Hand translations shape: {hand_translations.shape}")

    # Convert hand poses from axis-angle to quaternions
    print("Converting hand poses from axis-angle to quaternions...")
    N = hand_poses_axis_angle.shape[0]
    joint_poses = hand_poses_axis_angle.reshape(N, 15, 3)
    hand_poses_quat = np.zeros((N, 15, 4))

    for frame_idx in range(N):
        for joint_idx in range(15):
            axis_angle = joint_poses[frame_idx, joint_idx]

            if np.linalg.norm(axis_angle) > 1e-6:
                rot = R.from_rotvec(axis_angle)
                quat_xyzw = rot.as_quat()  # Returns [x, y, z, w]
                # MuJoCo uses [w, x, y, z] format
                hand_poses_quat[frame_idx, joint_idx] = [
                    quat_xyzw[3],  # w
                    quat_xyzw[0],  # x
                    quat_xyzw[1],  # y
                    quat_xyzw[2],  # z
                ]
            else:
                # Identity quaternion
                hand_poses_quat[frame_idx, joint_idx] = [1, 0, 0, 0]

    # Convert global orientations from axis-angle to quaternions
    print("Converting global orientations from axis-angle to quaternions...")
    hand_orientations_quat = np.zeros((N, 4))

    # Data specified, because this hand motion looks palm-up
    # Create a 180-degree rotation around X-axis to flip hand from palm-down to palm-up
    flip_rotation = R.from_rotvec([np.pi, 0, 0])  # 180 degrees around X-axis

    for frame_idx in range(N):
        axis_angle = hand_orientations_axis_angle[frame_idx]

        if np.linalg.norm(axis_angle) > 1e-6:
            rot = R.from_rotvec(axis_angle)
            # Apply the flip rotation to make hand back face up
            combined_rot = flip_rotation * rot
            quat_xyzw = combined_rot.as_quat()  # Returns [x, y, z, w]
            # MuJoCo uses [w, x, y, z] format
            hand_orientations_quat[frame_idx] = [
                quat_xyzw[3],  # w
                quat_xyzw[0],  # x
                quat_xyzw[1],  # y
                quat_xyzw[2],  # z
            ]
        else:
            # Apply flip rotation to identity
            quat_xyzw = flip_rotation.as_quat()  # Returns [x, y, z, w]
            hand_orientations_quat[frame_idx] = [
                quat_xyzw[3],  # w
                quat_xyzw[0],  # x
                quat_xyzw[1],  # y
                quat_xyzw[2],  # z
            ]

    print(f"Converted hand poses to quaternions shape: {hand_poses_quat.shape}")
    print(
        f"Converted hand orientations to quaternions shape: {hand_orientations_quat.shape}"
    )

    return {
        "hand_poses_quat": hand_poses_quat,
        "hand_orientations_quat": hand_orientations_quat,
        "hand_translations": hand_translations,
        "metadata": metadata,
        "fps": metadata["fps"],
    }


def main():
    """
    Main function to visualize a sequence of hand poses from recorded data.

    This function loads sequence data and plays back the hand motion using
    kinematic control (position control without dynamics).
    """
    print("=== MuJoCo Hand Model Sequence Visualization ===")

    # Default path to the MJCF file
    mjcf_path = "Models/mjcf/hand.xml"

    # Check if the MJCF file exists
    if not os.path.exists(mjcf_path):
        print(f"Error: MJCF file not found at {mjcf_path}")
        print("Please run 'python mjcf_generate.py' first to create the model.")
        return

    # Load sequence data
    try:
        sequence_data = load_sequence_data(mujoco_config["data_path"])
    except Exception as e:
        print(f"Error loading sequence data: {e}")
        return

    print(f"Loading MJCF model from: {mjcf_path}")

    try:
        # Load the model from the saved MJCF file
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        data = mujoco.MjData(model)

        # Disable gravity for kinematic visualization
        model.opt.gravity[:] = 0.0
        print(f"Model loaded successfully. Number of actuators: {model.nu}")
        print("Gravity disabled for kinematic visualization")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare sequence data
    hand_poses_quat = sequence_data["hand_poses_quat"]
    hand_orientations_quat = sequence_data["hand_orientations_quat"]
    hand_translations = sequence_data["hand_translations"]
    data_fps = sequence_data["fps"]

    # Note: hand_poses_quat and hand_orientations_quat are already converted to quaternions
    print("Data uses quaternion representation for hand poses and orientations")
    print(
        f"Global orientations (hand_orientations_quat): {hand_orientations_quat.shape}"
    )
    print(f"Global translations (hand_translations): {hand_translations.shape}")

    # Apply frame range limits
    start_frame = max(0, mujoco_config["start_frame"])
    end_frame = min(len(hand_poses_quat), mujoco_config["end_frame"])

    if start_frame >= end_frame:
        print(f"Error: Invalid frame range [{start_frame}, {end_frame})")
        return

    print(
        f"Visualizing frames {start_frame} to {end_frame-1} ({end_frame-start_frame} frames)"
    )
    print(f"Data FPS: {data_fps}, using original data FPS for playback")

    # Extract the frame range
    hand_poses_quat = hand_poses_quat[start_frame:end_frame]
    hand_orientations_quat = hand_orientations_quat[start_frame:end_frame]
    hand_translations = hand_translations[start_frame:end_frame]

    print("\nStarting MuJoCo sequence visualization. Close the viewer to exit.")
    print(
        "The hand will play back the recorded motion sequence using kinematic control."
    )
    print(f"Loop animation: {mujoco_config['loop_animation']}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        target_frametime = 1.0 / data_fps

        while viewer.is_running():
            step_start = time.time()

            # Set global transformation for freejoint (no actuators, set directly in qpos)
            current_translation = hand_translations[frame_idx]
            current_orientation_quat = hand_orientations_quat[frame_idx]

            # Set freejoint position and orientation directly
            data.qpos[0:3] = current_translation + np.array([0, 0, 0.5])

            # Set freejoint orientation using pre-computed quaternions
            data.qpos[3:7] = (
                current_orientation_quat  # Already in MuJoCo format [w, x, y, z]
            )

            # Set joint quaternions directly to qpos for ball joints
            qpos_offset = 7  # Skip freejoint (3 translation + 4 quaternion)
            current_joint_quats = hand_poses_quat[frame_idx]  # Shape: (15, 4)

            for joint_idx in range(15):
                # Use pre-computed quaternions
                data.qpos[qpos_offset : qpos_offset + 4] = current_joint_quats[
                    joint_idx
                ]
                qpos_offset += 4  # Each ball joint uses 4 quaternion components

            # Step the simulation to update joint positions
            mujoco.mj_step(model, data)

            # Sync the viewer to visualize the new state
            viewer.sync()

            # Update frame index
            frame_idx += 1
            if frame_idx >= len(hand_poses_quat):
                if mujoco_config["loop_animation"]:
                    frame_idx = 0
                    print(f"Looping animation - returning to frame {start_frame}")
                else:
                    print("Animation completed - holding last frame")
                    frame_idx = len(hand_poses_quat) - 1

            # Frame rate control
            elapsed_time = time.time() - step_start
            time_until_next_frame = target_frametime - elapsed_time
            if time_until_next_frame > 0:
                time.sleep(time_until_next_frame)


if __name__ == "__main__":
    main()
