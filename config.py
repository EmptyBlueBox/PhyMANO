import numpy as np
import os

num_frames_hand = 1
hand_shape = np.array(
    [
        [
            0.25624018907546997,
            -0.46635702252388,
            -1.6472688913345337,
            0.17522339522838593,
            -0.09414590150117874,
            0.9134469628334045,
            0.21890263259410858,
            1.025447964668274,
            0.03658989071846008,
            -0.29835939407348633,
        ]
    ]
)
hand_rest_pose = np.zeros((num_frames_hand, 45))

# Configuration settings for MJCF visualization
mujoco_config = {
    "data_path": os.path.join(
        "reference", "data", "Cuboid_00-fps_60-smooth_5Hz-upsample.npz"
    ),
    "start_frame": 0,
    "end_frame": 5000,  # Visualize specified frame range
    "loop_animation": True,  # Whether to loop the animation
}
