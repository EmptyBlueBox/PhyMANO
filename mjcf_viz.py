import os
import time

import mujoco
import mujoco.viewer
import numpy as np


def main():
    """
    Main function to visualize and test a pre-generated MJCF hand model in MuJoCo.

    This function loads the MJCF file and runs a simulation with random control inputs
    to test the visualization and physics.
    """
    print("=== MuJoCo Hand Model Visualization ===")

    # Default path to the MJCF file
    mjcf_path = "Models/mjcf/hand.xml"

    # Check if the MJCF file exists
    if not os.path.exists(mjcf_path):
        print(f"Error: MJCF file not found at {mjcf_path}")
        print("Please run 'python mjcf_generate.py' first to create the model.")
        return

    print(f"Loading MJCF model from: {mjcf_path}")

    try:
        # Load the model from the saved MJCF file
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        data = mujoco.MjData(model)
        print(f"Model loaded successfully. Number of actuators: {model.nu}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nStarting MuJoCo simulation. Close the viewer to exit.")
    print(
        "The hand will move to a fixed random target position using position control."
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Generate a fixed random target position for all actuators
        target_positions = None
        if model.nu > 0:  # Only generate targets if actuators exist
            target_positions = np.random.uniform(low=-np.pi, high=np.pi, size=model.nu)
            data.ctrl[:] = target_positions
            print(f"Set fixed target positions: {target_positions}")
            print(f"Total actuators: {model.nu} (3 per ball joint)")
            print("Each ball joint controlled by 3 actuators for X, Y, Z rotation")

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            # Sync the viewer to visualize the new state
            viewer.sync()

            # Rudimentary real-time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
