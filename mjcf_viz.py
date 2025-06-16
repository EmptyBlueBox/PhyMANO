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
    print("The hand will move with random control inputs to test the physics.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_update_time = time.time()
        control_update_interval = 0.1  # Update controls every 100ms

        while viewer.is_running():
            step_start = time.time()

            current_time = time.time()
            if current_time - last_update_time > control_update_interval:
                # Apply random control signals to the actuators
                if model.nu > 0:  # Only apply controls if actuators exist
                    random_ctrl = np.random.uniform(low=-1.0, high=1.0, size=model.nu)
                    data.ctrl[:] = random_ctrl
                last_update_time = current_time

            mujoco.mj_step(model, data)

            # Sync the viewer to visualize the new state
            viewer.sync()

            # Rudimentary real-time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
