import time

import numpy as np
import rclpy
import tyro

from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC
from decoupled_wbc.control.main.teleop.configs.configs import TeleopConfig
from decoupled_wbc.control.policy.motion_planning_policy import MotionPlanningPolicy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from decoupled_wbc.control.utils.telemetry import Telemetry

MOTION_PLANNING_NODE_NAME = "MotionPlanningPolicy"


def main(config: TeleopConfig):
    """
    Main loop for motion planning policy.
    
    This script runs the motion planning policy that automatically generates
    trajectories to grasp objects (like bottles) in the scene.
    
    The policy receives privileged observations (e.g., bottle_pos) from the 
    simulation and generates smooth trajectories using quintic polynomials.
    """
    ros_manager = ROSManager(node_name=MOTION_PLANNING_NODE_NAME)
    node = ros_manager.node

    if config.robot == "g1":
        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        robot_model = instantiate_g1_robot_model(
            waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
        )
        left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
    else:
        raise ValueError(f"Unsupported robot name: {config.robot}")

    print("Initializing motion planning policy...")
    
    # Create retargeting IK (needed by motion planning policy)
    retargeting_ik = TeleopRetargetingIK(
        robot_model=robot_model,
        left_hand_ik_solver=left_hand_ik_solver,
        right_hand_ik_solver=right_hand_ik_solver,
        enable_visualization=config.enable_visualization,
        body_active_joint_groups=["upper_body"],
    )
    
    # Create motion planning policy
    # You can customize these parameters:
    # - trajectory_duration: how long to reach the target (seconds)
    # - grasp_offset: offset from object center to grasp point [x, y, z] in meters
    # - wait_for_activation: countdown before policy starts (seconds)
    motion_policy = MotionPlanningPolicy(
        robot_model=robot_model,
        retargeting_ik=retargeting_ik,
        trajectory_duration=3.0,  # 3 seconds to reach target
        grasp_offset=np.array([0.0, 0.0, 0.1]),  # 10cm above object center
        wait_for_activation=5,  # 5 second countdown
        activate_keyboard_listener=True,  # Enable keyboard control (press 'l' to activate)
    )
    
    print("Motion planning policy initialized!")
    print("Press 'l' to activate/deactivate the policy")
    print("Press 'k' to reset the policy")
    print("Note: The policy requires 'bottle_pos' in observations (from privileged obs)")

    # Create a publisher for the navigation commands
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)

    # Create rate controller
    rate = node.create_rate(config.teleop_frequency)
    iteration = 0
    time_to_get_to_initial_pose = 2  # seconds

    telemetry = Telemetry(window_size=100)

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                
                # Get the current motion planning action
                with telemetry.timer("get_action"):
                    data = motion_policy.get_action()

                # Add timing information to the message
                t_now = time.monotonic()
                data["timestamp"] = t_now

                # Set target completion time - longer for initial pose, then match control frequency
                if iteration == 0:
                    data["target_time"] = t_now + time_to_get_to_initial_pose
                else:
                    data["target_time"] = t_now + (1 / config.teleop_frequency)

                # Publish the motion planning command
                with telemetry.timer("publish_motion_command"):
                    control_publisher.publish(data)

                # For the initial pose, wait the full duration before continuing
                if iteration == 0:
                    print(f"Moving to initial pose for {time_to_get_to_initial_pose} seconds")
                    time.sleep(time_to_get_to_initial_pose)
                iteration += 1
                
            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.teleop_frequency):
                telemetry.log_timing_info(context="Motion Planning Policy Loop Missed", threshold=0.001)
            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")

    finally:
        print("Cleaning up...")
        motion_policy.close()
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(TeleopConfig)
    main(config)
