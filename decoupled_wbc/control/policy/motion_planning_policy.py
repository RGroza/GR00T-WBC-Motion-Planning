from contextlib import contextmanager
import time as time_module
from typing import Optional, Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from decoupled_wbc.control.base.policy import Policy
from decoupled_wbc.control.robot_model import RobotModel
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK


class MotionPlanningPolicy(Policy):
    """
    Robot-agnostic motion planning policy for automated manipulation tasks.
    
    Generates smooth trajectories from the robot's current pose to target objects
    using straight-line interpolation with smooth acceleration profiles.
    
    Future improvements:
    - B-spline trajectory generation
    - Obstacle avoidance
    - Multi-waypoint planning
    """

    def __init__(
        self,
        robot_model: RobotModel,
        retargeting_ik: TeleopRetargetingIK,
        trajectory_duration: float = 3.0,
        grasp_offset: np.ndarray = np.array([0.0, 0.0, 0.15]),  # Offset above object for grasping
        wait_for_activation: int = 5,
        activate_keyboard_listener: bool = True,
    ):
        """
        Args:
            robot_model: Robot kinematic model
            retargeting_ik: IK solver for body pose
            trajectory_duration: Time to reach target (seconds)
            grasp_offset: Offset from object center to grasp point (x, y, z in meters)
            wait_for_activation: Seconds to wait before policy activation
            activate_keyboard_listener: Enable keyboard control
        """
        if activate_keyboard_listener:
            from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber

            self.keyboard_listener = KeyboardListenerSubscriber()
        else:
            self.keyboard_listener = None

        self.wait_for_activation = wait_for_activation
        self.robot_model = robot_model
        self.retargeting_ik = retargeting_ik
        self.is_active = False
        
        # Trajectory parameters
        self.trajectory_duration = trajectory_duration
        self.grasp_offset = grasp_offset
        
        # Trajectory state
        self.trajectory_start_time: Optional[float] = None
        self.start_wrist_pose: Optional[np.ndarray] = None
        self.target_wrist_pose: Optional[np.ndarray] = None
        self.target_bottle_pos: Optional[np.ndarray] = None
        
        # Current observation
        self.observation: Optional[Dict[str, Any]] = None

        # Calibrate initial wrist poses from robot model
        self._calibrate_initial_wrist_poses()

        # Latest wrist and finger data (initialized with calibrated poses)
        self.latest_left_wrist_data = self.initial_left_wrist_pose.copy()
        self.latest_right_wrist_data = self.initial_right_wrist_pose.copy()
        self.latest_left_fingers_data = {"position": np.ones((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.ones((25, 4, 4))}
        
        # Store current hand joint positions to keep fingers at current state
        # Initialize with robot model's initial hand positions (from initial_body_pose)
        left_hand_indices = self.robot_model.get_hand_actuated_joint_indices(side="left")
        right_hand_indices = self.robot_model.get_hand_actuated_joint_indices(side="right")
        self.current_left_hand_q = self.robot_model.initial_body_pose[left_hand_indices].copy()
        self.current_right_hand_q = self.robot_model.initial_body_pose[right_hand_indices].copy()

        # Initial wrist orientations
        self.initial_left_wrist_rot = R.from_matrix(self.initial_left_wrist_pose[:3, :3])
        self.initial_right_wrist_rot = R.from_matrix(self.initial_right_wrist_pose[:3, :3])

    def _calibrate_initial_wrist_poses(self):
        """
        Calibrate initial wrist poses from the robot model's initial configuration.
        
        This ensures the wrists start from a known position based on the robot's
        default pose, rather than at the origin [0,0,0].
        """
        # Use the robot model's initial body pose to compute wrist positions
        q_initial = self.robot_model.initial_body_pose.copy()
        
        # Compute forward kinematics for the initial configuration
        self.robot_model.cache_forward_kinematics(q_initial)
        
        # Get wrist frame names
        if self.robot_model.supplemental_info is None:
            left_wrist_name = "left_wrist_yaw_link"
            right_wrist_name = "right_wrist_yaw_link"
        else:
            left_wrist_name = self.robot_model.supplemental_info.hand_frame_names["left"]
            right_wrist_name = self.robot_model.supplemental_info.hand_frame_names["right"]
        
        # Get initial wrist placements
        left_wrist_placement = self.robot_model.frame_placement(left_wrist_name)
        right_wrist_placement = self.robot_model.frame_placement(right_wrist_name)
        
        # Convert SE3 placements to 4x4 matrices
        self.initial_left_wrist_pose = np.eye(4)
        self.initial_left_wrist_pose[:3, :3] = left_wrist_placement.rotation
        self.initial_left_wrist_pose[:3, 3] = left_wrist_placement.translation
        
        self.initial_right_wrist_pose = np.eye(4)
        self.initial_right_wrist_pose[:3, :3] = right_wrist_placement.rotation
        self.initial_right_wrist_pose[:3, 3] = right_wrist_placement.translation
        
        # Reset forward kinematics back to zero configuration
        self.robot_model.reset_forward_kinematics()
        
        print(f"Calibrated initial wrist poses:")
        print(f"  Left wrist position: {self.initial_left_wrist_pose[:3, 3]}")
        print(f"  Right wrist position: {self.initial_right_wrist_pose[:3, 3]}")


    def set_observation(self, observation: Dict[str, Any]):
        """Update the current environment observation."""
        self.observation = observation
        
        # Store current hand positions to keep fingers at their current state
        if "left_hand_q" in observation:
            self.current_left_hand_q = observation["left_hand_q"].copy()
        if "right_hand_q" in observation:
            self.current_right_hand_q = observation["right_hand_q"].copy()

    def set_goal(self, goal: Dict[str, Any]):
        """
        Set high-level goal for the motion planner.
        
        Args:
            goal: Dictionary containing task specification
                - "task": task type (e.g., "pick_object", "place_object")
                - "target_object": name of target object (optional)
        """
        # For now, we automatically target the object in the scene
        # Future: support multiple task types and objects
        pass

    def _get_current_wrist_pose(self, side: str = "right") -> np.ndarray:
        """
        Get current wrist pose using forward kinematics.
        
        Args:
            side: "left" or "right"
            
        Returns:
            4x4 transformation matrix
        """
        if self.observation is None or "q" not in self.observation:
            # Return calibrated initial pose if no observation available
            if side == "left":
                return self.initial_left_wrist_pose.copy()
            else:
                return self.initial_right_wrist_pose.copy()
        
        q = self.observation["q"]
        self.robot_model.cache_forward_kinematics(q)
        
        # Get wrist frame name (with safety check)
        if self.robot_model.supplemental_info is None:
            # Fallback to default frame names
            wrist_frame_name = f"{side}_wrist_yaw_link"
        else:
            wrist_frame_name = self.robot_model.supplemental_info.hand_frame_names[side]
        
        wrist_placement = self.robot_model.frame_placement(wrist_frame_name)
        
        # Convert SE3 to 4x4 matrix
        wrist_matrix = np.eye(4)
        wrist_matrix[:3, :3] = wrist_placement.rotation
        wrist_matrix[:3, 3] = wrist_placement.translation
        
        return wrist_matrix

    def _compute_trajectory_point(self, t: float) -> np.ndarray:
        """
        Compute wrist pose at time t using smooth quintic polynomial interpolation.
        
        Uses a quintic (5th order) polynomial for smooth acceleration/deceleration.
        Boundary conditions: zero velocity and acceleration at start and end.
        
        Args:
            t: Normalized time in [0, 1]
            
        Returns:
            4x4 transformation matrix for target wrist pose
        """
        if self.start_wrist_pose is None or self.target_wrist_pose is None:
            return np.eye(4)
        
        # Clamp t to [0, 1]
        t = np.clip(t, 0.0, 1.0)
        
        # Quintic polynomial: s(t) = 6t^5 - 15t^4 + 10t^3
        # This gives zero velocity and acceleration at t=0 and t=1
        s = 6 * t**5 - 15 * t**4 + 10 * t**3
        
        # Linear interpolation of position
        start_pos = self.start_wrist_pose[:3, 3]
        target_pos = self.target_wrist_pose[:3, 3]
        interp_pos = start_pos + s * (target_pos - start_pos)
        
        # SLERP for rotation (spherical linear interpolation)
        start_rot = R.from_matrix(self.start_wrist_pose[:3, :3])
        target_rot = R.from_matrix(self.target_wrist_pose[:3, :3])
        
        # Use scipy's Slerp for proper quaternion interpolation
        key_rots = R.from_quat([start_rot.as_quat(), target_rot.as_quat()])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp(s)
        
        # Construct 4x4 transformation matrix
        result = np.eye(4)
        result[:3, :3] = interp_rot.as_matrix()
        result[:3, 3] = interp_pos
        
        return result

    def get_action(self, time: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate motion planning action for the current timestep.
        
        Args:
            time: Optional monotonic time (not used in this implementation)
        
        Returns:
            Dictionary containing:
                - left_wrist, right_wrist: 4x4 transformation matrices
                - left_fingers, right_fingers: finger joint positions
                - wrist_pose: concatenated [left_pos, left_quat, right_pos, right_quat]
                - target_upper_body_pose: joint targets from IK
        """
        # Handle keyboard activation
        self.check_activation()

        action: Dict[str, Any] = {}

        if not self.is_active:
            # Return current pose when inactive
            left_wrist_matrix = self.latest_left_wrist_data
            right_wrist_matrix = self.latest_right_wrist_data
        else:
            # Initialize trajectory on first active step
            if self.trajectory_start_time is None:
                self._initialize_trajectory()
            
            # Compute current trajectory point
            if self.trajectory_start_time is not None:
                elapsed_time = time_module.monotonic() - self.trajectory_start_time
                t = elapsed_time / self.trajectory_duration
            else:
                t = 0.0
            
            # Generate right wrist trajectory (for grasping)
            right_wrist_matrix = self._compute_trajectory_point(t)
            self.latest_right_wrist_data = right_wrist_matrix
            
            # Keep left wrist at current position (not used for grasping)
            left_wrist_matrix = self.latest_left_wrist_data
            
            # if t >= 1.0:
            #     print(f"Trajectory complete! Right wrist reached target.")

        # Convert matrices to pose format (position + quaternion)
        left_wrist_pose = self._matrix_to_pose(left_wrist_matrix)
        right_wrist_pose = self._matrix_to_pose(right_wrist_matrix)

        # Construct IK data for retargeting
        if self.robot_model.supplemental_info is None:
            left_wrist_name = "left_wrist_yaw_link"
            right_wrist_name = "right_wrist_yaw_link"
        else:
            left_wrist_name = self.robot_model.supplemental_info.hand_frame_names["left"]
            right_wrist_name = self.robot_model.supplemental_info.hand_frame_names["right"]

        body_data = {
            left_wrist_name: left_wrist_matrix,
            right_wrist_name: right_wrist_matrix,
        }
        
        ik_data = {
            "body_data": body_data,
            "left_hand_data": self.latest_left_fingers_data,
            "right_hand_data": self.latest_right_fingers_data,
        }

        # Combine all action data
        action.update(
            {
                "left_wrist": left_wrist_matrix,
                "right_wrist": right_wrist_matrix,
                "left_fingers": self.latest_left_fingers_data,
                "right_fingers": self.latest_right_fingers_data,
                "wrist_pose": np.concatenate([left_wrist_pose, right_wrist_pose]),
                "ik_data": ik_data,
            }
        )

        # Run retargeting IK to get joint targets
        self.retargeting_ik.set_goal(ik_data)
        action["target_upper_body_pose"] = self.retargeting_ik.get_action()
        
        # Override hand joint positions with current hand state to keep fingers open
        # Get the full configuration from the IK solver
        full_q = self.retargeting_ik._most_recent_q.copy()
        
        # Update hand joints with stored positions
        left_hand_indices = self.robot_model.get_hand_actuated_joint_indices(side="left")
        full_q[left_hand_indices] = self.current_left_hand_q
        
        right_hand_indices = self.robot_model.get_hand_actuated_joint_indices(side="right")
        full_q[right_hand_indices] = self.current_right_hand_q
        
        # Extract upper body joints from the modified configuration
        upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")
        action["target_upper_body_pose"] = full_q[upper_body_indices]

        return action

    def _initialize_trajectory(self):
        """Initialize a new trajectory to the bottle."""
        if self.observation is None:
            print("Failed to initialize trajectory: No observation available")
            return

        self.trajectory_start_time = time_module.monotonic()

        # Get current right wrist pose
        self.start_wrist_pose = self._get_current_wrist_pose(side="right")
        self.latest_left_wrist_data = self._get_current_wrist_pose(side="left")

        # Get object position from observation
        if "obj_pos" not in self.observation:
            print("Failed to initialize trajectory: 'obj_pos' not in observation")
            return
        
        obj_pos_world = np.array(self.observation["obj_pos"])

        # Get robot base link pose from observation
        base_pose = self.observation.get("floating_base_pose", None)

        # Transform obj_pos in the robot base frame if base_pose is available
        obj_pos_robot = None
        if base_pose is not None:
            base_pos = np.array(base_pose[:3])
            base_quat = np.array(base_pose[3:7])  # [w, x, y, z] format
            base_rot = R.from_quat(base_quat, scalar_first=True)  # Tell scipy it's [w, x, y, z]
            obj_pos_robot = base_rot.inv().apply(obj_pos_world - base_pos)

        if obj_pos_robot is None:
            print("Failed to initialize trajectory: Object position not available in robot frame")
            return

        # Compute target wrist pose (object position + grasp offset)
        target_pos = obj_pos_robot + self.grasp_offset

        self.target_wrist_pose = np.eye(4)
        self.target_wrist_pose[:3, :3] = self.initial_right_wrist_rot.as_matrix() # Keep initial orientation
        self.target_wrist_pose[:3, 3] = target_pos

        print(f"Initialized trajectory:")
        print(f"  Start position: {self.start_wrist_pose[:3, 3]}")
        print(f"  Robot base pose: {base_pose}")
        print(f"  Object position (world): {obj_pos_world}")
        print(f"  Object position (robot): {obj_pos_robot}")
        print(f"  Target position: {target_pos}")
        print(f"  Duration: {self.trajectory_duration}s")

    def _matrix_to_pose(self, matrix: np.ndarray) -> np.ndarray:
        """
        Convert 4x4 transformation matrix to pose vector.
        
        Args:
            matrix: 4x4 transformation matrix
            
        Returns:
            7D vector [x, y, z, qw, qx, qy, qz]
        """
        pos = matrix[:3, 3]
        rot = R.from_matrix(matrix[:3, :3])
        # Get quaternion in [w, x, y, z] format
        quat_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.concatenate([pos, quat_wxyz])

    def check_activation(self):
        """Handle keyboard-based activation/deactivation."""
        key = self.keyboard_listener.read_msg() if self.keyboard_listener else ""
        toggle_activation_by_keyboard = key == "l"
        reset_by_keyboard = key == "k"

        if reset_by_keyboard:
            print("Resetting motion planning policy")
            self.reset()

        if toggle_activation_by_keyboard:
            self.is_active = not self.is_active
            if self.is_active:
                print("Starting motion planning policy")
                if self.wait_for_activation > 0:
                    print(f"Waiting {self.wait_for_activation} seconds before starting...")
                    for i in range(self.wait_for_activation, 0, -1):
                        print(f"Starting in {i}...")
                        time_module.sleep(1)
                print("Motion planning policy activated!")
            else:
                print("Stopping motion planning policy")
                self.trajectory_start_time = None

    def close(self) -> None:
        """Clean up resources."""
        pass

    @contextmanager
    def activate(self):
        """Context manager for policy activation."""
        try:
            yield self
        finally:
            self.close()

    def handle_keyboard_button(self, keycode):
        """
        Handle keyboard input for activation control.
        
        Args:
            keycode: Keyboard key pressed
                - "l": Toggle policy activation
                - "k": Reset policy
        """
        if keycode == "l":
            self.is_active = not self.is_active
            if not self.is_active:
                self.trajectory_start_time = None
        if keycode == "k":
            print("Resetting motion planning policy")
            self.reset()

    def activate_policy(self, wait_for_activation: int = 5):
        """
        Programmatically activate the motion planning policy.
        
        Args:
            wait_for_activation: Seconds to wait before activation
        """
        self.is_active = True
        if wait_for_activation > 0:
            print(f"Waiting {wait_for_activation} seconds before starting...")
            for i in range(self.wait_for_activation, 0, -1):
                print(f"Starting in {i}...")
                time_module.sleep(1)
        print("Motion planning policy activated!")

    def reset(self, wait_for_activation: int = 5, auto_activate: bool = False):
        """
        Reset the motion planning policy to initial state.
        
        Args:
            wait_for_activation: Seconds to wait before activation (if auto_activate=True)
            auto_activate: Whether to automatically activate after reset
        """
        self.retargeting_ik.reset()
        self.is_active = False
        self.trajectory_start_time = None
        self.start_wrist_pose = None
        self.target_wrist_pose = None
        self.target_bottle_pos = None
        
        # Reset to calibrated initial wrist poses
        self.latest_left_wrist_data = self.initial_left_wrist_pose.copy()
        self.latest_right_wrist_data = self.initial_right_wrist_pose.copy()
        self.latest_left_fingers_data = {"position": np.ones((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.ones((25, 4, 4))}
        
        # Reset stored hand positions to initial configuration (will be updated from observation)
        left_hand_indices = self.robot_model.get_hand_actuated_joint_indices(side="left")
        right_hand_indices = self.robot_model.get_hand_actuated_joint_indices(side="right")
        self.current_left_hand_q = self.robot_model.initial_body_pose[left_hand_indices].copy()
        self.current_right_hand_q = self.robot_model.initial_body_pose[right_hand_indices].copy()

        if auto_activate:
            self.activate_policy(wait_for_activation)
