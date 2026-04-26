"""
GR00T Deployment Bridge

Bridges the GEAR-SONIC deployment to the GR00T inference server by:
1. Subscribing to robot state from deployment (ZMQ SUB on port 5557)
2. Subscribing to camera images from camera server (ZMQ SUB on port 5558)
3. Sending observations (state + video) to GR00T server via REQ/REP (port 5555)
4. Publishing actions back to deployment (ZMQ PUB on port 5556)

Architecture:
  
  Deployment → [Port 5557] → Bridge → [REQ/REP Port 5555] → GR00T Server
  Camera     → [Port 5558] → Bridge → [REQ/REP Port 5555] → GR00T Server
  Deployment ← [Port 5556] ← Bridge ← [REQ/REP Port 5555] ← GR00T Server

Usage:
    python gear_sonic/scripts/groot_deployment_bridge.py \\
        --groot-host localhost \\
        --groot-port 5555 \\
        --deployment-state-port 5557 \\
        --camera-port 5558 \\
        --action-publish-port 5556 \\
        --lang-prompt "Walk forward slowly"
"""

from dataclasses import dataclass
from pathlib import Path
import base64
import json
import sys
import time
from typing import Any

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq
import tyro


def _bootstrap_venv():
    """Re-exec with the .venv_data_collection Python if dependencies not available."""
    try:
        import msgpack  # noqa: F401
        import zmq  # noqa: F401
        import numpy  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_data_collection" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: Required packages not installed and .venv_data_collection not found.\n"
            "  Run: bash install_scripts/install_data_collection.sh\n"
            "  Or install: pip install pyzmq msgpack numpy"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    import os
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()


class MsgSerializer:
    """Compatible with GR00T server's msgpack serialization."""
    
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes, raw=False)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            import io
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            import io
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class BridgeConfig:
    """Configuration for the GR00T deployment bridge."""

    groot_host: str = "localhost"
    """Host where the GR00T PolicyServer is running."""

    groot_port: int = 5555
    """Port where the GR00T PolicyServer is listening (REQ/REP)."""

    deployment_state_host: str = "localhost"
    """Host where deployment publishes state (usually localhost)."""

    deployment_state_port: int = 5557
    """Port where deployment publishes robot state (ZMQ PUB)."""

    deployment_state_topic: str = "g1_debug"
    """Topic that deployment publishes state on."""

    action_publish_port: int = 5556
    """Port to publish actions for deployment to consume (ZMQ PUB)."""

    action_publish_topic: str = "pose"
    """Topic to publish actions on (deployment expects 'pose')."""
    
    camera_host: str = "localhost"
    """Host where camera server publishes images."""
    
    camera_port: int = 5558
    """Port where camera server publishes images (ZMQ PUB)."""
    
    camera_topic: str = ""
    """Topic that camera server publishes on (empty string subscribes to all)."""

    groot_api_token: str | None = None
    """Optional API token for GR00T server authentication."""
    
    lang_prompt: str = "Walk forward"
    """Language prompt / task description to send to GR00T."""

    frequency: float = 50.0
    """Target frequency (Hz) for the bridge control loop."""

    verbose: bool = True
    """Enable verbose logging."""


class Gr00tDeploymentBridge:
    """Bridge between GEAR-SONIC deployment and GR00T inference server."""

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.context = zmq.Context()
        
        # Subscribe to deployment state
        print(f"Subscribing to deployment state at {config.deployment_state_host}:{config.deployment_state_port}")
        print(f"  Topic filter: '{config.deployment_state_topic}'")
        self.state_sub = self.context.socket(zmq.SUB)
        self.state_sub.connect(f"tcp://{config.deployment_state_host}:{config.deployment_state_port}")
        self.state_sub.setsockopt_string(zmq.SUBSCRIBE, config.deployment_state_topic)
        self.state_sub.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        print(f"  Connected and subscribed to topic '{config.deployment_state_topic}'")
        
        # Publish actions to deployment
        print(f"Publishing actions on port {config.action_publish_port}")
        print(f"  Topic prefix: '{config.action_publish_topic}'")
        self.action_pub = self.context.socket(zmq.PUB)
        self.action_pub.bind(f"tcp://*:{config.action_publish_port}")
        self.action_pub.setsockopt(zmq.SNDHWM, 10)
        self.action_pub.setsockopt(zmq.LINGER, 0)
        print(f"  Bound to port {config.action_publish_port}")
        
        # Subscribe to camera images
        print(f"Subscribing to camera at {config.camera_host}:{config.camera_port}")
        print(f"  Topic filter: '{config.camera_topic}'")
        self.camera_sub = self.context.socket(zmq.SUB)
        self.camera_sub.connect(f"tcp://{config.camera_host}:{config.camera_port}")
        self.camera_sub.setsockopt_string(zmq.SUBSCRIBE, config.camera_topic)
        self.camera_sub.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout (non-blocking)
        self.camera_sub.setsockopt(zmq.CONFLATE, 1)  # Keep only latest image
        print(f"  Connected and subscribed to topic '{config.camera_topic if config.camera_topic else '<all>'}'")
        
        self.last_camera_frame = None  # Store latest camera frame
        self._camera_connected = False  # Track first camera frame
        
        # Connect to GR00T server
        print(f"Connecting to GR00T server at {config.groot_host}:{config.groot_port}")
        self.groot_req = self.context.socket(zmq.REQ)
        self.groot_req.connect(f"tcp://{config.groot_host}:{config.groot_port}")
        self.groot_req.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.groot_req.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout
        
        self.running = False
        self.loop_count = 0
        self.last_state = None
        
        # Test GR00T server connection
        self._test_groot_connection()

    def _test_groot_connection(self):
        """Test connection to GR00T server with a ping."""
        try:
            request = {"endpoint": "ping"}
            if self.config.groot_api_token:
                request["api_token"] = self.config.groot_api_token
            
            self.groot_req.send(MsgSerializer.to_bytes(request))
            response_bytes = self.groot_req.recv()
            response = MsgSerializer.from_bytes(response_bytes)
            
            if response.get("status") == "ok":
                print(f"✓ GR00T server connection successful: {response.get('message')}")
            else:
                print(f"⚠ GR00T server responded but status not ok: {response}")
        except zmq.error.Again:
            print("✗ GR00T server connection timeout - is the server running?")
            raise RuntimeError("Cannot connect to GR00T server")
        except Exception as e:
            print(f"✗ Error testing GR00T connection: {e}")
            raise

    def _extract_observations(self, state_data: dict) -> dict:
        """
        Extract observations from deployment state for GR00T.
        
        Maps deployment state to the modality keys defined in g1_gear_sonic_config.py:
        - Video: ego_view (camera image)
        - Joint states: left_leg, right_leg, waist, left_arm, left_hand, right_arm, right_hand
        - End-effector states: left_wrist_pos, left_wrist_abs_quat, right_wrist_pos, right_wrist_abs_quat
        - Base/orientation: root_orientation, projected_gravity, cpp_rotation_offset, init_base_quat
        
        Deployment provides joint positions in IsaacLab order (body_q: 29 joints):
        [left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
        
        Note: Hand positions come from separate fields (left_hand_q, right_hand_q)
        """
        obs = {}
        state_obs = {}  # All state observations go here
        
        # VIDEO: Add camera image (required by embodiment config)
        if self.last_camera_frame is not None:
            # Reshape to (B, T, H, W, C) - batch=1, temporal=1, spatial=(H,W,C)
            video_frame = self.last_camera_frame
            if video_frame.ndim == 3:
                # Add batch and temporal dimensions: (H, W, C) -> (1, 1, H, W, C)
                video_frame = video_frame[np.newaxis, np.newaxis, ...]
            obs["video"] = {
                "ego_view": video_frame
            }
        else:
            print("WARNING: No camera frame available - GR00T requires 'video.ego_view'")
            # Create a dummy black frame (1, 1, 480, 640, 3) with batch and temporal dims
            obs["video"] = {
                "ego_view": np.zeros((1, 1, 480, 640, 3), dtype=np.uint8)
            }
        
        # STATE: Extract body joints (29 DOF) and split into modality groups
        # NOTE: All state observations need shape (B, T, D) - batch, temporal, features
        # Based on processor_config.json, delta_indices=[0] means T=1 (current timestep only)
        
        # Extract body joints (29 DOF) and split into modality groups
        if "body_q" in state_data:
            body_q = np.array(state_data["body_q"], dtype=np.float32)
            
            # Split body_q into modality groups (IsaacLab order)
            # Add batch and temporal dimensions: (n,) -> (1, 1, n)
            state_obs["left_leg"] = body_q[0:6][np.newaxis, np.newaxis, :]      # (1, 1, 6)
            state_obs["right_leg"] = body_q[6:12][np.newaxis, np.newaxis, :]    # (1, 1, 6)
            state_obs["waist"] = body_q[12:15][np.newaxis, np.newaxis, :]       # (1, 1, 3)
            state_obs["left_arm"] = body_q[15:22][np.newaxis, np.newaxis, :]    # (1, 1, 7)
            state_obs["right_arm"] = body_q[22:29][np.newaxis, np.newaxis, :]   # (1, 1, 7)
        
        # Hand joint positions (7 DOF each)
        if "left_hand_q" in state_data:
            state_obs["left_hand"] = np.array(state_data["left_hand_q"], dtype=np.float32)[np.newaxis, np.newaxis, :]  # (1, 1, 7)
        if "right_hand_q" in state_data:
            state_obs["right_hand"] = np.array(state_data["right_hand_q"], dtype=np.float32)[np.newaxis, np.newaxis, :]  # (1, 1, 7)
        
        # End-effector states - REQUIRED by GR00T (all 15 state keys must be present)
        # Use deployment data if available, otherwise use placeholder values
        # TODO: Compute from forward kinematics for accurate wrist poses
        
        # Left wrist position (3D)
        if "left_wrist_pos" in state_data:
            state_obs["left_wrist_pos"] = np.array(state_data["left_wrist_pos"], dtype=np.float32)[np.newaxis, np.newaxis, :]
        else:
            # Placeholder: approximate position in robot frame
            state_obs["left_wrist_pos"] = np.array([0.3, 0.2, 0.5], dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        # Left wrist absolute quaternion (4D: w, x, y, z)
        if "left_wrist_quat" in state_data:
            state_obs["left_wrist_abs_quat"] = np.array(state_data["left_wrist_quat"], dtype=np.float32)[np.newaxis, np.newaxis, :]
        else:
            # Placeholder: identity quaternion
            state_obs["left_wrist_abs_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        # Right wrist position (3D)
        if "right_wrist_pos" in state_data:
            state_obs["right_wrist_pos"] = np.array(state_data["right_wrist_pos"], dtype=np.float32)[np.newaxis, np.newaxis, :]
        else:
            # Placeholder: approximate position in robot frame
            state_obs["right_wrist_pos"] = np.array([0.3, -0.2, 0.5], dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        # Right wrist absolute quaternion (4D: w, x, y, z)
        if "right_wrist_quat" in state_data:
            state_obs["right_wrist_abs_quat"] = np.array(state_data["right_wrist_quat"], dtype=np.float32)[np.newaxis, np.newaxis, :]
        else:
            # Placeholder: identity quaternion
            state_obs["right_wrist_abs_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        # Root orientation (base quaternion) - REQUIRED
        if "base_quat" in state_data:
            state_obs["root_orientation"] = np.array(state_data["base_quat"], dtype=np.float32)[np.newaxis, np.newaxis, :]
        else:
            # Default: identity quaternion (upright)
            state_obs["root_orientation"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        # Projected gravity (3D) - REQUIRED
        if "projected_gravity" in state_data:
            state_obs["projected_gravity"] = np.array(state_data["projected_gravity"], dtype=np.float32)[np.newaxis, np.newaxis, :]
        else:
            # Default: gravity in world frame (pointing down)
            state_obs["projected_gravity"] = np.array([0.0, 0.0, -9.81], dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        # Rotation offset (quaternion) - REQUIRED
        if "cpp_rotation_offset" in state_data:
            state_obs["cpp_rotation_offset"] = np.array(state_data["cpp_rotation_offset"], dtype=np.float32)[np.newaxis, np.newaxis, :]
        else:
            # Default: identity quaternion (no offset)
            state_obs["cpp_rotation_offset"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        # Initial base quaternion - REQUIRED
        if "init_base_quat" in state_data:
            state_obs["init_base_quat"] = np.array(state_data["init_base_quat"], dtype=np.float32)[np.newaxis, np.newaxis, :]
        else:
            # Use current base quat as fallback
            if "base_quat" in state_data:
                state_obs["init_base_quat"] = np.array(state_data["base_quat"], dtype=np.float32)[np.newaxis, np.newaxis, :]
            else:
                # Default: identity quaternion (upright)
                state_obs["init_base_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        # Group all state observations under "state" key
        obs["state"] = state_obs
        
        # LANGUAGE: Task description / prompt
        # Language observations need shape (B, T) where B=batch, T=temporal (horizon)
        # Each element [b, t] is a string. delta_indices=[0] means T=1 (current timestep)
        obs["language"] = {
            "annotation.human.task_description": [[self.config.lang_prompt]]  # Shape (1, 1)
        }
        
        return obs

    def _request_action_from_groot(self, observations: dict) -> dict | None:
        """Send observations to GR00T and receive action."""
        try:
            request = {
                "endpoint": "get_action",
                "data": {
                    "observation": observations,  # Nest under "observation" key
                    "options": None  # Optional configuration
                }
            }
            if self.config.groot_api_token:
                request["api_token"] = self.config.groot_api_token
            
            self.groot_req.send(MsgSerializer.to_bytes(request))
            response_bytes = self.groot_req.recv()
            response = MsgSerializer.from_bytes(response_bytes)
            
            if isinstance(response, dict) and "error" in response:
                print(f"✗ GR00T error: {response['error']}")
                return None
            
            # Server returns [action, info] (tuple serialized as list by msgpack)
            if isinstance(response, (list, tuple)) and len(response) >= 2:
                action, info = response[0], response[1]
                if self.config.verbose and self.loop_count % 50 == 0:
                    print(f"  GR00T info: {info}")
                return action
            else:
                print(f"✗ Unexpected response format from GR00T: {type(response)}")
                return None
            
        except zmq.error.Again:
            print("✗ GR00T server timeout waiting for action")
            return None
        except Exception as e:
            print(f"✗ Error requesting action from GR00T: {e}")
            return None

    def _publish_action(self, action_data: dict):
        """
        Publish action to deployment in ZMQ Protocol v4 format.
        
        Protocol v4 is designed for token-based control (SONIC policy decoder):
        REQUIRED: token_state (motion token array - decoded by SONIC into joint commands)
        OPTIONAL: left_hand_joints, right_hand_joints, frame_index, body_quat_w
        
        GR00T response mapping (with action horizon extraction):
        - action_data["motion_token"][0, 0, :] → token_state (64-dim latent)
        - action_data["left_hand_joints"][0, 0, :] → left_hand_joints (7-dim)
        - action_data["right_hand_joints"][0, 0, :] → right_hand_joints (7-dim)
        
        Note: GR00T returns actions with shape (B, H, D) where H=40 (action horizon).
        We extract the first action (horizon index 0) for immediate execution.
        
        Format: [topic_prefix] [1280-byte JSON header] [concatenated binary fields]
        """
        # Build JSON header describing the binary payload
        header = {
            "v": 4,  # Protocol version 4 (token-based streaming)
            "endian": "le" if sys.byteorder == 'little' else "be",
            "count": 1,  # Single frame
            "fields": []
        }
        
        # Collect binary data in the order we'll concatenate it
        binary_parts = []
        
        # REQUIRED: token_state (motion tokens from GR00T)
        # GR00T returns this as "motion_token" with shape (B, H, D) where H=40 (action horizon)
        # Extract first action: [0, 0, :] to get shape (D,)
        if "motion_token" in action_data:
            motion_token = np.array(action_data["motion_token"], dtype=np.float64)
            # Extract first action from horizon if needed
            if motion_token.ndim == 3:  # Shape (B, H, D)
                token_state = motion_token[0, 0, :]  # Extract (D,)
            elif motion_token.ndim == 2:  # Shape (H, D) - batch already squeezed
                token_state = motion_token[0, :]  # Extract (D,)
            else:  # Already (D,)
                token_state = motion_token
            
            token_state = token_state.astype(np.float64)  # Ensure dtype
            header["fields"].append({
                "name": "token_state",
                "dtype": "f64",
                "shape": list(token_state.shape)  # Should be [64] after extraction
            })
            binary_parts.append(token_state.tobytes())
        else:
            print("ERROR: No 'motion_token' in GR00T response - v4 protocol requires token_state")
            print(f"Available keys: {list(action_data.keys())}")
            return
        
        # OPTIONAL: frame_index (helps with synchronization)
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0
        frame_index = np.array([self._frame_counter], dtype=np.int32)
        self._frame_counter += 1
        header["fields"].append({
            "name": "frame_index",
            "dtype": "i32",
            "shape": [1]
        })
        binary_parts.append(frame_index.tobytes())
        
        # OPTIONAL: hand joint actions (if present in GR00T response)
        # Also extract first action from horizon: (B, H, D) -> (D,)
        if "left_hand_joints" in action_data:
            left_hand_raw = np.array(action_data["left_hand_joints"], dtype=np.float64)
            # Extract first action from horizon if needed
            if left_hand_raw.ndim == 3:  # Shape (B, H, D)
                left_hand = left_hand_raw[0, 0, :]  # Extract (D,)
            elif left_hand_raw.ndim == 2:  # Shape (H, D)
                left_hand = left_hand_raw[0, :]  # Extract (D,)
            else:  # Already (D,)
                left_hand = left_hand_raw
            
            left_hand = left_hand.astype(np.float64)
            header["fields"].append({
                "name": "left_hand_joints",
                "dtype": "f64",
                "shape": list(left_hand.shape)  # Should be [7] from your config
            })
            binary_parts.append(left_hand.tobytes())
        
        if "right_hand_joints" in action_data:
            right_hand_raw = np.array(action_data["right_hand_joints"], dtype=np.float64)
            # Extract first action from horizon if needed
            if right_hand_raw.ndim == 3:  # Shape (B, H, D)
                right_hand = right_hand_raw[0, 0, :]  # Extract (D,)
            elif right_hand_raw.ndim == 2:  # Shape (H, D)
                right_hand = right_hand_raw[0, :]  # Extract (D,)
            else:  # Already (D,)
                right_hand = right_hand_raw
            
            right_hand = right_hand.astype(np.float64)
            header["fields"].append({
                "name": "right_hand_joints",
                "dtype": "f64",
                "shape": list(right_hand.shape)  # Should be [7] from your config
            })
            binary_parts.append(right_hand.tobytes())
        
        # Serialize JSON header and pad to exactly 1280 bytes
        header_json = json.dumps(header, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8')
        
        HEADER_SIZE = 1280
        if len(header_bytes) >= HEADER_SIZE:
            print(f"ERROR: JSON header too large ({len(header_bytes)} bytes, max {HEADER_SIZE})")
            return
        
        # Null-pad to HEADER_SIZE
        header_padded = header_bytes + b'\x00' * (HEADER_SIZE - len(header_bytes))
        
        # Concatenate: topic + header + binary payload
        topic_bytes = self.config.action_publish_topic.encode('utf-8')
        binary_payload = b''.join(binary_parts)
        
        message = topic_bytes + header_padded + binary_payload
        
        if self.config.verbose and self.loop_count % 50 == 0:
            print(f"  Publishing action: {len(message)} bytes (topic: {len(topic_bytes)}, header: {HEADER_SIZE}, payload: {len(binary_payload)})")
        
        self.action_pub.send(message, flags=zmq.NOBLOCK)

    def run(self):
        """Main bridge loop."""
        self.running = True
        loop_dt = 1.0 / self.config.frequency
        
        print("\n" + "=" * 60)
        print("  GR00T Deployment Bridge Running")
        print("=" * 60)
        print(f"  Deployment state: {self.config.deployment_state_host}:{self.config.deployment_state_port}")
        print(f"  GR00T server: {self.config.groot_host}:{self.config.groot_port}")
        print(f"  Action publish: port {self.config.action_publish_port}")
        print(f"  Camera: {self.config.camera_host}:{self.config.camera_port} (topic: {'<all>' if not self.config.camera_topic else repr(self.config.camera_topic)})")
        print(f"  Language prompt: '{self.config.lang_prompt}'")
        print(f"  Target frequency: {self.config.frequency} Hz")
        print("=" * 60)
        print("\nWaiting for deployment state and camera frames...")
        
        # Give publisher time to initialize
        time.sleep(0.5)
        
        while self.running:
            loop_start = time.time()
            
            try:
                # 0. Poll for latest camera frame (non-blocking)
                try:
                    camera_msg = self.camera_sub.recv()
                    # Camera server publishes msgpack with numpy arrays (uses msgpack_numpy)
                    camera_data = msgpack.unpackb(camera_msg, object_hook=m.decode, raw=False)
                    
                    # Expected format from SensorServer: {"images": {"ego_view": array, ...}, ...}
                    if isinstance(camera_data, dict) and 'images' in camera_data:
                        images = camera_data['images']
                        # Use ego_view camera if available
                        if 'ego_view' in images:
                            img = images['ego_view']
                            if isinstance(img, np.ndarray):
                                # Direct numpy array
                                self.last_camera_frame = img.astype(np.uint8)
                            elif isinstance(img, str):
                                # Base64-encoded JPEG image - decode it
                                color_data = base64.b64decode(img)
                                color_array = np.frombuffer(color_data, dtype=np.uint8)
                                self.last_camera_frame = cv2.imdecode(color_array, cv2.IMREAD_COLOR)
                            else:
                                print(f"  WARNING: ego_view is unexpected type: {type(img)}")
                            
                            # Log first connection
                            if self.last_camera_frame is not None and not self._camera_connected:
                                self._camera_connected = True
                                print(f"✓ Camera connected: ego_view {self.last_camera_frame.shape} @ {self.config.camera_port}")
                            if self.config.verbose and self.loop_count % 50 == 0 and self.last_camera_frame is not None:
                                print(f"  Camera frame (ego_view): {self.last_camera_frame.shape}")
                        else:
                            if self.loop_count % 50 == 0:
                                print(f"  WARNING: No 'ego_view' in camera images. Available: {list(images.keys())}")
                    else:
                        if self.loop_count % 50 == 0:
                            print(f"  WARNING: Unexpected camera data format: {type(camera_data)}, keys: {list(camera_data.keys()) if isinstance(camera_data, dict) else 'N/A'}")
                except zmq.error.Again:
                    # No camera frame available (non-blocking poll)
                    pass
                
                # 1. Receive state from deployment
                message = self.state_sub.recv()
                
                if self.loop_count % 50 == 0:
                    print(f"\n[{self.loop_count}] ✓ Received message ({len(message)} bytes)")
                
                # Strip topic prefix
                topic_len = len(self.config.deployment_state_topic)
                if len(message) < topic_len:
                    print(f"  ⚠ Message too short ({len(message)} bytes), expected topic prefix")
                    continue
                    
                msgpack_data = message[topic_len:]
                if self.loop_count % 50 == 0:
                    print(f"  Topic prefix ({topic_len} bytes): {message[:topic_len]}")
                    print(f"  Msgpack data: {len(msgpack_data)} bytes")
                
                state_data = msgpack.unpackb(msgpack_data, raw=False)
                
                self.last_state = state_data
                
                if self.config.verbose and self.loop_count % 50 == 0:
                    print(f"  State index: {state_data.get('index', 'N/A')}")
                    print(f"  State keys: {list(state_data.keys())[:10]}...")  # First 10 keys
                
                # 2. Extract observations for GR00T
                observations = self._extract_observations(state_data)
                
                if self.config.verbose and self.loop_count % 50 == 0:
                    print(f"  Extracted observations: {list(observations.keys())}")
                
                # 3. Request action from GR00T
                action_response = self._request_action_from_groot(observations)
                
                if action_response is None:
                    print("  ⚠ No action from GR00T, skipping")
                    continue
                
                if self.config.verbose and self.loop_count % 50 == 0:
                    print(f"  Received action from GR00T: {list(action_response.keys())}")
                
                # 4. Publish action to deployment
                self._publish_action(action_response)
                
                self.loop_count += 1
                
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = loop_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif self.config.verbose and self.loop_count % 50 == 0:
                    print(f"  ⚠ Loop running slow: {elapsed*1000:.1f}ms (target: {loop_dt*1000:.1f}ms)")
                
            except zmq.error.Again:
                # Timeout waiting for deployment state
                if self.loop_count == 0:
                    print("  Still waiting for deployment state...")
                    print(f"    Listening on: tcp://{self.config.deployment_state_host}:{self.config.deployment_state_port}")
                    print(f"    Topic filter: '{self.config.deployment_state_topic}'")
                    print(f"    Make sure deployment is publishing on this port!")
                elif self.loop_count % 10 == 0:
                    print(f"  [{self.loop_count}] Still waiting... (no messages received)")
                continue
            except KeyboardInterrupt:
                print("\n\nShutting down bridge...")
                self.running = False
            except Exception as e:
                print(f"\n✗ Error in bridge loop: {e}")
                import traceback
                traceback.print_exc()
                continue

    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        self.state_sub.close()
        self.camera_sub.close()
        self.action_pub.close()
        self.groot_req.close()
        self.context.term()
        print("Bridge shutdown complete")


def main(config: BridgeConfig):
    bridge = Gr00tDeploymentBridge(config)
    
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    config = tyro.cli(BridgeConfig)
    main(config)
