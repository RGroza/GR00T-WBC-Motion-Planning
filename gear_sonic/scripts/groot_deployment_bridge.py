"""
GR00T Deployment Bridge

Bridges the GEAR-SONIC deployment to the GR00T inference server by:
1. Subscribing to robot state from deployment (ZMQ SUB on port 5557)
2. Sending observations to GR00T server via REQ/REP (port 5555)
3. Publishing actions back to deployment (ZMQ PUB on port 5556)

Architecture:
  
  Deployment → [Port 5557] → Bridge → [REQ/REP Port 5555] → GR00T Server
  Deployment ← [Port 5556] ← Bridge ← [REQ/REP Port 5555] ← GR00T Server

Usage:
    python gear_sonic/scripts/groot_deployment_bridge.py \\
        --groot-host localhost \\
        --groot-port 5555 \\
        --deployment-state-port 5557 \\
        --action-publish-port 5556
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import msgpack
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

    groot_api_token: str | None = None
    """Optional API token for GR00T server authentication."""

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
        
        The exact format depends on what GR00T expects. This is a template
        that you'll need to customize based on your embodiment's modality config.
        """
        # Common fields from deployment state
        obs = {}
        
        # Base IMU
        if "base_quat" in state_data:
            obs["base_quat"] = np.array(state_data["base_quat"], dtype=np.float32)
        if "base_ang_vel" in state_data:
            obs["base_ang_vel"] = np.array(state_data["base_ang_vel"], dtype=np.float32)
        
        # Body joints
        if "body_q" in state_data:
            obs["joint_pos"] = np.array(state_data["body_q"], dtype=np.float32)
        if "body_dq" in state_data:
            obs["joint_vel"] = np.array(state_data["body_dq"], dtype=np.float32)
        
        # Hand joints
        if "left_hand_q" in state_data:
            obs["left_hand_joint_pos"] = np.array(state_data["left_hand_q"], dtype=np.float32)
        if "right_hand_q" in state_data:
            obs["right_hand_joint_pos"] = np.array(state_data["right_hand_q"], dtype=np.float32)
        
        # Last actions (for history)
        if "last_action" in state_data:
            obs["last_body_action"] = np.array(state_data["last_action"], dtype=np.float32)
        if "last_left_hand_action" in state_data:
            obs["last_left_hand_action"] = np.array(state_data["last_left_hand_action"], dtype=np.float32)
        if "last_right_hand_action" in state_data:
            obs["last_right_hand_action"] = np.array(state_data["last_right_hand_action"], dtype=np.float32)
        
        # VR 3-point data (if using)
        if "vr_3point_position" in state_data:
            obs["vr_position"] = np.array(state_data["vr_3point_position"], dtype=np.float32)
        if "vr_3point_orientation" in state_data:
            obs["vr_orientation"] = np.array(state_data["vr_3point_orientation"], dtype=np.float32)
        
        # Token state (for autoregressive models)
        if "token_state" in state_data and len(state_data["token_state"]) > 0:
            obs["token_state"] = np.array(state_data["token_state"], dtype=np.float32)
        
        return obs

    def _request_action_from_groot(self, observations: dict) -> dict | None:
        """Send observations to GR00T and receive action."""
        try:
            request = {
                "endpoint": "get_action",
                "data": observations
            }
            if self.config.groot_api_token:
                request["api_token"] = self.config.groot_api_token
            
            self.groot_req.send(MsgSerializer.to_bytes(request))
            response_bytes = self.groot_req.recv()
            response = MsgSerializer.from_bytes(response_bytes)
            
            if "error" in response:
                print(f"✗ GR00T error: {response['error']}")
                return None
            
            return response
            
        except zmq.error.Again:
            print("✗ GR00T server timeout waiting for action")
            return None
        except Exception as e:
            print(f"✗ Error requesting action from GR00T: {e}")
            return None

    def _publish_action(self, action_data: dict):
        """
        Publish action to deployment in the format it expects.
        
        The deployment expects a packed message with JSON header + binary payload.
        Format matches what ZMQPackedMessageSubscriber expects (see zmq_packed_message_subscriber.hpp).
        """
        # TODO: Convert action_data to the packed message format
        # For now, just publish as msgpack (you may need to adapt this)
        
        # Create a simple packed message
        # This is a simplified version - you may need to match the exact binary format
        topic_bytes = self.config.action_publish_topic.encode('utf-8')
        action_bytes = MsgSerializer.to_bytes(action_data)
        
        # Send topic + data
        message = topic_bytes + action_bytes
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
        print(f"  Target frequency: {self.config.frequency} Hz")
        print("=" * 60)
        print("\nWaiting for deployment state...")
        
        # Give publisher time to initialize
        time.sleep(0.5)
        
        while self.running:
            loop_start = time.time()
            
            try:
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
