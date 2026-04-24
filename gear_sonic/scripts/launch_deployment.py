"""
All-in-one tmux launcher for SONIC deployment with GR00T inference.

Starts the deployment stack in a tmux session for running the robot with
GR00T inference server:

    Window 0 — deployment (4 panes when using --sim --use-bridge):
    ┌──────────────────┬──────────────────┐
    │ Pane 0:          │ Pane 1:          │
    │ C++ Deploy       │ GR00T Bridge     │
    │ (gear_sonic_     │ (REQ/REP to      │
    │  deploy)         │  GR00T server)   │
    ├──────────────────┼──────────────────┤
    │ Pane 2:          │ Pane 3:          │
    │ MuJoCo Sim       │ Camera Viewer    │
    │ (.venv_sim)      │ (.venv_data_     │
    │                  │  collection)     │
    └──────────────────┴──────────────────┘

    OR (3 panes when using --use-bridge without --sim):
    ┌──────────────────┬──────────────────┬──────────────────┐
    │ Pane 0:          │ Pane 1:          │ Pane 2:          │
    │ C++ Deploy       │ GR00T Bridge     │ Camera Viewer    │
    │ (gear_sonic_     │ (REQ/REP to      │ (.venv_data_     │
    │  deploy)         │  GR00T server)   │  collection)     │
    └──────────────────┴──────────────────┴──────────────────┘

    OR (2 panes when NOT using --use-bridge):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 1: Camera Viewer │
    │ (gear_sonic_deploy)   │ (.venv_data_collection)│
    │ - Connects to GR00T   │ (optional)            │
    └───────────────────────┴───────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_data_collection.sh -> .venv_data_collection (for camera viewer)
        bash install_scripts/install_mujoco_sim.sh -> .venv_sim (for simulation)
    - gear_sonic_deploy built (see docs)
    - GR00T inference server running and publishing poses via ZMQ

Usage (from repo root — no venv activation needed):
    # RECOMMENDED: Use the bridge for REQ/REP communication with GR00T
    python gear_sonic/scripts/launch_deployment.py \\
        --use-bridge \\
        --groot-host localhost \\
        --groot-port 5555

    # Legacy: Direct ZMQ SUB connection (requires GR00T to publish)
    python gear_sonic/scripts/launch_deployment.py \\
        --groot-host 192.168.1.100 --groot-port 5556 --groot-topic pose

    # Enable ZMQ CONFLATE for lower latency (always use latest message)
    python gear_sonic/scripts/launch_deployment.py \\
        --use-bridge --groot-conflate

    # Run in simulation mode
    python gear_sonic/scripts/launch_deployment.py --sim --use-bridge

    # Skip camera viewer
    python gear_sonic/scripts/launch_deployment.py --no-camera-viewer

    # Use custom checkpoint
    python gear_sonic/scripts/launch_deployment.py \\
        --checkpoint policy/checkpoints/my_model/model_step_100000
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import signal
import socket
import subprocess
import sys
import time


def _bootstrap_venv():
    """Re-exec with the .venv_data_collection Python if tyro is not available."""
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_data_collection" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_data_collection not found.\n"
            "  Run: bash install_scripts/install_data_collection.sh"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

import tyro


def _get_local_ip() -> str:
    """Best-effort detection of the PC's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


@dataclass
class DeploymentLaunchConfig:
    """CLI config for the GR00T deployment tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim (deploy.sh sim) instead of real robot."""

    use_bridge: bool = False
    """Use the GR00T bridge (REQ/REP) instead of direct ZMQ SUB connection.
    RECOMMENDED for connecting to GR00T PolicyServer."""

    # GR00T ZMQ connection (for direct SUB mode - legacy)
    groot_host: str = "localhost"
    """Host where the GR00T inference server is running (publishes poses via ZMQ)."""

    groot_port: int = 5555
    """ZMQ port for direct SUB connection (legacy). Use 5555 with --use-bridge."""

    groot_topic: str = "pose"
    """ZMQ topic to subscribe to for pose data from GR00T (direct SUB mode only)."""

    groot_conflate: bool = False
    """Enable ZMQ CONFLATE (always use latest message, discard old ones in queue)."""

    groot_api_token: str | None = None
    """API token for GR00T server authentication (when using --use-bridge)."""

    # Bridge options (when use_bridge=True)
    bridge_frequency: float = 50.0
    """Bridge control loop frequency in Hz (when using --use-bridge)."""

    deployment_state_port: int = 5557
    """Port where deployment publishes robot state (bridge subscribes here)."""

    deployment_state_topic: str = "g1_debug"
    """Topic that deployment publishes robot state on (bridge subscribes to this topic)."""

    action_publish_port: int = 5556
    """Port where bridge publishes actions for deployment to consume."""

    # C++ deploy options
    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh (e.g., 'policy/checkpoints/my_model/model_step_100000').
    Leave empty to use the deploy.sh default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh (e.g., 'ros2', 'all'). Leave empty for default."""

    # Camera options
    camera_viewer: bool = True
    """Start the camera viewer pane."""

    camera_host: str = "localhost"
    """Camera server host for the viewer."""

    camera_port: int = 5558
    """Camera server port for the viewer."""


SESSION_NAME = "sonic_deployment"


def _check_prerequisites(sim: bool = False):
    """Verify that required tools and deploy directory exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    repo_root = Path(__file__).resolve().parent.parent.parent

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )

    if sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first "
            "(run: bash install_scripts/install_mujoco_sim.sh)."
        )

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    """Kill any existing tmux session with our name."""
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session(use_bridge: bool = False, use_sim: bool = False):
    """Create a 2, 3, or 4-pane tmux layout."""
    # Create detached session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )

    # Enable mouse support (click panes, scroll, resize)
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )

    # Bind Ctrl+\ to kill the entire session (no prefix needed)
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )

    # Rename default window
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "deployment"],
    )

    if use_sim and use_bridge:
        # 4-pane layout for sim + bridge:
        # ┌─────┬─────┐
        # │  0  │  1  │  (0: Deploy, 1: Bridge)
        # ├─────┼─────┤
        # │  2  │  3  │  (2: Sim, 3: Camera)
        # └─────┴─────┘
        
        # First horizontal split: top and bottom
        subprocess.run(
            ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-v"],
        )
        # Split top pane vertically: 0 (left) and 1 (right)
        subprocess.run(
            ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-h"],
        )
        # Split bottom pane vertically: 2 (left) and 3 (right)
        subprocess.run(
            ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.2", "-h"],
        )
    elif use_bridge:
        # 3-pane layout: 0 | 1 | 2 (Deploy | Bridge | Camera)
        subprocess.run(
            ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
        )
        subprocess.run(
            ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.1", "-h"],
        )
    else:
        # 2-pane layout: pane 0 (left) and pane 1 (right)
        subprocess.run(
            ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
        )

    # Let all pane shells finish initialization (.bashrc, conda, etc.)
    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    """Send a command string to a tmux pane."""
    target = f"{SESSION_NAME}:0.{pane_index}"

    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _check_pane_alive(pane_index: int) -> bool:
    """Check if a tmux pane's process is still running."""
    target = f"{SESSION_NAME}:0.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def main(config: DeploymentLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    _check_prerequisites(sim=config.sim)
    _kill_existing_session()

    print("=" * 60)
    print("  SONIC Deployment Launcher (GR00T Inference)")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Bridge mode:     {' Yes (REQ/REP)' if config.use_bridge else 'No (direct SUB)'}")
    print(f"  GR00T Server:    {config.groot_host}:{config.groot_port}")
    if not config.use_bridge:
        print(f"  ZMQ Topic:       {config.groot_topic}")
        print(f"  ZMQ Conflate:    {'Yes' if config.groot_conflate else 'No'}")
    if config.deploy_checkpoint:
        print(f"  Checkpoint:      {config.deploy_checkpoint}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  Camera viewer:   {'Yes' if config.camera_viewer else 'No'}")
    print(f"  PC IP:           {_get_local_ip()}")
    print("=" * 60)

    _create_tmux_session(use_bridge=config.use_bridge, use_sim=config.sim)
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Sim mode in pane 2 (when using 4-pane layout) ---
    if config.sim and config.use_bridge:
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        print("Starting MuJoCo simulator (pane 2)...")
        _send_to_pane(2, sim_cmd, wait=3.0)
    elif config.sim:
        # Old behavior: separate window for sim when not using bridge
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"],
        )
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        # Switch back to the deployment window
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:deployment"],
        )

    # --- Pane 0 (left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    
    if config.use_bridge:
        # When using bridge, deploy listens for actions on action_publish_port
        # and publishes state on deployment_state_port
        deploy_cmd = (
            f"cd {repo_root / 'gear_sonic_deploy'} && "
            f"export HAS_ROS2=0 && "
            f"./deploy.sh "
            f"--input-type zmq "
            f"--zmq-host localhost "
            f"--zmq-port {config.action_publish_port} "
            f"--zmq-topic {config.groot_topic} "
            f"--output-type zmq "
            f"--zmq-out-port {config.deployment_state_port} "
            f"--zmq-out-topic {config.deployment_state_topic} "
        )
        if config.groot_conflate:
            deploy_cmd += "--zmq-conflate "
    else:
        # Direct connection to GR00T (legacy SUB mode)
        deploy_cmd = (
            f"cd {repo_root / 'gear_sonic_deploy'} && "
            f"export HAS_ROS2=0 && "
            f"./deploy.sh "
            f"--input-type zmq "
            f"--zmq-host {config.groot_host} "
            f"--zmq-port {config.groot_port} "
            f"--zmq-topic {config.groot_topic} "
        )
        if config.groot_conflate:
            deploy_cmd += "--zmq-conflate "
    
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {config.deploy_checkpoint} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {config.deploy_obs_config} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {config.deploy_planner} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {config.deploy_motion_data} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {config.deploy_output_type} "
    
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 1: GR00T Bridge (if using bridge) OR Camera Viewer ---
    if config.use_bridge:
        bridge_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/groot_deployment_bridge.py "
            f"--groot-host {config.groot_host} "
            f"--groot-port {config.groot_port}"
        )
        print("Starting GR00T bridge (pane 1)...")
        _send_to_pane(1, bridge_cmd, wait=2.0)

        # --- Pane 2: Sim (if --sim) or empty ---
        if config.sim:
            sim_cmd = (
                f"cd {repo_root} && "
                f"source .venv_sim/bin/activate && "
                f"python gear_sonic/scripts/run_sim_loop.py"
            )
            print("Starting MuJoCo sim (pane 2)...")
            _send_to_pane(2, sim_cmd, wait=2.0)

        # --- Pane 3 (4-pane) or Pane 2 (3-pane): Camera Viewer ---
        if config.sim:
            # 4-pane layout: camera goes to pane 3
            camera_pane = 3
        else:
            # 3-pane layout: camera goes to pane 2
            camera_pane = 2
        
        if config.camera_viewer:
            viewer_cmd = (
                f"cd {repo_root} && "
                f"source .venv_data_collection/bin/activate && "
                f"python gear_sonic/scripts/run_camera_viewer.py "
                f"--camera-host {config.camera_host} "
                f"--camera-port {config.camera_port}"
            )
            print(f"Starting camera viewer (pane {camera_pane})...")
            _send_to_pane(camera_pane, viewer_cmd, wait=2.0)
        else:
            print(f"Camera viewer disabled. Pane {camera_pane} available for manual use.")
    else:
        # NOT using bridge: just camera viewer in pane 1 (2-pane layout)
        if config.camera_viewer:
            viewer_cmd = (
                f"cd {repo_root} && "
                f"source .venv_data_collection/bin/activate && "
                f"python gear_sonic/scripts/run_camera_viewer.py "
                f"--camera-host {config.camera_host} "
                f"--camera-port {config.camera_port}"
            )
            print("Starting camera viewer (pane 1)...")
            _send_to_pane(1, viewer_cmd, wait=2.0)
        else:
            print("Camera viewer disabled. Pane 1 available for manual use.")

    # Select the deploy pane so the user can monitor it
    subprocess.run(
        ["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.0"],
    )

    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print()
    print("  Window 'deployment':")
    
    if config.sim and config.use_bridge:
        # 4-pane layout
        print("    ┌──────────────────┬──────────────────┐")
        print("    │ Pane 0 (Deploy)  │ Pane 1 (Bridge)  │")
        print("    ├──────────────────┼──────────────────┤")
        print("    │ Pane 2 (Sim)     │ Pane 3 (Camera)  │")
        print("    └──────────────────┴──────────────────┘")
        print()
        print("  ** deploy.sh (pane 0) is waiting for confirmation —")
        print("     press Enter in pane 0 to proceed **")
        print()
        print("  NOTE: Make sure your GR00T inference server is running:")
        print(f"        uv run python gr00t/eval/run_gr00t_server.py \\")
        print(f"            --model-path <your-model> \\")
        print(f"            --embodiment-tag <tag> \\")
        print(f"            --port {config.groot_port}")
        print()
        print("  The bridge will:")
        print(f"    1. Subscribe to deployment state (port {config.deployment_state_port})")
        print(f"    2. Send observations to GR00T (REQ/REP port {config.groot_port})")
        print(f"    3. Publish actions back to deployment (port {config.action_publish_port})")
    elif config.use_bridge:
        # 3-pane layout (bridge but no sim)
        print("    ┌──────────────────┬──────────────────┬──────────────────┐")
        print("    │ Pane 0 (Deploy)  │ Pane 1 (Bridge)  │ Pane 2 (Camera)  │")
        print("    └──────────────────┴──────────────────┴──────────────────┘")
        print()
        print("  ** deploy.sh (pane 0) is waiting for confirmation —")
        print("     press Enter in pane 0 to proceed **")
        print()
        print("  NOTE: Make sure your GR00T inference server is running:")
        print(f"        uv run python gr00t/eval/run_gr00t_server.py \\")
        print(f"            --model-path <your-model> \\")
        print(f"            --embodiment-tag <tag> \\")
        print(f"            --port {config.groot_port}")
        print()
        print("  The bridge will:")
        print(f"    1. Subscribe to deployment state (port {config.deployment_state_port})")
        print(f"    2. Send observations to GR00T (REQ/REP port {config.groot_port})")
        print(f"    3. Publish actions back to deployment (port {config.action_publish_port})")
    else:
        # 2-pane layout (no bridge)
        print("    ┌───────────────────────┬───────────────────────┐")
        print("    │ Pane 0 (Deploy)       │ Pane 1 (Camera)       │")
        print("    └───────────────────────┴───────────────────────┘")
        print()
        print("  ** deploy.sh (pane 0) is waiting for confirmation —")
        print("     press Enter in pane 0 to proceed **")
        print()
        print("  NOTE: Make sure your GR00T inference server is PUBLISHING")
        print(f"        (PUB/SUB mode) at {config.groot_host}:{config.groot_port}")
        print(f"        on topic '{config.groot_topic}'")
        print()
        print("  WARNING: Direct SUB mode may not work with GR00T PolicyServer.")
        print("           Consider using --use-bridge for REQ/REP communication.")

    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    # Attach to the session
    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    # After detach/exit, offer cleanup
    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(sig, frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(DeploymentLaunchConfig)
    main(config)
