"""Online Planning with ROS Control

Run online planning in collision aware environments with ROS2 trajectory execution.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import numpy as np
import pyroki as pk
from pyroki.collision import HalfSpace, RobotCollision, Sphere
import pyroki_snippets as pks
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclDuration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robot_descriptions.loaders.yourdfpy import yourdfpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from wutility import sample_even_fit_mesh
from toppra_quintic_optimal import generate_optimal_trajectory

# Enable JAX persistent compilation cache
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Online planning with ROS control configuration."""
    urdf_path: str = str(Path(__file__).parent / "../ur5e/ur5e.urdf.sphere")
    sphere_json_path: str = str(Path(__file__).parent / "../ur5e/ur5e.urdf_spherized.json")
    target_link_name: str = "tool0"
    elbow_link_name: str = "forearm_link"
    default_cfg: tuple = (-0.523, -1.61, 1.544, -1.5, -1.57, -0.5)
    viser_port: int = 8080
    len_traj: int = 10
    dt: float = 0.3
    box_stl_path: str = str(Path(__file__).parent / "storage_box.stl")
    box_n_spheres: int = 500
    box_sphere_radius: float = 0.005


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class JointTrajectoryPublisher(Node):
    """ROS2 node for sending joint trajectory commands."""

    JOINT_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]

    def __init__(self):
        super().__init__("publisher_position_trajectory_controller")
        action_name = "/joint_trajectory_controller/follow_joint_trajectory"
        self.action_client = ActionClient(self, FollowJointTrajectory, action_name)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            qos,
        )

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected :(")
            return
        self.get_logger().info("Goal accepted :)")
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        error_code = future.result().result.error_code
        error_string = future.result().result.error_string
        if error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("Goal succeeded!")
        else:
            self.get_logger().info(f"Goal failed: {error_string}")

    def send_topic(self, joint_positions: list, duration_sec: int = 4) -> None:
        """Send a single joint position command via topic."""
        traj = JointTrajectory()
        traj.joint_names = list(self.JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(joint_positions)
        point.time_from_start = Duration(sec=duration_sec)
        traj.points.append(point)
        traj.header.frame_id = "base"
        self.publisher.publish(traj)

    def send_joint_trajectory(self, jnt_traj) -> None:
        """Send a time-parameterized trajectory via topic."""
        traj = JointTrajectory()
        traj.joint_names = list(self.JOINT_NAMES)

        duration = jnt_traj.duration
        ts_sample = np.linspace(0, duration, int(duration / 0.01))  # 100 Hz
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)

        for t, q, dq, ddq in zip(ts_sample, qs_sample, qds_sample, qdds_sample):
            point = JointTrajectoryPoint()
            point.positions = list(q)
            point.velocities = list(dq)
            point.accelerations = list(ddq)
            point.time_from_start = rclDuration(seconds=t).to_msg()
            traj.points.append(point)

        self.publisher.publish(traj)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class OnlinePlanningApp:
    """Online planning application with Viser visualization and ROS2 control."""

    def __init__(self, cfg: Config, node: JointTrajectoryPublisher):
        self.cfg = cfg
        self.node = node

        # --- Robot setup ---
        self.urdf = yourdfpy.URDF.load(cfg.urdf_path)
        with open(cfg.sphere_json_path) as f:
            sphere_decomposition = json.load(f)
        self.robot_coll = RobotCollision.from_sphere_decomposition(
            sphere_decomposition=sphere_decomposition,
            urdf=self.urdf,
        )
        self.default_cfg = np.array(cfg.default_cfg)
        self.robot = pk.Robot.from_urdf(self.urdf, default_joint_cfg=self.default_cfg)

        # --- Collision objects ---
        self.plane_coll = HalfSpace.from_point_and_normal(
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        self.box_mesh = trimesh.load_mesh(cfg.box_stl_path)
        self.box_mesh.apply_scale(0.005)
        pts, radius = sample_even_fit_mesh(
            self.box_mesh, n_spheres=cfg.box_n_spheres, sphere_radius=cfg.box_sphere_radius
        )
        self.box_spheres = pk.collision.Sphere.from_center_and_radius(
            center=pts, radius=radius
        )

        # --- Planning state ---
        self.sol_traj = np.array(
            self.robot.joint_var_cls.default_factory()[None].repeat(cfg.len_traj, axis=0)
        )
        self.current_mc_cfg = self.default_cfg.copy()
        self.is_executing = False       # ボタンを押すまで待機
        self.traj_index = 0
        self._pending_traj = None       # 確認待ち軌道
        self._pending_sol_traj = None   # 確認待ちsol_traj
        self._waiting_confirmation = False  # 確認待ちフラグ
        self._traj_confirmed = False    # 送信確定フラグ
        self._confirm_folder: Optional[viser.GuiFolderHandle] = None  # 確認パネル

        # --- Viser setup ---
        self.server = viser.ViserServer(port=cfg.viser_port)
        self.server.gui.configure_theme(dark_mode=True)
        self._setup_scene()
        self._setup_gui()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        """3Dシーンのセットアップ."""
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        self.urdf_vis = ViserUrdf(self.server, self.urdf, root_node_name="/robot")
        self.urdf_vis_mc = ViserUrdf(
            self.server, self.urdf, root_node_name="/robot_mc",
            mesh_color_override=(0.3, 0.3, 0.8, 0.5),
        )
        self.urdf_vis_mc.update_cfg(self.default_cfg)

        self.ik_target_handle = self.server.scene.add_transform_controls(
            "/ik_target", scale=0.2,
            position=(0.3398, -0.12158905, 0.5132406),
            wxyz=(0, 0.707, -0.707, 0),
        )
        self.ik_target_handle.on_update(self._on_ik_target_update)

        self.target_frame_handle = self.server.scene.add_batched_axes(
            "target_frame",
            axes_length=0.05,
            axes_radius=0.005,
            batched_positions=np.zeros((self.cfg.len_traj, 3)),
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * self.cfg.len_traj),
        )

        self.box_handle = self.server.scene.add_transform_controls(
            "/box", scale=0.2,
            wxyz=(0.707, 0.707, 0, 0),
            position=(5.07658497e-01, -7.54795097e-01, 1.37389611e-04),
        )
        self.server.scene.add_mesh_trimesh("/box/visual", mesh=self.box_mesh)
        self.server.scene.add_mesh_trimesh(
            "/box/coll", mesh=self.box_spheres.to_trimesh()
        )

    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------

    def _setup_gui(self) -> None:
        """GUIのセットアップ."""
        with self.server.gui.add_folder("Joint position", expand_by_default=False):
            self.slider_handles, _ = self._create_robot_control_sliders()

        with self.server.gui.add_folder("IK Weights"):
            self.rest_weight = self.server.gui.add_slider(
                "Rest Weight", 0.0, 30.0, 1.0, 0.0
            )
            self.pose_weight = self.server.gui.add_slider(
                "Pose Weight", 5.0, 100.0, 1.0, 19.0
            )
            self.collision_weight = self.server.gui.add_slider(
                "Collision Weight", 0.0, 30.0, 1.0, 15.0
            )
            self.elbow_height_weight = self.server.gui.add_slider(
                "Elbow Height Weight", 0.0, 10.0, 0.001, 0.25
            )
            self.realtime_ik = self.server.gui.add_checkbox(
                "Enable realtime IK", initial_value=True
            )

        with self.server.gui.add_folder("Cost Weights"):
            with self.server.gui.add_folder("Pose Costs"):
                self.w_pose_rot = self.server.gui.add_slider(
                    "Pose Match Rotation", 0.0, 200.0, 1.0, 100.0
                )
                self.w_pose_trans = self.server.gui.add_slider(
                    "Pose Match Translation", 0.0, 400.0, 1.0, 100.0
                )
                self.w_pose_smooth = self.server.gui.add_slider(
                    "Pose Smoothness", 0.0, 100.0, 0.1, 10.0
                )
                self.w_match_start = self.server.gui.add_slider(
                    "Match Start Pose", 0.0, 200.0, 1.0, 100.0
                )
                self.w_match_joint = self.server.gui.add_slider(
                    "Match Joint to Pose", 0.0, 200.0, 1.0, 100.0
                )

            with self.server.gui.add_folder("Joint Costs"):
                self.w_smooth = self.server.gui.add_slider(
                    "Smoothness", 0.0, 50.0, 1.0, 1.0
                )
                self.w_limit_vel = self.server.gui.add_slider(
                    "Limit Velocity", 0.0, 10.0, 0.1, 0.0
                )
                self.w_limit = self.server.gui.add_slider(
                    "Joint Limit", 0.0, 200.0, 1.0, 100.0
                )
                self.w_rest = self.server.gui.add_slider(
                    "Rest Pose", 0.0, 1.0, 0.01, 0.0
                )
                self.w_manip = self.server.gui.add_slider(
                    "Manipulability", 0.0, 1.0, 0.01, 0.0
                )

            with self.server.gui.add_folder("Collision Costs"):
                self.w_self_coll = self.server.gui.add_slider(
                    "Self Collision", 0.0, 50.0, 1.0, 0.0
                )
                self.w_world_coll = self.server.gui.add_slider(
                    "World Collision", 0.0, 100.0, 1.0, 21.0
                )

        self.speed_percentage = self.server.gui.add_slider(
            "Speed percentage", 0.1, 1.0, 0.1, 0.5
        )
        self.time_step = self.server.gui.add_slider(
            "Timestep", min=0, max=self.cfg.len_traj - 1, step=1, initial_value=0
        )
        self.time_step.on_update(
            lambda _: self.urdf_vis_mc.update_cfg(
                np.array(self.sol_traj[self.time_step.value])
            )
        )
        self.plan_button = self.server.gui.add_button(label="Planning")
        self.plan_button.on_click(self._on_plan_click)

        self._update_robot_visualization(self.sol_traj[0])

    def _create_robot_control_sliders(self) -> tuple:
        """ロボット制御スライダーを作成."""
        slider_handles = []
        initial_config = []
        for joint_name, (lower, upper) in self.urdf_vis.get_actuated_joint_limits().items():
            lower = lower if lower is not None else -np.pi
            upper = upper if upper is not None else np.pi
            initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
            slider = self.server.gui.add_slider(
                label=joint_name, min=lower, max=upper, step=1e-3,
                initial_value=initial_pos,
            )
            slider_handles.append(slider)
            initial_config.append(initial_pos)
        return slider_handles, initial_config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_robot_visualization(self, config: np.ndarray) -> None:
        """ロボットURDF・スライダー・衝突メッシュを更新."""
        self.urdf_vis.update_cfg(config)
        for slider, value in zip(self.slider_handles, config):
            slider.value = float(value)
        robot_coll_mesh = self.robot_coll.at_config(self.robot, config).to_trimesh()
        self.server.scene.add_mesh_trimesh(
            "/robot_coll", mesh=robot_coll_mesh, visible=True
        )

    def _get_world_coll(self) -> list:
        """現在のボックス位置から障害物リストを取得."""
        box_coll = self.box_spheres.transform_from_wxyz_position(
            wxyz=np.array(self.box_handle.wxyz),
            position=np.array(self.box_handle.position),
        )
        return [self.plane_coll, box_coll]

    def _get_rest_weights(self) -> np.ndarray:
        """rest weightの配列を生成."""
        weights = np.array(
            [self.rest_weight.value] * self.robot.joints.num_actuated_joints
        )
        weights[0] = 0  # allow joint rotation freely
        return weights

    def _run_ik(self, world_coll_list: list) -> Optional[np.ndarray]:
        """IKを解いて関節角度を返す。失敗時はNoneを返す。"""
        weights = np.ones(len(world_coll_list)) * self.collision_weight.value
        try:
            return pks.solve_ik_with_collision_custom(
                robot=self.robot,
                coll=self.robot_coll,
                world_coll_list=world_coll_list,
                target_link_name=self.cfg.target_link_name,
                target_position=np.array(self.ik_target_handle.position),
                target_wxyz=np.array(self.ik_target_handle.wxyz),
                weights=weights,
                pose_weight=self.pose_weight.value,
                rest_weight=self._get_rest_weights(),
                initial_joint_angles=self.current_mc_cfg,
                elbow_height_weight=self.elbow_height_weight.value,
                elbow_link_name=self.cfg.elbow_link_name,
            )
        except (ValueError, RuntimeError) as e:
            print(f"IK solver failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_ik_target_update(self, _: viser.TransformControlsHandle) -> None:
        """IKターゲット更新コールバック."""
        if not self.realtime_ik.value:
            return
        ik_sol = self._run_ik(self._get_world_coll())
        if ik_sol is not None:
            self.urdf_vis_mc.update_cfg(ik_sol)
            self.current_mc_cfg = ik_sol.copy()

    def _on_plan_click(self, _) -> None:
        """Planning ボタンコールバック."""
        if self._waiting_confirmation:
            return  # モーダル確認中は無視
        if self.is_executing:
            # 実行中なら停止
            self.is_executing = False
            self.traj_index = 0
            self.plan_button.label = "Start Planning"
        else:
            # 停止中なら計画開始
            self.is_executing = True
            self.traj_index = 0
            self.plan_button.label = "Planning..."

    # ------------------------------------------------------------------
    # Planning and execution
    # ------------------------------------------------------------------

    def _plan_trajectory(self, start_cfg: np.ndarray) -> np.ndarray:
        """軌道を計画してsol_trajを返す。"""
        world_coll_list = self._get_world_coll()
        print(f"Box position: {self.box_handle.position}, wxyz: {self.box_handle.wxyz}")

        prev_sols_interp = np.linspace(
            start_cfg, self.current_mc_cfg, self.cfg.len_traj + 1
        )[1:]

        print("Planning trajectory...")
        sol_traj, sol_pos, sol_wxyz = pks.wu_solve_online_planning(
            robot=self.robot,
            robot_coll=self.robot_coll,
            world_coll=world_coll_list,
            target_link_name=self.cfg.target_link_name,
            target_position=np.array(self.ik_target_handle.position),
            target_wxyz=np.array(self.ik_target_handle.wxyz),
            timesteps=self.cfg.len_traj,
            dt=self.cfg.dt,
            start_cfg=start_cfg,
            prev_sols=prev_sols_interp,
            weight_pose_match_rotation=self.w_pose_rot.value,
            weight_pose_match_translation=self.w_pose_trans.value,
            weight_pose_smoothness=self.w_pose_smooth.value,
            weight_match_start_pose=self.w_match_start.value,
            weight_match_joint_to_pose=self.w_match_joint.value,
            weight_smoothness=self.w_smooth.value,
            weight_limit_velocity=self.w_limit_vel.value,
            weight_limit=self.w_limit.value,
            weight_rest=self.w_rest.value,
            weight_manipulability=self.w_manip.value,
            weight_self_collision=self.w_self_coll.value,
            weight_world_collision=self.w_world_coll.value,
        )

        if hasattr(self.target_frame_handle, "batched_positions"):
            self.target_frame_handle.batched_positions = np.array(sol_pos)
            self.target_frame_handle.batched_wxyzs = np.array(sol_wxyz)
        else:
            self.target_frame_handle.positions_batched = np.array(sol_pos)
            self.target_frame_handle.wxyzs_batched = np.array(sol_wxyz)

        return sol_traj

    def _send_trajectory_to_robot(
        self, sol_traj: np.ndarray, start_cfg: np.ndarray
    ) -> None:
        """TOPPRAで時間最適化し、確認モーダルを表示する."""
        dof = len(start_cfg)
        spd = self.speed_percentage.value
        vel_limits = np.ones(dof) * 3.14 * spd
        acc_limits = np.ones(dof) * 14.0 * spd
        self._pending_traj = generate_optimal_trajectory(
            np.vstack([start_cfg, sol_traj]), vel_limits, acc_limits
        )
        self._pending_sol_traj = sol_traj
        self._waiting_confirmation = True
        self._traj_confirmed = False
        self._show_send_confirmation_panel()

    def _close_confirm_panel(self) -> None:
        """確認パネルを閉じる."""
        if self._confirm_folder is not None:
            self._confirm_folder.remove()
            self._confirm_folder = None

    def _show_send_confirmation_panel(self) -> None:
        """軌道送信前の確認パネルをGUIサイドバーに表示（ビューポート操作を妨げない）."""
        traj = self._pending_traj
        sol_traj = self._pending_sol_traj

        # 軌道の概要情報を作成
        duration = traj.duration
        n_steps = len(sol_traj)
        target_pos = np.array(self.ik_target_handle.position)
        target_link_idx = self.robot.links.names.index(self.cfg.target_link_name)
        end_fk = self.robot.forward_kinematics(sol_traj[-1])
        end_pos = np.array(end_fk[target_link_idx][4:])
        final_dist = float(np.linalg.norm(end_pos - target_pos))

        summary = (
            f"**軌道をロボットに送信しますか？**\n\n"
            f"- Duration: **{duration:.2f} s**\n"
            f"- Steps: **{n_steps}**\n"
            f"- Speed: **{self.speed_percentage.value * 100:.0f}%**\n"
            f"- Target distance: **{final_dist:.4f} m**"
        )

        # 既存の確認パネルがあれば削除
        self._close_confirm_panel()

        # GUIパネルに確認フォルダーを追加（order=-1で最上部に表示）
        self._confirm_folder = self.server.gui.add_folder(
            "⚠️ 送信確認", expand_by_default=True, order=-1.0
        )
        with self._confirm_folder:
            self.server.gui.add_markdown(summary)

            def _on_send(_):
                self._close_confirm_panel()
                print("Sending trajectory to robot...")
                self.node.send_joint_trajectory(traj)
                print("Trajectory sent!")
                self._traj_confirmed = True
                self._waiting_confirmation = False
                self.traj_index = 1  # 0のままだと再計画ループするので1に

            def _on_cancel(_):
                self._close_confirm_panel()
                self._traj_confirmed = False
                self._waiting_confirmation = False
                self.is_executing = False
                self.plan_button.label = "Start Planning"
                self.traj_index = 0
                print("Trajectory send cancelled.")

            self.server.gui.add_button("✅ 送信する", color="green").on_click(_on_send)
            self.server.gui.add_button("❌ キャンセル", color="red").on_click(_on_cancel)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """メインループ."""
        # Warm-up IK
        ik_sol = self._run_ik(self._get_world_coll())
        if ik_sol is not None:
            self.urdf_vis_mc.update_cfg(ik_sol)
            self.current_mc_cfg = ik_sol.copy()

        target_link_idx = self.robot.links.names.index(self.cfg.target_link_name)

        try:
            while True:
                # モーダル確認待ち中はメインループをスキップ
                if self._waiting_confirmation:
                    time.sleep(0.05)
                    continue

                if self.is_executing:
                    if self.traj_index == 0:
                        # 軌道を計画してモーダル表示（フラグを立てて次ループへ）
                        start_cfg = np.array(
                            [slider.value for slider in self.slider_handles]
                        )
                        self.sol_traj = self._plan_trajectory(start_cfg)
                        self._send_trajectory_to_robot(self.sol_traj, start_cfg)
                        # モーダルが表示されたので確認待ちに入る（次ループでskip）
                        time.sleep(0.05)
                        continue

                    if self.traj_index < len(self.sol_traj):
                        current_fk = self.robot.forward_kinematics(
                            self.sol_traj[self.traj_index]
                        )
                        current_pos = current_fk[target_link_idx][4:]
                        distance = np.linalg.norm(
                            current_pos - np.array(self.ik_target_handle.position)
                        )
                        print(
                            f"Step {self.traj_index}/{len(self.sol_traj)}: "
                            f"distance={distance:.6f}"
                        )
                        self._update_robot_visualization(
                            self.sol_traj[self.traj_index]
                        )
                        self.traj_index += 1
                    else:
                        self.is_executing = False
                        self.plan_button.label = "Start Planning"
                        print("Trajectory execution completed!")
                        self.traj_index = 0

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nShutting down...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: Config = Config()) -> None:
    """Main function for online planning with ROS control."""
    rclpy.init()
    node = JointTrajectoryPublisher()
    node.send_topic(list(cfg.default_cfg))

    app = OnlinePlanningApp(cfg, node)
    app.run()


if __name__ == "__main__":
    tyro.cli(main)
