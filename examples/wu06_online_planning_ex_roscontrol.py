"""Online Planning

Run online planning in collision aware environments.
"""


import json
import os
from pathlib import Path
import time

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import numpy as np
import pyroki as pk
from pyroki.collision import HalfSpace
from pyroki.collision import RobotCollision
from pyroki.collision import Sphere
import pyroki_snippets as pks
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclDuration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robot_descriptions.loaders.yourdfpy import yourdfpy
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import trimesh
import viser
from viser.extras import ViserUrdf
from wutility import sample_even_fit_mesh
from wutility import voxel_fit_volume_sample_surface_mesh
from toppra_quintic_optimal import generate_optimal_trajectory

# Enable JAX persistent compilation cache
# Use a permanent directory (not /tmp which is cleared on reboot)
os.environ["JAX_COMPILATION_CACHE_DIR"] = str(Path.home() / ".cache" / "jax")
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import jax
# JAX 0.7+ uses different config names
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        # slider.on_update(  # When sliders move, we update the URDF configuration.
        #     lambda _: viser_urdf.update_cfg(
        #         np.array([slider.value for slider in slider_handles])
        #     )
        # )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def update_robot_visualization(
    urdf_vis: ViserUrdf,
    slider_handles: list[viser.GuiInputHandle[float]],
    robot: pk.Robot,
    robot_coll: RobotCollision,
    server: viser.ViserServer,
    config: np.ndarray,
) -> None:
    """Update robot URDF visualization, sliders, and collision mesh."""
    urdf_vis.update_cfg(config)
    for slider, value in zip(slider_handles, config):
        slider.value = float(value)
    robot_coll_mesh = robot_coll.at_config(robot, config).to_trimesh()
    server.scene.add_mesh_trimesh("/robot_coll", mesh=robot_coll_mesh, visible=True)


class PublisherJointTrajectory(Node):
    def __init__(self):
        super().__init__("publisher_position_trajectory_controller")
        action_name = '/joint_trajectory_controller/follow_joint_trajectory'
        self.action_client = ActionClient(self, FollowJointTrajectory, action_name)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(JointTrajectory,
                                               "/joint_trajectory_controller/joint_trajectory",
                                               qos)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        error_string = future.result().result.error_string
        error_code = future.result().result.error_code
        if error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Goal succeeded!')
        else:
            self.get_logger().info(f'Goal failed with error string: {error_string}')

    def send_topic(self, joint_list):
        traj = JointTrajectory()
        # traj.joint_names = self.joints
        traj.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        # traj.points.append(self.goals[0])
        point = JointTrajectoryPoint()
        point.positions = joint_list
        point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point.time_from_start = Duration(sec=4)
        traj.points.append(point)
        # traj.header.stamp = rclpy.ti
        traj.header.frame_id = 'base'
        self.publisher.publish(traj)

    def send_joint_trajectory(self, jnt_traj
                              # ts_sample, qs_sample, qds_sample, qdds_sample,
                              # current_cfg
                              ):
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # Sample the time-parameterized trajectory
        duration = jnt_traj.duration
        ts_sample = np.linspace(0, duration, int(duration / 0.01))  # 100 Hz sampling
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)  # First derivative
        qdds_sample = jnt_traj(ts_sample, 2)  # Second derivative

        # Add trajectory points with offset time
        for t, q, dq, ddq in zip(ts_sample, qs_sample, qds_sample, qdds_sample):
            point = JointTrajectoryPoint()
            point.positions = list(q)
            point.velocities = list(dq)
            point.accelerations = list(ddq)
            point.time_from_start = rclDuration(seconds=t).to_msg()
            traj.points.append(point)

        # traj.header.frame_id = 'base'
        self.publisher.publish(traj)


def main():
    """Main function for online planning with collision."""
    rclpy.init()
    node = PublisherJointTrajectory()
    node.send_topic([-0.523, -1.61, 1.544, -1.5, -1.57, -0.5])
    urdf_path = str(Path(__file__).parent / '../ur5e/ur5e.urdf.sphere')
    target_link_name = 'tool0'
    urdf = yourdfpy.URDF.load(urdf_path)
    sphere_json_path = Path(__file__).parent / "../ur5e/ur5e.urdf_spherized.json"
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )
    # For UR5 it's important to initialize the robot in a safe configuration;
    # the zero-configuration puts the robot aligned with the wall obstacle.
    # default_cfg = np.array([0, -1.57, 0, -1.57, 0, 0])
    default_cfg = np.array([-0.523, -1.61, 1.544, -1.5, -1.57, -0.5])
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)

    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )

    # Define the online planning parameters.
    len_traj, dt = 10, 0.3

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.gui.configure_theme(dark_mode=True)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis_mc = ViserUrdf(server, urdf, root_node_name="/robot_mc",
                            mesh_color_override=(0.3, 0.3, 0.8, 0.5))
    urdf_vis_mc.update_cfg(default_cfg)
    current_mc_cfg = default_cfg.copy()

    with server.gui.add_folder("Joint   position", expand_by_default=False):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, urdf_vis
        )

    # Create interactive controller for IK target.
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2,
        # position=(0.3, 0.0, 0.5), wxyz=(0, 0, 1, 0)
        position=(0.3398, -0.12158905, 0.5132406), wxyz=(0, 0.707, -0.707, 0)
    )

    target_frame_handle = server.scene.add_batched_axes(
        "target_frame",
        axes_length=0.05,
        axes_radius=0.005,
        batched_positions=np.zeros((25, 3)),
        batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * 25),
    )

    with server.gui.add_folder("ik weights"):
        rest_weight = server.gui.add_slider(
            # "Rest Weight", 0.0, 0.5, 0.01, 0.02
            # "Rest Weight", 0.0, 0.5, 0.01, 0.5
            "Rest Weight", 0.0, 30.0, 1.0, 0.0,
        )
        pose_weight = server.gui.add_slider(
            "Pose Weight", 5.0, 100.0, 1.0, 19.0
        )

        collision_weight = server.gui.add_slider(
            "Collision Weight", 0.0, 30.0, 1.0, 15.0
        )

        elbow_height_weight = server.gui.add_slider(
            "Elbow Height Weight", 0.0, 10.0, 0.001, 0.25
        )

        realtime_ik = server.gui.add_checkbox("Enable realtime ik", initial_value=True)

    # Create weight sliders for cost functions
    with server.gui.add_folder("Cost Weights"):
        with server.gui.add_folder("Pose Costs"):
            weight_pose_match_rotation = server.gui.add_slider(
                label="Pose Match Rotation",
                min=0.0,
                max=200.0,
                step=1.0,
                initial_value=100.0,
            )
            weight_pose_match_translation = server.gui.add_slider(
                label="Pose Match Translation",
                min=0.0,
                max=400.0,
                step=1.0,
                initial_value=100.0,
            )
            weight_pose_smoothness = server.gui.add_slider(
                label="Pose Smoothness",
                min=0.0,
                max=100.0,
                step=0.1,
                initial_value=10.0,
            )
            weight_match_start_pose = server.gui.add_slider(
                label="Match Start Pose",
                min=0.0,
                max=200.0,
                step=1.0,
                initial_value=100.0,
            )
            weight_match_joint_to_pose = server.gui.add_slider(
                label="Match Joint to Pose",
                min=0.0,
                max=200.0,
                step=1.0,
                initial_value=100.0,
            )

        with server.gui.add_folder("Joint Costs"):
            weight_smoothness = server.gui.add_slider(
                label="Smoothness",
                min=0.0,
                max=50.0,
                step=1.0,
                initial_value=1.0,
            )
            weight_limit_velocity = server.gui.add_slider(
                label="Limit Velocity",
                min=0.0,
                max=10.0,
                step=0.1,
                initial_value=0.0,
            )
            weight_limit = server.gui.add_slider(
                label="Joint Limit",
                min=0.0,
                max=200.0,
                step=1.0,
                initial_value=100.0,
            )
            weight_rest = server.gui.add_slider(
                label="Rest Pose",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.0,
            )
            weight_manipulability = server.gui.add_slider(
                label="Manipulability",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.0,
            )

        with server.gui.add_folder("Collision Costs"):
            weight_self_collision = server.gui.add_slider(
                label="Self Collision",
                min=0.0,
                max=50.0,
                step=1.0,
                initial_value=0.0,
            )
            weight_world_collision = server.gui.add_slider(
                label="World Collision",
                min=0.0,
                max=100.0,
                step=1.0,
                initial_value=21.0,
            )

    sol_traj = np.array(
        robot.joint_var_cls.default_factory()[None].repeat(len_traj, axis=0)
    )
    update_robot_visualization(
        urdf_vis, slider_handles, robot, robot_coll, server, sol_traj[0]
    )

    target_link_idx = robot.links.names.index(target_link_name)
    target_link_pose = robot.forward_kinematics(sol_traj)[target_link_idx]
    print(f'target_link_pose initial: {target_link_pose}')

    mesh = trimesh.load_mesh(str(Path(__file__).parent / 'storage_box.stl'))
    mesh.apply_scale(0.005)
    box_handle = server.scene.add_transform_controls(
        "/box", scale=0.2,
        wxyz=(0.707, 0.707, 0, 0),
        position=(5.07658497e-01, -7.54795097e-01,  1.37389611e-04),
    )
    server.scene.add_mesh_trimesh("/box/visual", mesh=mesh)
    # pts, radius = voxel_fit_volume_sample_surface_mesh(mesh, n_spheres=500, surface_sphere_radius=0.005)
    pts, radius = sample_even_fit_mesh(mesh, n_spheres=500, sphere_radius=0.005)
    # print(f'{type(pts)=}, {pts=}')
    box_spheres = pk.collision.Sphere.from_center_and_radius(
        center=pts, radius=radius)
    server.scene.add_mesh_trimesh(
        "/box/coll", mesh=box_spheres.to_trimesh(),
    )

    # Get current configuration from sliders
    current_cfg = np.array([slider.value for slider in slider_handles])
    sol_traj = np.array([current_cfg] * len_traj)
    qs_sample = sol_traj.copy()
    # sol_traj_init = np.array([current_cfg] * len_traj)

    is_executing = True
    traj_index = 0

    speed_percentage = server.gui.add_slider(
        label="Speed percentage",
        min=0.0,
        max=1.0,
        step=0.1,
        initial_value=0.5,
    )

    plan_button = server.gui.add_button(
        label="Planning",
    )

    @plan_button.on_click
    def _(_) -> None:
        nonlocal is_executing, traj_index
        if not is_executing:
            is_executing = True
            traj_index = 0
            plan_button.name = "Planning..."
        else:
            is_executing = False
            plan_button.name = "Start Planning"

    # Add callback when IK target is moved

    time_step = server.gui.add_slider(
        "Timestep", min=0, max=len_traj - 1, step=1, initial_value=0
    )
    time_step.on_update(  # When sliders move, we update the URDF configuration.
        lambda _:  urdf_vis_mc.update_cfg(
            np.array([value for value in sol_traj[time_step.value]])
        )
    )

    """Callback when IK target is updated. warm up"""
    box_coll_world_current = box_spheres.transform_from_wxyz_position(
        wxyz=np.array(box_handle.wxyz),
        position=np.array(box_handle.position),
    )
    world_coll_list = [plane_coll, box_coll_world_current]
    weights = np.ones(len(world_coll_list)) * collision_weight.value
    rest_weights = np.array([rest_weight.value] * robot.joints.num_actuated_joints)
    rest_weights[0] = 0  # allow joint rotation intensly
    if realtime_ik.value:
        # Realtime IK mode
        ik_sol = pks.solve_ik_with_collision_custom(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=np.array(ik_target_handle.position),
            target_wxyz=np.array(ik_target_handle.wxyz),
            weights=weights,
            pose_weight=pose_weight.value,
            rest_weight=rest_weights,
            initial_joint_angles=current_mc_cfg,
            elbow_height_weight=elbow_height_weight.value,
            elbow_link_name='forearm_link',
        )
        urdf_vis_mc.update_cfg(ik_sol)
        current_mc_cfg = ik_sol.copy()

    @ik_target_handle.on_update
    def _(_: viser.TransformControlsHandle) -> None:
        nonlocal current_mc_cfg
        """Callback when IK target is updated."""
        box_coll_world_current = box_spheres.transform_from_wxyz_position(
            wxyz=np.array(box_handle.wxyz),
            position=np.array(box_handle.position),
        )
        world_coll_list = [plane_coll, box_coll_world_current]
        weights = np.ones(len(world_coll_list)) * collision_weight.value
        rest_weights = np.array([rest_weight.value] * robot.joints.num_actuated_joints)
        rest_weights[0] = 0  # allow joint rotation intensly
        if realtime_ik.value:
            # Realtime IK mode
            try:
                ik_sol = pks.solve_ik_with_collision_custom(
                    robot=robot,
                    coll=robot_coll,
                    world_coll_list=world_coll_list,
                    target_link_name=target_link_name,
                    target_position=np.array(ik_target_handle.position),
                    target_wxyz=np.array(ik_target_handle.wxyz),
                    weights=weights,
                    pose_weight=pose_weight.value,
                    rest_weight=rest_weights,
                    initial_joint_angles=current_mc_cfg,
                    elbow_height_weight=elbow_height_weight.value,
                    elbow_link_name='forearm_link',
                )
                urdf_vis_mc.update_cfg(ik_sol)
                current_mc_cfg = ik_sol.copy()
            except (ValueError, RuntimeError) as e:
                print(f"IK solver failed: {e}")
                print("Keeping previous configuration")
                # Keep current_mc_cfg unchanged

    while True:
        # Offline execution mode
        if is_executing:
            if traj_index == 0:
                # Plan once at the beginning
                current_cfg = np.array([slider.value for slider in slider_handles])
                box_coll_world_current = box_spheres.transform_from_wxyz_position(
                    wxyz=np.array(box_handle.wxyz),
                    position=np.array(box_handle.position),
                )
                print(f'Box position: {box_handle.position}, wxyz: {box_handle.wxyz}')
                world_coll_list = [plane_coll,
                                   box_coll_world_current]

                # Linear interpolation from current_cfg to current_mc_cfg
                prev_sols_interp = np.linspace(current_cfg, current_mc_cfg, len_traj + 1)[1:]

                print("Planning trajectory...")
                sol_traj, sol_pos, sol_wxyz = pks.wu_solve_online_planning(
                    robot=robot,
                    robot_coll=robot_coll,
                    world_coll=world_coll_list,
                    target_link_name=target_link_name,
                    target_position=np.array(ik_target_handle.position),
                    target_wxyz=np.array(ik_target_handle.wxyz),
                    timesteps=len_traj,
                    dt=dt,
                    start_cfg=current_cfg,
                    prev_sols=prev_sols_interp,
                    weight_pose_match_rotation=weight_pose_match_rotation.value,
                    weight_pose_match_translation=weight_pose_match_translation.value,
                    weight_pose_smoothness=weight_pose_smoothness.value,
                    weight_match_start_pose=weight_match_start_pose.value,
                    weight_match_joint_to_pose=weight_match_joint_to_pose.value,
                    weight_smoothness=weight_smoothness.value,
                    weight_limit_velocity=weight_limit_velocity.value,
                    weight_limit=weight_limit.value,
                    weight_rest=weight_rest.value,
                    weight_manipulability=weight_manipulability.value,
                    weight_self_collision=weight_self_collision.value,
                    weight_world_collision=weight_world_collision.value,
                )

                dof = 6
                max_vel = 3.14 * speed_percentage.value    # rad/s
                max_acc = 14.0 * speed_percentage.value    # rad/s² (800 deg/s² = 14 rad/s²)
                vel_limits = np.ones(dof) * max_vel
                acc_limits = np.ones(dof) * max_acc


                traj = generate_optimal_trajectory(np.vstack([current_cfg, sol_traj]),
                                                   vel_limits, acc_limits)
                node.send_joint_trajectory(traj)

                if hasattr(target_frame_handle, "batched_positions"):
                    target_frame_handle.batched_positions = np.array(sol_pos)
                    target_frame_handle.batched_wxyzs = np.array(sol_wxyz)
                else:
                    target_frame_handle.positions_batched = np.array(sol_pos)
                    target_frame_handle.wxyzs_batched = np.array(sol_wxyz)

            if traj_index < len(sol_traj):
                # Calculate and print distance to target
                target_link_idx = robot.links.names.index(target_link_name)
                current_fk = robot.forward_kinematics(sol_traj[traj_index])
                current_pos = current_fk[target_link_idx][4:]
                target_pos = np.array(ik_target_handle.position)
                distance = np.linalg.norm(current_pos - target_pos)
                print(f'Step {traj_index}/{len(sol_traj)}: {distance=:.6f}')
                update_robot_visualization(
                    urdf_vis, slider_handles, robot, robot_coll, server, sol_traj[traj_index]
                )
                traj_index += 1
            else:
                # Finished executing
                is_executing = False
                plan_button.name = "Start Planning"
                print("Trajectory execution completed!")
                traj_index = 0


if __name__ == "__main__":
    main()
