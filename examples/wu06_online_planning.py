"""Online Planning

Run online planning in collision aware environments.
"""

from wutility import voxel_fit_volume_sample_surface_mesh
from wutility import sample_even_fit_mesh
from viser.extras import ViserUrdf
import viser
import trimesh
from robot_descriptions.loaders.yourdfpy import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
import pyroki_snippets as pks
from pyroki.collision import Sphere
from pyroki.collision import RobotCollision
from pyroki.collision import HalfSpace
import pyroki as pk
import numpy as np
import json
import os
from pathlib import Path
import time

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

# Enable JAX persistent compilation cache
# Use a permanent directory (not /tmp which is cleared on reboot)
os.environ["JAX_COMPILATION_CACHE_DIR"] = str(Path.home() / ".cache" / "jax")
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"


# Enable JAX persistent compilation cache
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
# JAX 0.7+ uses different config names
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


NUM_OBJECT_SPHERES = 100
NUM_HAND_SPHERES = 100

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


def time_parameterize_toppra(
    waypoints: np.ndarray,
    max_velocity: float = 3.14,  # rad/s
    max_acceleration: float = 800.0 * np.pi / 180.0,  # 800 deg/s^2 -> rad/s^2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Time parameterization using TOPPRA with quintic spline interpolation.

    Args:
        waypoints: (N, n_dof) array of joint configurations
        max_velocity: Maximum joint velocity [rad/s]
        max_acceleration: Maximum joint acceleration [rad/s^2]

    Returns:
        ts_sample: Time points for interpolated trajectory
        qs_sample: Joint positions at sample points
        qds_sample: Joint velocities at sample points
        qdds_sample: Joint accelerations at sample points
    """

    n_waypoints, n_dof = waypoints.shape

    # Create path parameter (0 to 1)
    # Use arbitrary spacing - TOPPRA will optimize time intervals
    ss = np.linspace(0, 1, n_waypoints)

    # Create quintic spline path with zero velocity/acceleration at endpoints
    path = ta.SplineInterpolator(ss, waypoints, bc_type='clamped')

    # Create velocity and acceleration constraints
    # Stack lower and upper bounds as (n_dof, 2) array
    vlim = np.stack([
        np.full(n_dof, -max_velocity),
        np.full(n_dof, max_velocity)
    ], axis=1)  # shape (n_dof, 2)
    alim = np.stack([
        np.full(n_dof, -max_acceleration),
        np.full(n_dof, max_acceleration)
    ], axis=1)  # shape (n_dof, 2)

    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)

    # Setup and solve optimization problem
    instance = algo.TOPPRA(
        [pc_vel, pc_acc],
        path,
        solver_wrapper='seidel',
    )

    jnt_traj = instance.compute_trajectory()

    if jnt_traj is None:
        print("TOPPRA failed to find solution, using original waypoints")
        # Return simple linear interpolation as fallback
        ts_sample = np.linspace(0, (n_waypoints - 1) * 0.1, n_waypoints)
        qs_sample = waypoints
        qds_sample = np.zeros_like(waypoints)
        qdds_sample = np.zeros_like(waypoints)
        return ts_sample, qs_sample, qds_sample, qdds_sample

    # Sample the time-parameterized trajectory
    duration = jnt_traj.duration
    ts_sample = np.linspace(0, duration, int(duration / 0.1))  # 10 Hz sampling
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)  # First derivative
    qdds_sample = jnt_traj(ts_sample, 2)  # Second derivative

    print(f"TOPPRA: Original {n_waypoints} waypoints -> {len(ts_sample)} samples")
    print(f"Total duration: {duration:.3f}s")
    print(f"Max velocity: {np.max(np.abs(qds_sample)):.3f} rad/s")
    print(f"Max acceleration: {np.max(np.abs(qdds_sample)):.3f} rad/s^2")

    return ts_sample, qs_sample, qds_sample, qdds_sample


def spherelize_mesh(mesh: trimesh.Trimesh, n_spheres=500,
                    sphere_radius=0.005) -> pk.collision.Sphere:
    pts, radius = sample_even_fit_mesh(mesh, n_spheres=n_spheres,
                                       sphere_radius=sphere_radius)
    spheres = pk.collision.Sphere.from_center_and_radius(
        center=pts, radius=radius)
    return spheres


def main():
    """Main function for online planning with collision."""
    # urdf = load_robot_description("panda_description")
    # target_link_name = "panda_hand"
    # robot = pk.Robot.from_urdf(urdf)
    # robot_coll = RobotCollision.from_urdf(urdf)
    # urdf = load_robot_description("ur5_description")
    # target_link_name = "ee_link"
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
    hand_mesh = trimesh.load_mesh(
        str(Path(__file__).parent / '../cad/robotiq_2F_adaptive_gripper_rough.STL'))
    hand_mesh.apply_scale(0.001)
    sphere_hand_mesh = spherelize_mesh(
        hand_mesh, n_spheres=NUM_HAND_SPHERES, sphere_radius=0.002)

    # Attach hand as a new link to tool0
    robot_coll = robot_coll.attach_link(
        new_link_name='hand_gripper',
        parent_link_name='tool0',
        spheres=sphere_hand_mesh,
        ignore_self_collision=True,  # Ignore collision between hand and tool0
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

    # speed_percentage = server.gui.add_slider(
    #     label="Speed percentage",
    #     min=0.0,
    #     max=1.0,
    #     step=0.1,
    #     initial_value=0.5,
    # )

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
            np.array([value for value in qs_sample[time_step.value]])
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
                # sol_traj, sol_pos, sol_wxyz = pks.wu_solve_online_planning(
                qs_sample, sol_pos, sol_wxyz = pks.wu_solve_online_planning(
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

                # Apply TOPPRA time parameterization
                # print("Applying TOPPRA time parameterization...")
                # print(f'{sol_traj[0]=}')
                # ts_sample, qs_sample, qds_sample, qdds_sample = time_parameterize_toppra(
                #     waypoints=np.vstack([current_cfg, sol_traj]),
                #     max_velocity=speed_percentage.value * 3.14,  # rad/s
                #     max_acceleration=speed_percentage.value * 200.0 * np.pi / 180.0,  # 800 deg/s^2
                # )
                # print(f'{current_cfg=}')
                # print(f'{qs_sample=}')
                # print(f'{ts_sample=}')

                # # Update sol_traj with time-parameterized trajectory
                # # sol_traj = qs_sample

                # print(
                #     f"Time-parameterized trajectory: {len(sol_traj)} samples, "
                #     f"duration: {ts_sample[-1]:.3f}s"
                # )
                # node.send_joint_trajectory(
                #     ts_sample, qs_sample, qds_sample, qdds_sample, current_cfg)

                if hasattr(target_frame_handle, "batched_positions"):
                    target_frame_handle.batched_positions = np.array(sol_pos)
                    target_frame_handle.batched_wxyzs = np.array(sol_wxyz)
                else:
                    target_frame_handle.positions_batched = np.array(sol_pos)
                    target_frame_handle.wxyzs_batched = np.array(sol_wxyz)

            # Execute trajectory step by step
            # if traj_index < len(sol_traj):
            #     # Calculate and print distance to target
            #     target_link_idx = robot.links.names.index(target_link_name)
            #     current_fk = robot.forward_kinematics(sol_traj[traj_index])
            #     current_pos = current_fk[target_link_idx][4:]
            #     target_pos = np.array(ik_target_handle.position)
            #     distance = np.linalg.norm(current_pos - target_pos)
            #     print(f'Step {traj_index}/{len(sol_traj)}: {distance=:.6f}')
            #     print(f'{sol_traj[traj_index]=}')
            #     update_robot_visualization(
            #         urdf_vis, slider_handles, robot, robot_coll, server, sol_traj[traj_index]
            #     )
            #     traj_index += 1
            if traj_index < len(qs_sample):
                # Calculate and print distance to target
                target_link_idx = robot.links.names.index(target_link_name)
                current_fk = robot.forward_kinematics(qs_sample[traj_index])
                current_pos = current_fk[target_link_idx][4:]
                target_pos = np.array(ik_target_handle.position)
                distance = np.linalg.norm(current_pos - target_pos)
                print(f'Step {traj_index}/{len(qs_sample)}: {distance=:.6f}')
                update_robot_visualization(
                    urdf_vis, slider_handles, robot, robot_coll, server, qs_sample[traj_index]
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
