"""Online Planning

Run online planning in collision aware environments.
"""

import time
from pathlib import Path

import numpy as np
import pyroki_snippets as pks
import viser
from robot_descriptions.loaders.yourdfpy import (load_robot_description,
                                                 yourdfpy)
from viser.extras import ViserUrdf

import pyroki as pk
from pyroki.collision import HalfSpace, RobotCollision, Sphere


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
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main():
    # global sol_traj
    """Main function for online planning with collision."""
    # urdf = load_robot_description("panda_description")
    # target_link_name = "panda_hand"

    # urdf = load_robot_description("ur5_description")
    # target_link_name = "ee_link"
    urdf = yourdfpy.URDF.load(
        str(Path(__file__).parent / '../wur5e/ur5e.urdf'))
    target_link_name = "tool0"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)

    robot = pk.Robot.from_urdf(urdf)

    robot_coll = RobotCollision.from_urdf(urdf)
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Define the online planning parameters.
    len_traj, dt = 10, 0.1

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, urdf_vis
        )
        urdf_vis.update_cfg(initial_config)
        for q, s in zip(initial_config, slider_handles):
            s.value = q

    # Create interactive controller for IK target.
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.3, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )

    # Create interactive controller and mesh for the sphere obstacle.
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh(
        "/obstacle/mesh", mesh=sphere_coll.to_trimesh())
    target_frame_handle = server.scene.add_batched_axes(
        "target_frame",
        axes_length=0.05,
        axes_radius=0.005,
        batched_positions=np.zeros((25, 3)),
        batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * 25),
    )

    # timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # noneつけると[1,6]のshapeになる
    sol_traj = np.array(
        robot.joint_var_cls.default_factory()[None].repeat(len_traj, axis=0)
    )
    print(f'{robot.links.names=}')
    print(f'{sol_traj=}')

    # warming up
    plan_and_execute(sphere_coll, sphere_handle, ik_target_handle,
                     robot, robot_coll, plane_coll,
                     target_link_name, len_traj, dt,
                     urdf_vis, target_frame_handle, slider_handles)

    doit_btn = server.gui.add_button(
        label="plan&execute",  # ボタンに表示されるテキスト
    )

    @doit_btn.on_click
    def _(_) -> None:
        plan_and_execute(sphere_coll, sphere_handle, ik_target_handle,
                         robot, robot_coll, plane_coll,
                         target_link_name, len_traj, dt,
                         urdf_vis, target_frame_handle, slider_handles)

    print('ready to plan&execute')
    while True:
        time.sleep(10)


def plan_and_execute(sphere_coll, sphere_handle, ik_target_handle,
                     robot, robot_coll, plane_coll,
                     target_link_name, len_traj, dt,
                     urdf_vis, target_frame_handle, slider_handles):
    # start_time = time.time()
    sol_traj = []
    sol_pos, sol_wxyz = None, None
    for s in slider_handles:
        sol_traj.append(s.value)
    sol_traj = np.array(sol_traj)[None].repeat(len_traj, axis=0)

    sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
        wxyz=np.array(sphere_handle.wxyz),
        position=np.array(sphere_handle.position),
    )

    world_coll_list = [plane_coll, sphere_coll_world_current]
    sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(
        robot=robot,
        robot_coll=robot_coll,
        world_coll=world_coll_list,
        target_link_name=target_link_name,
        target_position=np.array(ik_target_handle.position),
        target_wxyz=np.array(ik_target_handle.wxyz),
        timesteps=len_traj,
        dt=dt,
        start_cfg=sol_traj[0],
        prev_sols=sol_traj,
    )

    # Update the planned trajectory visualization.
    if hasattr(target_frame_handle, "batched_positions"):
        target_frame_handle.batched_positions = np.array(
            sol_pos)  # type: ignore[attr-defined]
        target_frame_handle.batched_wxyzs = np.array(
            sol_wxyz)  # type: ignore[attr-defined]
    else:
        # This is an older version of Viser.
        target_frame_handle.positions_batched = np.array(
            sol_pos)  # type: ignore[attr-defined]
        target_frame_handle.wxyzs_batched = np.array(
            sol_wxyz)  # type: ignore[attr-defined]

    # Update visualizer.
    for sol in sol_traj:
        urdf_vis.update_cfg(sol)
        time.sleep(dt)
        for q, s in zip(sol, slider_handles):
            s.value = q


if __name__ == "__main__":
    main()
