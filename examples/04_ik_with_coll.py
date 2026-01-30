"""IK with Collision

Basic Inverse Kinematics with Collision Avoidance using PyRoKi.
"""

import time
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki_snippets as pks
import trimesh
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description, yourdfpy
from viser.extras import ViserUrdf
from wutility import sample_even_fit_mesh, voxel_fit_volume_sample_surface_mesh

import pyroki as pk
from pyroki.collision import Capsule, HalfSpace, RobotCollision, Sphere


def display_link_frames(
    server: viser.ViserServer,
    robot: pk.Robot,
    solution: np.ndarray,
    axes_length: float = 0.15,
    axes_radius: float = 0.002,
) -> jnp.ndarray:
    """Display coordinate frames for each link of the robot.

    Args:
        server: Viser server instance.
        robot: PyRoKi Robot instance.
        solution: Joint configuration.
        axes_length: Length of the coordinate frame axes.
        axes_radius: Radius of the coordinate frame axes.
    """
    Ts_link_world = robot.forward_kinematics(solution)
    for i, link_name in enumerate(robot.links.names):
        # Ts_link_world[i] is a SE3 representation (wxyz + xyz, shape (7,))
        # Convert to SE3 object first
        se3 = jaxlie.SE3(Ts_link_world[i])

        # Get position and quaternion
        wxyz_xyz = se3.wxyz_xyz
        wxyz = wxyz_xyz[:4]  # First 4 elements: quaternion (w, x, y, z)
        position = wxyz_xyz[4:]  # Last 3 elements: position (x, y, z)

        server.scene.add_frame(
            f"/link_frames/{link_name}",
            wxyz=np.array(wxyz),
            position=np.array(position),
            axes_length=axes_length,
            axes_radius=axes_radius,
        )
    return Ts_link_world


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


def spherelize_mesh(mesh: trimesh.Trimesh, n_spheres=500,
                    sphere_radius=0.005) -> pk.collision.Sphere:
    pts, radius = sample_even_fit_mesh(mesh, n_spheres=n_spheres,
                                       sphere_radius=sphere_radius)
    spheres = pk.collision.Sphere.from_center_and_radius(
        center=pts, radius=radius)
    return spheres


def main():
    """Main function for basic IK with collision."""
    # urdf = load_robot_description("panda_description")
    # target_link_name = "panda_hand"
    # urdf = load_robot_description("ur5_description")
    # target_link_name = "ee_link"
    # urdf = yourdfpy.URDF.load(
    #     str(Path(__file__).parent / '../ur5-bullet/UR5/ur_e_description/urdf/ur5e.urdf'))
    # target_link_name = "ee_link"

    # urdf_path = str(Path(__file__).parent / '../ur5e/ur5e.urdf')
    urdf_path = str(Path(__file__).parent / '../ur5e/ur5e.urdf.sphere')
    urdf = yourdfpy.URDF.load(urdf_path)
    target_link_name = "tool0"

    robot = pk.Robot.from_urdf(urdf)
    # robot_coll = RobotCollision.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf_spheres(urdf)
    print(f'{robot_coll=}')
    new_default = jnp.array([0.0, -1.57, 0., -1.57, 0., 0.], dtype=jnp.float32)
    robot.joint_var_cls.default_factory = staticmethod(lambda: new_default)

    # plane_coll = HalfSpace.from_point_and_normal(
    #     np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    # )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg([0.0] * robot.joints.num_actuated_joints)
    server.gui.configure_theme(dark_mode=True)

    with server.gui.add_folder("Joint position"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, urdf_vis
        )

    prev_solution = [0., -1.57, 0., -1.57, 0., 0.]
    rest_weight = server.gui.add_slider(
        # "Rest Weight", 0.0, 0.5, 0.01, 0.02
        # "Rest Weight", 0.0, 0.5, 0.01, 0.5
        "Rest Weight", 0.0, 30.0, 1.0, 0.0,
    )
    pose_weight = server.gui.add_slider(
        "Pose Weight", 5.0, 100.0, 1.0, 19.0
    )

    collision_weight = server.gui.add_slider(
        "Collision Weight", 0.0, 30.0, 1.0, 0.0
    )

    elbow_height_weight = server.gui.add_slider(
        "Elbow Height Weight", 0.0, 10.0, 0.001, 0.25
    )

    realtime_ik = server.gui.add_checkbox("Enable realtime ik", initial_value=True)

    # Create interactive controller for IK target.
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2,
        # panda
        # position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
        # ur
        position=(0.33980408, -0.12158905, 0.5132406),
        wxyz=(0, 0.707, -0.707, 0)
        # position=(-0.36595766, -0.19022489,  0.40833779),
        # wxyz=(-0.02906002,  0.99836426, -0.04604809,  0.00133575),

    )

    # Create interactive controller and mesh for the sphere obstacle.
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2,
        # position=(0.4, 0.3, 0.4)
        # position=(0.6857053, 0.11284989, 0.3196243),
        # position=(0.6857053, 0.08088909, 0.3196243),
        # wurdf
        # position=(0.6857053, 0.05182334, 0.3196243),
        # position=(0.47471599, -0.01089903,  0.21139717),
        position=(0.47471599, -0.34864163,  0.21139717),
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # wu try
    # calc_collision = server.gui.add_checkbox("collision",
    #                                          initial_value=True)

    hand_mesh = trimesh.load_mesh(
        str(Path(__file__).parent / '../cad/robotiq_2F_adaptive_gripper_rough.STL'))
    hand_mesh.apply_scale(0.001)
    sphere_hand_mesh = spherelize_mesh(hand_mesh, n_spheres=100, sphere_radius=0.002)

    offset = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.identity(),
        translation=jnp.array([0.0, 0.0, 0.1])  # Z軸方向に10cm
    )

    # Attach hand as a new link to tool0
    robot_coll = robot_coll.attach_link(
        new_link_name='hand_gripper',
        parent_link_name='tool0',
        spheres=sphere_hand_mesh,
        ignore_self_collision=True,  # Ignore collision between hand and tool0
    )
    print(f'After attaching hand: {robot_coll.link_names=}')
    print(f'After attaching hand: {robot_coll.num_spheres_per_link=}')
    print(f'After attaching hand: {robot_coll.parent_link_indices=}')

    # Create 100 spheres with radius 0 at origin
    dummy_centers = np.zeros((100, 3))  # All at [0, 0, 0]
    dummy_radii = np.zeros(100)  # All radius = 0
    dummy_obj = Sphere.from_center_and_radius(dummy_centers, dummy_radii)

    robot_coll = robot_coll.attach_link(
        new_link_name='object',
        parent_link_name='tool0',
        spheres=dummy_obj,
        ignore_self_collision=True,  # Ignore collision between hand and tool0
        offset=offset
    )

    obj_mesh = trimesh.load_mesh(
        str(Path(__file__).parent / '../cad/Bunny.stl'))
    obj_mesh.apply_scale(0.001)
    obj_spheres_raw = spherelize_mesh(obj_mesh, n_spheres=100, sphere_radius=0.002)

    server.scene.add_mesh_trimesh(
        "/bunny", mesh=obj_mesh, position=(0, 1, 0))

    # Ensure exactly 100 spheres by padding if necessary
    obj_centers = obj_spheres_raw.pose.translation()
    obj_radii = obj_spheres_raw.radius
    num_obj_spheres = obj_centers.shape[0]

    if num_obj_spheres < 100:
        # Pad to 100 spheres with zero radius
        pad = 100 - num_obj_spheres
        obj_centers = jnp.concatenate(
            [obj_centers, jnp.zeros((pad, 3), dtype=obj_centers.dtype)], axis=0)
        obj_radii = jnp.concatenate(
            [obj_radii, jnp.zeros((pad,), dtype=obj_radii.dtype)], axis=0)
    elif num_obj_spheres > 100:
        # Truncate to 100 spheres
        obj_centers = obj_centers[:100]
        obj_radii = obj_radii[:100]

    obj_spheres = Sphere.from_center_and_radius(obj_centers, obj_radii)
    print(f'Object spheres: {obj_spheres.get_batch_axes()[0]} spheres')

    mesh = trimesh.load_mesh(str(Path(__file__).parent / 'storage_box.stl'))
    mesh.apply_scale(0.005)
    box_handle = server.scene.add_transform_controls(
        "/box", scale=0.2,
        wxyz=(0.707, 0.707, 0, 0),
        position=(0.75, -0.3, 0)
    )
    # viserで可視化
    server.scene.add_mesh_trimesh("/box/visual", mesh=mesh)
    # pts, radius = voxel_fit_volume_sample_surface_mesh(mesh, n_spheres=500,
    #                                                    surface_sphere_radius=0.005)
    pts, radius = sample_even_fit_mesh(mesh, n_spheres=200, sphere_radius=0.005)
    print(f'{type(pts)=}, {pts=}')
    box_spheres = pk.collision.Sphere.from_center_and_radius(
        center=pts, radius=radius)
    server.scene.add_mesh_trimesh(
        "/box/coll", mesh=box_spheres.to_trimesh(),
    )

    detach_hand = server.gui.add_button(
        label="Detach Hand",
    )

    solve_ik = server.gui.add_button(
        label="Solve IK",
    )

    grasp_object = server.gui.add_button(
        label="Grasp Object",
    )

    release_object = server.gui.add_button(
        label="Release Object",
    )

    @grasp_object.on_click
    def _(_) -> None:
        nonlocal robot_coll
        # Update 'object' link spheres with actual object geometry
        robot_coll = robot_coll.update_link_spheres(
            link_name='object',
            new_spheres=obj_spheres,
            offset=offset
        )
        print(f'Object grasped: {robot_coll.num_spheres_per_link=}')

    @release_object.on_click
    def _(_) -> None:
        nonlocal robot_coll
        # Reset 'object' link spheres to zero radius (invisible)
        dummy_centers = np.zeros((100, 3))
        dummy_radii = np.zeros(100)
        dummy_obj = Sphere.from_center_and_radius(dummy_centers, dummy_radii)
        robot_coll = robot_coll.update_link_spheres(
            link_name='object',
            new_spheres=dummy_obj,
            offset=offset
        )
        print(f'Object released: {robot_coll.num_spheres_per_link=}')

    @detach_hand.on_click
    def _(_) -> None:
        nonlocal robot_coll
        # Detach the hand gripper link
        robot_coll = robot_coll.detach_link('hand_gripper')
        print(f'After detaching: {robot_coll.link_names=}')
        print(f'After detaching: {robot_coll.num_spheres_per_link=}')

    @solve_ik.on_click
    def _(_) -> None:
        nonlocal prev_solution
        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )
        world_coll_list = [sphere_coll_world_current]
        weights = np.ones(len(world_coll_list)) * 10.0
        # if not calc_collision.value:
        #     weights[-1] = 0.0
        weights[-1] = collision_weight.value

        # for rest weight per joint
        rest_weights = np.array([rest_weight.value] * robot.joints.num_actuated_joints)
        rest_weights[0] = 0  # allow joint rotation intensly

        solution = pks.solve_ik_with_collision(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=np.array(ik_target_handle.position),
            target_wxyz=np.array(ik_target_handle.wxyz),
            weights=weights,
            pose_weight=pose_weight.value,
            rest_weight=rest_weights,
            initial_joint_angles=prev_solution,
            elbow_height_weight=elbow_height_weight.value,
            elbow_link_name='forearm_link',
        )

        urdf_vis.update_cfg(solution)
        # prev_solution = solution
        prev_solution = solution.tolist()

        for s, q in zip(slider_handles, solution):
            s.value = q

        # update the robot collision mesh
        robot_coll_mesh = robot_coll.at_config(robot, solution).to_trimesh()
        # robot_coll_mesh = robot_coll.at_config(
        #     robot, solution).decompose_to_spheres(5).to_trimesh()
        server.scene.add_mesh_trimesh(
            "/robot_coll", mesh=robot_coll_mesh, visible=False)

        # Display coordinate frames for each link
        display_link_frames(server, robot, solution)

    while True:
        if not realtime_ik.value:
            time.sleep(0.1)
            continue

        start_time = time.time()
        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )

        # storage box
        box_coll_world_current = box_spheres.transform_from_wxyz_position(
            wxyz=np.array(box_handle.wxyz),
            position=np.array(box_handle.position),
        )

        # world_coll_list = [plane_coll, sphere_coll_world_current]
        world_coll_list = [sphere_coll_world_current,
                           box_coll_world_current]
        # world_coll_list = [sphere_coll_world_current]
        weights = np.ones(len(world_coll_list)) * collision_weight.value
        # if not calc_collision.value:
        #     weights[-1] = 0.0
        # weights[-1] = collision_weight.value

        # for rest weight per joint
        rest_weights = np.array([rest_weight.value] * robot.joints.num_actuated_joints)
        rest_weights[0] = 0  # allow joint rotation intensly

        solution = pks.solve_ik_with_collision(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=np.array(ik_target_handle.position),
            target_wxyz=np.array(ik_target_handle.wxyz),
            weights=weights,
            pose_weight=pose_weight.value,
            rest_weight=rest_weights,
            initial_joint_angles=prev_solution,
            elbow_height_weight=elbow_height_weight.value,
            elbow_link_name='forearm_link',
        )

        # print(f'{solution=}')
        # print(f'{ik_target_handle.position=}, {ik_target_handle.wxyz=}')
        # print(f'{sphere_handle.position=}, {sphere_handle.wxyz=}')

        # # Normalize solution to be within ±pi of prev_solution
        # diff = solution - prev_solution
        # diff = (diff + np.pi) % (2 * np.pi) - np.pi
        # solution = prev_solution + diff

        # wrapped = (solution + np.pi) % (2 * np.pi) - np.pi
        # prev_solution = np.where(np.abs(solution - np.pi) > 1e-8, wrapped, solution)
        # Update visualizer.
        # urdf_vis.update_cfg(prev_solution)
        # print(f'{prev_solution=}')

        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        urdf_vis.update_cfg(solution)
        # prev_solution = solution
        prev_solution = solution.tolist()

        for s, q in zip(slider_handles, solution):
            s.value = q
        # print(f'{prev_solution=}')
        # new_default = jnp.array(solution)
        # robot = pk.Robot.from_urdf(urdf)
        # robot.joint_var_cls.default_factory = staticmethod(lambda: new_default)

        # update the robot collision mesh
        robot_coll_mesh = robot_coll.at_config(robot, solution).to_trimesh()
        # robot_coll_mesh = robot_coll.at_config(
        #     robot, solution).decompose_to_spheres(5).to_trimesh()
        server.scene.add_mesh_trimesh(
            "/robot_coll", mesh=robot_coll_mesh, visible=False)

        # Display coordinate frames for each link
        Ts_link_world = display_link_frames(server, robot, solution)
        # target_link_index = robot.links.names.index('forearm_link')
        # pos = jaxlie.SE3(Ts_link_world[target_link_index]).translation()[2]
        # print(f'forearm_link position: {pos}')
        # server.scene.remove_by_name("/robot_coll")
        # time.sleep(1)
        # break
        # tool_index = robot.links.names.index('tool0')
        # tool_pose = jaxlie.SE3(Ts_link_world[tool_index])
        # wxyz_xyz = tool_pose.wxyz_xyz
        # server.scene.add_mesh_trimesh(
        #     "/robot/hand",
        #     mesh=sphere_hand_mesh.to_trimesh(),
        #     wxyz=np.array(wxyz_xyz[:4]),
        #     position=np.array(wxyz_xyz[4:]),
        # )


if __name__ == "__main__":
    main()
