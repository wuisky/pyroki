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
from robot_descriptions.loaders.yourdfpy import (load_robot_description,
                                                 yourdfpy)
from viser.extras import ViserUrdf
from wutility import sample_even_fit_mesh, voxel_fit_volume_sample_surface_mesh

import pyroki as pk
from pyroki.collision import Capsule, HalfSpace, RobotCollision, Sphere


def main():
    """Main function for basic IK with collision."""
    # urdf = load_robot_description("panda_description")
    # target_link_name = "panda_hand"
    # urdf = load_robot_description("ur5_description")
    # target_link_name = "ee_link"
    # urdf = yourdfpy.URDF.load(
    #     str(Path(__file__).parent / '../ur5-bullet/UR5/ur_e_description/urdf/ur5e.urdf'))
    # target_link_name = "ee_link"
    urdf = yourdfpy.URDF.load(
        str(Path(__file__).parent / '../wur5e/ur5e.urdf'))
    target_link_name = "tool0"

    robot = pk.Robot.from_urdf(urdf)
    new_default = jnp.array([0., -1.57, 0., -1.57, 0., 0.], dtype=jnp.float32)
    robot.joint_var_cls.default_factory = staticmethod(lambda: new_default)

    robot_coll = RobotCollision.from_urdf(urdf)
    print(f'{robot_coll=}')
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg([0.0] * robot.joints.num_actuated_joints)
    # urdf_vis.update_cfg([0., -1.57, 0., -1.57, 0., 0.])
    prev_solution = np.array([0., -1.57, 0., -1.57, 0., 0.])

    rest_weight = server.gui.add_slider(
        "Rest Weight", 0.0, 0.5, 0.01, 0.0
    )
    pose_weight = server.gui.add_slider(
        "Pose Weight", 5.0, 50.0, 1.0, 10.0
    )

    collision_weight = server.gui.add_slider(
        "Collision Weight", 0.0, 30.0, 1.0, 0.0
    )

    # Create interactive controller for IK target.
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2,
        # panda
        # position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
        # ur
        # position=(0.25,  0.13, 0.6), wxyz=(0, -0.707, 0.707, 0)
        position=(0.36595766, -0.19022489,  0.40833779),
        wxyz=(-0.02906002,  0.99836426, -0.04604809,  0.00133575),
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

    # mesh = trimesh.load_mesh("storage_box.stl")
    # mesh.apply_scale(0.005)
    # box_handle = server.scene.add_transform_controls(
    #     "/box", scale=0.2,
    #     wxyz=(0.707, 0.707, 0, 0),
    #     position=(0.75, -0.3, 0)
    # )
    # # viserで可視化
    # server.scene.add_mesh_trimesh("/box/visual", mesh=mesh)
    # # pts, radius = voxel_fit_volume_sample_surface_mesh(mesh, n_spheres=500,
    # #                                                    surface_sphere_radius=0.005)
    # pts, radius = sample_even_fit_mesh(mesh, n_spheres=500, sphere_radius=0.005)
    # print(f'{type(pts)=}, {pts=}')
    # box_spheres = pk.collision.Sphere.from_center_and_radius(
    #     center=pts, radius=radius)
    # server.scene.add_mesh_trimesh(
    #     "/box/coll", mesh=box_spheres.to_trimesh(),
    # )

    while True:
        start_time = time.time()
        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )

        # # storage box
        # box_coll_world_current = box_spheres.transform_from_wxyz_position(
        #     wxyz=np.array(box_handle.wxyz),
        #     position=np.array(box_handle.position),
        # )

        # world_coll_list = [plane_coll, sphere_coll_world_current]
        # world_coll_list = [sphere_coll_world_current,
        #                    box_coll_world_current]
        world_coll_list = [sphere_coll_world_current]
        weights = np.ones(len(world_coll_list)) * 10.0
        # if not calc_collision.value:
        #     weights[-1] = 0.0
        weights[-1] = collision_weight.value

        solution = pks.solve_ik_with_collision(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=np.array(ik_target_handle.position),
            target_wxyz=np.array(ik_target_handle.wxyz),
            weights=weights,
            pose_weight=pose_weight.value,
            rest_weight=rest_weight.value,
            initial_joint_angles=prev_solution,
        )
        # print(f'{solution=}')
        # print(f'{ik_target_handle.position=}, {ik_target_handle.wxyz=}')
        print(f'{sphere_handle.position=}, {sphere_handle.wxyz=}')

        # Update timing handle.
        timing_handle.value = (time.time() - start_time) * 1000

        # wrapped = (solution + np.pi) % (2 * np.pi) - np.pi
        # prev_solution = np.where(np.abs(solution - np.pi) > 1e-8, wrapped, solution)
        # Update visualizer.
        # urdf_vis.update_cfg(prev_solution)
        # print(f'{prev_solution=}')

        urdf_vis.update_cfg(solution)
        prev_solution = solution
        # print(f'{prev_solution=}')
        # new_default = jnp.array(solution)
        # robot = pk.Robot.from_urdf(urdf)
        # robot.joint_var_cls.default_factory = staticmethod(lambda: new_default)

        # update the robot collision mesh
        robot_coll_mesh = robot_coll.at_config(robot, solution).to_trimesh()
        # robot_coll_mesh = robot_coll.at_config(
        #     robot, solution).decompose_to_spheres(5).to_trimesh()
        server.scene.add_mesh_trimesh(
            "/robot_coll", mesh=robot_coll_mesh)

        # server.scene.remove_by_name("/robot_coll")
        # time.sleep(1)
        # break


if __name__ == "__main__":
    main()
