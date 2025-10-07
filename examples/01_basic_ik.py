"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time
from pathlib import Path

import numpy as np
import pyroki_snippets as pks
import trimesh
import viser
from robot_descriptions.loaders.yourdfpy import (load_robot_description,
                                                 yourdfpy)
from viser.extras import ViserUrdf
from wutility import (voxel_fit_volume_inside_mesh,
                      voxel_fit_volume_sample_surface_mesh)

import pyroki as pk
from pyroki.collision import Capsule, RobotCollision


def main():
    """Main function for basic IK."""

    # urdf = load_robot_description("panda_description")
    # target_link_name = "panda_hand"

    urdf = yourdfpy.URDF.load(
        str(Path(__file__).parent / '../wur5e/ur5e.urdf'))
    target_link_name = "tool0"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.2566333,  0.1340796, 0.5996458),
        wxyz=(0, -0.707, 0.707, 0)
        # wxyz=(0, -0.707, 0, 0.707)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    init_q = [0.0, -1.96, 1.31, -0.981, -1.57, 0.0]

    weight_handle = server.gui.add_slider(
        "Rest Weight", 0.0, 30.0, 1.0, 15.0
    )

    mybutton = server.gui.add_button(
        label="mybutton",  # ボタンに表示されるテキスト
    )

    @mybutton.on_click
    def _(_) -> None:
        print("Button pressed!")

    mesh = trimesh.load_mesh(str(Path(__file__).parent / './coord.stl'))
    mesh.apply_scale(0.001)
    # mesh = trimesh.load_mesh(str(Path(__file__).parent / './teapot.obj'))

    # mesh = trimesh.load_mesh(str(Path(__file__).parent / './storage_box.stl'))
    # mesh.apply_scale(0.005)
    # viserで可視化
    server.scene.add_mesh_trimesh("/imported_mesh",
                                  mesh=mesh,
                                  # wxyz=(0.707, 0.707, 0, 0),
                                  # position=(0.75, -0.3, 0)
                                  )

    # pts, radius = voxel_fit_volume_inside_mesh(mesh, n_spheres=200)
    pts, radius = voxel_fit_volume_sample_surface_mesh(
        mesh, n_spheres=500, surface_sphere_radius=0.005)
    spheres = pk.collision.Sphere.from_center_and_radius(
        center=pts, radius=radius)
    # server.scene.add_mesh_trimesh(
    #     "/box_coll", mesh=spheres.to_trimesh(),
    #     wxyz=(0.707, 0.707, 0, 0),
    #     position=(0.75, -0.3, 0)
    # )

    # capsule = Capsule.from_trimesh(mesh)

    testcap = pk.collision.Capsule.from_radius_height(
        position=np.array([[1.0, 0.0, 0.0]]),
        radius=np.array([[0.05]]),
        height=np.array([[0.2]]),
    )
    server.scene.add_mesh_trimesh("/testcap", mesh=testcap.to_trimesh())

    testsphere = pk.collision.Sphere.from_center_and_radius(
        np.array([0.0, 1.0, 0.0]), np.array([0.05]))
    server.scene.add_mesh_trimesh("/testshpere", mesh=testsphere.to_trimesh())

    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
            initial_joint_angles=init_q,
            rest_weight=weight_handle.value,
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)

        init_q = solution.tolist()
        robot_coll_mesh = robot_coll.at_config(robot, solution).to_trimesh()
        server.scene.add_mesh_trimesh(
            "/robot_coll", mesh=robot_coll_mesh)


if __name__ == "__main__":
    main()
