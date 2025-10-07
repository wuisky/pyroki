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

import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R
import torch


def main():
    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)

    urdf_path = Path(__file__).parent / '../wur5e/ur5e.urdf'
    target_link_name = "tool0"
    robot_name = 'wur5e'

    th = [-1.717292, -2.453, 2.034, -1.226, 0.0, 0.0]

    urdf = yourdfpy.URDF.load(str(urdf_path),
                              # mesh_dir='wur5e'
                              )
    visible_links = {
        link_name: str(Path(__file__).parent.parent /
                       f'{robot_name}'/link.visuals[0].geometry.mesh.filename)
        for link_name, link in urdf.link_map.items()
        if link.visuals and link.visuals[0].geometry and
        link.visuals[0].geometry.mesh
        # if urdf.link_map[link_name].visuals
    }
    print(f'{visible_links=}')
    chain = pk.build_serial_chain_from_urdf(open(urdf_path, mode='rb').read(),
                                            target_link_name)
    fk_ret = chain.forward_kinematics(th, end_only=False)
    print(f'{fk_ret[target_link_name].get_matrix()=}')

    # for link_name, mesh_path in visible_links.items():
    #     print(f'Link: {link_name}, Mesh: {mesh_path}')
    #     mesh = trimesh.load(mesh_path, force='mesh')
    #     m = fk_ret[link_name].get_matrix()
    #     pos = m[:, :3, 3]
    #     r = R.from_matrix(m[:, :3, :3])
    #     quat = r.as_quat()[0]
    #     print(f'{pos=}')

    #     server.scene.add_mesh_trimesh(link_name,
    #                                   mesh=mesh,
    #                                   wxyz=(quat[3], quat[0], quat[1], quat[2]),
    #                                   position=pos[0],
    #                                   )
    for link_name, mesh_path in visible_links.items():
        link = urdf.link_map[link_name]
        mesh = trimesh.load(mesh_path, force='mesh')
        m = fk_ret[link_name].get_matrix()[0]  # shape (4,4)
        # visualのoriginを取得
        if link.visuals and link.visuals[0].origin is not None:
            T_visual_link = link.visuals[0].origin
        else:
            T_visual_link = np.eye(4)
        # 合成
        T_visual_world = m @ T_visual_link
        pos = T_visual_world[:3, 3]
        r = R.from_matrix(T_visual_world[:3, :3])
        quat = r.as_quat()  # (x, y, z, w)
        server.scene.add_mesh_trimesh(
            link_name,
            mesh=mesh,
            wxyz=(quat[3], quat[0], quat[1], quat[2]),
            position=pos,
        )

    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
