"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki_snippets as pks
# from robot_descriptions.loaders.yourdfpy import load_robot_description
from robot_descriptions.loaders.yourdfpy import yourdfpy

import pyroki as pk


def main():
    """Main function for basic IK."""
    # urdf = load_robot_description("ur5_description")
    urdf = yourdfpy.URDF.load(
        str(Path(__file__).parent / '../ur5-bullet/UR5/ur_e_description/urdf/ur5e.urdf'))
    target_link_name = "tool0"

    # print(f'{urdf.link_map.values()=}')

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)
    print(f'{robot.links.names=}')
    target_link_index = robot.links.names.index(target_link_name)
    init_q = [0.0, -1.96, 1.31, -0.981, -1.57, 0.0]
    # result = robot.forward_kinematics(jnp.asarray([0.0, -1.96, 1.31, -0.981, -1.57, 0.0]))
    # result = robot.forward_kinematics(jnp.asarray(init_q))[target_link_index]
    result = robot.forward_kinematics(jnp.asarray(init_q))[target_link_index]
    print(f'{robot.forward_kinematics(jnp.asarray(init_q))=}')
    # result = robot.forward_kinematics(jnp.asarray(init_q))
    print(f'{result=}')
    Ts_link_world = jaxlie.SE3(result)
    print(f'{Ts_link_world.rotation().log()=}')
    print(f'{Ts_link_world.log()=}')

    target_position = [0.2566333,  0.1340796, 0.5996458]
    target_wxyz = [0.02156344, -0.70679474,  0.7067778, -0.02100065]

    T_world_targets = jaxlie.SE3(
        jnp.concatenate(
            [jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    print(f'{T_world_targets=}')

    T_offset = jaxlie.SE3(
        jnp.concatenate(
            [jnp.array([1, 0, 0, 0]), jnp.array([0.0, 0, 0.1])], axis=-1)
    )
    print(f'{T_world_targets @ T_offset=}')

    solution = pks.solve_ik(
        robot=robot,
        target_link_name=target_link_name,
        target_position=np.array(target_position),
        target_wxyz=np.array(target_wxyz),
        initial_joint_angles=init_q,
        rest_weight=1.3,
    )
    print(f'{solution=}')

    # while True:
    #     # Solve IK.
    #     start_time = time.time()
    #     solution = pks.solve_ik(
    #         robot=robot,
    #         target_link_name=target_link_name,
    #         target_position=np.array(ik_target.position),
    #         target_wxyz=np.array(ik_target.wxyz),
    #     )

    #     # Update timing handle.
    #     elapsed_time = time.time() - start_time
    #     timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

    #     # Update visualizer.
    #     urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
