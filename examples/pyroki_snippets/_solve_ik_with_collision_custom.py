"""
Solves the basic IK problem with collision avoidance.
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp

import pyroki as pk


def solve_ik_with_collision_custom(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    weights: Optional[onp.ndarray] = None,
    pose_weight: float = 5.0,
    rest_weight: float | onp.ndarray = 0.01,
    initial_joint_angles: Optional[onp.ndarray] = None,
    elbow_height_weight: float = 0.0,
    elbow_link_name: str = "forearm_link",
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: Sequence[str]. Length: num_targets.
        position: ArrayLike. Shape: (num_targets, 3), or (3,).
        wxyz: ArrayLike. Shape: (num_targets, 4), or (4,).

    Returns:
        cfg: ArrayLike. Shape: (robot.joint.actuated_count,).
    """
    if weights is None:
        weights = onp.ones(len(world_coll_list)) * 10
    assert len(weights) == len(world_coll_list)
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_idx = robot.links.names.index(target_link_name)

    # Convert elbow link name to index
    elbow_link_idx = robot.links.names.index(elbow_link_name)

    T_world_targets = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    cfg = _solve_ik_with_collision_custom_jax(
        robot,
        coll,
        world_coll_list,
        T_world_targets,
        jnp.array(target_link_idx),
        weights,
        pose_weight,
        rest_weight,
        jnp.array(
            initial_joint_angles) if initial_joint_angles is not None else robot.joint_var_cls.default_factory(),
        elbow_height_weight,
        jnp.array(elbow_link_idx, dtype=jnp.int32),  # Pass index instead of name
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_with_collision_custom_jax(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
    weights: onp.ndarray,
    pose_weight: float,
    rest_weight: float | jax.Array,
    initial_joint_angles: jnp.ndarray,
    elbow_height_weight: float,
    elbow_link_index: jax.Array,  # Changed from str to jax.Array
) -> jax.Array:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    # original_default = robot.joint_var_cls.default_factory()
    # robot.joint_var_cls.default_factory = staticmethod(lambda: jnp.array(initial_joint_angles))
    joint_var = robot.joint_var_cls(0)  # 0 is for id
    vars = [joint_var]

    # jax.debug.print("ini_q: {x}", x=initial_joint_angles)
    init_vals = jaxls.VarValues.make([vars[0].with_value(initial_joint_angles)])
    # init_vals = jaxls.VarValues.make({joint_var: initial_joint_angles})

    # jax.debug.print("rest val {q}", q=joint_var.default_factory())

    # Weights and margins defined directly in factors
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            # pos_weight=50.0,
            # ori_weight=10.0,
            pos_weight=pose_weight,
            ori_weight=pose_weight/5.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
        # default_joint_cfg = (joints.lower_limits + joints.upper_limits) / 2 defined in robot class
        # todo switch rest_pose to latest joint configuration
        pk.costs.rest_cost(
            joint_var,
            # rest_pose=jnp.array(joint_var.default_factory()),
            rest_pose=initial_joint_angles,
            weight=rest_weight,
        ),
        pk.costs.elbow_height_cost(
            robot,
            joint_var,
            link_index=elbow_link_index,
            weight=elbow_height_weight,
        ),
        # pk.costs.self_collision_cost(
        #     robot,
        #     robot_coll=coll,
        #     joint_var=joint_var,
        #     margin=0.02,
        #     weight=5.0,
        # ),

    ]
    costs.extend(
        [
            pk.costs.world_collision_cost(
                robot, coll, joint_var, world_coll, 0.05, weight
            )
            for world_coll, weight in zip(world_coll_list, weights)
        ]
    )

    sol = (
        jaxls.LeastSquaresProblem(costs, vars)
        .analyze()
        .solve(
            initial_vals=init_vals,
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
