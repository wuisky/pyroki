"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp

import pyroki as pk


def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    initial_joint_angles: onp.ndarray = None,
    rest_weight: float = 1.0,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)

    # Use provided initial joint angles or default to zeros
    if initial_joint_angles:
        initial_joint_angles = jnp.array(initial_joint_angles)

    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        initial_joint_angles,
        rest_weight,
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    initial_joint_angles: jax.Array = None,
    rest_weight: float = 1.0,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    T_world_target = jaxlie.SE3(
        jnp.concatenate(
            [jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )

    init_vals = jaxls.VarValues.make({joint_var: initial_joint_angles})

    factors = [
        # pk.costs.pose_cost_analytic_jac(
        #     robot,
        #     joint_var,
        #     jaxlie.SE3.from_rotation_and_translation(
        #         jaxlie.SO3(target_wxyz), target_position
        #     ),
        #     target_link_index,
        #     pos_weight=50.0,
        #     ori_weight=10.0,
        # ),
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]
    if initial_joint_angles is not None:
        factors.append(
            pk.costs.rest_cost(
                joint_var,
                initial_joint_angles,
                rest_weight,
            ),
        )
        # timesteps = 5
        # traj_var_prev = robot.joint_var_cls(jnp.arange(0, timesteps - 1))
        # traj_var_next = robot.joint_var_cls(jnp.arange(1, timesteps))
        # factors.append(pk.costs.limit_velocity_cost(
        #     jax.tree.map(lambda x: x[None], robot),
        #     traj_var_prev,
        #     traj_var_next,
        #     weight=100,
        #     dt=0.1,
        # ))

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            initial_vals=init_vals,
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
