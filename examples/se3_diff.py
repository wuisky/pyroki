import jax
import jax.numpy as jnp
import jaxlie

def pose_error(q, robot, target_pose, link_index):
    # q: (6,) など
    T_current = jaxlie.SE3(robot.forward_kinematics(q)[link_index])
    err_se3 = (target_pose.inverse() @ T_current).log()  # (6,)
    return jnp.linalg.norm(err_se3)

# サンプルデータ
q = jnp.array([0.0, -1.96, 1.31, -0.981, -1.57, 0.0])
target_pose = jaxlie.SE3(
    jnp.concatenate([jnp.array([1, 0, 0, 0]), jnp.array([0.0, 0, 0.1])], axis=-1)
)
link_index = 5  # 例

# 偏微分（ヤコビアン）を計算
grad_fn = jax.grad(pose_error)
dq = grad_fn(q, robot, target_pose, link_index)
print("dq:", dq)
