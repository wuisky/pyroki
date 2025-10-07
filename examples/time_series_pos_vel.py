import jax
import jax.numpy as jnp

# サンプル: 10ステップのjoint trajectory（6自由度ロボット）
timesteps = 10
dof = 6
q_traj = jnp.linspace(jnp.array([0.0, -1.0, 1.0, 0.0, -1.0, 0.0]),
                      jnp.array([1.0, 0.5, 0.0, 1.0, 0.5, 1.0]),
                      timesteps)  # shape: (10, 6)

dt = 0.1  # タイムステップ幅

# 速度計算関数（自動微分）
def joint_pos_fn(q):
    # q: (timesteps, dof)
    return q  # ここでは単純にq自身（forward_kinematics等に置き換えてもOK）

# 時系列の速度（有限差分）
vel_fd = (q_traj[1:] - q_traj[:-1]) / dt  # shape: (timesteps-1, dof)

# JAXの自動微分で時系列の速度を計算
# 各時刻tについて、q_traj[t]に対する勾配（速度）を計算
def get_velocity(q_traj):
    # q_traj: (timesteps, dof)
    # 各時刻の勾配（速度）をまとめて計算
    grad_fn = jax.vmap(jax.grad(lambda q: joint_pos_fn(q).sum()))
    return grad_fn(q_traj)

vel_ad = get_velocity(q_traj)  # shape: (timesteps, dof)

print("有限差分 velocity:\n", vel_fd)
print("自動微分 velocity:\n", vel_ad)
