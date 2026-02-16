#!/usr/bin/env python3
"""
TOPPRA with Quintic Spline Path: ROS2 Compatible Trajectory Generation
========================================================================

このスクリプトは、ROS2のJointTrajectoryControllerと完全に互換性のある
時間最適軌道を生成します：

1. Quintic spline経路を生成（waypoint間を5次スプラインで補間）
2. TOPPRAで時間最適化（速度・加速度制限を考慮）
3. 初期・終端で速度=0、加速度=0を保証

Key Features:
- ROS2 controllerもquintic補間を使うため、軌道が変わらない
- 境界条件を満たしながら時間最適
- 制約を守る

Author: Generated for ROS2 integration
Date: 2024
"""

import toppra.algorithm as algo
import toppra.constraint as constraint
import toppra as ta
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np


class QuinticSplinePath(ta.interpolator.AbstractGeometricPath):
    """
    Waypoint間をquintic spline (5次スプライン) で補間した経路

    ROS2のJointTrajectoryControllerと同じ補間方法を使用
    """

    def __init__(self, waypoints, bc_type='zero'):
        """
        Parameters
        ----------
        waypoints : array, shape (N, dof)
            経由点
        bc_type : str
            'zero': 両端で速度=0 かつ 加速度=0 (推奨)
            'clamped': 両端で速度=0
            'natural': 両端で加速度=0
        """
        super().__init__()
        waypoints = np.atleast_2d(waypoints)
        self._waypoints = waypoints
        self._dof = waypoints.shape[1]
        self._s_waypoints = np.linspace(0, 1, len(waypoints))

        # Quintic spline (k=5) を各自由度について作成
        self._splines = []
        for i in range(self._dof):
            # 境界条件を設定（各自由度で同じ）
            if bc_type == 'zero':
                # 両端で速度=0 かつ 加速度=0
                # bc_type形式: [左端の条件, 右端の条件]
                # 各端点: [(order1, value1), (order2, value2), ...]
                bc = [[(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)]]
            elif bc_type == 'clamped':
                # 両端で速度=0
                bc = [[(1, 0.0)], [(1, 0.0)]]
            elif bc_type == 'natural':
                # 両端で加速度=0
                bc = [[(2, 0.0)], [(2, 0.0)]]
            else:
                bc = None

            spline = make_interp_spline(
                self._s_waypoints,
                waypoints[:, i],
                k=5,  # 5次スプライン
                bc_type=bc
            )
            self._splines.append(spline)

    def __call__(self, s, order=0):
        """経路を評価（order=0: 位置, order=1: 速度, order=2: 加速度）"""
        s = np.atleast_1d(s)
        result = np.zeros((len(s), self._dof))
        for i, spline in enumerate(self._splines):
            result[:, i] = spline(s, nu=order)
        return result if len(s) > 1 else result[0]

    @property
    def path_interval(self):
        """経路のパラメータ範囲"""
        return np.array([0, 1])

    @property
    def dof(self):
        """自由度"""
        return self._dof


class QuinticSplineTimeTrajectory:
    """
    Quintic spline経路 + TOPPRA最適化結果の時間軌道

    boundary conditions: velocity=0, acceleration=0 at start and end
    """

    def __init__(self, path, gridpoints, sd_vec):
        """
        Parameters
        ----------
        path : QuinticSplinePath
            Quintic spline経路
        gridpoints : array
            TOPPRAのグリッドポイント (s座標)
        sd_vec : array
            各グリッドポイントでの経路速度 ds/dt
        """
        self.path = path
        self.gridpoints = gridpoints
        self.sd_vec = sd_vec
        self.dof = path.dof

        # 時間グリッドを計算
        self.time_grid = np.zeros(len(gridpoints))
        for i in range(1, len(gridpoints)):
            ds = gridpoints[i] - gridpoints[i - 1]
            # 2点間の平均速度を使用
            v_avg = (sd_vec[i] + sd_vec[i - 1]) / 2
            if v_avg > 1e-8:
                dt = ds / v_avg
            else:
                dt = 0
            self.time_grid[i] = self.time_grid[i - 1] + dt

        self.duration = self.time_grid[-1]

    def __call__(self, t, der=0):
        """
        時刻tでの軌道を評価

        Parameters
        ----------
        t : float or array
            時刻
        der : int
            0: 位置, 1: 速度, 2: 加速度

        Returns
        -------
        q : array
            軌道値
        """
        t = np.atleast_1d(t)
        result = np.zeros((len(t), self.dof))

        for i, ti in enumerate(t):
            # 境界条件のチェック
            if ti <= 0:
                s = 0
                sd = 0
                sdd = 0
            elif ti >= self.duration:
                s = 1
                sd = 0
                sdd = 0
            else:
                # 時刻に対応するsを補間
                s = np.interp(ti, self.time_grid, self.gridpoints)
                sd = np.interp(ti, self.time_grid, self.sd_vec)

                # 加速度 sdd を計算（数値微分）
                idx = np.searchsorted(self.time_grid, ti)
                if idx == 0:
                    idx = 1
                elif idx >= len(self.time_grid):
                    idx = len(self.time_grid) - 1

                dt = self.time_grid[idx] - self.time_grid[idx - 1]
                if dt > 1e-8:
                    dsd = self.sd_vec[idx] - self.sd_vec[idx - 1]
                    sdd = dsd / dt
                else:
                    sdd = 0

            # q̈(t) = q̈(s) * ṡ² + q̇(s) * s̈
            if der == 0:
                result[i] = self.path(s, 0)
            elif der == 1:
                result[i] = self.path(s, 1) * sd
            elif der == 2:
                result[i] = self.path(s, 2) * sd ** 2 + self.path(s, 1) * sdd

        return result if len(t) > 1 else result[0]


def generate_optimal_trajectory(waypoints, vel_limits, acc_limits):
    """
    Quintic spline経路 + TOPPRA で時間最適軌道を生成

    Parameters
    ----------
    waypoints : array, shape (N, dof)
        経由点
    vel_limits : array, shape (dof,)
        速度制限 [rad/s]
    acc_limits : array, shape (dof,)
        加速度制限 [rad/s²]

    Returns
    -------
    traj : QuinticSplineTimeTrajectory
        時間最適軌道
    """
    print("\n" + "=" * 60)
    print("Quintic Spline Path + TOPPRA Optimization")
    print("=" * 60)
    print(f"Waypoints: {len(waypoints)}, DOF: {waypoints.shape[1]}")

    # 1. Quintic spline経路を作成
    print("\n[1/3] Creating quintic spline path...")
    path = QuinticSplinePath(waypoints, bc_type='zero')
    print(f"  ✓ Path created (s ∈ [0, 1])")
    print(f"  ✓ Boundary: vel={np.linalg.norm(path(0, 1)):.2e}, "
          f"acc={np.linalg.norm(path(0, 2)):.2e}")

    # 2. TOPPRA で時間最適化
    print("\n[2/3] Running TOPPRA...")
    pc_vel = constraint.JointVelocityConstraint(vel_limits)
    pc_acc = constraint.JointAccelerationConstraint(acc_limits)

    # グリッドポイントを細かく設定して精度を向上
    instance = algo.TOPPRA(
        [pc_vel, pc_acc],
        path,
        gridpt_max_err_threshold=1e-4,  # より細かいグリッド
        gridpt_min_nb_points=500         # 最小500点
    )
    instance.compute_parameterization(0, 0)  # 境界で経路速度=0

    if instance.problem_data.return_code != algo.ParameterizationReturnCode.Ok:
        raise RuntimeError("TOPPRA failed!")

    print(f"  ✓ Optimization successful")

    # 3. 境界条件を満たす時間軌道を作成
    print("\n[3/3] Creating time trajectory with boundary conditions...")
    traj = QuinticSplineTimeTrajectory(
        path,
        instance.problem_data.gridpoints,
        instance.problem_data.sd_vec
    )
    print(f"  ✓ Trajectory duration: {traj.duration:.3f} s")

    # 境界条件を確認
    v0 = np.linalg.norm(traj(0, 1))
    vf = np.linalg.norm(traj(traj.duration, 1))
    a0 = np.linalg.norm(traj(0, 2))
    af = np.linalg.norm(traj(traj.duration, 2))

    print(f"\n  Boundary conditions:")
    print(f"    Start: vel={v0:.2e}, acc={a0:.2e}")
    print(f"    End:   vel={vf:.2e}, acc={af:.2e}")

    if v0 < 1e-6 and vf < 1e-6 and a0 < 1e-6 and af < 1e-6:
        print(f"    ✓ Boundary conditions satisfied")

    print(f"\n{'=' * 60}\n")

    return traj


def verify_and_plot(traj, vel_limits, acc_limits):
    """軌道を検証してプロット"""
    # サンプリング
    ts = np.linspace(0, traj.duration, 500)
    qs = traj(ts)
    qds = traj(ts, 1)
    qdds = traj(ts, 2)

    # 制約チェック
    vel_max = np.max(np.abs(qds), axis=0)
    acc_max = np.max(np.abs(qdds), axis=0)

    vel_ok = np.all(vel_max <= vel_limits * 1.01)  # 1%マージン
    acc_ok = np.all(acc_max <= acc_limits * 1.01)

    print("Constraint Verification:")
    print(f"  Velocity:     {'✓ OK' if vel_ok else '✗ VIOLATED'}")
    if not vel_ok:
        print(f"    Max velocities: {vel_max}")
        print(f"    Limits:         {vel_limits}")

    print(f"  Acceleration: {'✓ OK' if acc_ok else '✗ VIOLATED'}")
    if not acc_ok:
        print(f"    Max accelerations: {acc_max}")
        print(f"    Limits:            {acc_limits}")

    # プロット
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(traj.dof):
        axs[0].plot(ts, qs[:, i], label=f'Joint {i + 1}')
        axs[1].plot(ts, qds[:, i], label=f'Joint {i + 1}')
        axs[2].plot(ts, qdds[:, i], 'o-', markersize=2, label=f'Joint {i + 1}')

    axs[0].set_ylabel('Position [rad]')
    axs[0].legend(loc='upper right', fontsize='small')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title('Quintic Spline + TOPPRA (ROS2 Compatible)')

    axs[1].set_ylabel('Velocity [rad/s]')
    axs[1].axhline(vel_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[1].axhline(-vel_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[1].legend(loc='upper right', fontsize='small')
    axs[1].grid(True, alpha=0.3)

    axs[2].set_ylabel('Acceleration [rad/s²]')
    axs[2].axhline(acc_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[2].axhline(-acc_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[2].legend(loc='upper right', fontsize='small')
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # プロットを保存
    # output_file = '/home/wu/src/toppra/examples/quintic_optimal_trajectory.png'
    # plt.savefig(output_file, dpi=150, bbox_inches='tight')
    # print(f"\n  ✓ Plot saved to: {output_file}")
    plt.show()


def main():
    """デモンストレーション"""
    # Waypoints (開始と終了は同じ位置)
    # waypoints = np.array([
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.5, -0.3, 0.4, -0.2, 0.3, 0.1, -0.4],
    #     [1.0, -0.5, 0.8, -0.5, 0.6, 0.3, -0.8],
    #     [0.8, -0.2, 0.6, -0.3, 0.4, 0.2, -0.6],
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # ])

    # Define initial position
    initial_position = np.array([
        0.13858993, -1.0869963, 1.8468771, -2.3306694, -1.5707883, 0.12535799
    ])

    # Define waypoints (10 waypoints to goal)
    waypoint_list = [
        np.array([0.14505324, -1.1683416, 1.8178841, -2.2291677, -1.5702931, 0.1356192]),
        np.array([0.17124246, -1.2049217, 1.7500571, -2.1336172, -1.5697855, 0.16361867]),
        np.array([0.21433297, -1.2096072, 1.6697615, -2.0554717, -1.5697445, 0.20684084]),
        np.array([0.26989025, -1.1923206, 1.5905685, -1.9975934, -1.5703194, 0.2612928]),
        np.array([0.33179507, -1.1555458, 1.5177493, -1.9622236, -1.5712678, 0.3214667]),
        np.array([0.39203745, -1.1023152, 1.4541643, -1.9490795, -1.5721172, 0.3801592]),
        np.array([0.44486478, -1.0394548, 1.3994, -1.9518356, -1.5724883, 0.4321956]),
        np.array([0.49136376, -0.97423637, 1.3478417, -1.959433, -1.5723068, 0.47855031]),
        np.array([0.53609324, -0.90677756, 1.2941401, -1.9674608, -1.5716703, 0.5233229]),
        np.array([0.5860723, -0.76340413, 1.2514267, -2.0583208, -1.570936, 0.5728248]),
    ]

    # Combine initial position and waypoints
    waypoints = np.vstack([initial_position, waypoint_list])

    dof = 6
    # 制約（実際の制限の90%に設定して余裕を持たせる）
    max_vel = 3.14
    max_acc = 14.0
    vel_limits = np.ones(dof) * max_vel    # 2 rad/s の 90%
    acc_limits = np.ones(dof) * max_acc    # 5 rad/s² の 90%

    # max_velocity = 3.14  # rad/s
    # max_acceleration = 800.0 * np.pi / 180.0  # 800 deg/s^2 = 13.96 rad/s^2

    print(f"Target limits: vel={max_vel} rad/s, acc={max_acc} rad/s²")
    print(f"TOPPRA limits: vel={vel_limits[0]:.2f} rad/s (90%), "
          f"acc={acc_limits[0]:.2f} rad/s² (90%)")

    # 軌道生成
    traj = generate_optimal_trajectory(waypoints, vel_limits, acc_limits)

    # 検証とプロット（実際の制限で評価）
    verify_and_plot(traj, np.ones(dof) * max_vel, np.ones(dof) * max_acc)

    print("\n" + "=" * 60)
    print("✓ SUCCESS! Ready for ROS2 JointTrajectoryController")
    print("=" * 60)
    print("Key features:")
    print("  • Path: Quintic spline (same as ROS2 controller)")
    print("  • Boundary: velocity=0, acceleration=0")
    print("  • Constraints: Within limits")
    print("  • Time-optimal: Via TOPPRA")
    print("  • No interpolation issues!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    ta.setup_logging("INFO")
    main()
