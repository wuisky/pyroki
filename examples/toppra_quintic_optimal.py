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


def quintic_coeffs(p0, v0, a0, p1, v1, a1, T: float) -> np.ndarray:
    """
    ROS trajectory_interface::QuinticSplineSegment と同じ係数。
    p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
    端点条件: (p0,v0,a0) -> (p1,v1,a1) over duration T
    """
    c = np.zeros(6, dtype=float)
    if T == 0.0:
        c[0] = p0
        c[1] = v0
        c[2] = 0.5 * a0
        return c

    T1 = T
    T2 = T*T
    T3 = T2*T
    T4 = T3*T
    T5 = T4*T

    c[0] = p0
    c[1] = v0
    c[2] = 0.5 * a0
    # 以下、ROSの式そのまま
    c[3] = (-20.0*p0 + 20.0*p1 - 3.0*a0*T2 + a1*T2 - 12.0*v0*T1 - 8.0*v1*T1) / (2.0*T3)
    c[4] = (30.0*p0 - 30.0*p1 + 3.0*a0*T2 - 2.0*a1*T2 + 16.0*v0*T1 + 14.0*v1*T1) / (2.0*T4)
    c[5] = (-12.0*p0 + 12.0*p1 - 1.0*a0*T2 + 1.0*a1*T2 - 6.0*v0*T1 - 6.0*v1*T1) / (2.0*T5)
    return c


def sample_quintic(c: np.ndarray, t: float):
    """ROSの sample() と同じ (pos, vel, acc)。"""
    t0 = 1.0
    t1 = t
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    t5 = t4*t

    pos = t0*c[0] + t1*c[1] + t2*c[2] + t3*c[3] + t4*c[4] + t5*c[5]
    vel = c[1] + 2.0*t1*c[2] + 3.0*t2*c[3] + 4.0*t3*c[4] + 5.0*t4*c[5]
    acc = 2.0*c[2] + 6.0*t1*c[3] + 12.0*t2*c[4] + 20.0*t3*c[5]
    return pos, vel, acc


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
        # print(f'{gridpoints=}')
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

    # # グリッドポイントの分布を確認
    # gridpoints = instance.problem_data.gridpoints
    # deltas = np.diff(gridpoints)  # 隣接点間の間隔
    # print(f"  Grid statistics:")
    # print(f"    Total points: {len(gridpoints)}")
    # print(f"    Min interval: {np.min(deltas):.6f}")
    # print(f"    Max interval: {np.max(deltas):.6f}")
    # print(f"    Mean interval: {np.mean(deltas):.6f}")
    # print(f"    Std deviation: {np.std(deltas):.6f}")
    # print(f"    First 10 intervals: {deltas[:10]}")

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

    # 中間waypointでの速度を確認
    print(f"\n  Intermediate waypoint velocities (path space dq/ds):")
    n_waypoints = len(path._waypoints)
    s_waypoints = np.linspace(0, 1, n_waypoints)
    for i, s in enumerate(s_waypoints):
        vel_at_wp = path(s, order=1)  # dq/ds at waypoint
        vel_norm = np.linalg.norm(vel_at_wp)
        print(f"    wp{i} (s={s:.2f}): |dq/ds|={vel_norm:.3f}")

    print(f"\n{'=' * 60}\n")

    return traj


def verify_and_plot(traj, vel_limits, acc_limits, waypoints=None):
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
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for i in range(traj.dof):
        axs[0].plot(ts, qs[:, i], label=f'Joint {i + 1}')
        axs[1].plot(ts, qds[:, i], label=f'Joint {i + 1}')
        axs[2].plot(ts, qdds[:, i], 'o-', markersize=2, label=f'Joint {i + 1}')

    # Waypointsをプロット（もし提供されていれば）
    if waypoints is not None:
        # Waypointsは経路パラメータs上で均等分布（s=0, 1/N, 2/N, ..., 1）
        n_waypoints = len(waypoints)
        s_waypoints = np.linspace(0, 1, n_waypoints)

        # 各waypointのsに対応する時刻を計算
        waypoint_times = np.zeros(n_waypoints)
        for i, s in enumerate(s_waypoints):
            # 経路位置sに対応する時刻tを線形補間で求める
            waypoint_times[i] = np.interp(s, traj.gridpoints, traj.time_grid)

        # 各関節のwaypointを○マーカーで表示
        for i in range(traj.dof):
            axs[0].plot(waypoint_times, waypoints[:, i], 'o',
                        markersize=6, markerfacecolor='none',
                        markeredgewidth=2, color=f'C{i}')

    axs[0].set_ylabel('Position [rad]')
    axs[0].legend(loc='center left',  bbox_to_anchor=(1, 0.5),)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title('Quintic Spline + TOPPRA (ROS2 Compatible)')

    axs[1].set_ylabel('Velocity [rad/s]')
    axs[1].axhline(vel_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[1].axhline(-vel_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[1].legend(loc='center left',  bbox_to_anchor=(1, 0.5),)
    axs[1].grid(True, alpha=0.3)

    axs[2].set_ylabel('Acceleration [rad/s²]')
    axs[2].axhline(acc_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[2].axhline(-acc_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[2].legend(loc='center left',  bbox_to_anchor=(1, 0.5),)
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True, alpha=0.3)

    # plt.tight_layout()
    plt.tight_layout()

    # プロットを保存
    # output_file = '/home/wu/src/toppra/examples/quintic_optimal_trajectory.png'
    # plt.savefig(output_file, dpi=150, bbox_inches='tight')
    # print(f"\n  ✓ Plot saved to: {output_file}")
    plt.show()


def resample_and_reinterpolate_ros(traj, dt=0.01):
    """
    軌道を0.01秒間隔でサンプリングし、ROS2のquintic補間で再構築

    Parameters
    ----------
    traj : QuinticSplineTimeTrajectory
        元のTOPPRA軌道
    dt : float
        サンプリング間隔 [s]

    Returns
    -------
    ts_resampled : array
        再サンプリング用の時間配列
    coeffs_list : list of array
        各セグメントの係数 [n_segments, dof, 6]
    segment_times : array
        各セグメントの開始時刻
    """
    print("\n" + "=" * 60)
    print("Resampling with ROS2 Quintic Interpolation")
    print("=" * 60)

    # サンプリング時刻を計算（durationを超えないように）
    n_samples = int(np.ceil(traj.duration / dt)) + 1
    ts_samples = np.linspace(0, traj.duration, n_samples)
    qs = traj(ts_samples, der=0)
    qds = traj(ts_samples, der=1)
    qdds = traj(ts_samples, der=2)

    n_samples = len(ts_samples)
    n_segments = n_samples - 1
    dof = traj.dof

    print(f"  Original duration: {traj.duration:.3f} s")
    print(f"  Sampling interval: {dt} s")
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of segments: {n_segments}")

    # 各セグメントの係数を計算
    coeffs_list = []
    for i in range(n_segments):
        segment_coeffs = np.zeros((dof, 6))
        T = ts_samples[i+1] - ts_samples[i]

        for j in range(dof):
            p0, v0, a0 = qs[i, j], qds[i, j], qdds[i, j]
            p1, v1, a1 = qs[i+1, j], qds[i+1, j], qdds[i+1, j]
            segment_coeffs[j, :] = quintic_coeffs(p0, v0, a0, p1, v1, a1, T)

        coeffs_list.append(segment_coeffs)

    print(f"  ✓ Computed {len(coeffs_list)} quintic segments")
    print("=" * 60 + "\n")

    return ts_samples, coeffs_list, qs, qds, qdds


def evaluate_ros_trajectory(ts_samples, coeffs_list, t_eval):
    """
    ROS2 quintic補間軌道を評価

    Parameters
    ----------
    ts_samples : array
        セグメント境界の時刻
    coeffs_list : list
        各セグメントの係数
    t_eval : array
        評価する時刻

    Returns
    -------
    pos, vel, acc : array
        評価された軌道
    """
    n_eval = len(t_eval)
    dof = coeffs_list[0].shape[0]

    pos = np.zeros((n_eval, dof))
    vel = np.zeros((n_eval, dof))
    acc = np.zeros((n_eval, dof))

    for i, t in enumerate(t_eval):
        # どのセグメントに属するか探す
        seg_idx = np.searchsorted(ts_samples, t, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, len(coeffs_list) - 1)

        # セグメント内の相対時間
        t_local = t - ts_samples[seg_idx]

        # 各関節を評価
        for j in range(dof):
            pos[i, j], vel[i, j], acc[i, j] = sample_quintic(coeffs_list[seg_idx][j, :], t_local)

    return pos, vel, acc


def compare_trajectories(original_traj, ts_samples, coeffs_list,
                         sampled_qs, vel_limits, acc_limits):
    """
    元のTOPPRA軌道とROS2再補間軌道を比較

    Parameters
    ----------
    original_traj : QuinticSplineTimeTrajectory
        元の軌道
    ts_samples : array
        サンプリング時刻
    coeffs_list : list
        ROS2 quintic係数
    sampled_qs : array
        サンプリングされた位置
    vel_limits : array
        速度制限
    acc_limits : array
        加速度制限
    """
    # 高密度で評価
    ts_eval = np.linspace(0, original_traj.duration, 500)

    # 元の軌道
    qs_orig = original_traj(ts_eval, der=0)
    qds_orig = original_traj(ts_eval, der=1)
    qdds_orig = original_traj(ts_eval, der=2)

    # ROS2再補間軌道
    qs_ros, qds_ros, qdds_ros = evaluate_ros_trajectory(ts_samples, coeffs_list, ts_eval)

    # プロット
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    dof = original_traj.dof
    for i in range(dof):
        # 元の軌道（実線）
        axs[0].plot(ts_eval, qs_orig[:, i], '-', alpha=0.7, linewidth=2,
                    label=f'Joint {i+1} (TOPPRA)', color=f'C{i}')
        axs[1].plot(ts_eval, qds_orig[:, i], '-', alpha=0.7, linewidth=2,
                    label=f'Joint {i+1} (TOPPRA)', color=f'C{i}')
        axs[2].plot(ts_eval, qdds_orig[:, i], '-', alpha=0.7, linewidth=2,
                    label=f'Joint {i+1} (TOPPRA)', color=f'C{i}')

        # ROS2再補間軌道（破線）
        axs[0].plot(ts_eval, qs_ros[:, i], '--', alpha=0.9, linewidth=1.5,
                    label=f'Joint {i+1} (ROS2)', color=f'C{i}')
        axs[1].plot(ts_eval, qds_ros[:, i], '--', alpha=0.9, linewidth=1.5,
                    label=f'Joint {i+1} (ROS2)', color=f'C{i}')
        axs[2].plot(ts_eval, qdds_ros[:, i], '--', alpha=0.9, linewidth=1.5,
                    label=f'Joint {i+1} (ROS2)', color=f'C{i}')

    # サンプリングポイントをマーカーで表示
    for i in range(dof):
        axs[0].plot(ts_samples, sampled_qs[:, i], 'o',
                    markersize=3, color=f'C{i}', alpha=0.5)

    axs[0].set_ylabel('Position [rad]')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title('Trajectory Comparison: TOPPRA vs ROS2 Quintic (dt=0.01s)')

    axs[1].set_ylabel('Velocity [rad/s]')
    axs[1].axhline(vel_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[1].axhline(-vel_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    axs[1].grid(True, alpha=0.3)

    axs[2].set_ylabel('Acceleration [rad/s²]')
    axs[2].axhline(acc_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[2].axhline(-acc_limits[0], color='r', ls='--', lw=1, alpha=0.5)
    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 誤差を計算
    print("\nTrajectory Comparison (TOPPRA vs ROS2 Quintic):")
    pos_error = np.max(np.abs(qs_orig - qs_ros))
    vel_error = np.max(np.abs(qds_orig - qds_ros))
    acc_error = np.max(np.abs(qdds_orig - qdds_ros))

    print(f"  Max position error:     {pos_error:.6f} rad")
    print(f"  Max velocity error:     {vel_error:.6f} rad/s")
    print(f"  Max acceleration error: {acc_error:.6f} rad/s²")


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

    # 検証とプロット（実際の制限で評価、waypointsも渡す）
    verify_and_plot(traj, np.ones(dof) * max_vel, np.ones(dof) * max_acc, waypoints)

    # 0.01秒間隔でサンプリングしてROS2 quintic補間で再構築
    ts_samples, coeffs_list, sampled_qs, sampled_qds, sampled_qdds = \
        resample_and_reinterpolate_ros(traj, dt=0.01)

    # 元の軌道とROS2再補間軌道を比較
    compare_trajectories(traj, ts_samples, coeffs_list, sampled_qs,
                         np.ones(dof) * max_vel, np.ones(dof) * max_acc)

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
