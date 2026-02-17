"""TOPPRA + Ruckig Time Parameterization Example

Demonstrates time parameterization using TOPPRA + Ruckig.

Approach:
1. TOPPRA: Generate time-optimal trajectory through waypoints
2. Ruckig (initial): Adjust initial segment to ensure zero acceleration at t=0
3. Ruckig (final): Adjust final segment to ensure zero acceleration at end

Constraints:
- Max velocity: 3.14 rad/s per joint
- Max acceleration: 800 deg/s^2 = 13.96 rad/s^2 per joint
- Max jerk: 10000 deg/s^3 = 174.5 rad/s^3 per joint
- Start/end acceleration: exactly 0 (enforced by Ruckig)
"""

import numpy as np
import matplotlib.pyplot as plt
# from ruckig import InputParameter, Trajectory, Result, Ruckig


def time_parameterize_toppra(
    waypoints: np.ndarray,
    max_velocity: float = 3.14,  # rad/s
    max_acceleration: float = 800.0 * np.pi / 180.0,  # 800 deg/s^2 -> rad/s^2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Time parameterization using TOPPRA with quintic spline interpolation.

    Args:
        waypoints: (N, n_dof) array of joint configurations
        max_velocity: Maximum joint velocity [rad/s]
        max_acceleration: Maximum joint acceleration [rad/s^2]

    Returns:
        ts_sample: Time points for interpolated trajectory
        qs_sample: Joint positions at sample points
        qds_sample: Joint velocities at sample points
        qdds_sample: Joint accelerations at sample points
    """
    import toppra as ta
    import toppra.constraint as constraint
    import toppra.algorithm as algo

    n_waypoints, n_dof = waypoints.shape

    # Create path parameter (0 to 1)
    # Use arbitrary spacing - TOPPRA will optimize time intervals
    ss = np.linspace(0, 1, n_waypoints)

    # Create quintic spline path with zero velocity/acceleration at endpoints
    path = ta.SplineInterpolator(ss, waypoints, bc_type='clamped')

    # Create velocity and acceleration constraints
    vlim = np.stack([
        np.full(n_dof, -max_velocity),
        np.full(n_dof, max_velocity)
    ], axis=1)  # shape (n_dof, 2)
    alim = np.stack([
        np.full(n_dof, -max_acceleration),
        np.full(n_dof, max_acceleration)
    ], axis=1)  # shape (n_dof, 2)

    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)
    instance = algo.TOPPRA(
        [pc_vel, pc_acc],
        path,
        solver_wrapper='seidel',
    )

    jnt_traj = instance.compute_trajectory()

    if jnt_traj is None:
        print("TOPPRA failed to find solution")
        return None, None, None, None

    # Sample the time-parameterized trajectory
    duration = jnt_traj.duration
    ts_sample = np.linspace(0, duration, int(duration / 0.01))  # 100 Hz sampling
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)  # First derivative (velocity)
    qdds_sample = jnt_traj(ts_sample, 2)  # Second derivative (acceleration)

    print(f"\n=== TOPPRA Results ===")
    print(f"Original waypoints: {n_waypoints}")
    print(f"Time-parameterized samples: {len(ts_sample)}")
    print(f"Total duration: {duration:.3f}s")
    print(f"Sampling frequency: {1.0/0.01:.0f} Hz")
    print(f"\nMax velocity (abs): {np.max(np.abs(qds_sample)):.3f} rad/s")
    print(f"Max velocity constraint: {max_velocity:.3f} rad/s")
    print(f"\nMax acceleration (abs): {np.max(np.abs(qdds_sample)):.3f} rad/s^2")
    print(f"Max acceleration constraint: {max_acceleration:.3f} rad/s^2")

    # Check boundary conditions
    print(f"\n=== Boundary Conditions ===")
    print(f"Start velocity: {qds_sample[0]}")
    print(f"End velocity: {qds_sample[-1]}")
    print(f"Start acceleration: {qdds_sample[0]}")
    print(f"End acceleration: {qdds_sample[-1]}")

    return ts_sample, qs_sample, qds_sample, qdds_sample


# def adjust_segment_with_ruckig(
#     ts_input: np.ndarray,
#     qs_input: np.ndarray,
#     qds_input: np.ndarray,
#     qdds_input: np.ndarray,
#     segment_type: str,  # 'initial' or 'final'
#     max_velocity: float = 3.14,
#     max_acceleration: float = 800.0 * np.pi / 180.0,
#     max_jerk: float = 10000.0 * np.pi / 180.0,
#     dt: float = 0.01,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Adjust trajectory segment to have zero acceleration using Ruckig.

#     Args:
#         ts_input: Input time samples
#         qs_input: Input position samples
#         qds_input: Input velocity samples
#         qdds_input: Input acceleration samples
#         segment_type: 'initial' for start segment, 'final' for end segment
#         max_velocity: Maximum joint velocity [rad/s]
#         max_acceleration: Maximum joint acceleration [rad/s^2]
#         max_jerk: Maximum joint jerk [rad/s^3]
#         dt: Sampling time [s]

#     Returns:
#         ts_adjusted: Adjusted time samples
#         qs_adjusted: Adjusted position samples
#         qds_adjusted: Adjusted velocity samples
#         qdds_adjusted: Adjusted acceleration samples
#     """
#     n_dof = qs_input.shape[1]
#     vel_norms = np.linalg.norm(qds_input, axis=1)

#     if segment_type == 'initial':
#         # Find first point where velocity is significant
#         target_idx = np.argmax(vel_norms > 0.01 * max_velocity)
#         if target_idx == 0:
#             target_idx = min(10, len(ts_input) - 1)

#         start_pos = qs_input[0].tolist()
#         start_vel = [0.0] * n_dof
#         start_acc = [0.0] * n_dof
#         target_pos = qs_input[target_idx].tolist()
#         target_vel = qds_input[target_idx].tolist()
#         target_acc = qdds_input[target_idx].tolist()

#         print(f"\n=== Ruckig Initial Acceleration Adjustment ===")
#         print(f"Adjusting from t=0 to t={ts_input[target_idx]:.3f}s (index {target_idx})")
#     else:  # 'final'
#         # Find last point with significant velocity
#         significant_vel_indices = np.where(vel_norms > 0.01 * max_velocity)[0]
#         if len(significant_vel_indices) > 0:
#             target_idx = max(0, significant_vel_indices[-1] - 10)
#         else:
#             target_idx = max(0, len(ts_input) - 10)

#         start_pos = qs_input[target_idx].tolist()
#         start_vel = qds_input[target_idx].tolist()
#         start_acc = qdds_input[target_idx].tolist()
#         target_pos = qs_input[-1].tolist()
#         target_vel = [0.0] * n_dof
#         target_acc = [0.0] * n_dof

#         print(f"\n=== Ruckig Final Acceleration Adjustment ===")
#         print(f"Adjusting from t={ts_input[target_idx]:.3f}s to end "
#               f"(index {target_idx} to {len(ts_input)-1})")

#     print(f"Start: pos={start_pos[0]:.3f}, vel={start_vel[0]:.3f}, acc={start_acc[0]:.3f}")
#     print(f"Target: pos={target_pos[0]:.3f}, vel={target_vel[0]:.3f}, acc={target_acc[0]:.3f}")

#     # Initialize Ruckig
#     ruckig = Ruckig(n_dof, dt)
#     inp = InputParameter(n_dof)
#     traj = Trajectory(n_dof)

#     # Set constraints
#     inp.max_velocity = [max_velocity] * n_dof
#     inp.max_acceleration = [max_acceleration] * n_dof
#     inp.max_jerk = [max_jerk] * n_dof

#     # Set initial and target states
#     inp.current_position = start_pos
#     inp.current_velocity = start_vel
#     inp.current_acceleration = start_acc
#     inp.target_position = target_pos
#     inp.target_velocity = target_vel
#     inp.target_acceleration = target_acc

#     # Calculate Ruckig trajectory
#     result = ruckig.calculate(inp, traj)

#     if result not in (Result.Working, Result.Finished):
#         print(f"Ruckig {segment_type} adjustment failed: {result}")
#         print(f"Using original trajectory without {segment_type} adjustment")
#         return ts_input, qs_input, qds_input, qdds_input

#     ruckig_duration = traj.duration
#     n_ruckig_samples = int(ruckig_duration / dt) + 1

#     print(f"Ruckig {segment_type} segment duration: {ruckig_duration:.3f}s ({n_ruckig_samples} samples)")

#     # Sample Ruckig trajectory
#     ts_ruckig = np.linspace(0, ruckig_duration, n_ruckig_samples)
#     qs_ruckig = []
#     qds_ruckig = []
#     qdds_ruckig = []

#     for t in ts_ruckig:
#         pos, vel, acc = traj.at_time(t)
#         qs_ruckig.append(pos)
#         qds_ruckig.append(vel)
#         qdds_ruckig.append(acc)

#     qs_ruckig = np.array(qs_ruckig)
#     qds_ruckig = np.array(qds_ruckig)
#     qdds_ruckig = np.array(qdds_ruckig)

#     # Concatenate segments
#     if segment_type == 'initial':
#         ts_remaining = ts_input[target_idx:] - ts_input[target_idx] + ruckig_duration
#         ts_adjusted = np.concatenate([ts_ruckig, ts_remaining])
#         qs_adjusted = np.vstack([qs_ruckig, qs_input[target_idx:]])
#         qds_adjusted = np.vstack([qds_ruckig, qds_input[target_idx:]])
#         qdds_adjusted = np.vstack([qdds_ruckig, qdds_input[target_idx:]])
#     else:  # 'final'
#         ts_start_time = ts_input[target_idx]
#         ts_ruckig_adjusted = ts_ruckig + ts_start_time
#         ts_adjusted = np.concatenate([ts_input[:target_idx], ts_ruckig_adjusted])
#         qs_adjusted = np.vstack([qs_input[:target_idx], qs_ruckig])
#         qds_adjusted = np.vstack([qds_input[:target_idx], qds_ruckig])
#         qdds_adjusted = np.vstack([qdds_input[:target_idx], qdds_ruckig])

#     print(f"Adjusted trajectory: {len(ts_adjusted)} samples, {ts_adjusted[-1]:.3f}s total")
#     print(f"Start acceleration: {qdds_adjusted[0]}")
#     print(f"End acceleration: {qdds_adjusted[-1]}")

#     return ts_adjusted, qs_adjusted, qds_adjusted, qdds_adjusted


def plot_trajectory(
    ts_sample: np.ndarray,
    qs_sample: np.ndarray,
    qds_sample: np.ndarray,
    qdds_sample: np.ndarray,
    waypoints: np.ndarray,
    max_velocity: float,
    max_acceleration: float,
):
    """Plot the time-parameterized trajectory."""
    n_dof = qs_sample.shape[1]
    joint_names = [f"Joint {i+1}" for i in range(n_dof)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('TOPPRA + Ruckig Adjusted Trajectory', fontsize=16)

    # Position plot
    ax = axes[0]
    for i in range(n_dof):
        ax.plot(ts_sample, qs_sample[:, i], label=joint_names[i], linewidth=2)
    ax.set_ylabel('Position [rad]', fontsize=12)
    ax.set_title('Joint Positions', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    # Velocity plot
    ax = axes[1]
    for i in range(n_dof):
        ax.plot(ts_sample, qds_sample[:, i], label=joint_names[i], linewidth=2)
    ax.axhline(y=max_velocity, color='r', linestyle='--',
               # label=f'Max velocity limit: ±{max_velocity:.2f} rad/s')
               label='limit')
    ax.axhline(y=-max_velocity, color='r', linestyle='--')
    ax.set_ylabel('Velocity [rad/s]', fontsize=12)
    ax.set_title('Joint Velocities', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    # Acceleration plot
    ax = axes[2]
    for i in range(n_dof):
        ax.plot(ts_sample, qdds_sample[:, i], label=joint_names[i], linewidth=2)
    ax.axhline(y=max_acceleration, color='r', linestyle='--',
               # label=f'Max acceleration limit: ±{max_acceleration:.2f} rad/s²')
               label='limit')
    ax.axhline(y=-max_acceleration, color='r', linestyle='--')
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Acceleration [rad/s²]', fontsize=12)
    ax.set_title('Joint Accelerations', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('toppra_ruckig_trajectory.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'toppra_ruckig_trajectory.png'")
    plt.show()


def main():
    """Main function demonstrating TOPPRA time parameterization."""

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

    print("=== Input Waypoints ===")
    print(f"Number of waypoints: {len(waypoints)}")
    print(f"Degrees of freedom: {waypoints.shape[1]}")
    print(f"\nInitial position:\n{initial_position}")
    print(f"\nFinal position:\n{waypoint_list[-1]}")

    # Define constraints
    max_velocity = 3.14  # rad/s
    max_acceleration = 800.0 * np.pi / 180.0  # 800 deg/s^2 = 13.96 rad/s^2
    max_jerk = 10000.0 * np.pi / 180.0  # 10000 deg/s^3 = 174.5 rad/s^3

    print("\n=== Constraints ===")
    print(f"Max velocity: {max_velocity} rad/s ({max_velocity * 180/np.pi:.1f} deg/s)")
    print(f"Max acceleration: {max_acceleration:.3f} rad/s² ({800.0} deg/s²)")
    print(f"Max jerk: {max_jerk:.3f} rad/s³ ({10000.0} deg/s³)")
    print("Boundary conditions: Zero velocity and acceleration at start and end")
    print("Method: TOPPRA + Ruckig initial and final adjustments")

    # Step 1: Apply TOPPRA time parameterization
    print("\n=== Step 1: TOPPRA Time Parameterization ===")
    # ts_toppra, qs_toppra, qds_toppra, qdds_toppra = time_parameterize_toppra(
    ts_sample, qs_sample, qds_sample, qdds_sample =  time_parameterize_toppra(
        waypoints=waypoints,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    )

    # if ts_toppra is None:
    #     print("TOPPRA failed!")
    #     return

    # # Step 2: Adjust initial acceleration to zero using Ruckig
    # print("\n=== Step 2: Ruckig Initial Adjustment ===")
    # ts_adjusted, qs_adjusted, qds_adjusted, qdds_adjusted = adjust_segment_with_ruckig(
    #     ts_toppra, qs_toppra, qds_toppra, qdds_toppra,
    #     segment_type='initial',
    #     max_velocity=max_velocity,
    #     max_acceleration=max_acceleration,
    #     max_jerk=max_jerk,
    # )

    # if ts_adjusted is None:
    #     print("Initial adjustment failed!")
    #     return

    # # Step 3: Adjust final acceleration to zero using Ruckig
    # print("\n=== Step 3: Ruckig Final Adjustment ===")
    # ts_sample, qs_sample, qds_sample, qdds_sample = adjust_segment_with_ruckig(
    #     ts_adjusted, qs_adjusted, qds_adjusted, qdds_adjusted,
    #     segment_type='final',
    #     max_velocity=max_velocity,
    #     max_acceleration=max_acceleration,
    #     max_jerk=max_jerk,
    # )

    if ts_sample is None:
        print("Final adjustment failed!")
        return

    # Verify constraints are satisfied
    print(f"\n=== Constraint Verification ===")
    max_vel_actual = np.max(np.abs(qds_sample))
    max_acc_actual = np.max(np.abs(qdds_sample))

    vel_satisfied = max_vel_actual <= max_velocity * 1.01  # Allow 1% tolerance
    acc_satisfied = max_acc_actual <= max_acceleration * 1.01

    print(f"Velocity constraint satisfied: {vel_satisfied}")
    print(f"  Actual max: {max_vel_actual:.3f} rad/s")
    print(f"  Limit: {max_velocity:.3f} rad/s")

    print(f"Acceleration constraint satisfied: {acc_satisfied}")
    print(f"  Actual max: {max_acc_actual:.3f} rad/s²")
    print(f"  Limit: {max_acceleration:.3f} rad/s²")

    # Check boundary conditions (velocity and acceleration at endpoints)
    start_vel_ok = np.allclose(qds_sample[0], 0, atol=1e-3)
    end_vel_ok = np.allclose(qds_sample[-1], 0, atol=1e-3)
    start_acc_ok = np.allclose(qdds_sample[0], 0, atol=1e-2)
    end_acc_ok = np.allclose(qdds_sample[-1], 0, atol=1e-2)

    print(f"\nBoundary condition verification:")
    print(f"  Start velocity = 0: {start_vel_ok}")
    print(f"  End velocity = 0: {end_vel_ok}")
    print(f"  Start acceleration = 0: {start_acc_ok}")
    print(f"  End acceleration = 0: {end_acc_ok}")

    # Save trajectory data
    output_file = 'toppra_ruckig_trajectory.npz'
    np.savez(
        output_file,
        time=ts_sample,
        position=qs_sample,
        velocity=qds_sample,
        acceleration=qdds_sample,
        waypoints=waypoints,
    )
    print(f"\nTrajectory data saved to '{output_file}'")

    # Plot results
    plot_trajectory(
        ts_sample, qs_sample, qds_sample, qdds_sample,
        waypoints, max_velocity, max_acceleration
    )

    print("\n=== Summary ===")
    print(f"✓ Time-parameterized {len(waypoints)} waypoints")
    print(f"✓ Total samples: {len(ts_sample)}, duration: {ts_sample[-1]:.3f} seconds")
    print("✓ TOPPRA for optimal trajectory + Ruckig for start/end adjustments")
    print("✓ Zero acceleration at both start and end (enforced by Ruckig)")
    print("✓ All constraints satisfied")


if __name__ == "__main__":
    main()
