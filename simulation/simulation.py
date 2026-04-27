# Description: This script is used to simulate the full model of the robot in mujoco
import pathlib

# Authors:
# Giulio Turrisi, Daniel Ordonez
import time
from os import PathLike
from pprint import pprint

import copy
import numpy as np

import mujoco

# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

# Helper functions for plotting
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco

# PyMPC controller imports
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper


class WaypointNavigator:
    """Rotate-then-advance waypoint follower with adaptive speed control.

    Rather than a binary "go / stop" supervisor, the commanded forward speed is
    modulated continuously by three factors in [0, 1]:

      * f_dist  : decelerate near the waypoint (smooth ramp).
      * f_yaw   : slow down when heading is misaligned.
      * f_stab  : slow down when pitch / pitch_rate / roll grow toward instability.

    The internal commanded speed follows the (modulated) target through a slew
    limit (acceleration / deceleration caps) so the MPC never sees velocity
    steps. This avoids the forward pitch-and-tip failure mode caused by
    constant-speed commands.
    """

    ROTATING = "ROTATING"
    ADVANCING = "ADVANCING"
    ARRIVED = "ARRIVED"

    # Slew limits (m/s^2). Decel >> accel to brake hard, ramp up gently.
    ACC_LIMIT = 0.3
    DEC_LIMIT = 2.0

    # Continuous speed-governor thresholds.
    PITCH_ENTER = 0.22   # rad (~12.6 deg)
    PITCH_EXIT  = 0.06   # rad (~3.4 deg)
    ROLL_ENTER  = 0.25
    ROLL_EXIT   = 0.08
    WRATE_ENTER = 2.0    # rad/s
    WRATE_EXIT  = 0.6

    # Predictive emergency brake: pitch projected this far ahead must stay
    # below PITCH_HARD_LIMIT, otherwise speed target is forced to 0.
    PREDICT_HORIZON   = 0.30   # s
    PITCH_HARD_LIMIT  = 0.18   # rad
    WY_HARD_LIMIT     = 1.6    # rad/s instantaneous

    # Adaptive speed cap: shrinks on each near-fall event, slowly recovers.
    SPEED_CAP_ON_TRIGGER = 0.5    # multiply current cap by this on near-fall
    SPEED_CAP_FLOOR      = 0.25
    SPEED_CAP_RECOVERY   = 0.04   # per second when stable
    TRIGGER_COOLDOWN     = 1.2    # s, ignore further triggers during cooldown

    DECEL_ZONE = 0.6
    YAW_FULL_BRAKE = 0.6

    def __init__(self, waypoints, arrival_tol, yaw_tol, lin_vel, ang_vel):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.arrival_tol = float(arrival_tol)
        self.yaw_tol = float(yaw_tol)
        self.lin_vel = float(lin_vel)
        self.ang_vel = float(ang_vel)
        self.idx = 0
        self.state = self.ROTATING if self.waypoints else self.ARRIVED
        self.cmd_speed = 0.0  # slewed forward speed (m/s)
        self.cmd_wz = 0.0     # slewed yaw rate (rad/s)
        self._dt = 0.002      # default sim_dt; updated per call
        self.speed_cap = 1.0  # adaptive multiplier on lin_vel; learns from near-falls
        self.cooldown_t = 0.0 # s remaining of trigger cooldown
        self.brake_t = 0.0    # s remaining of forced full-stop after a trigger

    def _wrap(self, a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _ramp(self, x, lo, hi):
        """Return 1 when |x|<lo, 0 when |x|>hi, linear in between."""
        ax = abs(x)
        if ax <= lo:
            return 1.0
        if ax >= hi:
            return 0.0
        return (hi - ax) / (hi - lo)

    def _stability_factor(self, roll, pitch, wx, wy):
        f_p  = self._ramp(pitch, self.PITCH_EXIT, self.PITCH_ENTER)
        f_r  = self._ramp(roll,  self.ROLL_EXIT,  self.ROLL_ENTER)
        f_wx = self._ramp(wx, self.WRATE_EXIT, self.WRATE_ENTER)
        f_wy = self._ramp(wy, self.WRATE_EXIT, self.WRATE_ENTER)
        return min(f_p, f_r, f_wx, f_wy)

    def _slew(self, current, target, acc, dec, dt):
        delta = target - current
        if delta >= 0:
            return current + min(delta, acc * dt)
        return current + max(delta, -dec * dt)

    def compute_ref(self, base_pos_xy, yaw, roll=0.0, pitch=0.0,
                    wx=0.0, wy=0.0, dt=0.002):
        """Return (ref_lin_vel_world(3,), ref_ang_vel_world_z scalar)."""
        self._dt = float(dt)

        if self.state == self.ARRIVED or self.idx >= len(self.waypoints):
            self.cmd_speed = self._slew(self.cmd_speed, 0.0, self.ACC_LIMIT, self.DEC_LIMIT, self._dt)
            self.cmd_wz    = self._slew(self.cmd_wz,    0.0, 4.0, 4.0, self._dt)
            return np.zeros(3), 0.0

        target = self.waypoints[self.idx]
        delta = target - np.asarray(base_pos_xy, dtype=float)
        dist = float(np.linalg.norm(delta))

        if dist < self.arrival_tol:
            self.idx += 1
            if self.idx >= len(self.waypoints):
                self.state = self.ARRIVED
            else:
                self.state = self.ROTATING  # re-align before next leg
            # Brake immediately on arrival.
            self.cmd_speed = self._slew(self.cmd_speed, 0.0, self.ACC_LIMIT, self.DEC_LIMIT, self._dt)
            self.cmd_wz    = self._slew(self.cmd_wz,    0.0, 4.0, 4.0, self._dt)
            return np.zeros(3), 0.0

        desired_yaw = np.arctan2(delta[1], delta[0])
        yaw_err = self._wrap(desired_yaw - yaw)
        f_stab = self._stability_factor(roll, pitch, wx, wy)

        # --- Predictive emergency brake ---
        # Project pitch forward; if it crosses the hard limit, fire a trigger
        # event: full brake for 1 s and shrink the adaptive speed cap.
        pred_pitch = pitch + wy * self.PREDICT_HORIZON
        triggered = (
            abs(pred_pitch) > self.PITCH_HARD_LIMIT
            or abs(wy) > self.WY_HARD_LIMIT
            or abs(pitch) > self.PITCH_HARD_LIMIT
        )
        # Update timers.
        self.cooldown_t = max(0.0, self.cooldown_t - self._dt)
        self.brake_t    = max(0.0, self.brake_t    - self._dt)

        if triggered and self.cooldown_t == 0.0:
            self.speed_cap = max(
                self.SPEED_CAP_FLOOR, self.speed_cap * self.SPEED_CAP_ON_TRIGGER
            )
            self.brake_t    = 1.0
            self.cooldown_t = self.TRIGGER_COOLDOWN

        # Recover speed cap slowly when fully stable (no trigger active, low tilt).
        if self.brake_t == 0.0 and abs(pitch) < self.PITCH_EXIT and abs(wy) < self.WRATE_EXIT:
            self.speed_cap = min(1.0, self.speed_cap + self.SPEED_CAP_RECOVERY * self._dt)

        # --- Yaw-rate command (always active, modulated by stability) ---
        target_wz = np.clip(1.2 * yaw_err, -self.ang_vel, self.ang_vel) * f_stab
        if self.brake_t > 0.0:
            target_wz = 0.0
        self.cmd_wz = self._slew(self.cmd_wz, target_wz, 3.0, 6.0, self._dt)

        # --- Forward-speed target ---
        if self.brake_t > 0.0:
            target_speed = 0.0
        elif self.state == self.ROTATING:
            if abs(yaw_err) < self.yaw_tol:
                self.state = self.ADVANCING
            target_speed = 0.0
        else:
            f_dist = float(np.clip(dist / self.DECEL_ZONE, 0.0, 1.0))
            f_yaw  = float(np.clip(1.0 - abs(yaw_err) / self.YAW_FULL_BRAKE, 0.0, 1.0))
            target_speed = self.lin_vel * self.speed_cap * f_dist * f_yaw * f_stab

            if abs(yaw_err) > 3.0 * self.yaw_tol:
                self.state = self.ROTATING
                target_speed = 0.0

        self.cmd_speed = self._slew(
            self.cmd_speed, target_speed, self.ACC_LIMIT, self.DEC_LIMIT, self._dt
        )

        vx_world = self.cmd_speed * np.cos(yaw)
        vy_world = self.cmd_speed * np.sin(yaw)
        return np.array([vx_world, vy_world, 0.0]), float(self.cmd_wz)


def run_simulation(
    qpympc_cfg,
    process=0,
    num_episodes=500,
    num_seconds_per_episode=60,
    ref_base_lin_vel=(0.0, 4.0),
    ref_base_ang_vel=(-0.4, 0.4),
    friction_coeff=(0.5, 1.0),
    base_vel_command_type="human",
    seed=0,
    render=True,
    recording_path: PathLike = None,
):
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed)

    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    robot_leg_joints = qpympc_cfg.robot_leg_joints
    robot_feet_geom_names = qpympc_cfg.robot_feet_geom_names
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    # Save all observables available.
    state_obs_names = [] #list(QuadrupedEnv.ALL_OBS)  # + list(IMU.ALL_OBS)

    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray(ref_base_lin_vel) * hip_height,  # pass a float for a fixed value
        ref_base_ang_vel=ref_base_ang_vel,  # pass a float for a fixed value
        ground_friction_coeff=friction_coeff,  # pass a float for a fixed value
        base_vel_command_type=base_vel_command_type,  # "forward", "random", "forward+rotate", "human"
        state_obs_names=tuple(state_obs_names),  # Desired quantities in the 'state' vec
    )
    pprint(env.get_hyperparameters())
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant

    # Some robots require a change in the zero joint-space configuration. If provided apply it
    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

    # Lower the default spawn height so the feet start touching the ground
    # (the default puts them ~2.4 cm in the air, causing a destabilizing drop).
    spawn_z_offset = 0.024
    env.mjModel.qpos0[2] = max(0.0, env.mjModel.qpos0[2] - spawn_z_offset)

    env.reset(random=False)

    # Build the waypoint navigator if the sim is in 'waypoints' mode.
    waypoint_nav = None
    if qpympc_cfg.simulation_params.get("mode") == "waypoints":
        wp_cfg = qpympc_cfg.simulation_params
        waypoint_nav = WaypointNavigator(
            waypoints=wp_cfg["waypoints"],
            arrival_tol=wp_cfg["waypoint_arrival_tol"],
            yaw_tol=wp_cfg["waypoint_yaw_tol"],
            lin_vel=wp_cfg["waypoint_lin_vel"],
            ang_vel=wp_cfg["waypoint_ang_vel"],
        )

    if render:
        env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    # Initialization of variables used in the main control loop --------------------------------

    # Torque vector
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    # Torque limits
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    # Feet positions and Legs order
    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]

    # Create HeightMap -----------------------------------------------------------------------
    if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
        from gym_quadruped.sensors.heightmap import HeightMap

        resolution_heightmap = 0.04
        num_rows_heightmap = 7
        num_cols_heightmap = 7
        heightmaps = LegsAttr(
            FL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, 
                         dist_x=resolution_heightmap, dist_y=resolution_heightmap, 
                         mj_model=env.mjModel, mj_data=env.mjData),
            FR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, 
                         dist_x=resolution_heightmap, dist_y=resolution_heightmap, 
                         mj_model=env.mjModel, mj_data=env.mjData),
            RL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, 
                         dist_x=resolution_heightmap, dist_y=resolution_heightmap, 
                         mj_model=env.mjModel, mj_data=env.mjData),
            RR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, 
                         dist_x=resolution_heightmap, dist_y=resolution_heightmap, 
                         mj_model=env.mjModel, mj_data=env.mjData),
        )
    else:
        heightmaps = None

    # Quadruped PyMPC controller initialization -------------------------------------------------------------
    quadrupedpympc_observables_names = (
        "ref_base_height",
        "ref_base_angles",
        "ref_feet_pos",
        "nmpc_GRFs",
        "nmpc_footholds",
        "swing_time",
        "phase_signal",
        "lift_off_positions",
        # "base_lin_vel_err",
        # "base_ang_vel_err",
        # "base_poz_z_err",
    )

    quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
        quadrupedpympc_observables_names=quadrupedpympc_observables_names,
    )

    # Data recording -------------------------------------------------------------------------------------------
    if recording_path is not None:
        from gym_quadruped.utils.data.h5py import H5Writer

        root_path = pathlib.Path(recording_path)
        root_path.mkdir(exist_ok=True)
        dataset_path = (
            root_path
            / f"{robot_name}/{scene_name}"
            / f"lin_vel={ref_base_lin_vel} ang_vel={ref_base_ang_vel} friction={friction_coeff}"
            / f"ep={num_episodes}_steps={int(num_seconds_per_episode // simulation_dt):d}.h5"
        )
        h5py_writer = H5Writer(
            file_path=dataset_path,
            env=env,
            extra_obs=None,  # TODO: Make this automatically configured. Not hardcoded
        )
        print(f"\n Recording data to: {dataset_path.absolute()}")
    else:
        h5py_writer = None

    # -----------------------------------------------------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    N_EPISODES = num_episodes
    N_STEPS_PER_EPISODE = int(num_seconds_per_episode // simulation_dt)
    last_render_time = time.time()

    state_obs_history, ctrl_state_history = [], []
    for episode_num in range(N_EPISODES):
        ep_state_history, ep_ctrl_state_history, ep_time = [], [], []
        for _ in tqdm(range(N_STEPS_PER_EPISODE), desc=f"Ep:{episode_num:d}-steps:", total=N_STEPS_PER_EPISODE):
            # Update value from SE or Simulator ----------------------
            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame='world')
            hip_pos = env.hip_positions(frame="world")
            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = copy.deepcopy(env.base_pos)
            com_pos = copy.deepcopy(env.com)

            # Get the reference base velocity in the world frame
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

            # Override with waypoint navigator (rotate-then-advance) if active.
            if waypoint_nav is not None:
                ref_lin, ref_wz = waypoint_nav.compute_ref(
                    base_pos_xy=base_pos[:2],
                    yaw=base_ori_euler_xyz[2],
                    roll=base_ori_euler_xyz[0],
                    pitch=base_ori_euler_xyz[1],
                    wx=base_ang_vel[0],
                    wy=base_ang_vel[1],
                    dt=simulation_dt,
                )
                ref_base_lin_vel = ref_lin
                ref_base_ang_vel = np.array([0.0, 0.0, ref_wz])

            # Get the inertia matrix
            if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                inertia = env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
            else:
                inertia = qpympc_cfg.inertia.flatten()

            # Get the qpos and qvel
            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            # Idx of the leg
            legs_qvel_idx = env.legs_qvel_idx  # leg_name: [idx1, idx2, idx3] ...
            legs_qpos_idx = env.legs_qpos_idx  # leg_name: [idx1, idx2, idx3] ...
            joints_pos = LegsAttr(FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR, RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR)

            # Get Centrifugal, Coriolis, Gravity, Friction for the swing controller
            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive

            # Compute feet jacobians
            feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame='world', return_rot_jac=False)

            # Quadruped PyMPC controller --------------------------------------------------------------
            tau = quadrupedpympc_wrapper.compute_actions(
                com_pos,
                base_pos,
                base_lin_vel,
                base_ori_euler_xyz,
                base_ang_vel,
                feet_pos,
                hip_pos,
                joints_pos,
                heightmaps,
                legs_order,
                simulation_dt,
                ref_base_lin_vel,
                ref_base_ang_vel,
                env.step_num,
                qpos,
                qvel,
                feet_jac,
                feet_jac_dot,
                feet_vel,
                legs_qfrc_passive,
                legs_qfrc_bias,
                legs_mass_matrix,
                legs_qpos_idx,
                legs_qvel_idx,
                tau,
                inertia,
                env.mjData.contact,
            )
            # Limit tau between tau_limits
            for leg in ["FL", "FR", "RL", "RR"]:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            # Set control and mujoco step -------------------------------------------------------------------------
            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR


            # Apply the action to the environment and evolve sim --------------------------------------------------
            state, reward, is_terminated, is_truncated, info = env.step(action=action)

            # Get Controller state observables
            ctrl_state = quadrupedpympc_wrapper.get_obs()

            # Store the history of observations and control -------------------------------------------------------
            base_poz_z_err = ctrl_state["ref_base_height"] - base_pos[2]
            ctrl_state["base_poz_z_err"] = base_poz_z_err

            ep_state_history.append(state)
            ep_time.append(env.simulation_time)
            ep_ctrl_state_history.append(ctrl_state)

            # Render only at a certain frequency -----------------------------------------------------------------
            if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

                # Plot the swing trajectory
                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=quadrupedpympc_wrapper.wb_interface.stc,
                    swing_period=quadrupedpympc_wrapper.wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=ctrl_state["swing_time"][0],
                        FR=ctrl_state["swing_time"][1],
                        RL=ctrl_state["swing_time"][2],
                        RR=ctrl_state["swing_time"][3],
                    ),
                    lift_off_positions=ctrl_state["lift_off_positions"],
                    nmpc_footholds=ctrl_state["nmpc_footholds"],
                    ref_feet_pos=ctrl_state["ref_feet_pos"],
                    early_stance_detector=quadrupedpympc_wrapper.wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )

                # Update and Plot the heightmap
                if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
                    # if(stc.check_apex_condition(current_contact, interval=0.01)):
                    for leg_id, leg_name in enumerate(legs_order):
                        data = heightmaps[
                            leg_name
                        ].data  # .update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
                        if data is not None:
                            for i in range(data.shape[0]):
                                for j in range(data.shape[1]):
                                    heightmaps[leg_name].geom_ids[i, j] = render_sphere(
                                        viewer=env.viewer,
                                        position=([data[i][j][0][0], data[i][j][0][1], data[i][j][0][2]]),
                                        diameter=0.01,
                                        color=[0, 1, 0, 0.5],
                                        geom_id=heightmaps[leg_name].geom_ids[i, j],
                                    )

                # Plot the GRF
                for leg_id, leg_name in enumerate(legs_order):
                    feet_GRF_geom_ids[leg_name] = render_vector(
                        env.viewer,
                        vector=feet_GRF[leg_name],
                        pos=feet_pos[leg_name],
                        scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=feet_GRF_geom_ids[leg_name],
                    )

                env.render()
                last_render_time = time.time()

            # Reset the environment if the episode is terminated ------------------------------------------------
            if env.step_num >= N_STEPS_PER_EPISODE or is_terminated or is_truncated:
                if is_terminated:
                    print("Environment terminated")
                else:
                    state_obs_history.append(ep_state_history)
                    ctrl_state_history.append(ep_ctrl_state_history)     

                env.reset(random=False)
                quadrupedpympc_wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))
                if waypoint_nav is not None:
                    waypoint_nav.idx = 0
                    waypoint_nav.state = WaypointNavigator.ROTATING
                    waypoint_nav.cmd_speed = 0.0
                    waypoint_nav.cmd_wz = 0.0
                    waypoint_nav.speed_cap = 1.0
                    waypoint_nav.cooldown_t = 0.0
                    waypoint_nav.brake_t = 0.0

        if h5py_writer is not None:  # Save episode trajectory data to disk.
            ep_obs_history = collate_obs(ep_state_history)  # | collate_obs(ep_ctrl_state_history)
            ep_traj_time = np.asarray(ep_time)[:, np.newaxis]
            h5py_writer.append_trajectory(state_obs_traj=ep_obs_history, time=ep_traj_time)

    env.close()
    if h5py_writer is not None:
        return h5py_writer.file_path


def collate_obs(list_of_dicts) -> dict[str, np.ndarray]:
    """Collates a list of dictionaries containing observation names and numpy arrays
    into a single dictionary of stacked numpy arrays.
    """
    if not list_of_dicts:
        raise ValueError("Input list is empty.")

    # Get all keys (assumes all dicts have the same keys)
    keys = list_of_dicts[0].keys()

    # Stack the values per key
    collated = {key: np.stack([d[key] for d in list_of_dicts], axis=0) for key in keys}
    collated = {key: v[:, None] if v.ndim == 1 else v for key, v in collated.items()}
    return collated


if __name__ == "__main__":
    from quadruped_pympc import config as cfg

    qpympc_cfg = cfg
    # Custom changes to the config here:
    pass

    # Run the simulation with the desired configuration.....
    run_simulation(qpympc_cfg=qpympc_cfg)

    # run_simulation(num_episodes=1, render=False)
