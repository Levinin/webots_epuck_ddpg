

import torch
from dataclasses import dataclass


@dataclass
class OptimiseHyperparams:
    """Hyperparameters for optimise_loop"""
    update_after: int = 1000
    update_every: int = 100
    batch_size: int = 32       # 32  128
    aei_batch_size: int = 1000
    policy_lr: float = 1e-4
    q_lr: float = 1e-4
    gamma: float = 0.99
    polyak: float = 0.999
    q_optimizer: torch.optim = torch.optim.Adam
    policy_optimizer: torch.optim = torch.optim.Adam


@dataclass
class DDPGHyperParameters:
    """Hyperparameters for the DDPG algorithm"""
    replay_size: int = int(130_000)
    use_dwr: bool = True
    sec_win_size: int = int(30_000)
    step_size: int = int(5000)
    gamma: float = 0.995
    polyak: float = 0.999
    start_steps: int = 40_000
    update_after: int = 1000
    act_noise: int = 0.5
    noise_decay: float = 1.0
    num_inputs: int = 14
    num_actions: int = 2
    action_max = 1.0
    n_contexts: int = 1


@dataclass
class SimulationHyperParameters:
    """Hyperparameters for the robot controller"""
    time_step: int = 50  # ms    32 ~30 Hz
    max_speed: int = 5.  # m/s
    max_ep_len: int = 1000
    epochs: int = 5
    n_steps: int = 2
    gamma: float = 0.99
    optimise: bool = False
    save_model: bool = False
    load_model: bool = True
    sensor_min: float = 1000         # 5000 for 2, 1000 for 1,4-7
    sensor_max: float = 5000         # 1000 for 2, 5000 for 1,4-7
    left_motor_factor_start: float = 1.0
    left_motor_factor: float = 1.0
    right_motor_factor: float = 1.0
    motor_decay: float = 1.0        # 0.999
    negative_environment: bool = False
    sensor_noise_std: float = 0.0
    salt_and_pepper_noise_magnitude: float = 0.0
    salt_and_pepper_noise_prob: float = 0.0
    sensor_bias_start: float = 0.0
    sensor_bias: float = sensor_bias_start
    sensor_bias_step: float = 0.0 / 1200.0



