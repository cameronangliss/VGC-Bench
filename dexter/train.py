import os
import time
from subprocess import PIPE, STDOUT, Popen

from src.callback import Callback
from src.env import ShowdownEnv
from src.policy import MaskedActorCriticPolicy
from src.utils import (
    battle_format,
    device,
    num_envs,
    num_frames,
    port,
    run_name,
    self_play,
    steps,
    teams,
)
from stable_baselines3 import PPO


def train():
    server = Popen(
        ["node", "pokemon-showdown", "start", str(port), "--no-security"],
        stdout=PIPE,
        stderr=STDOUT,
        cwd="pokemon-showdown",
    )
    time.sleep(10)
    env = ShowdownEnv.create_env(num_envs, battle_format, num_frames, port, teams, self_play)
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=1e-5,
        n_steps=2048 // num_envs,
        batch_size=64,
        gamma=1,
        ent_coef=0.02,
        tensorboard_log="logs",
        policy_kwargs={"num_frames": num_frames},
        device=device,
    )
    if os.path.exists(f"saves/{run_name}") and len(os.listdir(f"saves/{run_name}")) > 0:
        num_saved_timesteps = max([int(file[:-4]) for file in os.listdir(f"saves/{run_name}")])
        ppo.num_timesteps = num_saved_timesteps
        ppo.set_parameters(f"saves/{run_name}/{num_saved_timesteps}.zip", device=ppo.device)
    # ppo.policy.actor_grad = num_saved_timesteps > 0  # type: ignore
    callback = Callback(steps, battle_format, num_frames, teams, port, self_play)
    ppo.learn(steps, callback=callback, tb_log_name=run_name, reset_num_timesteps=False)
    server.terminate()
    server.wait()


if __name__ == "__main__":
    train()
