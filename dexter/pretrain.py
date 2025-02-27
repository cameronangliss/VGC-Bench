import os
import pickle

import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from imitation.util.logger import configure
from poke_env.player import RandomPlayer, SingleAgentWrapper
from src.env import ShowdownEnv
from src.policy import MaskedActorCriticPolicy
from src.utils import (
    battle_format,
    behavior_clone,
    device,
    doubles_chunk_obs_len,
    num_frames,
    run_name,
)
from stable_baselines3 import PPO


def pretrain():
    env = ShowdownEnv(
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
    )
    opponent = RandomPlayer(
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
    )
    single_agent_env = SingleAgentWrapper(env, opponent)
    ppo = PPO(
        MaskedActorCriticPolicy,
        single_agent_env,
        policy_kwargs={"num_frames": num_frames},
        device=device,
    )
    with open("data/trajs.pkl", "rb") as f:
        trajs: list[Trajectory] = pickle.load(f)
    stacked_trajs = []
    for i in range(len(trajs) // 10):
        print(f"progress: {round(100 * i / len(trajs), ndigits=2)}%", end="\r")
        stacked_trajs += [frame_stack_traj(trajs[i])]
    print(f"finished loading {len(stacked_trajs)} trajectories")
    bc = BC(
        observation_space=ppo.observation_space,  # type: ignore
        action_space=ppo.action_space,  # type: ignore
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        demonstrations=stacked_trajs,
        batch_size=1024,
        device=device,
        custom_logger=configure(f"logs/{run_name}", ["tensorboard"]),
    )
    print("finished initing")
    bc.train(n_epochs=100)
    ppo.save(f"saves/{run_name}/0")


def frame_stack_traj(traj: Trajectory) -> Trajectory:
    zero_obs = np.zeros([12, doubles_chunk_obs_len])
    obs_list = [
        np.stack([traj.obs[i - j] if i - j >= 0 else zero_obs for j in range(num_frames)], axis=0)
        for i in range(len(traj.obs))
    ]
    return Trajectory(obs=np.stack(obs_list, axis=0), acts=traj.acts, infos=None, terminal=True)


if __name__ == "__main__":
    if not behavior_clone:
        print("behavior cloning toggled off - aborting")
    elif os.path.exists(f"saves/{run_name}/0.zip"):
        print("already have pretrained NN - aborting")
    else:
        pretrain()
