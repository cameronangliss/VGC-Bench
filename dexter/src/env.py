from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStackObservation
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import (
    DoublesEnv,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    SingleAgentWrapper,
)
from src.agent import Agent
from src.teams import RandomTeamBuilder
from src.utils import doubles_chunk_obs_len, moves
from stable_baselines3.common.monitor import Monitor


class ShowdownEnv(DoublesEnv[npt.NDArray[np.float32]]):
    _teampreview_draft: list[str]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.metadata = {"name": "showdown_v1", "render_modes": ["human"]}
        self.render_mode: str | None = None
        self.observation_spaces = {
            agent: Box(-1, len(moves), shape=(12, doubles_chunk_obs_len), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._teampreview_draft = []

    @classmethod
    def create_env(
        cls,
        i: int,
        battle_format: str,
        num_frames: int,
        port: int,
        teams: list[int],
        self_play: bool,
        device: str,
    ) -> Monitor:
        env = cls(
            account_configuration1=AccountConfiguration(f"Agent{port}-{i + 1}", None),
            account_configuration2=AccountConfiguration(f"Opponent{port}-{i + 1}", None),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format),
            start_challenging=True,
        )
        if self_play:
            num_gpus = torch.cuda.device_count()
            other_gpu_ids = [f"cuda:{i}" for i in range(num_gpus) if f"cuda:{i}" != device]
            opponent = Agent(
                None,
                num_frames=num_frames,
                device=torch.device(
                    other_gpu_ids[i % len(other_gpu_ids)] if num_gpus > 0 else "cpu"
                ),
                account_configuration=AccountConfiguration(f"Internal{i + 1}", None),
                server_configuration=ServerConfiguration(
                    f"ws://localhost:{port}/showdown/websocket",
                    "https://play.pokemonshowdown.com/action.php?",
                ),
                battle_format=battle_format,
                log_level=40,
                accept_open_team_sheet=True,
                open_timeout=None,
                team=RandomTeamBuilder(teams, battle_format),
            )
        elif "vgc" in battle_format:
            opponent = MaxBasePowerPlayer(battle_format=battle_format, log_level=40)
        else:
            opp_classes = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
            opponent = opp_classes[i % 3](battle_format=battle_format, log_level=40)
        env = SingleAgentWrapper(env, opponent)
        env = FrameStackObservation(env, num_frames)
        env = Monitor(env)
        return env

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, dict[str, Any]]]:
        result = super().reset(seed=seed, options=options)
        tags = [k for k in self.agent1.battles.keys()]
        for tag in tags:
            if self.battle1 is None or tag != self.battle1.battle_tag:
                self.agent1._battles.pop(tag)
                self.agent2._battles.pop(tag)
        return result

    def calc_reward(self, battle: AbstractBattle) -> float:
        if not battle.finished:
            return 0
        elif battle.won:
            return 1
        elif battle.lost:
            return -1
        else:
            return 0

    def embed_battle(self, battle: AbstractBattle) -> npt.NDArray[np.float32]:
        return Agent.embed_battle(battle, self._teampreview_draft, fake_ratings=True)
