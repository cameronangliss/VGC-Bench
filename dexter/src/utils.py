import json

import numpy as np
import numpy.typing as npt
from poke_env.environment import (
    Effect,
    Field,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)

# training params
battle_format = "gen9vgc2024regh"
device = "cuda:3"
num_envs = 32
num_frames = 3
num_teams = 16
port = 8000
self_play = True
steps = 98_304

teams = list(range(num_teams))
run_name = ",".join([str(t) for t in teams])

# observation length constants
singles_act_len = 26
doubles_act_len = 107
singles_glob_obs_len = singles_act_len + len(Field) + len(Weather) + 7
doubles_glob_obs_len = 2 * doubles_act_len + len(Field) + len(Weather) + 8
side_obs_len = len(SideCondition) + 5
pokemon_obs_len = len(Effect) + len(PokemonGender) + 2 * len(PokemonType) + len(Status) + 41
singles_chunk_obs_len = singles_glob_obs_len + side_obs_len + pokemon_obs_len
doubles_chunk_obs_len = doubles_glob_obs_len + side_obs_len + pokemon_obs_len

# pokemon data
with open("data/abilities.json") as f:
    ability_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    abilities = list(ability_descs.keys())
    ability_embeds = list(ability_descs.values())
with open("data/items.json") as f:
    item_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    items = list(item_descs.keys())
    item_embeds = list(item_descs.values())
with open("data/moves.json") as f:
    move_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    moves = list(move_descs.keys())
    move_embeds = list(move_descs.values())
