import json
from enum import Enum, auto, unique

import numpy as np
import numpy.typing as npt
from poke_env.battle import (
    Effect,
    Field,
    MoveCategory,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)


@unique
class LearningStyle(Enum):
    EXPLOITER = auto()
    PURE_SELF_PLAY = auto()
    LAST_SELF = auto()
    FICTITIOUS_PLAY = auto()
    DOUBLE_ORACLE = auto()

    @property
    def is_self_play(self) -> bool:
        return self in {
            LearningStyle.PURE_SELF_PLAY,
            LearningStyle.LAST_SELF,
            LearningStyle.FICTITIOUS_PLAY,
            LearningStyle.DOUBLE_ORACLE,
        }

    @property
    def abbrev(self) -> str:
        match self:
            case LearningStyle.EXPLOITER:
                return "ex"
            case LearningStyle.PURE_SELF_PLAY:
                return "sp"
            case LearningStyle.LAST_SELF:
                return "ls"
            case LearningStyle.FICTITIOUS_PLAY:
                return "fp"
            case LearningStyle.DOUBLE_ORACLE:
                return "do"


# training params
battle_format = "gen9vgc2025regi"
num_envs = 24
steps = 98_304
allow_mirror_match = True
chooses_on_teampreview = True

# observation length constants
act_len = 107
glob_obs_len = 2 * act_len + len(Field) + len(Weather) + 3
side_obs_len = len(SideCondition) + 5
move_obs_len = len(MoveCategory) + len(Target) + len(PokemonType) + 11
pokemon_obs_len = (
    4 * move_obs_len + len(Effect) + len(PokemonGender) + 2 * len(PokemonType) + len(Status) + 39
)
chunk_obs_len = glob_obs_len + side_obs_len + pokemon_obs_len

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
