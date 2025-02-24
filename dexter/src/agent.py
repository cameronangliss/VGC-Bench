from typing import Any, Deque

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from poke_env.environment import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from poke_env.player import BattleOrder, DoublesEnv, Player, SinglesEnv
from poke_env.player.env import _EnvPlayer
from src.policy import MaskedActorCriticPolicy
from src.utils import (
    abilities,
    doubles_act_len,
    doubles_chunk_obs_len,
    items,
    move_obs_len,
    moves,
    pokemon_obs_len,
    singles_act_len,
    singles_chunk_obs_len,
)
from stable_baselines3.common.policies import ActorCriticPolicy


class Agent(Player):
    __policy: ActorCriticPolicy
    frames: Deque[npt.NDArray[np.float32]]
    __teampreview_draft: list[str]

    def __init__(
        self, policy: ActorCriticPolicy | None, num_frames: int, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        if policy is not None:
            self.__policy = policy
        elif self.format_is_doubles:
            self.__policy = MaskedActorCriticPolicy(
                observation_space=Box(
                    -1, len(moves), shape=(12, doubles_chunk_obs_len), dtype=np.float32
                ),
                action_space=MultiDiscrete([doubles_act_len, doubles_act_len]),
                lr_schedule=lambda _: 1e-5,
                num_frames=num_frames,
            )
        else:
            self.__policy = MaskedActorCriticPolicy(
                observation_space=Box(
                    -1, len(moves), shape=(12, singles_chunk_obs_len), dtype=np.float32
                ),
                action_space=Discrete(singles_act_len),
                lr_schedule=lambda _: 1e-5,
                num_frames=num_frames,
            )
        self.frames = Deque(maxlen=num_frames)
        self.__teampreview_draft = []

    def set_policy(self, policy: ActorCriticPolicy):
        self.__policy = policy.to(self.__policy.device)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self.frames.appendleft(
            self.embed_battle(battle, self.__teampreview_draft, fake_ratings=True)
        )
        obs = np.stack(self.frames)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.__policy.device).unsqueeze(0)
            action, _, _ = self.__policy.forward(obs_tensor)
        if isinstance(battle, Battle):
            return SinglesEnv.action_to_order(action.cpu().numpy()[0], battle)
        elif isinstance(battle, DoubleBattle):
            return DoublesEnv.action_to_order(action.cpu().numpy()[0], battle)
        else:
            raise TypeError()

    def teampreview(self, battle: AbstractBattle) -> str:
        if isinstance(battle, Battle):
            return self.random_teampreview(battle)
        elif isinstance(battle, DoubleBattle):
            assert self.frames.maxlen is not None
            for _ in range(self.frames.maxlen):
                self.frames.appendleft(np.zeros([12, doubles_chunk_obs_len], dtype=np.float32))
            order1 = self.choose_move(battle)
            upd_battle = _EnvPlayer._simulate_teampreview_switchin(order1, battle)
            order2 = self.choose_move(upd_battle)
            action1 = DoublesEnv.order_to_action(order1, battle)
            action2 = DoublesEnv.order_to_action(order2, upd_battle)
            return f"/team {action1[0]}{action1[1]}{action2[0]}{action2[1]}"
        else:
            raise TypeError()

    @staticmethod
    def embed_battle(
        battle: AbstractBattle, teampreview_draft: list[str], fake_ratings: bool = False
    ) -> npt.NDArray[np.float32]:
        glob = Agent.embed_global(battle, teampreview_draft)
        side = Agent.embed_side(battle, fake_ratings)
        opp_side = Agent.embed_side(battle, fake_ratings, opp=True)
        [a1, a2, *_] = (
            [battle.active_pokemon] if isinstance(battle, Battle) else battle.active_pokemon
        )
        [o1, o2, *_] = (
            [battle.opponent_active_pokemon]
            if isinstance(battle, Battle)
            else battle.opponent_active_pokemon
        )
        pokemons = [
            Agent.embed_pokemon(
                p,
                i,
                from_opponent=False,
                active_a=a1 is not None and p.name == a1.name,
                active_b=a2 is not None and p.name == a2.name,
            )
            for i, p in enumerate(battle.team.values())
        ]
        pokemons += [np.zeros(pokemon_obs_len, dtype=np.float32)] * (6 - len(pokemons))
        opp_pokemons = [
            Agent.embed_pokemon(
                p,
                i,
                from_opponent=True,
                active_a=o1 is not None and p.name == o1.name,
                active_b=o2 is not None and p.name == o2.name,
            )
            for i, p in enumerate(battle.opponent_team.values())
        ]
        opp_pokemons += [np.zeros(pokemon_obs_len, dtype=np.float32)] * (6 - len(opp_pokemons))
        return np.stack(
            [np.concatenate([glob, side, p]) for p in pokemons]
            + [np.concatenate([glob, opp_side, p]) for p in opp_pokemons],
            dtype=np.float32,
        )

    @staticmethod
    def embed_global(
        battle: AbstractBattle, teampreview_draft: list[str]
    ) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            if not battle._last_request:
                mask = np.zeros(singles_act_len, dtype=np.float32)
            else:
                action_space = Agent.get_action_space(battle)
                mask = [float(i not in action_space) for i in range(singles_act_len)]
            force_switch = [float(battle.force_switch), 0]
        elif isinstance(battle, DoubleBattle):
            if not battle._last_request:
                mask = np.zeros(2 * doubles_act_len, dtype=np.float32)
            else:
                action_space1 = Agent.get_action_space(battle, 0)
                mask1 = [float(i not in action_space1) for i in range(doubles_act_len)]
                action_space2 = Agent.get_action_space(battle, 1)
                mask2 = [float(i not in action_space2) for i in range(doubles_act_len)]
                mask = mask1 + mask2
            force_switch = [float(f) for f in battle.force_switch]
        else:
            raise TypeError()
        weather = [
            min(battle.turn - battle.weather[w], 8) / 8 if w in battle.weather else 0
            for w in Weather
        ]
        fields = [
            min(battle.turn - battle.fields[f], 8) / 8 if f in battle.fields else 0 for f in Field
        ]
        if battle.teampreview:
            if not battle.active_pokemon:
                teampreview_draft = []
            else:
                teampreview_draft = teampreview_draft[:2]
        draft_positions = [float(p.name in teampreview_draft) for p in battle.team.values()]
        return np.concatenate(
            [mask, weather, fields, draft_positions, force_switch], dtype=np.float32
        )

    @staticmethod
    def embed_side(
        battle: AbstractBattle, fake_ratings: bool, opp: bool = False
    ) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            gims = [
                battle.can_mega_evolve,
                battle.can_z_move,
                battle.can_dynamax,
                battle.can_tera is not False,
            ]
            opp_gims = [
                battle.opponent_can_mega_evolve,
                battle.opponent_can_z_move,
                battle.opponent_can_dynamax,
                battle._opponent_can_terrastallize,
            ]
        elif isinstance(battle, DoubleBattle):
            gims = [
                battle.can_mega_evolve[0],
                battle.can_z_move[0],
                battle.can_dynamax[0],
                battle.can_tera[0] is not False,
            ]
            opp_gims = [
                battle.opponent_can_mega_evolve[0],
                battle.opponent_can_z_move[0],
                battle.opponent_can_dynamax[0],
                battle._opponent_can_terrastallize,
            ]
        else:
            raise TypeError()
        side_conds = battle.opponent_side_conditions if opp else battle.side_conditions
        side_conditions = [
            (
                0
                if s not in side_conds
                else (
                    1
                    if s == SideCondition.STEALTH_ROCK
                    else (
                        side_conds[s] / 2
                        if s == SideCondition.TOXIC_SPIKES
                        else (
                            side_conds[s] / 3
                            if s == SideCondition.SPIKES
                            else min(battle.turn - side_conds[s], 8) / 8
                        )
                    )
                )
            )
            for s in SideCondition
        ]
        gims = opp_gims if opp else gims
        gimmicks = [float(g) for g in gims]
        rat = battle.opponent_rating if opp else battle.rating
        rating = 1 if fake_ratings else (rat or 0) / 2000
        return np.array([*side_conditions, *gimmicks, rating], dtype=np.float32)

    @staticmethod
    def embed_pokemon(
        pokemon: Pokemon, pos: int, from_opponent: bool, active_a: bool, active_b: bool
    ) -> npt.NDArray[np.float32]:
        # (mostly) stable fields
        ability_id = abilities.index("null" if pokemon.ability is None else pokemon.ability)
        item_id = items.index("null" if pokemon.item is None else pokemon.item)
        move_ids = [
            moves.index("hiddenpower" if move.id.startswith("hiddenpower") else move.id)
            for move in pokemon.moves.values()
        ]
        move_ids += [0] * (4 - len(move_ids))
        move_details = [Agent.embed_move(m) for m in pokemon.moves.values()]
        move_details = np.concatenate(
            [*move_details, np.zeros(move_obs_len * (4 - len(move_details)))]
        )
        types = [float(t in pokemon.types) for t in PokemonType]
        tera_type = [float(t == pokemon.tera_type) for t in PokemonType]
        stats = [(s or 0) / 1000 for s in pokemon.stats.values()]
        gender = [float(g == pokemon.gender) for g in PokemonGender]
        weight = pokemon.weight / 1000
        # volatile fields
        hp_frac = pokemon.current_hp_fraction
        status = [float(s == pokemon.status) for s in Status]
        status_counter = pokemon.status_counter / 16
        boosts = [b / 6 for b in pokemon.boosts.values()]
        effects = [(min(pokemon.effects[e], 8) / 8 if e in pokemon.effects else 0) for e in Effect]
        first_turn = float(pokemon.first_turn)
        protect_counter = pokemon.protect_counter / 5
        must_recharge = float(pokemon.must_recharge)
        preparing = float(pokemon.preparing)
        gimmicks = [float(s) for s in [pokemon.is_dynamaxed, pokemon.is_terastallized]]
        pos_onehot = [float(pos == i) for i in range(6)]
        return np.array(
            [
                ability_id,
                item_id,
                *move_ids,
                *move_details,
                *types,
                *tera_type,
                *stats,
                *gender,
                weight,
                hp_frac,
                *status,
                status_counter,
                *boosts,
                *effects,
                first_turn,
                protect_counter,
                must_recharge,
                preparing,
                *gimmicks,
                float(active_a),
                float(active_b),
                *pos_onehot,
                float(from_opponent),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def embed_move(move: Move) -> npt.NDArray[np.float32]:
        power = move.base_power / 250
        acc = move.accuracy / 100
        category = [float(c == move.category) for c in MoveCategory]
        target = [float(t == move.target) for t in Target]
        priority = (move.priority + 7) / 12
        crit_ratio = move.crit_ratio
        drain = move.drain
        force_switch = float(move.force_switch)
        recoil = move.recoil
        self_destruct = float(move.self_destruct is not None)
        self_switch = float(move.self_switch is not False)
        pp = move.max_pp / 64
        pp_frac = move.current_pp / move.max_pp
        move_type = [float(t == move.type) for t in PokemonType]
        return np.array(
            [
                power,
                acc,
                *category,
                *target,
                priority,
                crit_ratio,
                drain,
                force_switch,
                recoil,
                self_destruct,
                self_switch,
                pp,
                pp_frac,
                *move_type,
            ]
        )

    @staticmethod
    def get_action_space(battle: AbstractBattle, pos: int | None = None) -> npt.NDArray[np.int64]:
        if isinstance(battle, Battle):
            switch_space = [
                i
                for i, pokemon in enumerate(battle.team.values())
                if not battle.maybe_trapped
                and pokemon.species in [p.species for p in battle.available_switches]
            ]
            if battle.active_pokemon is None:
                return np.array(switch_space)
            else:
                move_space = [
                    i + 6
                    for i, move in enumerate(battle.active_pokemon.moves.values())
                    if move.id in [m.id for m in battle.available_moves]
                ]
                mega_space = [i + 4 for i in move_space if battle.can_mega_evolve]
                zmove_space = [
                    i + 8
                    for i, move in enumerate(battle.active_pokemon.moves.values())
                    if move.id in [m.id for m in battle.active_pokemon.available_z_moves]
                    and battle.can_z_move
                ]
                dynamax_space = [i + 12 for i in move_space if battle.can_dynamax]
                tera_space = [i + 16 for i in move_space if battle.can_tera]
                return np.array(
                    switch_space
                    + move_space
                    + mega_space
                    + zmove_space
                    + dynamax_space
                    + tera_space
                )
        elif isinstance(battle, DoubleBattle):
            assert pos is not None
            switch_space = [
                i + 1
                for i, pokemon in enumerate(battle.team.values())
                if battle.force_switch != [[False, True], [True, False]][pos]
                and not battle.trapped[pos]
                and not (
                    len(battle.available_switches[0]) == 1
                    and battle.force_switch == [True, True]
                    and pos == 1
                )
                and not pokemon.active
                and pokemon.species in [p.species for p in battle.available_switches[pos]]
            ]
            active_mon = battle.active_pokemon[pos]
            if battle.teampreview:
                return np.array(switch_space)
            elif battle.finished or battle._wait:
                return np.array([0])
            elif active_mon is None or battle.force_switch[pos]:
                return np.array(switch_space or [0])
            else:
                move_spaces = [
                    [
                        7 + 5 * i + j + 2
                        for j in battle.get_possible_showdown_targets(move, active_mon)
                    ]
                    for i, move in enumerate(active_mon.moves.values())
                    if move.id in [m.id for m in battle.available_moves[pos]]
                ]
                move_space = [i for s in move_spaces for i in s]
                tera_space = [i + 80 for i in move_space if battle.can_tera[pos]]
                if (
                    not move_space
                    and len(battle.available_moves[pos]) == 1
                    and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                ):
                    move_space = [9]
                return np.array((switch_space + move_space + tera_space) or [0])
        else:
            raise TypeError()
