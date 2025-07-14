import re
from typing import Any

import numpy as np
import torch
import transformers
from poke_env.environment import AbstractBattle, DoubleBattle, Move, Pokemon
from poke_env.player import BattleOrder, DoublesEnv, Player
from src.agent import Agent
from src.utils import doubles_act_len


class LLMPlayer(Player):
    def __init__(self, device: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.__teampreview_draft = []
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", use_auth_token=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            use_auth_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        self.model = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)  # type: ignore

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        action1 = self.choose_move_individual(battle, 0, None)
        prev_action = action1 if action1 >= 0 else None
        action2 = self.choose_move_individual(battle, 1, prev_action)
        action = np.array([action1, action2])
        order = DoublesEnv.action_to_order(action, battle)
        return order

    def choose_move_individual(
        self, battle: DoubleBattle, pos: int, prev_action: int | None
    ) -> int:
        action_space = Agent.get_action_space(battle, pos)
        mask = torch.tensor(
            [float("-inf") if i not in action_space else 0 for i in range(doubles_act_len)]
        )
        last_order = None
        if pos == 1:
            assert prev_action is not None
            mask += LLMPlayer._get_mask(torch.tensor([[prev_action]]))[0]
            last_order = DoublesEnv._action_to_order_individual(
                np.int64(prev_action), battle, False, 0
            )
        action_space = [i for i, m in enumerate(mask.tolist()) if m == 0]
        if not action_space:
            return 0
        elif len(action_space) == 1:
            return action_space[0]
        order_space = [
            DoublesEnv._action_to_order_individual(np.int64(a), battle, False, pos)
            for a in action_space
        ]
        action_names = [
            self.readable_battle_order(battle, o, pos) for o in order_space if o is not None
        ]
        prompt = self.explain_battle(
            battle, self.__teampreview_draft, action_names, last_order, pos
        )
        input_dict = [
            {
                "role": "system",
                "content": f"You are an expert Pokemon VGC competitor playing a Pokemon battle in the {battle.format} format.",
            },
            {"role": "user", "content": prompt},
        ]
        response: str = self.model(input_dict)[0]["generated_text"][-1]["content"]  # type: ignore
        try:
            action_index = int(response) - 1
            action = action_space[action_index]
        except ValueError:
            print(f"FAULTY RESPONSE: {response}", flush=True)
            action = -2
        return action

    @staticmethod
    def readable_battle_order(battle: DoubleBattle, order: BattleOrder, pos: int) -> str:
        order_str = str(order).removeprefix("/choose ")
        if order_str.endswith(" 1"):
            target = (
                battle.opponent_active_pokemon[0].base_species
                if battle.opponent_active_pokemon[0] is not None
                else "empty slot"
            )
            order_str = f"{order_str[:-2]} targeting foe's {target}"
        elif order_str.endswith(" 2"):
            target = (
                battle.opponent_active_pokemon[1].base_species
                if battle.opponent_active_pokemon[1] is not None
                else "empty slot"
            )
            order_str = f"{order_str[:-2]} targeting foe's {target}"
        elif order_str.endswith(" -1"):
            target = (
                battle.active_pokemon[0].base_species
                if battle.active_pokemon[0] is not None
                else "empty slot"
            )
            order_str = f"{order_str[:-3]} targeting your {target}"
        elif order_str.endswith(" -2"):
            target = (
                battle.active_pokemon[1].base_species
                if battle.active_pokemon[1] is not None
                else "empty slot"
            )
            order_str = f"{order_str[:-3]} targeting your {target}"
        if "terastallize" in order_str:
            active_mon = battle.active_pokemon[pos]
            assert active_mon is not None
            order_str = order_str.replace(
                "terastallize", f"activating {active_mon.tera_type} tera type"
            )
        return order_str

    @staticmethod
    def readable_boost(boost: int) -> float:
        if boost >= 0:
            modifier = (2 + boost) / 2
        else:
            modifier = 2 / (2 - boost)
        return round(modifier, ndigits=2)

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        team_pokemon = list(battle.team.values())
        opponent_pokemon = list(battle.opponent_team.values())
        prompt = f"""
Here is the following observation:

Your Pokemon:
    1. {team_pokemon[0].base_species}
{LLMPlayer.explain_pokemon(team_pokemon[0])}
    2. {team_pokemon[1].base_species}
{LLMPlayer.explain_pokemon(team_pokemon[1])}
    3. {team_pokemon[2].base_species}
{LLMPlayer.explain_pokemon(team_pokemon[2])}
    4. {team_pokemon[3].base_species}
{LLMPlayer.explain_pokemon(team_pokemon[3])}
    5. {team_pokemon[4].base_species}
{LLMPlayer.explain_pokemon(team_pokemon[4])}
    6. {team_pokemon[5].base_species}
{LLMPlayer.explain_pokemon(team_pokemon[5])}
Opponent Pokemon:
    1. {opponent_pokemon[0].base_species}
{LLMPlayer.explain_pokemon(opponent_pokemon[0])}
    2. {opponent_pokemon[1].base_species}
{LLMPlayer.explain_pokemon(opponent_pokemon[1])}
    3. {opponent_pokemon[2].base_species}
{LLMPlayer.explain_pokemon(opponent_pokemon[2])}
    4. {opponent_pokemon[3].base_species}
{LLMPlayer.explain_pokemon(opponent_pokemon[3])}
    5. {opponent_pokemon[4].base_species}
{LLMPlayer.explain_pokemon(opponent_pokemon[4])}
    6. {opponent_pokemon[5].base_species}
{LLMPlayer.explain_pokemon(opponent_pokemon[5])}

You must respond with the indices of which Pokemon you wish to bring to the battle from teampreview. You can only select from your Pokemon, NOT the opponent's Pokemon.
Please respond with the format /team <action1><action2><action3><action4>. The order that you set the numbers determines the order that they come in. So, if you send /turn 1246, 1 and 2 will lead and 4 and 6 will be on the bench, whereas if you do /turn 4162, then 4 and 1 will lead and 6 and 2 will be on the bench. You need not limit yourself to the set of numbers 1, 2, 4, and 6; any number from 1-6 is acceptable, and each can only be used once.
Do **not** include any extra text, punctuation, or explanation.
"""
        input_dict = [
            {
                "role": "system",
                "content": f"You are an expert Pokemon VGC competitor playing a Pokemon battle in the {battle.format} format. You are currently in teampreview.",
            },
            {"role": "user", "content": prompt},
        ]
        response: str = self.model(input_dict)[0]["generated_text"][-1]["content"]  # type: ignore
        if re.match(r"^/team (?!.*([1-6]).*\1)[1-6]{4}$", response) is None:
            response = self.random_teampreview(battle)[:-2]
        self.__teampreview_draft = [int(i) for i in response[6:]]
        return response

    @staticmethod
    def explain_battle(
        battle: DoubleBattle,
        teampreview_draft: list[int],
        action_names: list[str],
        last_order: BattleOrder | None,
        pos: int,
    ) -> str:
        active_mon = battle.active_pokemon[pos]
        [a1, a2] = battle.active_pokemon
        [o1, o2] = battle.opponent_active_pokemon
        benched_pokemon = [
            p
            for i, p in enumerate(battle.team.values())
            if i + 1 in teampreview_draft and p not in [a1, a2]
        ]
        opp_benched_pokemon = [p for p in battle.opponent_team.values() if p not in [o1, o2]]
        listed_action_space = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(action_names))
        return f"""The following is what you are currently observing:

########## GLOBAL EFFECTS ##########

Active weather: {list(battle.weather.keys())[0] if battle.weather else "None"}
Active fields: {", ".join([str(f) for f in battle.fields.keys()]) or "None"}

########## YOUR SIDE ##########

Terastallization available: {any([c is not False for c in battle.can_tera])}
Active side conditions: {", ".join([str(s) for s in battle.side_conditions.keys()]) or None}

### Active Pokemon ###

Slot 1: {LLMPlayer.explain_pokemon(a1) if a1 is not None else "empty"}

Slot 2: {LLMPlayer.explain_pokemon(a2) if a2 is not None else "empty"}

### Benched Pokemon ###

{LLMPlayer.explain_pokemon(benched_pokemon[0])}

{LLMPlayer.explain_pokemon(benched_pokemon[1])}

{LLMPlayer.explain_pokemon(benched_pokemon[2]) if len(benched_pokemon) > 2 else "empty"}

########## OPPONENT SIDE ##########

Rating: {battle.opponent_rating}
Terastallization available: {battle._opponent_can_terrastallize}
Active side conditions: {", ".join([str(s) for s in battle.opponent_side_conditions.keys()]) or "None"}

### Active Pokemon ###

{LLMPlayer.explain_pokemon(o1) if o1 is not None else "empty"}

{LLMPlayer.explain_pokemon(o2) if o2 is not None else "empty"}

### Benched Pokemon ###

{LLMPlayer.explain_pokemon(opp_benched_pokemon[0])}

{LLMPlayer.explain_pokemon(opp_benched_pokemon[1])}

{LLMPlayer.explain_pokemon(opp_benched_pokemon[2])}

{LLMPlayer.explain_pokemon(opp_benched_pokemon[3])}

{LLMPlayer.explain_pokemon(opp_benched_pokemon[4]) if len(opp_benched_pokemon) > 4 else "empty"}

########## MAKE YOUR DECISION ##########

Please select the optimal action for slot {pos + 1}{f" (your {active_mon.base_species})" if active_mon is not None else ""}.

{f'The action you already chose for your first slot was {last_order}.' if pos == 1 else ''}

Here are your available actions:
{listed_action_space}

Respond with the number corresponding to your chosen action. PLEASE GIVE NO FURTHER RESPONSE THAN THAT, JUST THE NUMBER WITH NO PUNCTUATION!"""

    @staticmethod
    def explain_pokemon(pokemon: Pokemon) -> str:
        if pokemon.fainted:
            return f"{pokemon.base_species} (fainted)"
        elif not pokemon.active:
            return LLMPlayer.explain_inactive_pokemon(pokemon)
        else:
            return LLMPlayer.explain_inactive_pokemon(pokemon) + f"""
{LLMPlayer.explain_boosts(pokemon.boosts)}
Effects: {", ".join([str(e) for e in pokemon.effects]) or "None"}"""

    @staticmethod
    def explain_inactive_pokemon(pokemon: Pokemon) -> str:
        moves = list(pokemon.moves.values())
        reveal_str = "revealed" if pokemon.revealed else "unrevealed"
        type_str = "/".join([str(t) for t in pokemon.types])
        tera_type_str = (
            f"terastallized to {pokemon.tera_type}-type"
            if pokemon.is_terastallized
            else f"and unused tera-type of {pokemon.tera_type}"
        )
        hp_str = f"{round(100 * pokemon.current_hp_fraction)}%" if pokemon.max_hp > 0 else "unknown"
        if pokemon.fainted:
            return f"{pokemon.base_species} (fainted)"
        return f"""{pokemon.base_species} ({reveal_str} in battle), a {type_str}-type pokemon ({tera_type_str}) with {hp_str} HP, ability {pokemon.ability}, and held item {pokemon.item or "None"}.
Status Effect: {pokemon.status}
Moves:
    - {LLMPlayer.explain_move(moves[0]) if len(moves) > 0 else "None"}
    - {LLMPlayer.explain_move(moves[1]) if len(moves) > 1 else "None"}
    - {LLMPlayer.explain_move(moves[2]) if len(moves) > 2 else "None"}
    - {LLMPlayer.explain_move(moves[3]) if len(moves) > 3 else "None"}
Base stats:
    {pokemon.base_stats["hp"]} HP
    {pokemon.base_stats["atk"]} Attack
    {pokemon.base_stats["def"]} Defense
    {pokemon.base_stats["spa"]} Special Attack
    {pokemon.base_stats["spd"]} Special Defense
    {pokemon.base_stats["spe"]} Speed"""

    @staticmethod
    def explain_move(move: Move) -> str:
        return f"{move.id}, a {move.type}-type move with {move.base_power} power, {int(100 * move.accuracy)}% accuracy, and {move.current_pp}/{move.max_pp} pp"

    @staticmethod
    def explain_boosts(boosts: dict[str, int]) -> str:
        boost_str = "Stat Modifiers:"
        if boosts['atk'] != 0:
            boost_str += f"\n    Attack: x{LLMPlayer.readable_boost(boosts['atk'])}"
        if boosts['def'] != 0:
            boost_str += f"\n    Defense: x{LLMPlayer.readable_boost(boosts['def'])}"
        if boosts['spa'] != 0:
            boost_str += f"\n    Special Attack: x{LLMPlayer.readable_boost(boosts['spa'])}"
        if boosts['spd'] != 0:
            boost_str += f"\n    Special Defense: x{LLMPlayer.readable_boost(boosts['spd'])}"
        if boosts['spe'] != 0:
            boost_str += f"\n    Speed: x{LLMPlayer.readable_boost(boosts['spe'])}"
        if boosts['accuracy'] != 0:
            boost_str += f"\n    Accuracy: x{LLMPlayer.readable_boost(boosts['accuracy'])}"
        if boosts['evasion'] != 0:
            boost_str += f"\n    Evasion: x{LLMPlayer.readable_boost(boosts['evasion'])}"
        if boost_str == "Stat Modifiers:":
            boost_str += " None"
        return boost_str

    @staticmethod
    def _get_mask(ally_actions: torch.Tensor) -> torch.Tensor:
        indices = (
            torch.arange(doubles_act_len, device=ally_actions.device)
            .unsqueeze(0)
            .expand(len(ally_actions), -1)
        )
        ally_switched = (1 <= ally_actions) & (ally_actions <= 6)
        ally_terastallized = ally_actions >= 87
        mask = (
            ((27 <= indices) & (indices < 87))
            | ((indices == ally_actions) & ally_switched)
            | ((indices >= 87) & ally_terastallized)
        )
        mask = torch.where(mask == 1, float("-inf"), 0)
        return mask
