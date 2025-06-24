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
                battle.active_pokemon[0].base_species
                if battle.active_pokemon[0] is not None
                else "empty slot"
            )
            order_str = f"{order_str[:-2]} targeting {target} (foe's pokemon)"
        elif order_str.endswith(" 2"):
            target = (
                battle.active_pokemon[1].base_species
                if battle.active_pokemon[1] is not None
                else "empty slot"
            )
            order_str = f"{order_str[:-2]} targeting {target} (foe's pokemon)"
        elif order_str.endswith(" -1"):
            target = (
                battle.opponent_active_pokemon[0].base_species
                if battle.opponent_active_pokemon[0] is not None
                else "empty slot"
            )
            order_str = f"{order_str[:-3]} targeting {target} (your pokemon)"
        elif order_str.endswith(" -2"):
            target = (
                battle.opponent_active_pokemon[1].base_species
                if battle.opponent_active_pokemon[1] is not None
                else "empty slot"
            )
            order_str = f"{order_str[:-3]} targeting {target} (your pokemon)"
        if "terastallize" in order_str:
            active_mon = battle.active_pokemon[pos]
            assert active_mon is not None
            order_str = order_str.replace(
                "terastallize", f"activating {active_mon.tera_type} tera type"
            )
        return order_str

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
        glob = LLMPlayer.explain_global(battle)
        side = LLMPlayer.explain_side(battle)
        opp_side = LLMPlayer.explain_side(battle, opp=True)
        [a1, a2] = battle.active_pokemon
        [o1, o2] = battle.opponent_active_pokemon
        benched_pokemon = [
            p
            for i, p in enumerate(battle.team.values())
            if i + 1 in teampreview_draft and p not in [a1, a2]
        ]
        opp_benched_pokemon = [p for p in battle.opponent_team.values() if p not in [o1, o2]]
        listed_action_space = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(action_names))
        return f"""
The following is what you are currently observing:

Global Conditions in Battle:
{glob}
Side Conditions:
{side}
Active Pokemon:
    1. {a1.base_species if a1 is not None else "empty"}
{LLMPlayer.explain_pokemon(a1) if a1 is not None else ""}
    2. {a2.base_species if a2 is not None else "empty"}
{LLMPlayer.explain_pokemon(a2) if a2 is not None else ""}
Benched Pokemon:
    1. {benched_pokemon[0].base_species}
{LLMPlayer.explain_pokemon(benched_pokemon[0])}
    2. {benched_pokemon[1].base_species}
{LLMPlayer.explain_pokemon(benched_pokemon[1])}
    3. {benched_pokemon[2].base_species if len(benched_pokemon) > 2 else "empty"}
{LLMPlayer.explain_pokemon(benched_pokemon[2]) if len(benched_pokemon) > 2 else ""}
Opponent Side Conditions:
{opp_side}
Opponent Active Pokemon:
    1. {o1.base_species if o1 is not None else "empty"}
{LLMPlayer.explain_pokemon(o1) if o1 is not None else ""}
    2. {o2.base_species if o2 is not None else "empty"}
{LLMPlayer.explain_pokemon(o2) if o2 is not None else ""}
Opponent Benched Pokemon:
    1. {opp_benched_pokemon[0].base_species}
{LLMPlayer.explain_pokemon(opp_benched_pokemon[0])}
    2. {opp_benched_pokemon[1].base_species}
{LLMPlayer.explain_pokemon(opp_benched_pokemon[1])}
    3. {opp_benched_pokemon[2].base_species}
{LLMPlayer.explain_pokemon(opp_benched_pokemon[2])}
    4. {opp_benched_pokemon[3].base_species}
{LLMPlayer.explain_pokemon(opp_benched_pokemon[3])}
    5. {opp_benched_pokemon[4].base_species if len(opp_benched_pokemon) > 4 else "empty"}
{LLMPlayer.explain_pokemon(opp_benched_pokemon[4]) if len(opp_benched_pokemon) > 4 else ""}

Please select the optimal action given this observation for your {['first', 'second'][pos]} active pokemon: {battle.active_pokemon[pos]}.

{f'The action you already chose for your first pokemon, {battle.active_pokemon[0]}, was {last_order}.' if pos == 1 else ''}

Here are your available actions:
{listed_action_space}

Respond with the number corresponding to your chosen action. PLEASE GIVE NO FURTHER RESPONSE THAN THAT, JUST THE NUMBER WITH NO PUNCTUATION!
"""

    @staticmethod
    def explain_global(battle: DoubleBattle) -> str:
        return f"""
    Current Turn: {battle.turn}
    Your Pokemon 1 is being forced to switch: {battle.force_switch[0]}
    Your Pokemon 2 is being forced to switch: {battle.force_switch[1]}
    The active weather in the game (as a dictionary, mapping to its starting turn): {battle.weather}
    The active fields in the game (as a dictionary, mapping to its starting turn): {battle.fields}
"""

    @staticmethod
    def explain_side(battle: DoubleBattle, opp: bool = False) -> str:
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
        side_conds = battle.opponent_side_conditions if opp else battle.side_conditions
        gims = opp_gims if opp else gims
        rat = battle.opponent_rating if opp else battle.rating
        return f"""
Player Rating: {rat}
Player can still mega evolve: {gims[0]}
Player can still z-move: {gims[1]}
Player can still dynamax/gigantamax: {gims[2]}
Player can still terastallize: {gims[3]}
Side conditions on this player's side (the number of layers of the SideCondition if the side condition is stackable, or the turn where the SideCondition was setup otherwise): {side_conds}
"""

    @staticmethod
    def explain_pokemon(pokemon: Pokemon) -> str:
        move_names = list(pokemon.moves.keys())
        moves = list(pokemon.moves.values())
        return f"""
        Ability: {pokemon.ability}
        Item: {pokemon.item}
        Moves:
            1. {move_names[0] if move_names[0] else ""}
        {LLMPlayer.explain_move(moves[0]) if moves[0] else ""}
            2. {move_names[1] if move_names[1] else ""}
        {LLMPlayer.explain_move(moves[1]) if moves[1] else ""}
            3. {move_names[2] if move_names[2] else ""}
        {LLMPlayer.explain_move(moves[2]) if moves[2] else ""}
            4. {move_names[3] if move_names[3] else ""}
        {LLMPlayer.explain_move(moves[3]) if moves[3] else ""}
        Types: {pokemon.types[0]}, {pokemon.types[1] if len(pokemon.types) == 2 else ""}
        Tera Type: {pokemon.tera_type}
        Stats: {pokemon.stats["hp"]} HP, {pokemon.stats["atk"]} Attack, {pokemon.stats["def"]} Defense, {pokemon.stats["spa"]} Special Attack, {pokemon.stats["spd"]} Special Defense, {pokemon.stats["spe"]} Speed
        Gender: {pokemon.gender}
        Weight: {pokemon.weight}
        Current HP Fraction: {pokemon.current_hp_fraction}
        Has been revealed in battle: {pokemon.revealed}
        Status effect: {pokemon.status}
        Number of turns with that status effect (only for toxic and sleep): {pokemon.status_counter}
        Boosts: {pokemon.boosts["accuracy"]} Accuracy, {pokemon.boosts["atk"]} Attack, {pokemon.boosts["def"]} Defense, {pokemon.boosts["evasion"]} Evasion, {pokemon.boosts["spa"]} Special Attack, {pokemon.boosts["spd"]} Special Defense, {pokemon.boosts["spe"]} Speed
        Effects (mapping effect name to number of turns left for the effect): {pokemon.effects}
        Is first turn being in (effects moves like fake out): {pokemon.first_turn}
        Number of turns protect has been used in a row: {pokemon.protect_counter}
        Currently recharging (from a move like hyper beam): {pokemon.must_recharge}
        Is currently preparing a move (like solar beam): {pokemon.preparing}
        Is dynamaxed: {pokemon.is_dynamaxed}
        Is terastallized: {pokemon.is_terastallized}
"""

    @staticmethod
    def explain_move(move: Move) -> str:
        return f"""
                Power: {move.base_power}
                Accuracy: {move.accuracy}
                Type: {move.type}
                Category: {move.category}
                Target: {move.target}
                Priority: {move.priority}
                Critical-hit Ratio: {move.crit_ratio}
                Drain ratio: {move.drain}
                Forces switch: {move.force_switch}
                Has recoil damage: {move.recoil}
                Switches self out: {move.self_switch}
                Current PP: {move.current_pp}
                Max PP: {move.max_pp}
"""

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
