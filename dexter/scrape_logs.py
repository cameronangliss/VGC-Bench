import json
import os
from typing import Any

import requests
from src.utils import battle_format


def scrape_logs(increment: int):
    if os.path.exists("data/logs.json"):
        with open("data/logs.json", "r") as f:
            old_logs = json.load(f)
        log_times = [int(time) for time, _ in old_logs.values()]
        before = min(log_times)
        newest = max(log_times)
    else:
        old_logs = {}
        before = 2_000_000_000
        newest = None
    print("total battle logs:", len(old_logs), end="\r")
    battle_idents = get_battle_idents(increment, before + 1, newest)
    log_jsons = [get_log_json(ident) for ident in battle_idents]
    new_logs = {
        lj["id"]: (lj["uploadtime"], lj["log"])
        for lj in log_jsons
        if lj is not None
        and lj["log"].count("|poke|p1|") == 6
        and lj["log"].count("|poke|p2|") == 6
        and "|showteam|" in lj["log"]
        and "Zoroark" not in lj["log"]
        and "Zorua" not in lj["log"]
    }
    logs = {**old_logs, **new_logs}
    with open("data/logs.json", "w") as f:
        json.dump(logs, f)


def get_battle_idents(num_battles: int, before: int, newest: int | None) -> list[str]:
    battle_idents = set()
    if newest is not None:
        before_ = 2_000_000_000
        while len(battle_idents) < num_battles and before >= newest:
            battle_idents, before_ = update_battle_idents(battle_idents, before_)
    while len(battle_idents) < num_battles:
        battle_idents, before = update_battle_idents(battle_idents, before)
    return list(battle_idents)[:num_battles]


def update_battle_idents(battle_idents: set[str], before: int) -> tuple[set[str], int]:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/search.json?format={battle_format}&before={before}")
    new_battle_jsons = json.loads(response.text)
    before = new_battle_jsons[-1]["uploadtime"] + 1
    battle_idents |= {
        bj["id"]
        for bj in new_battle_jsons
        if bj["id"].startswith(battle_format) and bj["rating"] is None
    }
    return battle_idents, before


def get_log_json(ident: str) -> dict[str, Any] | None:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/{ident}.json")
    if response:
        return json.loads(response.text)


if __name__ == "__main__":
    while True:
        scrape_logs(1000)
