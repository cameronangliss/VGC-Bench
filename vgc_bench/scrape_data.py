import json
import os
import re
import warnings

import requests
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def update_desc_embeddings(url: str, file: str, extras: dict[str, dict[str, str]] = {}):
    response = requests.get(f"{url}/{file}")
    if ".json" in file:
        json_text = response.text
    else:
        js_text = response.text
        i = js_text.index("{")
        js_literal = js_text[i:-1]
        json_text = re.sub(r"([{,])([a-zA-Z0-9_]+)(:)", r'\1"\2"\3', js_literal)
        file += "on"
    dex = {k: v["desc"] for k, v in {**extras, **json.loads(json_text)}.items() if "desc" in v}
    with open(f"data/{file}", "w") as f:
        json.dump(dex, f)


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.mkdir("data")
    update_desc_embeddings(
        "https://play.pokemonshowdown.com/data", "abilities.js", extras={"null": {"desc": "null"}}
    )
    update_desc_embeddings(
        "https://play.pokemonshowdown.com/data",
        "items.js",
        extras={
            "null": {"desc": "null"},
            "": {"desc": "empty"},
            "unknown_item": {"desc": "unknown item"},
        },
    )
    update_desc_embeddings(
        "https://play.pokemonshowdown.com/data", "moves.js", extras={"no move": {"desc": "no move"}}
    )
