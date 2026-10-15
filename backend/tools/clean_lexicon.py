from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys

try:
    from wordfreq import zipf_frequency
except ImportError as exc:
    raise SystemExit(
        "Install build-time lexicon tooling with "
        "'python -m pip install -r requirements-dev.txt'."
    ) from exc

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SKILLS_PATH = BACKEND_ROOT / "data" / "skills.txt"
OVERRIDES_PATH = BACKEND_ROOT / "data" / "skills_overrides.txt"
PRODUCT_WORDS = {
    "nanoweb", "tapewound", "roundwound", "flatwound", "daddario",
    "ernie ball", "stringjoy", "bass guitar",
}
MUSIC_WORDS = {
    "plugin alliance", "universal audio", "apollo", "ableton", "cubase",
    "reaper", "pro tools", "fabfilter", "izotope", "waves", "lufs",
    "true peak", "foley", "adr", "ambisonics", "binaural", "loudness",
    "mastering", "shotgun mic", "field recorder",
}
CRYPTO_WORDS = {
    "bitcoin", "cryptocurrency", "crypto", "blockchain", "lightning",
    "defi", "nft", "trezor", "coldcard", "bitbox", "electrum",
    "bluewallet", "satoshi", "bolt11", "bolt12", "zeusln",
}
CATEGORY_WORDS = {
    "category theory", "profunctor", "contravariant", "bifunctor",
    "monad", "monaderror", "functor", "colax", "oplax", "unital",
}
SOFT_MARKER = "soft"


def read_overrides() -> tuple[set[str], set[str]]:
    whitelist: set[str] = set()
    blacklist: set[str] = set()
    for raw in OVERRIDES_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rule, separator, term = line.partition("|")
        if not separator:
            raise ValueError(f"Invalid override line: {line}")
        term = term.strip().casefold()
        if rule == "whitelist":
            whitelist.add(term)
        elif rule == "blacklist":
            blacklist.add(term)
        else:
            raise ValueError(f"Unknown override rule: {rule}")
    return whitelist, blacklist


def off_domain(line: str) -> str | None:
    folded = line.casefold()
    terms = set(re.findall(r"[a-z0-9+#.-]+", folded))
    if (
        any(term in folded for term in PRODUCT_WORDS)
        or ("guitar" in terms and "elixir" in terms)
        or ("bass" in terms and "string" in terms)
    ):
        return "music_products"
    if any(term in folded for term in MUSIC_WORDS):
        return "audio"
    if any(term in terms for term in CRYPTO_WORDS):
        return "crypto"
    if any(term in folded for term in CATEGORY_WORDS):
        return "category_theory"
    return None


def clean() -> Counter[str]:
    whitelist, blacklist = read_overrides()
    counts: Counter[str] = Counter()
    output: list[str] = ["# canonical|alias|alias"]
    canonicals: set[str] = set()
    aliases: set[str] = set()

    for raw in SKILLS_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.isdigit():
            continue
        parts = [part.strip().casefold() for part in line.split("|")]
        canon = parts[0]
        skill_type = "soft" if SOFT_MARKER in parts[1:2] else "hard"
        aliases_in = [part for part in parts if part and part != SOFT_MARKER]
        if canon in blacklist or any(
            part in blacklist for part in aliases_in if part != SOFT_MARKER
        ):
            counts["blacklist"] += 1
            continue
        reason = off_domain(line)
        if reason:
            counts[reason] += 1
            continue
        if len(canon.split()) >= 4:
            counts["product_like"] += 1
            continue
        if canon in canonicals:
            counts["duplicate_canonical"] += 1
            continue
        if (
            len(canon.split()) == 1
            and zipf_frequency(canon, "en") >= 3.5
            and canon not in whitelist
        ):
            counts["common_english"] += 1
            continue

        kept: list[str] = []
        for alias in aliases_in:
            if alias == "soft":
                continue
            if (
                len(alias.split()) == 1
                and zipf_frequency(alias, "en") >= 3.5
                and alias not in whitelist
            ):
                counts["common_english"] += 1
                continue
            compact = re.sub(r"\s+", "", alias)
            if alias in aliases or compact in aliases:
                counts["alias_collision"] += 1
                continue
            kept.append(alias)
            aliases.add(alias)
            aliases.add(compact)
        if not kept:
            counts["empty_after_alias_filter"] += 1
            continue
        if canon not in kept:
            kept.insert(0, canon)
            aliases.add(canon)
            aliases.add(re.sub(r"\s+", "", canon))
        canonicals.add(canon)
        output.append("|".join([canon, *kept[1:]] + (
            ["soft"] if skill_type == "soft" else []
        )))
        counts["kept"] += 1

    SKILLS_PATH.write_bytes(("\r\n".join(output) + "\r\n").encode("utf-8"))
    return counts


def main() -> None:
    counts = clean()
    print("Lexicon cleanup counts:")
    for reason, count in sorted(counts.items()):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
