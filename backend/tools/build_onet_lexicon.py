from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
import unicodedata

try:
    from wordfreq import zipf_frequency
except ImportError as exc:
    raise SystemExit(
        "Install build-time lexicon tooling with "
        "'python -m pip install -r requirements-dev.txt'."
    ) from exc

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_ROOT / "data"
SKILLS_DIR = DATA_DIR / "skills"
META_PATH = DATA_DIR / "skills_meta.tsv"
ATTRIBUTION_PATH = SKILLS_DIR / "ATTRIBUTION.md"
OCCUPATION_PATH = DATA_DIR / "occupation_titles.txt"
COMMON_TOOLS = {
    "excel", "word", "powerpoint", "quickbooks", "sap",
    "microsoft word", "microsoft powerpoint",
}
DOMAIN_OVERRIDES = {"plc": {"engineering", "trades"}}
GENERIC_CATEGORIES = {
    "electronic mail software",
    "word processing software",
    "spreadsheet software",
    "database management software",
    "presentation software",
    "office suite software",
}
SOFT_SKILLS = {
    "active listening",
    "critical thinking",
    "complex problem solving",
    "coordination",
    "instructing",
    "learning strategies",
    "monitoring",
    "negotiation",
    "persuasion",
    "service orientation",
    "social perceptiveness",
    "time management",
    "writing",
    "speaking",
    "reading comprehension",
    "active learning",
    "judgment and decision making",
    "systems analysis",
    "systems evaluation",
    "operations analysis",
    "quality control analysis",
    "troubleshooting",
    "operation monitoring",
    "dependability",
    "integrity",
    "cooperation",
    "stress tolerance",
    "adaptability/flexibility",
    "initiative",
    "leadership",
    "persistence",
    "achievement/effort",
    "self-control",
    "independence",
    "innovation",
    "analytical thinking",
    "attention to detail",
    "empathy",
    "reliability",
}
DOMAIN_BY_SOC = {
    "11": "hr",
    "13": "finance",
    "15": "software",
    "17": "engineering",
    "19": "engineering",
    "21": "general",
    "23": "legal",
    "25": "teaching",
    "27": "design_creative",
    "29": "healthcare",
    "31": "healthcare",
    "33": "general",
    "35": "culinary",
    "37": "general",
    "39": "healthcare",
    "41": "sales_marketing",
    "43": "admin_office",
    "45": "general",
    "47": "trades",
    "49": "trades",
    "51": "trades",
    "53": "logistics",
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[®™©]", "", value)
    return " ".join(value.casefold().split())


def read_tables(
        root: Path,
) -> list[tuple[Path, list[str], list[dict[str, str]]]]:
    tables = []
    for path in sorted(root.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(stream, dialect=dialect)
            headers = [
                str(name or "").strip()
                for name in reader.fieldnames or []
            ]
            rows = [
                {
                    (key or "").strip(): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
                for row in reader
            ]
        tables.append((path, headers, rows))
    if not tables:
        raise FileNotFoundError(f"No O*NET CSV files found under {root}")
    return tables


def find_table(
    tables: list[tuple[Path, list[str], list[dict[str, str]]]],
    required_headers: set[str],
) -> tuple[Path, list[str], list[dict[str, str]]] | None:
    required = {header.casefold() for header in required_headers}
    for table in tables:
        headers = {header.casefold() for header in table[1]}
        if required.issubset(headers):
            return table
    return None


def get_field(headers: list[str], wanted: str) -> str | None:
    return next(
        (
            header for header in headers
            if header.casefold() == wanted.casefold()
        ),
        None,
    )


def get_domain(code: str) -> str:
    return DOMAIN_BY_SOC.get(code[:2], "general")


def truthy(value: str) -> bool:
    return normalize(value) in {"y", "yes", "true", "1", "hot"}


def title_version(root: Path, override: str | None) -> str:
    if override:
        return override
    for path in root.rglob("*.zip"):
        match = re.search(r"_(\d+\.\d+)_csv\.zip$", path.name, re.I)
        if match:
            return match.group(1)
    for path in root.rglob("*"):
        if path.is_file() and path.name.casefold().startswith("read me"):
            text = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(
                r"O\*NET(?:-SOC)?\s+Database\s+(\d+\.\d+)",
                text,
                re.I,
            )
            if match:
                return match.group(1)
    return "unknown"


def common_english_single(value: str) -> bool:
    return (
        len(value.split()) == 1
        and zipf_frequency(value, "en") >= 3.5
    )


def is_candidate(value: str) -> bool:
    value = normalize(value)
    words = re.findall(r"[a-z0-9+#.]+", value)
    return (
        bool(value)
        and value not in GENERIC_CATEGORIES
        and len(value) > 2
        and len(words) < 5
        and not (len(words) == 1 and common_english_single(value))
    )


def load_existing() -> tuple[dict[str, str], set[str]]:
    mapping: dict[str, str] = {}
    paths = [DATA_DIR / "skills.txt"]
    paths.extend(
        path for path in sorted(SKILLS_DIR.glob("*.txt"))
        if not path.name.startswith("onet_")
    )
    for path in paths:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            parts = [normalize(part) for part in raw.split("|")]
            if not parts or not parts[0] or parts[0].startswith("#"):
                continue
            canonical = parts[0]
            for alias in parts:
                if (
                    alias
                    and alias not in {"hard", "soft"}
                    and not alias.isdigit()
                ):
                    mapping.setdefault(alias, canonical)
                    mapping.setdefault(alias.replace(" ", ""), canonical)
    keys = set(mapping)
    return mapping, keys


def load_previous_onet() -> tuple[dict[str, set[str]], dict[str, str]]:
    entries: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, str] = {}
    for path in sorted(SKILLS_DIR.glob("onet_*.txt")):
        domain = path.stem.removeprefix("onet_")
        for raw in path.read_text(encoding="utf-8").splitlines():
            parts = [normalize(part) for part in raw.split("|")]
            if not parts or not parts[0]:
                continue
            canonical = parts[0]
            row = "|".join(parts)
            entries[domain].add(row)
            for alias in parts:
                if alias:
                    aliases.setdefault(alias, canonical)
                    aliases.setdefault(alias.replace(" ", ""), canonical)
    return entries, aliases


def write_meta(
    additions: dict[str, tuple[str, str, int, str, str]],
) -> None:
    existing: dict[str, list[str]] = {}
    if META_PATH.is_file():
        with META_PATH.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                key = row.get("canonical", "").strip().casefold()
                if key:
                    existing[key] = [
                        row.get(field, "").strip()
                        for field in (
                            "type", "domains", "specificity", "source",
                            "ambiguity_rule",
                        )
                    ]
    domains_by_canonical: dict[str, set[str]] = defaultdict(set)
    for path in sorted(SKILLS_DIR.glob("*.txt")):
        if path.name == "ATTRIBUTION.md":
            continue
        is_onet = path.name.startswith("onet_")
        domain = path.stem[5:] if is_onet else path.stem
        source = "onet" if is_onet else "curated"
        for raw in path.read_text(encoding="utf-8").splitlines():
            canon = raw.split("|", 1)[0].strip().casefold()
            if canon:
                domains_by_canonical[canon].add(domain)
                additions.setdefault(
                    canon, ("hard", domain, 2, source, "")
                )
    legacy_path = DATA_DIR / "skills.txt"
    for raw in legacy_path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip().casefold() for part in raw.split("|")]
        canon = parts[0] if parts else ""
        if not canon or canon.startswith("#"):
            continue
        skill_type = "soft" if "soft" in parts[1:] else "hard"
        additions.setdefault(
            canon,
            (
                skill_type,
                "general" if skill_type == "soft" else "software",
                2,
                "legacy",
                "",
            ),
        )
    for canon, values in additions.items():
        if canon in COMMON_TOOLS:
            domains = {"general"}
        elif canon in DOMAIN_OVERRIDES:
            domains = set(DOMAIN_OVERRIDES[canon])
        else:
            domains = set(values[1].split(","))
            domains.update(domains_by_canonical.get(canon, set()))
        if canon in existing:
            if (
                canon not in COMMON_TOOLS
                and canon not in DOMAIN_OVERRIDES
            ):
                domains.update(existing[canon][1].split(","))
            existing[canon][1] = ",".join(sorted(
                domain for domain in domains if domain
            ))
        else:
            existing[canon] = [
                values[0],
                ",".join(sorted(domain for domain in domains if domain)),
                values[2],
                values[3],
                values[4],
            ]
    with META_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["canonical", "type", "domains", "specificity", "source",
             "ambiguity_rule"]
        )
        for canon, values in sorted(existing.items()):
            writer.writerow([canon, *values])


def build(root: Path, release: str | None) -> None:
    tables = read_tables(root)
    occupation_table = find_table(
        tables,
        {"O*NET-SOC Code", "Title", "Description"},
    )
    if occupation_table is None:
        raise ValueError("O*NET Occupation Data table was not found")
    occ_path, occ_headers, occupations = occupation_table
    code_field = get_field(occ_headers, "O*NET-SOC Code")
    title_field = get_field(occ_headers, "Title")
    if not code_field or not title_field:
        raise ValueError(f"Invalid occupation table: {occ_path}")
    occupation_domains = {
        row[code_field]: get_domain(row[code_field])
        for row in occupations
        if row.get(code_field)
    }
    all_occupations = set(occupation_domains)
    if not all_occupations:
        raise ValueError("O*NET Occupation Data table is empty")

    software_table = find_table(
        tables,
        {
            "O*NET-SOC Code", "Workplace Example", "Hot Technology",
            "In Demand",
        },
    )
    tool_tables = [
        table for table in tables
        if any("tool" in header.casefold() for header in table[1])
        and get_field(table[1], "O*NET-SOC Code")
    ]
    existing_aliases, _keys = load_existing()
    previous_entries, previous_aliases = load_previous_onet()
    occurrences: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    flags: dict[str, bool] = defaultdict(bool)
    before: dict[str, set[str]] = defaultdict(set)
    files_used = [occ_path.name]

    source_tables = (
        [(software_table, "software")] if software_table else []
    )
    source_tables.extend((table, "tools") for table in tool_tables)
    for table, label in source_tables:
        path, headers, rows = table
        code_key = get_field(headers, "O*NET-SOC Code")
        if not code_key:
            continue
        example_fields = [
            header for header in headers
            if (
                "example" in header.casefold()
                or "tool" in header.casefold()
            )
        ]
        if label == "software":
            example_fields = ["Workplace Example"]
        if not example_fields:
            continue
        hot_key = get_field(headers, "Hot Technology")
        demand_key = get_field(headers, "In Demand")
        files_used.append(path.name)
        for row in rows:
            code = row.get(code_key, "")
            domain = occupation_domains.get(code)
            if not domain:
                continue
            is_hot = bool(
                (hot_key and truthy(row.get(hot_key, "")))
                or (demand_key and truthy(row.get(demand_key, "")))
            )
            for field in example_fields:
                name = normalize(row.get(field, ""))
                if not is_candidate(name):
                    continue
                before[domain].add(name)
                occurrences[name][domain].add(code)
                flags[name] = flags[name] or is_hot

    filtered: dict[str, list[tuple[str, str]]] = defaultdict(list)
    meta_additions: dict[str, tuple[str, str, int, str, str]] = {}
    additions: dict[str, set[str]] = defaultdict(set)
    for name, domain_counts in occurrences.items():
        total = sum(len(codes) for codes in domain_counts.values())
        if total / len(all_occupations) > 0.05:
            continue
        ranked = sorted(
            (
                (domain, len(codes))
                for domain, codes in domain_counts.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not ranked:
            continue
        main_domain, main_count = ranked[0]
        if main_count < 2 and not flags[name]:
            continue
        share = main_count / total
        domain = (
            "general"
            if name in COMMON_TOOLS
            else main_domain
        )
        if share < 0.5 and name not in COMMON_TOOLS:
            continue
        if domain == "general" and name not in COMMON_TOOLS:
            continue

        existing_canon = existing_aliases.get(name)
        if not existing_canon:
            existing_canon = existing_aliases.get(name.replace(" ", ""))
        if existing_canon:
            if name not in existing_aliases:
                additions[domain].add(f"{existing_canon}|{name}")
            continue
        if (
            name in previous_aliases
            or name.replace(" ", "") in previous_aliases
        ):
            continue
        filtered[domain].append((name, name))
        meta_additions[name] = (
            "hard", domain, 2 if share < 0.75 else 3, "onet", ""
        )

    soft_hits: set[str] = set()
    soft_sources: set[str] = set()
    for path, headers, rows in tables:
        code_key = get_field(headers, "O*NET-SOC Code")
        name_key = get_field(headers, "Element Name")
        if not code_key or not name_key:
            continue
        for row in rows:
            name = normalize(row.get(name_key, ""))
            if name not in SOFT_SKILLS or not is_candidate(name):
                continue
            if row.get(code_key, "") not in occupation_domains:
                continue
            soft_hits.add(name)
            soft_sources.add(path.name)
    files_used.extend(sorted(soft_sources))
    for name in sorted(soft_hits):
        filtered["general"].append((name, name))
        before["general"].add(name)
        meta_additions[name] = ("soft", "general", 2, "onet", "")

    title_domains: dict[str, set[str]] = defaultdict(set)
    for row in occupations:
        code = row.get(code_field, "")
        title = normalize(row.get(title_field, ""))
        if title:
            title_domains[title].add(occupation_domains.get(code, "general"))
    for path, headers, rows in tables:
        code_key = get_field(headers, "O*NET-SOC Code")
        if not code_key:
            continue
        title_fields = [
            header for header in headers
            if (
                "job title" in header.casefold()
                or "reported title" in header.casefold()
            )
        ]
        for row in rows:
            code = row.get(code_key, "")
            domain = occupation_domains.get(code)
            if not domain:
                continue
            for field in title_fields:
                title = normalize(row.get(field, ""))
                if title and len(title.split()) <= 8:
                    title_domains[title].add(domain)
        if title_fields:
            files_used.append(path.name)
    OCCUPATION_PATH.write_text(
        "".join(
            f"{title}|{','.join(sorted(domains))}\n"
            for title, domains in sorted(title_domains.items())
        ),
        encoding="utf-8",
        newline="\n",
    )

    counts_after: Counter[str] = Counter()
    counts_added: Counter[str] = Counter()
    domains = set(previous_entries) | set(filtered) | set(additions)
    for domain in sorted(domains):
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        path = SKILLS_DIR / f"onet_{domain}.txt"
        entries = set(previous_entries[domain])
        new_entries = set(additions[domain])
        new_entries.update(
            f"{canon}|{alias}" for canon, alias in filtered[domain]
        )
        entries.update(new_entries)
        path.write_bytes(
            ("\r\n".join(sorted(entries, key=str.casefold)) + "\r\n")
            .encode("utf-8")
        )
        counts_after[domain] = len(entries)
        counts_added[domain] = len(new_entries - previous_entries[domain])
    write_meta(meta_additions)
    version = title_version(root, release)
    ATTRIBUTION_PATH.write_text(
        "# O*NET data attribution\n\n"
        f"Contains filtered and modified data from the O*NET {version} "
        "Database "
        "published by the U.S. Department of Labor, National Center for "
        "O*NET Development. The source is licensed under CC BY 4.0. "
        f"Release used: {version}.\n\n"
        "The source data was normalized, filtered by occupation domain, "
        "frequency, specificity, and common-English rules, and deduplicated "
        "against the existing skill lexicon.\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"O*NET release: {version}")
    print("Files used: " + ", ".join(sorted(set(files_used))))
    print("Entries by domain (candidates before / retained):")
    for domain in sorted(set(before) | set(counts_after)):
        examples = {
            normalize(row.split("|")[-1])
            for row in previous_entries.get(domain, set())
        }
        examples.update(
            name for name, _alias in filtered.get(domain, [])
        )
        print(
            f"  {domain}: {len(before[domain])} / "
            f"{counts_after[domain]} retained "
            f"({counts_added[domain]} added)"
        )
        print("    samples: " + ", ".join(sorted(examples)[:20]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build domain skill lists from O*NET CSV data."
    )
    parser.add_argument(
        "--onet-dir",
        type=Path,
        default=BACKEND_ROOT / "samples" / "onet",
    )
    parser.add_argument("--release")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.onet_dir, args.release)


if __name__ == "__main__":
    sys.path.insert(0, str(BACKEND_ROOT))
    main()
