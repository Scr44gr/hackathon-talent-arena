import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def weighted_choice(items: list[Any], weights: list[float]) -> Any:
    if len(items) != len(weights) or not items:
        raise ValueError("weighted_choice: items/weights mismatch or empty")
    total = float(sum(max(0.0, float(w)) for w in weights))
    if total <= 0:
        return random.choice(items)
    r = random.random() * total
    upto = 0.0
    for item, weight in zip(items, weights):
        upto += max(0.0, float(weight))
        if upto >= r:
            return item
    return items[-1]



QWERTY_NEIGHBORS = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
    'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
    'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
}

COMMON_PRETERITE_2SG = [
    "dijiste",
    "hiciste",
    "pudiste",
    "quisiste",
    "supiste",
    "fuiste",
    "tuviste",
    "viste",
    "pusiste",
    "trajiste",
    "viniste",
    "saliste",
    "llegaste",
    "preguntaste",
    "contestaste",
    "intentaste",
    "probaste",
    "buscaste",
    "encontraste",
    "mandaste",
    "enviaste",
    "escribiste",
    "leiste",
    "creiste",
    "pediste",
    "notaste",
    "cambiaste",
    "borraste",
    "pegaste",
    "editaste",
]


class TypoOps:
    def _qwerty_candidates(self, text: str) -> list[int]:
        chars = list(text)
        return [
            i
            for i in range(1, len(chars) - 1)
            if chars[i].isalpha() and chars[i].lower() in QWERTY_NEIGHBORS
        ]

    def qwerty_once(self, text: str) -> str:
        chars = list(text)
        candidates = self._qwerty_candidates(text)
        if not candidates:
            return text
        idx = random.choice(candidates)
        base = chars[idx].lower()
        repl = random.choice(QWERTY_NEIGHBORS[base])
        chars[idx] = repl.upper() if chars[idx].isupper() else repl
        return "".join(chars)

    def omission_once(self, text: str, vowel_bias: float = 0.8) -> str:
        chars = list(text)
        if len(chars) < 3:
            return text
        vowels = set("aeiouaeiouuAEIOUAEIOUU")
        vowel_candidates = [i for i, c in enumerate(chars) if c in vowels]
        alpha_candidates = [i for i, c in enumerate(chars) if c.isalpha()]
        if not alpha_candidates:
            return text
        idx = (
            random.choice(vowel_candidates)
            if vowel_candidates and random.random() < vowel_bias
            else random.choice(alpha_candidates)
        )
        del chars[idx]
        return "".join(chars)

    def abbr_once(self, text: str, weight_q: float, weight_pq: float) -> str:
        candidates: list[tuple[str, float]] = []
        if re.search(r"\bque\b", text, flags=re.IGNORECASE):
            candidates.append(("q", max(0.0, float(weight_q))))
        if re.search(r"\bporque\b", text, flags=re.IGNORECASE) or re.search(
            r"\bpor\s+que\b", text, flags=re.IGNORECASE
        ):
            candidates.append(("pq", max(0.0, float(weight_pq))))
        if not candidates:
            return text

        op = weighted_choice([c[0] for c in candidates], [c[1] for c in candidates])
        if op == "q":
            return re.sub(r"\bque\b", "q", text, flags=re.IGNORECASE, count=1)
        t = re.sub(r"\bpor\s+que\b", "pq", text, flags=re.IGNORECASE, count=1)
        return re.sub(r"\bporque\b", "pq", t, flags=re.IGNORECASE, count=1)

    def remove_space_once(self, text: str) -> str:
        if " " not in text:
            return text
        chars = list(text)
        candidates = [
            i
            for i in range(1, len(chars) - 1)
            if chars[i] == " " and chars[i - 1] != " " and chars[i + 1] != " "
        ]
        if not candidates:
            return text
        del chars[random.choice(candidates)]
        return "".join(chars)


def apply_typos_weighted_exact(
    text: str,
    n_typos: int,
    ops: TypoOps,
    typo_type_weights: dict[str, float],
    vowel_delete_bias: float,
    abbr_q_weight: float,
    abbr_pq_weight: float,
    max_attempts: int = 120,
) -> str:
    applied = 0
    attempts = 0
    type_names = ["qwerty", "omission", "abbr", "space_remove"]

    while applied < n_typos and attempts < max_attempts:
        attempts += 1
        before = text
        weights = [
            float(typo_type_weights.get("qwerty", 0.5)),
            float(typo_type_weights.get("omission", 0.3)),
            float(typo_type_weights.get("abbr", 0.2)),
            float(typo_type_weights.get("space_remove", 0.0)),
        ]
        chosen = weighted_choice(type_names, weights)

        if chosen == "qwerty":
            text = ops.qwerty_once(text)
        elif chosen == "omission":
            text = ops.omission_once(text, vowel_bias=vowel_delete_bias)
        elif chosen == "abbr":
            text = ops.abbr_once(text, weight_q=abbr_q_weight, weight_pq=abbr_pq_weight)
        else:
            text = ops.remove_space_once(text)

        if text != before:
            applied += 1
    return text


def normalize_block1(text: str, accents_drop_prob: float) -> str:
    text = re.sub(r"¿", "", text)
    if random.random() < accents_drop_prob:
        text = strip_accents(text)
    return text


GrammarRule = Callable[[str], str]


class GrammarRules:
    def __init__(self):
        self.homophone_pairs = [
            (r"\bhecho\b", "echo"),
            (r"\becho\b", "hecho"),
            (r"\bvaya\b", "valla"),
            (r"\bvalla\b", "vaya"),
            (r"\bhaber\b", "a ver"),
            (r"\ba ver\b", "haber"),
            (r"\bhay\b", "ay"),
            (r"\bay\b", "hay"),
            (r"\boye\b", "olle"),
            (r"\bolle\b", "oye"),
        ]
        self.porque_pairs = [
            (r"\bporque\b", "por que"),
            (r"\bpor\s+que\b", "porque"),
            (r"\bpor qué\b", "porque"),
            (r"\bporqué\b", "porque"),
        ]

    def habia_to_habian(self, text: str) -> str:
        t = strip_accents(text)
        if re.search(r"\bhabia\b", t, flags=re.IGNORECASE):
            return re.sub(
                r"\bhabia\b",
                "habian",
                strip_accents(text),
                flags=re.IGNORECASE,
                count=1,
            )
        return text

    def hemos_to_habemos(self, text: str) -> str:
        return re.sub(r"\bhemos\b", "habemos", text, flags=re.IGNORECASE, count=1)

    def homophones(self, text: str) -> str:
        for pat, repl in self.homophone_pairs:
            if re.search(pat, text, flags=re.IGNORECASE):
                return re.sub(pat, repl, text, flags=re.IGNORECASE, count=1)
        return text

    def porque(self, text: str) -> str:
        for pat, repl in self.porque_pairs:
            if re.search(pat, text, flags=re.IGNORECASE):
                return re.sub(pat, repl, text, flags=re.IGNORECASE, count=1)
        return text

    def seseo_ceceo(self, text: str, max_replacements: int = 2) -> str:
        pairs = [
            (r"za", "sa"),
            (r"zo", "so"),
            (r"zu", "su"),
            (r"ce", "se"),
            (r"ci", "si"),
            (r"sa", "za"),
            (r"so", "zo"),
            (r"su", "zu"),
            (r"se", "ce"),
            (r"si", "ci"),
        ]
        made = 0
        for pat, repl in pairs:
            if made >= max_replacements:
                break
            if re.search(pat, text, flags=re.IGNORECASE):
                text = re.sub(pat, repl, text, flags=re.IGNORECASE, count=1)
                made += 1
        return text

    def preterite_s(self, text: str) -> str:
        t = strip_accents(text)
        earliest = None
        for verb in COMMON_PRETERITE_2SG:
            match = re.search(rf"\b{re.escape(verb)}\b", t, flags=re.IGNORECASE)
            if match and (earliest is None or match.start() < earliest[0]):
                earliest = (match.start(), verb)
        if earliest is None:
            return text
        return re.sub(
            rf"\b{re.escape(earliest[1])}\b",
            earliest[1] + "s",
            strip_accents(text),
            flags=re.IGNORECASE,
            count=1,
        )

    def drop_initial_h(self, text: str) -> str:
        pattern = re.compile(r"\b([hH])([A-Za-zÁÉÍÓÚÜáéíóúüÑñ])")
        match = pattern.search(text)
        if not match:
            return text
        start, end = match.span(1)
        return text[:start] + text[end:]

    def swap_bv(self, text: str) -> str:
        match = re.search(r"[bB]", text)
        if match:
            idx = match.start()
            return text[:idx] + ("v" if text[idx] == "b" else "V") + text[idx + 1 :]
        match = re.search(r"[vV]", text)
        if match:
            idx = match.start()
            return text[:idx] + ("b" if text[idx] == "v" else "B") + text[idx + 1 :]
        return text

    def registry(self) -> dict[str, GrammarRule]:
        return {
            "habia_to_habian": self.habia_to_habian,
            "hemos_to_habemos": self.hemos_to_habemos,
            "homophones": self.homophones,
            "porque": self.porque,
            "seseo_ceceo": self.seseo_ceceo,
            "preterite_s": self.preterite_s,
            "drop_initial_h": self.drop_initial_h,
            "swap_bv": self.swap_bv,
        }


def normalize_block2(text: str) -> str:
    text = strip_accents(text)
    text = re.sub(r"¿", "", text)
    text = re.sub(r",", "", text)
    return text


def apply_grammar_ordered(
    text: str,
    n_changes: int,
    rule_order: list[str],
    rule_registry: dict[str, GrammarRule],
) -> str:
    applied = 0
    for name in rule_order:
        if applied >= n_changes:
            break
        updated = rule_registry[name](text)
        if updated != text:
            text = updated
            applied += 1
    return text


@dataclass
class CustomConfig:
    n_typos: int = 2
    n_grammar_changes: int = 2
    typo_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "qwerty": 0.5,
            "omission": 0.3,
            "abbr": 0.2,
            "space_remove": 0.0,
        }
    )
    vowel_delete_bias: float = 0.9
    abbr_q_weight: float = 0.6
    abbr_pq_weight: float = 0.4
    grammar_rule_weights: dict[str, float] = field(
        default_factory=lambda: {
            "habia_to_habian": 1.0,
            "hemos_to_habemos": 0.7,
            "homophones": 0.7,
            "porque": 0.9,
            "seseo_ceceo": 0.4,
            "preterite_s": 0.3,
            "drop_initial_h": 0.3,
            "swap_bv": 0.2,
        }
    )
    remove_open_questions: bool = True
    strip_accents: bool = True
    remove_commas: bool = True
    lowercase: bool = True


def normalize_custom(text: str, cfg: CustomConfig) -> str:
    if cfg.remove_open_questions:
        text = re.sub(r"¿", "", text)
    if cfg.strip_accents:
        text = strip_accents(text)
    if cfg.remove_commas:
        text = re.sub(r",", "", text)
    if cfg.lowercase:
        text = text.lower()
    return text


def apply_grammar_weighted(
    text: str,
    n_changes: int,
    rule_registry: dict[str, GrammarRule],
    weights_by_rule: dict[str, float],
    max_attempts: int = 120,
) -> str:
    applied = 0
    attempts = 0
    while applied < n_changes and attempts < max_attempts:
        attempts += 1
        applicable: list[tuple[str, str]] = []
        weights: list[float] = []
        for name, fn in rule_registry.items():
            updated = fn(text)
            if updated != text:
                applicable.append((name, updated))
                weights.append(float(weights_by_rule.get(name, 1.0)))
        if not applicable:
            break
        _, text = weighted_choice(applicable, weights)
        applied += 1
    return text


def process_prompts(
    prompts: list[str],
    custom_cfg: Optional[CustomConfig] = None,
    typos_range: tuple[int, int] = (1, 2),
    grammar_range: tuple[int, int] = (3, 4),
    typos_accents_drop_prob: float = 0.60,
) -> list[dict[str, str]]:
    custom_cfg = custom_cfg or CustomConfig()
    typo_ops = TypoOps()
    grammar = GrammarRules()
    rule_registry = grammar.registry()

    block2_order = [
        "habia_to_habian",
        "hemos_to_habemos",
        "homophones",
        "porque",
        "seseo_ceceo",
        "preterite_s",
        "drop_initial_h",
        "swap_bv",
    ]

    out: list[dict[str, str]] = []
    for prompt in prompts:
        n_typos_block1 = random.randint(typos_range[0], typos_range[1])
        prompt_typos = apply_typos_weighted_exact(
            prompt,
            n_typos=n_typos_block1,
            ops=typo_ops,
            typo_type_weights={
                "qwerty": 0.55,
                "omission": 0.4,
                "abbr": 0.4,
                "space_remove": 0.5,
            },
            vowel_delete_bias=0.8,
            abbr_q_weight=0.6,
            abbr_pq_weight=0.4,
        )
        prompt_typos = normalize_block1(
            prompt_typos, accents_drop_prob=typos_accents_drop_prob
        )

        n_grammar_block2 = random.randint(grammar_range[0], grammar_range[1])
        prompt_grammatical = normalize_block2(prompt)
        prompt_grammatical = apply_grammar_ordered(
            prompt_grammatical,
            n_changes=n_grammar_block2,
            rule_order=block2_order,
            rule_registry=rule_registry,
        )
        if re.search(r"\bhabia\b", prompt_grammatical, flags=re.IGNORECASE):
            prompt_grammatical = re.sub(
                r"\bhabia\b", "habian", prompt_grammatical, flags=re.IGNORECASE
            )

        prompt_custom = prompt
        if custom_cfg.n_grammar_changes > 0:
            prompt_custom = apply_grammar_weighted(
                prompt_custom,
                n_changes=custom_cfg.n_grammar_changes,
                rule_registry=rule_registry,
                weights_by_rule=custom_cfg.grammar_rule_weights,
            )
        if custom_cfg.n_typos > 0:
            prompt_custom = apply_typos_weighted_exact(
                prompt_custom,
                n_typos=custom_cfg.n_typos,
                ops=typo_ops,
                typo_type_weights=custom_cfg.typo_type_weights,
                vowel_delete_bias=custom_cfg.vowel_delete_bias,
                abbr_q_weight=custom_cfg.abbr_q_weight,
                abbr_pq_weight=custom_cfg.abbr_pq_weight,
            )
        prompt_custom = normalize_custom(prompt_custom, cfg=custom_cfg)

        out.append(
            {
                "prompt_original": prompt,
                "prompt_typos": prompt_typos,
                "prompt_grammatical_errors": prompt_grammatical,
                "prompt_custom": prompt_custom,
            }
        )
    return out
