"""Torch-free helpers for evaluation-only replays under the exact storage budget.

The functions in this module work in *level units* inside equal-size groups.
Under the group-wise exact budget used by the full-space experiments
(``preserve_equal_size_group_costs=True`` in ``src.compression_budget``), every
size group ``G_k`` must satisfy

    sum_{p in G_k, p active} b_p = b_ref * |G_k| + mu * |G_k \\ active|,

where ``mu = (scale_bits + zero_point_bits) / group_size`` is the metadata cost
per weight. Because all projections in a group have the same size, this is an
exact integer statement about bit-widths and can be checked without a model.

The module is used to plan replays (which crossed candidates need repair, which
are repair-free), to construct heuristic baseline candidates, and to summarise
replay results. The actual repair applied before evaluation is always the
production ``repair_quant_state_to_budget`` (see ``evo_exact_replay.py``), so the
replays use the same feasibility rule as the searches.
"""

from __future__ import annotations

import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LAYER_INDEX_RE = re.compile(r"layers\.(\d+)\.")

# Projection sizes of Mistral-7B-v0.3 (number of weights per matrix).
MISTRAL_7B_V03_PROJECTION_SIZES = {
    "self_attn.q_proj": 4096 * 4096,
    "self_attn.k_proj": 1024 * 4096,
    "self_attn.v_proj": 1024 * 4096,
    "self_attn.o_proj": 4096 * 4096,
    "mlp.gate_proj": 14336 * 4096,
    "mlp.up_proj": 14336 * 4096,
    "mlp.down_proj": 4096 * 14336,
}
MISTRAL_7B_V03_NUM_LAYERS = 32


def mistral_module_sizes(num_layers: int = MISTRAL_7B_V03_NUM_LAYERS) -> dict[str, int]:
    """Return ``{module_name: num_weights}`` for the 224 searched projections."""
    return {
        f"model.layers.{layer}.{suffix}": size
        for layer in range(num_layers)
        for suffix, size in MISTRAL_7B_V03_PROJECTION_SIZES.items()
    }


def layer_index(module_name: str) -> int:
    match = LAYER_INDEX_RE.search(module_name)
    if match is None:
        raise ValueError(f"Cannot determine layer index of {module_name!r}.")
    return int(match.group(1))


def module_kind(module_name: str) -> str:
    if ".self_attn." in module_name:
        return "attn"
    if ".mlp." in module_name:
        return "mlp"
    raise ValueError(f"Module {module_name!r} is neither attention nor MLP.")


def module_is_active(module_name: str, drop: Mapping[str, Sequence[bool]]) -> bool:
    return not bool(drop[module_kind(module_name)][layer_index(module_name)])


def size_groups(module_sizes: Mapping[str, int]) -> list[list[str]]:
    """Group modules by size, ordered by size and then by name (layer order)."""
    by_size: dict[int, list[str]] = {}
    for name, size in module_sizes.items():
        by_size.setdefault(int(size), []).append(name)

    def order(name: str) -> tuple[int, str]:
        return (layer_index(name), name)

    return [sorted(by_size[size], key=order) for size in sorted(by_size)]


def normalize_drop(drop: Mapping[str, Sequence[Any]]) -> dict[str, list[bool]]:
    attn = [bool(int(x)) if not isinstance(x, bool) else x for x in drop["attn"]]
    mlp = [bool(int(x)) if not isinstance(x, bool) else x for x in drop["mlp"]]
    if len(attn) != len(mlp):
        raise ValueError("Attention and MLP masks must have equal length.")
    return {"attn": attn, "mlp": mlp}


def load_final_candidate(path: str | Path) -> dict[str, Any]:
    """Load a ``final_candidate.json`` written by the search programs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "attention_mask" in data and "mlp_mask" in data:
        drop = normalize_drop({"attn": data["attention_mask"], "mlp": data["mlp_mask"]})
    else:
        raw = data.get("candidate_vector_raw") or {}
        source = raw.get("drop", raw)
        drop = normalize_drop(source)
    bits = data.get("bitwidth_by_module") or {}
    return {
        "drop": drop,
        "bits": {str(k): int(v) for k, v in bits.items()},
        "candidate_type": data.get("candidate_type"),
        "source": str(path),
    }


def metadata_bits_per_weight(group_size: int = 128, scale_bits: int = 16, zero_point_bits: int = 16) -> Fraction:
    return Fraction(scale_bits + zero_point_bits, group_size)


def required_level_sums(
    groups: Sequence[Sequence[str]],
    drop: Mapping[str, Sequence[bool]],
    reference_bitwidth: int = 3,
    mu: Fraction = Fraction(1, 4),
) -> list[int]:
    """Required active level sum of every size group (group-wise exact budget)."""
    sums = []
    for group in groups:
        inactive = sum(not module_is_active(name, drop) for name in group)
        value = reference_bitwidth * len(group) + mu * inactive
        if value.denominator != 1:
            raise ValueError(
                "No integer assignment satisfies the group-wise budget: "
                f"{inactive} removed projections in a group of {len(group)} "
                f"with metadata cost {mu} bits per weight."
            )
        sums.append(int(value))
    return sums


def active_level_sums(
    groups: Sequence[Sequence[str]],
    bits: Mapping[str, int],
    drop: Mapping[str, Sequence[bool]],
) -> list[int]:
    return [
        sum(int(bits[name]) for name in group if module_is_active(name, drop))
        for group in groups
    ]


def level_deficits(
    groups: Sequence[Sequence[str]],
    bits: Mapping[str, int],
    drop: Mapping[str, Sequence[bool]],
    reference_bitwidth: int = 3,
    mu: Fraction = Fraction(1, 4),
) -> list[int]:
    """Signed level correction a repair must apply per group (required - actual)."""
    required = required_level_sums(groups, drop, reference_bitwidth, mu)
    actual = active_level_sums(groups, bits, drop)
    return [r - a for r, a in zip(required, actual)]


def min_changed_genes(deficits: Sequence[int], bits_range: tuple[int, int] = (2, 6)) -> int:
    """Lower bound on bit-width genes a repair must change (max step per gene)."""
    max_step = bits_range[1] - bits_range[0]
    return sum(math.ceil(abs(delta) / max_step) for delta in deficits)


def hamming_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, int]:
    """Distance of Equation (4.1): mask entries plus all bit-width genes."""
    mask = sum(x != y for x, y in zip(a["drop"]["attn"], b["drop"]["attn"]))
    mask += sum(x != y for x, y in zip(a["drop"]["mlp"], b["drop"]["mlp"]))
    genes = sum(a["bits"][name] != b["bits"][name] for name in a["bits"])
    return {"mask": int(mask), "bits": int(genes), "total": int(mask + genes)}


def mask_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    """Jaccard overlap of the removed sublayers, per type and combined."""
    result = {}
    total_inter = total_union = 0
    for kind in ("attn", "mlp"):
        removed_a = {i for i, v in enumerate(a["drop"][kind]) if v}
        removed_b = {i for i, v in enumerate(b["drop"][kind]) if v}
        inter = len(removed_a & removed_b)
        union = len(removed_a | removed_b)
        result[f"{kind}_shared"] = inter
        result[f"{kind}_jaccard"] = inter / union if union else 1.0
        total_inter += inter
        total_union += union
    result["jaccard"] = total_inter / total_union if total_union else 1.0
    return result


def crossed_candidate(mask_source: Mapping[str, Any], bit_source: Mapping[str, Any]) -> dict[str, Any]:
    return {"drop": normalize_drop(mask_source["drop"]), "bits": dict(bit_source["bits"])}


def crossed_pair_report(
    label_a: str,
    cand_a: Mapping[str, Any],
    label_b: str,
    cand_b: Mapping[str, Any],
    groups: Sequence[Sequence[str]],
    reference_bitwidth: int = 3,
    mu: Fraction = Fraction(1, 4),
) -> dict[str, Any]:
    """Feasibility of the two crossed candidates (D_A, Q_B) and (D_B, Q_A)."""
    if sum(cand_a["drop"]["attn"]) != sum(cand_b["drop"]["attn"]) or sum(
        cand_a["drop"]["mlp"]
    ) != sum(cand_b["drop"]["mlp"]):
        raise ValueError(f"{label_a} and {label_b} remove different numbers of sublayers.")
    ab = level_deficits(groups, cand_b["bits"], cand_a["drop"], reference_bitwidth, mu)
    ba = level_deficits(groups, cand_a["bits"], cand_b["drop"], reference_bitwidth, mu)
    distance = hamming_distance(cand_a, cand_b)
    active_both_differ = sum(
        cand_a["bits"][name] != cand_b["bits"][name]
        for group in groups
        for name in group
        if module_is_active(name, cand_a["drop"]) and module_is_active(name, cand_b["drop"])
    )
    report = {
        "a": label_a,
        "b": label_b,
        "deficits_DA_QB": ab,
        "deficits_DB_QA": ba,
        "repair_free_DA_QB": all(d == 0 for d in ab),
        "repair_free_DB_QA": all(d == 0 for d in ba),
        "min_repair_genes_DA_QB": min_changed_genes(ab),
        "min_repair_genes_DB_QA": min_changed_genes(ba),
        "hamming_mask": distance["mask"],
        "hamming_bits": distance["bits"],
        "active_in_both_bits_differ": int(active_both_differ),
    }
    report["repair_free_pair"] = report["repair_free_DA_QB"] and report["repair_free_DB_QA"]
    report.update(mask_overlap(cand_a, cand_b))
    return report


# ---------------------------------------------------------------------------
# Heuristic baseline construction.
# ---------------------------------------------------------------------------

HEURISTIC_MASK_RULES = ("late_layer", "early_layer", "random", "late_layer_keep_last")


def heuristic_layer_indices(rule: str, num_layers: int, count: int, rng: random.Random) -> list[int]:
    """Indices removed by a simple rule.

    ``late_layer``, ``early_layer`` and ``random`` reproduce
    ``scripts/build_depth_baseline_config.choose_indices`` (without layer-0
    protection), the rules of the early depth-only baseline experiment.
    ``late_layer_keep_last`` removes the ``count`` layers immediately before the
    final decoder layer, which is kept.
    """
    candidates = list(range(num_layers))
    if count > len(candidates):
        raise ValueError("Cannot remove more sublayers than layers.")
    if rule == "random":
        return sorted(rng.sample(candidates, count))
    if rule == "late_layer":
        return candidates[-count:] if count else []
    if rule == "early_layer":
        return candidates[:count]
    if rule == "late_layer_keep_last":
        return candidates[-count - 1 : -1] if count else []
    raise ValueError(f"Unknown heuristic mask rule: {rule}")


def heuristic_mask(rule: str, num_layers: int, k_attn: int, k_mlp: int, seed: int = 0) -> dict[str, list[bool]]:
    """Build a mask; for ``random`` attention is drawn before MLP from one RNG."""
    rng = random.Random(seed)
    attn_ids = heuristic_layer_indices(rule, num_layers, k_attn, rng)
    mlp_ids = heuristic_layer_indices(rule, num_layers, k_mlp, rng)
    return {
        "attn": [i in attn_ids for i in range(num_layers)],
        "mlp": [i in mlp_ids for i in range(num_layers)],
    }


def score_mask(scores: Mapping[str, Sequence[float]], k_attn: int, k_mlp: int) -> dict[str, list[bool]]:
    """Remove the ``k`` sublayers of each type with the lowest score.

    Ties are broken by layer index (later layers removed first), so the result is
    deterministic.
    """
    mask = {}
    for kind, count in (("attn", k_attn), ("mlp", k_mlp)):
        values = list(scores[kind])
        order = sorted(range(len(values)), key=lambda i: (values[i], -i))
        removed = set(order[:count])
        mask[kind] = [i in removed for i in range(len(values))]
    return mask


def near_uniform_bits(
    groups: Sequence[Sequence[str]],
    drop: Mapping[str, Sequence[bool]],
    reference_bitwidth: int = 3,
    mu: Fraction = Fraction(1, 4),
    order: str = "spread",
    seed: int = 0,
    inactive_bitwidth: int | None = None,
) -> dict[str, int]:
    """Spend the group budget as evenly as possible over the active projections.

    Every active projection receives ``floor(B_k / a_k)`` bits and ``B_k mod a_k``
    of them one more bit, where ``B_k`` is the required active level sum and
    ``a_k`` the number of active projections of group ``k``. ``order='spread'``
    gives the extra bits to active projections evenly spaced in layer order;
    ``order='random'`` draws them with ``seed``. Inactive genes are set to
    ``inactive_bitwidth`` (default: the reference bit-width); they do not affect
    the model or the cost.
    """
    required = required_level_sums(groups, drop, reference_bitwidth, mu)
    rng = random.Random(seed)
    bits: dict[str, int] = {}
    fill = reference_bitwidth if inactive_bitwidth is None else inactive_bitwidth
    for group, total in zip(groups, required):
        active = [name for name in group if module_is_active(name, drop)]
        for name in group:
            bits[name] = fill
        if not active:
            continue
        base, extra = divmod(total, len(active))
        for name in active:
            bits[name] = base
        if order == "spread":
            chosen = [active[(i * len(active)) // extra] for i in range(extra)] if extra else []
        elif order == "random":
            chosen = rng.sample(active, extra)
        else:
            raise ValueError(f"Unknown order: {order}")
        for name in chosen:
            bits[name] += 1
    return bits


def validate_group_budget(
    groups: Sequence[Sequence[str]],
    bits: Mapping[str, int],
    drop: Mapping[str, Sequence[bool]],
    reference_bitwidth: int = 3,
    mu: Fraction = Fraction(1, 4),
    levels: Iterable[int] = (2, 3, 4, 5, 6),
) -> None:
    allowed = set(levels)
    bad = [name for group in groups for name in group if bits[name] not in allowed]
    if bad:
        raise ValueError(f"Bit-widths outside the database levels: {bad[:5]}")
    deficits = level_deficits(groups, bits, drop, reference_bitwidth, mu)
    if any(deficits):
        raise ValueError(f"Group-wise budget violated, deficits {deficits}.")


# ---------------------------------------------------------------------------
# Interaction contrast on replay metrics.
# ---------------------------------------------------------------------------


def interaction_terms(j_aa: float, j_bb: float, j_ab: float, j_ba: float) -> dict[str, float]:
    """Contrast of Equation (3.10) for a loss-like metric (lower is better).

    ``j_ab`` is the metric of (D_A, Q_B) and ``j_ba`` that of (D_B, Q_A).
    Returns ``I = (j_aa + j_bb) - (j_ab + j_ba)`` and the per-mask changes
    ``delta_a = j_ab - j_aa`` and ``delta_b = j_ba - j_bb``, so that
    ``I = -(delta_a + delta_b)``.
    """
    delta_a = j_ab - j_aa
    delta_b = j_ba - j_bb
    return {"I": (j_aa + j_bb) - (j_ab + j_ba), "delta_a": delta_a, "delta_b": delta_b}


def log_ppl(ppl: float) -> float:
    """Mean negative log-likelihood per token from a perplexity."""
    return math.log(ppl)


# ---------------------------------------------------------------------------
# Replay job construction (pure part).
# ---------------------------------------------------------------------------

FILL_MODES = ("donor", "owner")
REPAIR_SCOPES = ("all", "exclusive", "none")


def exclusive_sublayers(mask_drop: Mapping[str, Sequence[bool]], donor_drop: Mapping[str, Sequence[bool]]) -> dict[str, list[bool]]:
    """Sublayers active under ``mask_drop`` but removed under ``donor_drop``."""
    return {
        kind: [(not m) and bool(d) for m, d in zip(mask_drop[kind], donor_drop[kind])]
        for kind in ("attn", "mlp")
    }


def eligibility_drop_state(mask_drop: Mapping[str, Sequence[bool]], donor_drop: Mapping[str, Sequence[bool]]) -> dict[str, list[bool]]:
    """Drop state under which only the exclusive sublayers count as active.

    Passing this state to the production repair DP restricts the repair to the
    projections of sublayers that are active in the evaluated mask but were
    removed in the bit-width donor.
    """
    exclusive = exclusive_sublayers(mask_drop, donor_drop)
    return {kind: [not value for value in exclusive[kind]] for kind in ("attn", "mlp")}


def fill_bits(
    mask_owner: Mapping[str, Any] | None,
    donor: Mapping[str, Any],
    mask_drop: Mapping[str, Sequence[bool]],
    groups: Sequence[Sequence[str]],
    fill: str,
) -> dict[str, int]:
    """Bit-widths of a crossed candidate before repair.

    ``donor``: the donor's bit-widths for every gene, i.e. (D_A, Q_B).
    ``owner``: the donor's bit-widths on projections active under both masks and
    the mask owner's own bit-widths on projections active only under the mask
    (and on inactive genes). Exchanging only the shared genes keeps the
    contrast zero under per-projection additivity without any repair.
    """
    if fill not in FILL_MODES:
        raise ValueError(f"Unknown fill mode: {fill}")
    bits = dict(donor["bits"])
    if fill == "owner":
        if mask_owner is None:
            raise ValueError("Owner fill requires a candidate that owns the mask.")
        for group in groups:
            for name in group:
                shared = module_is_active(name, mask_drop) and module_is_active(name, donor["drop"])
                if not shared:
                    bits[name] = int(mask_owner["bits"][name])
    return bits


def shift_to_budget(
    bits: Mapping[str, int],
    groups: Sequence[Sequence[str]],
    drop: Mapping[str, Sequence[bool]],
    reference_bitwidth: int = 3,
    mu: Fraction = Fraction(1, 4),
    levels: Sequence[int] = (2, 3, 4, 5, 6),
) -> dict[str, int]:
    """Spend a group's deficit as evenly as possible on its active projections.

    Each active projection is moved by the same number of levels (clipped to the
    available levels), and the remainder is spread evenly in layer order. Any
    residual caused by clipping is left for the repair step. Used to compose an
    independently searched mask with a quantization-only profile: the profile's
    relative allocation is kept while the storage freed by removal is spent.
    """
    lo, hi = min(levels), max(levels)
    out = dict(bits)
    deficits = level_deficits(groups, out, drop, reference_bitwidth, mu)
    for group, delta in zip(groups, deficits):
        active = [name for name in group if module_is_active(name, drop)]
        if not active or delta == 0:
            continue
        sign = 1 if delta > 0 else -1
        base, extra = divmod(abs(delta), len(active))
        chosen = {active[(i * len(active)) // extra] for i in range(extra)} if extra else set()
        for name in active:
            step = base + (1 if name in chosen else 0)
            out[name] = max(lo, min(hi, out[name] + sign * step))
    return out


def build_replay_bits(
    job: Mapping[str, Any],
    mask_drop: Mapping[str, Sequence[bool]],
    sources: Mapping[str, Mapping[str, Any]],
    groups: Sequence[Sequence[str]],
    reference_bitwidth: int = 3,
    mu: Fraction = Fraction(1, 4),
) -> tuple[dict[str, int], dict[str, list[bool]] | None]:
    """Pre-repair bit-widths of a job and the repair eligibility drop state.

    Returns ``(bits, eligibility)``; ``eligibility`` is ``None`` when every
    active projection may be repaired (production repair).
    """
    spec = job["bits"]
    scope = (job.get("repair") or {}).get("scope", "all")
    if scope not in REPAIR_SCOPES:
        raise ValueError(f"Unknown repair scope: {scope}")
    if "rule" in spec:
        if spec["rule"] == "near_uniform":
            bits = near_uniform_bits(
                groups, mask_drop, reference_bitwidth, mu,
                order=spec.get("order", "spread"), seed=int(spec.get("seed", 0)),
            )
            return bits, None
        if spec["rule"] == "uniform":
            value = int(spec["bitwidth"])
            return {name: value for group in groups for name in group}, None
        raise ValueError(f"Unknown bit rule: {spec['rule']}")
    donor = sources[spec["source"]]
    owner_label = (job.get("mask") or {}).get("source")
    owner = sources.get(owner_label) if owner_label else None
    bits = fill_bits(owner, donor, mask_drop, groups, spec.get("fill", "donor"))
    if spec.get("transform") == "shift":
        bits = shift_to_budget(bits, groups, mask_drop, reference_bitwidth, mu)
    elif spec.get("transform") not in (None, "none"):
        raise ValueError(f"Unknown transform: {spec['transform']}")
    eligibility = None
    if scope == "exclusive":
        eligibility = eligibility_drop_state(mask_drop, donor["drop"])
    return bits, eligibility


def repair_capacity(
    groups: Sequence[Sequence[str]],
    bits: Mapping[str, int],
    eligible_drop: Mapping[str, Sequence[bool]],
    deficits: Sequence[int],
    levels: Sequence[int] = (2, 3, 4, 5, 6),
) -> list[bool]:
    """Whether each group's deficit can be absorbed by the eligible genes.

    The production repair changes every eligible projection at most once and in
    the direction of the deficit; with contiguous levels, a deficit is reachable
    exactly when the total room in that direction is large enough.
    """
    lo, hi = min(levels), max(levels)
    feasible = []
    for group, delta in zip(groups, deficits):
        eligible = [name for name in group if module_is_active(name, eligible_drop)]
        if delta > 0:
            room = sum(hi - bits[name] for name in eligible)
        elif delta < 0:
            room = sum(bits[name] - lo for name in eligible)
        else:
            room = 0
        feasible.append(abs(delta) <= room)
    return feasible
