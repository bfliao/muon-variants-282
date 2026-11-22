"""Run identity is frozen: adding an optimizer (or a hashed field) must never
move an existing run.

Every run on disk is keyed by ``runner.config_hash``; the results are tracked
in git and the whole analysis is built on them. Lion-Muon added one hashed
field ("lionmuon"), which ``runner.canonical`` omits when absent/None exactly
so that the checks below hold:

1.  every tracked ``results/<study>/<hash>.json`` still hashes to its own
    filename (and to its stored "hash" field);
2.  the planned job set of the five pre-Lion-Muon optimizers is byte-for-byte
    the set the code produced before Lion-Muon existed. The digests in
    ``PLANNED_BEFORE_LIONMUON`` were computed by running commit 8b0f14f
    (the last commit without Lion-Muon) in a git worktree:
    sha256 of the comma-joined sorted hashes of
    ``studies.planned_jobs(study)``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from muonarch import runner, studies

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BEFORE_LIONMUON = ("adamw", "muon", "moonlight", "manifold", "muonall")

#: study -> (number of planned jobs, digest) at commit 8b0f14f.
PLANNED_BEFORE_LIONMUON = {
    "mlp128": (101, "67faf6c5408ba42024109705d3a8674dd26eb2d3b444e4a0c2514080b4c64423"),
    "mlp256": (161, "65ecdbb3637c6e9661100bb4e5536a490c6c947eb259960b6e5909105a71c26d"),
    "mlp512": (161, "c957e5e7969e99b203b37d1c8e799afab34e0227401e29b2677c023a174dc508"),
    "dense": (101, "7d4dbc1a78b11f06e6d8f6f48efa1d7f835be480372f225a061ff6d46dc545ab"),
    "moe": (94, "d71a17e5621fde42d69f074ec87781c02230c4b9dee621db2432ef9a7e2f7ae1"),
}


def _result_files() -> list[Path]:
    return sorted(RESULTS.glob("*/*.json"))


def test_every_result_file_still_hashes_to_its_filename():
    files = _result_files()
    if not files:
        pytest.skip("no tracked results in this checkout")
    wrong = []
    for path in files:
        res = json.loads(path.read_text())
        h = runner.config_hash(res["config"])
        if h != path.stem or res.get("hash", h) != h \
                or res["config"]["study"] != path.parent.name:
            wrong.append(f"{path.parent.name}/{path.name} -> {h}")
    assert not wrong, f"{len(wrong)} of {len(files)} results moved: {wrong[:5]}"


def test_existing_results_carry_no_lionmuon_identity():
    """The canonical (hashed) view of a pre-Lion-Muon run has no "lionmuon"
    key at all — neither a value nor a null."""
    files = [p for p in _result_files()
             if json.loads(p.read_text())["config"]["optimizer"] in BEFORE_LIONMUON]
    if not files:
        pytest.skip("no tracked results in this checkout")
    for path in files:
        cfg = json.loads(path.read_text())["config"]
        assert "lionmuon" not in runner.canonical(cfg), path.name


@pytest.mark.parametrize("study", sorted(PLANNED_BEFORE_LIONMUON))
def test_planned_jobs_of_existing_optimizers_unchanged(study):
    hashes = sorted(runner.config_hash(c) for c in studies.planned_jobs(study)
                    if c["optimizer"] in BEFORE_LIONMUON)
    n, digest = PLANNED_BEFORE_LIONMUON[study]
    assert len(hashes) == len(set(hashes)) == n
    assert hashlib.sha256(",".join(hashes).encode()).hexdigest() == digest


@pytest.mark.parametrize("study", sorted(PLANNED_BEFORE_LIONMUON))
def test_generated_configs_of_existing_optimizers_have_no_lionmuon_key(study):
    """Not just the hash: the generated config dict itself is what it was
    (stored result configs stay dict-equal to regenerated ones)."""
    for cfg in studies.planned_jobs(study):
        if cfg["optimizer"] in BEFORE_LIONMUON:
            assert "lionmuon" not in cfg
        else:
            assert cfg["optimizer"] == "lionmuon"
            assert set(cfg["lionmuon"]) == {"period", "alpha", "beta1", "beta2"}


def test_null_or_absent_lionmuon_is_not_hashed_but_a_value_is():
    cfg = next(c for c in studies.planned_jobs("mlp128") if c["optimizer"] == "muon")
    h = runner.config_hash(cfg)
    assert runner.config_hash(dict(cfg, lionmuon=None)) == h
    assert set(runner.canonical(cfg)) == (
        set(runner.HASHED_FIELDS) - set(runner.OMITTED_WHEN_NULL))
    lm = {"period": 5, "alpha": 30.0, "beta1": 0.9, "beta2": 0.99}
    assert runner.config_hash(dict(cfg, lionmuon=lm)) != h
    assert runner.canonical(dict(cfg, lionmuon=lm))["lionmuon"] == lm


def test_omitted_when_null_covers_exactly_the_post_freeze_fields():
    """The 31 fields frozen by the identity review are always emitted; only
    fields added after results existed may be omitted when null."""
    assert runner.OMITTED_WHEN_NULL == frozenset({"lionmuon"})
    assert runner.OMITTED_WHEN_NULL <= runner.HASHED_FIELDS
    assert len(runner.HASHED_FIELDS - runner.OMITTED_WHEN_NULL) == 31
