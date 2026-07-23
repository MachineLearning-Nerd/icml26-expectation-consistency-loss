#!/usr/bin/env python3
"""Exact certificates for ECL compatibility with three calibration paradigms.

This is a clean-room, standard-library-only reproduction of the population
identities in Theorem 3.1 / Eq. (4), Theorems D.1-D.2 / Eqs. (15)-(23),
and the hard-group versions of Eqs. (24)-(29) in Dong et al. (2026).

Unlike the earlier random six-state floating-point check, every probability in
this module is a ``fractions.Fraction``.  The executed certificate therefore
compares exact rational values, separately for:

* canonical calibration: group by the complete confidence vector S and observe
  the complete posterior vector P(Y|X);
* class-wise calibration: for every class k, group by S_k and observe
  P(Y_k=1|X);
* top-label calibration: group by max_k S_k and observe the event posterior
  P(Y*=Yhat|X) = P(Y=Yhat(X)|X), not max_k P(Y=k|X).

The fixtures contain a fixed three-class classifier, twelve covariate atoms,
positive source/target mass on every atom, and a strict covariate shift.  One
fixture is exactly calibrated and has zero ECL in all three paradigms; another
has a nonzero, exactly detected ECL in all three.  Negative controls deliberately
use wrong groupings and break the shared-posterior assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Hashable, Mapping, Sequence


Scalar = Fraction
Vector = tuple[Fraction, ...]
Value = Scalar | Vector
Group = Hashable

PAPER_SHA256 = "fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab"
OFFICIAL_REPOSITORY = "https://github.com/NeuroDong/ECL"
OFFICIAL_COMMIT = "aae77f890f1e4ebc13dad135b5e29758d98d318d"
OFFICIAL_LOSSES_SHA256 = "1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754"


def _q(numerator: int, denominator: int = 1) -> Fraction:
    return Fraction(numerator, denominator)


def _add(a: Value, b: Value) -> Value:
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        return tuple(x + y for x, y in zip(a, b, strict=True))
    assert isinstance(b, Fraction)
    return a + b


def _sub(a: Value, b: Value) -> Value:
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        return tuple(x - y for x, y in zip(a, b, strict=True))
    assert isinstance(b, Fraction)
    return a - b


def _scale(a: Value, weight: Fraction) -> Value:
    if isinstance(a, tuple):
        return tuple(weight * x for x in a)
    return weight * a


def _zero_like(a: Value) -> Value:
    if isinstance(a, tuple):
        return tuple(Fraction(0) for _ in a)
    return Fraction(0)


def _l1(a: Value) -> Fraction:
    if isinstance(a, tuple):
        return sum((abs(x) for x in a), start=Fraction(0))
    return abs(a)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _serialize(value: object) -> object:
    """Turn exact objects into stable, JSON-safe strings/containers."""
    if isinstance(value, Fraction):
        return _fraction_text(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Atom:
    name: str
    score: Vector
    posterior: Vector

    def __post_init__(self) -> None:
        if len(self.score) != 3 or len(self.posterior) != 3:
            raise ValueError("The certificate is intentionally three-class.")
        if sum(self.score, start=Fraction(0)) != 1:
            raise ValueError(f"{self.name}: score is not on the simplex")
        if sum(self.posterior, start=Fraction(0)) != 1:
            raise ValueError(f"{self.name}: posterior is not on the simplex")
        if any(value < 0 or value > 1 for value in (*self.score, *self.posterior)):
            raise ValueError(f"{self.name}: probability outside [0,1]")
        maximum = max(self.score)
        if sum(value == maximum for value in self.score) != 1:
            raise ValueError(f"{self.name}: top-label prediction must be unique")

    @property
    def predicted_class(self) -> int:
        return max(range(len(self.score)), key=self.score.__getitem__)

    @property
    def top_confidence(self) -> Fraction:
        return self.score[self.predicted_class]

    @property
    def correct_event_posterior(self) -> Fraction:
        return self.posterior[self.predicted_class]


@dataclass(frozen=True)
class ExactFixture:
    name: str
    atoms: tuple[Atom, ...]
    source_mass: tuple[Fraction, ...]
    target_mass: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        n = len(self.atoms)
        if len(self.source_mass) != n or len(self.target_mass) != n:
            raise ValueError("Each atom must have one source and target mass.")
        for domain, masses in (("source", self.source_mass), ("target", self.target_mass)):
            if any(value <= 0 for value in masses):
                raise ValueError(f"{domain} masses must have common positive support")
            if sum(masses, start=Fraction(0)) != 1:
                raise ValueError(f"{domain} masses must sum to one")
        if self.source_mass == self.target_mass:
            raise ValueError("The certificate must contain a strict covariate shift.")


def make_fixture(*, calibrated: bool) -> ExactFixture:
    """Return an exact 12-atom covariate-shift fixture.

    Every confidence vector occurs twice.  Source pair weights are (1,3)/24
    and target pair weights are (3,1)/24, so the full support is shared but
    P_s(X) != P_t(X).  For the calibrated fixture q(x)=S(x).  For the
    diagnostic fixture duplicate posteriors are S +/- (1/12,-1/24,-1/24),
    making every correct grouping detect a nonzero domain gap.
    """
    patterns: tuple[Vector, ...] = (
        (_q(1, 2), _q(1, 3), _q(1, 6)),
        (_q(1, 2), _q(1, 6), _q(1, 3)),
        (_q(1, 3), _q(1, 2), _q(1, 6)),
        (_q(1, 6), _q(1, 3), _q(1, 2)),
        (_q(3, 5), _q(1, 5), _q(1, 5)),
        (_q(1, 5), _q(3, 5), _q(1, 5)),
    )
    delta = (_q(1, 12), -_q(1, 24), -_q(1, 24))
    atoms: list[Atom] = []
    for pattern_index, score in enumerate(patterns):
        if calibrated:
            pair = (score, score)
        else:
            pair = (
                tuple(value + change for value, change in zip(score, delta, strict=True)),
                tuple(value - change for value, change in zip(score, delta, strict=True)),
            )
        for duplicate_index, posterior in enumerate(pair):
            atoms.append(
                Atom(
                    name=f"x{pattern_index}_{duplicate_index}",
                    score=score,
                    posterior=posterior,
                )
            )

    source_pair = (_q(1, 24), _q(3, 24))
    target_pair = (_q(3, 24), _q(1, 24))
    return ExactFixture(
        name="calibrated-zero-loss" if calibrated else "diagnostic-nonzero-loss",
        atoms=tuple(atoms),
        source_mass=source_pair * len(patterns),
        target_mass=target_pair * len(patterns),
    )


def _conditional_expectations(
    fixture: ExactFixture,
    masses: Sequence[Fraction],
    group_of: Callable[[Atom], Group],
    value_of: Callable[[Atom], Value],
) -> dict[Group, Value]:
    """Compute E[value(X)|group(X)] directly via P(X|group)."""
    totals: dict[Group, Fraction] = defaultdict(Fraction)
    numerators: dict[Group, Value] = {}
    for atom, mass in zip(fixture.atoms, masses, strict=True):
        group = group_of(atom)
        value = value_of(atom)
        totals[group] += mass
        if group not in numerators:
            numerators[group] = _zero_like(value)
        numerators[group] = _add(numerators[group], _scale(value, mass))
    return {
        group: _scale(numerators[group], Fraction(1, total))
        for group, total in totals.items()
    }


def _joint_conditionals(
    fixture: ExactFixture,
    masses: Sequence[Fraction],
    group_of: Callable[[Atom], Group],
    event_probability_of: Callable[[Atom], Value],
) -> dict[Group, Value]:
    """Independently build P(observation, group)/P(group).

    This is the direct calibration side of Eqs. (13), (15), and (17), kept
    separate from the P(X|summary) expectation implementation above.
    """
    group_probability: dict[Group, Fraction] = defaultdict(Fraction)
    joint_probability: dict[Group, Value] = {}
    for index in range(len(fixture.atoms)):
        atom = fixture.atoms[index]
        group = group_of(atom)
        mass = masses[index]
        event_probability = event_probability_of(atom)
        group_probability[group] += mass
        if group not in joint_probability:
            joint_probability[group] = _zero_like(event_probability)
        joint_probability[group] = _add(
            joint_probability[group], _scale(event_probability, mass)
        )
    return {
        group: _scale(joint_probability[group], Fraction(1, probability))
        for group, probability in group_probability.items()
    }


def _group_masses(
    fixture: ExactFixture,
    masses: Sequence[Fraction],
    group_of: Callable[[Atom], Group],
) -> dict[Group, Fraction]:
    result: dict[Group, Fraction] = defaultdict(Fraction)
    for atom, mass in zip(fixture.atoms, masses, strict=True):
        result[group_of(atom)] += mass
    return dict(result)


def _residual_map(source: Mapping[Group, Value], target: Mapping[Group, Value]) -> dict[Group, Value]:
    if source.keys() != target.keys():
        raise AssertionError("Source and target summary support must agree.")
    return {group: _sub(source[group], target[group]) for group in source}


def _loss(
    residuals: Mapping[Group, Value],
    target_group_mass: Mapping[Group, Fraction],
) -> Fraction:
    return sum(
        (target_group_mass[group] * _l1(value) for group, value in residuals.items()),
        start=Fraction(0),
    )


def canonical_certificate(fixture: ExactFixture) -> dict[str, object]:
    group = lambda atom: atom.score
    observe = lambda atom: atom.posterior
    source_ecl = _conditional_expectations(fixture, fixture.source_mass, group, observe)
    target_ecl = _conditional_expectations(fixture, fixture.target_mass, group, observe)
    source_calibration = _joint_conditionals(fixture, fixture.source_mass, group, observe)
    target_calibration = _joint_conditionals(fixture, fixture.target_mass, group, observe)
    ecl_residual = _residual_map(source_ecl, target_ecl)
    calibration_gap = _residual_map(source_calibration, target_calibration)
    target_mass = _group_masses(fixture, fixture.target_mass, group)
    return {
        "grouping": "complete three-class confidence vector S",
        "observation": "complete posterior vector P(Y|X)",
        "paper_anchors": ["Theorem 3.1", "Eq. 4"],
        "summary_state_count": len(ecl_residual),
        "ecl_residual": ecl_residual,
        "direct_calibration_gap": calibration_gap,
        "identity_exact": ecl_residual == calibration_gap,
        "ecl_l1_loss": _loss(ecl_residual, target_mass),
    }


def classwise_certificate(fixture: ExactFixture) -> dict[str, object]:
    all_residuals: dict[str, object] = {}
    identity_exact = True
    total_loss = Fraction(0)
    for class_index in range(3):
        group = lambda atom, k=class_index: atom.score[k]
        observe = lambda atom, k=class_index: atom.posterior[k]
        source_ecl = _conditional_expectations(fixture, fixture.source_mass, group, observe)
        target_ecl = _conditional_expectations(fixture, fixture.target_mass, group, observe)
        source_calibration = _joint_conditionals(fixture, fixture.source_mass, group, observe)
        target_calibration = _joint_conditionals(fixture, fixture.target_mass, group, observe)
        ecl_residual = _residual_map(source_ecl, target_ecl)
        calibration_gap = _residual_map(source_calibration, target_calibration)
        target_mass = _group_masses(fixture, fixture.target_mass, group)
        class_loss = _loss(ecl_residual, target_mass)
        identity_exact = identity_exact and ecl_residual == calibration_gap
        total_loss += class_loss
        all_residuals[f"class_{class_index}"] = {
            "summary_state_count": len(ecl_residual),
            "ecl_residual": ecl_residual,
            "direct_calibration_gap": calibration_gap,
            "identity_exact": ecl_residual == calibration_gap,
            "ecl_l1_loss": class_loss,
        }
    return {
        "grouping": "each scalar coordinate S_k, separately for every k",
        "observation": "scalar event posterior P(Y_k=1|X)",
        "paper_anchors": ["Theorem D.2", "Eqs. 17-19", "Eqs. 22-23", "Eqs. 26-29"],
        "classes_executed": 3,
        "per_class": all_residuals,
        "identity_exact": identity_exact,
        "ecl_l1_loss": total_loss,
    }


def toplabel_certificate(fixture: ExactFixture) -> dict[str, object]:
    group = lambda atom: atom.top_confidence
    observe = lambda atom: atom.correct_event_posterior
    source_ecl = _conditional_expectations(fixture, fixture.source_mass, group, observe)
    target_ecl = _conditional_expectations(fixture, fixture.target_mass, group, observe)
    source_calibration = _joint_conditionals(fixture, fixture.source_mass, group, observe)
    target_calibration = _joint_conditionals(fixture, fixture.target_mass, group, observe)
    ecl_residual = _residual_map(source_ecl, target_ecl)
    calibration_gap = _residual_map(source_calibration, target_calibration)
    target_mass = _group_masses(fixture, fixture.target_mass, group)
    return {
        "grouping": "top confidence S_hat=max_k S_k",
        "observation": "correctness-event posterior P(Y*=Y_hat|X)=P(Y=Y_hat(X)|X)",
        "paper_anchors": ["Theorem D.1", "Eqs. 15-16", "Eqs. 20-21", "Eqs. 24-25"],
        "summary_state_count": len(ecl_residual),
        "predicted_classes_exercised": sorted({atom.predicted_class for atom in fixture.atoms}),
        "ecl_residual": ecl_residual,
        "direct_calibration_gap": calibration_gap,
        "identity_exact": ecl_residual == calibration_gap,
        "ecl_l1_loss": _loss(ecl_residual, target_mass),
    }


def _replace_posteriors(fixture: ExactFixture, target_posteriors: Sequence[Vector]) -> ExactFixture:
    """Helper only for representing a target-domain posterior in a control."""
    if len(target_posteriors) != len(fixture.atoms):
        raise ValueError("One target posterior is required for every atom.")
    return ExactFixture(
        name=f"{fixture.name}-target-posterior",
        atoms=tuple(
            Atom(atom.name, atom.score, posterior)
            for atom, posterior in zip(fixture.atoms, target_posteriors, strict=True)
        ),
        source_mass=fixture.source_mass,
        target_mass=fixture.target_mass,
    )


def _make_grouping_control_fixture() -> ExactFixture:
    """Exact fixture whose aggregate value changes under wrong partitions."""
    calibrated = make_fixture(calibrated=True)
    delta = (_q(1, 24), -_q(1, 48), -_q(1, 48))
    # These signs make both the canonical max(S) merge and the class-1/S_0
    # partition yield a provably different rational loss from the correct one.
    signs = (1, 1, 1, -1, 1, 1)
    atoms: list[Atom] = []
    for pattern_index, sign in enumerate(signs):
        score = calibrated.atoms[2 * pattern_index].score
        signed_delta = tuple(sign * value for value in delta)
        for duplicate_index, posterior_sign in enumerate((1, -1)):
            posterior = tuple(
                value + posterior_sign * change
                for value, change in zip(score, signed_delta, strict=True)
            )
            atoms.append(Atom(f"control_{pattern_index}_{duplicate_index}", score, posterior))
    return ExactFixture(
        "wrong-grouping-control",
        tuple(atoms),
        calibrated.source_mass,
        calibrated.target_mass,
    )


def negative_controls(fixture: ExactFixture) -> dict[str, object]:
    """Controls that exact checks must reject.

    1. Merge canonical states by top confidence instead of the full vector.
    2. For class-wise k=1, group by S_0 instead of S_1.
    3. For top-label, use max_k P(Y=k|X) instead of P(Y=Yhat(X)|X).
    4. Break covariate shift by changing P_t(Y|X), while an invalid EC path
       continues to reuse the source posterior in the target expectation.
    """
    grouping_fixture = _make_grouping_control_fixture()
    correct_canonical_certificate = canonical_certificate(grouping_fixture)
    correct_canonical = correct_canonical_certificate["ecl_l1_loss"]
    wrong_canonical_group = lambda atom: atom.top_confidence
    source = _conditional_expectations(
        grouping_fixture,
        grouping_fixture.source_mass,
        wrong_canonical_group,
        lambda atom: atom.posterior,
    )
    target = _conditional_expectations(
        grouping_fixture,
        grouping_fixture.target_mass,
        wrong_canonical_group,
        lambda atom: atom.posterior,
    )
    wrong_canonical_loss = _loss(
        _residual_map(source, target),
        _group_masses(grouping_fixture, grouping_fixture.target_mass, wrong_canonical_group),
    )

    correct_canonical_groups = tuple(atom.score for atom in grouping_fixture.atoms)
    wrong_canonical_groups = tuple(atom.top_confidence for atom in grouping_fixture.atoms)
    canonical_partition_mismatch = (
        len(set(correct_canonical_groups)) != len(set(wrong_canonical_groups))
        and any(
            correct_canonical_groups[i] != correct_canonical_groups[j]
            and wrong_canonical_groups[i] == wrong_canonical_groups[j]
            for i in range(len(grouping_fixture.atoms))
            for j in range(i + 1, len(grouping_fixture.atoms))
        )
    )

    correct_classwise = classwise_certificate(grouping_fixture)
    correct_class_1_loss = correct_classwise["per_class"]["class_1"]["ecl_l1_loss"]
    wrong_classwise_group = lambda atom: atom.score[0]
    class_1_observe = lambda atom: atom.posterior[1]
    source = _conditional_expectations(
        grouping_fixture,
        grouping_fixture.source_mass,
        wrong_classwise_group,
        class_1_observe,
    )
    target = _conditional_expectations(
        grouping_fixture,
        grouping_fixture.target_mass,
        wrong_classwise_group,
        class_1_observe,
    )
    wrong_class_1_loss = _loss(
        _residual_map(source, target),
        _group_masses(grouping_fixture, grouping_fixture.target_mass, wrong_classwise_group),
    )

    correct_class_1_groups = tuple(atom.score[1] for atom in grouping_fixture.atoms)
    wrong_class_1_groups = tuple(atom.score[0] for atom in grouping_fixture.atoms)
    classwise_partition_mismatch = any(
        (correct_class_1_groups[i] == correct_class_1_groups[j])
        != (wrong_class_1_groups[i] == wrong_class_1_groups[j])
        for i in range(len(grouping_fixture.atoms))
        for j in range(i + 1, len(grouping_fixture.atoms))
    )

    correct_top = toplabel_certificate(fixture)["ecl_l1_loss"]
    top_group = lambda atom: atom.top_confidence
    wrong_top_observe = lambda atom: max(atom.posterior)
    source = _conditional_expectations(fixture, fixture.source_mass, top_group, wrong_top_observe)
    target = _conditional_expectations(fixture, fixture.target_mass, top_group, wrong_top_observe)
    wrong_top_loss = _loss(
        _residual_map(source, target),
        _group_masses(fixture, fixture.target_mass, top_group),
    )
    event_vs_max_disagreement_atoms = [
        atom.name
        for atom in fixture.atoms
        if atom.correct_event_posterior != max(atom.posterior)
    ]
    # Dedicated exact semantic counterexample: the fixed classifier predicts
    # class 0 from S, while the posterior's most likely class is class 1.
    # Top-label correctness is therefore q_0=1/5, not max(q)=7/10.
    top_event_counterexample = Atom(
        "top-event-counterexample",
        (_q(3, 5), _q(1, 5), _q(1, 5)),
        (_q(1, 5), _q(7, 10), _q(1, 10)),
    )
    event_semantics_rejected = (
        top_event_counterexample.correct_event_posterior == _q(1, 5)
        and max(top_event_counterexample.posterior) == _q(7, 10)
        and top_event_counterexample.correct_event_posterior
        != max(top_event_counterexample.posterior)
    )

    posterior_change = (_q(1, 48), -_q(1, 96), -_q(1, 96))
    target_posteriors: list[Vector] = []
    for atom in fixture.atoms:
        # The same nonzero target-only perturbation keeps every posterior on the
        # simplex but violates P_s(Y|X)=P_t(Y|X).
        target_posteriors.append(
            tuple(
                probability + change
                for probability, change in zip(atom.posterior, posterior_change, strict=True)
            )
        )
    target_fixture = _replace_posteriors(fixture, target_posteriors)

    def broken_identity(
        group_of: Callable[[Atom], Group],
        source_observe: Callable[[Atom], Value],
        target_observe: Callable[[Atom], Value],
    ) -> bool:
        # Invalid theorem-side calculation reuses q_s on both domains.
        invalid_ec = _residual_map(
            _conditional_expectations(fixture, fixture.source_mass, group_of, source_observe),
            _conditional_expectations(fixture, fixture.target_mass, group_of, source_observe),
        )
        # True calibration gap uses q_s in source and q_t in target.
        true_gap = _residual_map(
            _joint_conditionals(fixture, fixture.source_mass, group_of, source_observe),
            _joint_conditionals(target_fixture, fixture.target_mass, group_of, target_observe),
        )
        return invalid_ec != true_gap

    broken_canonical = broken_identity(
        lambda atom: atom.score,
        lambda atom: atom.posterior,
        lambda atom: atom.posterior,
    )
    broken_classwise = all(
        broken_identity(
            lambda atom, k=k: atom.score[k],
            lambda atom, k=k: atom.posterior[k],
            lambda atom, k=k: atom.posterior[k],
        )
        for k in range(3)
    )
    broken_top = broken_identity(
        lambda atom: atom.top_confidence,
        lambda atom: atom.correct_event_posterior,
        lambda atom: atom.correct_event_posterior,
    )

    controls = {
        "canonical_wrong_grouping": {
            "mistake": "grouping by max(S) instead of the complete vector S",
            "correct_loss": correct_canonical,
            "wrong_loss": wrong_canonical_loss,
            "correct_state_count": len(set(correct_canonical_groups)),
            "wrong_state_count": len(set(wrong_canonical_groups)),
            "numeric_mismatch": correct_canonical != wrong_canonical_loss,
            "partition_mismatch": canonical_partition_mismatch,
            "rejected": canonical_partition_mismatch and correct_canonical != wrong_canonical_loss,
        },
        "classwise_wrong_coordinate": {
            "mistake": "grouping class 1 posterior by S_0 instead of S_1",
            "correct_loss": correct_class_1_loss,
            "wrong_loss": wrong_class_1_loss,
            "numeric_mismatch": correct_class_1_loss != wrong_class_1_loss,
            "partition_mismatch": classwise_partition_mismatch,
            "rejected": classwise_partition_mismatch and correct_class_1_loss != wrong_class_1_loss,
        },
        "toplabel_wrong_event": {
            "mistake": "using max_k P(Y=k|X) instead of P(Y=Y_hat(X)|X)",
            "correct_loss": correct_top,
            "wrong_loss": wrong_top_loss,
            "event_vs_max_disagreement_atoms": event_vs_max_disagreement_atoms,
            "semantic_counterexample": {
                "score": top_event_counterexample.score,
                "posterior": top_event_counterexample.posterior,
                "predicted_class": top_event_counterexample.predicted_class,
                "correct_event_posterior": top_event_counterexample.correct_event_posterior,
                "incorrect_max_posterior": max(top_event_counterexample.posterior),
            },
            "rejected": event_semantics_rejected,
        },
        "broken_shared_posterior": {
            "mistake": "P_t(Y|X) differs from P_s(Y|X) while EC reuses q_s for target",
            "canonical_identity_rejected": broken_canonical,
            "classwise_identity_rejected": broken_classwise,
            "toplabel_identity_rejected": broken_top,
            "rejected": broken_canonical and broken_classwise and broken_top,
        },
    }
    controls["all_rejected"] = all(
        control["rejected"] for control in controls.values() if isinstance(control, dict)
    )
    return controls


def audit_official_source(repository_root: Path) -> dict[str, object]:
    """Pin and semantically audit all three official ECL mode branches."""
    source_path = repository_root / "repro/evidence/claim3/official_losses.py"
    source = source_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != OFFICIAL_LOSSES_SHA256:
        raise AssertionError(
            f"official_losses.py SHA256 changed: expected {OFFICIAL_LOSSES_SHA256}, got {digest}"
        )

    requirements = {
        "TopLabel": {
            "cache_shape": "torch.zeros(num_bins)",
            "grouping": "top_conf_train = train_probs.max(dim=1).values",
            "target_event": "p_correct_source = head_probs_correct(h2_train)",
            "bin_statistics": "m_s_batch = (w_s * p_correct_source.unsqueeze(1)).sum(dim=0)",
        },
        "Classwise": {
            "cache_shape": "torch.zeros(num_classes, num_bins)",
            "grouping": "conf_train_k = train_probs[:, k]",
            "target_event": "p_true_s_k = p_correct_source[:, k]",
            "bin_statistics": "m_s_batch = (w_s * p_true_s_k.unsqueeze(1)).sum(dim=0)",
        },
        "Canonical": {
            "cache_shape": "torch.zeros(self.num_bins, num_classes)",
            "grouping": "diffs_s = train_probs.unsqueeze(1) - anchors.unsqueeze(0)",
            "target_event": "p_correct_source = torch.softmax(h2_train, dim=1)",
            "bin_statistics": "m_s_batch = (w_s.unsqueeze(2) * p_correct_source.unsqueeze(1)).sum(dim=0)",
        },
    }
    line_numbers: dict[str, int] = {}
    source_lines = source.splitlines()
    for mode, checks in requirements.items():
        for label, fragment in checks.items():
            matches = [index + 1 for index, line in enumerate(source_lines) if fragment in line]
            if not matches:
                raise AssertionError(f"Official {mode} branch is missing {label}: {fragment}")
            line_numbers[f"{mode}.{label}"] = matches[-1]
    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_COMMIT,
        "vendored_path": source_path.relative_to(repository_root).as_posix(),
        "sha256": digest,
        "sha256_matches_pin": True,
        "all_three_branches_present": True,
        "semantic_checks": requirements,
        "line_numbers": line_numbers,
        "assessment": (
            "The pinned implementation separates TopLabel scalar confidence/correctness, "
            "Classwise per-coordinate confidence/posterior, and Canonical full-vector "
            "confidence/posterior branches. The returned objective is the paper's "
            "auxiliary/proximal mini-batch loss, not this population certificate."
        ),
    }


def _fixture_report(fixture: ExactFixture) -> dict[str, object]:
    canonical = canonical_certificate(fixture)
    classwise = classwise_certificate(fixture)
    top = toplabel_certificate(fixture)
    all_exact = bool(
        canonical["identity_exact"]
        and classwise["identity_exact"]
        and top["identity_exact"]
    )
    losses = {
        "canonical": canonical["ecl_l1_loss"],
        "classwise": classwise["ecl_l1_loss"],
        "toplabel": top["ecl_l1_loss"],
    }
    return {
        "name": fixture.name,
        "atoms": len(fixture.atoms),
        "classes": 3,
        "all_atoms_positive_in_both_domains": all(
            mass > 0 for mass in (*fixture.source_mass, *fixture.target_mass)
        ),
        "strict_covariate_shift": fixture.source_mass != fixture.target_mass,
        "source_mass": fixture.source_mass,
        "target_mass": fixture.target_mass,
        "canonical": canonical,
        "classwise": classwise,
        "toplabel": top,
        "all_population_identities_exact": all_exact,
        "losses": losses,
        "all_losses_zero": all(loss == 0 for loss in losses.values()),
        "all_losses_positive": all(loss > 0 for loss in losses.values()),
    }


def build_report(repository_root: Path | None = None) -> dict[str, object]:
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[2]
    calibrated = make_fixture(calibrated=True)
    diagnostic = make_fixture(calibrated=False)
    calibrated_report = _fixture_report(calibrated)
    diagnostic_report = _fixture_report(diagnostic)
    controls = negative_controls(diagnostic)
    source_audit = audit_official_source(repository_root)

    criteria = {
        "calibrated_fixture_exact_and_zero": bool(
            calibrated_report["all_population_identities_exact"]
            and calibrated_report["all_losses_zero"]
        ),
        "diagnostic_fixture_exact_and_nonzero": bool(
            diagnostic_report["all_population_identities_exact"]
            and diagnostic_report["all_losses_positive"]
        ),
        "negative_controls_rejected": bool(controls["all_rejected"]),
        "official_source_pin_and_semantics_pass": bool(
            source_audit["sha256_matches_pin"]
            and source_audit["all_three_branches_present"]
        ),
    }
    return {
        "schema_version": 1,
        "claim": "ECL is compatible with canonical calibration, class-wise calibration, and top-label calibration.",
        "method": "exact rational finite-distribution certificates plus general total-probability derivation",
        "paper": {
            "title": "Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift",
            "openreview_id": "gFPPTokv9C",
            "pdf_path": "repro/evidence/claim3/2605.21552v1.pdf",
            "pdf_sha256": PAPER_SHA256,
            "anchors": [
                "Theorem 3.1 and Eq. 4",
                "Appendix D Theorems D.1-D.2 and Eqs. 15-19",
                "Appendix E Eqs. 20-23",
                "Appendix F Eqs. 24-29",
            ],
        },
        "general_identity": (
            "For every deterministic summary T(X), P_d(A|T=t) equals "
            "sum_{x:T(x)=t} P_d(x) P(A|x) / P_d(T=t). Hence the ECL "
            "source-target residual exactly equals the direct calibration gap. "
            "Canonical uses T=S and vector A=Y; class-wise uses T=S_k and "
            "A={Y_k=1}; top-label uses T=max(S) and A={Y=argmax(S)}."
        ),
        "fixtures": {
            "calibrated": calibrated_report,
            "diagnostic": diagnostic_report,
        },
        "negative_controls": controls,
        "official_source_audit": source_audit,
        "success_criteria": criteria,
        "all_success_criteria_pass": all(criteria.values()),
        "scope": {
            "executed": (
                "Population and hard-group empirical formulas for all three paradigms, "
                "with exact arithmetic and official branch semantics."
            ),
            "not_executed": (
                "Neural training, image benchmarks, learned auxiliary-head estimation, "
                "soft-anchor numerical equivalence, or performance improvement."
            ),
            "cost_usd": 0,
            "random_seeds": "none; deterministic exact arithmetic",
        },
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "hardware": "local CPU only",
            "third_party_runtime_dependencies": "none",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/claim2_three_paradigm_certificate.json"),
        help="Stable JSON evidence path (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = build_report()
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_serialize(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diagnostic = report["fixtures"]["diagnostic"]
    print("Claim 2 exact three-paradigm certificate")
    print("  exact calibrated zero-loss fixture:", report["success_criteria"]["calibrated_fixture_exact_and_zero"])
    print("  exact diagnostic nonzero fixture:", report["success_criteria"]["diagnostic_fixture_exact_and_nonzero"])
    for name, loss in diagnostic["losses"].items():
        print(f"  diagnostic {name:9s} ECL L1 loss: {_fraction_text(loss)}")
    print("  negative controls all rejected:", report["negative_controls"]["all_rejected"])
    print("  official source audit:", report["success_criteria"]["official_source_pin_and_semantics_pass"])
    print("  all success criteria:", report["all_success_criteria_pass"])
    print(f"  runtime_seconds: {elapsed:.6f}")
    print("  wrote:", args.output.as_posix())
    return 0 if report["all_success_criteria_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
