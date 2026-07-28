from __future__ import annotations

"""The invariant-relational philosophy behind the method.

The philosophy is operational, not decorative: each principle maps to a
mathematical object and to a verifiable implementation rule.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CorePrinciple:
    name: str
    mathematical_form: str
    implementation_rule: str
    real_world_meaning: str


PRINCIPLES = (
    CorePrinciple(
        name="Identity before inscription",
        mathematical_form="T = D + N; preserve D = diag(lambda_1,...,lambda_4)",
        implementation_rule="Never modify the virtual Schur diagonal; embed only in strict-upper couplings.",
        real_world_meaning="A media asset should retain a stable structural identity while carrying provenance.",
    ),
    CorePrinciple(
        name="Relations carry meaning",
        mathematical_form="n* = n + H^T(t-Hn)",
        implementation_rule="Project relational coordinates onto the nearest parity coset with minimum Frobenius change.",
        real_world_meaning="Information is carried by relationships among coefficients, not by overwriting isolated pixels.",
    ),
    CorePrinciple(
        name="Identity witnesses relation",
        mathematical_form="omega_i = exp(-((log s_i - median(log s)) / sigma)^2)",
        implementation_rule="Use unchanged diagonal-derived spectral scale to gate confidence in damaged coupling evidence.",
        real_world_meaning="The same invariant that was protected during embedding later tells the decoder which evidence to trust.",
    ),
    CorePrinciple(
        name="Verify meaning, not resemblance",
        mathematical_form="valid = ECC decode AND CRC AND digital signature",
        implementation_rule="Treat NC as a diagnostic metric; accept real payloads only after ECC decoding, CRC validation, and signature verification.",
        real_world_meaning="A recovered logo that looks similar is weaker evidence than an exactly verified signed record.",
    ),
)


def philosophy_summary() -> dict[str, object]:
    return {
        "name": "Invariant-Relational DCT-Schur Watermarking",
        "thesis": (
            "Preserve spectral identity, inscribe information in relational degrees "
            "of freedom, and use preserved identity as a witness during recovery."
        ),
        "principles": [principle.__dict__ for principle in PRINCIPLES],
    }


__all__ = ["CorePrinciple", "PRINCIPLES", "philosophy_summary"]
