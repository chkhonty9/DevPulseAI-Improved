"""Link critic claims to the exact sentences of a digest.

The critic is asked to quote unsupported claims verbatim, but a small model
often paraphrases them or quotes text that is not in the report at all (for
example the developer profile). A revision cannot fix a sentence it cannot
find, so every claim is resolved to a real report sentence here, in code.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable, List, Mapping, Optional, Tuple

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_EDGE_PUNCTUATION = " \t\n\"'“”‘’`.,;:!?()[]"

# A paraphrase must be close to one report sentence to count as that sentence.
MATCH_THRESHOLD = 0.6
MIN_SUBSTRING_LENGTH = 20


def normalize(text: str) -> str:
    """Casefold, collapse whitespace, and trim quotes/punctuation at the ends."""
    return " ".join(str(text).casefold().split()).strip(_EDGE_PUNCTUATION)


def report_sentences(digest: Mapping[str, Any]) -> List[str]:
    """Return the summary and recommendation sentences, in report order."""
    texts: List[str] = [str(digest.get("executive_summary", ""))]
    for recommendation in digest.get("recommendations", []) or []:
        if isinstance(recommendation, Mapping):
            texts.append(str(recommendation.get("text", "")))
    sentences: List[str] = []
    for text in texts:
        for sentence in _SENTENCE_BOUNDARY.split(text.strip()):
            if sentence.strip():
                sentences.append(sentence.strip())
    return sentences


def cited_ids_for_sentence(digest: Mapping[str, Any], sentence: str) -> List[str]:
    """Return the evidence IDs cited by the part of the report a sentence is in."""
    if contains_sentence(str(digest.get("executive_summary", "")), sentence):
        return list(digest.get("summary_evidence_ids", []) or [])
    for recommendation in digest.get("recommendations", []) or []:
        if isinstance(recommendation, Mapping) and contains_sentence(
            str(recommendation.get("text", "")), sentence
        ):
            return list(recommendation.get("evidence_ids", []) or [])
    return []


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.casefold()) if len(word) > 3}


def _similarity(claim: str, sentence: str) -> float:
    claim_norm, sentence_norm = normalize(claim), normalize(sentence)
    if len(claim_norm) >= MIN_SUBSTRING_LENGTH and (
        claim_norm in sentence_norm or sentence_norm in claim_norm
    ):
        return 1.0
    ratio = SequenceMatcher(None, claim_norm, sentence_norm).ratio()
    claim_words = _content_words(claim)
    overlap = (
        len(claim_words & _content_words(sentence)) / len(claim_words)
        if claim_words
        else 0.0
    )
    return max(ratio, overlap)


def match_claim(claim: str, sentences: Iterable[str]) -> Optional[str]:
    """Return the report sentence a claim refers to, or None if none is close."""
    best_sentence, best_score = None, 0.0
    for sentence in sentences:
        score = _similarity(claim, sentence)
        if score > best_score:
            best_sentence, best_score = sentence, score
    return best_sentence if best_score >= MATCH_THRESHOLD else None


def resolve_claims(
    claims: Iterable[str],
    sentences: List[str],
) -> Tuple[List[str], List[str]]:
    """Split claims into (exact report sentences, claims not in the report)."""
    resolved: List[str] = []
    not_in_report: List[str] = []
    for claim in claims:
        sentence = match_claim(claim, sentences)
        if sentence is None:
            not_in_report.append(claim)
        elif sentence not in resolved:
            resolved.append(sentence)
    return resolved, not_in_report


def contains_sentence(text: str, sentence: str) -> bool:
    """True when the normalized sentence still appears in the text."""
    return normalize(sentence) in normalize(text)


def drop_sentences(text: str, flagged: Iterable[str]) -> str:
    """Remove every sentence of text that is (or contains) a flagged sentence."""
    flagged_norm = [normalize(item) for item in flagged if normalize(item)]
    kept = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(str(text).strip())
        if sentence.strip()
        and not any(
            item in normalize(sentence) or normalize(sentence) in item
            for item in flagged_norm
        )
    ]
    return " ".join(kept)
