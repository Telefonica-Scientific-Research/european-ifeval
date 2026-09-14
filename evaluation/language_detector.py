# coding=utf-8
"""GlotLID-backed language detection for European IFEval."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Optional


DEFAULT_MODEL_REPO = "cis-lmu/glotlid"
DEFAULT_MODEL_FILENAME = "model.bin"
DEFAULT_MIN_CONFIDENCE = 0.5
MIN_REPEATED_TOKEN_TAIL_REPEATS = 8
MIN_REPEATED_TOKEN_TAIL_CHARS = 20
MIN_COMPACT_TAIL_REPEATS = 8
MIN_COMPACT_TAIL_CHARS = 24
MAX_REPEATED_TOKEN_UNIT_TOKENS = 3
MAX_COMPACT_UNIT_CHARS = 6


class LanguageDetectionError(Exception):
  """Raised when language detection cannot produce a usable prediction."""

  def __init__(
      self,
      message: str,
      *,
      glotlid_label: str | None = None,
      confidence: float | None = None,
      detector_input: str | None = None,
      repo_code: str | None = None,
  ):
    super().__init__(message)
    self.glotlid_label = glotlid_label
    self.confidence = confidence
    self.detector_input = detector_input
    self.repo_code = repo_code


class LanguageDetectorConfigurationError(LanguageDetectionError):
  """Raised when GlotLID cannot be loaded or configured."""


@dataclass(frozen=True)
class LanguagePrediction:
  repo_code: str | None
  glotlid_label: str
  confidence: float


_ISO3_TO_REPO_CODE = {
    "ara": "ar",
    "ast": "ast",
    "arb": "ar",
    "bel": "be",
    "ben": "bn",
    "bos": "bs",
    "bre": "br",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "cym": "cy",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    # GlotLID can label short modern-English text as Middle English.
    "enm": "en",
    "est": "et",
    "eus": "eu",
    "fao": "fo",
    "fas": "fa",
    "fin": "fi",
    "fra": "fr",
    "fur": "fur",
    "gla": "gd",
    "gle": "ga",
    "glg": "gl",
    "guj": "gu",
    "heb": "he",
    "hin": "hi",
    "hrv": "hr",
    "hun": "hu",
    "isl": "is",
    "ita": "it",
    "jpn": "ja",
    "kat": "ka",
    "kan": "kn",
    "kor": "ko",
    "ltg": "ltg",
    "lij": "lij",
    "lit": "lt",
    "lmo": "lmo",
    "ltz": "lb",
    "lav": "lv",
    "mar": "mr",
    "mkd": "mk",
    "mal": "ml",
    "mlt": "mt",
    "nld": "nl",
    "nno": "nn",
    "nob": "no",
    "nor": "no",
    "nep": "ne",
    "npi": "ne",
    "oci": "oc",
    "pan": "pa",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "srd": "sc",
    "scn": "scn",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "sqi": "sq",
    "srp": "sr",
    "swe": "sv",
    "swh": "sw",
    "szl": "szl",
    "tam": "ta",
    "tel": "te",
    "tha": "th",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "vec": "vec",
    "vie": "vi",
    "zho": "zh-cn",
    "cmn": "zh-cn",
    # Alternate ISO 639-2 bibliographic codes that may appear in tooling.
    "alb": "sq",
    "baq": "eu",
    "chi": "zh-cn",
    "cze": "cs",
    "dut": "nl",
    "fre": "fr",
    "geo": "ka",
    "ger": "de",
    "gre": "el",
    "ice": "is",
    "mac": "mk",
    "may": "ms",
    "per": "fa",
    "rum": "ro",
    "slo": "sk",
    "wel": "cy",
}


_model = None
_model_path = None
_min_confidence = DEFAULT_MIN_CONFIDENCE


def configure(*, model_path: Optional[str] = None,
              min_confidence: Optional[float] = None):
  """Configures model loading and confidence threshold."""
  global _model
  global _model_path
  global _min_confidence

  configured_model_path = model_path or os.environ.get("GLOTLID_MODEL_PATH")
  if configured_model_path != _model_path:
    _model = None
    _model_path = configured_model_path

  if min_confidence is None:
    min_confidence = float(
        os.environ.get("GLOTLID_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
    )
  _min_confidence = min_confidence


def get_min_confidence() -> float:
  return _min_confidence


def _load_model():
  global _model
  if _model is not None:
    return _model

  try:
    import fasttext  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise LanguageDetectorConfigurationError(
        "fasttext is required for GlotLID language detection. "
        "Install the repository root requirements.txt."
    ) from exc

  model_path = _model_path
  if not model_path:
    try:
      from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
      raise LanguageDetectorConfigurationError(
          "Set GLOTLID_MODEL_PATH or install huggingface-hub so the "
          "GlotLID model can be loaded from the Hugging Face cache."
      ) from exc
    try:
      model_path = hf_hub_download(
          repo_id=DEFAULT_MODEL_REPO,
          filename=DEFAULT_MODEL_FILENAME,
      )
    except Exception as exc:
      raise LanguageDetectorConfigurationError(
          "Unable to load GlotLID from Hugging Face. Set GLOTLID_MODEL_PATH "
          "to a local model.bin if the model is not already cached."
      ) from exc

  try:
    _model = fasttext.load_model(model_path)
  except Exception as exc:
    raise LanguageDetectorConfigurationError(
        f"Unable to load GlotLID model from {model_path!r}."
    ) from exc
  return _model


def _normalize_label(
    raw_label: str,
    *,
    confidence: float | None = None,
    detector_input: str | None = None,
) -> tuple[str, str | None]:
  label = raw_label.removeprefix("__label__")
  iso3 = label.split("_", maxsplit=1)[0]
  repo_code = _ISO3_TO_REPO_CODE.get(iso3)
  return label, repo_code


def _strip_repeated_token_tail(text: str) -> str:
  """Removes obvious whitespace-separated repeated-token tails."""
  tokens = text.split()
  for unit_size in range(1, MAX_REPEATED_TOKEN_UNIT_TOKENS + 1):
    if len(tokens) < unit_size * MIN_REPEATED_TOKEN_TAIL_REPEATS:
      continue

    unit = tokens[-unit_size:]
    repeats = 0
    index = len(tokens)
    while index >= unit_size and tokens[index - unit_size:index] == unit:
      repeats += 1
      index -= unit_size

    if repeats < MIN_REPEATED_TOKEN_TAIL_REPEATS:
      continue

    repeated_tail = " ".join(tokens[index:])
    if len(repeated_tail) < MIN_REPEATED_TOKEN_TAIL_CHARS:
      continue

    return " ".join(tokens[:index]).strip()

  return text


def _strip_repeated_compact_tail(text: str) -> str:
  """Removes obvious compact repeated tails such as 'usususususus'."""
  for unit_size in range(1, MAX_COMPACT_UNIT_CHARS + 1):
    min_chars = unit_size * MIN_COMPACT_TAIL_REPEATS
    if len(text) < min_chars:
      continue

    unit = text[-unit_size:]
    repeats = 0
    index = len(text)
    while index >= unit_size and text[index - unit_size:index] == unit:
      repeats += 1
      index -= unit_size

    tail_chars = repeats * unit_size
    if (
        repeats >= MIN_COMPACT_TAIL_REPEATS
        and tail_chars >= MIN_COMPACT_TAIL_CHARS
    ):
      return text[:index].strip()

  return text


def _build_language_detection_text(value: str) -> str:
  """Builds a GlotLID-only text by removing generation artifacts at the end."""
  text = value.replace("\n", " ")
  text = re.sub(r"\s+", " ", text).strip()
  text = _strip_repeated_token_tail(text)
  text = _strip_repeated_compact_tail(text)
  return text.strip()


def build_language_detection_text(value: str) -> str:
  """Returns the exact text used as input to GlotLID."""
  return _build_language_detection_text(value)


def detect(value: str) -> LanguagePrediction:
  """Returns the mapped GlotLID prediction for a response."""
  if not value or not value.strip():
    raise LanguageDetectionError("empty response")

  model = _load_model()
  language_detection_text = _build_language_detection_text(value)
  if not language_detection_text:
    raise LanguageDetectionError(
        "empty response after language-detection preprocessing"
    )

  labels, probabilities = model.predict(language_detection_text, k=1)
  if not labels:
    raise LanguageDetectionError("GlotLID returned no prediction")

  confidence = float(probabilities[0])
  if confidence < _min_confidence:
    raise LanguageDetectionError(
        f"GlotLID confidence {confidence:.4f} is below "
        f"threshold {_min_confidence:.4f} for label {labels[0]!r}",
        glotlid_label=labels[0].removeprefix("__label__"),
        confidence=confidence,
        detector_input=language_detection_text,
    )

  label, repo_code = _normalize_label(
      labels[0],
      confidence=confidence,
      detector_input=language_detection_text,
  )
  return LanguagePrediction(
      repo_code=repo_code,
      glotlid_label=label,
      confidence=confidence,
  )


def is_language(value: str, expected_language: str) -> bool:
  return detect(value).repo_code == expected_language
