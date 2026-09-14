"""Tests for language-detector normalization without loading GlotLID."""

from absl.testing import absltest

from evaluation import language_detector


class LanguageDetectorTest(absltest.TestCase):

  def test_middle_english_label_maps_to_english(self):
    label, repo_code = language_detector._normalize_label("__label__enm_Latn")

    self.assertEqual(label, "enm_Latn")
    self.assertEqual(repo_code, "en")


if __name__ == "__main__":
  absltest.main()
