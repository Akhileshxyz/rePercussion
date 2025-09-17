import unittest
from unittest.mock import MagicMock

from app.services.analysis_service import _describe_personality


class TestAnalysisService(unittest.TestCase):
    def test_describe_personality_high_energy(self):
        """
        Tests that a high energy rating results in a personality description
        containing "high-energy".
        """
        ratings = {"energy": "High", "danceability": "High"}
        summary = {}
        favorite_genre = "pop"
        description = _describe_personality(ratings, summary, favorite_genre)
        self.assertIn("high‑energy", description)

    def test_describe_personality_low_valence(self):
        """
        Tests that a low valence rating results in a personality description
        containing "introspective and moody".
        """
        ratings = {"valence": "Low"}
        summary = {}
        favorite_genre = "ambient"
        description = _describe_personality(ratings, summary, favorite_genre)
        self.assertIn("introspective and moody", description)


if __name__ == "__main__":
    unittest.main()
