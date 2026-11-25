import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reject_generic_topic_matches import is_pure_generic, routes


class GenericTopicRejectionTests(unittest.TestCase):
    def test_whole_fragment_can_contain_several_generic_phrases(self):
        for text in ("A classic masterpiece, I love it.", "太好玩了，强烈推荐！", "10/10", "It’s such a good game."):
            with self.subTest(text=text):
                self.assertTrue(is_pure_generic(text))

    def test_generic_openers_do_not_hide_concrete_evidence(self):
        cases = (
            "Great combat, especially the parry timing.",
            "好玩，就是一进城就只剩十几帧。",
            "这299很值。",
            "Good game. The art direction is gorgeous.",
            "画面很好。",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(is_pure_generic(text))

    def test_semantic_rejection_only_vetoes_selected_topics(self):
        policy = {"semantic_topics": ["T05"], "minimum_topic_over_generic_margin": 0.02}
        self.assertEqual(routes("A classic masterpiece!", "T05", .85, .72, policy), ("A01", "A01"))
        self.assertEqual(routes("Not quite what I hoped for", "T05", .75, .74, policy), (None, "A03"))
        self.assertEqual(routes("Not quite what I hoped for", None, .75, .74, policy), ("A03", "A03"))
        self.assertEqual(routes("The art direction is gorgeous", "T05", .78, .70, policy), (None, None))
        self.assertEqual(routes("这299很值", "T07", .75, .74, policy), (None, None))


if __name__ == "__main__":
    unittest.main()
