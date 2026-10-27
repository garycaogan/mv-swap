"""Checks that need no network: frame maths, shot finding, sizes, prompts, the workflow graph."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvswap import config, media, prompts, rh  # noqa: E402

CAST = [{"id": "P1", "gender": "woman", "desc": "long black hair with red ribbons", "picture": "/a.jpg", "src_desc": "the blonde woman"},
        {"id": "P2", "gender": "man", "desc": "short black hair, black jacket", "picture": "/b.jpg", "src_desc": "the man in the black cap"}]


class Frames(unittest.TestCase):
    def test_gen_frames_is_17k_plus_5_with_padding(self):
        for f in range(1, 360):
            n = media.gen_frames(f)
            self.assertEqual(n % 17, 5)
            self.assertGreaterEqual(n - f, config.PAD_MIN)
            self.assertLess(n - f, config.PAD_MIN + 17)

    def test_workflow_math_gives_back_the_same_frames(self):
        # node 389: max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17, a = seconds we send
        for f in (11, 89, 313, 350):
            n = media.gen_frames(f)
            m = max(5, round(n / 24 * 24))
            self.assertEqual(m + (5 - m % 17) % 17, n)


class Shots(unittest.TestCase):
    def test_cuts_hidden_cuts_and_long_shots(self):
        scores = [0.0] * 1000
        scores[100] = 0.6            # obvious cut
        scores[103] = 0.5            # 3 frames later: a flash, dropped
        scores[300] = 0.2            # quiet cut, found by the second pass
        scores[302] = 0.15           # too close to 300, ignored
        shots = media.find_shots(scores, 1000, 334)
        edges = [s["f0"] for s in shots]
        self.assertIn(100, edges)
        self.assertIn(300, edges)
        self.assertNotIn(302, edges)
        self.assertEqual(sum(s["frames"] for s in shots), 1000)
        self.assertTrue(all(s["frames"] <= 334 for s in shots))
        self.assertTrue(all(s["gen_frames"] <= config.MAX_GEN_FRAMES for s in shots))

    def test_flashes_do_not_cut(self):
        scores = [0.0] * 200
        sims = [1.0] * 200
        scores[50], sims[50] = 0.6, 0.9      # a bright flash: big score, same layout
        scores[120], sims[120] = 0.4, 0.1    # a real cut
        scores[123], sims[123] = 0.5, 0.0    # 3 frames later: dropped
        edges = [s["f0"] for s in media.find_shots(scores, 200, 334, sims)]
        self.assertEqual(edges, [0, 120])

    def test_sizes(self):
        self.assertEqual(media.target_size(1920, 1080), (1184, 672))
        self.assertEqual(media.target_size(1080, 1920), (672, 1184))
        w, h = media.target_size(1440, 1080)
        self.assertEqual((w % 32, h % 32), (0, 0))


class Prompts(unittest.TestCase):
    def test_solo_shot_gets_one_picture_and_no_second_person(self):
        p, pics = prompts.build(CAST, ["P2"], False)
        self.assertEqual(pics, ["/b.jpg"])
        self.assertIn("only person", p)
        self.assertIn("young man", p)
        self.assertIn("his face", p)
        self.assertNotIn("<Picture 2>", p)
        self.assertNotIn("words", p)

    def test_two_people_map_in_order(self):
        p, pics = prompts.build(CAST, ["P1", "P2"], True)
        self.assertEqual(pics, ["/a.jpg", "/b.jpg"])
        self.assertIn("The blonde woman in <Video 1> becomes the young woman in <Picture 1>", p)
        self.assertIn("becomes the young man in <Picture 2>", p)
        self.assertIn("words already written on screen", p)
        for bad in ("every cut", "sticker", "amplifier"):
            self.assertNotIn(bad, p)


class SecondPass(unittest.TestCase):
    def test_second_pass_keeps_the_people_already_swapped(self):
        p, pics = prompts.build(CAST, ["P1"], False, second_pass=True)
        self.assertEqual(pics, ["/a.jpg"])
        self.assertNotIn("only person", p)
        self.assertIn("Everyone else in <Video 1> stays exactly as they are", p)

    def test_backdrop_is_named_per_shot(self):
        # an animated MV: without the shot's own setting, dark scenes took the cast pictures' grey studio background
        p, _ = prompts.build(CAST, ["P1"], False, backdrop="a dark palace hall lit by a golden beam from above")
        self.assertIn("The backdrop is a dark palace hall lit by a golden beam from above, exactly as in <Video 1>", p)
        self.assertIn("never the plain background behind them", p)
        self.assertNotIn(prompts.SET, p)
        p, _ = prompts.build(CAST, ["P1"], False)
        self.assertIn(prompts.SET, p)

    def test_animation_style_sentence(self):
        p, _ = prompts.build(CAST, ["P1"], False, style={"kind": "animation", "desc": "a stylized 3D animated music video"})
        self.assertIn("a stylized 3D animated music video", p)


# the nodes of the swap workflow we fill in (the full graph is fetched from RunningHub at run time)
FIXTURE = {n: {"class_type": c, "inputs": {}} for n, c in rh.SWAP_NODES.items()}


class Graph(unittest.TestCase):
    def test_graph_takes_our_inputs(self):
        g = rh.build_graph(FIXTURE, "v.mp4", ["a.png", "b.png"], "hello", 13.6666, 1184, 672, 7)
        self.assertEqual(g["554"]["inputs"]["video"], "v.mp4")
        self.assertEqual(g["200"]["inputs"]["image"], "a.png")
        self.assertEqual(g["901"]["inputs"]["image"], "b.png")
        self.assertEqual(g["385"]["inputs"]["ref_images.ref_image_1"], ["901", 0])
        self.assertEqual((g["385"]["inputs"]["width"], g["385"]["inputs"]["height"]), (1184, 672))
        self.assertEqual(g["456"]["inputs"]["seed"], 7)
        self.assertEqual(g["557"]["inputs"]["text"], "hello")
        self.assertNotIn("901", FIXTURE)          # the cached base graph is never changed in place

    def test_changed_workflow_is_refused(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config.CACHE = Path(tmp)
            bad = dict(FIXTURE, **{"557": {"class_type": "SomethingElse", "inputs": {}}})
            (Path(tmp) / "workflow-x.json").write_text(json.dumps(bad))
            with self.assertRaises(SystemExit):
                rh.workflow("x", rh.SWAP_NODES)
            (Path(tmp) / "workflow-y.json").write_text(json.dumps(FIXTURE))
            self.assertIn("557", rh.workflow("y", rh.SWAP_NODES))


if __name__ == "__main__":
    unittest.main()
