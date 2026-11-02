"""Framing, strict review gates and segmentation; every remote service is fake."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from mvswap import ai, framing, media, pipeline, rh, vision


class Framing(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(mock.patch.stopall)
        mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")).start()
        mock.patch.object(ai, "upload", side_effect=AssertionError("upload forbidden")).start()
        self.pic = self.root / "full.jpg"
        im = Image.new("RGB", (600, 1200), "white")
        draw = ImageDraw.Draw(im)
        draw.rectangle((180, 40, 420, 550), fill="blue")
        draw.rectangle((180, 600, 420, 1150), fill="red")
        im.save(self.pic, quality=100)
        self.cast = {"id": "P1", "picture": str(self.pic), "desc": "black hair and red ribbons, red eyes, navy shirt, red stockings and shoes", "src_desc": "the woman",
                     "references": [{"picture": str(self.pic), "source": "primary.jpg", "framing": "full", "compatible": True,
                                     "bbox": [.3, .03, .7, .96], "chin_y": .2, "waist_y": .46, "knee_y": .8}]}

    def test_full_reference_is_cropped_to_waist_without_upscaling_or_lower_garments(self):
        r = framing.reference(self.cast, {"framing": "半身"}, self.root / "half.jpg")
        self.assertEqual(r["method"], "crop")
        self.assertEqual(r["framing"], "medium")
        with Image.open(r["picture"]) as im:
            self.assertEqual(im.height, 552)
            self.assertLessEqual(im.width, 600)
            self.assertLess(im.getpixel((im.width//2, im.height-3))[0], 30)
        self.assertNotIn("stockings", r["desc"])
        self.assertIn("black hair", r["desc"])
        self.assertIn("red ribbons", r["desc"])
        self.assertIn("navy shirt", r["desc"])
        self.assertEqual(r["crop_box"][1], 0)  # space above the hair preserved

    def test_existing_compatible_half_body_wins_over_crop_and_wrong_identity(self):
        self.cast["references"] += [
            {"picture": "wrong-face.jpg", "framing": "medium", "compatible": False},
            {"picture": "correct-half.jpg", "framing": "medium", "compatible": True, "desc": "navy shirt"}]
        r = framing.reference(self.cast, {"framing": "medium"}, self.root / "unused.jpg")
        self.assertEqual(r["picture"], "correct-half.jpg")
        self.assertFalse((self.root / "unused.jpg").exists())

    def test_missing_landmark_or_unknown_framing_blocks_submission(self):
        for target in ("medium", "unknown"):
            self.cast["references"][0].pop("waist_y", None)
            with self.subTest(target=target), self.assertRaises(ValueError):
                framing.reference(self.cast, {"framing": target}, self.root / "out.jpg")

    def test_close_reference_cannot_be_used_to_invent_full_body(self):
        self.cast["references"][0]["framing"] = "close-up"
        with self.assertRaisesRegex(ValueError, "补全身体"):
            framing.reference(self.cast, {"framing": "full"}, self.root / "out.jpg")

    def test_existing_label_call_carries_each_person_geometry(self):
        reply = {"kind": "swap", "people": ["P1"], "framing": "medium",
                 "geometry": {"P1": {"framing": "medium", "bbox": [.1, .1, .9, 1]}}}
        with mock.patch.object(ai, "ask_json", return_value=reply) as call:
            result = vision.label_shot([self.cast], [self.pic])
        call.assert_called_once()
        self.assertEqual(result["geometry"], reply["geometry"])

    def test_reference_stage_checkpoints_and_resume_does_not_call_vision(self):
        job = pipeline.Job(self.root / "job")
        job.state = {"cast": [self.cast], "shots": [{"id": "s001", "kind": "swap", "people": ["P1"], "framing": "medium"}]}
        with mock.patch.object(ai, "ask_json", side_effect=AssertionError("extra billed call")), redirect_stdout(io.StringIO()):
            pipeline.stage_references(job)
            saved = pipeline.Job(job.dir)
            pipeline.stage_references(saved)
        self.assertEqual(saved.state["shots"][0]["references"]["P1"]["method"], "crop")

    def reply(self):
        return {"right_people": True, "originals_left": False, "extra_people": False, "added_overlays": False,
                "broken": False, "style_kept": True, "framing": 2, "score": 10, "missing": [], "issue": "",
                "per_second": [{"second": s, "source_size": "medium", "remake_size": "medium", "camera_match": True,
                                "pose_match": True, "reference_pose": False, "newly_visible_lower_body": False} for s in range(9)]}

    def review(self, reply):
        with mock.patch.object(ai, "ask_json", return_value=reply):
            return vision.review_take([self.cast], ["P1"], [self.pic]*9, [], seconds=list(range(9)))

    def test_ten_points_still_fails_on_one_second_wider_or_tighter(self):
        for size in ("three-quarter", "full", "close-up"):
            reply = self.reply()
            reply["per_second"][5]["remake_size"] = size
            r = self.review(reply)
            self.assertFalse(r["pass"])
            self.assertEqual(r["drift_seconds"], [5])
            self.assertIn("取景/姿势漂到参考图（第 5 秒）", r["issue"])

    def test_pose_camera_reference_pose_and_new_lower_body_are_independent_vetoes(self):
        for key, value in (("camera_match", False), ("pose_match", False), ("reference_pose", True), ("newly_visible_lower_body", True)):
            reply = self.reply()
            reply["per_second"][7][key] = value
            with self.subTest(key=key):
                r = self.review(reply)
                self.assertFalse(r["pass"])
                self.assertIn("第 7 秒", r["issue"])

    def test_missing_duplicate_unknown_or_nonboolean_check_fails_closed(self):
        for change in (lambda rows: rows.pop(3), lambda rows: rows.append(rows[3]),
                       lambda rows: rows[3].update(source_size="unknown"), lambda rows: rows[3].update(pose_match="true")):
            reply = self.reply(); change(reply["per_second"])
            self.assertFalse(self.review(reply)["pass"])

    def test_all_seconds_clean_passes_and_identity_failure_still_fails(self):
        reply = self.reply()
        self.assertTrue(self.review(reply)["pass"])
        reply["originals_left"] = True
        self.assertFalse(self.review(reply)["pass"])

    def test_review_sampling_includes_global_seconds_and_segment_edges(self):
        self.assertEqual(framing.sample_indices({"f0": 70, "frames": 53}), [0, 2, 26, 50, 52])

    def test_sample_ids_keep_fractional_segment_times_exact_even_if_model_rounds(self):
        reply = self.reply()
        reply["per_second"] = [{**row, "sample": i} for i, row in enumerate(reply["per_second"][:3])]
        times = [2.917, 3, 3.958]
        with mock.patch.object(ai, "ask_json", return_value=reply):
            result = vision.review_take([self.cast], ["P1"], [self.pic]*3, [], seconds=times)
        self.assertTrue(result["pass"])
        self.assertEqual([r["second"] for r in result["per_second"]], times)
        reply["per_second"][1]["sample"] = 0
        with mock.patch.object(ai, "ask_json", return_value=reply):
            self.assertFalse(vision.review_take([self.cast], ["P1"], [self.pic]*3, [], seconds=times)["pass"])

    def test_fake_runninghub_retry_uses_cropped_reference_and_rejects_high_score_drift(self):
        job = pipeline.Job(self.root / "retry")
        ref = framing.reference(self.cast, {"framing": "medium"}, job.dir / "refs/half.jpg")
        job.state = {"cast": [self.cast], "prepared": {"mv": "fake.mp4", "width": 1080, "height": 1920},
                     "shots": [{"id": "s001", "kind": "swap", "people": ["P1"], "f0": 0, "frames": 216, "gen_frames": 243,
                                "references": {"P1": ref}}]}
        bad, good = self.reply(), self.reply()
        bad["per_second"][5]["remake_size"] = "full"
        def generate(takes, callback):
            self.assertEqual(takes[0]["pictures"], [ref["picture"]])
            self.assertNotIn("stockings", takes[0]["prompt"])
            self.assertIn("black hair", takes[0]["prompt"])
            self.assertIn("red ribbons", takes[0]["prompt"])
            callback(takes[0]["id"], "done", {"cny": .01})
        with mock.patch.object(rh, "Scheduler") as scheduler, mock.patch.object(media, "cut_clip"), \
                mock.patch.object(pipeline, "_backdrop_lost", return_value=0), \
                mock.patch.object(framing, "pairs", return_value=([self.pic]*9, list(range(9)))), \
                mock.patch.object(ai, "ask_json", side_effect=[bad, good]), redirect_stdout(io.StringIO()):
            scheduler.return_value.run.side_effect = generate
            pipeline.stage_generate(job, 2)
        shot = job.state["shots"][0]
        self.assertEqual(len(shot["attempts"]), 2)
        self.assertFalse(shot["attempts"][0]["review"]["pass"])
        self.assertEqual(shot["chosen"], 2)


class Segments(unittest.TestCase):
    def test_maximum_coverage_real_cuts_and_pause_selection(self):
        motion = [10.0]*240
        motion[60] = 0; motion[132] = 0
        original = [{"f0": 0, "frames": 216}, {"f0": 216, "frames": 24}]
        result = media.split_shots(original, 84, motion)
        self.assertEqual(result[0]["frames"], 60)
        self.assertIn(216, [s["f0"] for s in result])
        self.assertEqual(sum(s["frames"] for s in result), 240)
        self.assertTrue(all(0 < s["frames"] <= 84 and (s["gen_frames"]-5)%17 == 0 for s in result))
        for left, right in zip(result, result[1:]):
            self.assertEqual(left["f0"]+left["frames"], right["f0"])

    def test_small_remainders_and_one_second_cap_never_lose_frames(self):
        for frames in (1, 23, 24, 25, 47, 49, 216, 341):
            for cap in (24, 25, 72, 96):
                with self.subTest(frames=frames, cap=cap):
                    shots = media.split_shots([{"f0": 0, "frames": frames}], cap, [])
                    self.assertEqual(sum(s["frames"] for s in shots), frames)
                    self.assertTrue(all(0 < s["frames"] <= cap for s in shots))

    def test_disabled_segmentation_keeps_original_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = pipeline.Job(Path(tmp))
            job.state = {"args": {"max_segment_seconds": 0}, "prepared": {"mv": "fake.mp4", "frames": 216}}
            with mock.patch.object(media, "scene_scores", return_value=[0]*216), mock.patch.object(media, "structure", return_value=[1]*216), \
                    mock.patch.object(media, "raw_frames", side_effect=AssertionError("disabled")), redirect_stdout(io.StringIO()):
                pipeline.stage_shots(job)
            self.assertEqual(len(job.state["shots"]), 1)

    def test_enabled_segmentation_is_saved_and_reused_without_recutting(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            job = pipeline.Job(Path(tmp))
            job.state = {"args": {"max_segment_seconds": 3}, "prepared": {"mv": "fake.mp4", "frames": 216}}
            with mock.patch.object(media, "scene_scores", return_value=[0]*216), mock.patch.object(media, "structure", return_value=[1]*216), \
                    mock.patch.object(media, "raw_frames", return_value=np.zeros((216, 3, 3, 3), np.uint8)) as raw, redirect_stdout(io.StringIO()):
                pipeline.stage_shots(job)
                resumed = pipeline.Job(job.dir)
                pipeline.stage_shots(resumed)
            raw.assert_called_once()
            self.assertEqual([s["frames"] for s in resumed.state["shots"]], [72, 72, 72])

    def test_invalid_caps_rejected_before_job_creation(self):
        for value in (-1, .5, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                pipeline.run(Path("must-not-create"), None, [], max_segment_seconds=value)
