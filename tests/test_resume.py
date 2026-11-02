"""Paid-work checkpoints and local diagnostics; every RunningHub and APIMart reply here is fake."""
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from threading import Event
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from mvswap import ai, config, finish, media, pipeline, rh  # noqa: E402

ENTRY = Path(__file__).resolve().parents[1] / "mvswap.py"
# A real ffmpeg for the local checks: MVSWAP_TEST_FFMPEG (e.g. an ffmpeg 9 build), else the one on PATH.
FFMPEG = Path(os.environ.get("MVSWAP_TEST_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg-not-found")


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="mvswap-resume-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.dict(ai._spent, {"vision": 0.0, "image": 0.0}))
        self.stack.enter_context(mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")))
        self.stack.enter_context(mock.patch.object(ai, "_req", side_effect=AssertionError("unexpected APIMart call")))
        self.stack.enter_context(mock.patch.object(ai, "upload", side_effect=AssertionError("unexpected APIMart upload")))
        self.stack.enter_context(mock.patch.object(rh, "call", side_effect=AssertionError("unexpected RunningHub call")))
        self.stack.enter_context(mock.patch.object(rh, "upload", side_effect=AssertionError("unexpected RunningHub upload")))
        self.job = pipeline.Job(self.root / "任务 中文")
        self.job.state = {
            "args": {"source": "source.mp4", "cast": ["女生=千咲·真人"], "start": 0, "end": 1, "upscale": True},
            "prepared": {"mv": "source.mp4", "frames": 24, "width": 1080, "height": 1920},
            "cast": [{"id": "P1", "hint": "女生", "name": "千咲·真人", "picture": "ref.jpg",
                      "desc": "black hair and red ribbons", "src_desc": "the blonde woman", "gender": "woman"}],
            "shots": [{"id": "s001", "kind": "swap", "people": ["P1"], "f0": 0, "frames": 24,
                       "gen_frames": media.gen_frames(24), "attempts": []}],
        }
        self.shot = self.job.state["shots"][0]

    def reload(self):
        return pipeline.Job(self.job.dir)

    def attempt(self, n, passed=False):
        return {"n": n, "status": "done", "file": str(self.job.dir / f"take-{n}.mp4"), "seed": n,
                "prompt": "fake prompt", "pictures": ["ref.jpg"], "cny": 0.1,
                "review": {"pass": passed, "score": 7 if passed else 3, "issue": "" if passed else "原女生仍在"}}

    def fake_scheduler(self, responder, keys=("fake-rh-key",)):
        """RunningHub with fake keys (named k1, k2, … in order); `self.graphs` counts every graph built for a new submission."""
        self.stack.enter_context(mock.patch.object(config, "rh_keys", return_value=list(keys)))
        self.stack.enter_context(mock.patch.object(rh.time, "sleep"))
        self.graphs = self.stack.enter_context(mock.patch.object(rh.Scheduler, "_graph", return_value=({}, "fake-workflow")))
        self.calls = self.stack.enter_context(mock.patch.object(rh, "call", side_effect=responder))

        def download(url, out):
            out.write_bytes(b"fake-video")
            return True

        self.stack.enter_context(mock.patch.object(rh, "download", side_effect=download))

    def output(self):
        return {"code": 0, "data": [{"fileUrl": "https://example.invalid/result.mp4", "fileType": "mp4",
                                     "consumeMoney": 0.25, "taskCostTime": 12}]}

    def fake_review(self):
        self.stack.enter_context(mock.patch.object(pipeline, "_backdrop_lost", return_value=0))
        self.stack.enter_context(mock.patch.object(pipeline, "_keys", return_value=[]))
        self.stack.enter_context(mock.patch.object(pipeline.vision, "review_take", return_value={"pass": True, "score": 8}))


class VisionCheckpoints(Fixture):
    def prepare_cast(self, count=2):
        self.job.state.pop("cast")
        specs = []
        for i in range(count):
            pic = self.root / f"角色{i}.jpg"
            Image.new("RGB", (4, 4), "red").save(pic)
            specs.append(f"女生{i}={pic}")
        self.job.save()
        self.stack.enter_context(mock.patch.object(config, "vision_key", return_value="fake-api-key"))
        return specs

    def answer(self, data):
        return {"choices": [{"message": {"content": json.dumps(data)}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    def test_each_character_is_saved_before_the_next_paid_call_and_reused(self):
        specs = self.prepare_cast()

        def first_run(*args, **kwargs):
            if request.call_count == 1:
                return self.answer({"gender": "woman", "desc": "black hair"})
            saved = self.reload().state
            self.assertEqual(saved["cast"][0]["desc"], "black hair")
            self.assertGreater(saved["vision_cny"], 0)
            raise RuntimeError("second visual call interrupted")

        with mock.patch.object(ai, "_req", side_effect=first_run) as request:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                pipeline.stage_cast(self.job, specs)
        saved_cost = self.reload().state["vision_cny"]
        with mock.patch.object(ai, "_req", return_value=self.answer({"gender": "woman", "desc": "red jacket"})) as request, \
                mock.patch.object(pipeline, "_keys", side_effect=RuntimeError("ffmpeg interrupted")):
            resumed = self.reload()
            with self.assertRaisesRegex(RuntimeError, "ffmpeg"):
                pipeline.stage_cast(resumed, specs)
        request.assert_called_once()                             # only the second role needed describing
        self.assertEqual([c["desc"] for c in self.reload().state["cast"]], ["black hair", "red jacket"])
        self.assertGreater(self.reload().state["vision_cny"], saved_cost)
        self.assertEqual(self.reload().state["vision_cny"], ai.vision_spent())

    def test_performer_results_and_cost_survive_later_failure_without_repeating_calls(self):
        specs = self.prepare_cast(1)
        replies = [self.answer({"gender": "woman", "desc": "black hair"}),
                   self.answer({"P1": {"gender": "woman", "desc": "the blonde woman"},
                                "style": {"kind": "live", "desc": "a live dance"}})]
        frame = self.root / "角色0.jpg"
        original_log = self.job.log

        def fail_after_performers(event, **fields):
            if event == "style":
                saved = self.reload().state
                self.assertEqual(saved["cast"][0]["src_desc"], "the blonde woman")
                self.assertEqual(saved["style"]["desc"], "a live dance")
                self.assertGreater(saved["vision_cny"], 0)
                raise RuntimeError("after performers")
            original_log(event, **fields)

        with mock.patch.object(ai, "_req", side_effect=replies) as request, \
                mock.patch.object(pipeline, "_keys", return_value=[frame]), \
                mock.patch.object(self.job, "log", side_effect=fail_after_performers):
            with self.assertRaisesRegex(RuntimeError, "after performers"):
                pipeline.stage_cast(self.job, specs)
        self.assertEqual(request.call_count, 2)
        cost = self.reload().state["vision_cny"]
        pipeline.stage_cast(self.reload(), specs)                 # the default fake APIMart raises if called
        self.assertEqual(self.reload().state["vision_cny"], cost)

    def test_finally_keeps_cost_when_a_billed_reply_cannot_be_parsed(self):
        specs = self.prepare_cast(1)
        reply = self.answer({})
        reply["choices"][0]["message"]["content"] = "invalid JSON"
        with mock.patch.object(ai, "_req", return_value=reply) as request:
            with self.assertRaisesRegex(RuntimeError, "no JSON"):
                pipeline.stage_cast(self.job, specs)
        self.assertEqual(request.call_count, 2)
        self.assertGreater(self.reload().state["vision_cny"], 0)

    def test_parallel_visual_success_is_saved_before_an_earlier_call_fails_and_is_reused(self):
        self.shot.pop("kind")
        second = {**self.shot, "id": "s002"}
        self.job.state["shots"].append(second)
        saved = Event()
        original_save = self.job.save

        def checkpoint():
            original_save()
            if second.get("kind"):
                saved.set()

        label = {"kind": "keep", "people": [], "framing": "wide"}

        def interrupted(cast, frames, style):
            if frames[0].name == "s001":
                self.assertTrue(saved.wait(5), "the completed call was not checkpointed independently")
                raise RuntimeError("another visual call failed")
            ai._spent["vision"] += 0.1
            return label

        with mock.patch.object(pipeline, "_keys", side_effect=lambda job, shot, *args: [Path(shot["id"])]), \
                mock.patch.object(self.job, "save", side_effect=checkpoint), \
                mock.patch.object(pipeline.vision, "label_shot", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "another visual"):
                pipeline.stage_label(self.job)
        self.assertEqual(self.reload().state["shots"][1]["kind"], "keep")
        self.assertEqual(self.reload().state["vision_cny"], 0.1)
        with mock.patch.object(pipeline, "_keys", return_value=[]), \
                mock.patch.object(pipeline.vision, "label_shot", return_value=label) as call:
            pipeline.stage_label(self.reload())
        call.assert_called_once()                                # the already-labelled second shot is free on resume

    def test_finished_reviews_are_saved_when_another_review_fails(self):
        self.shot["attempts"] = [self.attempt(1), self.attempt(2)]
        for a in self.shot["attempts"]:
            a.pop("review")
        self.stack.enter_context(mock.patch.object(pipeline, "_backdrop_lost", return_value=0))
        self.stack.enter_context(mock.patch.object(pipeline, "_keys", side_effect=lambda job, shot, video, offset, tag: [Path(tag)]))
        done = Event()
        original_save = self.job.save

        def checkpoint():
            original_save()
            if "review" in self.shot["attempts"][1]:
                done.set()

        def review(cast, people, src, take):
            if take[0].name == "t1":
                self.assertTrue(done.wait(5), "the finished review was not checkpointed independently")
                raise RuntimeError("review call failed")
            return {"pass": True, "score": 8}

        with mock.patch.object(self.job, "save", side_effect=checkpoint), \
                mock.patch.object(pipeline.vision, "review_take", side_effect=review):
            with self.assertRaisesRegex(RuntimeError, "review call failed"):
                pipeline.stage_review(self.job)
        saved = self.reload().state["shots"][0]["attempts"]
        self.assertNotIn("review", saved[0])
        self.assertTrue(saved[1]["review"]["pass"])


class GenerateResume(Fixture):
    def test_last_submitted_attempt_is_collected_even_with_lowered_limit(self):
        self.shot["attempts"] = [self.attempt(1)]
        self.stack.enter_context(mock.patch.object(media, "cut_clip"))
        self.fake_review()

        def interrupt(key, path, body):
            if path.endswith("/create"):
                return {"code": 0, "data": {"taskId": "generation-2"}}
            saved = self.reload().state["shots"][0]["attempts"][-1]
            self.assertEqual((saved["status"], saved["task"], saved["key"]), ("submitted", "generation-2", "k1"))
            raise KeyboardInterrupt()

        self.fake_scheduler(interrupt)
        with self.assertRaises(KeyboardInterrupt):
            pipeline.stage_generate(self.job, 2)
        for limit in (2, 1, 0):
            with self.subTest(limit=limit):
                resumed = self.reload()

                def collect(key, path, body):
                    self.assertTrue(path.endswith("/outputs"))
                    self.assertEqual(body, {"taskId": "generation-2"})
                    return self.output()

                with mock.patch.object(rh, "call", side_effect=collect) as calls, \
                        mock.patch.object(media, "cut_clip") as cut:
                    pipeline.stage_generate(resumed, limit)
                calls.assert_called_once()
                cut.assert_not_called()
                self.assertEqual(resumed.state["shots"][0]["chosen"], 2)
                self.assertEqual(len(resumed.state["shots"][0]["attempts"]), 2)
                # Keep the interrupted checkpoint for the next limit's independent replay.
                self.job.save()
        self.assertEqual(self.graphs.call_count, 1)

    def test_exhausted_failed_attempts_do_not_submit_more(self):
        self.shot["attempts"] = [self.attempt(1), self.attempt(2)]
        with mock.patch.object(rh, "Scheduler") as scheduler:
            pipeline.stage_generate(self.job, 2)
        scheduler.assert_not_called()
        self.assertEqual(self.shot["chosen"], 1)

    def test_reserved_pending_last_attempt_can_still_be_submitted(self):
        pending = self.attempt(1)
        pending.pop("review")
        pending["status"] = "pending"
        self.shot["attempts"] = [pending]
        self.fake_review()
        self.stack.enter_context(mock.patch.object(media, "cut_clip"))

        def respond(key, path, body):
            return {"code": 0, "data": {"taskId": "reserved-1"}} if path.endswith("/create") else self.output()

        self.fake_scheduler(respond)
        pipeline.stage_generate(self.job, 1)
        self.assertEqual(self.shot["chosen"], 1)
        self.assertEqual(self.graphs.call_count, 1)
        self.assertEqual(len(self.shot["attempts"]), 1)

    def test_remote_tasks_are_collected_before_new_shots_are_submitted(self):
        pending = self.attempt(1)
        pending.pop("review")
        pending.update(status="submitted", task="old-task", key="k1")
        self.shot["attempts"] = [pending]
        second = {**self.shot, "id": "s002", "f0": 24, "attempts": []}
        self.job.state["shots"].append(second)
        self.fake_review()
        self.stack.enter_context(mock.patch.object(media, "cut_clip"))
        paths = []

        def respond(key, path, body):
            paths.append(path)
            return {"code": 0, "data": {"taskId": "new-task"}} if path.endswith("/create") else self.output()

        self.fake_scheduler(respond)
        pipeline.stage_generate(self.job, 1)
        self.assertEqual(paths, ["/task/openapi/outputs", "/task/openapi/create", "/task/openapi/outputs"])


    def test_a_task_under_a_key_no_longer_configured_stops_instead_of_spinning(self):
        pending = self.attempt(1)
        pending.pop("review")
        pending.update(status="submitted", task="old-task", key="k2")
        self.shot["attempts"] = [pending]
        self.fake_scheduler(lambda key, path, body: self.fail("nothing may be submitted or polled"))
        with mock.patch.object(rh.Scheduler, "run", side_effect=AssertionError("scheduler must not run")):
            with self.assertRaisesRegex(SystemExit, "k2"):
                pipeline.stage_generate(self.job, 3)
        self.assertEqual(self.reload().state["shots"][0]["attempts"][0]["task"], "old-task")


class UpscaleResume(Fixture):
    def prepare_upscale(self):
        self.shot["attempts"] = [self.attempt(1, passed=True)]
        self.shot["chosen"] = 1
        self.stack.enter_context(mock.patch.object(media, "ff"))
        self.stack.enter_context(mock.patch.object(media, "frame_count", return_value=24))
        self.colour = self.stack.enter_context(mock.patch.object(pipeline, "_colour_fix", return_value=[1, 0, 1, 0, 1, 0]))

    def test_submitted_task_and_key_are_saved_and_resume_only_collects(self):
        self.prepare_upscale()

        def interrupt(key, path, body):
            if path.endswith("/create"):
                return {"code": 0, "data": {"taskId": "upscale-1"}}
            saved = self.reload().state["shots"][0]["upscaled"]
            self.assertEqual((saved["status"], saved["task"], saved["key"]), ("submitted", "upscale-1", "k1"))
            raise KeyboardInterrupt()

        self.fake_scheduler(interrupt)
        with self.assertRaises(KeyboardInterrupt):
            pipeline.stage_upscale(self.job)

        def collect(key, path, body):
            self.assertTrue(path.endswith("/outputs"))
            self.assertEqual(body, {"taskId": "upscale-1"})
            return self.output()

        with mock.patch.object(rh, "call", side_effect=collect) as calls:
            resumed = self.reload()
            pipeline.stage_upscale(resumed)
            pipeline.stage_upscale(resumed)
        calls.assert_called_once()
        self.assertEqual(self.graphs.call_count, 1)
        up = self.reload().state["shots"][0]["upscaled"]
        self.assertEqual((up["status"], up["cny"], up["task"]), ("done", 0.25, "upscale-1"))
        self.assertTrue(Path(up["file"]).is_file())

    def test_downloaded_result_and_cost_survive_local_colour_fix_failure(self):
        self.prepare_upscale()

        def respond(key, path, body):
            return {"code": 0, "data": {"taskId": "upscale-1"}} if path.endswith("/create") else self.output()

        self.fake_scheduler(respond)
        self.colour.side_effect = RuntimeError("local processing interrupted")
        with self.assertRaisesRegex(RuntimeError, "local processing"):
            pipeline.stage_upscale(self.job)
        up = self.reload().state["shots"][0]["upscaled"]
        self.assertEqual((up["status"], up["cny"]), ("downloaded", 0.25))
        self.colour.side_effect = None
        with mock.patch.object(rh, "Scheduler") as scheduler:
            pipeline.stage_upscale(self.reload())
        scheduler.assert_not_called()
        self.assertEqual(self.reload().state["shots"][0]["upscaled"]["status"], "done")

    def test_an_upscale_under_a_key_no_longer_configured_stops_instead_of_resubmitting(self):
        self.prepare_upscale()
        self.shot["upscaled"] = {"n": 1, "status": "submitted", "task": "upscale-1", "key": "k2",
                                 "out": str(self.job.dir / "upscale" / "s001-1-up.mp4")}
        self.fake_scheduler(lambda key, path, body: self.fail("nothing may be submitted or polled"))
        with self.assertRaisesRegex(SystemExit, "k2"):
            pipeline.stage_upscale(self.job)
        self.assertEqual(self.reload().state["shots"][0]["upscaled"]["task"], "upscale-1")

    def test_upscales_recorded_before_this_change_are_not_redone(self):
        self.prepare_upscale()
        self.shot["upscaled"] = {"n": 1, "file": str(self.root / "old-up.mp4"), "cny": 0.1, "run_s": 9, "fix": [1, 0, 1, 0, 1, 0]}
        second = {**self.shot, "id": "s002", "upscaled": {"n": 1, "failed": "超分只出了 20/24 帧，用没超分的版本"}}
        self.job.state["shots"].append(second)
        with mock.patch.object(rh, "Scheduler") as scheduler:
            pipeline.stage_upscale(self.job)
        scheduler.assert_not_called()


class ReviewGate(Fixture):
    def run_pipeline(self, accept_failed=False, max_attempts=2):
        # Keep the real generation/review gate, assembly and reporting; bypass completed early stages.
        names = ("stage_prepare", "stage_shots", "stage_cast", "stage_label", "stage_backdrops", "stage_graphic")
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(mock.patch.object(pipeline, name))
            stack.enter_context(mock.patch.object(media, "ff"))
            stack.enter_context(mock.patch.object(media, "frame_count", return_value=24))
            return pipeline.run(self.job.dir, None, [], max_attempts=max_attempts, accept_failed=accept_failed)

    def test_any_failed_shot_blocks_all_upscale_and_accept_resumes_without_new_generation(self):
        self.shot["attempts"] = [self.attempt(1), self.attempt(2)]
        passed = {**self.shot, "id": "s002", "attempts": [self.attempt(1, passed=True)]}
        self.job.state["shots"].append(passed)
        self.job.save()
        with mock.patch.object(pipeline, "stage_upscale") as upscale:
            result = self.run_pipeline()
        upscale.assert_not_called()
        self.assertEqual(Path(result["compare"]).name, "compare.mp4")
        report = (self.job.dir / "report.md").read_text()
        self.assertIn("诊断用成片", report)
        self.assertIn("s001（2 个版本）", report)
        self.assertIn("原女生仍在", report)
        self.assertIn("--accept-failed", report)
        self.assertNotIn("三次都没过审", report)
        with mock.patch.object(pipeline, "stage_upscale") as upscale, mock.patch.object(rh, "Scheduler") as scheduler:
            self.run_pipeline(accept_failed=True, max_attempts=3)
        upscale.assert_called_once()
        scheduler.assert_not_called()                            # accepting does not buy a third generation
        saved = self.reload().state
        self.assertEqual(len(saved["shots"][0]["attempts"]), 2)
        self.assertEqual(saved["review_blocked"], [])
        self.assertTrue(saved["accept_failed"])
        self.assertIn("--accept-failed", (self.job.dir / "report.md").read_text())
        self.assertNotIn("诊断用成片", (self.job.dir / "report.md").read_text())

    def test_all_passed_shots_continue_to_upscale(self):
        self.shot["attempts"] = [self.attempt(1, passed=True)]
        self.job.save()
        with mock.patch.object(pipeline, "stage_upscale") as upscale:
            self.run_pipeline()
        upscale.assert_called_once()

    def test_no_upscale_job_is_not_held_for_acceptance(self):
        self.job.state["args"]["upscale"] = False
        self.shot["attempts"] = [self.attempt(1), self.attempt(2)]
        self.job.save()
        with mock.patch.object(pipeline, "stage_upscale") as upscale:
            self.run_pipeline()
        upscale.assert_not_called()
        self.assertEqual(self.reload().state["review_blocked"], [])
        report = (self.job.dir / "report.md").read_text()
        self.assertIn("s001（2 个版本）", report)
        self.assertNotIn("诊断用成片", report)
        self.assertNotIn("--accept-failed", report)

    def test_accepted_diagnostic_interrupted_during_upscale_resumes_the_original_task(self):
        self.shot["attempts"] = [self.attempt(1), self.attempt(2)]
        self.job.save()
        self.run_pipeline()

        def interrupt(key, path, body):
            if path.endswith("/create"):
                return {"code": 0, "data": {"taskId": "accepted-upscale"}}
            raise KeyboardInterrupt()

        self.fake_scheduler(interrupt)
        self.stack.enter_context(mock.patch.object(pipeline, "_colour_fix", return_value=[1, 0, 1, 0, 1, 0]))
        with self.assertRaises(KeyboardInterrupt):
            self.run_pipeline(accept_failed=True, max_attempts=3)
        self.assertTrue(self.reload().state["accept_failed"])

        def collect(key, path, body):
            self.assertTrue(path.endswith("/outputs"))
            self.assertEqual(body["taskId"], "accepted-upscale")
            return self.output()

        with mock.patch.object(rh, "call", side_effect=collect) as calls:
            self.run_pipeline(max_attempts=3)                    # the saved acceptance survives a plain resume
        calls.assert_called_once()
        self.assertEqual(self.graphs.call_count, 1)
        saved = self.reload().state["shots"][0]
        self.assertEqual(len(saved["attempts"]), 2)
        self.assertEqual(saved["upscaled"]["status"], "done")

    def test_missing_generation_uses_source_in_free_diagnostic(self):
        self.job.save()
        with mock.patch.object(pipeline, "stage_upscale") as upscale:
            self.run_pipeline(max_attempts=0)
        upscale.assert_not_called()
        report = (self.job.dir / "report.md").read_text()
        self.assertIn("没有生成结果，片里暂用原片", report)

    def run_cli(self, *extra):
        argv = [str(ENTRY), "run", str(self.job.dir), *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "path", sys.path.copy()), \
                mock.patch.object(pipeline, "run") as run:
            runpy.run_path(str(ENTRY), run_name="__main__")
        return run.call_args

    def test_cli_accept_failed_reaches_pipeline_and_the_old_call_is_unchanged(self):
        self.assertTrue(self.run_cli("--accept-failed").kwargs["accept_failed"])
        call = self.run_cli("--max-attempts", "2")
        self.assertFalse(call.kwargs["accept_failed"])
        self.assertEqual(call.args[1:], (None, [], 0, None, False, 2, "auto", True, False))

    def test_report_layout_matches_the_actual_stack_for_each_orientation(self):
        self.shot["kind"] = "keep"
        for width, height, stack, layout in ((1080, 1920, "hstack", "左原片右成片"),
                                             (1920, 1080, "vstack", "上原片下成片"),
                                             (1080, 1080, "vstack", "上原片下成片")):
            with self.subTest(width=width, height=height), mock.patch.object(media, "ff") as ff, \
                    mock.patch.object(media, "frame_count", return_value=24):
                self.job.state["prepared"].update(width=width, height=height)
                finish.assemble(self.job.state, self.job.dir)
                compare = next(c.args for c in ff.call_args_list if c.args[-1].endswith("compare.mp4"))
                self.assertIn(stack, compare[compare.index("-filter_complex") + 1])
                self.assertIn(layout, finish.report(self.job.state, self.job.dir).read_text())


class FrameCompatibility(Fixture):
    def real_ffmpeg(self):
        """Put the chosen ffmpeg/ffprobe first on PATH; media.py calls them by name."""
        return mock.patch.dict(os.environ, {"PATH": str(FFMPEG.parent) + os.pathsep + os.environ.get("PATH", "")})

    def test_both_extraction_commands_use_passthrough_without_vsync(self):
        with mock.patch.object(media, "ff") as ff:
            media.keyframes(Path("source.mp4"), [5], self.root, "test")
        command = ff.call_args.args
        self.assertNotIn("-vsync", command)
        self.assertEqual(command[command.index("-fps_mode") + 1], "passthrough")
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(stdout=b"")) as run:
            media.raw_frames(Path("source.mp4"), 1)
        command = run.call_args.args[0]
        self.assertNotIn("-vsync", command)
        self.assertEqual(command[command.index("-fps_mode") + 1], "passthrough")

    @unittest.skipUnless(FFMPEG.is_file(), "no local ffmpeg")
    def test_real_ffmpeg_extracts_a_jpeg_and_rgb_frame(self):
        with self.real_ffmpeg():
            version = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True).stdout
            self.assertIn("ffmpeg version", version)
            src = self.root / "source.mp4"
            media.ff("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=24:duration=1", "-an", "-c:v", "libx264", str(src))
            frames = media.keyframes(src, [5], self.root, "real", width=96)
            self.assertEqual(len(frames), 1)
            with Image.open(frames[0]) as frame:
                self.assertEqual(frame.size, (96, 64))
            self.assertEqual(media.raw_frames(src, 1, start=5).shape, (1, 54, 96, 3))

    @unittest.skipUnless(FFMPEG.is_file() and FFMPEG.with_name("ffprobe").is_file(), "no local ffmpeg/ffprobe")
    def test_failed_review_really_produces_decodable_free_diagnostic_outputs(self):
        with self.real_ffmpeg(), mock.patch.object(config, "FINAL_LONG_SIDE", 96):
            src, take = self.root / "source.mp4", self.root / "take.mp4"
            for path, colour in ((src, "red"), (take, "blue")):
                media.ff("-f", "lavfi", "-i", f"color=c={colour}:size=64x96:rate=24:duration=1", "-an", "-c:v", "libx264", str(path))
            self.job.state["prepared"].update(mv=str(src), width=64, height=96)
            self.shot["backdrop"] = "a room"
            self.shot["attempts"] = [{**self.attempt(1), "file": str(take)}]
            self.job.save()
            with mock.patch.object(pipeline, "stage_upscale") as upscale, mock.patch.object(rh, "Scheduler") as scheduler:
                result = pipeline.run(self.job.dir, None, [], max_attempts=1)
            upscale.assert_not_called()
            scheduler.assert_not_called()
            self.assertEqual((result["frames"], result["want"]), (24, 24))
            for kind in ("film", "compare", "sheet"):
                self.assertGreater(Path(result[kind]).stat().st_size, 0)
            for kind in ("film", "compare"):
                media.ff("-i", result[kind], "-f", "null", "-")
            compare = media.raw_frames(Path(result["compare"]), 1, w=128, h=96)[0]
            self.assertGreater(compare[:, :60, 0].mean(), 200)     # the red source is on the left
            self.assertGreater(compare[:, 68:, 2].mean(), 200)    # the blue diagnostic take is on the right
            report = (self.job.dir / "report.md").read_text()
            self.assertIn("--accept-failed", report)
            self.assertIn("左原片右成片", report)


if __name__ == "__main__":
    unittest.main()
