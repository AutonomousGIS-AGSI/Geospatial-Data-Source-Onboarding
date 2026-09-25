import json
import os
import tempfile
import unittest
from unittest import mock

from WebUI import handbook_studio_runner as runner
from WebUI import handbook_studio_store as store
from agents.data_agent import handbook_generator


def exhaust(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as done:
            return events, done.value


def fake_generate_handbook(query, **kwargs):
    from agents.data_agent.handbook_generator import _log
    _log("[stage:source:start] Finding the best official source for your request.")
    _log("[stage:source:complete] Selected OpenTopography from OpenTopography.org.")
    _log("[stage:docs:start] Reading the official API documentation and access requirements.")
    _log("[stage:docs:complete] Confirmed REST access.")
    _log("[stage:draft:start] Writing retrieval instructions and a runnable Python example.")
    _log("[stage:draft:complete] The handbook draft and sample code are ready.")
    _log("[stage:verify:start] Running a small sample download in an isolated process.")
    _log("[stage:verify:complete] The sample download completed successfully.")
    return {
        "data_source_name": "OpenTopography", "brief_description": "DEM tiles",
        "handbook": "1. Base URL...",
        "code_example": "def download_data(): pass\ndownload_data()",
        "website": "https://opentopography.org", "requires_key": "",
        "key_name": "", "caveats": "", "key_signup_url": "",
        "_verification": {
            "verified": True, "attempts": 1, "error": "", "files": [],
            "note": "The sample download completed successfully.",
        },
    }


def fake_select_access_mechanism(query, website, user_key=None, model=None,
                                 **kwargs):
    return {"mechanism": "rest", "why": "It is a documented REST API."}


def fake_evaluate_retrieval_result(current, task, execution_result, mechanism="",
                                   user_key=None, model=None, **kwargs):
    return {
        "task_completed": True, "confidence": 0.95,
        "summary": "The output satisfies the task.",
        "mechanism_correctness": "pass", "parameter_correctness": "pass",
        "spatial_query_correctness": "not_applicable",
        "temporal_query_correctness": "not_applicable",
        "output_correctness": "pass", "checks": [],
        "failure_category": "", "should_refine_handbook": False,
        "handbook_gap": "",
    }


class FakeRetriever:
    """Matches DataRetrieverAgent's constructor/trial-call shape without the
    real LLM/network work, mirroring test_experiment2_corrections.py."""

    def __init__(self, **kwargs):
        self.model = ""

    def run_controlled_handbook_trial(self, data_request, source_ID,
                                       source_name=None, user_keys=None,
                                       stream_callback=None, output_dir=None,
                                       try_count=10, attempt_timeout=None,
                                       use_handbook=True,
                                       required_key_names=None):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "result.geojson")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"type": "FeatureCollection", "features": []}, handle)
        if stream_callback:
            stream_callback("Generating retrieval code from the pinned handbook.")
            stream_callback("Executing the generated retrieval program.")
            stream_callback("Code executed successfully.")
        return {
            "status": "passed", "error": "", "source_id": source_ID,
            "generated_code": "def download_data(): pass",
            "execution": {"success": True, "attempts_used": 1},
            "http_requests": [{
                "attempt": 1, "method": "GET",
                "url": "https://example.test/data",
            }],
            "downloaded_files": [{
                "name": "result.geojson", "path": path,
                "size_bytes": os.path.getsize(path),
            }],
        }


class HandbookStudioStoreTests(unittest.TestCase):
    def test_extra_keys_in_generated_source_survive_a_save_load_cycle(self):
        # This is the exact regression reusing Experiment 2's whitelisting
        # _clean_record would cause: it only copies fields it already knows
        # about, so anything new attached to generated_source would vanish.
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as store_root:
            with mock.patch.object(store, "STORE_ROOT", store_root):
                session = store.new_session("Round trip test")
                session["generated_source"] = {
                    "data_source_name": "X", "_weird_extra_key": "kept?",
                }
                saved = store.save_session("user123", session)
                loaded = store.get_session("user123", saved["id"])
        self.assertEqual(
            loaded["generated_source"]["_weird_extra_key"], "kept?")


class HandbookStudioRunnerTests(unittest.TestCase):
    def test_generate_stream_auto_selects_mechanism_and_pauses_for_review(self):
        user_id = "f" * 64
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_root:
            store_root = os.path.join(temp_root, "sessions")
            output_root = os.path.join(temp_root, "outputs")
            with mock.patch.object(store, "STORE_ROOT", store_root), \
                    mock.patch.object(runner, "OUTPUT_ROOT", output_root), \
                    mock.patch.object(
                        handbook_generator, "select_access_mechanism",
                        side_effect=fake_select_access_mechanism), \
                    mock.patch.object(
                        handbook_generator, "generate_handbook",
                        side_effect=fake_generate_handbook):
                session = store.new_session("Gen test")
                session.update({
                    "source_name": "OpenTopography",
                    "retrieval_task": "Download a small area as GeoJSON.",
                })
                saved = store.save_session(user_id, session)
                events, _ = exhaust(runner.run_generate_stream(
                    user_id, "sk-test", saved["id"]))
                completed = store.get_session(user_id, saved["id"])

        terminal = events[-1]
        self.assertEqual(terminal["type"], "handbook_ready_for_review")
        self.assertEqual(completed["status"], "awaiting_review")
        self.assertEqual(completed["access_mechanism"], "rest")
        self.assertEqual(completed["access_mechanism_source"], "auto")
        self.assertEqual(
            completed["generated_source"]["data_source_name"],
            "OpenTopography")

    def test_test_stream_runs_one_controlled_trial_to_completion(self):
        user_id = "f" * 64
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_root:
            store_root = os.path.join(temp_root, "sessions")
            output_root = os.path.join(temp_root, "outputs")
            with mock.patch.object(store, "STORE_ROOT", store_root), \
                    mock.patch.object(runner, "OUTPUT_ROOT", output_root), \
                    mock.patch.object(
                        runner, "DataRetrieverAgent", FakeRetriever), \
                    mock.patch.object(
                        handbook_generator, "evaluate_retrieval_result",
                        side_effect=fake_evaluate_retrieval_result):
                session = store.new_session("Test phase")
                session.update({
                    "source_name": "OpenTopography",
                    "retrieval_task": "Download a small area as GeoJSON.",
                    "generated_source": {
                        "data_source_name": "OpenTopography",
                        "handbook": "1. Base URL...",
                        "code_example": "def download_data(): pass\ndownload_data()",
                        "requires_key": "", "key_name": "",
                    },
                    "status": "awaiting_review",
                    "access_mechanism": "rest",
                })
                saved = store.save_session(user_id, session)
                events, _ = exhaust(runner.run_test_stream(
                    user_id, "sk-test", saved["id"]))
                completed = store.get_session(user_id, saved["id"])

        terminal = events[-1]
        self.assertEqual(terminal["type"], "test_finished")
        self.assertEqual(terminal["outcome"], "passed")
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["test_result"]["status"], "passed")
        self.assertTrue(completed["test_result"]["downloaded_files"])
        semantic = completed["test_result"]["semantic_validation"]
        self.assertTrue(semantic["task_completed"])
        self.assertEqual(semantic["parameter_correctness"], "pass")
        self.assertEqual(
            completed["test_runs"][-1]["result"]["semantic_validation"],
            semantic)

    def test_semantic_validation_failure_marks_outcome_failed(self):
        user_id = "f" * 64

        def fake_evaluate_not_completed(*args, **kwargs):
            result = fake_evaluate_retrieval_result(*args, **kwargs)
            result.update({
                "task_completed": False,
                "failure_category": "Spatial validation",
                "summary": "The output was not filtered to the requested area.",
            })
            return result

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_root:
            store_root = os.path.join(temp_root, "sessions")
            output_root = os.path.join(temp_root, "outputs")
            with mock.patch.object(store, "STORE_ROOT", store_root), \
                    mock.patch.object(runner, "OUTPUT_ROOT", output_root), \
                    mock.patch.object(
                        runner, "DataRetrieverAgent", FakeRetriever), \
                    mock.patch.object(
                        handbook_generator, "evaluate_retrieval_result",
                        side_effect=fake_evaluate_not_completed):
                session = store.new_session("Test phase — semantic failure")
                session.update({
                    "source_name": "OpenTopography",
                    "retrieval_task": "Download a small area as GeoJSON.",
                    "generated_source": {
                        "data_source_name": "OpenTopography",
                        "handbook": "1. Base URL...",
                        "code_example": "def download_data(): pass\ndownload_data()",
                        "requires_key": "", "key_name": "",
                    },
                    "status": "awaiting_review",
                    "access_mechanism": "rest",
                })
                saved = store.save_session(user_id, session)
                events, _ = exhaust(runner.run_test_stream(
                    user_id, "sk-test", saved["id"]))
                completed = store.get_session(user_id, saved["id"])

        terminal = events[-1]
        self.assertEqual(terminal["outcome"], "failed")
        self.assertEqual(
            completed["test_result"]["status"], "failed_validation")
        self.assertFalse(
            completed["test_result"]["semantic_validation"]["task_completed"])


if __name__ == "__main__":
    unittest.main()
