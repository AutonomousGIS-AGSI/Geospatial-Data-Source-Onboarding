"""Tests for the mechanical/manual validation instrument.

The behaviour these lock down is the reason the module exists: an
unverifiable requirement must surface as "undecidable" and route to a human,
never collapse into a pass, and a human verdict must survive a later
LLM-judge run over the same test run.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_agent import validation as V
from WebUI import handbook_studio_store as store
from WebUI import handbook_studio_runner as runner


# The WorldPop Kenya run from the case study: a real 179 MB GeoTIFF covering
# Kenya, correctly unconstrained, correctly 2025 -- but the task also asked
# for the UN-adjusted product, which is not encoded anywhere in the path.
KENYA_RESULT = {
    "status": "passed",
    "validation": {"passed": True, "output_present": True, "format_match": True},
    "output_evidence": [
        {"name": "ken_pop_2025_UC_100m_R2024B_v1.metadata.json",
         "extension": ".json", "size_bytes": 852},
        {"name": "ken_pop_2025_UC_100m_R2024B_v1.tif", "extension": ".tif",
         "size_bytes": 179024303, "bands": 1, "crs": "EPSG:4326",
         "bbox": {"west": 33.90999914436, "south": -4.737499645050004,
                  "east": 41.90666577904, "north": 5.034166982529996}},
    ],
    "http_requests": [
        {"attempt": 1, "method": "GET", "url":
         "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/"
         "2025/KEN/v1/100m/unconstrained/ken_pop_2025_UC_100m_R2024B_v1.tif"},
    ],
    "downloaded_files": [],
}

KENYA_SPEC = """
[expect.output]
format = "GeoTIFF"
count_min = 1
bands = 1

[expect.spatial]
place = "Kenya"
mode = "covers"

[expect.temporal]
year = 2025

[expect.dataset]
filename_must_match = ["_UC_"]
filename_must_not_match = ["/constrained/"]
undecidable_claims = ["UN-adjusted"]
"""

# Kenya's real extent, so the spatial check never touches the network in
# tests (reference_geometry consults the local index/cache first).
KENYA_BBOX = {"west": 33.89, "south": -4.72, "east": 41.92, "north": 5.03}


def _local_kenya(place, allow_network=True):
    from shapely.geometry import box
    if place.strip().lower() != "kenya":
        return None, f"no fixture for {place}"
    return box(KENYA_BBOX["west"], KENYA_BBOX["south"],
               KENYA_BBOX["east"], KENYA_BBOX["north"]), "test fixture"


class VerdictAlgebraTests(unittest.TestCase):
    def test_undecidable_never_rolls_up_to_pass(self):
        self.assertEqual(
            V.roll_up({"a": V.PASS, "b": V.UNDECIDABLE, "c": V.NA}),
            V.UNDECIDABLE)

    def test_fail_dominates_undecidable(self):
        self.assertEqual(
            V.roll_up({"a": V.FAIL, "b": V.UNDECIDABLE}), V.FAIL)

    def test_nothing_checked_is_undecidable_not_pass(self):
        """The specific defect in the old metadata checker: when every
        dimension was not-applicable it reported task_completed=True."""
        self.assertEqual(V.roll_up({k: V.NA for k in V.DIMENSIONS}),
                         V.UNDECIDABLE)

    def test_all_applicable_pass_is_a_pass(self):
        self.assertEqual(V.roll_up({"a": V.PASS, "b": V.NA}), V.PASS)

    def test_undecidable_record_is_not_task_completed(self):
        record = V.build_record("mechanical", V.UNDECIDABLE,
                                {"output_correctness": V.PASS}, [], "")
        self.assertFalse(record["task_completed"])
        self.assertFalse(record["should_refine_handbook"])


class TaskSpecTests(unittest.TestCase):
    def test_expect_prefix_is_flattened(self):
        spec = V.parse_task_spec('[expect.spatial]\nplace = "Kenya"\n')
        self.assertEqual(spec["spatial"]["place"], "Kenya")

    def test_json_specs_are_accepted(self):
        spec = V.parse_task_spec('{"spatial": {"place": "Kenya"}}')
        self.assertEqual(spec["spatial"]["place"], "Kenya")

    def test_malformed_spec_raises_rather_than_silently_emptying(self):
        with self.assertRaises(ValueError):
            V.parse_task_spec("[expect.spatial\nplace = ")

    def test_shipped_template_parses(self):
        self.assertIn("output", V.parse_task_spec(V.SPEC_TEMPLATE))


class MechanicalCheckerTests(unittest.TestCase):
    def _evaluate(self, spec=KENYA_SPEC, result=None):
        with mock.patch.object(V, "reference_geometry", _local_kenya):
            return V.evaluate_mechanical(
                "Download the WorldPop unconstrained, UN-adjusted 2025 "
                "population count raster for Kenya and save the GeoTIFF",
                result or KENYA_RESULT, spec_text=spec, mechanism="http_file")

    def test_unverifiable_claim_makes_the_run_undecidable(self):
        record = self._evaluate()
        self.assertEqual(record["verdict"], V.UNDECIDABLE)
        self.assertFalse(record["task_completed"])
        self.assertEqual(record["dimensions"]["dataset_selection"],
                         V.UNDECIDABLE)

    def test_decidable_dimensions_still_report_their_own_verdicts(self):
        record = self._evaluate()
        for dimension in ("output_correctness", "spatial_query_correctness",
                          "temporal_query_correctness"):
            self.assertEqual(record["dimensions"][dimension], V.PASS, dimension)

    def test_no_spec_reports_undecidable_rather_than_passing(self):
        record = self._evaluate(spec="")
        self.assertEqual(record["verdict"], V.UNDECIDABLE)
        self.assertFalse(record["spec_declared"])

    def test_wrong_place_fails_spatially(self):
        result = dict(KENYA_RESULT)
        result["output_evidence"] = [dict(
            KENYA_RESULT["output_evidence"][1],
            bbox={"west": -10.0, "south": 40.0, "east": -5.0, "north": 45.0})]
        record = self._evaluate(result=result)
        self.assertEqual(record["dimensions"]["spatial_query_correctness"],
                         V.FAIL)
        self.assertEqual(record["verdict"], V.FAIL)

    def test_a_sliver_inside_the_place_does_not_satisfy_covers(self):
        """The old check asked only whether the bboxes overlapped, so a
        single tile inside Kenya passed a whole-country request."""
        result = dict(KENYA_RESULT)
        result["output_evidence"] = [dict(
            KENYA_RESULT["output_evidence"][1],
            bbox={"west": 36.7, "south": -1.4, "east": 37.0, "north": -1.1})]
        record = self._evaluate(result=result)
        self.assertEqual(record["dimensions"]["spatial_query_correctness"],
                         V.FAIL)

    def test_wrong_year_fails_temporally(self):
        spec = KENYA_SPEC.replace("year = 2025", "year = 2020")
        record = self._evaluate(spec=spec)
        self.assertEqual(record["dimensions"]["temporal_query_correctness"],
                         V.FAIL)

    def test_forbidden_variant_in_the_captured_url_fails(self):
        spec = KENYA_SPEC.replace('filename_must_not_match = ["/constrained/"]',
                                  'filename_must_not_match = ["unconstrained"]')
        record = self._evaluate(spec=spec)
        self.assertEqual(record["dimensions"]["dataset_selection"], V.FAIL)

    def test_parameters_are_read_from_captured_requests(self):
        result = dict(KENYA_RESULT)
        result["http_requests"] = [
            {"url": "https://example.org/wfs?service=WFS&bbox=1,2,3,4"}]
        record = self._evaluate(
            spec='[expect.parameters]\nrequired = { service = "WFS", '
                 'bbox = "*" }\nforbidden = ["apikey"]\n',
            result=result)
        self.assertEqual(record["dimensions"]["parameter_correctness"], V.PASS)

    def test_missing_required_parameter_fails(self):
        result = dict(KENYA_RESULT)
        result["http_requests"] = [{"url": "https://example.org/wfs?service=WFS"}]
        record = self._evaluate(
            spec='[expect.parameters]\nrequired = { bbox = "*" }\n',
            result=result)
        self.assertEqual(record["dimensions"]["parameter_correctness"], V.FAIL)

    def test_provenance_hashes_the_file_on_disk(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "test_1"))
            path = os.path.join(root, "test_1", "artifact.tif")
            with open(path, "wb") as handle:
                handle.write(b"raster bytes")
            result = dict(KENYA_RESULT)
            result["downloaded_files"] = [
                {"name": "artifact.tif", "artifact_ref": "test_1/artifact.tif"}]
            with mock.patch.object(V, "reference_geometry", _local_kenya):
                record = V.evaluate_mechanical(
                    "task", result, spec_text='[expect.provenance]\n'
                    'verify_remote = false\n', artifact_root=root)
        self.assertEqual(record["dimensions"]["provenance"], V.PASS)
        self.assertIn("sha256", record["checks"][-1]["evidence"])

    def test_missing_artifact_is_undecidable_not_a_pass(self):
        result = dict(KENYA_RESULT)
        result["downloaded_files"] = [
            {"name": "gone.tif", "artifact_ref": "test_1/gone.tif"}]
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(V, "reference_geometry", _local_kenya):
                record = V.evaluate_mechanical(
                    "task", result, spec_text="[expect.provenance]\n",
                    artifact_root=root)
        self.assertEqual(record["dimensions"]["provenance"], V.UNDECIDABLE)


class ManualRubricTests(unittest.TestCase):
    def test_per_dimension_verdicts_are_retained(self):
        record = V.manual_record({
            "dimensions": {"dataset_selection": "pass",
                           "spatial_query_correctness": "fail"},
            "evidence": {"dataset_selection": "checked the release statement"},
            "rater": "R1", "blinded": True, "elapsed_seconds": 90.0})
        self.assertEqual(record["dataset_selection"], V.PASS)
        self.assertEqual(record["spatial_query_correctness"], V.FAIL)
        self.assertEqual(record["verdict"], V.FAIL)
        self.assertEqual(record["rater"], "R1")
        self.assertTrue(record["blinded"])
        self.assertEqual(record["elapsed_seconds"], 90.0)

    def test_cited_evidence_becomes_a_check(self):
        record = V.manual_record({
            "dimensions": {"dataset_selection": "pass"},
            "evidence": {"dataset_selection": "release statement, page 3"}})
        self.assertIn("release statement, page 3",
                      record["checks"][0]["evidence"])

    def test_explicit_overall_verdict_overrides_the_roll_up(self):
        record = V.manual_record({
            "dimensions": {"output_correctness": "pass"},
            "verdict": "undecidable"})
        self.assertEqual(record["verdict"], V.UNDECIDABLE)


class PrecedenceTests(unittest.TestCase):
    def test_human_verdict_outranks_mechanical_and_llm(self):
        log = [V.build_record("mechanical", V.UNDECIDABLE, {}, [], ""),
               V.build_record("manual", V.PASS, {}, [], ""),
               V.build_record("llm", V.FAIL, {}, [], "")]
        self.assertEqual(V.primary_validation(log)["method"], "manual")

    def test_mechanical_outranks_the_llm_judge(self):
        log = [V.build_record("llm", V.PASS, {}, [], ""),
               V.build_record("mechanical", V.FAIL, {}, [], "")]
        self.assertEqual(V.primary_validation(log)["method"], "mechanical")

    def test_latest_record_wins_within_one_method(self):
        log = [V.build_record("manual", V.FAIL, {}, [], "first"),
               V.build_record("manual", V.PASS, {}, [], "second")]
        self.assertEqual(V.primary_validation(log)["summary"], "second")


class AppendOnlyStoreTests(unittest.TestCase):
    """The data-integrity fix: validation used to be one slot, so a second
    method silently destroyed the first verdict."""

    def _session(self, temp_root):
        session = store.new_session("append-only")
        session.update({
            "source_name": "WorldPop", "retrieval_task": "task",
            "generated_source": {"data_source_name": "WorldPop"},
            "status": "complete", "access_mechanism": "http_file",
            "validation_spec": KENYA_SPEC,
            "test_runs": [{"id": "run-1", "test_number": 1,
                           "result": dict(KENYA_RESULT)}],
        })
        return store.save_session("d" * 64, session)

    def test_two_methods_produce_two_records_and_the_human_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.object(store, "STORE_ROOT",
                                   os.path.join(temp_root, "sessions")), \
                 mock.patch.object(runner, "OUTPUT_ROOT",
                                   os.path.join(temp_root, "outputs")), \
                 mock.patch.object(V, "reference_geometry", _local_kenya):
                saved = self._session(temp_root)
                user_id = "d" * 64
                list(runner.run_validation_stream(
                    user_id, "sk-x", saved["id"], "run-1", "mechanical"))
                list(runner.run_validation_stream(
                    user_id, "sk-x", saved["id"], "run-1", "manual",
                    manual_verdict={"dimensions": {"dataset_selection": "pass"},
                                    "rater": "R1"}))
                final = store.get_session(user_id, saved["id"])

        result = final["test_runs"][0]["result"]
        self.assertEqual([v["method"] for v in result["validations"]],
                         ["mechanical", "manual"])
        self.assertEqual(result["validation_method"], "manual")
        # The mechanical verdict is still readable, unchanged.
        self.assertEqual(result["validations"][0]["verdict"], V.UNDECIDABLE)

    def test_legacy_single_slot_verdict_is_migrated_not_dropped(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.object(store, "STORE_ROOT",
                                   os.path.join(temp_root, "sessions")):
                session = store.new_session("legacy")
                session.update({
                    "source_name": "WorldPop", "status": "complete",
                    "test_runs": [{"id": "run-1", "test_number": 1, "result": {
                        "validation": {"output_present": True},
                        "semantic_validation": {"task_completed": True,
                                                "summary": "old verdict"},
                        "validation_method": "llm"}}]})
                saved = store.save_session("c" * 64, session)

        log = saved["test_runs"][0]["result"]["validations"]
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["summary"], "old verdict")
        self.assertTrue(log[0]["migrated_from_legacy_slot"])

    def test_malformed_spec_is_rejected_at_save_time(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.object(store, "STORE_ROOT",
                                   os.path.join(temp_root, "sessions")):
                session = store.new_session("bad spec")
                session.update({"source_name": "X",
                                "validation_spec": "[expect.spatial\nplace ="})
                with self.assertRaises(ValueError):
                    store.save_session("b" * 64, session)

    def test_refinement_refuses_an_undecidable_verdict(self):
        """Revising a handbook against a non-finding is how a requirement
        gets argued away instead of met."""
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.object(store, "STORE_ROOT",
                                   os.path.join(temp_root, "sessions")), \
                 mock.patch.object(runner, "OUTPUT_ROOT",
                                   os.path.join(temp_root, "outputs")), \
                 mock.patch.object(V, "reference_geometry", _local_kenya):
                saved = self._session(temp_root)
                user_id = "d" * 64
                list(runner.run_validation_stream(
                    user_id, "sk-x", saved["id"], "run-1", "mechanical"))
                events = list(runner.run_validation_refine_stream(
                    user_id, "sk-x", saved["id"], "run-1"))

        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("undecidable", events[-1]["error"].lower())


if __name__ == "__main__":
    unittest.main()
