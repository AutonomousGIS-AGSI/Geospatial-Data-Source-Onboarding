"""The repair ladder LLM_Find climbs when a download's debug loop is spent.

Before this existed there were two rungs and no triage: retry the code five
times, then throw the handbook away and re-research the source from its live
documentation. That fired on any exhausted failure, so a 503 or a rate limit
cost a full regeneration -- and could replace a working handbook with one
written against a service that merely happened to be down.

The ladder is now: diagnose, then revise the handbook we have, then re-research.
These tests pin the ORDER and the STOPPING RULES, which is where the cost and
the risk live. The LLM calls each rung makes are stubbed out; what is under
test is which rungs get climbed, with what evidence, and what gets persisted.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_agent import DataRetrieverAgent as agent_module
from agents.data_agent import handbook_generator
from agents.data_agent.DataRetrieverAgent import DataRetrieverAgent

HANDBOOK_TOML = '''data_source_name = "Test Source"
brief_description = "A source used only by tests."
handbook = """Call https://example.invalid/api and save the response as CSV."""
code_example = "print('original')"
website = "https://example.invalid/docs"
requires_key = "False"
key_name = ""
caveats = ""
key_signup_url = ""
'''

REVISED_BOOK = {
    "data_source_name": "Test Source",
    "brief_description": "A source used only by tests.",
    "handbook": "REVISED: call https://example.invalid/v2/api instead.",
    "code_example": "print('revised')",
    "website": "https://example.invalid/docs",
    "requires_key": "False",
    "key_name": "",
    "caveats": "",
    "key_signup_url": "",
}


class LadderAgent(DataRetrieverAgent):
    """DataRetrieverAgent with everything outside the repair ladder stubbed.

    Source selection, code generation and execution are replaced by scripted
    outcomes, and each rung records that it ran. Nothing here reaches a model
    or the network, so a test asserts on the sequence of decisions alone.
    """

    def __init__(self, *args, **kwargs):
        self.script = list(kwargs.pop("script", []))
        super().__init__(*args, **kwargs)
        self.calls = []
        self.prompts = []
        self.triage_verdict = {"failure_category": "handbook_deficiency",
                               "summary": "wrong endpoint"}
        self.refine_result = dict(REVISED_BOOK)
        self.research_result = None

    # -- stubbed surroundings -------------------------------------------------
    def create_select_prompt(self, task):
        return "select"

    def select_source(self, select_prompt_str, stream_callback=None):
        return "{'Selected data source': 'Test Source', 'Confidence': 9}"

    def generate_data_fetching_code(self, download_prompt_str, stream_callback=None):
        self.prompts.append(download_prompt_str)
        return "```python\npass\n```"

    def execute_complete_program(self, code, try_cnt, task, model_name,
                                 handbook_str, **kwargs):
        outcome = self.script.pop(0) if self.script else "fail"
        self.calls.append(f"execute:{outcome}")
        if outcome == "deliver":
            path = os.path.join(self.output_dir, "delivered.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("id,value\n1,2\n")
            self.last_execution_report = {"success": True, "traceback": ""}
        else:
            self.last_execution_report = {"success": False,
                                          "traceback": "HTTPError: 503"}
        return code

    # -- the rungs ------------------------------------------------------------
    def _triage_download_failure(self, source_ID, data_request, code,
                                 failure_note, http_requests, _stream, _warn):
        self.calls.append("triage")
        self.last_failure_note = failure_note
        return dict(self.triage_verdict)

    def _refine_skill_from_failure(self, source_ID, data_request, analysis,
                                   failure_note, exec_ok, _stream, _warn):
        self.calls.append("refine")
        self.refine_analysis = analysis
        return (dict(self.refine_result) if self.refine_result is not None
                else None)

    def _re_research_skill(self, source_name, source_ID, data_request,
                           user_keys, _stream, _warn, failure_note=""):
        self.calls.append("re_research")
        return self.research_result

    def _persist_refined_skill(self, source_ID, refined_book, _stream, _warn):
        self.calls.append("persist")


class RepairLadderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.books = os.path.join(self.tmp.name, "books")
        self.outputs = os.path.join(self.tmp.name, "out")
        os.makedirs(self.books)
        os.makedirs(self.outputs)
        with open(os.path.join(self.books, "test_source.toml"), "w",
                  encoding="utf-8") as fh:
            fh.write(HANDBOOK_TOML)
        self.addCleanup(self.tmp.cleanup)

    def run_request(self, script, refine_attempts=1, **attrs):
        agent = LadderAgent(api_key="sk-test", extra_handbook_dirs=[self.books],
                            script=script)
        agent.output_dir = self.outputs
        for name, value in attrs.items():
            setattr(agent, name, value)
        with mock.patch.object(agent_module, "HANDBOOK_REFINE_ATTEMPTS",
                               refine_attempts):
            try:
                agent.LLM_Find("Download the test data.")
            except Exception:
                # Post-download handling (file classification, reprojection) is
                # out of scope here; the ladder has already run by then.
                pass
        return agent

    def test_external_failure_stops_before_any_repair(self):
        """The saving that motivated triage: an outage must not cost a
        regeneration, nor replace a handbook that is already correct."""
        agent = self.run_request(
            ["fail"], triage_verdict={"failure_category": "external",
                                      "summary": "service is down"})
        self.assertEqual(agent.calls, ["execute:fail", "triage"])

    def test_external_verdict_is_matched_literally(self):
        """Only the literal category stops the ladder. Anything else -- an
        empty analysis, a drifted label -- escalates as before, so a broken
        classifier can never strand a request that a repair would have fixed."""
        for verdict in ({}, {"failure_category": "Unknown"},
                        {"failure_category": "external/platform_issue"}):
            with self.subTest(verdict=verdict):
                agent = self.run_request(["fail", "fail"],
                                         triage_verdict=verdict)
                self.assertIn("refine", agent.calls)

    def test_refinement_is_tried_before_re_research(self):
        agent = self.run_request(["fail", "fail"])
        self.assertEqual(
            agent.calls,
            ["execute:fail", "triage", "refine", "execute:fail", "triage",
             "re_research"])

    def test_retry_runs_the_revised_handbook_not_the_saved_one(self):
        """A revision is held in memory until it proves itself, so the retry
        has to render THAT rather than re-reading the .toml -- otherwise the
        revision is judged on a run it took no part in."""
        agent = self.run_request(["fail", "deliver"])
        self.assertIn("Call https://example.invalid/api", agent.prompts[0])
        self.assertNotIn("REVISED", agent.prompts[0])
        self.assertIn("REVISED", agent.prompts[1])
        self.assertIn("print('revised')", agent.prompts[1])

    def test_refinement_that_delivers_is_persisted(self):
        agent = self.run_request(["fail", "deliver"])
        self.assertEqual(
            agent.calls,
            ["execute:fail", "triage", "refine", "execute:deliver", "persist"])

    def test_refinement_that_never_delivers_is_discarded(self):
        """The rule that keeps a bad revision from poisoning a handbook other
        tasks rely on: persistence is earned by delivering data, not attempted."""
        agent = self.run_request(["fail", "fail"])
        self.assertNotIn("persist", agent.calls)

    def test_refinement_returning_no_change_falls_through_to_re_research(self):
        agent = self.run_request(["fail"], refine_result=None)
        self.assertEqual(agent.calls,
                         ["execute:fail", "triage", "refine", "re_research"])

    def test_zero_budget_restores_the_previous_ladder(self):
        """AGSI_HANDBOOK_REFINE_ATTEMPTS=0 is the documented way back to
        debug-then-re-research, so it must skip refinement entirely."""
        agent = self.run_request(["fail"], refine_attempts=0)
        self.assertEqual(agent.calls,
                         ["execute:fail", "triage", "re_research"])

    def test_empty_download_is_repaired_on_its_own_evidence(self):
        """A run that "succeeds" while writing a header-only file is a failure
        the traceback cannot describe, so the ladder gets the empty-result
        evidence instead."""
        class EmptyAgent(LadderAgent):
            def execute_complete_program(self, code, try_cnt, task, model_name,
                                         handbook_str, **kwargs):
                self.calls.append("execute:empty")
                path = os.path.join(self.output_dir, "empty.csv")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("id,value\n")
                self.last_execution_report = {"success": True, "traceback": ""}
                return code

        agent = EmptyAgent(api_key="sk-test",
                           extra_handbook_dirs=[self.books], script=[])
        agent.output_dir = self.outputs
        with mock.patch.object(agent_module, "HANDBOOK_REFINE_ATTEMPTS", 1):
            try:
                agent.LLM_Find("Download the test data.")
            except Exception:
                pass
        self.assertIn("refine", agent.calls)
        self.assertIn("ZERO records", agent.last_failure_note)


class RefinementContractTests(unittest.TestCase):
    """The wording and the no-change rule are shared with Handbook Studio.
    They live in handbook_generator so both repair paths ask for the same
    revision; these pin what each side depends on."""

    ANALYSIS = {"failure_category": "handbook_deficiency", "summary": "s"}

    def test_non_regression_note_only_when_the_code_already_ran(self):
        ran = handbook_generator.build_refinement_feedback(
            "task", self.ANALYSIS, error="e", prior_execution_ok=True)
        crashed = handbook_generator.build_refinement_feedback(
            "task", self.ANALYSIS, error="e", prior_execution_ok=False)
        self.assertIn("already ran successfully", ran)
        self.assertNotIn("already ran successfully", crashed)

    def test_mechanism_clause_is_omitted_rather_than_left_empty(self):
        pinned = handbook_generator.build_refinement_feedback(
            "task", self.ANALYSIS, mechanism="rest")
        unpinned = handbook_generator.build_refinement_feedback(
            "task", self.ANALYSIS)
        self.assertIn("Do not change the access mechanism (rest).", pinned)
        self.assertNotIn("access mechanism", unpinned)

    def test_evidence_is_carried_into_the_request(self):
        feedback = handbook_generator.build_refinement_feedback(
            "download faults", self.ANALYSIS, error="HTTPError: 503")
        self.assertIn("download faults", feedback)
        self.assertIn("HTTPError: 503", feedback)
        self.assertIn("handbook_deficiency", feedback)

    def test_only_operative_fields_count_as_a_change(self):
        base = {"handbook": "h", "code_example": "c", "brief_description": "d"}
        self.assertFalse(handbook_generator.refinement_changed(
            base, dict(base, brief_description="reworded")))
        self.assertTrue(handbook_generator.refinement_changed(
            base, dict(base, handbook="h2")))
        self.assertTrue(handbook_generator.refinement_changed(
            base, dict(base, code_example="c2")))


class InMemoryRenderTests(unittest.TestCase):
    """A revision under test must render exactly as it will once saved, or a
    retry can pass or fail for reasons unrelated to the revision."""

    def test_in_memory_record_renders_like_one_loaded_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "test_source.toml"), "w",
                      encoding="utf-8") as fh:
                fh.write(HANDBOOK_TOML)
            agent = DataRetrieverAgent(api_key="sk-test",
                                       extra_handbook_dirs=[tmp])
            from_disk = agent.collect_a_handbook(
                source_ID="test_source", source_dir=tmp,
                keys_dir=agent.keys_dir)
            book = agent._load_book(os.path.join(tmp, "test_source.toml"))
            self.assertEqual(agent.render_handbook_text(book, {}), from_disk)
            self.assertEqual(
                agent.render_code_example(book, {}),
                agent.collect_code_example("test_source", source_dir=tmp,
                                           keys_dir=agent.keys_dir))

    def test_credential_placeholders_are_substituted_case_insensitively(self):
        book = {"data_source_name": "S", "handbook": "key={MY_KEY} done",
                "code_example": "k = '{my_key}'"}
        agent = DataRetrieverAgent(api_key="sk-test")
        rendered = agent.render_handbook_text(book, {"my_key": "SECRET"})
        self.assertIn("key=SECRET", rendered)
        self.assertEqual(agent.render_code_example(book, {"MY_KEY": "SECRET"}),
                         "k = 'SECRET'")


if __name__ == "__main__":
    unittest.main()
