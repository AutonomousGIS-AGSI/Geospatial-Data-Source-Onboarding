"""The ablation's one variable: prompts differ by the handbook and nothing else.

The control arm withholds the handbook. It must not, in doing so, be told that
a handbook "is provided", be handed an empty "Technical handbook:" heading, or
be pointed at a naming pattern "documented in the handbook" -- all three said
something untrue about its own inputs. Equally, removing them must not perturb
the treatment prompt, or the arms would differ by more than the handbook.
"""

import re
import types
import unittest
import unittest.mock

from agents.data_agent import DataRetrieverAgent as agent_module
from agents.data_agent.DataRetrieverAgent import DataRetrieverAgent

TASK = "All active fire hotspots in California for the last 7 days"
SOURCE = "NASA/LANCE FIRMS"
HANDBOOK = ("1. Base URL: https://firms.modaps.eosdis.nasa.gov/api/area/csv\n"
            "2. MAP_KEY required.\n")
EXAMPLE = "import requests\nprint(requests.get('https://example').status_code)\n"

# The treatment prompt's length for these exact inputs. A change here means
# the treatment prompt moved, which must only ever happen on purpose. Last
# re-measured when the schema-vs-coverage requirement was added to the
# download prompt (the one that forbids raising DATA_UNAVAILABLE over a
# missing column/field name -- the year/variable may be implicit in a
# release-year or wide-format dataset).
TREATMENT_LENGTH = 8706


def _stub():
    return types.SimpleNamespace(output_dir="/tmp/out", model="gpt-4o")


def _download(handbook, example):
    return DataRetrieverAgent.create_download_prompt(
        _stub(), TASK, SOURCE, handbook, code_example=example)


def _debug(handbook):
    return DataRetrieverAgent.get_debug_prompt(
        _stub(), Exception("HTTP 401"), "code()", TASK, handbook)


class TreatmentPromptUnchanged(unittest.TestCase):
    def test_length_matches_pre_change_measurement(self):
        self.assertEqual(len(_download(HANDBOOK, EXAMPLE)), TREATMENT_LENGTH)

    def test_still_carries_handbook_role_clause_and_heading(self):
        prompt = _download(HANDBOOK, EXAMPLE)
        self.assertIn(agent_module._HANDBOOK_ROLE_CLAUSE, prompt)
        self.assertIn("Technical handbook: \n", prompt)
        self.assertIn(agent_module._HANDBOOK_PATTERN_CLAUSE, prompt)
        self.assertIn(HANDBOOK.strip(), prompt)

    def test_debug_prompt_still_carries_the_guidelines(self):
        self.assertIn("The technical guidelines for the code:", _debug(HANDBOOK))


class ControlPromptMentionsNoHandbook(unittest.TestCase):
    """The invariant that catches a future edit to the role sentence: if the
    clause in `download_role` is reworded, the swap becomes a silent no-op and
    the word reappears here."""

    def test_generation_prompt_never_says_handbook(self):
        prompt = _download("", "")
        found = re.findall(r"[^.]*handbook[^.]*\.", prompt, flags=re.I)
        self.assertEqual(found, [], f"handbook still referenced: {found}")

    def test_debug_prompt_never_says_handbook_or_guidelines(self):
        prompt = _debug("")
        self.assertNotIn("handbook", prompt.lower())
        self.assertNotIn("The technical guidelines for the code:", prompt)

    def test_no_empty_handbook_heading(self):
        self.assertNotIn("Technical handbook:", _download("", ""))

    def test_blank_and_whitespace_only_handbooks_are_both_control(self):
        self.assertEqual(_download("", ""), _download("   \n  ", ""))
        self.assertEqual(_debug(""), _debug("  \n "))


class ClaudePathSaysNoHandbookEither(unittest.TestCase):
    """The Claude Agent SDK path appends its own write/run/fix instructions
    after the download prompt, so it is a second place a handbook reference can
    survive -- and did."""

    def test_control_instructions_never_say_handbook(self):
        text = agent_module._claude_agent_instructions("retrieve.py", 3, control=True)
        self.assertNotIn("handbook", text.lower())

    def test_treatment_instructions_unchanged(self):
        text = agent_module._claude_agent_instructions("retrieve.py", 3, control=False)
        self.assertIn("under the exact names documented in the handbook", text)

    def test_both_arms_keep_the_credential_instruction(self):
        for control in (True, False):
            text = agent_module._claude_agent_instructions("retrieve.py", 3, control)
            self.assertIn("read them via os.environ", text)
            self.assertIn("never invent or hard-code a value", text)
            self.assertIn("retrieve.py", text)

    def test_full_claude_control_prompt_is_handbook_free(self):
        """What the Claude control agent actually receives, end to end."""
        prompt = (_download("", "") + "\n\n"
                  + agent_module._claude_agent_instructions("retrieve.py", 3, True))
        self.assertNotIn("handbook", prompt.lower())


class OnlyTheHandbookDiffers(unittest.TestCase):
    def test_control_keeps_the_coding_instruction_from_the_dropped_clause(self):
        # The clause carried two things: a handbook claim and "write Python
        # code carefully". Only the first may go.
        self.assertIn("write Python code carefully to download the data",
                      _download("", ""))

    @staticmethod
    def _requirements_block(prompt):
        """Just the numbered requirements — counting "\n1. " over the whole
        prompt also catches a numbered list inside the handbook body."""
        return prompt.split("Requirements: \n", 1)[1].split("\n\nData source:", 1)[0]

    def test_control_keeps_every_requirement(self):
        control = self._requirements_block(_download("", ""))
        treatment = self._requirements_block(_download(HANDBOOK, EXAMPLE))
        for n in range(1, 10):
            self.assertIn(f"{n}. ", control, f"requirement {n} missing")
        # Identical except the one clause that had to be reworded.
        self.assertEqual(
            control,
            treatment.replace(agent_module._HANDBOOK_PATTERN_CLAUSE,
                              agent_module._NO_HANDBOOK_PATTERN_CLAUSE))

    def test_control_keeps_task_source_outdir_and_reply_example(self):
        control = _download("", "")
        for fragment in (TASK, SOURCE, "/tmp/out", "download_data()",
                         "Current date-time:", "Your reply example:"):
            self.assertIn(fragment, control)

    def test_the_naming_pattern_requirement_survives_reworded(self):
        control = _download("", "")
        self.assertIn(agent_module._NO_HANDBOOK_PATTERN_CLAUSE, control)
        self.assertIn("parse the actual listing content", control)

    def test_shared_prose_is_identical_line_for_line(self):
        control = _download("", "")
        treatment = _download(HANDBOOK, EXAMPLE)
        # Every control line must appear in the treatment verbatim, except the
        # two lines carrying a reworded clause.
        reworded = (agent_module._NO_HANDBOOK_ROLE_CLAUSE,
                    agent_module._NO_HANDBOOK_PATTERN_CLAUSE)
        treatment_lines = set(treatment.splitlines())
        for line in control.splitlines():
            if any(clause in line for clause in reworded):
                continue
            self.assertIn(line, treatment_lines, f"control-only line: {line!r}")


if __name__ == "__main__":
    unittest.main()


class FailureAnalysisFitsTheArm(unittest.TestCase):
    """A control run has no handbook, so it must not be diagnosed as having a
    deficient one -- and the analyst must not be shown handbook text the run
    never received."""

    SOURCE = {
        "data_source_name": "NASA/LANCE FIRMS",
        "handbook": "SECRET-HANDBOOK-BODY: call /api/area/csv with MAP_KEY.",
        "code_example": "SECRET-EXAMPLE: requests.get('https://firms/api/area/csv')",
        "website": "https://firms.modaps.eosdis.nasa.gov",
    }
    TRACE = {"status": "failed", "error": "HTTP 404 on /v2/area",
             "generated_code": "requests.get('https://firms/v2/area')"}

    def _analyze(self, control, reply):
        from agents.data_agent import handbook_generator as hg
        seen = {}

        def fake_call_model(messages, **kwargs):
            seen["prompt"] = messages[0]["content"]
            return reply

        with unittest.mock.patch.object(hg, "_call_model", fake_call_model):
            out = hg.analyze_execution_trace(
                self.SOURCE, "fires in California", self.TRACE,
                user_key="sk-test", control=control)
        return out, seen["prompt"]

    GOOD_CONTROL = ('{"failure_category": "knowledge_gap", "summary": "wrong path", '
                    '"deficiencies": [{"handbook_gap": "assumed /v2/area", '
                    '"evidence": "HTTP 404", "recommended_change": "use /api/area/csv"}]}')
    GOOD_TREATMENT = ('{"failure_category": "handbook_deficiency", "summary": "s", '
                      '"deficiencies": []}')

    def test_control_analyst_never_sees_the_handbook_or_example(self):
        _, prompt = self._analyze(True, self.GOOD_CONTROL)
        self.assertNotIn("SECRET-HANDBOOK-BODY", prompt)
        self.assertNotIn("SECRET-EXAMPLE", prompt)
        # It still needs the source name -- that much the run WAS told.
        self.assertIn("NASA/LANCE FIRMS", prompt)
        self.assertIn("HTTP 404 on /v2/area", prompt)

    def test_treatment_analyst_still_sees_the_handbook(self):
        _, prompt = self._analyze(False, self.GOOD_TREATMENT)
        self.assertIn("SECRET-HANDBOOK-BODY", prompt)
        self.assertIn("SECRET-EXAMPLE", prompt)

    def test_control_is_asked_the_knowledge_gap_question(self):
        _, prompt = self._analyze(True, self.GOOD_CONTROL)
        self.assertIn('"knowledge_gap" | "external"', prompt)
        self.assertIn("do not propose handbook changes", prompt)
        self.assertNotIn('"handbook_deficiency" | "external"', prompt)

    def test_control_verdict_is_recorded_with_its_own_vocabulary(self):
        out, _ = self._analyze(True, self.GOOD_CONTROL)
        self.assertEqual(out["failure_category"], "knowledge_gap")
        self.assertEqual(out["condition"], "control")
        self.assertEqual(out["deficiencies"][0]["handbook_gap"], "assumed /v2/area")

    def test_a_control_run_can_never_be_labelled_handbook_deficiency(self):
        # Even if the model ignores the instruction and answers with the
        # treatment vocabulary, the record must not claim a handbook was
        # deficient in a run that had none.
        out, _ = self._analyze(True, self.GOOD_TREATMENT)
        self.assertEqual(out["failure_category"], "Unknown")

    def test_a_treatment_run_can_never_be_labelled_knowledge_gap(self):
        out, _ = self._analyze(False, self.GOOD_CONTROL)
        self.assertEqual(out["failure_category"], "Unknown")

    def test_external_is_valid_in_both_arms(self):
        reply = '{"failure_category": "external", "summary": "outage", "deficiencies": []}'
        for control in (True, False):
            out, _ = self._analyze(control, reply)
            self.assertEqual(out["failure_category"], "external")
            self.assertEqual(out["condition"], "control" if control else "with_handbook")

    def test_unparseable_reply_degrades_to_unknown_in_both_arms(self):
        for control in (True, False):
            out, _ = self._analyze(control, "not json at all")
            self.assertEqual(out["failure_category"], "Unknown")
            self.assertEqual(out["deficiencies"], [])


class TokensAreBookedToTheRightPhase(unittest.TestCase):
    """A control session generates no handbook, so nothing in it may be booked
    as "generation". The failure analysis and the semantic judge both run
    during a TEST, and both used to be recorded as generation because
    _call_model hard-coded the phase."""

    class _Usage:
        input_tokens, output_tokens, total_tokens = 1000, 200, 1200

    def _phases_for(self, call):
        """Run one analysis call inside a retrieval ledger, with the streaming
        client stubbed, and report which phases its tokens landed in."""
        from agents.data_agent import handbook_generator as hg
        from agents.data_agent import usage_ledger

        class Chunk:
            usage = TokensAreBookedToTheRightPhase._Usage()
            choices = []

        def fake_stream(**kwargs):
            return [Chunk()]

        with unittest.mock.patch.object(
                hg.helper, "client_chat_completion_stream", fake_stream), \
                unittest.mock.patch.object(
                    hg, "_parse_json_object",
                    lambda text: {"failure_category": "external", "summary": "",
                                  "deficiencies": [], "task_completed": False}):
            with usage_ledger.collect("retrieval") as ledger:
                call(hg)
                return ledger.totals({})["by_phase"]

    def test_failure_analysis_is_booked_as_evaluation_not_generation(self):
        phases = self._phases_for(lambda hg: hg.analyze_execution_trace(
            {"data_source_name": "X"}, "task", {"status": "failed"},
            user_key="sk-test", control=True))
        self.assertIn("evaluation", phases)
        self.assertNotIn("generation", phases,
                         "a control test run must not book tokens as generation")

    def test_semantic_evaluation_is_booked_as_evaluation(self):
        phases = self._phases_for(lambda hg: hg.evaluate_retrieval_result(
            {"data_source_name": "X"}, "task", {"status": "failed"},
            user_key="sk-test"))
        self.assertIn("evaluation", phases)
        self.assertNotIn("generation", phases)

    def test_call_model_still_defaults_to_generation(self):
        # refine_handbook authors handbook text and keeps the generation
        # label; only the two evaluation callers were relabelled.
        import inspect
        from agents.data_agent import handbook_generator as hg
        sig = inspect.signature(hg._call_model)
        self.assertEqual(sig.parameters["phase"].default, "generation")
