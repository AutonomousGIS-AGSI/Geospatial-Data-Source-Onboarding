"""A key the download code cannot find is asked for, not debugged around.

The failure this pins: a synthesized NASA FIRMS skill verified fine (the
verification trial exports the user's key into the environment), then every
real download died on ``KeyError: 'FIRMS_MAP_KEY'`` -- the production path
only substituted ``{PLACEHOLDER}`` tokens in the handbook prose, never
exported anything, and a KeyError is not a 401, so it went to the debugger
five times, then triage, then three whole-request retries.

Two rules now: the real download exports the source's credentials exactly
like verification does, and a failure that says a credential is ABSENT is a
CredentialError carrying the missing names, so the user is asked for them.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_agent import DataRetrieverAgent as agent_module
from agents.data_agent.DataRetrieverAgent import (
    DataRetrieverAgent, CredentialError, _missing_credential_names)

KEYED_HANDBOOK = '''data_source_name = "Keyed Source"
brief_description = "A source used only by tests."
handbook = """Call https://example.invalid/api with your key."""
code_example = "import os\\nk = os.environ['TEST_MAP_KEY']"
website = "https://example.invalid/docs"
requires_key = "true"
key_name = "TEST_MAP_KEY"
caveats = ""
key_signup_url = "https://example.invalid/signup"
'''


class MissingNameDetectionTests(unittest.TestCase):
    def test_keyerror_from_os_environ(self):
        self.assertEqual(
            _missing_credential_names(KeyError("FIRMS_MAP_KEY"),
                                      ["FIRMS_MAP_KEY"]),
            ["FIRMS_MAP_KEY"])

    def test_message_naming_the_variable_as_unset(self):
        err = RuntimeError("Missing required environment variable "
                           "FIRMS_MAP_KEY. Set it before running.")
        self.assertEqual(_missing_credential_names(err, ["FIRMS_MAP_KEY"]),
                         ["FIRMS_MAP_KEY"])

    def test_wrapped_keyerror_is_seen_through_the_chain(self):
        try:
            try:
                os.environ["TEST_ABSENT_KEY_XYZ"]
            except KeyError as e:
                raise RuntimeError("setup failed") from e
        except RuntimeError as outer:
            self.assertEqual(
                _missing_credential_names(outer, ["TEST_ABSENT_KEY_XYZ"]),
                ["TEST_ABSENT_KEY_XYZ"])

    def test_unrelated_errors_and_unknown_names_do_not_match(self):
        for err in (KeyError("features"),
                    ValueError("required parameter bbox is missing"),
                    RuntimeError("FIRMS_MAP_KEY_STATUS was 200")):
            with self.subTest(err=err):
                self.assertEqual(
                    _missing_credential_names(err, ["FIRMS_MAP_KEY"]), [])
        self.assertEqual(_missing_credential_names(KeyError("X"), []), [])


def _reply(text):
    return [types.SimpleNamespace(
        usage=None,
        choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content=text))])]


class ExecuteLoopTests(unittest.TestCase):
    def setUp(self):
        self.agent = DataRetrieverAgent(api_key="sk-test")
        self.debug_calls = 0
        patcher = mock.patch.object(
            agent_module, "_stream_with_usage",
            side_effect=lambda **kw: self._debug())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _debug(self):
        self.debug_calls += 1
        return _reply("fix\n```python\nprint('ok')\n```")

    def test_missing_declared_key_is_asked_for_not_debugged(self):
        with self.assertRaises(CredentialError) as ctx:
            self.agent.execute_complete_program(
                code="import os\nos.environ['TEST_ABSENT_KEY_XYZ']",
                try_cnt=5, task="t", model_name="m", handbook_str="h",
                attempt_timeout=30, credential_names=["TEST_ABSENT_KEY_XYZ"])
        self.assertEqual(ctx.exception.missing_keys, ["TEST_ABSENT_KEY_XYZ"])
        self.assertIn("TEST_ABSENT_KEY_XYZ", str(ctx.exception))
        self.assertEqual(self.debug_calls, 0)

    def test_undeclared_name_read_by_the_code_counts_too(self):
        """The handbook never declared it, but the code reads it: the code
        is the ground truth for what it needs."""
        with self.assertRaises(CredentialError) as ctx:
            self.agent.execute_complete_program(
                code="import os\nk = os.environ['TEST_OTHER_KEY_XYZ']",
                try_cnt=5, task="t", model_name="m", handbook_str="h",
                attempt_timeout=30, credential_names=[])
        self.assertEqual(ctx.exception.missing_keys, ["TEST_OTHER_KEY_XYZ"])

    def test_plain_bugs_still_go_to_the_debugger(self):
        self.agent.execute_complete_program(
            code="d = {}\nd['features']", try_cnt=5, task="t",
            model_name="m", handbook_str="h", attempt_timeout=30,
            credential_names=["TEST_ABSENT_KEY_XYZ"])
        self.assertEqual(self.debug_calls, 1)
        self.assertTrue(self.agent.last_execution_report["success"])


class EnvExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(self.tmp.name, "keyed_source.toml"), "w",
                  encoding="utf-8") as fh:
            fh.write(KEYED_HANDBOOK)
        self.addCleanup(self.tmp.cleanup)
        self.agent = DataRetrieverAgent(api_key="sk-test",
                                        extra_handbook_dirs=[self.tmp.name])
        os.environ.pop("TEST_MAP_KEY", None)

    def test_user_key_is_exported_for_the_download_and_restored_after(self):
        with self.agent._source_keys_in_env(
                "keyed_source", {"TEST_MAP_KEY": "secret-1"}) as names:
            self.assertEqual(names, ["TEST_MAP_KEY"])
            self.assertEqual(os.environ.get("TEST_MAP_KEY"), "secret-1")
        self.assertNotIn("TEST_MAP_KEY", os.environ)

    def test_case_of_the_entered_key_does_not_matter(self):
        with self.agent._source_keys_in_env(
                "keyed_source", {"test_map_key": "secret-2"}):
            self.assertEqual(os.environ.get("TEST_MAP_KEY"), "secret-2")

    def test_names_the_code_reads_are_exported_when_a_value_exists(self):
        code = "import os\nos.environ['TEST_EXTRA_KEY']"
        with self.agent._source_keys_in_env(
                "keyed_source", {"TEST_EXTRA_KEY": "v"}, code) as names:
            self.assertIn("TEST_EXTRA_KEY", names)
            self.assertEqual(os.environ.get("TEST_EXTRA_KEY"), "v")
        self.assertNotIn("TEST_EXTRA_KEY", os.environ)

    def test_placeholder_values_are_not_exported(self):
        with self.agent._source_keys_in_env(
                "keyed_source", {"TEST_MAP_KEY": "XXXX"}):
            self.assertNotIn("TEST_MAP_KEY", os.environ)

    def test_previous_value_is_put_back(self):
        os.environ["TEST_MAP_KEY"] = "before"
        try:
            with self.agent._source_keys_in_env(
                    "keyed_source", {"TEST_MAP_KEY": "during"}):
                self.assertEqual(os.environ["TEST_MAP_KEY"], "during")
            self.assertEqual(os.environ["TEST_MAP_KEY"], "before")
        finally:
            os.environ.pop("TEST_MAP_KEY", None)

    def test_download_prompt_names_the_exported_variables(self):
        _, exported = self.agent._credential_env_plan(
            "keyed_source", {"TEST_MAP_KEY": "secret"})
        prompt = self.agent.create_download_prompt(
            "task", "Keyed Source", "handbook",
            credential_env_names=sorted(exported))
        self.assertIn("named exactly: TEST_MAP_KEY", prompt)
        bare = self.agent.create_download_prompt("task", "Keyed Source", "handbook")
        self.assertNotIn("named exactly", bare)


if __name__ == "__main__":
    unittest.main()
