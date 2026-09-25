"""Token accounting for the Research / Spatial Analysis / Data Retrieval modes.

These tests load the shipped source of each function under test directly, so
they exercise the real code without needing the project's heavy runtime
dependencies (numpy, flask, langchain) to be installed.

What they pin down:

* a call is attributed to the phase that was running when it was made
* a call whose backend reported no usage is COUNTED and disclosed, never
  estimated -- an invented number here would be indistinguishable from a
  measured one, and the whole point of the figure is that it is measured
* worker threads record into the run that spawned them, and only that run
* the browser receives cumulative phase totals but only newly-added call rows
"""
import ast
import importlib.util
import io
import sys
import types
import unittest


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_functions(path, names, namespace):
    """Exec the named top-level functions from *path* into *namespace*."""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    wanted = [node for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.name in names]
    missing = set(names) - {node.name for node in wanted}
    assert not missing, "not found in %s: %s" % (path, missing)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), path, "exec"), namespace)
    return namespace


def _module_constant(path, name):
    """Evaluate one module-level constant assignment from *path*."""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("%s not found in %s" % (name, path))


def _module_function(path, name):
    """Load one top-level function, resolving its module-level dependencies."""
    return load_functions(path, {name}, {})[name]


ledger_mod = load_module("agents/data_agent/usage_ledger.py", "usage_ledger_under_test")


class LedgerTests(unittest.TestCase):
    def test_phase_attribution_follows_set_phase(self):
        with ledger_mod.collect() as ledger:
            ledger.set_phase("data_audit")
            ledger_mod.record("gpt-5.2", {"prompt_tokens": 100, "completion_tokens": 20})
            ledger_mod.record("gpt-5.2", {"input_tokens": 50, "output_tokens": 5})
            ledger.set_phase("rq_breakdown")
            ledger_mod.record("gpt-5.2", {"input_tokens": 7, "output_tokens": 3})
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 3)
        self.assertEqual(totals["total_tokens"], 120 + 55 + 10)
        self.assertEqual(totals["by_phase"]["data_audit"]["total_tokens"], 175)
        self.assertEqual(totals["by_phase"]["rq_breakdown"]["total_tokens"], 10)

    def test_call_without_usage_is_counted_not_estimated(self):
        with ledger_mod.collect() as ledger:
            ledger_mod.record("gpt-5.2", {"input_tokens": 7, "output_tokens": 3})
            ledger_mod.record("gpt-5.2", {})     # backend returned nothing
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 2)
        self.assertEqual(totals["calls_missing_usage"], 1)
        self.assertEqual(totals["total_tokens"], 10)   # not inflated by a guess

    def test_records_since_returns_only_new_records(self):
        with ledger_mod.collect() as ledger:
            for i in range(3):
                ledger_mod.record("m", {"input_tokens": i, "output_tokens": 0})
            first, offset = ledger.records_since(0)
            self.assertEqual((len(first), offset), (3, 3))
            ledger_mod.record("m", {"input_tokens": 9, "output_tokens": 1})
            new, offset = ledger.records_since(offset)
        self.assertEqual((len(new), offset), (1, 4))
        self.assertEqual(new[0]["input_tokens"], 9)

    def test_workflow_phase_wins_over_a_callers_own_label(self):
        # The retrieval agent labels its calls "retrieval" and the handbook
        # generator labels its own "generation". In the chat modes the phases
        # are workflow CARDS, so those labels match nothing on screen and a
        # whole download run showed no tokens on any card. The card wins; the
        # caller's label survives as `stage`.
        with ledger_mod.collect() as ledger:
            ledger.set_phase("data_download_req_1")
            ledger_mod.record("gpt-5.2", {"input_tokens": 900, "output_tokens": 100},
                              phase="retrieval", source="generate_code")
            totals = ledger.totals()
            entry = ledger.records[0]
        self.assertEqual(entry["phase"], "data_download_req_1")
        self.assertEqual(entry["stage"], "retrieval")
        self.assertEqual(totals["by_phase"]["data_download_req_1"]["total_tokens"], 1000)
        self.assertNotIn("retrieval", totals["by_phase"])

    def test_a_callers_label_still_wins_when_no_phase_is_set(self):
        # Generate & Test opens a ledger per phase and never calls set_phase,
        # so its own labels must keep working exactly as before.
        with ledger_mod.collect("generation") as ledger:
            ledger_mod.record("gpt-5.2", {"input_tokens": 10, "output_tokens": 2},
                              phase="retrieval", source="generate_code")
            ledger_mod.record("gpt-5.2", {"input_tokens": 10, "output_tokens": 2})
            totals = ledger.totals()
        self.assertEqual(totals["by_phase"]["retrieval"]["total_tokens"], 12)
        self.assertEqual(totals["by_phase"]["generation"]["total_tokens"], 12)

    def test_no_active_ledger_is_inert(self):
        # Notebooks, the CLI and the test suite call the agents with no ledger
        # open; instrumenting a call site must not change behaviour there.
        self.assertIsNone(ledger_mod.record("m", {"input_tokens": 1}))


class RawCompletionCaptureTests(unittest.TestCase):
    """The second LLM family: helper.client_chat_completion[_stream].

    GIBDChat is not the only path. The RQ understanding agent (which produces
    the Task Breakdown in every mode), the task manager, the spatial analysis
    executor and the modeling agent all call these raw helpers instead, so a
    run whose work happens there records nothing unless they are instrumented
    too.
    """

    def setUp(self):
        self.routed = []

        def fake_route(model, messages, user_key=None, base_url=None, **kwargs):
            self.routed.append(kwargs)
            return types.SimpleNamespace(usage={"prompt_tokens": 60,
                                                "completion_tokens": 12})

        def fake_route_stream(model, messages, user_key=None, base_url=None, **kwargs):
            self.routed.append(kwargs)
            if kwargs.get("stream_options") and self.reject:
                raise RuntimeError("stream_options not supported by this backend")
            yield types.SimpleNamespace(usage=None, text="a")
            yield types.SimpleNamespace(usage=None, text="b")
            if kwargs.get("stream_options"):
                yield types.SimpleNamespace(
                    usage={"prompt_tokens": 30, "completion_tokens": 5})

        self.reject = False
        self.ns = load_functions(
            "utils/agm_helper.py",
            {"_usage_from_chunk", "_record_llm_usage",
             "client_chat_completion", "client_chat_completion_stream"},
            {"_usage_ledger": lambda: ledger_mod,
             "_route_chat_completion": fake_route,
             "_route_chat_completion_stream": fake_route_stream,
             "print": lambda *a, **k: None})

    def test_non_streaming_call_is_recorded(self):
        with ledger_mod.collect() as ledger:
            self.ns["client_chat_completion"]("gpt-4o", [{"role": "user", "content": "x"}])
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["total_tokens"], 72)

    def test_streamed_call_is_recorded_once_from_the_trailing_chunk(self):
        with ledger_mod.collect() as ledger:
            chunks = list(self.ns["client_chat_completion_stream"](
                "gpt-4o", [{"role": "user", "content": "x"}]))
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 1)        # one call, not one per chunk
        self.assertEqual(totals["total_tokens"], 35)
        self.assertEqual(self.routed[0].get("stream_options"),
                         {"include_usage": True})
        self.assertEqual([c.text for c in chunks if getattr(c, "text", None)],
                         ["a", "b"])

    def test_a_backend_rejecting_stream_options_still_streams(self):
        # The router is a generator, so the rejection surfaces only when the
        # first chunk is pulled. Content must survive the fallback intact.
        self.reject = True
        with ledger_mod.collect() as ledger:
            chunks = list(self.ns["client_chat_completion_stream"](
                "gpt-4o", [{"role": "user", "content": "x"}]))
            totals = ledger.totals()
        self.assertEqual([c.text for c in chunks if getattr(c, "text", None)],
                         ["a", "b"])
        self.assertIsNone(self.routed[1].get("stream_options"))
        self.assertEqual(totals["calls"], 0)   # accounting lost, stream intact

    def test_record_usage_false_opts_a_self_accounting_caller_out(self):
        # DataRetrieverAgent records each call itself; recording here too
        # would count every retrieval call twice.
        with ledger_mod.collect() as ledger:
            list(self.ns["client_chat_completion_stream"](
                "gpt-4o", [{"role": "user", "content": "x"}], record_usage=False))
            self.ns["client_chat_completion"](
                "gpt-4o", [{"role": "user", "content": "x"}], record_usage=False)
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 0)

    def test_caller_supplied_stream_options_passes_through_unrecorded(self):
        # handbook_generator asks for usage itself and records in its own
        # consume(); this must not add a second record for the same call.
        with ledger_mod.collect() as ledger:
            list(self.ns["client_chat_completion_stream"](
                "gpt-4o", [{"role": "user", "content": "x"}],
                stream_options={"include_usage": True}))
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 0)
        self.assertEqual(len(self.routed), 1)   # no probe-then-retry


if __name__ == "__main__":
    unittest.main(verbosity=2)
