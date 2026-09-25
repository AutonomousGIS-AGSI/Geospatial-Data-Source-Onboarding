import os
import tempfile
import time
import unittest
from unittest import mock

from WebUI import output_retention as ret

MB = 1024 * 1024
DAY = 86400


def _write(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\0" * size)
    return path


def _age(path, seconds):
    past = time.time() - seconds
    os.utime(path, (past, past))


class RetentionPolicyTests(unittest.TestCase):
    """The janitor must be inert unless a quota is configured, and once it is,
    must never touch small files, bookkeeping, or anything inside a claim
    window."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        # quota 10 MB, keep <=1 MB, retention 7 d, run cap 5 MB, claim 24 h
        self.env = mock.patch.dict(os.environ, {
            "GIS_COSCI_OUTPUT_QUOTA_MB": "10",
            "GIS_COSCI_OUTPUT_KEEP_SMALL_MB": "1",
            "GIS_COSCI_OUTPUT_RETENTION_DAYS": "7",
            "GIS_COSCI_RUN_CAP_MB": "5",
            "GIS_COSCI_RUN_CAP_RETENTION_HOURS": "24",
            "GIS_COSCI_OUTPUT_MIN_FREE_MB": "0",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_dir(self, user="u1", session="s1", run="test_1"):
        return os.path.join(self.root, user, session, run)

    # ── policy switch ───────────────────────────────────────────────────

    def test_disabled_without_quota_deletes_nothing(self):
        with mock.patch.dict(os.environ, {"GIS_COSCI_OUTPUT_QUOTA_MB": ""}):
            big = _write(os.path.join(self.run_dir(), "old.tif"), 3 * MB)
            _age(big, 365 * DAY)
            report = ret.sweep(self.root)
            self.assertFalse(report["enabled"])
            self.assertTrue(os.path.exists(big))
            self.assertIsNone(ret.preflight(self.root))
            self.assertFalse(ret.mark_run(self.run_dir())["policy_enabled"])
            self.assertFalse(os.path.exists(os.path.join(self.run_dir(), ".retention.json")))

    # ── what is never touched ───────────────────────────────────────────

    def test_small_files_and_handbook_and_bookkeeping_survive_everything(self):
        rd = self.run_dir()
        small = _write(os.path.join(rd, "sample.csv"), 100 * 1024)
        boot = _write(os.path.join(rd, "_rerun_boot.py"), 3 * MB)  # absurd but internal
        hb = _write(os.path.join(self.root, "u1", "s1", "_handbook", "x.toml"), 3 * MB)
        for p in (small, boot, hb):
            _age(p, 400 * DAY)
        ret.sweep(self.root)
        for p in (small, boot, hb):
            self.assertTrue(os.path.exists(p), p)

    # ── expiry ──────────────────────────────────────────────────────────

    def test_bulk_file_evicted_after_retention_period(self):
        rd = self.run_dir()
        fresh = _write(os.path.join(rd, "fresh.tif"), 2 * MB)
        stale = _write(os.path.join(rd, "stale.tif"), 2 * MB)
        _age(stale, 8 * DAY)
        report = ret.sweep(self.root)
        self.assertTrue(os.path.exists(fresh))
        self.assertFalse(os.path.exists(stale))
        self.assertEqual(len(report["expired"]), 1)
        # and the eviction is recorded with a user-facing reason
        self.assertIn("stale.tif", ret.evicted_files(rd))
        self.assertIn("retention policy", ret.evicted_reason(rd, "stale.tif"))
        self.assertIsNone(ret.evicted_reason(rd, "fresh.tif"))

    def test_marked_run_uses_its_deadline_not_file_mtime(self):
        rd = self.run_dir()
        f = _write(os.path.join(rd, "data.tif"), 2 * MB)
        finished = time.time() - 3 * DAY
        record = ret.mark_run(rd, finished_at=finished)
        self.assertFalse(record["over_cap"])
        # file mtime is "now", but the run finished 3 days ago -> still within
        # 7 d retention: kept
        self.assertTrue(os.path.exists(f))
        ret.sweep(self.root)
        self.assertTrue(os.path.exists(f))
        # 8 days after the recorded finish: gone
        ret.sweep(self.root, now=finished + 8 * DAY)
        self.assertFalse(os.path.exists(f))

    def test_large_run_gets_claim_window_then_expires(self):
        rd = self.run_dir()
        f = _write(os.path.join(rd, "huge.tif"), 6 * MB)  # over the 5 MB cap
        finished = time.time()
        record = ret.mark_run(rd, finished_at=finished)
        self.assertTrue(record["over_cap"])
        ret.sweep(self.root, now=finished + 23 * 3600)
        self.assertTrue(os.path.exists(f), "inside the 24 h claim window")
        ret.sweep(self.root, now=finished + 25 * 3600)
        self.assertFalse(os.path.exists(f), "claim window passed")

    # ── quota ───────────────────────────────────────────────────────────

    def test_quota_evicts_oldest_eligible_first_but_never_inside_claim_window(self):
        rd = self.run_dir()
        newest = _write(os.path.join(rd, "newest.tif"), 4 * MB)   # 1 h old
        middle = _write(os.path.join(rd, "middle.tif"), 4 * MB)   # 2 d old
        oldest = _write(os.path.join(rd, "oldest.tif"), 4 * MB)   # 3 d old
        _age(newest, 3600)
        _age(middle, 2 * DAY)
        _age(oldest, 3 * DAY)
        # 12 MB bulk > 10 MB quota: evict oldest only (brings it to 8 MB)
        report = ret.sweep(self.root)
        self.assertFalse(os.path.exists(oldest))
        self.assertTrue(os.path.exists(middle))
        self.assertTrue(os.path.exists(newest))
        self.assertEqual(report["quota"], [os.path.relpath(oldest, self.root)])

    def test_quota_pressure_cannot_evict_files_inside_claim_window(self):
        rd = self.run_dir()
        a = _write(os.path.join(rd, "a.tif"), 6 * MB)
        b = _write(os.path.join(rd, "b.tif"), 6 * MB)   # 12 MB > quota, both fresh
        report = ret.sweep(self.root)
        self.assertTrue(os.path.exists(a) and os.path.exists(b))
        self.assertEqual(report["quota"], [])
        self.assertGreater(report["bulk_bytes_after"], ret.policy()["quota_bytes"])

    def test_dry_run_reports_without_deleting(self):
        rd = self.run_dir()
        stale = _write(os.path.join(rd, "stale.tif"), 2 * MB)
        _age(stale, 8 * DAY)
        report = ret.sweep(self.root, dry_run=True)
        self.assertEqual(len(report["expired"]), 1)
        self.assertTrue(os.path.exists(stale))
        self.assertEqual(ret.evicted_files(rd), {})

    # ── disk floor ──────────────────────────────────────────────────────

    def test_preflight_and_watchdog_refuse_when_below_min_free(self):
        with mock.patch.dict(os.environ, {"GIS_COSCI_OUTPUT_MIN_FREE_MB": "1"}), \
             mock.patch.object(ret, "free_disk_bytes", return_value=0):
            self.assertIn("Not enough free disk", ret.preflight(self.root))
            self.assertIn("Aborted", ret.watchdog(self.root))
        with mock.patch.dict(os.environ, {"GIS_COSCI_OUTPUT_MIN_FREE_MB": "1"}), \
             mock.patch.object(ret, "free_disk_bytes", return_value=50 * MB):
            self.assertIsNone(ret.preflight(self.root))
            self.assertIsNone(ret.watchdog(self.root))

    # ── usage accounting ────────────────────────────────────────────────

    def test_usage_splits_small_and_bulk(self):
        rd = self.run_dir()
        _write(os.path.join(rd, "small.csv"), 10 * 1024)
        _write(os.path.join(rd, "big.tif"), 3 * MB)
        use = ret.usage(self.root)
        self.assertEqual(use["runs"], 1)
        self.assertEqual(use["small_bytes"], 10 * 1024)
        self.assertEqual(use["bulk_bytes"], 3 * MB)


if __name__ == "__main__":
    unittest.main()
