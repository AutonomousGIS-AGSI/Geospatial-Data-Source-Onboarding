"""Public (read-only) sharing of Handbook Studio sessions.

The contract: a session is invisible to anyone but its owner until the owner
flags it public; then anyone can READ it and its files, nobody but the owner
can change it, and turning the flag off closes the door again.
"""
import hashlib
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "x")

from WebUI import app as appmod
from WebUI import handbook_studio_runner as runner
from WebUI import handbook_studio_store as store


OWNER = {"X-API-Key": "sk-test-share-owner"}
OTHER = {"X-API-Key": "sk-test-share-other"}
ANON = {}


class StudioSharingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(store, "STORE_ROOT", os.path.join(self.tmp.name, "sessions")),
            mock.patch.object(runner, "OUTPUT_ROOT", os.path.join(self.tmp.name, "outputs")),
        ]
        for p in self.patches:
            p.start()
        self.client = appmod.app.test_client()
        self.uid = hashlib.sha256(OWNER["X-API-Key"].encode()).hexdigest()[:16]
        sess = store.save_session(
            self.uid, {**store.new_session("share test"), "status": "complete"})
        self.sid = sess["id"]
        run_dir = os.path.join(runner._session_output_root(self.uid, self.sid), "test_s1")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "out.csv"), "w") as handle:
            handle.write("a,b\n1,2\n")
        sess["test_runs"] = [{
            "id": "r1", "test_number": 1, "outcome": "passed",
            "result": {"downloaded_files": [
                {"name": "out.csv", "size_bytes": 8, "artifact_ref": "test_s1/out.csv"}]},
        }]
        store.save_session(self.uid, sess, session_id=self.sid)
        self.S = f"/api/handbook-studio-sessions/{self.sid}"

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def get(self, headers, path):
        return self.client.get(path, headers=headers).status_code

    def share(self, headers, public):
        return self.client.post(self.S + "/share", json={"public": public}, headers=headers)

    def test_private_session_is_owner_only(self):
        self.assertEqual(self.get(ANON, self.S), 401)
        self.assertEqual(self.get(OTHER, self.S), 404)
        self.assertEqual(self.get(OWNER, self.S), 200)
        self.assertEqual(self.get(ANON, self.S + "/output-preview?ref=test_s1/out.csv"), 401)
        self.assertEqual(self.get(OTHER, self.S + "/test-runs/r1/download-all"), 404)

    def test_only_owner_can_toggle_sharing(self):
        self.assertEqual(self.share(ANON, True).status_code, 401)
        self.assertEqual(self.share(OTHER, True).status_code, 404)
        r = self.share(OWNER, True)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["public"])
        self.assertTrue(r.get_json()["url"].endswith(f"/handbook-studio/shared/{self.sid}"))

    def test_public_session_is_readable_by_anyone_but_not_writable(self):
        self.share(OWNER, True)
        for headers in (ANON, OTHER):
            r = self.client.get(self.S, headers=headers)
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.get_json()["shared_view"])
            self.assertFalse(r.get_json()["is_owner"])
            self.assertEqual(self.get(headers, self.S + "/output-preview?ref=test_s1/out.csv"), 200)
            self.assertEqual(self.get(headers, self.S + "/test-runs/r1/artifacts"), 200)
            self.assertEqual(self.get(headers, self.S + "/test-runs/r1/artifacts/out.csv?raw=1"), 200)
            self.assertEqual(self.get(headers, self.S + "/test-runs/r1/download-all"), 200)
        owner = self.client.get(self.S, headers=OWNER).get_json()
        self.assertTrue(owner["is_owner"])
        self.assertTrue(owner["session"]["public_since"])
        # Writes stay owner-only.
        self.assertEqual(self.client.put(self.S, json={"name": "x"}, headers=ANON).status_code, 401)
        self.assertEqual(self.client.put(self.S, json={"name": "x"}, headers=OTHER).status_code, 404)
        self.assertEqual(self.client.delete(self.S, headers=OTHER).status_code, 404)
        self.assertEqual(store.get_session(self.uid, self.sid)["name"], "share test")

    def test_turning_sharing_off_closes_access_again(self):
        self.share(OWNER, True)
        self.assertEqual(self.get(ANON, self.S), 200)
        r = self.share(OWNER, False)
        self.assertFalse(r.get_json()["public"])
        self.assertNotIn("public_since", store.get_session(self.uid, self.sid))
        self.assertEqual(self.get(ANON, self.S), 401)
        self.assertEqual(self.get(OTHER, self.S), 404)

    def test_unknown_and_malformed_ids_do_not_leak(self):
        self.assertEqual(self.get(ANON, "/api/handbook-studio-sessions/00000000-0000-0000-0000-000000000000"), 401)
        self.assertIsNone(store.find_public_session("../../etc/passwd"))
        self.assertIsNone(store.find_public_session(""))

    def test_shared_page_route_serves_the_app_shell(self):
        r = self.client.get(f"/handbook-studio/shared/{self.sid}")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"handbook-studio-workspace", r.data)

    def test_ordinary_save_preserves_the_flag(self):
        self.share(OWNER, True)
        sess = store.get_session(self.uid, self.sid)
        store.save_session(self.uid, {**sess, "name": "renamed"}, session_id=self.sid)
        after = store.get_session(self.uid, self.sid)
        self.assertTrue(after["public"])
        self.assertEqual(after["name"], "renamed")
        self.assertIn(True, [s["public"] for s in store.list_sessions(self.uid) if s["id"] == self.sid])


if __name__ == "__main__":
    unittest.main()
