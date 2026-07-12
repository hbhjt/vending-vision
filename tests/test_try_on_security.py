import unittest

import vision.try_on_session as try_on
from vision.camera_owner import FrontCameraOwner
from vision.config import settings


class TryOnSecurityTest(unittest.TestCase):
    def setUp(self):
        self.original_acquire = try_on.acquire_front_camera
        self.original_release = try_on.release_front_camera
        try_on.acquire_front_camera = lambda owner, reason=None: {"ok": True, "owner": owner}
        try_on.release_front_camera = lambda owner, reason=None: {"ok": True, "owner": "idle"}
        self.manager = try_on.TryOnSessionManager()

    def tearDown(self):
        try_on.acquire_front_camera = self.original_acquire
        try_on.release_front_camera = self.original_release

    def test_other_client_cannot_replace_or_stop_active_session(self):
        session = self.manager.start("try-on-a", owner_id="client-a")

        with self.assertRaises(PermissionError):
            self.manager.start("try-on-b", owner_id="client-b")

        with self.assertRaises(PermissionError):
            self.manager.stop("try-on-a", owner_id="client-b")

        self.assertTrue(self.manager.is_active("try-on-a"))
        stopped = self.manager.stop("try-on-a", owner_id="client-a")
        self.assertEqual(stopped["sessionId"], "try-on-a")
        self.assertFalse(self.manager.is_active("try-on-a"))
        self.assertIn("token=", session["previewUrl"])

    def test_stream_requires_opaque_capability_token(self):
        session = self.manager.start("try-on-token", owner_id="client-a")

        self.assertFalse(self.manager.can_stream("try-on-token", "wrong-token"))
        self.assertTrue(
            self.manager.can_stream("try-on-token", session["streamToken"])
        )

        status = self.manager.status()["activeSession"]
        self.assertNotIn("streamToken", status)
        self.assertNotIn("ownerId", status)

    def test_same_owner_start_is_idempotent(self):
        first = self.manager.start("try-on-repeat", owner_id="client-a")
        second = self.manager.start("try-on-repeat", owner_id="client-a")
        self.assertEqual(first["streamToken"], second["streamToken"])
        self.assertEqual(self.manager.status()["sessionCount"], 1)

    def test_camera_owner_lease_can_be_renewed(self):
        owner = FrontCameraOwner()
        original_timeout = settings.FRONT_CAMERA_OWNER_TIMEOUT_MS
        settings.FRONT_CAMERA_OWNER_TIMEOUT_MS = 100
        try:
            self.assertTrue(owner.acquire("tryon_frontend")["ok"])
            owner.updated_monotonic -= 0.05
            self.assertTrue(owner.renew("tryon_frontend")["ok"])
            self.assertEqual(owner.status()["owner"], "tryon_frontend")
        finally:
            settings.FRONT_CAMERA_OWNER_TIMEOUT_MS = original_timeout


if __name__ == "__main__":
    unittest.main()
