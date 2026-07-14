# Vision Camera Maintenance Contract

`vem.vision.camera-maintenance/v2` is a loopback-only, Vision-owned contract.
It is protected by a short-lived, single-use maintenance capability with the
purpose `vision.camera-maintenance`; callers need the exact endpoint scope
(`camera.read`, `camera.refresh`, `camera.preview`, `camera.test`, or
`camera.confirm`). A
customer capability, an expired capability, or a replayed capability is
rejected. VEM may render returned values but never persists device identities
or backend observations.

`GET /maintenance/cameras` returns a cached candidate `generation` and top /
front readiness. Candidate identity and capture source must come from the same
Windows Media Foundation enumeration boundary. Vision never sorts and zips PnP
devices with OpenCV indexes. If the installed capture adapter cannot prove that
source mapping, the candidate has `mappingState: "unproven"`, is unavailable,
and a bound role is explicitly `ambiguous` rather than ready.

The numeric backend index is therefore an authenticated maintenance
observation only. It is not in managed site configuration, release `/version`,
or Vision's persisted binding.

Operator workflow:

1. Read candidates and role readiness with a `camera.read` capability.
2. Preview with `camera.preview`; frames are local and use `Cache-Control:
   no-store`.
3. Run `POST /maintenance/cameras/{top|front}/test` with `camera.test`. The
   successful response returns role-specific candidate, generation, expiry,
   and one-use evidence ID.
4. Confirm with `camera.confirm`, the matching `testEvidenceId`, and the same
   generation. If the operator made a visual placement verification instead,
   confirm may explicitly send `operatorVisualConfirmation: true`.

Preview, test, and runtime capture share one local lease namespace, so a role
cannot silently open a second capture while another owner holds it. Discovery
runs only for explicit `POST /maintenance/cameras/refresh` (with
`camera.refresh`), device/read failure, or initial snapshot;
normal role resolution uses the cached generation.

Vision atomically stores confirmations in
`%ProgramData%\VendingVision\camera-bindings.json`. Only the stable identity
and confirmation evidence are persisted. Duplicate persisted identities across
roles, missing devices, unproven mappings, and duplicate candidates are all
non-ready role states. Only `ready` permits runtime acquisition.

The response schema is
`config/vending-vision-camera-maintenance-v2.schema.json`; the versioned test
and confirm request schemas are in
`config/vending-vision-camera-maintenance-v2.requests.schema.json`, and test
and error response schemas are in
`config/vending-vision-camera-maintenance-v2.responses.schema.json`.
