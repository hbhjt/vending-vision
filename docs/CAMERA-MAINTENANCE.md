# Vision Camera Maintenance Contract

`vem.vision.camera-maintenance/v2` is a loopback-only, Vision-owned contract.
VEM may render its opaque candidates and readiness evidence, but must never
persist a camera identity, moniker, PnP path, or backend index.

## Windows adapter and bindings

Production Windows discovery is the pinned `cv2-enumerate-cameras==1.1.16`
DirectShow adapter. Its one native enumeration returns both a DirectShow
moniker path (the persistent Vision identity) and the corresponding
`cv2.VideoCapture(index, CAP_DSHOW)` opening source. The adapter is included in
the PyInstaller bundle. Vision never combines Windows Runtime/Media Foundation
identities with a separately probed OpenCV index, and production does not use
an `msmf`/`mediafoundation` fallback naming path.

After a USB replug, DirectShow may assign a different index. Refreshing the
same stable moniker resolves that current source without changing the confirmed
role binding. Missing monikers, duplicate monikers, or an adapter that cannot
prove an identity-to-source mapping produce explicit non-ready evidence;
`ambiguous` is reserved for an actually unresolvable candidate, not the normal
Windows production path.

The backend index is an authenticated maintenance observation only. It is not
part of managed site configuration, `/version`, or persisted binding state.

## Daemon-issued maintenance capability

Vision is verifier-only. The Vending Daemon is the only issuer and owns the
Ed25519 private key. The managed site configuration points only to three
daemon-owned local files; it never contains a shared secret or signing key:

```json
{
  "maintenance_capability_keyring_path": "C:\\ProgramData\\VEM\\vision\\daemon-maintenance-keys.json",
  "maintenance_session_path": "C:\\ProgramData\\VEM\\vision\\daemon-maintenance-session.json",
  "maintenance_replay_path": "C:\\ProgramData\\VEM\\vision\\camera-maintenance-replay.json"
}
```

The keyring has `version: 1`, issuer `vem.vending-daemon`, and key records with
`id`, raw Ed25519 `publicKey` (base64url), `notBefore`, and `notAfter`. The
active session material has `version: 1`, `machineCode`, `sessionId`, `keyId`,
and `expiresAt`. VEM atomically rotates either file when a maintenance session
or signing key rotates. Their Windows ACL must allow the daemon to write and
Vision to read, but never make the daemon private key available to Vision.

Each JWT uses `alg: EdDSA`, `typ: JWT`, and the active `kid`; claims must bind
`iss=vem.vending-daemon`, `aud=vem.vision.camera-maintenance`, `machine`,
`session`, `purpose=vision.camera-maintenance`, exact endpoint `scope`, `iat`,
`exp`, and `jti`. Vision accepts at most a 300-second TTL, validates the key
and session lifetimes, and atomically persists consumed JTIs in the replay
ledger so restart does not permit replay. Missing/invalid daemon material is
an explicit HTTP 503 maintenance blocker, not a development fallback.

## Operator workflow

Every endpoint needs a fresh, single-use capability with its exact scope:
`camera.read`, `camera.refresh`, `camera.preview`, `camera.test`, or
`camera.confirm`.

1. Read candidates and role readiness with `GET /maintenance/cameras`.
2. Optionally refresh with `POST /maintenance/cameras/refresh`.
3. Preview with `GET /maintenance/cameras/{candidateId}/preview.jpg`; the
   response is local-only and `Cache-Control: no-store`.
4. Run `POST /maintenance/cameras/{top|front}/test` with `candidateId`. The
   successful response is role-specific and returns a one-use evidence ID and
   candidate generation.
5. Confirm with `POST /maintenance/cameras/{top|front}/confirm`, supplying
   `candidateId`, `testEvidenceId`, `operatorVisualConfirmation: true`, and
   `expectedGeneration`. Confirmation atomically rejects an index/replug
   generation change, wrong role, expired/used evidence, or a missing visual
   check.

Preview and role test perform a protected runtime handoff: a persistent Vision
stream releases its lease for the short maintenance capture, then resumes
lazily through the same single-owner pipeline. Maintenance therefore does not
remain permanently blocked by an active runtime stream.

Vision atomically stores only the stable identity and confirmation method in
`%ProgramData%\VendingVision\camera-bindings.json`. Only `ready` roles may be
acquired at runtime.

The contract response schema is
`config/vending-vision-camera-maintenance-v2.schema.json`; requests are in
`config/vending-vision-camera-maintenance-v2.requests.schema.json`, and
contract/refresh/test/confirm/error responses are in
`config/vending-vision-camera-maintenance-v2.responses.schema.json`.
