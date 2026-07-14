# Vision Camera Maintenance Contract

`vem.vision.camera-maintenance/v1` is a loopback-only, Vision-owned contract.
VEM may render its values in protected maintenance UI, but must treat candidate
IDs and backend observations as opaque and must never persist them.

`GET /maintenance/cameras` returns candidate devices plus `top` and `front`
role readiness. Candidate `id` is a stable Windows PnP identity. Its
`backendObservation.index` is only the current DirectShow/OpenCV index and can
change after reboot or replug.

The operator workflow is local and non-retaining:

1. Read candidates and readiness from `GET /maintenance/cameras`.
2. Request `GET /maintenance/cameras/{candidateId}/preview.jpg`; responses use
   `Cache-Control: no-store` and do not upload or retain frames.
3. Submit `{ "candidateId": "..." }` to
   `POST /maintenance/cameras/{top|front}/test`.
4. Submit the same payload to `/confirm` only after the role-specific result is
   acceptable.

Vision atomically stores confirmations in its local application-data file
(`%ProgramData%\VendingVision\camera-bindings.json` on Windows). The file has
only Vision's binding format and never appears in the VEM site-config schema.
The response names `unbound`, `missing`, and `ambiguous` states explicitly;
only `ready` permits runtime acquisition of a role.
