CatVTON vendored source closure

Upstream: https://github.com/Zheng-Chong/CatVTON
Revision: 3b795364a4d2f3b5adb365f39cdea376d20bc53c
Vendored for VEM Issue06 Stage9 official AI attempt worker.
Only source code needed by the CPU CatVTON attempt child is included; model weights are excluded.

Files below record the upstream bytes at revision
`3b795364a4d2f3b5adb365f39cdea376d20bc53c` before the minimal VEM audit
patch.  The local patch only changes package-relative imports and enforces
offline/local model loading from the verified pack.

Upstream files:
- LICENSE sha256:095791f7f830b929a5c4c5bd8fb9e62a1694336fc6b915f771e25a44015c403c
- model/SCHP/__init__.py sha256:d895bce7e45e5ed6c8073a38f47e015195f9fe042e2b6da3a12202761d479d39
- model/SCHP/networks/AugmentCE2P.py sha256:9092adf2e3c932f5559c4b16e1067cff268c2959358a9f2a5ab4f4497d7703b1
- model/SCHP/networks/__init__.py sha256:5f97fe315303d2daee1a211c5e4435d1cb516941af278a530ebe5c29bccf2479
- model/SCHP/utils/transforms.py sha256:196e0ef23983b20838001a832250dd72819ffe17dfb583fa949093dcfb70af07
- model/attn_processor.py sha256:51f793caeb7c828b0d479120142db5b735f8fe7c3efcf56ad8c130388d7a6244
- model/pipeline.py sha256:7231a2d50be5fee3958a8b77643959b3c88f0385a6c6bbc3d910804f40974349
- model/utils.py sha256:9e5041c963126fba8053648cb0b52b2427eaf817c02bdd23421175e80c24eb7f
- utils.py sha256:3d31693279cf9d217fc6efa9f7499d9a9bd58fb71e9b679dcad12a58cc2ab710

Local patched source hashes are not duplicated here.  The canonical runtime
source hash allowlist is `official-ai-source-descriptor.json`, which is checked
by the worker probe against the actual deployed source layout.
