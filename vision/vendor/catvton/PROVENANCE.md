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

Standalone migration reference

The production integration was migrated from the non-production reference
repository `https://github.com/hbhjt/virtual-tryon` at commit
`c0a76e499a620a253b7ac0a6a07f8ee0754c2c10`. That repository is source history,
not a build, test, packaging, runtime, download, fallback, or deployment
dependency of Vending Vision.

Reference files used to verify the migration:

- `app/ai_masks.py` sha256:487ac2261ae102a80f8a2142d2a369af7776869cc3e91d9b6729a122bd49af03
  became the maintained preprocessing/postprocessing behavior in
  `vision/catvton_preprocess.py`.
- `app/ai_tryon.py` sha256:3076d494da15f30421f74fb9e9be4949c973b373d7822af727bd19d5d053c8f0
  and `scripts/catvton_worker.py`
  sha256:4fae4fa44ee8ab75c94869680deea944de5e5c03a4b56689e8c23422c3cfc18d
  informed the attempt-only worker boundary now owned by
  `vision/catvton_pose_masks.py` and `vision/ai_attempt_worker.py`.
- `samples/inputs/person-man-front.png`
  sha256:659f08c709c8d526552713741f5e2cfe3fa819a34a63a34a8372a3404890952c
  is the sole retained standalone fixture. Its maintained owner and generated
  recordings are documented in `fixtures/recorded-video/README.md`.

Deliberately excluded reference assets:

- `samples/inputs/person-woman-front.png`
  sha256:2587576cc8359c9717ea63a7efa1eb76e62bdcfdde7bb2248d4f37b6953b13d4
  is not needed by the production acquisition fixture matrix, so repository
  history remains its only recovery mechanism.
- The four local wardrobe images `coral-tee`, `cream-sweater`,
  `midnight-jacket`, and `ocean-polo` have respective SHA-256 values
  `eb2b5029578b879c503c1418860e5b0959c7cb6e4beb84e4ab8a355775937913`,
  `41d16eb98ec430672e8cdbf0297cf4cb7c6e01a83309e80cad5d93553e555ea8`,
  `fe4d998e1c713429c3dcc346716a150a6257f82d75fe4217a7d2aa417db6a149`,
  and `5b48bd28bdf2499554e77d259f32f62932cc705b258ab99c53dba25b1ed2c0d9`.
  Their ImageGen prompt record has SHA-256
  `4e6fb9ffcc8a70fc13b482981143af088385d6a74515d3008c12c65e3168b469`.
  They are excluded because production garments are platform-owned verified
  media, not a Vision-local wardrobe.
- The hash-named inputs `652ab2a22dd83ec45e81e283af5310ec.jpg` and
  `c196741201df156a8a2ff68fabd2d034.jpg` have SHA-256 values
  `9227b6b0d32c6b7666023019f510576d06dfacc8058d7850a06ef3ebed9681cc`
  and `435e9d33ad184842c58ceb74576732614028c1edec7793ff906ac63e0b88b6a4`.
  They and their derived custom garments are excluded because the reference
  repository contains no adequate origin or fictional-person provenance.
