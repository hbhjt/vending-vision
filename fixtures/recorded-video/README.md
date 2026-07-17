# Recorded Video Fixtures

`top.mp4` and `front.mp4` are short deterministic recordings for the Vision
`recorded_video` frame source. The expected behavior and SHA-256 digests are
in `expected-results.json`; changing either recording requires updating that
manifest and the recorded-frame integration tests.

The top recording is finite (`loop: false`) so the fixture can assert a
departure event. The front recording loops at EOF (`loop: true`) for repeated
profile and try-on frame reads; a non-looping source stays exhausted until an
explicit reset.

These files are intentionally separate from the Windows runtime bundle. The
release workflow archives this directory as the fixture artifact.
