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

`man-front.mp4` is a six-frame recorded-video fixture derived only from
`sources/person-man-front.png` (SHA-256 is guarded by the recorded-video
tests).  That source is an unmodified vendored copy of
`virtual-tryon/samples/inputs/person-man-front.png`, selected because it is a
known fictional/provenance-tracked test input in the parent try-on project;
it is not a customer or field photograph.  Its generator is
`generate-man-front.py`; it resizes the source to 512x768 and uses OpenCV's
`mp4v` encoder.  It exists solely for production YOLO and MediaPipe Pose
acquisition acceptance and has no runtime-bundle dependency.  The same
generator emits `man-unaligned-front.mp4`, a deterministic left crop which
still has one YOLO person but fails the centered-pose rule, and
`empty-front.mp4`, a black no-person recording.  SHA-256 values for all
three are in the manifest.
