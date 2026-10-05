# Reimplementation requirements

This repository reproduces the paper's central modular interaction loop rather than its unavailable institute implementation.

## Required behavior

1. Accept a still image, video camera, or caller-supplied frame.
2. Run a modern Ultralytics YOLO detector with either public pretrained weights or user-trained weights. No weights are committed.
3. Resolve the screen-center gaze/pointer to a detected bounding box and keep the last selected object as conversational context.
4. Enter listening only after a continuous four-second dwell. Gaze loss resets the dwell; silent gaze loss ends listening.
5. Answer from a user-authored domain knowledge file using `(query, object)` and emit parallel speech, expression, gesture, and viseme-friendly events.
6. Keep spatial anchors behind a provider interface. The included JSON provider demonstrates room-scoped persistence, not HoloLens spatial mapping.
7. Keep detection, interaction, knowledge, behavior, and anchors replaceable and independently callable.

## Deliberate boundaries

- This is a portable Python research implementation, not the original Unity/HoloLens application.
- It does not reproduce the paper's private flower images, chatbot knowledge, models, 3D characters, cloud endpoints, or measured latency.
- Anchor transforms are opaque application values. Device SDK adapters must establish and resolve real world coordinates.
- The rule-based responder is grounded only in the supplied knowledge JSON; it does not claim open-domain chatbot coverage.

## Acceptance checks

- Unit tests cover exact dwell transitions, gaze loss, bounding-box selection, contextual follow-ups, behavior events, and room-scoped anchors.
- The CLI can perform image inference and a camera interaction loop when optional YOLO weights and a camera are available.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; speaking envelopes approximate mouth motion rather than phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.

## Local recorded co-speech integration

The browser application retrieves prepared BEAT body-motion clips with `wearable` mode: the paper's early exact-rule co-speech component. The first `python scripts/start_demo.py` run fetches a small official BVH/TextGrid sample, constructs a nine-clip bank, and fits the local retrieval artifact under ignored `outputs/beat-library/`. Install `scripts/requirements-demo.txt` first. Preparation code and method dependencies are vendored in this repository; no sibling clone, original institute library, full dataset, or pretrained weights are bundled. The object-focus, dwell, knowledge-answer, and anchor code remains this application's core. The separate recorded-motion companion remains available for local motion/face/audio inspection.
