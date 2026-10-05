# Reimplementation requirements

This repository implements the paper's multimodal interaction framework as a portable Python core with a local web demo. It does not reuse the institute's HoloLens/Unity application, assets or services.

## Required behavior

1. **Gaze on the character.** Continuous gaze on the virtual character for the dwell time (4 s, configurable) starts listening. Leaving the character before the dwell resets it.
2. **Proactive greeting.** If the user stays silent after the dwell, the character says "Hello, do you need help?" (configurable), once per conversation.
3. **Ending.** While listening, gazing away *and* being silent ends the conversation and clears its object and history. Gazing away while speaking keeps it. A configurable grace period (default 1.5 s; 0 for the strict rule) lets the user turn to an object before a voice command. The runtime receives the user's speaking state with every gaze sample.
4. **Voice commands.** A configurable list ("what is this", "tell me about this", …), matched on word boundaries, triggers recognition of the object under the user's gaze. Commands can also open a conversation from idle.
5. **Follow-ups and general conversation.** Later queries reuse the recognised object without re-gazing. General queries (greeting, identity, help, yes/no answers, thanks, farewell) are answered with or without an object, in any order.
6. **Chatbot.** `(Query, Object) → (Reply, Sentiment class, Sentiment level)`. The language part is pluggable: an OpenAI-compatible endpoint grounded with curated facts, a local command, or the offline curated-knowledge chatbot. Topic matching uses whole words, not substrings. A failed client falls back to the offline chatbot.
7. **Sentiment engine.** Exactly one of Joy, Angry, Sad or Fear, and one of High, Medium or Low. The classifier is pluggable (JSON service or OpenAI-compatible model); an offline lexicon classifier handles fallback. The face uses the renderer's `setExpression(name, level)` with happiness, anger, sadness or fear and preset levels 1–3, for the whole utterance.
8. **Animation builder.** An editable phrase/word → animation table maps to prepared BEAT clip ids and procedural gestures. Phrases are matched longest first, then single words in unclaimed text. The builder emits a timed animation list ordered by position in the reply. Missing clips fall back to the entry's procedural gesture.
9. **Latency hiding.** A query enters an asynchronous THINKING state with a thinking animation and a filler utterance while recognition and the chatbot are pending.
10. **Anchors.** A recognised object label selects and loads that room's anchors; identical-room ambiguity waits for more evidence. A command on an anchored object in the loaded room skips recognition.
11. **Recognition.** A YOLO detector on camera frames (the box under the gaze point), a remote vision endpoint, or a simulated recogniser for web-demo props.
12. **Web demo.** The character is the gaze target (mouse ray or screen-centre ray). The demo provides proactive greeting, voice commands (browser speech recognition with a typed fallback), thinking and filler, webcam frames to the YOLO endpoint (with a clear message when no weights are configured), two simulated rooms and a voice per character.

## Deliberate boundaries

- Hosted services are interfaces with offline fallbacks; no service, model or key is bundled or downloaded automatically.
- The offline chatbot answers only from the curated knowledge JSON and a small set of general intents.
- The lexicon sentiment classifier defaults to Joy/Low when the reply has no cue words.
- **Not reproduced in the web demo:**
  - **Persistent spatial anchors.** These need a device runtime (WebXR Anchors exist only in Android/Quest browsers). Desktop rooms and anchors are simulated, and `AnchorProvider` is the device-adapter interface.
  - **WebXR head gaze.**
  - **Recognition of real flowers in a mixed-reality scene.** Props are procedural, with labels from the room description.
  - **The paper's latency measurements.**
- The local Kokoro backend has one voice. Per-character voices use browser speech synthesis.

## Acceptance checks

- Unit tests cover:
  - dwell timing and cancellation;
  - greeting timing and suppression;
  - look-away while speaking or while the agent thinks;
  - strict and grace-period endings;
  - response timeout;
  - voice commands from idle and while listening;
  - follow-ups without recognition;
  - general chat with and without an object;
  - whole-word topic matching (the `use`/`because` regression);
  - OpenAI-compatible, command and HTTP clients against stubs, with fallbacks;
  - sentiment classes, levels and renderer mapping;
  - the builder's ordering, longest-phrase-first rule, gap rule and clip resolution;
  - room anchor loading, ambiguity and recognition skipping;
  - the asynchronous thinking state;
  - the demo HTTP routes, including webcam requests without weights.
- `scripts/verify.py` drives a complete offline visit: dwell, greeting, command with anchor loading, follow-up, anchored object, general chat and Fear sentiment. It writes `outputs/verify/response.json`.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally. No dataset recordings or trained weights are bundled.

## Local recorded co-speech integration

The first `python scripts/start_demo.py` run fetches a small official BVH/TextGrid sample and builds a nine-clip bank under ignored `outputs/beat-library/`. Install `scripts/requirements-demo.txt` first. The repository's animation table (`demo/animation-table.json`) names clips from that bank; the demo server attaches their frames to each reply. Without a prepared bank, the table's procedural gestures play instead. The vendored BEAT preparation code is shared with the other repositories; no sibling clone, institute library, full dataset or pretrained weights are bundled.
