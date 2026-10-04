# Design of Seamless Multi-modal Interaction Framework for Intelligent Virtual Agents in Wearable Mixed Reality Environment

**Ghazanfar Ali, Hong-Quan Le, Junho Kim, Seung-Won Hwang, Jae-In Hwang**

**CASA · 2019** · Published

[Paper / publisher](https://doi.org/10.1145/3328756.3328758) · [Project page](https://ghazanfarali.com/research/wearable-mr-agent/) · [Video presentation](https://www.youtube.com/watch?v=zlmVpUgBdew) · [BibTeX](CITATION.bib) · [Requirements](REQUIREMENTS.md) · [Code & setup](#implementation-and-usage)

> A modular virtual guide combines speech, gaze, and spatial context in wearable MR.

![Wearable MR Agent architecture: camera, gaze and speech inputs, object-grounded dialogue, coordinated agent behavior, spatial anchors and engagement loop](paper-assets/wearable-system.png)

*Graphical abstract diagram. Camera, gaze and speech inputs support object-grounded conversation and coordinated virtual-agent behavior.*

## Why this research

A wearable virtual guide must connect what a visitor says and looks at with the surrounding scene. Modular processing distributes these capabilities across the device and supporting services.

The framework integrates spatial mapping, speech recognition, gaze, object recognition, a domain-specific chatbot, and virtual-character animation. Computationally intensive components run on a cloud platform to support a wearable device with limited resources.

## Method at a glance

**Speech + gaze + scene** → **Modular agent / cloud** → **Embodied MR response**

| | Research system |
|---|---|
| Input | Speech, gaze, recognized objects, and spatial mapping |
| Method | Modular interaction framework with cloud-assisted processing |
| Output | An interactive virtual guide with verbal and animated responses |

## Evidence and scope

Paper reports responses within 2–4 seconds in its tests

**Attribution:** These findings describe the paper or manuscript, not results obtained with this repository's code.

**Study context:** Wearable mixed-reality application scenarios.

**Limitations:** The response measurements describe the tested hardware, network, and scenarios; cloud connectivity and spatial mapping are required by the design.

## Explore the implementation

Modern YOLO image/camera input, gaze dwell, object-grounded responses and behavior events. Device spatial anchors are an integration interface; this is a portable core rather than the original HoloLens application.

This repository contains independently written research code. The institute's original source, datasets and trained models are not distributed. Public-data preparation, commands, assumptions and checks are documented below and in [REQUIREMENTS.md](REQUIREMENTS.md).

## Resources and citation

Read the paper through its [publisher record](https://doi.org/10.1145/3328756.3328758). PDFs are hosted by publishers or preprint archives rather than stored in this repository.

Watch the [existing YouTube presentation](https://www.youtube.com/watch?v=zlmVpUgBdew).

Please cite the research paper when using its ideas; [download the BibTeX citation](CITATION.bib). The implementation has its own documented scope.

## Implementation and usage

<!-- implementation-guide -->

This standalone repository reimplements the core loop from **“Design of Seamless Multi-modal Interaction Framework for Intelligent Virtual Agents in Wearable Mixed Reality Environment”** by Ghazanfar Ali, Hong-Quan Le, Junho Kim, Seung-Won Hwang, and Jae-In Hwang, CASA 2019, pp. 47–52. DOI: [10.1145/3328756.3328758](https://doi.org/10.1145/3328756.3328758).

The original implementation and its assets are held by the institute. This is a new educational implementation built from the paper. It includes real object detection, gaze selection, the four-second dwell state machine, grounded object conversation, coordinated behavior events, and a portable anchor abstraction. It does not include the original Unity project, HoloLens application, flower dataset, knowledge base, cloud services, characters, animations, or trained weights, and it does not reproduce the paper's latency measurements.

### Setup

Python 3.10 or newer is required. From this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[yolo,dev]"
pytest -q
```

Ultralytics downloads a named public checkpoint on first use. To avoid downloads, pass a local checkpoint. No model is bundled here.

### Synthetic quickstart

Run `python scripts/smoke.py` after installation. It feeds a contract-valid procedural detection through the real gaze selector and four-second dwell state machine, then resolves a curator-authored object query into speech, expression, gesture, and viseme events. Inspect `outputs/smoke/response.json` and the annotated frame. For real use, replace the procedural `Detection` list with `YoloDetector.detect(frame)` output and replace the inline knowledge mapping with `DomainKnowledge.load("knowledge.json")`; the downstream interaction contract stays the same.

If the optional `yolo` dependency is installed, `python scripts/smoke.py --with-yolo` additionally initializes YOLO11n from its bundled architecture configuration with random weights and sends the synthetic frame through `YoloDetector`. This checks the adapter and dependency boundary without downloading pretrained weights; random detections have no semantic meaning.

### Knowledge contract

Create `knowledge.json` with curator-written, verified content:

```json
{
  "objects": {
    "detector class name": {
      "overview": "A curator-authored description.",
      "topics": {"care": "A curator-authored answer about care."},
      "emotion": "joy"
    }
  }
}
```

Keys must match YOLO class names. Topic keys are simple query triggers. The responder never fills missing facts from a language model.

### Detect, train, and interact

Run public YOLO weights on an image:

```powershell
wearable-mr-agent image .\frame.jpg --weights yolo11n.pt
```

Run the camera loop. The screen center is the portable gaze proxy; dwell continuously on one detected instance for four seconds and press `A` to type an utterance:

```powershell
wearable-mr-agent camera --weights yolo11n.pt --knowledge .\knowledge.json --camera 0
```

For a new domain, prepare a standard Ultralytics detection dataset (`images/train`, `images/val`, matching YOLO text labels, and `dataset.yaml`) and train from architecture configuration:

```powershell
yolo detect train model=yolo11n.yaml data=.\dataset.yaml epochs=100 imgsz=640
wearable-mr-agent camera --weights .\runs\detect\train\weights\best.pt --knowledge .\knowledge.json
```

The first command initializes a detector from scratch; using `model=yolo11n.pt` instead fine-tunes pretrained weights. See the official [Ultralytics training guide](https://docs.ultralytics.com/modes/train/) and [dataset format guide](https://docs.ultralytics.com/datasets/detect/).

### Architecture and adapters

- `YoloDetector` owns inference only.
- `select_detection` resolves a pointer/gaze point to the tightest containing box.
- `InteractionStateMachine` implements idle → dwell → listen → think → respond and resets incomplete dwell on gaze loss.
- `DomainKnowledge` accepts `(query, object)` and emits speech, expression, gesture, and viseme-channel events.
- `AnchorProvider` separates application logic from device spatial APIs. `JsonAnchorProvider` only proves room-scoped save/load semantics; an OpenXR/ARCore adapter must supply real transforms and room localization.

This scope is intentional: it preserves the paper's modular contribution while avoiding a false claim that JSON transforms recreate native HoloLens spatial anchors or that a desktop camera is a wearable MR system.
