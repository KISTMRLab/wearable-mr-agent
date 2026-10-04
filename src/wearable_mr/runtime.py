from __future__ import annotations

from .gaze import select_detection
from .interaction import InteractionStateMachine
from .knowledge import DomainKnowledge
from .types import AgentReply, Detection


class AgentRuntime:
    def __init__(self, knowledge: DomainKnowledge, dwell_seconds: float = 4.0):
        self.knowledge = knowledge
        self.interaction = InteractionStateMachine(dwell_seconds=dwell_seconds)
        self.focused_object: str | None = None
        self._focused_instance: str | None = None
        self._focused_detection: Detection | None = None
        self._instance_counter = 0

    @staticmethod
    def _iou(first: Detection, second: Detection) -> float:
        left, top = max(first.box.left, second.box.left), max(first.box.top, second.box.top)
        right, bottom = min(first.box.right, second.box.right), min(first.box.bottom, second.box.bottom)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = first.box.area + second.box.area - intersection
        return intersection / union if union else 0.0

    def observe(self, detections: list[Detection], pointer_x: float, pointer_y: float, now: float) -> list[str]:
        hit = select_detection(detections, pointer_x, pointer_y)
        target = None
        if hit is not None:
            if hit.track_id is not None:
                target = f"{hit.label}:{hit.track_id}"
            elif self._focused_detection is not None and hit.label == self._focused_detection.label and self._iou(hit, self._focused_detection) >= .45:
                target = self._focused_instance
            else:
                self._instance_counter += 1
                target = f"{hit.label}:local-{self._instance_counter}"
            self._focused_instance = target
            self._focused_detection = hit
            self.focused_object = hit.label
        return self.interaction.update_gaze(target, now)

    def ask(self, query: str) -> AgentReply:
        self.interaction.submit_utterance()
        reply = self.knowledge.answer(query, self.focused_object)
        self.interaction.begin_response()
        self.interaction.finish_response()
        return reply
