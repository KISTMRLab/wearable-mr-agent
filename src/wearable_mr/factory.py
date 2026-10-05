"""Assemble the runtime from the authored scenario, knowledge, animation table and optional services."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .anchors import AnchorManager, InMemoryAnchorProvider, JsonAnchorProvider, SpatialAnchor
from .animation import AnimationBuilder, AnimationTable, ClipLibrary
from .chatbot import Chatbot, CommandChatbot, OpenAICompatibleChatbot
from .clients import OpenAICompatibleClient
from .knowledge import DomainKnowledge
from .recognition import DetectorRecognizer, HttpRecognizer, RoutingRecognizer, SimulatedRecognizer
from .runtime import AgentConfig, AgentRuntime
from .sentiment import HttpSentimentClassifier, OpenAICompatibleSentimentClassifier, SentimentEngine

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
DEFAULT_BANK = ROOT / "outputs" / "beat-library" / "bank.json"


def add_service_arguments(parser: argparse.ArgumentParser, chat: bool = True) -> None:
    if chat:
        parser.add_argument("--knowledge", default=str(DEMO / "knowledge.json"), help="Curated knowledge JSON (offline chatbot and LLM grounding)")
        parser.add_argument("--scenario", default=str(DEMO / "scenario.json"), help="Agent settings, characters and simulated rooms")
        parser.add_argument("--animation-table", default=str(DEMO / "animation-table.json"), help="Phrase/word -> animation table")
        parser.add_argument("--bank", default=str(DEFAULT_BANK), help="Prepared BEAT clip bank (scripts/prepare_beat_demo.py)")
        parser.add_argument("--dwell", type=float, help="Override the gaze dwell time in seconds (paper: 4)")
        parser.add_argument("--chat-endpoint", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
        parser.add_argument("--chat-model", help="Model name for --chat-endpoint")
        parser.add_argument("--chat-command", help="Local chatbot command: JSON {query, object, history} on stdin")
        parser.add_argument("--anchors", help="Persist anchors to this JSON file instead of memory")
        parser.add_argument("--vision-endpoint", help="Remote vision API for camera frames: POST {image} -> {label, confidence}")
        parser.add_argument("--recognition-latency", type=float, help="Simulated recogniser delay in seconds")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the API key for LLM endpoints")
    parser.add_argument("--sentiment-endpoint", help="Sentiment JSON service: POST {text} -> {class, level} or {label, score}")
    parser.add_argument("--sentiment-llm-endpoint", help="OpenAI-compatible endpoint used as the sentiment classifier")
    parser.add_argument("--sentiment-model", help="Model for --sentiment-llm-endpoint")


def sentiment_engine(args: Any) -> SentimentEngine:
    if getattr(args, "sentiment_endpoint", None):
        return SentimentEngine(HttpSentimentClassifier(args.sentiment_endpoint))
    if getattr(args, "sentiment_llm_endpoint", None):
        if not args.sentiment_model:
            raise SystemExit("--sentiment-llm-endpoint needs --sentiment-model")
        client = OpenAICompatibleClient(args.sentiment_llm_endpoint, args.sentiment_model, args.api_key_env, temperature=0)
        return SentimentEngine(OpenAICompatibleSentimentClassifier(client))
    return SentimentEngine()


def load_scenario(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scene_props(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**prop, "room": room["id"]} for room in scenario.get("rooms", []) for prop in room.get("props", [])]


def build_runtime(args: Any, *, detector: Any = None, executor: Any = None, clock: Any = None) -> AgentRuntime:
    knowledge = DomainKnowledge.load(args.knowledge)
    scenario = load_scenario(getattr(args, "scenario", None))
    settings = dict(scenario.get("agent", {}))
    if getattr(args, "dwell", None) is not None:
        settings["dwell_seconds"] = args.dwell
    config = AgentConfig.from_dict(settings)
    if getattr(args, "chat_endpoint", None):
        if not args.chat_model:
            raise SystemExit("--chat-endpoint needs --chat-model")
        client = OpenAICompatibleChatbot(OpenAICompatibleClient(args.chat_endpoint, args.chat_model, args.api_key_env), knowledge)
    elif getattr(args, "chat_command", None):
        client = CommandChatbot(args.chat_command)
    else:
        client = knowledge
    chatbot = Chatbot(client, sentiment_engine(args), fallback=None if client is knowledge else knowledge)
    table = AnimationTable.load(args.animation_table)
    builder = AnimationBuilder(table, ClipLibrary.load(args.bank) if getattr(args, "bank", None) else None)
    provider = JsonAnchorProvider(args.anchors) if getattr(args, "anchors", None) else InMemoryAnchorProvider()
    anchors = AnchorManager(provider)
    known = {a.anchor_id for a in provider.all()}
    props = scene_props(scenario)
    for prop in props:  # anchor placement done once per room (paper Figure 11a)
        if prop.get("anchor", True) and prop["id"] not in known:
            anchors.place(SpatialAnchor(prop["id"], prop["room"], prop["label"], [float(v) for v in prop.get("position", [0, 0, 0])]))
    latency = args.recognition_latency if getattr(args, "recognition_latency", None) is not None else float(scenario.get("simulated_recognition_seconds", 0.0))
    camera = DetectorRecognizer(detector) if detector is not None else (
        HttpRecognizer(args.vision_endpoint) if getattr(args, "vision_endpoint", None) else None)
    recognizer = RoutingRecognizer(SimulatedRecognizer({p["id"]: p["label"] for p in props}, latency) if props else None, camera)
    return AgentRuntime(chatbot, builder=builder, recognizer=recognizer, anchors=anchors, config=config,
                        executor=executor, clock=clock or time.monotonic)


def default_args(**overrides: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_service_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args
