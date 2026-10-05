import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from wearable_mr.chatbot import Chatbot, CommandChatbot, OpenAICompatibleChatbot
from wearable_mr.clients import OpenAICompatibleClient, ServiceError
from wearable_mr.knowledge import DomainKnowledge
from wearable_mr.sentiment import (HttpSentimentClassifier, LexiconSentimentClassifier, OpenAICompatibleSentimentClassifier,
                                   Sentiment, SentimentEngine)

KNOWLEDGE = DomainKnowledge({
    "chair": {"overview": "A chair supports a seated person.", "topics": {"use": "A chair provides a seat."}},
    "rose": {"overview": "This is a rose.", "topics": {"care": {"text": "Water roses at the base.", "keywords": ["water", "watering"]},
                                                       "flower season": "Roses flower from late spring."}},
})


def test_topics_match_whole_words_not_substrings():
    # Regression: "use" used to fire inside "because".
    assert KNOWLEDGE.match("Why is it here, because I wonder?", "chair").kind == "overview"
    assert KNOWLEDGE.match("What is the use of it?", "chair").topic == "use"
    assert KNOWLEDGE.match("How often should I be watering it?", "rose").topic == "care"
    assert KNOWLEDGE.match("When is the flower season?", "rose").topic == "flower season"


def test_general_conversation_works_with_and_without_an_object():
    assert KNOWLEDGE.match("Hello there", None).kind == "general"
    assert KNOWLEDGE.match("Who are you?", None).intent == "identity"
    # Chit-chat during an object conversation is not answered with the object overview.
    assert KNOWLEDGE.match("Who are you?", "rose").kind == "general"
    assert KNOWLEDGE.match("What is this?", None).kind == "need_object"
    assert KNOWLEDGE.match("What is this?", "rose").kind == "overview"
    assert KNOWLEDGE.match("Tell me about roses", None).object_label == "rose"
    assert KNOWLEDGE.match("Yes please", None).intent == "affirm"
    assert "rose" in KNOWLEDGE.match("What flowers do you know?", None).text


def test_lexicon_sentiment_covers_four_classes_and_three_levels():
    classify = LexiconSentimentClassifier().classify
    careful = classify("Careful, it is sharp.")
    assert (careful.sentiment_class, careful.level) == ("Fear", "Medium")
    assert classify("Careful, the thorns are sharp.").level == "High"
    assert classify("Warning: this flower is deadly and toxic to cats!").level == "High"
    assert classify("Sadly, the old tree died last winter.").sentiment_class == "Sad"
    assert classify("Vandals destroyed it; that is outrageous.").sentiment_class == "Angry"
    assert classify("It is nice.").level == "Low"
    neutral = classify("A rose is a flower.")
    assert (neutral.sentiment_class, neutral.level, neutral.source) == ("Joy", "Low", "lexicon-default")
    assert classify("I am not happy about it.").sentiment_class == "Sad"
    assert classify("It is very very wonderful!").level == "High"


def test_sentiment_maps_to_renderer_expression_presets():
    assert Sentiment("Joy", "High").expression == {"name": "happiness", "level": 3}
    assert Sentiment("Angry", "Low").expression == {"name": "anger", "level": 1}
    assert Sentiment("Sad", "Medium").expression == {"name": "sadness", "level": 2}
    assert Sentiment("Fear", "High").to_dict()["expression"] == {"name": "fear", "level": 3}
    with pytest.raises(ValueError):
        Sentiment("Neutral", "Low")


class _Stub(BaseHTTPRequestHandler):
    calls: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append((self.path, body, self.headers.get("Authorization")))
        if self.path.endswith("/sentiment"):
            answer = {"label": "scared", "score": 0.91}
        elif self.path.endswith("/broken/chat/completions"):
            self.send_response(500); self.end_headers(); return
        elif "classify" in body["messages"][0]["content"].lower():
            answer = {"choices": [{"message": {"content": "```json\n{\"class\": \"Sad\", \"level\": \"Medium\"}\n```"}}]}
        else:
            answer = {"choices": [{"message": {"content": "Roses are wonderful, " + body["messages"][-1]["content"]}}]}
        data = json.dumps(answer).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


@pytest.fixture()
def stub():
    _Stub.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown(); server.server_close()


def test_openai_compatible_chatbot_is_grounded_and_returns_reply_class_level(stub, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret")
    client = OpenAICompatibleChatbot(OpenAICompatibleClient(stub + "/v1", "guide-model", "TEST_KEY"), KNOWLEDGE)
    bot = Chatbot(client, fallback=KNOWLEDGE)
    reply = bot.ask("how do I water it?", "rose")
    assert reply.text.startswith("Roses are wonderful") and reply.source == "openai-compatible"
    assert (reply.sentiment.sentiment_class, reply.sentiment.level) == ("Joy", "Medium")
    path, body, auth = _Stub.calls[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer secret" and body["model"] == "guide-model"
    assert "Water roses at the base." in body["messages"][0]["content"]
    bot.ask("and in summer?", "rose")  # follow-up carries history
    assert [m["role"] for m in _Stub.calls[-1][1]["messages"]] == ["system", "user", "assistant", "user"]


def test_failed_client_falls_back_to_offline_knowledge(stub):
    bot = Chatbot(OpenAICompatibleChatbot(OpenAICompatibleClient(stub + "/broken", "m", None)), fallback=KNOWLEDGE)
    reply = bot.ask("What is this?", "rose")
    assert reply.text == "This is a rose." and reply.source.endswith("(fallback)") and "HTTP 500" in reply.error
    with pytest.raises(ServiceError):
        Chatbot(OpenAICompatibleChatbot(OpenAICompatibleClient(stub + "/broken", "m", None))).ask("hi", None)


def test_remote_sentiment_classifiers_and_fallback(stub):
    assert HttpSentimentClassifier(stub + "/sentiment").classify("x") == Sentiment("Fear", "High", 0.91, "http")
    llm = OpenAICompatibleSentimentClassifier(OpenAICompatibleClient(stub + "/v1", "m", None))
    assert (llm.classify("It wilted.").sentiment_class, llm.classify("It wilted.").level) == ("Sad", "Medium")
    engine = SentimentEngine(HttpSentimentClassifier("http://127.0.0.1:9/none", timeout=.5))
    result = engine.analyze("Beware, it is poisonous!")
    assert result.sentiment_class == "Fear" and result.source == "lexicon-fallback" and engine.last_error


def test_local_command_chatbot(tmp_path):
    script = tmp_path / "bot.py"
    script.write_text("import json,sys\nq=json.load(sys.stdin)\nprint(json.dumps({'reply': 'You asked about ' + str(q['object'])}))\n")
    reply = Chatbot(CommandChatbot([sys.executable, str(script)])).ask("what is this", "lily")
    assert reply.text == "You asked about lily" and reply.source == "command"
