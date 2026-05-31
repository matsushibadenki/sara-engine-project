from sara_engine.utils.tokenizer import SaraTokenizer
from sara_engine.utils.project_paths import workspace_path
from sara_engine.pipelines.text_generation import TextGenerationPipeline
from sara_engine.models.spiking_llm import SpikingLLM, SpikingLayerNorm
import json
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))


def test_sara_tokenizer_split_text_keeps_whitespace_and_punctuation():
    tokenizer = SaraTokenizer(
        vocab_size=128, model_path="workspace/test_tokenizer_split.json")
    pieces = tokenizer.split_text("猫 は 走る。")

    assert " " in pieces
    assert "。" in pieces
    assert "猫" in pieces or "走" in pieces


def test_sara_tokenizer_save_writes_complete_json_without_temp_artifact():
    model_path = workspace_path("tests", "tokenizer_atomic_save.json")
    temp_path = f"{model_path}.tmp"
    for path in (model_path, temp_path):
        if os.path.exists(path):
            os.remove(path)

    tokenizer = SaraTokenizer(vocab_size=64, model_path=model_path)
    tokenizer.train(["alpha beta alpha"])

    assert os.path.exists(model_path)
    assert not os.path.exists(temp_path)
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["next_id"] == tokenizer.next_id
    assert payload["vocab"]["<unk>"] == tokenizer.vocab["<unk>"]
    assert isinstance(payload["merges"], list)


def test_spiking_llm_exposes_candidate_scores_and_streaming():
    model = SpikingLLM(num_layers=1, sdr_size=64,
                       vocab_size=256, context_window=4)
    tok_cat = model.tokenizer._add_token("猫")
    tok_topic = model.tokenizer._add_token("は")
    tok_run = model.tokenizer._add_token("走る")
    tok_stop = model.tokenizer._add_token("。")
    model.pretrained_synapses = {
        1: {
            tok_cat: {tok_topic: 1.0},
            tok_topic: {tok_run: 1.0},
            tok_run: {tok_stop: 1.0},
        },
        2: {tok_cat: {tok_run: 1.0}},
    }

    candidates = model.predict_next_tokens(
        prompt_tokens=[tok_cat, tok_topic], top_k=3)

    assert candidates
    assert candidates[0]["token"]
    assert 0.0 <= candidates[0]["confidence"] <= 1.0
    assert isinstance(candidates[0]["supporting_delays"], list)
    assert isinstance(candidates[0]["supporting_chunks"], list)

    streamed = list(model.generate_stream(prompt_tokens=[
                    tok_cat, tok_topic], max_new_tokens=2, top_k=3, temperature=0.0))

    assert streamed
    assert "text" in streamed[0]
    assert "candidates" in streamed[0]
    assert isinstance(streamed[0]["generated_tokens"], list)
    assert [step["token_id"] for step in streamed] == [tok_run, tok_stop]


def test_spiking_llm_registers_unknown_prompt_tokens_and_respects_stop_conditions():
    model = SpikingLLM(num_layers=1, sdr_size=64,
                       vocab_size=256, context_window=4)
    registered = model.generate(prompt="未学習語", max_new_tokens=1)
    prepared_tokens, _, _ = model._prepare_prompt_tokens(prompt="未学習語")
    assert "未知の単語" not in registered
    assert prepared_tokens

    tok_new = prepared_tokens[-1]
    tok_answer = model.tokenizer._add_token("応答")
    tok_stop = model.tokenizer._add_token("END")
    first_delay: dict[int, dict[int, float]] = {}
    for current_token, next_token in zip(prepared_tokens[:-1], prepared_tokens[1:]):
        first_delay.setdefault(current_token, {})[next_token] = 1.0
    first_delay.setdefault(tok_new, {})[tok_answer] = 1.0
    first_delay.setdefault(tok_answer, {})[tok_stop] = 1.0
    model.pretrained_synapses = {
        1: first_delay,
    }

    candidates = model.predict_next_tokens(
        prompt_tokens=prepared_tokens, top_k=3)
    streamed = list(model.generate_stream(prompt_tokens=prepared_tokens,
                    max_new_tokens=4, top_k=3, temperature=0.0, stop_conditions=["応答"]))

    assert candidates
    assert candidates[0]["token_id"] == tok_answer
    assert streamed
    assert streamed[0]["text"] == "応答"
    assert len(streamed) == 1


def test_spiking_llm_status_message_supports_japanese():
    message = SpikingLLM._status_message("missing_knowledge", lang="ja")
    assert "知識ネットワーク" in message


def test_spiking_layer_norm_uses_peak_based_global_inhibition():
    layer_norm = SpikingLayerNorm(
        sdr_size=100,
        base_threshold=1.0,
        target_active_ratio=0.01,
    )

    spikes = layer_norm.forward([3.0, 2.8, 1.2] + [0.0] * 97)

    assert 0 in spikes
    assert 1 in spikes
    assert 2 not in spikes


def test_spiking_llm_uses_hierarchical_chunk_summary_for_older_context():
    model = SpikingLLM(num_layers=1, sdr_size=64, vocab_size=256, context_window=2)
    tok_topic = model.tokenizer._add_token("topic")
    tok_fill_1 = model.tokenizer._add_token("f1")
    tok_fill_2 = model.tokenizer._add_token("f2")
    tok_fill_3 = model.tokenizer._add_token("f3")
    tok_answer = model.tokenizer._add_token("answer")

    model.pretrained_synapses = {
        1: {
            tok_topic: {tok_answer: 1.4},
            tok_fill_1: {tok_fill_2: 0.2},
            tok_fill_2: {tok_fill_3: 0.2},
        }
    }

    candidates, should_block = model._score_next_tokens(
        [tok_topic, tok_fill_1, tok_fill_2, tok_fill_3],
        suppress_initial_symbols=False,
    )

    assert should_block is False
    assert candidates
    assert candidates[0]["token_id"] == tok_answer
    assert candidates[0]["supporting_chunks"] == [0]
    assert candidates[0]["hierarchical_score"] > 0.0


class _DummyTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, token_ids):
        return "".join(chr(i) for i in token_ids)


class _StreamingModel:
    def __init__(self):
        self.last_generate_kwargs = None
        self.last_predict_kwargs = None
        self.last_stream_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return {"sequences": "ok", "scores": [[{"token": "a", "score": 1.0}]]}

    def predict_next_tokens(self, **kwargs):
        self.last_predict_kwargs = kwargs
        return [{"token_id": 97, "token": "a", "score": 1.0, "confidence": 1.0}]

    def generate_stream(self, **kwargs):
        self.last_stream_kwargs = kwargs
        yield {"token_id": 97, "token": "a", "text": "a", "step": 0, "candidates": []}


def test_text_generation_pipeline_exposes_scores_prediction_and_stream():
    model = _StreamingModel()
    pipe = TextGenerationPipeline(model=model, tokenizer=_DummyTokenizer())

    generated = pipe("hello", return_dict_in_generate=True, output_scores=True,
                     top_p=0.8, presence_penalty=0.4, frequency_penalty=0.6)
    predicted = pipe.predict_next_tokens(
        "hello", top_k=1, presence_penalty=0.4, frequency_penalty=0.6)
    streamed = list(pipe.stream("hello", max_new_tokens=1,
                    top_p=0.8, stop_conditions=["a"]))

    assert generated["sequences"] == "ok"
    assert predicted[0]["token"] == "a"
    assert streamed[0]["text"] == "a"
    assert model.last_generate_kwargs["top_p"] == 0.8  # type: ignore[index]
    assert model.last_generate_kwargs["presence_penalty"] == 0.4  # type: ignore[index]
    assert model.last_generate_kwargs["frequency_penalty"] == 0.6  # type: ignore[index]
    assert model.last_predict_kwargs["presence_penalty"] == 0.4  # type: ignore[index]
    assert model.last_stream_kwargs["stop_conditions"] == [  # type: ignore[index]
        "a"]
