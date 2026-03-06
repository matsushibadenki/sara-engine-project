import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.pipelines import pipeline
from sara_engine.pipelines.audio_classification import AudioClassificationPipeline
from sara_engine.pipelines.feature_extraction import FeatureExtractionPipeline
from sara_engine.pipelines.image_classification import ImageClassificationPipeline
from sara_engine.pipelines.summarization import SummarizationPipeline
from sara_engine.pipelines.text_generation import TextGenerationPipeline


class _DummyTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, token_ids):
        return "".join(chr(i) for i in token_ids)


class _DummyGeneratorModel:
    def __init__(self):
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return "generated"


class _DummySummaryModel:
    def __init__(self):
        self.learn_input = None

    def generate(self, **kwargs):
        return "summary"

    def learn_sequence(self, token_ids):
        self.learn_input = token_ids


class _DummyFeatureModel:
    def forward(self, token_ids):
        return [float(len(token_ids))]


class _DummyImageModel:
    def forward(self, image, learning=False, target_class=None):
        return 1


class _DummyAudioModel:
    def forward(self, waveform, learning=False, target_class=None):
        return 0


def test_pipeline_dispatch_for_implemented_tasks():
    tok = _DummyTokenizer()

    assert isinstance(
        pipeline("summarization", model=_DummySummaryModel(), tokenizer=tok),
        SummarizationPipeline,
    )
    assert isinstance(
        pipeline("feature-extraction", model=_DummyFeatureModel(), tokenizer=tok),
        FeatureExtractionPipeline,
    )
    assert isinstance(
        pipeline("image-classification", model=_DummyImageModel()),
        ImageClassificationPipeline,
    )
    assert isinstance(
        pipeline("audio-classification", model=_DummyAudioModel()),
        AudioClassificationPipeline,
    )


def test_text_generation_pipeline_uses_supported_generate_arguments():
    tok = _DummyTokenizer()
    model = _DummyGeneratorModel()
    gen = TextGenerationPipeline(model=model, tokenizer=tok)

    out = gen("hello", max_new_tokens=12, temperature=0.7, top_k=4, repetition_penalty=1.5)

    assert out == "generated"
    assert model.last_kwargs is not None
    assert model.last_kwargs["prompt"] == "hello"
    assert model.last_kwargs["max_new_tokens"] == 12
    assert model.last_kwargs["temperature"] == 0.7
    assert model.last_kwargs["top_k"] == 4
    assert model.last_kwargs["repetition_penalty"] == 1.5


def test_summarization_pipeline_encodes_before_learning():
    tok = _DummyTokenizer()
    model = _DummySummaryModel()
    summ = SummarizationPipeline(model=model, tokenizer=tok)

    summ.learn("abc", "def")

    assert isinstance(model.learn_input, list)
    assert all(isinstance(v, int) for v in model.learn_input)
