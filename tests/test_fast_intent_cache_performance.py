"""Reuse fixed identity replies without changing temporal state refresh."""
from unittest.mock import patch

from sara_engine.inference import SaraInference


def engine():
    result = SaraInference.__new__(SaraInference)
    result.session_memory = {}
    result.lif_network = None
    return result


def test_identity_cache_reuses_latest_turn_across_multilingual_history():
    for question in ('Who are you?', 'あなたは誰ですか？'):
        runtime = engine()
        with patch.object(runtime, '_fast_intent_response', wraps=runtime._fast_intent_response) as build:
            first = runtime.generate('You: Older English history\nSARA: Old response\nYou: '+question+'\nSARA:')
            second = runtime.generate('You: 過去の会話，之前的对话\nSARA: 別の返答\nYou: '+question+'\nSARA:')
        assert first == second == 'I am SARA, a CPU-first spiking neural network assistant.'
        assert build.call_count == 1
        assert len(runtime.fast_intent_cache) == 1
        assert len(next(iter(runtime.fast_intent_cache))[0]) == len(question.lower())


def test_other_intents_keep_full_prompt_key():
    runtime = engine()
    for question in ('What is my name?', '你好', 'こんにちは', 'who is SARA?'):
        prompt = 'You: Who are you?\nSARA: Older answer\nYou: '+question+'\nSARA:'
        assert runtime._fast_intent_cache_key(prompt) == (prompt.strip().lower(), '||')
    assert runtime._fast_intent_cache_key(123) == ('123', '||')


def test_identity_cache_keeps_session_signature_separation():
    runtime = engine()
    prompt = 'You: Who are you?\nSARA:'
    runtime.generate(prompt)
    runtime.session_memory['location'] = 'Tokyo'
    with patch.object(runtime, '_fast_intent_response', wraps=runtime._fast_intent_response) as build:
        runtime.generate(prompt)
    assert build.call_count == 1


def test_outputs_diagnostics_and_temporal_refresh_match_legacy_keys():
    updated, legacy = engine(), engine()
    legacy._fast_intent_cache_key = lambda prompt: (str(prompt).strip().lower(), legacy._session_memory_signature())
    for runtime in (updated, legacy):
        runtime.session_memory = {'goal': 'learn programming', 'task': 'study python', 'location': 'Tokyo'}
    with patch.object(updated, '_refresh_predictor_state', wraps=updated._refresh_predictor_state) as new_refresh, patch.object(
            legacy, '_refresh_predictor_state', wraps=legacy._refresh_predictor_state) as old_refresh:
        for i in range(5):
            prompt = f'You: Archived turn {i}\nSARA: Older answer\nYou: Who are you?\nSARA:'
            assert updated.generate(prompt) == legacy.generate(prompt)
            for name in ('session_memory', 'predictor_state', 'adaptation_state', 'future_state_runtime_state', 'retrieval_diagnostics'):
                assert getattr(updated, name) == getattr(legacy, name)
        assert new_refresh.call_count == old_refresh.call_count
