# パス: src/sara_engine/pipelines/text_generation.py
# 英語タイトル: Text Generation Pipeline
# 目的や内容: SNNを用いたテキスト生成機能のTransformers互換パイプライン実装。ジェネレータ式への移行等によるメモリ効率と速度の向上を図り、GPU・誤差逆伝播なしでの自己回帰的な推論、オンライン学習(STDP)、ストリーミング生成を提供する。

from typing import Any


class TextGenerationPipeline:
    """Pipeline for text generation using a causal Spiking Neural Network.
    Fully compatible with Transformers-like APIs but runs strictly without GPU,
    backpropagation, or matrix multiplication.
    """

    def __init__(self, model: Any, tokenizer: Any = None):
        self.model = model
        # モデルがトークナイザを内包している場合のフォールバック
        self.tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
        
        # モデルがサポートしているメソッドを事前にキャッシュして高速化
        self._has_model_generate = hasattr(self.model, "generate") and callable(self.model.generate)
        self._has_model_predict = hasattr(self.model, "predict_next_tokens")
        self._has_model_stream = hasattr(self.model, "generate_stream")
        self._has_tokenizer_encode = hasattr(self.tokenizer, "encode") or hasattr(self.tokenizer, "encode_text")
        self._has_tokenizer_decode = hasattr(self.tokenizer, "decode") or hasattr(self.tokenizer, "decode_text")

    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, "encode_text"):
            return self.tokenizer.encode_text(text)
        elif hasattr(self.model, "encode_text"):
            return self.model.encode_text(text)
        return [ord(c) for c in text]

    def _decode(self, tokens: list[int]) -> str:
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens)
        elif hasattr(self.tokenizer, "decode_text"):
            return self.tokenizer.decode_text(tokens)
        elif hasattr(self.model, "decode_text"):
            return self.model.decode_text(tokens)
        return "".join(chr(i) for i in tokens if i < 0x110000)

    def __call__(self, text: str, max_new_tokens: int = 50, **kwargs) -> Any:
        """
        Generates text based on the provided prompt string.
        Utilizes biologically inspired parameters like refractory_period instead of repetition_penalty.
        """
        # Ensure biological consistency in parameters mapping from Transformers kwargs
        refractory_penalty = kwargs.get("repetition_penalty", 1.2)
        refractory_period = kwargs.get("refractory_period", 10)
        temperature = kwargs.get("temperature", 0.5)
        top_k = kwargs.get("top_k", 3)
        top_p = kwargs.get("top_p", 1.0)
        presence_penalty = kwargs.get("presence_penalty", 0.0)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)

        stop_conditions = kwargs.get("stop_conditions")
        return_dict_in_generate = kwargs.get("return_dict_in_generate", False)
        output_scores = kwargs.get("output_scores", False)
        output_tokens = kwargs.get("output_tokens", False)

        if self._has_model_generate:
            try:
                generated = self.model.generate(
                    prompt=text,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=refractory_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop_conditions=stop_conditions,
                    refractory_period=refractory_period,
                    return_dict_in_generate=return_dict_in_generate,
                    output_scores=output_scores,
                    output_tokens=output_tokens,
                )
            except TypeError:
                input_ids = self._encode(text)
                generated = self.model.generate(
                    prompt_tokens=input_ids,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=refractory_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop_conditions=stop_conditions,
                    refractory_period=refractory_period,
                    return_dict_in_generate=return_dict_in_generate,
                    output_scores=output_scores,
                    output_tokens=output_tokens,
                )
            except Exception as e:
                print(f"[TextGenerationPipeline] Error during generation: {e}")
                return text

            if isinstance(generated, dict):
                return generated
            if isinstance(generated, str):
                return generated
            if self._has_tokenizer_decode or hasattr(self.model, "decode_text"):
                return self._decode(generated)
            # ジェネレータ式を使用してメモリ効率を向上
            return "".join(chr(i) for i in generated if i < 0x110000)

        return text

    def predict_next_tokens(self, text: str, top_k: int = 5, **kwargs) -> list[dict[str, Any]]:
        if self._has_model_predict:
            try:
                return self.model.predict_next_tokens(
                    prompt=text,
                    top_k=top_k,
                    repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                    presence_penalty=kwargs.get("presence_penalty", 0.0),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                )
            except TypeError:
                input_ids = self._encode(text)
                return self.model.predict_next_tokens(
                    prompt_tokens=input_ids,
                    top_k=top_k,
                    repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                    presence_penalty=kwargs.get("presence_penalty", 0.0),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                )
        return []

    def stream(self, text: str, max_new_tokens: int = 50, **kwargs):
        if self._has_model_stream:
            try:
                return self.model.generate_stream(
                    prompt=text,
                    max_new_tokens=max_new_tokens,
                    top_k=kwargs.get("top_k", 3),
                    top_p=kwargs.get("top_p", 1.0),
                    temperature=kwargs.get("temperature", 0.5),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                    presence_penalty=kwargs.get("presence_penalty", 0.0),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                    stop_conditions=kwargs.get("stop_conditions"),
                )
            except TypeError:
                input_ids = self._encode(text)
                return self.model.generate_stream(
                    prompt_tokens=input_ids,
                    max_new_tokens=max_new_tokens,
                    top_k=kwargs.get("top_k", 3),
                    top_p=kwargs.get("top_p", 1.0),
                    temperature=kwargs.get("temperature", 0.5),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                    presence_penalty=kwargs.get("presence_penalty", 0.0),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                    stop_conditions=kwargs.get("stop_conditions"),
                )
        return iter(())

    def learn(self, text: str) -> None:
        """
        Trains the SNN locally on the provided sequence using STDP.
        This allows continuous, energy-efficient learning on the edge.
        """
        if hasattr(self.model, "learn_sequence"):
            input_ids = self._encode(text)
            self.model.learn_sequence(input_ids)
        else:
            print("[Warning] This model does not support online STDP learning.")

    def save_pretrained(self, save_directory: str) -> None:
        """Saves the SNN configuration and synaptic weights to disk."""
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_directory)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save"):
            self.tokenizer.save()


def pipeline(task: str, model: Any = None, tokenizer: Any = None) -> Any:
    if task == "text-generation":
        if model is None:
            raise ValueError("The 'model' must be provided for the pipeline.")
        # tokenizerが未提供でも、SpikingLLMのように内部に保持しているケースを許容
        return TextGenerationPipeline(model=model, tokenizer=tokenizer)
    else:
        raise NotImplementedError(
            f"The task '{task}' is not yet supported in the SARA engine.")