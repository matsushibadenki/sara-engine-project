# ディレクトリパス: src/sara_engine/pipelines/text_generation.py
# ファイルの日本語タイトル: テキスト生成パイプライン
# ファイルの目的や内容: SNNを用いたテキスト生成機能のTransformers互換パイプライン実装。

from typing import Any


class TextGenerationPipeline:
    """Pipeline for text generation using a causal Spiking Neural Network.
    Fully compatible with Transformers-like APIs but runs strictly without GPU,
    backpropagation, or matrix multiplication.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

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

        if hasattr(self.model, "generate") and callable(self.model.generate):
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
                    refractory_penalty=refractory_penalty,
                    refractory_period=refractory_period,
                    return_dict_in_generate=return_dict_in_generate,
                    output_scores=output_scores,
                    output_tokens=output_tokens,
                )
            except TypeError:
                if hasattr(self.tokenizer, "encode"):
                    input_ids = self.tokenizer.encode(text)
                else:
                    input_ids = [ord(c) for c in text]
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
                    refractory_penalty=refractory_penalty,
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
            if hasattr(self.tokenizer, "decode"):
                return self.tokenizer.decode(generated)
            return "".join([chr(i) for i in generated if i < 0x110000])

        return text

    def predict_next_tokens(self, text: str, top_k: int = 5, **kwargs) -> list[dict[str, Any]]:
        if hasattr(self.model, "predict_next_tokens"):
            try:
                return self.model.predict_next_tokens(
                    prompt=text,
                    top_k=top_k,
                    repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                    presence_penalty=kwargs.get("presence_penalty", 0.0),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                )
            except TypeError:
                if hasattr(self.tokenizer, "encode"):
                    input_ids = self.tokenizer.encode(text)
                else:
                    input_ids = [ord(c) for c in text]
                return self.model.predict_next_tokens(
                    prompt_tokens=input_ids,
                    top_k=top_k,
                    repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                    presence_penalty=kwargs.get("presence_penalty", 0.0),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                )
        return []

    def stream(self, text: str, max_new_tokens: int = 50, **kwargs):
        if hasattr(self.model, "generate_stream"):
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
                if hasattr(self.tokenizer, "encode"):
                    input_ids = self.tokenizer.encode(text)
                else:
                    input_ids = [ord(c) for c in text]
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
            if hasattr(self.tokenizer, "encode"):
                input_ids = self.tokenizer.encode(text)
            else:
                input_ids = [ord(c) for c in text]
            self.model.learn_sequence(input_ids)
        else:
            print("[Warning] This model does not support online STDP learning.")

    def save_pretrained(self, save_directory: str) -> None:
        """Saves the SNN configuration and synaptic weights to disk."""
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_directory)
        if hasattr(self.tokenizer, "save"):
            self.tokenizer.save()


def pipeline(task: str, model: Any = None, tokenizer: Any = None) -> Any:
    if task == "text-generation":
        if model is None or tokenizer is None:
            raise ValueError(
                "Both 'model' and 'tokenizer' must be provided for the pipeline.")
        return TextGenerationPipeline(model=model, tokenizer=tokenizer)
    else:
        raise NotImplementedError(
            f"The task '{task}' is not yet supported in the SARA engine.")
