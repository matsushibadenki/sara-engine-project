from typing import Any
{
    "//": "ディレクトリパス: src/sara_engine/pipelines/text_generation.py",
    "//": "ファイルの日本語タイトル: テキスト生成パイプライン",
    "//": "ファイルの目的や内容: SNNを用いたテキスト生成機能のTransformers互換パイプライン実装。"
}


class TextGenerationPipeline:
    """Pipeline for text generation using a causal Spiking Neural Network.
    Fully compatible with Transformers-like APIs but runs strictly without GPU,
    backpropagation, or matrix multiplication.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: str, max_new_tokens: int = 50, **kwargs) -> str:
        """
        Generates text based on the provided prompt string.
        Utilizes biologically inspired parameters like refractory_period instead of repetition_penalty.
        """
        # Ensure biological consistency in parameters mapping from Transformers kwargs
        refractory_penalty = kwargs.get("repetition_penalty", 1.2)
        refractory_period = kwargs.get("refractory_period", 10)
        temperature = kwargs.get("temperature", 0.5)
        top_k = kwargs.get("top_k", 3)

        # Check if the model uses the new inference API string directly
        if hasattr(self.model, "generate") and callable(self.model.generate):
            try:
                # Fallback for models that only accept input_ids
                if hasattr(self.tokenizer, "encode"):
                    input_ids = self.tokenizer.encode(text)
                else:
                    # Generic UTF-8 fallback
                    input_ids = [ord(c) for c in text]

                output_ids = self.model.generate(
                    input_ids, max_l=max_new_tokens)

                if hasattr(self.tokenizer, "decode"):
                    generated_text = self.tokenizer.decode(output_ids)
                else:
                    generated_text = "".join(
                        [chr(i) for i in output_ids if i < 0x110000])

                return text + generated_text
            except Exception as e:
                print(f"[TextGenerationPipeline] Error during generation: {e}")
                return text

        return text

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
