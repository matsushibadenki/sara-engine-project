_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/text_generation.py",
    "//": "ファイルの日本語タイトル: テキスト生成パイプライン",
    "//": "ファイルの目的や内容: SNNモデルを利用して自然言語を自己回帰的に生成するパイプライン。結合時のスペース処理を改善。"
}

import re
from typing import Union, List, Dict
from .base import Pipeline

class TextGenerationPipeline(Pipeline):
    """
    Text generation pipeline using SNN Transformer.
    Replaces traditional ANN-based text generation pipelines.
    It purely relies on biological spike patterns (no backprop, no dense matrix mult, CPU-only).
    """
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)

    def __call__(self, text_inputs: Union[str, List[str]], max_new_tokens: int = 50, **kwargs) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
        """
        Generates text based on the provided input strings.
        
        Args:
            text_inputs: A string or a list of strings to be used as prompts.
            max_new_tokens: The maximum number of new tokens to generate.
            
        Returns:
            A list of dictionaries containing the generated text.
        """
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        results = []
        for text in text_inputs:
            if self.tokenizer is not None and hasattr(self.model, 'forward_step'):
                # Encode text to discrete token IDs for multi-lingual spike translation
                token_ids = self.tokenizer.encode(text)
                
                # Reset biological states (membrane potentials, delay buffers) for a fresh context
                self.model.reset_state()
                
                # Context feeding (Establish reservoir states without STDP weight updates)
                last_token = 0
                for tid in token_ids:
                    last_token = self.model.forward_step(tid, learning=False)
                
                # Auto-regressive generation using spike events
                generated_ids = []
                current_token = last_token
                for _ in range(max_new_tokens):
                    # Check for EOS token (ID 3). If model predicts end of sequence, stop generating.
                    if current_token == 3:
                        break
                    # Avoid appending special tokens like UNK(0), PAD(1), BOS(2)
                    if current_token > 3:
                        generated_ids.append(current_token)
                        
                    current_token = self.model.forward_step(current_token, learning=False)
                
                # Decode the generated biological spike patterns back to human-readable text
                generated_text = self.tokenizer.decode(generated_ids)
                
                # プロンプト(text)の末尾と生成テキストの先頭が英数字なら空白を挟む
                if text and re.match(r'[A-Za-z0-9]', text[-1]) and generated_text and re.match(r'[A-Za-z0-9]', generated_text[0]):
                    full_text = text + " " + generated_text
                else:
                    full_text = text + generated_text
            else:
                # Fallback to the model's native byte-level string generation
                full_text = self.model.generate(text, max_length=max_new_tokens)

            results.append({"generated_text": full_text})

        if len(results) == 1:
            return results
        return results