_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/token_classification.py",
    "//": "ファイルの日本語タイトル: トークン分類パイプライン",
    "//": "ファイルの目的や内容: Transformersのpipeline('token-classification')をSNNで再現。SpikingTokenClassifierのforward仕様に合わせてシーケンス全体を一括処理するように修正。"
}

from typing import Union, List, Dict, Any
import inspect

class TokenClassificationPipeline:
    """
    Token classification pipeline using an SNN Token Classifier.
    Used for Named Entity Recognition (NER), POS tagging, etc.
    Evaluates spike rates sequentially without backpropagation.
    """
    def __init__(self, model: Any, tokenizer: Any, **kwargs: Any):
        self.model = model
        self.tokenizer = tokenizer
        # デフォルトのNER用ラベル（必要に応じてkwargsから上書き可能）
        self.id2label = kwargs.get("id2label", {
            0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 
            5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
        })

    def __call__(self, text: Union[str, List[str]], **kwargs: Any) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Classifies each token in the provided text(s) sequentially using SNN dynamics.
        """
        is_batched = isinstance(text, list)
        # mypyエラー対策: strのリストであることを明示的に定義
        texts: List[str] = text if isinstance(text, list) else [text]
        
        all_results = []
        for t in texts:
            # バイトレベル等でエンコード
            token_ids = self.tokenizer.encode(t)
            tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
            
            # SNNによる推論 (内部状態の初期化)
            if hasattr(self.model, 'reset_state'):
                self.model.reset_state()
                
            # モデルのforwardにシーケンス全体を渡す
            predicted_class_ids = self.model.forward(token_ids, learning=False)
            
            # 万が一スカラーが返された場合のフォールバック
            if not isinstance(predicted_class_ids, list):
                predicted_class_ids = [predicted_class_ids] * len(token_ids)
            
            result = []
            current_offset = 0
            for idx, (tok, cid) in enumerate(zip(tokens, predicted_class_ids)):
                if idx >= len(predicted_class_ids):
                    break
                
                label = self.id2label.get(cid, f"LABEL_{cid}")
                start = current_offset
                
                # トークンの文字列長(バイト長)を計算
                tok_len = len(tok.encode('utf-8')) if hasattr(tok, 'encode') else 1
                end = current_offset + tok_len
                current_offset = end
                
                # スパイク駆動のため厳密な確率(Softmax)は存在しない
                result.append({
                    "entity": label,
                    "score": 1.0,
                    "index": idx,
                    "word": tok,
                    "start": start,
                    "end": end
                })
            all_results.append(result)

        return all_results if is_batched else all_results[0]

    def learn(self, text: str, labels: List[int]) -> None:
        """
        Trains the SNN classifier locally on the sequence using STDP and Reward-modulated learning.
        """
        token_ids = self.tokenizer.encode(text)
        
        # 配列長の調整 (トークン数とラベル数が一致しない場合の安全処理)
        if len(token_ids) != len(labels):
             if len(token_ids) > len(labels):
                 token_ids = token_ids[:len(labels)]
             else:
                 labels = labels[:len(token_ids)]
                 
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state()
            
        # forwardメソッドの引数にシーケンス全体と正解ラベルのリストを渡す
        sig = inspect.signature(self.model.forward)
        
        if 'target_classes' in sig.parameters:
            self.model.forward(token_ids, learning=True, target_classes=labels)
        else:
            # 万が一別のモデルでループ処理が必要な場合へのフォールバック
            for tid, tgt_cid in zip(token_ids, labels):
                self.model.forward([tid], learning=True, target_classes=[tgt_cid])

    def save_pretrained(self, save_directory: str) -> None:
        """Saves the SNN model state (synaptic weights, thresholds, etc.)."""
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_directory)