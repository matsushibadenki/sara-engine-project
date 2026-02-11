About Tools  
  
### 1. 診断・テストツールの確認

`mode` 引数（`debug` または `test`）が必要です。

```bash
# デバッグモード（エンジンの内部状態チェック）
python examples/run_diagnostics.py debug

# 学習テストモード（簡単な単語ペアの学習確認）
python examples/run_diagnostics.py test

```

### 2. 分類タスクの確認

`task` 引数（`text` または `mnist`）が必要です。

```bash
# テキスト分類（すぐに終わります）
python examples/run_classifier.py text

# MNIST画像分類（データダウンロードが発生します。時間がなければ省略可）
python examples/run_classifier.py mnist --epochs 1 --samples 100

```

### 3. チャットの確認（完了済み）

```bash
# チャット開始
python examples/run_chat.py
```
---