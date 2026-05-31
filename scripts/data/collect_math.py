# ディレクトリパス: scripts/collect_math.py
# ファイルの日本語タイトル: 数式・自然言語等価コーパスジェネレーター
# ファイルの目的や内容: LaTeX形式の数式と、その自然言語による説明を紐付けた多様なテキストデータを生成する（モジュール化対応）。

import os
import json

default_math_database = [
    {"category": "相対性理論", "name": "質量とエネルギーの等価性", "latex": "E = mc^2", "text": "エネルギーEは、質量mと光速cの2乗の積に等しい"},
    {"category": "相対性理論", "name": "エネルギー・運動量関係", "latex": "E^2 = (mc^2)^2 + (pc)^2", "text": "エネルギーEの2乗は、静止エネルギー(mc^2)の2乗と運動量pと光速cの積の2乗の和に等しい"},
    {"category": "量子力学", "name": "時間依存シュレーディンガー方程式", "latex": "i\\hbar \\frac{\\partial}{\\partial t}\\Psi(\\mathbf{r},t) = \\hat{H}\\Psi(\\mathbf{r},t)", "text": "波動関数の時間微分に虚数単位と換算プランク定数を掛けたものは、ハミルトニアン演算子を波動関数に作用させたものに等しい"},
    {"category": "一般相対論", "name": "アインシュタイン方程式", "latex": "G_{\\mu\\nu} = \\frac{8\\pi G}{c^4}T_{\\mu\\nu}", "text": "時空の歪みを表すアインシュタインテンソルは、物質やエネルギーの分布を表すエネルギー・運動量テンソルに比例する"},
    {"category": "熱力学", "name": "ボルツマンのエントロピー公式", "latex": "S = k_B \\ln \\Omega", "text": "エントロピーSは、微視的状態数Ωの自然対数にボルツマン定数k_Bを掛けたものに等しい"},
    {"category": "ニューラルネット", "name": "多層パーセプトロンの順伝播", "latex": "\\mathbf{y} = \\sigma(W\\mathbf{x} + \\mathbf{b})", "text": "出力ベクトルyは、入力ベクトルxに重み行列Wを掛け、バイアスベクトルbを足し合わせた結果に活性化関数σを適用したものである"}
]

def generate_math_corpus(math_concepts, output_txt_path, output_jsonl_path):
    templates = [
        "{category}における{name}の方程式は、数式で {latex} と記述されます。",
        "数式 {latex} は、{category}の{name}を表しています。",
        "{name}の意味を言葉で説明すると「{text}」となりますが、これを式で表すと {latex} となります。",
        "「{text}」という関係性は、物理学や数学において {latex} （{name}）として知られています。",
        "もし{category}の{name}について問われたら、 {latex} という式を思い浮かべるべきです。"
    ]

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    generated_lines = []
    for concept in math_concepts:
        for template in templates:
            sentence = template.format(**concept)
            generated_lines.append(sentence)
            
    with open(output_txt_path, "w", encoding="utf-8") as f_txt, \
         open(output_jsonl_path, "w", encoding="utf-8") as f_jsonl:
        
        for line in generated_lines:
            f_txt.write(line + "\n")
            
        for concept in math_concepts:
            q1 = {"text": f"You: {concept['category']}の{concept['name']}を数式で教えて。\nSARA: はい、{concept['latex']} です。"}
            q2 = {"text": f"You: {concept['latex']} という数式は何を表していますか？\nSARA: それは{concept['category']}における{concept['name']}を表しており、言葉にすると「{concept['text']}」という意味になります。"}
            f_jsonl.write(json.dumps(q1, ensure_ascii=False) + "\n")
            f_jsonl.write(json.dumps(q2, ensure_ascii=False) + "\n")

    print(f"✅ テキストコーパスを {output_txt_path} に生成しました。")
    print(f"✅ Q&Aコーパスを {output_jsonl_path} に生成しました。")

if __name__ == "__main__":
    generate_math_corpus(default_math_database, "data/interim/math_corpus.txt", "data/interim/math_corpus.jsonl")