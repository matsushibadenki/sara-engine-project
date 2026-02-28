# ディレクトリパス: scripts/collect_math.py
# ファイルの日本語タイトル: 数式・自然言語等価コーパスジェネレーター
# ファイルの目的や内容: LaTeX形式の数式と、その自然言語による説明を紐付け、双方向の関連性を学習させるための多様なテキストデータを生成・収集する。

import os
import json

def generate_math_corpus(math_concepts, output_path):
    """
    数式の概念リストから、多様な表現パターンのテキストを生成し、
    コーパスファイルに追記する。
    """
    # 表現のバリエーション（テンプレート）
    # これらの多様な文脈を与えることで、数式と文字が等価であることを学習させます。
    templates = [
        "{category}における{name}の方程式は、数式で {latex} と記述されます。",
        "数式 {latex} は、{category}の{name}を表しています。",
        "{name}の意味を言葉で説明すると「{text}」となりますが、これを式で表すと {latex} となります。",
        "「{text}」という関係性は、物理学や数学において {latex} （{name}）として知られています。",
        "もし{category}の{name}について問われたら、 {latex} という式を思い浮かべるべきです。"
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    generated_lines = []
    
    for concept in math_concepts:
        category = concept.get("category", "")
        name = concept.get("name", "")
        latex = concept.get("latex", "")
        text = concept.get("text", "")
        
        # 各テンプレートに当てはめて文章を生成
        for template in templates:
            sentence = template.format(
                category=category,
                name=name,
                latex=latex,
                text=text
            )
            generated_lines.append(sentence)
            
    # Q&A形式（jsonl）としての保存も並行して行うと、対話モデルの学習に有利です。
    jsonl_path = output_path.replace(".txt", ".jsonl")
    
    with open(output_path, "a", encoding="utf-8") as f_txt, \
         open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
        
        for line in generated_lines:
            # コーパス用のテキスト書き込み
            f_txt.write(line + "\n")
            
        for concept in math_concepts:
            # 対話学習用のJSONL書き込み（一問一答形式）
            q1 = {"text": f"ユーザー: {concept['category']}の{concept['name']}を数式で教えて。\nシステム: はい、{concept['latex']} です。"}
            q2 = {"text": f"ユーザー: {concept['latex']} という数式は何を表していますか？\nシステム: それは{concept['category']}における{concept['name']}を表しており、言葉にすると「{concept['text']}」という意味になります。"}
            f_jsonl.write(json.dumps(q1, ensure_ascii=False) + "\n")
            f_jsonl.write(json.dumps(q2, ensure_ascii=False) + "\n")

    print(f"✅ 合計 {len(generated_lines)} 行のテキストコーパスを {output_path} に追加しました。")
    print(f"✅ Q&A形式のデータを {jsonl_path} に追加しました。")

if __name__ == "__main__":
    # 学習させたい数式のデータベース
    # latex(記号)とtext(意味・読み下し)の両方を定義します
    math_database = [
        {
            "category": "相対性理論",
            "name": "質量とエネルギーの等価性",
            "latex": "E = mc^2",
            "text": "エネルギーEは、質量mと光速cの2乗の積に等しい"
        },
        {
            "category": "相対性理論",
            "name": "エネルギー・運動量関係",
            "latex": "E^2 = (mc^2)^2 + (pc)^2",
            "text": "エネルギーEの2乗は、静止エネルギー(mc^2)の2乗と運動量pと光速cの積の2乗の和に等しい"
        },
        {
            "category": "量子力学",
            "name": "時間依存シュレーディンガー方程式",
            "latex": "i\\hbar \\frac{\\partial}{\\partial t}\\Psi(\\mathbf{r},t) = \\hat{H}\\Psi(\\mathbf{r},t)",
            "text": "波動関数の時間微分に虚数単位と換算プランク定数を掛けたものは、ハミルトニアン演算子を波動関数に作用させたものに等しい"
        },
        {
            "category": "一般相対論",
            "name": "アインシュタイン方程式",
            "latex": "G_{\\mu\\nu} = \\frac{8\\pi G}{c^4}T_{\\mu\\nu}",
            "text": "時空の歪みを表すアインシュタインテンソルは、物質やエネルギーの分布を表すエネルギー・運動量テンソルに比例する"
        },
        {
            "category": "熱力学",
            "name": "ボルツマンのエントロピー公式",
            "latex": "S = k_B \\ln \\Omega",
            "text": "エントロピーSは、微視的状態数Ωの自然対数にボルツマン定数k_Bを掛けたものに等しい"
        },
        {
            "category": "ニューラルネット",
            "name": "多層パーセプトロンの順伝播",
            "latex": "\\mathbf{y} = \\sigma(W\\mathbf{x} + \\mathbf{b})",
            "text": "出力ベクトルyは、入力ベクトルxに重み行列Wを掛け、バイアスベクトルbを足し合わせた結果に活性化関数σを適用したものである"
        }
    ]
    
    output_filepath = "data/math_corpus.txt"
    generate_math_corpus(math_database, output_filepath)