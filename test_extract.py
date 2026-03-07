from scripts.data.collect_docs import process_document

# ダミーのHTML抽出テスト（Wikipediaなどを指定）
process_document(
    source_type="html",
    source="https://ja.wikipedia.org/wiki/Python",
    output_path="data/interim/test_corpus.txt"
)