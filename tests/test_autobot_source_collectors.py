from bot.collectors_plugins import arxiv_abstract_collector, official_docs_collector


def test_official_docs_collector_extracts_visible_text_and_metadata():
    html = """
    <html>
      <head><style>.hidden { display: none; }</style><script>secret()</script></head>
      <body>
        <h1>Sparse Runtime Guide</h1>
        <p>SARA uses sparse event routing for CPU first operation.</p>
        <p>Local updates keep runtime learning bounded and inspectable.</p>
      </body>
    </html>
    """

    text = official_docs_collector.extract_visible_text(html)
    record = official_docs_collector.build_doc_record("https://docs.example.org/sara/runtime", html)

    assert "Sparse Runtime Guide" in text
    assert "secret()" not in text
    assert record["source"] == "official_docs"
    assert record["meta"]["source_type"] == "official_docs"
    assert record["meta"]["source_domain"] == "docs.example.org"
    assert record["meta"]["license_hint"] == "official_documentation_reference"


def test_arxiv_collector_parses_abstract_metadata_only():
    atom_xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2601.00001</id>
        <published>2026-01-01T00:00:00Z</published>
        <title> Sparse Events for Efficient Local Learning </title>
        <summary>
          We study local plasticity and sparse event routing for CPU first systems.
        </summary>
        <author><name>Ada Researcher</name></author>
        <author><name>Lin Scientist</name></author>
      </entry>
    </feed>
    """

    entries = arxiv_abstract_collector.parse_arxiv_entries(atom_xml)
    record = arxiv_abstract_collector.build_abstract_record(entries[0], "sparse event routing")

    assert len(entries) == 1
    assert entries[0]["title"] == "Sparse Events for Efficient Local Learning"
    assert entries[0]["authors"] == ["Ada Researcher", "Lin Scientist"]
    assert record["source"] == "arxiv_abstract"
    assert record["meta"]["source_type"] == "arxiv_abstract"
    assert record["meta"]["license_hint"] == "abstract_metadata_only_mixed_license_preprint"
    assert "Abstract:" in record["record_text"]


def test_arxiv_query_url_is_bounded_and_uses_query_terms():
    url = arxiv_abstract_collector.build_query_url("neuromorphic sparse learning", max_results=100)

    assert url.startswith("https://export.arxiv.org/api/query?")
    assert "neuromorphic+sparse+learning" in url
    assert "max_results=25" in url
