from tokenslim.detector import ContentType, detect_content_type


def test_detect_json_object():
    r = detect_content_type('{"a": 1, "b": [1, 2, 3]}')
    assert r.content_type is ContentType.JSON
    assert r.confidence > 0.9


def test_detect_json_array():
    assert detect_content_type("[1, 2, 3]").content_type is ContentType.JSON


def test_invalid_json_is_not_json():
    assert detect_content_type("{not valid json").content_type is not ContentType.JSON


def test_detect_diff():
    diff = "diff --git a/x.py b/x.py\n@@ -1,2 +1,2 @@\n-old\n+new\n"
    assert detect_content_type(diff).content_type is ContentType.DIFF


def test_detect_log():
    log = "\n".join(
        [
            "2024-01-02 13:45:01 INFO starting up",
            "2024-01-02 13:45:02 WARNING low memory",
            "2024-01-02 13:45:03 ERROR boom",
        ]
    )
    assert detect_content_type(log).content_type is ContentType.LOG


def test_detect_code():
    code = "\n".join(
        [
            "def add(a, b):",
            "    return a + b",
            "",
            "class Foo:",
            "    def bar(self):",
            "        return self.x == 1",
        ]
    )
    assert detect_content_type(code).content_type is ContentType.CODE


def test_detect_markdown():
    md = "# Title\n\n- item one\n- item two\n\n```\ncode\n```\n"
    assert detect_content_type(md).content_type is ContentType.MARKDOWN


def test_detect_search_results():
    search = "\n".join(["12:def foo():", "45:    return 1", "78:class Bar:", "90:    pass"])
    assert detect_content_type(search).content_type is ContentType.SEARCH


def test_detect_plain_text():
    text = "This is just an ordinary sentence about a cat sitting on a warm mat."
    assert detect_content_type(text).content_type is ContentType.TEXT


def test_empty_is_text():
    assert detect_content_type("").content_type is ContentType.TEXT
    assert detect_content_type("   \n  ").content_type is ContentType.TEXT


def test_detect_csv_comma():
    text = "id,name,amount\n1,alpha,10\n2,beta,20\n3,gamma,30\n"
    r = detect_content_type(text)
    assert r.content_type is ContentType.CSV
    assert r.confidence > 0.6


def test_detect_csv_semicolon_tab_pipe():
    for delim in (";", "\t", "|"):
        text = "\n".join(delim.join(("a", "b", "c")) for _ in range(5))
        assert detect_content_type(text).content_type is ContentType.CSV, repr(delim)


def test_csv_with_date_fields_beats_search():
    # Leading date-like fields also match the grep-hit regex ("2024-...").
    text = "\n".join(f"2024-12-{d:02d},order-{d},19.99" for d in range(1, 8))
    assert detect_content_type(text).content_type is ContentType.CSV


def test_inconsistent_columns_not_csv():
    text = "a,b\n1,2,3\njust a line\nx,y\n"
    assert detect_content_type(text).content_type is not ContentType.CSV


def test_two_lines_not_csv():
    assert detect_content_type("a,b\n1,2").content_type is not ContentType.CSV


def test_semicolon_terminated_code_not_csv():
    code = "const a = 1;\nconst b = 2;\nlet c = a + b;\n"
    r = detect_content_type(code)
    assert r.content_type is not ContentType.CSV
    assert r.content_type is ContentType.CODE


def test_quoted_delimiters_still_csv():
    text = 'id,note\n1,"hello, world"\n2,"a, b, and c"\n3,plain\n'
    assert detect_content_type(text).content_type is ContentType.CSV


def test_uniform_jsonl_not_csv():
    # Uniform-schema JSON lines have a constant comma field count too, but
    # they are JSON payloads — the CSV branch must not steal them.
    jsonl = "\n".join(f'{{"id": {i}, "name": "item-{i}", "price": {10 + i}}}' for i in range(10))
    assert detect_content_type(jsonl).content_type is not ContentType.CSV
    arrays = "\n".join(f'[{i}, "item-{i}", {10 + i}]' for i in range(10))
    assert detect_content_type(arrays).content_type is not ContentType.CSV
