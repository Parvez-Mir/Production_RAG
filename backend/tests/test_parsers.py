import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.config import Settings
from app.services.parsers import (
    CsvParser,
    DocumentParseError,
    FileValidationError,
    JsonParser,
    ParserFactory,
    PdfParser,
    TextParser,
    UnsupportedFileTypeError,
)


def write_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_text_and_markdown_parsers_preserve_content(tmp_path: Path) -> None:
    text_path = write_file(tmp_path / "notes.txt", "First line\n\nSecond line")
    markdown_path = write_file(tmp_path / "notes.md", "# Heading\n\n- Item")

    text_document = ParserFactory.parse(text_path)
    markdown_document = ParserFactory.parse(markdown_path)

    assert text_document.content == "First line\n\nSecond line"
    assert text_document.metadata["file_type"] == "txt"
    assert markdown_document.content == "# Heading\n\n- Item"
    assert markdown_document.metadata["file_type"] == "md"


def test_csv_parser_includes_headers_and_rows(tmp_path: Path) -> None:
    path = write_file(tmp_path / "people.csv", "name,age\nJohn,30\nMaria,28\n")

    document = ParserFactory.parse(path)

    assert document.content == "name: John | age: 30\nname: Maria | age: 28"
    assert document.metadata["rows"] == 2
    assert document.metadata["columns"] == 2


def test_json_parser_preserves_nested_structure(tmp_path: Path) -> None:
    path = write_file(tmp_path / "config.json", json.dumps({"app": {"debug": True}}))

    document = ParserFactory.parse(path)

    assert '"debug": true' in document.content
    assert document.metadata["file_type"] == "json"


def test_pdf_parser_extracts_pages(tmp_path: Path) -> None:
    path = tmp_path / "empty-pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as output:
        writer.write(output)

    document = ParserFactory.parse(path)

    assert document.metadata["pages"] == 1
    assert document.content.startswith("[Page 1]")


@pytest.mark.parametrize(
    ("suffix", "parser_type"),
    [(".pdf", PdfParser), (".txt", TextParser), (".md", TextParser), (".csv", CsvParser), (".json", JsonParser)],
)
def test_factory_selects_parser(tmp_path: Path, suffix: str, parser_type: type) -> None:
    path = tmp_path / f"document{suffix}"
    write_file(path, "{}" if suffix == ".json" else "value" if suffix != ".csv" else "value\n1")

    parser = ParserFactory.get_parser(path)

    assert isinstance(parser, parser_type)


def test_factory_handles_uppercase_extensions(tmp_path: Path) -> None:
    path = write_file(tmp_path / "README.MD", "# Read me")

    assert isinstance(ParserFactory.get_parser(path), TextParser)


def test_parser_rejects_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(FileValidationError):
        ParserFactory.parse(tmp_path / "missing.txt")

    unsupported = write_file(tmp_path / "document.exe", "content")
    with pytest.raises(UnsupportedFileTypeError):
        ParserFactory.parse(unsupported)


def test_parser_rejects_empty_and_oversized_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.touch()
    with pytest.raises(FileValidationError):
        ParserFactory.parse(empty)

    oversized = write_file(tmp_path / "large.txt", "x" * (1024 * 1024 + 1))
    settings = Settings(max_file_size_mb=1)
    with pytest.raises(FileValidationError):
        ParserFactory.parse(oversized, settings=settings)


def test_parser_reports_malformed_json(tmp_path: Path) -> None:
    path = write_file(tmp_path / "broken.json", "{not valid json}")

    with pytest.raises(DocumentParseError):
        ParserFactory.parse(path)