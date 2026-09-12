import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Base exception for document parser failures."""


class FileValidationError(ParserError):
    """Raised when a source file cannot be safely parsed."""


class UnsupportedFileTypeError(ParserError):
    """Raised when no parser is registered for a file extension."""


class DocumentParseError(ParserError):
    """Raised when a supported file cannot be decoded or parsed."""


@dataclass
class Document:
    content: str
    metadata: dict[str, Any]


class BaseParser:
    file_type: str

    def parse(self, file_path: Path) -> Document:
        raise NotImplementedError

    def _metadata(self, file_path: Path, **extra: Any) -> dict[str, Any]:
        return {
            "source": file_path.name,
            "file_type": self.file_type,
            "size_bytes": file_path.stat().st_size,
            **extra,
        }


class TextParser(BaseParser):
    def __init__(self, file_type: str) -> None:
        self.file_type = file_type

    def parse(self, file_path: Path) -> Document:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except OSError as exc:
                raise DocumentParseError(f"Could not read {file_path.name}") from exc
        except OSError as exc:
            raise DocumentParseError(f"Could not read {file_path.name}") from exc

        return Document(content=content, metadata=self._metadata(file_path))


class JsonParser(BaseParser):
    file_type = "json"

    def parse(self, file_path: Path) -> Document:
        try:
            with file_path.open(encoding="utf-8") as source:
                data = json.load(source)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentParseError(f"Could not parse JSON file {file_path.name}") from exc

        return Document(
            content=json.dumps(data, indent=2, ensure_ascii=False),
            metadata=self._metadata(file_path),
        )


class CsvParser(BaseParser):
    file_type = "csv"

    def parse(self, file_path: Path) -> Document:
        try:
            frame = pd.read_csv(file_path, keep_default_na=False)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DocumentParseError(f"Could not parse CSV file {file_path.name}") from exc

        rows = [
            " | ".join(f"{column}: {row[column]}" for column in frame.columns)
            for _, row in frame.iterrows()
        ]
        return Document(
            content="\n".join(rows),
            metadata=self._metadata(
                file_path,
                rows=len(frame.index),
                columns=len(frame.columns),
            ),
        )


class PdfParser(BaseParser):
    file_type = "pdf"

    def parse(self, file_path: Path) -> Document:
        try:
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise DocumentParseError(f"Could not parse PDF file {file_path.name}") from exc

        content = "\n\n".join(
            f"[Page {page_number}]\n{text}" for page_number, text in enumerate(pages, start=1)
        )
        return Document(
            content=content,
            metadata=self._metadata(file_path, pages=len(pages)),
        )


class ParserFactory:
    _parsers = {
        ".pdf": PdfParser,
        ".txt": lambda: TextParser("txt"),
        ".md": lambda: TextParser("md"),
        ".csv": CsvParser,
        ".json": JsonParser,
    }

    @classmethod
    def get_parser(cls, file_path: Path) -> BaseParser:
        parser_factory = cls._parsers.get(file_path.suffix.lower())
        if parser_factory is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_path.suffix or '<none>'}"
            )
        return parser_factory()

    @classmethod
    def parse(cls, file_path: str | Path, settings: Settings | None = None) -> Document:
        path = Path(file_path)
        settings = settings or get_settings()
        cls._validate(path, settings)
        parser = cls.get_parser(path)
        logger.info("Parsing %s as %s", path.name, path.suffix.lower().lstrip("."))
        document = parser.parse(path)
        logger.info("Parsed %s successfully", path.name)
        return document

    @staticmethod
    def _validate(file_path: Path, settings: Settings) -> None:
        if not file_path.exists() or not file_path.is_file():
            raise FileValidationError(f"File does not exist: {file_path}")

        extension = file_path.suffix.lower().lstrip(".")
        allowed_types = {file_type.lower().lstrip(".") for file_type in settings.allowed_file_types}
        if extension not in allowed_types:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_path.suffix or '<none>'}"
            )

        size_bytes = file_path.stat().st_size
        if size_bytes == 0:
            raise FileValidationError(f"File is empty: {file_path.name}")
        if size_bytes > settings.max_file_size_mb * 1024 * 1024:
            raise FileValidationError(
                f"File exceeds the {settings.max_file_size_mb} MB limit: {file_path.name}"
            )