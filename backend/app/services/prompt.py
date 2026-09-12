from dataclasses import dataclass, field
from typing import TypeAlias


class PromptError(Exception):
    """Raised when a prompt template cannot be built or selected."""


ExampleInteraction: TypeAlias = tuple[str, str]


@dataclass(frozen=True)
class PromptOutput:
    system_prompt: str
    user_message: str


@dataclass(frozen=True)
class PromptTemplate:
    """Versioned system and user prompt templates for an LLM."""

    name: str
    version: str
    system_prompt: str
    user_template: str
    examples: tuple[ExampleInteraction, ...] = field(default_factory=tuple)

    def build(
        self,
        context: str,
        query: str,
        examples: list[ExampleInteraction] | None = None,
    ) -> PromptOutput:
        if not context or not context.strip():
            raise PromptError("Context cannot be empty")
        if not query or not query.strip():
            raise PromptError("Query cannot be empty")

        selected_examples = self.examples if examples is None else tuple(examples)
        examples_text = self._format_examples(selected_examples)
        user_message = self.user_template.format(
            context=context,
            query=query,
            examples=examples_text,
        )
        return PromptOutput(
            system_prompt=self.system_prompt,
            user_message=user_message,
        )

    def _format_examples(self, examples: tuple[ExampleInteraction, ...]) -> str:
        if not examples:
            return ""

        formatted: list[str] = []
        for question, answer in examples:
            if not question.strip() or not answer.strip():
                raise PromptError("Example questions and answers cannot be empty")
            formatted.append(f"Question: {question}\nAnswer: {answer}")
        return "\n\n".join(formatted)


class PromptFactory:
    """Registry for named prompt templates and prompt versions."""

    DEFAULT_NAME = "default"

    def __init__(self, templates: list[PromptTemplate] | None = None) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        for template in templates or [self.default_template()]:
            self.register(template)

    def register(self, template: PromptTemplate) -> None:
        if not template.name.strip():
            raise PromptError("Template name cannot be empty")
        if not template.version.strip():
            raise PromptError("Template version cannot be empty")
        self._templates[self._key(template.name, template.version)] = template

    def get_template(self, name: str, version: str | None = None) -> PromptTemplate:
        if version is not None:
            template = self._templates.get(self._key(name, version))
        else:
            matching = [
                template
                for template in self._templates.values()
                if template.name == name
            ]
            template = matching[-1] if matching else None

        if template is None:
            version_text = f" version '{version}'" if version else ""
            raise PromptError(f"Unknown prompt template '{name}'{version_text}")
        return template

    def list_templates(self) -> list[str]:
        return sorted(self._templates)

    @classmethod
    def default_template(cls) -> PromptTemplate:
        return PromptTemplate(
            name=cls.DEFAULT_NAME,
            version="v1",
            system_prompt=(
                "You are a helpful assistant that answers questions based on provided documents.\n\n"
                "Instructions:\n"
                "1. Answer based ONLY on the provided context.\n"
                "2. If the information is not in the context, say \"I don't have this information\".\n"
                "3. Cite the source document for important claims.\n"
                "4. Be concise but thorough.\n"
                "5. If the question is ambiguous, ask for clarification."
            ),
            user_template=(
                "Here are relevant documents to answer your question:\n\n"
                "{context}\n\n"
                "{examples}\n\n"
                "Question: {query}\n\n"
                "Answer:"
            ),
        )

    def _key(self, name: str, version: str) -> str:
        return f"{name}:{version}"