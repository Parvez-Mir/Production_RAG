import pytest

from app.services.prompt import PromptError, PromptFactory, PromptTemplate


def test_default_prompt_substitutes_context_and_query() -> None:
    template = PromptFactory().get_template("default")

    result = template.build("Context from documents:\nSource text", "What is this?")

    assert "Source text" in result.user_message
    assert "Question: What is this?" in result.user_message
    assert "Answer:" in result.user_message
    assert "provided documents" in result.system_prompt


def test_prompt_includes_example_interactions() -> None:
    template = PromptTemplate(
        name="few-shot",
        version="v1",
        system_prompt="Be precise.",
        user_template="Examples:\n{examples}\n\nContext:\n{context}\nQuestion: {query}",
    )

    result = template.build(
        "Paris is the capital of France.",
        "What is the capital of France?",
        examples=[("What is 2 + 2?", "4")],
    )

    assert "Question: What is 2 + 2?" in result.user_message
    assert "Answer: 4" in result.user_message


def test_factory_supports_versions_and_lists_template_keys() -> None:
    factory = PromptFactory()
    factory.register(
        PromptTemplate("default", "v2", "New system", "{context}\n{query}")
    )

    assert factory.get_template("default", "v1").version == "v1"
    assert factory.get_template("default", "v2").system_prompt == "New system"
    assert factory.list_templates() == ["default:v1", "default:v2"]


@pytest.mark.parametrize("context, query", [("", "question"), ("context", " ")])
def test_prompt_rejects_empty_inputs(context: str, query: str) -> None:
    template = PromptFactory().get_template("default")

    with pytest.raises(PromptError):
        template.build(context, query)


def test_factory_rejects_unknown_template() -> None:
    with pytest.raises(PromptError):
        PromptFactory().get_template("missing")