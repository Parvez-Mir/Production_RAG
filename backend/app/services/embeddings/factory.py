from app.config import Settings, get_settings
from app.services.embeddings.base import EmbeddingError, EmbeddingProvider
from app.services.embeddings.sentence_transformer import SentenceTransformerProvider


class EmbeddingFactory:
    @staticmethod
    def create(
        settings: Settings | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> EmbeddingProvider:
        if provider is not None:
            return provider

        settings = settings or get_settings()
        provider_name = settings.embedding_provider.lower().replace("_", "-")
        if provider_name in {"sentence-transformers", "sentence-transformer"}:
            return SentenceTransformerProvider(settings)
        raise EmbeddingError(f"Unknown embedding provider: {settings.embedding_provider}")