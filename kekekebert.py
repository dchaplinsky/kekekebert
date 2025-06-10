"""
Token-level embeddings extraction using SBERT and spaCy.

This module provides functionality to extract token-level embeddings from text
using sentence-transformers (SBERT) while maintaining alignment with spaCy
tokenization. It also includes utilities for pooling embeddings using various
strategies (mean, max, weighted mean, etc.).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sentence_transformers.util import cos_sim

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


@dataclass
class TokenEmbeddingsResult:
    """Result container for token-level embedding extraction.

    Attributes:
        token_embeddings: Embeddings for each SBERT token as numpy array of shape (n_tokens, embedding_dim)
        word_to_tokens_mapping: Dictionary mapping spaCy spans to lists of SBERT token indices
        text_embedding: Full text embedding as numpy array of shape (embedding_dim,)
    """

    token_embeddings: np.ndarray
    word_to_tokens_mapping: Dict[Span, List[int]]
    text_embedding: np.ndarray


def pool_embeddings(
    embeddings: List[np.ndarray],
    pooling_method: str = "mean",
    weights: Optional[List[float]] = None,
    attention_mask: Optional[List[bool]] = None,
) -> np.ndarray:
    """Apply pooling to a list of embedding vectors.

    This function combines multiple embedding vectors using the specified pooling strategy.
    Supports various pooling methods commonly used in transformer-based models.

    Args:
        embeddings: List of embedding vectors, each as numpy array of shape (embedding_dim,)
        pooling_method: Pooling strategy to use. Options: "mean", "max", "sum", "weighted_mean", "min"
        weights: Optional weights for weighted mean pooling. Must have same length as embeddings
        attention_mask: Optional mask to exclude certain embeddings (False = exclude, True = include)

    Returns:
        Combined embedding vector as numpy array of shape (embedding_dim,)

    Raises:
        ValueError: If embeddings list is empty, pooling method is unsupported, or weights/mask dimensions don't match

    Examples:
        >>> embeddings = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        >>> pool_embeddings(embeddings, "mean")
        array([4., 5., 6.])
        >>> pool_embeddings(embeddings, "max")
        array([7., 8., 9.])
    """
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty")

    # Convert to numpy array for easier manipulation
    try:
        embeddings_array = np.array(embeddings)
    except Exception as e:
        print(embeddings)
        raise

    # Validate dimensions
    if len(embeddings_array.shape) != 2:
        raise ValueError("All embeddings must have the same dimensionality")

    n_embeddings, embedding_dim = embeddings_array.shape

    # Apply attention mask if provided
    if attention_mask is not None:
        if len(attention_mask) != n_embeddings:
            raise ValueError(
                f"Attention mask length ({len(attention_mask)}) must match "
                f"number of embeddings ({n_embeddings})"
            )

        mask_array = np.array(attention_mask)
        if not np.any(mask_array):
            raise ValueError("Attention mask cannot exclude all embeddings")

        embeddings_array = embeddings_array[mask_array]
        n_embeddings = embeddings_array.shape[0]

        # Update weights if provided
        if weights is not None:
            weights = [w for i, w in enumerate(weights) if attention_mask[i]]

    # Validate weights for weighted mean pooling
    if pooling_method == "weighted_mean":
        if weights is None:
            raise ValueError("Weights must be provided for weighted_mean pooling")
        if len(weights) != n_embeddings:
            raise ValueError(
                f"Weights length ({len(weights)}) must match "
                f"number of embeddings ({n_embeddings})"
            )
        weights_array = np.array(weights)
        if np.sum(weights_array) == 0:
            raise ValueError("Sum of weights cannot be zero")

    # Apply pooling strategy
    if pooling_method == "mean":
        result = np.mean(embeddings_array, axis=0)

    elif pooling_method == "max":
        result = np.max(embeddings_array, axis=0)

    elif pooling_method == "min":
        result = np.min(embeddings_array, axis=0)

    elif pooling_method == "sum":
        result = np.sum(embeddings_array, axis=0)

    elif pooling_method == "weighted_mean":
        weights_normalized = weights_array / np.sum(weights_array)
        result = np.average(embeddings_array, axis=0, weights=weights_normalized)

    else:
        raise ValueError(
            f"Unsupported pooling method: '{pooling_method}'. "
            f"Supported methods: 'mean', 'max', 'min', 'sum', 'weighted_mean'"
        )

    logger.debug(
        f"Applied {pooling_method} pooling to {n_embeddings} embeddings, "
        f"output shape: {result.shape}"
    )

    return result


def get_word_embeddings(
    result: TokenEmbeddingsResult, pooling_method: str = "mean"
) -> Dict[Span, np.ndarray]:
    """Extract word-level embeddings by pooling token embeddings for each spaCy word.

    This convenience function pools the SBERT token embeddings that correspond to each
    spaCy word span, providing a single embedding vector per word.

    Args:
        result: TokenEmbeddingsResult from extract_token_embeddings()
        pooling_method: Pooling strategy to apply to tokens within each word

    Returns:
        Dictionary mapping spaCy word spans to their pooled embedding vectors

    Example:
        >>> doc = nlp("Hello world")
        >>> result = extract_token_embeddings(doc)
        >>> word_embeddings = get_word_embeddings(result, "mean")
        >>> print(word_embeddings[doc[0:1]].shape)  # "Hello" word embedding
    """
    word_embeddings = {}

    for span, token_indices in result.word_to_tokens_mapping.items():
        # Get embeddings for all tokens that make up this word
        span_token_embeddings = [result.token_embeddings[i] for i in token_indices]

        # Pool them to get a single word embedding
        word_embedding = pool_embeddings(span_token_embeddings, pooling_method)
        word_embeddings[span] = word_embedding

    logger.debug(
        f"Generated embeddings for {len(word_embeddings)} words using {pooling_method} pooling"
    )

    return word_embeddings


def extract_token_embeddings(
    doc: Doc, model_name: str = "all-MiniLM-L6-v2"
) -> TokenEmbeddingsResult:
    """Extract token-level embeddings from a spaCy document using SBERT.

    This function processes a spaCy document to extract embeddings at multiple levels:
    - Individual token embeddings from the SBERT model
    - Mapping between spaCy word spans and SBERT tokens
    - Overall text embedding

    Uses the official SentenceTransformer API with output_value="token_embeddings"
    for reliable token-level embedding extraction.

    Args:
        doc: spaCy document containing the tokenized text
        model_name: Name of the sentence-transformers model to use

    Returns:
        TokenEmbeddingsResult containing token embeddings, word-to-token mapping,
        and text embedding

    Raises:
        ValueError: If the document is empty or model fails to load

    Note:
        The SentenceTransformer.encode() method supports other output_value options:
        - "sentence_embedding" (default): Full sentence embedding
        - "token_embeddings": Token-level embeddings
        - Both can be combined with convert_to_numpy=True for numpy arrays
    """
    if len(doc) == 0:
        raise ValueError("Input document is empty")

    logger.info(
        f"Processing document with {len(doc)} tokens using model '{model_name}'"
    )

    # Load the sentence transformer model
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise ValueError(f"Could not load sentence transformer model: {e}")

    text = doc.text

    # Get token embeddings using the official SentenceTransformer API
    # This is the recommended approach as it handles all internal processing automatically
    token_embeddings = model.encode(
        text,
        output_value="token_embeddings",
        convert_to_numpy=True,
        device="cpu",
        normalize_embeddings=False,
    )

    # Get text embedding using the model's encode method
    # Note: SBERT uses mean pooling by default to aggregate token embeddings into sentence embeddings
    text_embedding = model.encode(
        text, convert_to_numpy=True, device="cpu", normalize_embeddings=False
    )

    # Get tokenizer and create offset mapping for word-to-token alignment
    tokenizer = model.tokenizer
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
    )

    # Create mapping between spaCy words and SBERT tokens
    word_to_tokens_mapping = _create_word_to_tokens_mapping(
        doc, encoded["offset_mapping"].squeeze(0).tolist(), tokenizer
    )

    logger.info(
        f"Extracted embeddings using official API: {token_embeddings.shape[0]} tokens, "
        f"embedding dimension: {token_embeddings.shape[1]}"
    )

    return TokenEmbeddingsResult(
        token_embeddings=token_embeddings,
        word_to_tokens_mapping=word_to_tokens_mapping,
        text_embedding=text_embedding,
    )


def _create_word_to_tokens_mapping(
    doc: Doc, offset_mapping: List[Tuple[int, int]], tokenizer
) -> Dict[Span, List[int]]:
    """Create mapping between spaCy word spans and SBERT token indices.

    This function aligns spaCy's tokenization with SBERT's subword tokenization
    by finding overlapping character spans between the two tokenization approaches.

    Args:
        doc: spaCy document
        offset_mapping: Character offset mapping from SBERT tokenizer
        tokenizer: SBERT tokenizer instance

    Returns:
        Dictionary mapping spaCy spans to lists of SBERT token indices
    """
    word_to_tokens = {}

    for token in doc:
        # Skip whitespace tokens that might not align well
        if token.is_space:
            continue

        # Find overlapping SBERT tokens for this spaCy token
        token_start = token.idx
        token_end = token.idx + len(token.text)

        overlapping_tokens = []
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            # Skip special tokens and padding (they have offset (0, 0))
            if start_char == 0 and end_char == 0:
                continue

            # Check for character-level overlap between spaCy token and SBERT token
            # Use intersection over union approach for better alignment
            overlap_start = max(start_char, token_start)
            overlap_end = min(end_char, token_end)

            if overlap_start < overlap_end:  # There is an overlap
                overlapping_tokens.append(token_idx)

        if overlapping_tokens:
            # Create span for the spaCy token
            span = doc[token.i : token.i + 1]
            word_to_tokens[span] = overlapping_tokens
        else:
            # Log cases where no alignment is found
            logger.warning(
                f"No SBERT token alignment found for spaCy token: '{token.text}'"
            )

    return word_to_tokens


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process sample text
    text = "Peculiarish sample sentence for testing token embeddings."
    doc = nlp(text)

    # Extract embeddings
    result = extract_token_embeddings(doc)

    print(f"Token embeddings shape: {result.token_embeddings.shape}")
    print(f"Text embedding shape: {result.text_embedding.shape}")
    print(f"Word to tokens mapping: {len(result.word_to_tokens_mapping)} words")

    # Print some mappings
    for span, token_indices in list(result.word_to_tokens_mapping.items()):
        print(f"'{span.text}' -> SBERT tokens {token_indices}")

    # Demonstrate pooling function usage
    print("\n--- Pooling Function Examples ---")

    # Example 1: Get word-level embeddings using the helper function
    word_embeddings_dict = get_word_embeddings(result, "mean")
    print(f"Generated word embeddings for {len(word_embeddings_dict)} words")

    for span, embedding in list(word_embeddings_dict.items())[:3]:
        print(f"Word '{span.text}' embedding shape: {embedding.shape}")

    # Example 2: Manual pooling of specific words
    word_embeddings = []
    word_names = []
    for span, token_indices in list(result.word_to_tokens_mapping.items())[:3]:
        # Get embeddings for tokens that make up this word
        span_token_embeddings = [result.token_embeddings[i] for i in token_indices]
        # Pool them to get a single word embedding
        word_embedding = pool_embeddings(span_token_embeddings, "mean")
        word_embeddings.append(word_embedding)
        word_names.append(span.text)

    # Example 3: Pool word embeddings using different strategies
    if len(word_embeddings) >= 2:
        print(f"\nPooling {len(word_embeddings)} word embeddings ({word_names}):")

        # Mean pooling
        mean_pooled = pool_embeddings(word_embeddings, "mean")
        print(f"Mean pooled shape: {mean_pooled.shape}")

        # Max pooling
        max_pooled = pool_embeddings(word_embeddings, "max")
        print(f"Max pooled shape: {max_pooled.shape}")

        # Weighted mean pooling (give more weight to first word)
        weights = [0.5, 0.3, 0.2][: len(word_embeddings)]
        weighted_pooled = pool_embeddings(
            word_embeddings, "weighted_mean", weights=weights
        )
        print(f"Weighted mean pooled shape: {weighted_pooled.shape}")

        # With attention mask (exclude last word)
        attention_mask = [True] * (len(word_embeddings) - 1) + [False]
        masked_pooled = pool_embeddings(
            word_embeddings, "mean", attention_mask=attention_mask
        )
        print(f"Masked pooled shape: {masked_pooled.shape}")

    # Example 3: Compare different approaches for getting token embeddings
    print(f"\nToken embeddings extracted using official API:")
    print(f"Shape: {result.token_embeddings.shape}")
    print(f"First token embedding (first 5 dims): {result.token_embeddings[0][:5]}")

    # Example 4: Compare with SBERT's original text embedding
    all_token_embeddings = [
        result.token_embeddings[i] for i in range(len(result.token_embeddings))
    ]
    manual_mean_pooled = pool_embeddings(all_token_embeddings, "mean")

    print(result.text_embedding[:10])
    print(manual_mean_pooled[:10])

    normalized_embeddings = manual_mean_pooled / np.linalg.norm(manual_mean_pooled)
    print(normalized_embeddings[:10])

    print(f"\nComparison with SBERT text embedding:")
    print(f"SBERT text embedding shape: {result.text_embedding.shape}")
    print(f"Manual mean pooled shape: {manual_mean_pooled.shape}")
    print(
        f"Embeddings are similar: {np.allclose(result.text_embedding, normalized_embeddings, atol=1e-3)}"
    )

    print("\nCosine similarity between SBERT text embedding and manual mean pooled:")
    print(cos_sim(result.text_embedding, normalized_embeddings).item())
    print(cos_sim(result.text_embedding, manual_mean_pooled).item())
