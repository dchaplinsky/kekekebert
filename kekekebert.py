"""
Token-level embeddings extraction using SBERT and spaCy.

This module provides functionality to extract token-level embeddings from text
using sentence-transformers (SBERT) while maintaining alignment with spaCy
tokenization. It also includes utilities for pooling embeddings using various
strategies (mean, max, weighted mean, etc.).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from sentence_transformers.util import cos_sim

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, Span
import jinja2

# Load Jinja2 template
template_loader = jinja2.FileSystemLoader("templates")
template_env = jinja2.Environment(loader=template_loader)

logger = logging.getLogger(__name__)


@dataclass
class TokenEmbeddingsResult:
    """Result container for token-level embedding extraction.

    Attributes:
        token_embeddings: Embeddings for each SBERT token as numpy array of shape (n_tokens, embedding_dim)
        word_to_tokens_mapping: Dictionary mapping spaCy spans to lists of SBERT token indices
        token_to_sentence_mapping: Dictionary mapping SBERT token indices to spaCy sentence indices
        text_embedding: Full text embedding as numpy array of shape (embedding_dim,)
        tokens_ids: List of SBERT token IDs corresponding to the input text
        tokens: List of SBERT tokens corresponding to the input text, including subwords
    """

    token_embeddings: np.ndarray
    word_to_tokens_mapping: Dict[Span, List[int]]
    token_to_sentence_mapping: Dict[int, int]
    text_embedding: np.ndarray
    tokens_ids: List[int]
    tokens: List[str]


@dataclass
class TokenScoreReport:
    """Container for token for report.

    Attributes:
        token: The token string
        background_color: CSS color for the token background
        text_color: CSS color for the token text
        tooltip: Tooltip text to show on hover
        css_classes: Additional CSS classes for styling
    """

    token: str
    background_color: str
    text_color: str
    tooltip: str
    css_classes: str


@dataclass
class ScoreReport:
    """Container for score report.

    Attributes:
        grouped_tokens: List of lists of TokenScoreReport objects, each representing a group of tokens
        min_score: Minimum score across all tokens in the report
        max_score: Maximum score across all tokens in the report
    """

    grouped_tokens: List[List[TokenScoreReport] | str]  # List of token groups or strings (for spaces)
    min_score: float
    max_score: float


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
    doc: Doc,
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
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

    # Create mapping between SBERT tokens and sentences
    token_to_sentence_mapping = _create_token_to_sentence_mapping(
        doc, encoded["offset_mapping"].squeeze(0).tolist()
    )

    logger.info(
        f"Extracted embeddings using official API: {token_embeddings.shape[0]} tokens, "
        f"embedding dimension: {token_embeddings.shape[1]}"
    )

    return TokenEmbeddingsResult(
        token_embeddings=token_embeddings,
        word_to_tokens_mapping=word_to_tokens_mapping,
        token_to_sentence_mapping=token_to_sentence_mapping,
        text_embedding=text_embedding,
        tokens_ids=encoded["input_ids"].squeeze(0).tolist(),
        tokens=tokenizer.convert_ids_to_tokens(
            encoded["input_ids"].squeeze(0).tolist()
        ),
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

    logger.debug(f"Created word-to-tokens mapping for {len(word_to_tokens)} words")
    return word_to_tokens


def _create_token_to_sentence_mapping(
    doc: Doc, offset_mapping: List[Tuple[int, int]]
) -> Dict[int, int]:
    """Create mapping between SBERT token indices and spaCy sentence indices.

    This function maps each SBERT token to the sentence it belongs to,
    using character offset information to determine sentence boundaries.

    Args:
        doc: spaCy document with sentence segmentation
        offset_mapping: Character offset mapping from SBERT tokenizer

    Returns:
        Dictionary mapping SBERT token indices to spaCy sentence indices
    """
    token_to_sentence = {}

    # Create sentence boundaries based on character positions
    sentence_boundaries = []
    for sent_idx, sent in enumerate(doc.sents):
        start_char = sent.start_char
        end_char = sent.end_char
        sentence_boundaries.append((sent_idx, start_char, end_char))

    # Map each SBERT token to its sentence
    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
        # Skip special tokens and padding (they have offset (0, 0))
        if start_char == 0 and end_char == 0:
            continue

        # Find which sentence this token belongs to
        token_center = (start_char + end_char) / 2

        for sent_idx, sent_start, sent_end in sentence_boundaries:
            if sent_start <= token_center < sent_end:
                token_to_sentence[token_idx] = sent_idx
                break
        else:
            # If no sentence found, assign to the last sentence
            if sentence_boundaries:
                token_to_sentence[token_idx] = sentence_boundaries[-1][0]

    return token_to_sentence


def render_tokens_html(
    score_groups: List[List[Union[Dict[str, Union[str, float]], Tuple[str, float]]]],
    title: str = "Token Visualization",
    colormap: str = "red",
    show_scores: bool = True,
    dialect: str = "mpnet",
) -> str:
    """Render tokenized text as HTML with color-coded heatmap based on scores.

    Creates an HTML visualization where each token is color-coded according to its score,
    with higher scores showing more intense colors. Tokens belonging to the same word
    are grouped closer together, and subword tokens (##) are displayed without the prefix.

    Args:
        score_groups: List of lists of token/score pairs. Each item can be:
            - Dict with 'token' and 'score' keys: {'token': 'hello', 'score': 0.8}
            - Tuple: ('hello', 0.8)
        title: HTML page title
        colormap: Color scheme to use. Options: 'red', 'blue', 'green', 'purple', 'orange'
        show_scores: Whether to show numeric scores as tooltips
        dialect: Dialect of the model used for tokenization (default is 'mpnet',
            'minilm' is also supported)

    Returns:
        Complete HTML string with embedded CSS styling

    Raises:
        ValueError: If token_scores is empty or contains invalid format

    Examples:
        >>> tokens = [('Hello', 0.8), ('##world', 0.3), ('!', 0.1)]
        >>> html = render_tokens_html([tokens], title="Attention Weights")
        >>> with open('output.html', 'w') as f:
        ...     f.write(html)
    """
    if dialect not in ["mpnet", "minilm"]:
        raise ValueError(
            f"Unsupported dialect: {dialect}. Supported: 'mpnet', 'minilm'"
        )
    if not score_groups or not all(score_groups):
        raise ValueError("score_groups must be a non-empty list of token/score pairs")

    # Color schemes with good readability
    color_schemes = {
        "red": "rgba(220, 53, 69, {alpha})",  # Bootstrap red
        "blue": "rgba(13, 110, 253, {alpha})",  # Bootstrap blue
        "green": "rgba(25, 135, 84, {alpha})",  # Bootstrap green
        "purple": "rgba(111, 66, 193, {alpha})",  # Bootstrap purple
        "orange": "rgba(253, 126, 20, {alpha})",  # Bootstrap orange
    }

    if colormap not in color_schemes:
        raise ValueError(
            f"Unsupported colormap: {colormap}. Available: {list(color_schemes.keys())}"
        )

    color_template = color_schemes[colormap]
    template = template_env.get_template("report.jinja")
    score_reports = []

    for token_scores in score_groups:
        score_report = ScoreReport(grouped_tokens=[], min_score=0, max_score=1.0)
        # Normalize and process tokens
        processed_tokens = []
        scores = []

        for item in token_scores:
            if isinstance(item, dict):
                if "token" not in item or "score" not in item:
                    raise ValueError("Dict items must contain 'token' and 'score' keys")
                token, score = item["token"], item["score"]
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                token, score = item
            else:
                raise ValueError(
                    "Each item must be a dict with 'token'/'score' keys or a (token, score) tuple"
                )

            # Ensure score is in 0-1 range
            score = max(0.0, min(1.0, float(score)))
            scores.append(score)
            processed_tokens.append(str(token))

        # Generate HTML
        min_score = min(scores)
        max_score = max(scores)
        score_report.min_score = min_score
        score_report.max_score = max_score

        # Group tokens by words and add them with proper spacing
        i = 0
        while i < len(processed_tokens):
            # Start a word group
            current_group = []

            # Process tokens in current word
            word_token_indices = [i]

            # Look ahead for subword tokens (starting with ##)
            if dialect == "minilm":
                j = i + 1
                while j < len(processed_tokens) and processed_tokens[j].startswith(
                    "##"
                ):
                    word_token_indices.append(j)
                    j += 1
            else:  # mpnet or other dialects
                j = i + 1
                while j < len(processed_tokens) and not processed_tokens[j].startswith(
                    "▁"
                ):
                    word_token_indices.append(j)
                    j += 1

            # Add all tokens in this word group
            for idx_in_word, token_idx in enumerate(word_token_indices):
                token = processed_tokens[token_idx]
                score = scores[token_idx]

                # Remove ## prefix for display
                if dialect == "minilm":
                    display_token = token[2:] if token.startswith("##") else token
                else:
                    display_token = token[1:] if token.startswith("▁") else token

                # Calculate alpha based on score (minimum 0.1 for readability, maximum 0.9)
                alpha = (
                    0.1 + ((score - min_score) / (max_score - min_score)) * 0.8
                    if max_score > min_score
                    else 0.5
                )
                background_color = color_template.format(alpha=alpha)

                # Determine text color for readability
                text_color = "#000" if alpha < 0.5 else "#fff"

                # Create tooltip text
                tooltip = (
                    f"Score: {score:.3f}" if show_scores else f"Token {token_idx+1}"
                )

                # Determine CSS classes
                css_classes = ["token"]
                if token.startswith("##"):
                    css_classes.append("subword")
                    if idx_in_word == 1:  # First subword token
                        css_classes.append("first-subword")

                current_group.append(
                    TokenScoreReport(
                        token=display_token,
                        background_color=background_color,
                        text_color=text_color,
                        tooltip=tooltip,
                        css_classes=" ".join(css_classes),
                    )
                )

            # Close word group
            score_report.grouped_tokens.append(current_group)

            # Add space after word group (but not after punctuation)
            if j < len(processed_tokens):
                next_token = processed_tokens[j]
                if next_token not in ".,!?;:)]}":
                    score_report.grouped_tokens.append(" ")

            # Move to next word
            i = j

        score_reports.append(score_report)
    # Add legend and closing HTML

    return template.render(
        title=title,
        score_reports=score_reports,
        color_template=color_template,
        show_scores=show_scores,
        dialect=dialect,
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process sample text with multiple sentences
    text = """Ontologies of Time: Review and Trends.
    Time, as a phenomenon, has been in the focus of scientific thought from ancient times.
    It continues to be an important subject of research in many disciplines due to its importance as a basic aspect for understanding and formally representing change."""
    doc = nlp(text)

    # Extract embeddings
    # result = extract_token_embeddings(doc, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    result = extract_token_embeddings(doc, model_name="all-MiniLM-L6-v2")

    similarity_scores = []

    for token_emb in result.token_embeddings:
        # Calculate cosine similarity
        similarity = cos_sim(token_emb, result.text_embedding).item()
        # # Normalize to 0-1 range
        # similarity_normalized = (similarity + 1) / 2
        similarity_scores.append(similarity)

    embedding_token_scores = [
        (
            result.tokens[i] if i < len(result.tokens) else f"token_{i}",
            similarity_scores[i],
        )
        for i in range(len(similarity_scores))
    ]

    html_embedding_viz = render_tokens_html(
        [embedding_token_scores],
        title="Token Embedding Similarity to Mean (Grouped by Words)",
        colormap="red",
        show_scores=True,
        dialect="minilm",  # Use 'minilm' for better subword handling
    )

    with open(
        "html_results/embedding_similarity_grouped.html", "w", encoding="utf-8"
    ) as f:
        f.write(html_embedding_viz)
    print(
        "Embedding similarity visualization saved to 'html_results/embedding_similarity_grouped.html'"
    )
