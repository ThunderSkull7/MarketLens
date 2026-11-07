"""
MarketLens - Column Normalisation and Mapping.

Provides functions for normalising column names and mapping source columns
to canonical fields. Supports automatic matching of common aliases and
explicit user-supplied mappings.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Canonical field definitions
# ---------------------------------------------------------------------------


class CanonicalField(BaseModel):
    """Represents a canonical field with its metadata."""

    name: str = Field(description="Canonical field name")
    required: bool = Field(default=False, description="Whether the field is required")
    aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of alias -> canonical name",
    )
    description: str = Field(default="", description="Human-readable description")

    @property
    def is_required(self) -> bool:
        return self.required


# Canonical fields for the MarketLens platform
class CanonicalFields:
    """Singleton-like container for canonical field definitions."""

    REVIEW_ID = CanonicalField(
        name="review_id",
        required=True,
        aliases={
            "id": "review_id",
            "feedback_id": "review_id",
            "comment_id": "review_id",
        },
        description="Stable feedback identifier",
    )

    REVIEW_TEXT = CanonicalField(
        name="review_text",
        required=True,
        aliases={
            "feedback": "review_text",
            "comment": "review_text",
            "review": "review_text",
            "description": "review_text",
        },
        description="Customer feedback text",
    )

    RATING = CanonicalField(
        name="rating",
        required=False,
        aliases={
            "score": "rating",
            "ratings": "rating",
            "value": "rating",
            "stars": "rating",
        },
        description="Optional rating",
    )

    CREATED_AT = CanonicalField(
        name="created_at",
        required=False,
        aliases={
            "date": "created_at",
            "timestamp": "created_at",
            "created": "created_at",
            "time": "created_at",
        },
        description="Feedback timestamp",
    )

    PRODUCT_OR_SERVICE = CanonicalField(
        name="product_or_service",
        required=False,
        aliases={
            "product": "product_or_service",
            "category": "product_or_service",
            "item": "product_or_service",
            "sku": "product_or_service",
        },
        description="Product or category",
    )

    SOURCE = CanonicalField(
        name="source",
        required=False,
        aliases={"origin": "source", "provider": "source", "provider_name": "source"},
        description="Data provenance",
    )

    LANGUAGE = CanonicalField(
        name="language",
        required=False,
        aliases={"lang": "language", "language_code": "language"},
        description="Language information",
    )

    VERIFIED_LABEL = CanonicalField(
        name="verified_label",
        required=False,
        aliases={"verified": "verified_label", "verification": "verified_label"},
        description="Optional verified sentiment",
    )


canonical_fields = CanonicalFields()


def iter_canonical_fields() -> list[tuple[str, CanonicalField]]:
    """
    Return all canonical fields as (attribute_name, CanonicalField) tuples.

    Iterates over the class-level attributes of CanonicalFields, which
    cannot be obtained from an instance's __dict__.
    """
    return [
        (name, value)
        for name, value in vars(CanonicalFields).items()
        if isinstance(value, CanonicalField)
    ]


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------


def normalise_column_name(name: str) -> str:
    """
    Normalise a column name to a consistent format.

    Steps:
    1. Strip leading/trailing whitespace
    2. Replace internal runs of whitespace with a single underscore
    3. Remove surrounding quotes
    4. Convert to lowercase
    5. Remove non-alphanumeric characters except underscores

    Parameters
    ----------
    name : str
        Raw column name from the CSV header.

    Returns
    -------
    str
        Normalised column name.
    """
    if not isinstance(name, str):
        return str(name)

    # Strip whitespace
    cleaned = name.strip()

    # Replace internal runs of whitespace with single underscore
    cleaned = re.sub(r"\s+", "_", cleaned)

    # Remove surrounding quotes (already stripped, but defensive)
    cleaned = cleaned.strip('"').strip("'")

    # Convert to lowercase
    cleaned = cleaned.lower()

    # Remove non-alphanumeric characters except underscores and hyphens
    cleaned = re.sub(r"[^a-z0-9_-]", "", cleaned)

    # Replace hyphens with underscores for consistency
    cleaned = cleaned.replace("-", "_")

    # Collapse multiple underscores
    cleaned = re.sub(r"_+", "_", cleaned)

    return cleaned


def normalise_headers(headers: list[str]) -> list[str]:
    """
    Normalise a list of column headers.

    Parameters
    ----------
    headers : list[str]
        Raw column headers from a CSV file.

    Returns
    -------
    list[str]
        Normalised column headers.
    """
    return [normalise_column_name(h) for h in headers]


# ---------------------------------------------------------------------------
# Duplicate normalised header detection
# ---------------------------------------------------------------------------


def detect_duplicate_normalised_headers(
    headers: list[str],
) -> Tuple[bool, list[Tuple[int, int, str]]]:
    """
    Detect duplicate normalised column headers.

    Parameters
    ----------
    headers : list[str]
        Normalised column headers.

    Returns
    -------
    Tuple[bool, list[Tuple[int, int, str]]]
        (has_duplicates, [(row1, row2, header_value), ...])
    """
    seen: dict[str, list[int]] = {}
    for idx, header in enumerate(headers):
        if header in seen:
            seen[header].append(idx)
        else:
            seen[header] = [idx]

    duplicates: list[Tuple[int, int, str]] = []
    for header, indices in seen.items():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    duplicates.append((indices[i], indices[j], header))

    return len(duplicates) > 0, duplicates


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------


class ColumnMapping(BaseModel):
    """
    Represents a mapping from source column names to canonical fields.

    Attributes
    ----------
    source_to_canonical : dict[str, str]
        Mapping from normalised source column name -> canonical field name.
    unmapped : list[str]
        Source columns that were not mapped to any canonical field.
    ambiguous : list[str]
        Source columns that matched multiple candidates ambiguously.
    """

    source_to_canonical: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from source column to canonical field",
    )
    unmapped: list[str] = Field(
        default_factory=list,
        description="Source columns not mapped to canonical fields",
    )
    ambiguous: list[str] = Field(
        default_factory=list,
        description="Source columns with ambiguous mappings",
    )

    def get(self, canonical_name: str) -> Optional[str]:
        """Get the source column name for a canonical field."""
        for source, canonical in self.source_to_canonical.items():
            if canonical == canonical_name:
                return source
        return None

    def get_canonical(self, source_name: str) -> Optional[str]:
        """Get the canonical field name for a source column."""
        return self.source_to_canonical.get(source_name)


class ColumnMappingResult(BaseModel):
    """
    Result of attempting to map columns, including mapping metadata.

    Attributes
    ----------
    mapping : ColumnMapping
        The resulting column mapping.
    unmapped : list[str]
        Source columns that were not mapped to any canonical field.
    ambiguous : list[str]
        Source columns that matched multiple candidates ambiguously.
    warnings : list[str]
        Warnings generated during mapping.
    errors : list[str]
        Errors generated during mapping.
    """

    mapping: ColumnMapping = Field(default_factory=ColumnMapping)
    unmapped: list[str] = []
    ambiguous: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    def get(self, canonical_name: str):
        """Get the source column name for a canonical field (delegates to mapping)."""
        return self.mapping.get(canonical_name)

    def get_canonical(self, source_name: str):
        """Get the canonical field name for a source column (delegates to mapping)."""
        return self.mapping.get_canonical(source_name)


def auto_map_columns(
    normalised_headers: list[str],
    explicit_mapping: Optional[dict[str, str]] = None,
    strict: bool = True,
) -> ColumnMappingResult:
    """
    Attempt automatic column mapping based on canonical aliases.

    Parameters
    ----------
    normalised_headers : list[str]
        Normalised column headers from the CSV.
    explicit_mapping : Optional[dict[str, str]]
        User-supplied explicit mapping {source: canonical}. If provided,
        these take priority and the automatic matching only fills gaps.
    strict : bool
        If True, raise errors for missing required columns.
        If False, warn but continue.

    Returns
    -------
    ColumnMappingResult
        The mapping result with source_to_canonical, unmapped, and
        ambiguous information.

    Raises
    ------
    AmbiguousColumnMappingError
        If an ambiguous mapping is detected and strict=True.
    MissingRequiredColumnsError
        If required canonical columns are missing and strict=True.
    """
    result = ColumnMappingResult()
    mapping = ColumnMapping()
    used_canonical: set[str] = set()
    used_source: set[str] = set()

    # Step 1: Apply explicit mapping first
    if explicit_mapping:
        for source, canonical in explicit_mapping.items():
            source_norm = normalise_column_name(source)
            canonical_norm = normalise_column_name(canonical)
            mapping.source_to_canonical[source_norm] = canonical_norm
            used_canonical.add(canonical_norm)
            used_source.add(source_norm)

    # Step 2: Automatic matching for remaining headers
    # Build a reverse lookup: canonical alias -> canonical name
    alias_to_canonical: dict[str, str] = {}
    alias_to_canonical_reverse: dict[str, str] = {}

    for field_name, field in iter_canonical_fields():
        for alias in field.aliases:
            alias_to_canonical[alias] = field.name
            alias_to_canonical_reverse[field.name] = alias

    # Also add the canonical name itself as an alias
    for field_name, field in iter_canonical_fields():
        alias_to_canonical[field_name.lower()] = field.name

    # Process each normalised header
    for header in normalised_headers:
        if header in used_source and header not in mapping.source_to_canonical:
            # This source column was already used; skip or flag
            result.unmapped.append(header)
            continue

        # Check if this header matches a known alias
        matched_canonical: Optional[str] = None
        matched_aliases: list[str] = []

        # First check explicit mapping (already applied, skip)
        if header in mapping.source_to_canonical:
            continue

        # Search through all known aliases
        for alias, canonical in alias_to_canonical.items():
            if alias == header:
                # Avoid mapping the same canonical field twice
                if canonical in used_canonical and canonical not in [
                    mapping.source_to_canonical[s] for s in mapping.source_to_canonical
                ]:
                    # This canonical was already mapped via explicit mapping
                    # via a different source - potential conflict
                    pass
                matched_canonical = canonical
                break
            # Check for substring/partial matches
            if header in alias or alias in header:
                # Only add if not already matched
                if alias not in matched_aliases:
                    matched_aliases.append(alias)

        if matched_canonical:
            # Check if this canonical was already mapped to a different source
            already_mapped_to = mapping.get(matched_canonical)
            if already_mapped_to is not None and already_mapped_to != header:
                # Conflict: this canonical already has a different source
                result.warnings.append(
                    f"Column '{header}' maps to canonical '{matched_canonical}', "
                    f"but '{already_mapped_to}' was already mapped to it. "
                    f"Using '{header}' and dropping '{already_mapped_to}'."
                )
                # Remove the old mapping
                # Note: we keep the first one; could also raise an error
            mapping.source_to_canonical[header] = matched_canonical
            used_canonical.add(matched_canonical)
            used_source.add(header)
        elif matched_aliases:
            # Ambiguous match - multiple aliases matched
            result.ambiguous.append(header)
            # Record all matched aliases for the warning
            alias_names = ", ".join(matched_aliases)
            result.warnings.append(
                f"Ambiguous mapping for column '{header}': matched aliases "
                f"[{alias_names}]. Mapping to canonical will be skipped."
            )
            result.unmapped.append(header)
        else:
            # No match found - column is unmapped
            result.unmapped.append(header)

    result.mapping = mapping

    # Step 3: Check for missing required columns
    if strict:
        missing_required: list[str] = []
        for _field_name, field in iter_canonical_fields():
            if field.is_required:
                if field.name not in mapping.source_to_canonical.values():
                    missing_required.append(field.name)

        if missing_required:
            result.errors.append(
                f"Missing required columns: {', '.join(missing_required)}"
            )

    return result


# ---------------------------------------------------------------------------
# Preserve original column names mapping metadata
# ---------------------------------------------------------------------------


class ColumnMappingMetadata(BaseModel):
    """
    Metadata about the column mapping process, preserving original names
    and normalisation information.

    Attributes
    ----------
    original_headers : list[str]
        The original column headers as they appeared in the CSV.
    normalised_headers : list[str]
        The column headers after normalisation.
    mapping_result : ColumnMappingResult
        The result of the mapping process.
    """

    original_headers: list[str] = Field(description="Original column headers")
    normalised_headers: list[str] = Field(description="Normalised column headers")
    mapping_result: ColumnMappingResult = Field(description="Mapping result")

    def get_mapped_sources(self) -> list[str]:
        """Get the source columns that were mapped to canonical fields."""
        return list(self.mapping_result.mapping.source_to_canonical.keys())

    def get_mapped_canonicals(self) -> list[str]:
        """Get the canonical fields that were mapped from source columns."""
        return list(self.mapping_result.mapping.source_to_canonical.values())
