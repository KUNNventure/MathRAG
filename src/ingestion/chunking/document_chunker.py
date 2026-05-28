"""Document chunking module - adapts libs.splitter for business layer.

This module serves as the adapter layer between libs.splitter (pure text splitting)
and Ingestion Pipeline (business object transformation). It transforms Document
objects into Chunk objects with proper ID generation, metadata inheritance, and
traceability.

Core Value-Add (vs libs.splitter):
1. Chunk ID Generation: Deterministic and unique IDs for each chunk
2. Metadata Inheritance: Propagates Document metadata to all chunks
3. chunk_index: Records sequential position within document
4. source_ref: Establishes parent-child traceability
5. Type Conversion: str → Chunk object (core.types contract)

Design Principles:
- Adapter Pattern: Bridges text splitter tool with business objects
- Config-Driven: Uses SplitterFactory for configuration-based strategy selection
- Deterministic: Same Document produces same Chunk IDs on repeat splits
- Type-Safe: Enforces core.types.Chunk contract
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, List, Optional

from src.core.types import Chunk, Document
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings

# Strip `[IMAGE: …]` placeholders when measuring non-placeholder text ratio.
_IMAGE_PLACEHOLDER_RATIO_RE = re.compile(r"\[IMAGE:[^\]]+\]")


class DocumentChunker:
    """Converts Documents into Chunks with business-level enrichment.
    
    This class wraps a text splitter (from libs) and adds business logic:
    - Generates stable chunk IDs
    - Inherits and extends metadata
    - Maintains document traceability
    
    Attributes:
        _splitter: The underlying text splitter from libs layer
        _settings: Configuration settings for chunking behavior
    
    Example:
        >>> from src.core.settings import load_settings
        >>> from src.core.types import Document
        >>> settings = load_settings("config/settings.yaml")
        >>> chunker = DocumentChunker(settings)
        >>> document = Document(
        ...     id="doc_123",
        ...     text="Long document content...",
        ...     metadata={"source_path": "data/report.pdf"}
        ... )
        >>> chunks = chunker.split_document(document)
        >>> print(f"Generated {len(chunks)} chunks")
        >>> print(f"First chunk ID: {chunks[0].id}")
        >>> print(f"First chunk index: {chunks[0].metadata['chunk_index']}")
    """
    
    def __init__(self, settings: Settings):
        """Initialize DocumentChunker with configuration.
        
        Args:
            settings: Configuration settings containing splitter configuration.
                     The splitter config is expected at settings.splitter.*
        
        Raises:
            ValueError: If splitter configuration is invalid or provider unknown
        """
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)
        self._merge_guard_max_chars = self._resolve_merge_guard_max_chars()
    
    def split_document(self, document: Document) -> List[Chunk]:
        """Split a Document into Chunks with full business enrichment.
        
        This is the main entry point that orchestrates the transformation:
        1. Uses underlying splitter to get text fragments
        2. Generates deterministic IDs for each chunk
        3. Inherits and extends metadata from document
        4. Creates Chunk objects conforming to core.types contract
        
        Args:
            document: Source document to split into chunks
        
        Returns:
            List of Chunk objects with:
            - Unique, deterministic IDs
            - Inherited metadata + chunk_index + source_ref
            - Proper type contract (core.types.Chunk)
        
        Raises:
            ValueError: If document has no text or invalid structure
        
        Example:
            >>> doc = Document(
            ...     id="doc_abc",
            ...     text="Section 1 content.\\n\\nSection 2 content.",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> chunker = DocumentChunker(settings)
            >>> chunks = chunker.split_document(doc)
            >>> len(chunks) >= 1
            True
            >>> chunks[0].metadata["source_path"]
            'file.pdf'
            >>> chunks[0].metadata["chunk_index"]
            0
            >>> chunks[0].metadata["source_ref"]
            'doc_abc'
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")
        
        # Step 1: Use underlying splitter to get text fragments
        splitter_kwargs = {}
        for key in ("grade", "volume"):
            if key in document.metadata and document.metadata[key] is not None:
                splitter_kwargs[key] = document.metadata[key]

        try:
            text_fragments = self._splitter.split_text(document.text, **splitter_kwargs)
        except TypeError:
            # Backward compatibility for splitters that don't accept kwargs.
            text_fragments = self._splitter.split_text(document.text)
        
        if not text_fragments:
            raise ValueError(
                f"Splitter returned no chunks for document {document.id}. "
                f"Text length: {len(document.text)}"
            )
        
        structure_meta_list: Optional[List[dict]] = None
        if hasattr(self._splitter, "get_chunk_metadata"):
            structure_meta_list = self._splitter.get_chunk_metadata()

        text_fragments, structure_meta_list = self._merge_image_only_fragments(
            text_fragments, structure_meta_list
        )
        text_fragments, structure_meta_list = self._guard_oversized_fragments(
            text_fragments, structure_meta_list
        )

        # Step 2: Transform text fragments into Chunk objects with enrichment
        chunks: List[Chunk] = []
        for index, text in enumerate(text_fragments):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)

            if structure_meta_list and index < len(structure_meta_list):
                chunk_metadata.update(structure_meta_list[index])

            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)

        # Drop chunks that are mostly IMAGE placeholders (low retrieval value).
        chunks = [
            c
            for c in chunks
            if len(_IMAGE_PLACEHOLDER_RATIO_RE.sub("", c.text).strip())
            / max(len(c.text), 1)
            > 0.2
        ]
        if not chunks:
            raise ValueError(
                f"Document {document.id}: all chunks dropped by placeholder-to-text "
                "ratio filter (>20% non-placeholder text required per chunk)."
            )
        # Renumber indices and IDs after drops (merge logic unchanged above).
        reordered: List[Chunk] = []
        for i, c in enumerate(chunks):
            meta = dict(c.metadata)
            meta["chunk_index"] = i
            reordered.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, i, c.text),
                    text=c.text,
                    metadata=meta,
                    start_offset=c.start_offset,
                    end_offset=c.end_offset,
                    source_ref=c.source_ref,
                )
            )
        return reordered

    def _merge_image_only_fragments(
        self,
        fragments: List[str],
        metadata_list: Optional[List[dict]] = None,
    ) -> tuple[List[str], Optional[List[dict]]]:
        """Attach image-only fragments to neighboring text chunks.

        Some PDF pages contain dense figure regions, and splitters may create
        standalone chunks that are only `[IMAGE: ...]` placeholders. Merging
        them into adjacent text chunks keeps multimodal context together and
        avoids long runs of image-only chunks in retrieval views.
        """
        if not fragments:
            return fragments, metadata_list

        merged: List[str] = []
        merged_metadata: Optional[List[dict]] = (
            [] if metadata_list is not None and len(metadata_list) == len(fragments) else None
        )
        pending_prefix = ""
        pending_meta: Optional[dict] = None

        for index, fragment in enumerate(fragments):
            current_meta = metadata_list[index] if merged_metadata is not None else None
            if self._is_image_only_fragment(fragment):
                pending_prefix = (
                    f"{pending_prefix}\n{fragment}".strip()
                    if pending_prefix
                    else fragment
                )
                if pending_meta is None:
                    pending_meta = current_meta
                continue

            if pending_prefix:
                fragment = f"{pending_prefix}\n{fragment}"
                pending_prefix = ""
                pending_meta = None
            merged.append(fragment)
            if merged_metadata is not None:
                merged_metadata.append(dict(current_meta or {}))

        # Trailing image-only content: append to the previous text chunk.
        if pending_prefix:
            if merged:
                merged[-1] = f"{merged[-1]}\n{pending_prefix}"
            else:
                merged.append(pending_prefix)
                if merged_metadata is not None:
                    merged_metadata.append(dict(pending_meta or {}))

        return merged, merged_metadata

    def _guard_oversized_fragments(
        self,
        fragments: List[str],
        metadata_list: Optional[List[dict]] = None,
    ) -> tuple[List[str], Optional[List[dict]]]:
        """Apply post-merge fallback split for oversized fragments.

        Image-only merge improves multimodal context, but in dense-image PDFs it
        can create very large trailing fragments. This guard re-splits only
        fragments larger than a conservative safety threshold to avoid embedding
        API input-limit failures.
        """
        if not fragments:
            return fragments, metadata_list

        has_aligned_metadata = metadata_list is not None and len(metadata_list) == len(fragments)
        guarded_fragments: List[str] = []
        guarded_metadata: Optional[List[dict]] = [] if has_aligned_metadata else None

        for index, fragment in enumerate(fragments):
            current_meta = metadata_list[index] if has_aligned_metadata else None
            if len(fragment) <= self._merge_guard_max_chars:
                guarded_fragments.append(fragment)
                if guarded_metadata is not None:
                    guarded_metadata.append(dict(current_meta or {}))
                continue

            split_parts = self._split_oversized_fragment(fragment)
            for part in split_parts:
                if not part.strip():
                    continue
                guarded_fragments.append(part)
                if guarded_metadata is not None:
                    guarded_metadata.append(dict(current_meta or {}))

        return guarded_fragments, guarded_metadata

    def _resolve_merge_guard_max_chars(self) -> int:
        """Compute max fragment size for post-merge fallback splitting."""
        ingestion_cfg = getattr(self._settings, "ingestion", None)
        configured_chunk_size = getattr(ingestion_cfg, "chunk_size", None)
        base_chunk_size = (
            configured_chunk_size
            if isinstance(configured_chunk_size, int) and configured_chunk_size > 0
            else 1000
        )
        hierarchical_large_threshold = 1500
        using_hierarchical = self._splitter.__class__.__name__.lower() == "hierarchicalsplitter"
        if using_hierarchical:
            return max(base_chunk_size, hierarchical_large_threshold)
        return base_chunk_size

    def _split_oversized_fragment(self, fragment: str) -> List[str]:
        """Split large post-merge fragment with recursive splitter fallback."""
        try:
            ingestion_cfg = getattr(self._settings, "ingestion", None)
            configured_overlap = getattr(ingestion_cfg, "chunk_overlap", 200)
            effective_overlap = (
                configured_overlap
                if isinstance(configured_overlap, int) and configured_overlap >= 0
                else 200
            )
            effective_overlap = min(effective_overlap, self._merge_guard_max_chars - 1)
            recursive_splitter = RecursiveSplitter(
                self._settings,
                chunk_size=self._merge_guard_max_chars,
                chunk_overlap=effective_overlap,
            )
            parts = recursive_splitter.split_text(fragment)
            valid_parts = [p for p in parts if p and p.strip()]
            if valid_parts:
                return valid_parts
        except Exception:
            # Fall back to deterministic slicing if recursive splitter unavailable.
            pass
        return self._hard_split(fragment, self._merge_guard_max_chars)

    @staticmethod
    def _hard_split(text: str, max_chars: int) -> List[str]:
        """Deterministically split text by length when no splitter is available."""
        if len(text) <= max_chars:
            return [text]

        parts: List[str] = []
        start = 0
        overlap = min(200, max_chars // 10)
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                newline_idx = text.rfind("\n", start + max_chars // 2, end)
                if newline_idx > start:
                    end = newline_idx
            chunk = text[start:end].strip()
            if chunk:
                parts.append(chunk)
            if end >= len(text):
                break
            next_start = max(end - overlap, start + 1)
            start = next_start
        return parts or [text]

    @staticmethod
    def _is_image_only_fragment(fragment: str) -> bool:
        """Return True when a fragment only contains image placeholders."""
        if not fragment or not fragment.strip():
            return False
        stripped = re.sub(r"\[IMAGE:\s*[^\]]+\]", "", fragment)
        stripped = re.sub(r"\[FIGURE_TEXT:[^\]]+\]", "", stripped)
        stripped = re.sub(r"\s+", "", stripped)
        return stripped == ""
    
    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate unique and deterministic chunk ID.
        
        ID format: {doc_id}_{index:04d}_{content_hash}
        - doc_id: Parent document identifier
        - index: Sequential position (zero-padded to 4 digits)
        - content_hash: First 8 chars of text SHA256 hash
        
        This ensures:
        - Uniqueness: Combination of doc_id + index + content_hash
        - Determinism: Same input always produces same ID
        - Debuggability: Human-readable structure
        
        Args:
            doc_id: Parent document ID
            index: Sequential position of chunk (0-based)
            text: Chunk text content
        
        Returns:
            Unique chunk ID string
        
        Example:
            >>> chunker._generate_chunk_id("doc_123", 0, "Hello world")
            'doc_123_0000_c0535e4b'
        """
        # Compute content hash for uniqueness
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        
        # Format: {doc_id}_{index:04d}_{hash_8chars}
        return f"{doc_id}_{index:04d}_{content_hash}"
    
    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str = "") -> dict:
        """Inherit metadata from document and add chunk-specific fields.
        
        This creates a new metadata dict containing:
        - All fields from document.metadata (copied, not referenced)
        - chunk_index: Sequential position (0-based)
        - source_ref: Reference to parent document ID
        Note: The document-level 'images' field is excluded from chunk metadata.
        
        Args:
            document: Source document whose metadata to inherit
            chunk_index: Sequential position of this chunk
            chunk_text: The text content of this chunk (used to extract image_refs)
        
        Returns:
            Metadata dict with inherited and chunk-specific fields
        
        Example:
            >>> doc = Document(
            ...     id="doc_123",
            ...     text="Content",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> metadata = chunker._inherit_metadata(doc, 2, "See [IMAGE: img_001]")
            >>> metadata["source_path"]
            'file.pdf'
            >>> metadata["chunk_index"]
            2
            >>> metadata["source_ref"]
            'doc_123'
            >>> metadata["image_refs"]
            ['img_001']
        """
        # Copy all document metadata (shallow copy is sufficient for primitives)
        chunk_metadata = document.metadata.copy()
        
        # Get document-level images for lookup
        doc_images = document.metadata.get("images", [])
        
        # Remove document-level 'images' field - we'll add chunk-specific images below
        chunk_metadata.pop("images", None)
        
        # Add chunk-specific fields
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id

        return chunk_metadata
