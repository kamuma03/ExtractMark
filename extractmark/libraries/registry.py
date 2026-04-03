"""Library registry -- resolves library IDs to adapter instances."""

from __future__ import annotations

from extractmark.config import LibraryConfig
from extractmark.libraries.base import LibraryAdapter
from extractmark.libraries.camelot_adapter import CamelotAdapter
from extractmark.libraries.docling_adapter import DoclingAdapter
from extractmark.libraries.marker_adapter import MarkerAdapter
from extractmark.libraries.markitdown_adapter import MarkItDownAdapter
from extractmark.libraries.mineru_adapter import MinerUAdapter
from extractmark.libraries.nougat_adapter import NougatAdapter
from extractmark.libraries.pdfminer_adapter import PdfminerAdapter
from extractmark.libraries.pdfplumber_adapter import PdfplumberAdapter
from extractmark.libraries.pymupdf import PyMuPDFAdapter
from extractmark.libraries.pypdfium2_adapter import Pypdfium2Adapter
from extractmark.libraries.python_docx_adapter import PythonDocxAdapter
from extractmark.libraries.python_pptx_adapter import PythonPptxAdapter
from extractmark.libraries.surya_adapter import SuryaAdapter
from extractmark.libraries.tabula_adapter import TabulaAdapter
from extractmark.libraries.table_transformer_adapter import TableTransformerAdapter
from extractmark.libraries.tesseract_adapter import TesseractAdapter
from extractmark.libraries.unstructured_adapter import UnstructuredAdapter


_LIB_MAP: dict[str, type] = {
    "LIB-01": PyMuPDFAdapter,
    "LIB-02": PdfplumberAdapter,
    "LIB-03": Pypdfium2Adapter,
    "LIB-04": CamelotAdapter,
    "LIB-05": TabulaAdapter,
    "LIB-06": PythonDocxAdapter,
    "LIB-07": PythonPptxAdapter,
    "LIB-08": DoclingAdapter,
    "LIB-09": MinerUAdapter,
    "LIB-10": MarkerAdapter,
    "LIB-11": SuryaAdapter,
    "LIB-12": UnstructuredAdapter,
    "LIB-13": MarkItDownAdapter,
    "LIB-14": TesseractAdapter,
    "LIB-15": TableTransformerAdapter,
    "LIB-16": NougatAdapter,
    "LIB-17": PdfminerAdapter,
}


def get_library(lib_id: str, config: LibraryConfig) -> LibraryAdapter:
    """Create a library adapter for the given library ID."""
    adapter_cls = _LIB_MAP.get(lib_id)
    if adapter_cls is None:
        raise ValueError(
            f"Library adapter not implemented: {lib_id}. "
            f"Available: {list(_LIB_MAP.keys())}"
        )
    return adapter_cls()
