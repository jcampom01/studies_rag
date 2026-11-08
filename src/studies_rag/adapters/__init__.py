from studies_rag.adapters.base import Adapter, PageRecord, normalize
from studies_rag.adapters.master_notes import MasterNotesAdapter

# Name used on the command line -> adapter class
ADAPTERS: dict[str, type[Adapter]] = {
    "master_notes": MasterNotesAdapter,
}

__all__ = ["ADAPTERS", "Adapter", "PageRecord", "normalize"]
