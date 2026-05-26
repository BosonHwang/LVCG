"""Abstract base class for dataset providers used by the probing pipeline.

A ``DatasetProvider`` decouples ``run_probing.py`` from any specific dataset
implementation: each provider knows how to materialise its own ``DataLoader``
trio and how to expose meaningful per-task metadata (label names, class count).

Adding a new dataset requires only:

1. Subclassing :class:`DatasetProvider` and implementing :meth:`build`.
2. Registering the subclass in :mod:`datasets.__init__`.
3. Referring to its ``type`` from ``configs/probing.yaml``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from torch.utils.data import DataLoader


@dataclass
class DatasetBundle:
    """Output of :meth:`DatasetProvider.build`.

    Attributes:
        train_loader: training DataLoader (may be ``None`` if labels are scarce).
        val_loader:   validation DataLoader.
        test_loader:  test DataLoader.
        num_classes:  number of binary heads of the linear probe.
        label_names:  optional human-readable names of the ``num_classes`` outputs.
        meta:         arbitrary metadata for logging / downstream filtering.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    label_names: Optional[List[str]] = None
    meta: dict = field(default_factory=dict)

    def as_tuple(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Backward-compatible tuple unpacking used by ``run_probing.py``."""
        return self.train_loader, self.val_loader, self.test_loader, self.num_classes


class DatasetProvider(ABC):
    """Source of train/val/test ECG :class:`~torch.utils.data.DataLoader` triples.

    Subclasses are configured purely from ``**kwargs`` so they can be
    constructed straight from a YAML config block.  All probing-specific knobs
    (``label_ratio``, ``batch_size``, ``num_workers``) are passed to
    :meth:`build` rather than the constructor; this keeps providers cheap to
    instantiate and lets a single provider serve multiple ``label_ratio`` runs.
    """

    #: Provider name (filled in by the registry).  Optional, used for logging.
    name: str = ""

    @abstractmethod
    def build(
        self,
        label_ratio: float,
        batch_size: int,
        num_workers: int,
    ) -> DatasetBundle:
        """Return a :class:`DatasetBundle` for the requested configuration."""
        raise NotImplementedError
