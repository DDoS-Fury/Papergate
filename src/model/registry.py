"""Dynamic node registry for the streaming TGN.

A Zero Trust access stream contains an *unbounded* set of entities (users, IPs,
resources). The TGN memory, however, is a fixed-size buffer of ``capacity`` slots.
``NodeRegistry`` bridges the two: it maps arbitrary, hashable **external keys**
(e.g. ``"user:42"`` or an integer node id) to contiguous internal slot indices in
``[0, capacity)``, assigning a fresh slot on first sight.

When the registry is full a new key evicts the least-recently-updated slot — its
recency is read from the TGN ``memory.last_update`` buffer — and the caller is
asked (via ``on_evict``) to cold-start that memory slot. The mapping is JSON
serialisable so train-time and serve-time indices stay consistent across restarts.
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, Optional, Sequence


class NodeRegistry:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self._key_to_idx: dict[Hashable, int] = {}
        self._idx_to_key: dict[int, Hashable] = {}
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._key_to_idx)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._key_to_idx

    def get(self, key: Hashable) -> Optional[int]:
        """Return the internal index for ``key`` or ``None`` if unseen."""
        return self._key_to_idx.get(key)

    def preregister(self, keys: Iterable[Hashable]) -> None:
        """Assign slots to ``keys`` in order (identity mapping for ``range(n)``)."""
        for key in keys:
            self.get_or_add(key)

    def get_or_add(
        self,
        key: Hashable,
        *,
        recency: Optional[Sequence[float]] = None,
        on_evict: Optional[Callable[[int], None]] = None,
    ) -> tuple[int, bool]:
        """Return ``(idx, created)`` for ``key``, allocating a slot if needed.

        ``recency`` (e.g. ``memory.last_update``) drives least-recently-updated
        eviction once the registry is full; ``on_evict(idx)`` is called so the
        caller can reset the reused memory slot.
        """
        idx = self._key_to_idx.get(key)
        if idx is not None:
            return idx, False

        if self._next_idx < self.capacity:
            idx = self._next_idx
            self._next_idx += 1
        else:
            idx = self._select_eviction(recency)
            old_key = self._idx_to_key.pop(idx)
            self._key_to_idx.pop(old_key, None)
            if on_evict is not None:
                on_evict(idx)

        self._key_to_idx[key] = idx
        self._idx_to_key[idx] = key
        return idx, True

    def _select_eviction(self, recency: Optional[Sequence[float]]) -> int:
        used = list(self._idx_to_key.keys())
        if recency is None:
            return min(used)
        return min(used, key=lambda i: float(recency[i]))

    # --- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        # Store entries as [key, idx] pairs to preserve key types (JSON object
        # keys would coerce ints to strings).
        return {
            "capacity": self.capacity,
            "next_idx": self._next_idx,
            "entries": [[key, idx] for key, idx in self._key_to_idx.items()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NodeRegistry":
        reg = cls(int(data["capacity"]))
        reg._next_idx = int(data["next_idx"])
        for key, idx in data["entries"]:
            idx = int(idx)
            reg._key_to_idx[key] = idx
            reg._idx_to_key[idx] = key
        return reg
