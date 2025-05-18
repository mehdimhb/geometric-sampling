import copy
import time
from functools import lru_cache
from math import isclose
from typing import List, Optional

from geometric_sampling.design import DesignGenetic
from geometric_sampling.structs import Sample

import numpy as np


class GeneticOptimizer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def partition_design(
        self, fip_list: list[float], num_partitions: int
    ) -> tuple[dict[int, list[int]], set[int]]:
        target = sum(fip_list) / num_partitions
        partitions, border_units = {}, set()
        current_partition, cumulative_sum = 0, 0.0
        partitions[current_partition] = []

        for index, fip in enumerate(fip_list):
            if (
                isclose(cumulative_sum + fip, target, abs_tol=1e-9)
                or cumulative_sum + fip < target
            ):
                partitions[current_partition].append(index)
                cumulative_sum += fip
            else:
                if not isclose(cumulative_sum, target, abs_tol=1e-9):
                    border_units.add(index)
                    cumulative_sum = fip - (target - cumulative_sum)
                    current_partition += 1
                    partitions[current_partition] = []
                    continue

                current_partition += 1
                partitions[current_partition] = []
                cumulative_sum = fip
                partitions[current_partition].append(index)

        return partitions, border_units

    def _chunk_ids(self, ids: frozenset[int], n_parts: int) -> List[set[int]]:
        lst = sorted(ids)
        base, rem = divmod(len(lst), n_parts)
        chunks, idx = [], 0
        for i in range(n_parts):
            size = base + (1 if i < rem else 0)
            chunks.append(set(lst[idx : idx + size]))
            idx += size
        return chunks

    @lru_cache(maxsize=None)
    def _chunk_ids_cached(self, ids_tuple: tuple[int, ...], n_parts: int) -> List[set]:
        return self._chunk_ids(frozenset(ids_tuple), n_parts)

    def combine_n_parents(
        self,
        parents: List[DesignGenetic],
        random_pull: bool = False,
    ) -> tuple[DesignGenetic, DesignGenetic]:

        parents = [par.copy() for par in parents]

        n = len(parents)
        rng = parents[0].rng

        cum_bounds: List[np.ndarray] = []
        samples: List[List[Sample]] = []
        all_chunks: List[List[List[set[int]]]] = []

        for p in parents:
            segs: List[Sample] = list(p.heap)
            samples.append(segs)
            weights = np.array([s.probability for s in segs], dtype=float)
            bounds = np.concatenate(([0.0], weights.cumsum()))
            cum_bounds.append(bounds)

            chunks_for_parent = [
                self._chunk_ids_cached(tuple(sorted(s.ids)), n)
                for s in segs
            ]
            all_chunks.append(chunks_for_parent)

        if not cum_bounds or cum_bounds[0].size <= 1:
            empty = DesignGenetic(inclusions=None, rng=rng)
            return empty, empty
        #
        # --- ۳) ادغام همهٔ مرزها و پیداکردن طول زیر-قطعه‌ها ---
        raw = np.concatenate(cum_bounds)
        # sort کنید
        sorted_raw = np.sort(raw)
        # با tol دسته‌بندی
        tol = 1e-2  # یا هر tol معقولی که مناسب مسئله‌ی شماست
        unified = [sorted_raw[0]]
        for v in sorted_raw[1:]:
            if v - unified[-1] > tol:
                unified.append(v)
        unified = np.array(unified)
        # آنگاه
        lengths = np.diff(unified)
        valid_idx = np.where(lengths > tol)[0]
        if valid_idx.size == 0:
            empty = DesignGenetic(inclusions=None, rng=rng)
            return empty, empty

        seg_indices = []
        for b in cum_bounds:
            idxs = np.searchsorted(b, unified[valid_idx], side='right') - 1
            seg_indices.append(idxs)
        #

        child1 = DesignGenetic(inclusions=None, rng=rng)
        child2 = DesignGenetic(inclusions=None, rng=rng)

        for pos in valid_idx:
            L = float(lengths[pos])
            ids_c1: set[int] = set()
            ids_c2: set[int] = set()

            for i in range(n):
                j = int(seg_indices[i][np.where(valid_idx == pos)[0][0]])
                chunks = all_chunks[i][j]
                ids_c1.update(chunks[i])
                ids_c2.update(chunks[(i + 1) % n])

            s1 = Sample(L, frozenset(ids_c1), index=[child1.step, []])
            s2 = Sample(L, frozenset(ids_c2), index=[child2.step, []])
            child1.push(s1)
            child2.push(s2)

            child1.step += 1
            child1.changes += 1
            child2.step += 1
            child2.changes += 1

        return child1, child2

    def design_fragmentation_n(
        self, parent: DesignGenetic, n_parts: int, random_pull: bool = False
    ) -> List[DesignGenetic]:
        children: List[DesignGenetic] = [
            DesignGenetic(inclusions=None, rng=parent.rng) for _ in range(n_parts)
        ]

        while parent.heap:
            sample = parent.pull(random_pull)
            ids_chunks = self._chunk_ids(sample.ids, n_parts)
            # total_ids = len(sample.ids)

            for i, ids_part in enumerate(ids_chunks):
                if not ids_part:
                    continue
                weight_i = sample.probability
                s = Sample(weight_i, frozenset(ids_part), index=sample.index)
                children[i].push(s)
                children[i].changes += 1

        return children

    def combine_fragments_n(
        self, fragments: List[DesignGenetic], random_pull: bool = False
    ) -> DesignGenetic:
        n = len(fragments)
        leftovers: List[Optional[Sample]] = [None] * n

        child = DesignGenetic(inclusions=None, rng=fragments[0].rng)

        while any(leftovers) or any(frag.heap for frag in fragments):
            pulled: List[Sample] = []
            for i, frag in enumerate(fragments):
                if leftovers[i] is not None:
                    r = leftovers[i]
                    leftovers[i] = None
                else:
                    r = frag.pull(random_pull)
                pulled.append(r)

            length = min(r.probability for r in pulled)
            if length <= 0:
                break

            combined_ids: set[int] = set()
            for r in pulled:
                combined_ids |= r.ids

            # idx = pulled[0].index

            child_sample = Sample(
                length, frozenset(combined_ids), index=[child.step, []]
            )
            child.push(child_sample)

            for i, r in enumerate(pulled):
                rem = r.probability - length
                if rem > 0:
                    leftovers[i] = Sample(rem, r.ids, index=[-1, []])
                else:
                    leftovers[i] = None

            child.step += 1
            child.changes += 1

        return child
