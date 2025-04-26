from math import isclose
from typing import List, Optional

import numpy as np

from geometric_sampling.design import DesignGenetic
from geometric_sampling.structs import  SampleGenetic



class GeneticOptimizer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def partition_design(self,
                         fip_list: list[float],
                         num_partitions: int
    ) -> tuple[dict[int, list[int]], set[int]]:

        target = sum(fip_list) / num_partitions
        partitions, border_units = {}, set()
        current_partition, cumulative_sum = 0, 0.0
        partitions[current_partition] = []

        for index, fip in enumerate(fip_list):
            if isclose(cumulative_sum + fip, target, abs_tol=1e-9) or cumulative_sum + fip < target:
                partitions[current_partition].append(index)
                cumulative_sum += fip
            else:
                if not isclose(cumulative_sum, target, abs_tol=1e-9):
                    border_units.add(index)
                    cumulative_sum = fip - (target-cumulative_sum)
                    current_partition += 1
                    partitions[current_partition] = []
                    continue

                current_partition += 1
                partitions[current_partition] = []
                cumulative_sum = fip
                partitions[current_partition].append(index)

        return partitions, border_units


    def _chunk_ids(
            self,
            ids: frozenset[int],
            n_parts: int
    ) -> List[set[int]]:

        lst = sorted(ids)
        base, rem = divmod(len(lst), n_parts)
        chunks, idx = [], 0
        for i in range(n_parts):
            size = base + (1 if i < rem else 0)
            chunks.append(set(lst[idx: idx + size]))
            idx += size
        return chunks

    def combine_n_parents(
            self,
            parents: List[DesignGenetic],
            random_pull: bool = False,
    ) -> DesignGenetic:

        n = len(parents)
        leftovers: List[Optional[SampleGenetic]] = [None] * n
        child_design = DesignGenetic(inclusions=None, rng=parents[0].rng)

        while any(leftovers) or any(parent.heap for parent in parents):

            pulled: List[SampleGenetic] = []
            for i, par in enumerate(parents):
                if leftovers[i] is not None:
                    r = leftovers[i]
                    leftovers[i] = None
                else:
                    r = par.pull(random_pull)
                pulled.append(r)

            length = min(r.probability for r in pulled)
            if length <= 0:
                break

            all_chunks = [self._chunk_ids(r.ids, n) for r in pulled]
            child_ids: set[int] = set()
            for i in range(n):
                child_ids |= all_chunks[i][i]

            child = SampleGenetic(length, frozenset(child_ids), index=[child_design.step, []])
            child_design.push(child)

            for i, r in enumerate(pulled):
                rem = r.probability - length
                if rem > 0:
                    leftovers[i] = SampleGenetic(rem, r.ids, index=[-1, []])
                else:
                    leftovers[i] = None

            child_design.step += 1
            child_design.changes += 1

        return child_design

