from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray
import heapq


rng = np.random.default_rng()


class Design:
    def __init__(self, inclusions: NDArray = None) -> None:
        self.heap: tuple[float, set[int]] = []

    def copy(self) -> Design:
        new_design = Design()
        new_design.heap = self.heap[:]
        return new_design

    def pull(self) -> tuple[float, set[int]]:
        return heapq.heappop(self.heap)

    def push(self, *args: tuple[float, set[int]]) -> None:
        for arg in args:
            heapq.heappush(self.heap, arg)


def generate_design(design: Design, num_changes: int, length_function: Callable[[float, float], float]) -> Design:
    new_design = design.copy()
    for _ in range(num_changes):
        sample_1_prob, sample_1_IDs = new_design.pull()
        sample_2_prob, sample_2_IDs = new_design.pull()
        if sample_1_IDs == sample_2_IDs:
            new_design.push((sample_1_prob+sample_2_prob, sample_1_IDs))
        else:
            length = length_function(sample_1_prob, sample_2_prob)
            n1 = rng.choice(list(sample_1_IDs - sample_2_IDs))
            n2 = rng.choice(list(sample_2_IDs - sample_1_IDs))
            new_design.push(
                (length, sample_1_IDs-{n1}|{n2}),
                (sample_1_prob-length, sample_1_IDs),
                (length, sample_2_IDs-{n2}|{n1}),
                (sample_2_prob-length, sample_2_IDs)
            )
    return new_design

# TODO
# make bars out of design
# make sample from design

class Old:
    def __init__(
        self,
        x: ndarray,
        y: ndarray,
        inclusions: ndarray,
        threshold_x: float = 1e-2,
        threshold_y: float = 1e-2,
        length: float = 1e-5

    ) -> None:
        self.x = x
        self.y = y
        self.inclusions = inclusions
        self.threshold_y = threshold_y
        self.threshold_x = threshold_x
        self.length = length
        self.rng = np.random.default_rng()

        self.best_design = None
        self.best_cost = None

    def generate_initial_design(self):
        bars = []
        level = 0
        for p in self.inclusions:
            next_level = level + p
            if next_level < 1-1e-9:
                interval = P.closed(level, next_level)
                level = next_level
            elif next_level > 1+1e-9:
                interval = P.closed(level, 1) | P.closed(0, next_level-1)
                level = next_level-1
            else:
                interval = P.closed(level, 1)
                level = 0
            bars.append(interval)

        events = []
        for i, bar in enumerate(bars):
            for interval in bar:
                events.append((interval.lower, 'start', i))
                events.append((interval.upper, 'end', i))

        events.sort()

        active = set()
        design = P.IntervalDict()
        last_point = 0

        for point, event_type, bar_index in events:
            if event_type == 'start':
                active.add(bar_index)
            elif event_type == 'end':
                if last_point != point:
                    design[P.open(round(last_point, 9), round(point, 9))] = set(active)
                active.remove(bar_index)

            last_point = point
        return design

    def clean_design(self, design: P.IntervalDict) -> P.IntervalDict:
        cleaned_design = P.IntervalDict()
        samples = design.values()
        for i, intervals in enumerate(design):
            for interval in intervals:
                cleaned_design[interval.replace(P.OPEN, interval.lower, interval.upper, P.OPEN)] = samples[i]
        return cleaned_design

    def length_of_interval(self, interval):
        length = []
        for i in interval:
            if i.upper - i.lower != 0:
                length.append(i.upper - i.lower)
        return length

    def match_interval(self, interval, length):
        #print("itervalleng", interval, length)
        for i in interval:
            if i.upper - i.lower >= length - 1e-6:
                return i.lower

    def generate_new_design(self, design: P.IntervalDict, num_changes: int) -> P.IntervalDict:
        new_design = design.copy()
        for _ in range(num_changes):
            intervals = new_design.keys()
            samples = new_design.values()
            index1, index2 = self.rng.choice(len(intervals), size=2, replace=False)
            interval1, interval2 = intervals[index1], intervals[index2]
            if self.length_of_interval(interval1) and self.length_of_interval(interval2):
                sample1, sample2 = samples[index1], samples[index2]
                n1 = self.rng.choice(list(sample1 - sample2))
                n2 = self.rng.choice(list(sample2 - sample1))
                #length = self.rng.choice([0.01, 0.05, 0.1])
                length = self.length

                valid_length = round(min(*self.length_of_interval(interval1), *self.length_of_interval(interval2), length), 9)
                interval1_lower = self.match_interval(interval1, valid_length)
                interval2_lower = self.match_interval(interval2, valid_length)
                new_design[P.open(interval1_lower, round(interval1_lower+valid_length, 9))] = sample1 - {n1} | {n2}
                new_design[P.open(interval2_lower, round(interval2_lower+valid_length, 9))] = sample2 - {n2} | {n1}


        return new_design

    def make_output_design(self, design: P.IntervalDict) -> list:
        output = []
        samples = design.values()
        for i, intervals in enumerate(design):
            length = 0
            for interval in intervals:
                length += interval.upper - interval.lower
            output.append((np.array(list(samples[i])), round(length, 9)))
        return output

    def criteria(self, design):
        output_design = self.make_output_design(design)
        #print(output_design)
        NHT_estimator = np.array([np.sum(self.x[sample[0]] / self.inclusions[sample[0]]) for sample in output_design])
        NHT_estimator_y = np.array([np.sum(self.y[sample[0]] / self.inclusions[sample[0]]) for sample in output_design])
        probabilities = np.array([sample[1] for sample in output_design])
        #print(NHT_estimator)
        var_NHT = np.sum((NHT_estimator - np.sum(self.x)) ** 2 * probabilities)
        #print('cojaiie?',var_NHT)
        var_NHT_y = np.sum((NHT_estimator_y - np.sum(self.y)) ** 2 * probabilities)
        var_NHT_yr = np.sum((NHT_estimator_y*np.sum(self.x)/NHT_estimator - np.sum(self.y)) ** 2 * probabilities)
        NHT_yr_Bias= (np.sum( probabilities * NHT_estimator_y*np.sum(self.x)/NHT_estimator) - np.sum(self.y))/np.sum(self.y)
        NHT_y_Bias= (np.sum(NHT_estimator_y * probabilities) - np.sum(self.y))/np.sum(self.y)
        return var_NHT, var_NHT_y, var_NHT_yr, NHT_estimator, NHT_estimator_y, NHT_yr_Bias, NHT_y_Bias


    def save_as_json(self, design):
        # Saving list to JSON file
        list_design = []
        for d in design:
            list_design.append((d[0].tolist(), d[1]))
        with open('best_design.json', 'w') as f:
            json.dump(list_design, f)

    def run(self, max_iterations, num_new_nodes, max_open_set_size, num_changes):
        open_set = []
        initial_design = self.generate_initial_design()
        heapq.heappush(open_set, (self.criteria(initial_design)[0], initial_design))
        closed_set = set()
        iterations = 0
        new_cost = self.criteria(initial_design)[0]


        while open_set and iterations < max_iterations:
            iterations += 1
            print(f'\r{iterations/max_iterations}', end=' ')

            print('new', np.round(self.threshold_x/self.best_cost, 4) if self.best_cost is not None else 0, np.round(self.threshold_y/self.best_cost_y, 4) if self.best_cost is not None else 0)
            _, current_design = heapq.heappop(open_set)
            if tuple(map(tuple, current_design)) in closed_set:
                continue
            closed_set.add(tuple(map(tuple, current_design)))
            for _ in range(num_new_nodes):
                new_design = self.generate_new_design(current_design, num_changes)
                if tuple(map(tuple, new_design)) not in closed_set:
                    new_cost = self.criteria(new_design)[0] + self.rng.random() * 0.0000001
                    if len(open_set) < max_open_set_size:
                        heapq.heappush(open_set, (new_cost, new_design))
                    else:
                        heapq.heappushpop(open_set, (new_cost, new_design))
                    if self.best_design is None or new_cost < self.best_cost:
                        self.best_design = self.make_output_design(new_design)
                        self.best_cost = self.criteria(new_design)[0]
                        self.best_cost_y = self.criteria(new_design)[1]
                        self.best_cost_yr = self.criteria(new_design)[2]
                        NHT_corr = np.corrcoef(self.criteria(new_design)[3], self.criteria(new_design)[4])[0,1]
                        print('Congrat!', 'cor',np.round(NHT_corr,2), 'bias 1 and 2',round(self.criteria(new_design)[5],3), round(self.criteria(new_design)[6],3) , 'x:',  np.round(self.threshold_x/self.best_cost, 4), 'y:', np.round(self.threshold_y/self.best_cost_y, 4), 'yr:', np.round(self.threshold_y/self.best_cost_yr, 4) )
                        self.save_as_json(self.best_design)

                        if self.best_cost < self.threshold_x:
                            print('costs', round(self.best_cost), round(self.best_cost_y))
                            print('CONGRATS! You Are Where You Wish To')
                            return iterations
        return iterations
