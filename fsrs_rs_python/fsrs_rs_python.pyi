from typing import List, Optional, Tuple

class FSRS:
    def __init__(self, parameters: List[float]) -> None: ...
    def compute_parameters(
        self, fsrs_items: List[FSRSItem]
    ) -> Tuple[List[float], float]: ...
    def benchmark(self, fsrs_items: List[FSRSItem]) -> List[float]: ...
    def evaluate(self, fsrs_items: List[FSRSItem]) -> float: ...
    def memory_state_batch(
        self,
        items: List[FSRSItem],
        starting_states: Optional[List[Optional[MemoryState]]] = None,
    ) -> List[MemoryState]: ...

class FSRSItem:
    def __init__(self, reviews: List[FSRSReview]) -> None: ...

class FSRSReview:
    def __init__(self, rating: int, delta_t: float) -> None: ...

class MemoryState:
    def __init__(self, stability: float, difficulty: float) -> None: ...
    stability: float
    difficulty: float

DEFAULT_PARAMETERS: List[float]
