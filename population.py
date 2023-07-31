from typing import List, Dict
from dataclasses import dataclass, field


@dataclass(order=True)
class model_AIPot():
    sort_index: int = field(init=False, repr=False)
    model_name: str
    model: int
    rating: float
    enviroment: str
    team: str


    def __post_init__(self):
        self.sort_index = self.rating

@dataclass(order=True)
class thread_AIPot():
    thread: Dict[str,model_AIPot]
