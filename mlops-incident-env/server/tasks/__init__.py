from .easy_task import EasyTaskGrader
from .medium_task import MediumTaskGrader
from .hard_task import HardTaskGrader
from .cascade_task import CascadeTaskGrader

GRADERS = {
    "easy":    EasyTaskGrader(),
    "medium":  MediumTaskGrader(),
    "hard":    HardTaskGrader(),
    "cascade": CascadeTaskGrader(),
}
