import enum
import typing

import numpy as np

from Region import Region

class DecisionTreeNode(object):
    class Type(enum.Enum):
        TERMINAL = 0
        SPLIT = 1
    #:

    # Should not be constructed directly!
    # Use TerminalNode or SplitNode.
    def __init__(self, node_type: Type, region: Region) -> None:
        self.type = node_type
        self.region = region
    #:

    def is_terminal(self):
        return self.type == DecisionTreeNode.Type.TERMINAL
    #:

    def is_split(self):
        return self.type == DecisionTreeNode.Type.SPLIT
    #:
#:

class TerminalNode(DecisionTreeNode):
    def __init__(self,
                 region: Region,
                 value: int) -> None:
        super().__init__(self, DecisionTreeNode.Type.TERMINAL, region)
        self.value = value
    #:

    def __call__(self, x) -> int:
        return self.value
    #:

    def label(self):
        return f'{self.value}'
    #:
#:class

class SplitNode(DecisionTreeNode):
    def __init__(self,
                 region: Region,
                 j:int, n:int) -> None:
        super().__init__(self, DecisionTreeNode.Type.SPLIT, region)
        self.region: Region = region
        self.j: int = j
        self.n:int  = n
        self.children: typing.List[DecisionTreeNode] = []
    #:

    def add_child(self, child):
        self.children.append(child)
    #:

    def label(self):
        assert len(self.children) == 2
        return self.children[0].region.predicate.label(
            Region.dataset.attribute_name(self.j)
        )
    #:

    def __call__(self, x) -> bool:
        assert len(self.children) == 2
        return self.children[0].region.predicate(x[self.j])
    #:__call__()

#:class

