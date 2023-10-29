import enum

class Inequality(enum.Enum):
    LT = 1
    GE = 2
#:

class PredicateBase(object):
    def __init__(self, inequality: Inequality, theta: float)  -> None:
        self.inequality = inequality # enum Inequality above
        self.theta = theta
    #:

    def label(self, variable_name: str):
        s = f'{variable_name} ' \
            + ('<' if self.is_lt() else '>=') \
            + f'{self.theta}'
        return s
    #:

    def is_lt(self):
        return self.inequality == Inequality.LT
    #:

    def is_ge(self):
        return self.inequality == Inequality.GE
    #:

    def __call__(self, x: float) -> bool:
        return (x < self.theta) if self.is_lt() else (x >= self.theta)
    #:
#:class

class Predicate_LessThan(PredicateBase):
    def __init__(self, theta: float):
        super(Predicate_LessThan, self).__init__(Inequality.LT, theta)
    #:
#:

class Predicate_GreaterEqual(PredicateBase):
    def __init__(self, theta: float):
        super(Predicate_GreaterEqual, self).__init__(Inequality.GE, theta)
    #:
#:
