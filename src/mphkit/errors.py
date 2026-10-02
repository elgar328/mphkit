"""Exceptions and warnings raised by mphkit."""


class LicenseError(RuntimeError):
    """Raised when a feature needs a COMSOL license that is not available."""


class StepWarning(UserWarning):
    """
    Warns that a number given as `step=` or `outer=`, a position counted
    from 1, is also the value of another position: `step=10` on times
    0, 1, ..., 10 is t = 9, not t = 10. Pick by value to mean the time,
    `step={'t': 10}`; code that means positions can turn it off with
    `warnings.filterwarnings('ignore', category=mk.StepWarning)`, or on
    the command line, where the class cannot be named, with
    `python -W "ignore:step=" -W "ignore:outer=" script.py`.
    """
