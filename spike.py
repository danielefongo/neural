import inspect

from neural.activations import Linear
from neural.losses import MSE
from neural.optimizers import Adam


class A:
    def __init__(self):
        frame = inspect.currentframe()
        stack = inspect.getouterframes(frame)
        for stackFrame in [a[0] for a in stack]:
            if inspect.getframeinfo(stackFrame).function != "__init__":
                break
            frame = stackFrame

        args, varargs, _, values = inspect.getargvalues(frame)

        argument_list = [values[i] for i in args[1:]]
        if varargs is not None:
            argument_list += [value for value in values[varargs]]

        self.init = argument_list

class B(A):
    pass


class C(B):
    def __init__(self, *args):
        super().__init__()

def prova():
    c = C(1,2,3)

prova()