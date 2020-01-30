import inspect
import re
from pydoc import locate


class Exportable:
    def __init__(self):
        frame = inspect.currentframe()
        stack = inspect.getouterframes(frame)
        for stackFrame in [frameInfo[0] for frameInfo in stack]:
            if inspect.getframeinfo(stackFrame).function != "__init__":
                break
            frame = stackFrame

        args, varargs, _, values = inspect.getargvalues(frame)

        argument_list = [values[i] for i in args[1:]]
        if varargs is not None:
            argument_list += [value for value in values[varargs]]

        self.init = argument_list

    def self_structure(self):
        return dict(
            clazz=re.search('\'(.*)\'', str(self.__class__)).group(1),
            hash=hash(self),
            init=[initElement.self_structure() if isinstance(initElement, Exportable) else initElement for initElement in self.init],
        )

    @staticmethod
    def self_create(config):
        exportableType = locate(config["clazz"])
        init = config["init"]
        for i in range(len(init)):
            if isinstance(init[i], dict) and "clazz" in init[i].keys():
                clazz = locate(init[i]["clazz"])
                init[i] = clazz.self_create(init[i])

        return exportableType(*init) if len(init) else exportableType()
