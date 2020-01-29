import inspect
import re
from pydoc import locate


class Config:
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


    def self_structure(self):
        return dict(
            clazz=re.search('\'(.*)\'', str(self.__class__)).group(1),
            hash=hash(self),
            init=[a.self_structure() if isinstance(a, Config) else a for a in self.init],
        )

    @staticmethod
    def self_create(config):
        unittype = locate(config["clazz"])
        hashino = config["hash"]
        init = config["init"]
        for i in range(len(init)):
            if isinstance(init[i], dict) and "clazz" in init[i].keys():
                clazz = locate(init[i]["clazz"])
                init[i] = clazz.self_create(init[i])

        return unittype(*init) if len(init) else unittype()

    @staticmethod
    def params_as_list():
        frame = inspect.currentframe().f_back
        args, varargs, _, values = inspect.getargvalues(frame)
        args.remove("self")
        listina = [values[i] for i in args]
        if varargs is not None:
            listina += [value for value in values[varargs]]

        return listina
