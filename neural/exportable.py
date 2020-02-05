import inspect
import re
from pydoc import locate


class Exportable:
    def __init__(self):
        actual_frame = inspect.currentframe()
        frame = self._last_frame_for_self_init(actual_frame)

        args, varargs, _, values = inspect.getargvalues(frame)

        argument_list = [values[i] for i in args[1:]]
        if varargs is not None:
            argument_list += [value for value in values[varargs]]

        self.init = argument_list

    def export(self):
        return dict(
            clazz=re.search('\'(.*)\'', str(self.__class__)).group(1),
            hash=hash(self),
            init=[initElement.export() if isinstance(initElement, Exportable) else initElement for initElement in self.init]
        )

    @staticmethod
    def use(config):
        exportableType = locate(config["clazz"])
        init = config["init"]
        for i in range(len(init)):
            if isinstance(init[i], dict) and "clazz" in init[i].keys():
                clazz = locate(init[i]["clazz"])
                init[i] = clazz.use(init[i])

        return exportableType(*init) if len(init) else exportableType()

    def _last_frame_for_self_init(self, actual_frame):
        stack = inspect.getouterframes(actual_frame)
        frames = [frameInfo[0] for frameInfo in stack]
        valid_frames = [frame for frame in frames if "self" in frame.f_locals.keys() and frame.f_locals["self"] == self]
        return valid_frames[-1]
