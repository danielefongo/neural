import re
from pydoc import locate


class Config:
    def __init__(self, *init):
        self.init = init

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
