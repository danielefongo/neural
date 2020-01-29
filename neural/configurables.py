import re


class Config:
    def __init__(self, init: list):
        self.init = init

    def self_structure(self):
        return dict(
            clazz=re.search('\'(.*)\'', str(self.__class__)).group(1),
            hash=hash(self),
            init=[a.self_structure() if isinstance(a, Config) else a for a in self.init],
        )


class UnderConfig(Config):
    def self_structure(self):
        a = super().self_structure()
        a["d"] = 3
        return a