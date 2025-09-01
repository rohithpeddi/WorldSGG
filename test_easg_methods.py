from lib.supervised.ego_config import EgoConfig
from lib.supervised import EASG
from test_easg_base import TestEASGBase


class TestEASG(TestEASGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = EASG(
            verbs=self._val_dataset.verbs,
            objs=self._val_dataset.objs,
            rels=self._val_dataset.rels
        )
        self._model.to(device=self._device)


def main():
    conf = EgoConfig()
    if conf.method_name == "easg":
        test_class = TestEASG(conf)
    else:
        raise NotImplementedError

    test_class.init_method_evaluation()


if __name__ == "__main__":
    main()