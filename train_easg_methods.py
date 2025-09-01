from lib.supervised.ego_config import EgoConfig
from lib.supervised import EASG
from train_easg_base import TrainEASGBase


class TrainEASG(TrainEASGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = EASG(
            verbs=self._train_dataset.verbs,
            objs=self._train_dataset.objs,
            rels=self._train_dataset.rels
        )
        self._model.to(device=self._device)


def main():
    conf = EgoConfig()
    if conf.method_name == "easg":
        train_class = TrainEASG(conf)
    else:
        raise NotImplementedError

    train_class.init_method_training()


if __name__ == "__main__":
    main()