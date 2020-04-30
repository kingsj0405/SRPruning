import fire
import torch

from config import Config
from dataset import SRDatasetFromDIV2K


class Main:
    def test(self):
        raise NotImplementedError

    def train(self):
        print("[INFO] Set configuration")
        config = Config()
        config.prepare_experiment()
        # FIXME: After this issue resolved
        # https://github.com/makinacorpus/easydict/issues/20
        config = config.cfg


if __name__ == '__main__':
    fire.Fire(Main)
