from mmseg.datasets import DATASETS
from mmseg.datasets.builder import build_dataset


@DATASETS.register_module()
class MixDataset(object):

    def __init__(self, source: dict, target: dict, cfg: dict):

        self.source = build_dataset(source)
        self.target = build_dataset(target)
        self.ignore_index = self.target.ignore_index
        self.CLASSES = self.target.CLASSES
        self.PALETTE = self.target.PALETTE
        assert self.target.ignore_index == self.source.ignore_index
        assert self.target.CLASSES == self.source.CLASSES
        assert self.target.PALETTE == self.source.PALETTE

    def __getitem__(self, idx):
        s1 = self.source[idx // len(self.target)]
        s2 = self.target[idx % len(self.target)]
        return {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img']
        }

    def __len__(self):
        return len(self.source) * len(self.target)
