import os.path as osp

from mmseg.datasets import DATASETS
from mmseg.datasets.pipelines import Compose


@DATASETS.register_module()
class ZurichPairDataset(object):

    def __init__(self,
                 data_root: str,
                 pair_list_path: str,
                 pipeline: dict,
                 repeat_times=None):

        self.pipeline = Compose(pipeline)
        self.img_dir = data_root
        self.ann_dir = data_root
        self.img_infos = {}
        with open(pair_list_path, 'r') as f:
            line = f.readline()
            idx = 0
            while line:
                night, day = line.strip().split(',')

                day = osp.join(data_root, f'{day}_rgb_anon.png')
                night = osp.join(data_root, f'{night}_rgb_anon.png')

                day_info = dict(filename=day)
                night_info = dict(filename=night)

                self.img_infos[idx] = (day_info, night_info)
                idx += 1
                line = f.readline()
        self._ori_len = len(self.img_infos)
        self.repeat_times = 1
        if repeat_times:
            self.repeat_times = repeat_times

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = osp.join(self.img_dir)
        results['seg_prefix'] = self.ann_dir

    def __getitem__(self, idx):

        day_info, night_info = self.img_infos[idx % self._ori_len]
        day_result = dict(img_info=day_info, ann_info=None)
        night_result = dict(img_info=night_info, ann_info=None)
        # self.pre_pipeline(day_result)
        # self.pre_pipeline(night_result)
        img_day = self.pipeline(day_result)
        img_night = self.pipeline(night_result)
        data = dict(img_day=img_day['img'].data,
                    img_day_metas=img_day['img_metas'].data,
                    img_night=img_night['img'].data,
                    img_night_metas=img_night['img_metas'].data)

        return data

    def __len__(self):
        return self._ori_len * self.repeat_times
