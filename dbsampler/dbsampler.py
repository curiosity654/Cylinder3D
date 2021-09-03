import numpy as np
import open3d as o3d
import mmcv
import yaml
import os

class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]

class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 classes=None,):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.classes = classes
        # self.cat2label = {name: i for i, name in enumerate(classes)}
        # self.label2cat = {i: name for i, name in enumerate(classes)}

        db_infos = mmcv.load(info_path)

        # filter database infos
        for k, v in db_infos.items():
            print(f'load {len(v)} {k} database infos')
        print('After filter database:')
        for k, v in db_infos.items():
            print(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        self.group_db_infos = self.db_infos  # just use db_infos

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_class(self, points, labels, sample_dict, cat2id, avoid_class):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """

        for class_name, sampled_num in sample_dict.items():
            if sampled_num > 0:
                sampled = self.sampler_dict[class_name].sample(sampled_num)
            
            for object_info in sampled:
                points_vec = o3d.utility.Vector3dVector(points)
                boxpoints = object_info['box3d_lidar']
                boxpoints_vec = o3d.utility.Vector3dVector(boxpoints)
                bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxpoints_vec)

                ind = bbox.get_point_indices_within_bounding_box(points_vec)
                labels_in_box = labels[ind]
                unq_labels = np.unique(labels_in_box)

                for l in unq_labels:
                    if l in avoid_class:
                        continue

                # filtered_points = np.delete(points, ind, axis=0)
                # filtered_labels = np.delete(labels, ind, axis=0)
                new_points = np.fromfile(os.path.join(self.data_root, 'bin', object_info['filename'])).reshape(-1,3)
                new_labels = np.expand_dims(np.array([cat2id[class_name]]*len(new_points)), axis=1)

                points = np.concatenate([points, new_points], axis=0)
                labels = np.concatenate([labels, new_labels], axis=0)
                # print(object_info['filename'])

        return points, labels

    def sample_class_batch(self, points, labels, sample_dict, cat2id, avoid_class):
        new_points = []
        new_labels = []
        for i in range(len(points)):
            point, label = self.sample_class(points[i], labels[i], sample_dict, cat2id, avoid_class)
            new_points.append(point)
            new_labels.append(label)
        return new_points, new_labels

def load_data(path, map):
    raw_data = np.fromfile(path, dtype=np.float32).reshape((-1, 4))[:,:3]
    annotated_data = np.fromfile(path.replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32).reshape((-1, 1))
    annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
    annotated_data = np.vectorize(map.__getitem__)(annotated_data).squeeze()

    return raw_data, annotated_data

if __name__ == "__main__":
    info_path = "../data/database/semantickitti_gt_database.pkl"
    database_root = "../data/database"
    data_path = "../data/dataset/08/velodyne/000000.bin"
    label_mapping = "../data/dataset/label-mapping.yaml"
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
        learning_map = semkittiyaml['learning_map']

    points, labels = load_data(data_path, learning_map)
    sampler = DataBaseSampler(info_path, database_root)

    sample_dict = {'car': 1, "bicycle": 1, "motorcycle": 1, 'person': 1, 'bicyclist': 1, 'motorcyclist': 1}
    cat2id = {'car': 1, 'bicycle': 2, 'motorcycle': 3, 'person': 6, 'bicyclist': 7, 'motorcyclist': 8}
    avoid_class = np.array([1,2,3,4,5,6,7,8,13,14,16,18,19])
    
    new_points, new_labels = sampler.sample_class(points, labels, sample_dict, cat2id, avoid_class)