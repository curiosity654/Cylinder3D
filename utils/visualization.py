import numpy as np
import os
import os.path as osp

PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
    ]

def show(points, gt_seg, pred_seg, out_dir, file_name, ignore_index=0):
    """Results visualization.

    Args:
        results (list[dict]): List of bounding boxes results.
        out_dir (str): Output directory of visualization result.
        show (bool): Visualize the results online.
        pipeline (list[dict], optional): raw data loading for showing.
            Default: None.
    """
    assert out_dir is not None, 'Expect out_dir, got none.'
    points = points.numpy()
    gt_sem_mask = gt_seg.numpy()
    pred_sem_mask = pred_seg
    show_seg_result(points, gt_sem_mask,
                    pred_sem_mask, out_dir, file_name,
                    palette=np.array(PALETTE), ignore_index=ignore_index, norm_color=True)

def show_seg_result(points,
                    gt_seg,
                    pred_seg,
                    out_dir,
                    filename,
                    palette,
                    ignore_index=None,
                    show=False,
                    snapshot=False,
                    norm_color=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_seg (np.ndarray): Ground truth segmentation mask.
        pred_seg (np.ndarray): Predicted segmentation mask.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        palette (np.ndarray): Mapping between class labels and colors.
        ignore_index (int, optional): The label index to be ignored, e.g. \
            unannotated points. Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results. \
            Defaults to False.
    """
    # we need 3D coordinates to visualize segmentation mask
    if gt_seg is not None or pred_seg is not None:
        assert points is not None, \
            '3D coordinates are required for segmentation visualization'

    # filter out ignored points
    if gt_seg is not None and ignore_index is not None:
        if points is not None:
            points = points[gt_seg != ignore_index]
        if pred_seg is not None:
            pred_seg = pred_seg[gt_seg != ignore_index]
        gt_seg = gt_seg[gt_seg != ignore_index]

    if gt_seg is not None:
        gt_seg_color = palette[gt_seg]
        if norm_color:
            gt_seg_color = np.array(gt_seg_color, dtype='float32')/255
        gt_seg_color = np.concatenate([points[:, :3], gt_seg_color], axis=1)
    if pred_seg is not None:
        pred_seg_color = palette[pred_seg]
        if norm_color:
            pred_seg_color = np.array(pred_seg_color, dtype='float32')/255
        pred_seg_color = np.concatenate([points[:, :3], pred_seg_color],
                                        axis=1)

    result_path = osp.join(out_dir, filename)
    mkdir_or_exist(result_path)

    # online visualization of segmentation mask
    # we show three masks in a row, scene_points, gt_mask, pred_mask
    # if show:
    #     from .open3d_vis import Visualizer
    #     mode = 'xyzrgb' if points.shape[1] == 6 else 'xyz'
    #     vis = Visualizer(points, mode=mode)
    #     if gt_seg is not None:
    #         vis.add_seg_mask(gt_seg_color)
    #     if pred_seg is not None:
    #         vis.add_seg_mask(pred_seg_color)
    #     show_path = osp.join(result_path,
    #                          f'{filename}_online.png') if snapshot else None
    #     vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_seg is not None:
        _write_obj(gt_seg_color, osp.join(result_path, f'{filename}_gt.obj'), norm_color=norm_color)

    if pred_seg is not None:
        _write_obj(pred_seg_color, osp.join(result_path, f'{filename}_pred.obj'), norm_color=norm_color)

def _write_obj(points, out_filename, norm_color=False):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            if norm_color:
                c = points[i, 3:]
                fout.write(
                    'v %f %f %f %f %f %f\n' %
                    (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
            else:
                c = points[i, 3:].astype(int)
                fout.write(
                    'v %f %f %f %d %d %d\n' %
                    (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)