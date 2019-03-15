# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo
import glob
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import models
import moviepy.editor as mpe
from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig
import colorsys
from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str
import pickle
thrs = np.arange(0.3, 0.5, 0.05)

model_zoo = sorted(name for name in models.__dict__
            if not name.startswith("__")
            and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=model_zoo + ['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2018', 
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')

def create_unique_color_float(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b

def create_unique_color_uchar(tag, hue_step=0.41):
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)

def draw_rectangle(image, color, min_x, min_y, max_x, max_y, thickness=2, label=None):
    pt1 = int(min_x), int(min_y)
    pt2 = int(max_x), int(max_y)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
            text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), thickness)

def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
    return bbox

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(im, target_pos, target_sz, model, hp=None):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)

    p.renew()

    net = model
    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = len(p.ratios) * len(p.scales)
    p.anchor = generate_anchor(model.anchors, p.score_size)

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.template(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


def siamese_track(state, im, mask_enable=False, refine_enable=False):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    if mask_enable:
        score, delta, mask = net.track_mask(x_crop.cuda())
    else:
        score, delta = net.track(x_crop.cuda())

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_sz*scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    # for Mask Branch
    if mask_enable:
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        if refine_enable:
            mask = net.track_refine((delta_y, delta_x)).cuda().sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop

        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))

        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            # box_in_img = pbox
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    state['mask'] = mask_in_img if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []
    state['best_pscore'] = score[best_pscore_id]
    bbox = bbox1(target_mask)
    state['bbox'] = bbox
    return state


def track_vot(model, video, hp=None, mask_enable=False, refine_enable=False):
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']

    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, model, hp)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            state = siamese_track(state, im, mask_enable, refine_enable)  # track
            if mask_enable:
                location = state['ploygon'].flatten()
                mask = state['mask']
            else:
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                mask = []

            if 'VOT' in args.dataset:
                gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                              (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
                if mask_enable:
                    pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                    (location[4], location[5]), (location[6], location[7]))
                else:
                    pred_polygon = ((location[0], location[1]),
                                    (location[0] + location[2], location[1]),
                                    (location[0] + location[2], location[1] + location[3]),
                                    (location[0], location[1] + location[3]))
                b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
            else:
                b_overlap = 1

            if b_overlap:
                regions.append(location)
            else:  # lost
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()
            if gt.shape[0] > f:
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                if mask_enable:
                    mask = mask > state['p'].seg_thr
                    im_show[:, :, 2] = mask * 255 + (1 - mask) * im_show[:, :, 2]
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save result
    name = args.arch.split('.')[0] + '_' + ('mask_' if mask_enable else '') + ('refine_' if refine_enable else '') +\
           args.resume.split('/')[-1].split('.')[0]

    if 'VOT' in args.dataset:
        video_path = join('test', args.dataset, name,
                          'baseline', video['name'])
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                        fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
    else:  # OTB
        video_path = join('test', args.dataset, name)
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x])+'\n')

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
        v_id, video['name'], toc, f / toc, lost_times))

    return lost_times, f / toc


def MultiBatchIouMeter(thrs, outputs, targets, start=None, end=None):
    targets = np.array(targets)
    outputs = np.array(outputs)

    num_frame = targets.shape[0]
    if start is None:
        object_ids = np.array(list(range(outputs.shape[0]))) + 1
    else:
        object_ids = [int(id) for id in start]

    num_object = len(object_ids)
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)

    output_max_id = np.argmax(outputs, axis=0).astype('uint8')+1
    outputs_max = np.max(outputs, axis=0)
    for k, thr in enumerate(thrs):
        output_thr = outputs_max > thr
        for j in range(num_object):
            target_j = targets == object_ids[j]

            if start is None:
                start_frame, end_frame = 1, num_frame - 1
            else:
                start_frame, end_frame = start[str(object_ids[j])] + 1, end[str(object_ids[j])] - 1
            iou = []
            for i in range(start_frame, end_frame):
                pred = (output_thr[i] * output_max_id[i]) == (j+1)
                mask_sum = (pred == 1).astype(np.uint8) + (target_j[i] > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            res[j, k] = np.mean(iou)
    return res


def track_vos(model, video, hp=None, mask_enable=False, refine_enable=False, mot_enable=False, is_av=False):
    if is_av:
        start_image_name = {
            1: '1539644288208_2e0c2683ca37b7bc60a73ed4a5e56c07.jpeg',
            2: '1539644288208_2e0c2683ca37b7bc60a73ed4a5e56c07.jpeg',
            3: '1539644288208_2e0c2683ca37b7bc60a73ed4a5e56c07.jpeg',
            7: '1539644288208_2e0c2683ca37b7bc60a73ed4a5e56c07.jpeg',
            8: '1539644288306_fe57e9c4532ef666a09e3f739f5a5453.jpeg',
            11: '1539644288506_9eb46d04b55095cf1e46e284ebd9dc74.jpeg',
            15: '1539644288806_a60c70392fe388ba65149af1f05c82f1.jpeg',
            25: '1539644289508_bf85b246ffb2b5422daf92b198b4962f.jpeg',
            31: '1539644290307_c887872be7aad089acd43d379522b2e0.jpeg',
            36: '1539644290707_e82a8e9bc300ed70ba295209fa630b18.jpeg',
            46: '1539644291205_4c93de9a25a05ad66a2c56acdba2a597.jpeg',
            64: '1539644292205_5dd64ec5619d051218fb9ef5d0dc66c5.jpeg',
            70: '1539644292806_a6c3f0e2227a7aa69121e0fa2d9c8204.jpeg',
            81: '1539644295609_a2d13da21de24aa67e2937e9d6e9d578.jpeg',
            85: '1539644297208_c86e5aa24eeb1bd019a9b8ccadd20391.jpeg',
            88: '1539644297906_6ef4e56500f601420d9378a93fb82702.jpeg',
            95: '1539644300108_4634964ecba62e723641a86e9d6acbca.jpeg',
            98: '1539644301204_39be9d8c63f655563163d53077b31ee6.jpeg',
            102: '1539644302106_67579806270edff237d28babe618f1c1.jpeg'
        }
#         end_image_name = {
#             1: '1539644303205_af37301f17a8339c9e7d9a23840e43ce.jpeg',
#             2: '1539644303107_7966554f2bdabc5e829d724f2c2b196c.jpeg',
#             3: '1539644290407_e2f74e04fb17ce283b8a0a8cff9ea1de.jpeg',
#             7: '1539644302304_20bf85e62420d22f70f02e3dad4d01bb.jpeg'
#         }
        video = {'name': 'av_15s', 'start_image_name': start_image_name}
        image_files = sorted(glob.glob('/mnt/user-home/SiamMask/experiments/siammask/av_data/images/*.jpeg'))
        image_names = list(map(lambda x : x.split('/')[-1], image_files))
        image_names_dict = {}
        i = 0
        for image_name in image_names:
            image_names_dict[image_name] = i
            i += 1
        object_ids = [1,2,3,7,8,11,15,25,31,36,46,64,70,81,85,88,95,98,102]
        object_num = len(object_ids)
        pred_masks = {}
        # pred_masks = np.zeros((len(image_files), 1024, 1224)) - 1
        init_bbox = [[488, 552, 551, 596], [716, 549, 820, 625], [320, 557, 359, 584], [680, 543, 737, 594], [573, 547, 612, 573],[717, 545, 796, 607], [481, 540, 546, 567], [237, 549, 304, 575], [39, 518, 189, 579], [755, 536, 824, 562], [824, 532, 836, 569], [928, 527, 954, 593], [1051, 527, 1098, 638], [5, 555, 269, 855], [462, 541, 505, 568], [236, 551, 308, 588], [90, 556, 214, 606], [423, 547, 477, 577], [1186, 512, 1223, 675]]

        # locations_objs = []
        all_bboxes = []
        next_track_id = 1
        alive_tracks = {}
        dead_tracks = {}
        seg_thr = 0.35 # for DAVIS

        toc = 0

        def match_IOU(bboxes, target):
            for track_id, bbox in bboxes:
                iou = get_iou({'x1': bbox[0], 'x2': bbox[2], 'y1': bbox[1], 'y2': bbox[3]},
                        {'x1': target[0], 'x2': target[2], 'y1': target[1], 'y2': target[3]})
                if iou > 0.4:
                    return True
            return False

        with open('detectron_result_15s', 'rb') as fp:
            detectron_result = pickle.load(fp)
        detectron_result = list(map(lambda r: r['bboxes'], detectron_result))

        for f, image_file in enumerate(image_files):
            image = cv2.imread(image_file)
            detections = detectron_result[f]

            current_bboxes = []
            all_bboxes.append(current_bboxes)
            to_delete = []
            # update current tracks
            for track_id, states in alive_tracks.items():
                tic = cv2.getTickCount()
                new_state = siamese_track(states[-1][1], image, mask_enable, refine_enable)  # track
                toc += cv2.getTickCount() - tic
                mask = new_state['mask']
                if new_state['best_pscore'] < .80:
                    print("track: {} stops at frame: {:3d}, score: {:3.4f}".format(track_id, f, new_state['best_pscore']))
                    to_delete.append(track_id)
                    dead_tracks[track_id] = states
                else:
                    print("track: {}, frame: {:3d}, score: {:3.4f}".format(track_id, f, new_state['best_pscore']))
                    # location = state['ploygon'].flatten()
                    # locations[f] = location
                    states.append((f, new_state))
                    current_bboxes.append((track_id, new_state['bbox']))
                    if track_id in pred_masks:
                        pred_mask = pred_masks[track_id]
                    else:
                        pred_mask = np.zeros((len(image_files), 1024, 1224)) - 1
                    pred_mask[f, :, :] = mask
                    pred_masks[track_id] = pred_mask
            for track_id in to_delete:
                del alive_tracks[track_id]
            # new tracks
            for detection in detections:
                bbox = detection
                if not match_IOU(current_bboxes, bbox):
                    track_id = next_track_id
                    current_bboxes.append((track_id, bbox))

                    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    target_pos = np.array([cx, cy])
                    target_sz = np.array([w, h])
                    tic = cv2.getTickCount()
                    new_state = siamese_init(image, target_pos, target_sz, model, hp)  # init tracker
                    toc += cv2.getTickCount() - tic
                    new_states = [(f, new_state)]
                    alive_tracks[next_track_id] = new_states
                    next_track_id += 1

        toc /= cv2.getTickFrequency()

        video_path = join('test', args.dataset, 'vis', video['name'])
        if not isdir(video_path): makedirs(video_path)
        num_tracks = next_track_id - 1
        pred_mask_final = np.array([pred_masks[track_id] for track_id in range(1, num_tracks + 1)])
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > seg_thr).astype('uint8')
        colors = []
        for track_id in range(1, num_tracks + 1):
            color = create_unique_color_uchar(track_id)
            colors.append(color)
        colors = np.array(colors)
        # COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        colors = np.vstack([[0, 0, 0], colors]).astype("uint8")
        mask = colors[pred_mask_final]

        frames = []
        for f, image_file in tqdm(enumerate(image_files)):
            output = ((0.4 * cv2.imread(image_file)) + (0.6 * mask[f, :, :, :])).astype("uint8")
            pp = join(video_path, image_file.split('/')[-1].split('.')[0] + '.png')
            # if f > 0:
            for track_id, locations in all_bboxes[f]:
                location = locations[f]
                color = create_unique_color_uchar(track_id)
                draw_rectangle(output, color, *location, label=str(track_id))
            cv2.imwrite(pp, output)
            copy_output = np.ascontiguousarray(np.copy(output), dtype=np.uint8)
            frames.append(copy_output)
            # cv2.imshow("mask", output)
            cv2.waitKey(1)

        clip = mpe.ImageSequenceClip(frames, fps=10)
        clip.write_videofile(join(video_path, video['name'] + '.avi'), codec='png')

        multi_mean_iou = []
        object_ids = range(1, num_tracks + 1)

    else:
        image_files = video['image_files']

        annos = [np.array(Image.open(x)) for x in video['anno_files']]
        if 'anno_init_files' in video:
            annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
        else:
            annos_init = [annos[0]]

        if not mot_enable:
            annos = [(anno > 0).astype(np.uint8) for anno in annos]
            annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

        if 'start_frame' in video:
            object_ids = [int(id) for id in video['start_frame']]
        else:
            object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
            if len(object_ids) != len(annos_init):
                annos_init = annos_init * len(object_ids)
        object_num = len(object_ids)
        pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1])) - 1

        toc = 0
        locations_objs = []
        for obj_id, o_id in enumerate(object_ids):
            mask = None
            locations = {}
            locations_objs.append((o_id, locations))
            if 'start_frame' in video:
                start_frame = video['start_frame'][str(o_id)]
                end_frame = video['end_frame'][str(o_id)]
            elif 'start_image_name' in video:
                # start_image_name = video['start_image_name'][o_id]
    #             end_image_name = video['end_image_name'][o_id]
    #             start_frame = image_names_dict[start_image_name]
    #             end_frame = image_names_dict[end_image_name]
                end_frame = len(image_files)
            else:
                start_frame, end_frame = 0, len(image_files)

            track_stop = False
            for f, image_file in enumerate(image_files):
                tic = cv2.getTickCount()
                if f == start_frame:  # init
                    im = cv2.imread(image_file)
                    # if is_av:
                    #     bbox = init_bbox[obj_id]
                    #     cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    #     w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    #     target_pos = np.array([cx, cy])
                    #     target_sz = np.array([w, h])
                    #     state = siamese_init(im, target_pos, target_sz, model, hp)  # init tracker
                    # else:
                    mask = annos_init[obj_id] == o_id
                    x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                    cx, cy = x + w / 2, y + h / 2
                    target_pos = np.array([cx, cy])
                    target_sz = np.array([w, h])
                    state = siamese_init(im, target_pos, target_sz, model, hp)  # init tracker
                    #                 location = state['ploygon'].flatten()
                elif end_frame >= f > start_frame:  # tracking
                    if not track_stop:
                        im = cv2.imread(image_file)
                        state = siamese_track(state, im, mask_enable, refine_enable)  # track
                        mask = state['mask']
                        if state['best_pscore'] < .80:
                            print("track: {} stops at frame: {:3d}, score: {:3.4f}".format(o_id, f, state['best_pscore']))
                            track_stop = True
                        else:
                            print("track: {}, frame: {:3d}, score: {:3.4f}".format(o_id, f, state['best_pscore']))
                            # location = state['ploygon'].flatten()
                            # locations[f] = location
                            locations[f] = state['bbox']
                toc += cv2.getTickCount() - tic
                if end_frame >= f > start_frame and not track_stop:
                    pred_masks[obj_id, f, :, :] = mask
        toc /= cv2.getTickFrequency()

    #     if len(annos) == len(image_files):
    #         multi_mean_iou = MultiBatchIouMeter(thrs, pred_masks, annos,
    #                                             start=video['start_frame'] if 'start_frame' in video else None,
    #                                             end=video['end_frame'] if 'end_frame' in video else None)
    #         for i in range(object_num):
    #             for j, thr in enumerate(thrs):
    #                 logger.info('Fusion Multi Object{:20s} IOU at {:.2f}: {:.4f}'.format(video['name'] + '_' + str(i + 1), thr,
    #                                                         multi_mean_iou[i, j]))
    #     else:
    #         multi_mean_iou = []
        multi_mean_iou = []

    # if False:
    #     video_path = join('test', args.dataset, 'SiamMask', video['name'])
    #     if not isdir(video_path): makedirs(video_path)
    #     pred_mask_final = np.array(pred_masks)
    #     pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
    #             np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
    #     for i in range(pred_mask_final.shape[0]):
    #         pp = join(video_path, image_files[i].split('/')[-1].split('.')[0] + '.png')
    #         cv2.imwrite(pp, pred_mask_final[i].astype(np.uint8))

        if True:
            video_path = join('test', args.dataset, 'vis', video['name'])
            if not isdir(video_path): makedirs(video_path)
            pred_mask_final = np.array(pred_masks)
            pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                    np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
            colors = []
            for track_id in object_ids:
                color = create_unique_color_uchar(track_id)
                colors.append(color)
            colors = np.array(colors)
            # COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
            colors = np.vstack([[0, 0, 0], colors]).astype("uint8")
            mask = colors[pred_mask_final]

            frames = []
            for f, image_file in tqdm(enumerate(image_files)):
                output = ((0.4 * cv2.imread(image_file)) + (0.6 * mask[f, :, :, :])).astype("uint8")
                pp = join(video_path, image_file.split('/')[-1].split('.')[0] + '.png')
                # if f > 0:
                for track_id, locations in locations_objs:
                    if f in locations:
                        location = locations[f]
                        color = create_unique_color_uchar(track_id)
                        # cv2.polylines(output, [np.int0(location).reshape((-1, 1, 2))], True, color, 3)
    #                         xmin = xmax = location[0]
    #                         ymin = ymax = location[1]
    #                         for i in range(2, 8, 2):
    #                             if location[i] > xmax:
    #                                 xmax = location[i]
    #                             if location[i] < xmin:
    #                                 xmin = location[i]
    #                         for i in range(3, 8, 2):
    #                             if location[i] > ymax:
    #                                 ymax = location[i]
    #                             if location[i] < ymin:
    #                                 ymin = location[i]
                        draw_rectangle(output, color, *location, label=str(track_id))
                cv2.imwrite(pp, output)
                copy_output = np.ascontiguousarray(np.copy(output), dtype=np.uint8)
                frames.append(copy_output)
                # cv2.imshow("mask", output)
                cv2.waitKey(1)

            clip = mpe.ImageSequenceClip(frames, fps=10)
            clip.write_videofile(join(video_path, video['name'] + '.avi'), codec='png')

        logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
            v_id, video['name'], toc, f * len(object_ids) / toc))

    return multi_mean_iou, f * len(object_ids) / toc


def main():
    global args, logger, v_id
    args = parser.parse_args()
    cfg = load_config(args)

    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)

    # setup model
    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(anchors=cfg['anchors'])
    else:
        model = models.__dict__[args.arch](anchors=cfg['anchors'])

    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)
    model.eval()
    model = model.cuda()

    # setup dataset
    dataset = load_dataset(args.dataset)

    # VOS or VOT?
    if args.dataset in ['DAVIS2016', 'DAVIS2017', 'ytb_vos'] and args.mask:
        vos_enable = True  # enable Mask output
    else:
        vos_enable = False

    total_lost = 0  # VOT
    iou_lists = []  # VOS
    speed_list = []

    for v_id, video in enumerate(dataset.keys(), start=1):
        if vos_enable:
            iou_list, speed = track_vos(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                                 args.mask, args.refine, args.dataset in ['DAVIS2017', 'ytb_vos'], is_av=True)
            iou_lists.append(iou_list)
            speed_list.append(speed)
            break
        else:
            lost, speed = track_vot(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                             args.mask, args.refine)
            total_lost += lost
        speed_list.append(speed)

    # report final result
    if vos_enable:
        for thr, iou in zip(thrs, np.mean(np.concatenate(iou_lists), axis=0)):
            logger.info('Segmentation Threshold {:.2f} mIoU: {:.3f}'.format(thr, iou))
    else:
        logger.info('Total Lost: {:d}'.format(total_lost))

    logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))


if __name__ == '__main__':
    main()
