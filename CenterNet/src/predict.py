from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cv2 import data
from torch.functional import split

import _init_paths

import os
import json
import glob
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

class PrefetchDataset(torch.utils.data.Dataset):
  class_name = [
        '__background__', 
        "No entry",
        "No parking/waiting",
        "No turning",
        "Max speed",
        "Other prohibition signs",
        "Warning",
        "Mandatory",]

  _valid_ids = [1, 2, 3, 4, 5, 6, 7]
  def __init__(self, opt, data_dir, pre_process_func):
    self.images = glob.glob(os.path.join(data_dir, '*.*'))
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_path = self.images[index]
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ctdet':
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    image_id = os.path.basename(img_path).split('.')[0]
    return str(image_id), {'images': images, 'image': image, 'meta': meta}


  def __len__(self):
    return len(self.images)

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": str(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.5f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def save_results(self, results, save_dir, split):
    json.dump(self.convert_eval_format(results), 
                open('{}/results_{}_crop.json'.format(save_dir, split), 'w'), indent=4)

  def post_process(self, save_dir, split):
    results = json.load(open('{}/results_{}_crop.json'.format(save_dir, split), 'r'))
    list_of_dict = []
    for d in results:
      flag, img_id = d["image_id"].split('_')
      center = 811
      delta = center // 2
      if  flag == '2':
        d['bbox'][0] += 811
        d['image_id'] = int(img_id)
      elif flag == '1':
        d['image_id'] = int(img_id)
      elif flag == '3':
        d['bbox'][0] += delta
        d['image_id'] = int(img_id)
      if d['score'] < 0.0075:
        continue
      list_of_dict.append(d)
    json.dump(list_of_dict, open(os.path.join(self.opt.save_dir, 'submission.json').format(split), 'w'))

  def draw_label(self, boxes, image, img_id):
    thresh = 0.3
    
    for cls_ind in boxes:
      category_id = self._valid_ids[cls_ind - 1]
      cls = self.class_name[category_id]
      for bbox in boxes[cls_ind]:
      
        score = bbox[4]
        if score > thresh:
          cv2.putText(image, cls, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), lineType=2)
          cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    cv2.imwrite('{}/{}.jpg'.format(self.opt.result_crop_dir, img_id), image)
    
  def get_flag(self, img_dir):
    return os.path.basename(img_dir).split('.')[0].split('_')[1] + os.path.basename(img_dir).split('_')[0]
  
  def get_img_id(self, img_dir):
    return os.path.basename(img_dir).split('.')[0].split('_')[1]
    
  def merge_crop_image(self, crop_image_dir):

    list_crop_imgs_path = glob.glob(os.path.join(crop_image_dir, '*.*'))
    list_crop_imgs_path = sorted(list_crop_imgs_path, key=self.get_flag)
    
    for i in range(len(list_crop_imgs_path)//2):
      img1 = cv2.imread(list_crop_imgs_path[2*i])
      img2 = cv2.imread(list_crop_imgs_path[2*i+1])
      img = np.concatenate((img1, img2), axis=1)
      cv2.imwrite('{}/{}.png'.format(self.opt.results_image_dir, self.get_img_id(list_crop_imgs_path[2*i])), img)
    
def pre_process(IMAGES_FOLDER, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    ls_img = glob.glob(os.path.join(IMAGES_FOLDER, '*.*'))
    print('Before preprocess:\t ', len(ls_img))
    for path in ls_img:
        filename = os.path.basename(path)
        ori_img = cv2.imread(path)
        ori_h, ori_w = ori_img.shape[:2]
        center = ori_w // 2
        left = center // 2
        right = left + center
        file1 = '1_' + filename
        file2 = '2_' + filename
        
        cv2.imwrite(os.path.join(dst, file1), ori_img[:, :center])
        cv2.imwrite(os.path.join(dst, file2), ori_img[:, center:])

    print('After preprocess:\t ', len(os.listdir(dst)))

def prefetch_test(opt):

  IMAGES_FOLDER = opt.data_dir
  DST_IMAGES = os.path.join(opt.data_dir, 'croped')
  os.makedirs(DST_IMAGES, exist_ok=True)
  
  pre_process(IMAGES_FOLDER, DST_IMAGES)

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'test'
  data_dir = os.path.join(DST_IMAGES)
  print(data_dir)
  detector = Detector(opt)
  dataset = PrefetchDataset(opt, data_dir, detector.pre_process)
  print('loading {} samples for testing'.format(len(dataset)))
  data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.avg:.3f}s '.format(
        t, tm = avg_time_stats[t])
    bar.next()
    dataset.draw_label(ret['results'], pre_processed_images['image'].numpy()[0],
                       img_id[0])
  bar.finish()
  
  dataset.save_results(results, opt.save_dir, split)
  dataset.post_process(opt.save_dir, split)
  dataset.merge_crop_image(opt.result_crop_dir)
if __name__ == '__main__':
  opt = opts().parse()
  prefetch_test(opt)
