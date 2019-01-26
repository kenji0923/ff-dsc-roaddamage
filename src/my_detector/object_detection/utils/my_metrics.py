from abc import ABCMeta
from abc import abstractmethod
import collections
import logging
import unicodedata
import numpy as np
import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.utils import label_map_util
from object_detection.utils import metrics
from object_detection.utils import per_image_evaluation
from object_detection.utils import object_detection_evaluation
from object_detection.utils import json_utils

def get_f(precision, recall):
  return 2 * precision * recall / (precision + recall)

class MyEvaluator(object_detection_evaluation.ObjectDetectionEvaluator):
  """A class to evaluate detections."""

  def __init__(self,
               categories,
               export_path='',
               matching_iou_threshold=0.5,
               evaluate_corlocs=False,
               evaluate_precision_recall=False,
               metric_prefix=None,
               use_weighted_mean_ap=False,
               evaluate_masks=False,
               group_of_weight=0.0):

    super(MyEvaluator, self).__init__(
               categories,
               matching_iou_threshold=0.5,
               evaluate_corlocs=False,
               evaluate_precision_recall=False,
               metric_prefix=None,
               use_weighted_mean_ap=False,
               evaluate_masks=False,
               group_of_weight=0.0)

    self.b_export = False
    if export_path and not export_path =='':
        self.b_export = True
        self.export_path = export_path

    self.initialize_my_vars()

  def initialize_my_vars(self):
    self.fs_per_class = [np.nan] * self._evaluation.num_class
    self.detection_boxes_list = {}

  def clear(self):
    super(MyEvaluator, self).clear()
    self.initialize_my_vars()

  def _build_metric_names(self):
    """Builds a list with metric names."""

    self._metric_names = [
        self._metric_prefix + 'Precision@{}IOU'.format(
            self._matching_iou_threshold),
        self._metric_prefix + 'Recall@{}IOU'.format(
            self._matching_iou_threshold),
        self._metric_prefix + 'F@{}IOU'.format(
            self._matching_iou_threshold),
    ]

    category_index = label_map_util.create_category_index(self._categories)
    for idx in range(self._num_classes):
      if idx + self._label_id_offset in category_index:
        category_name = category_index[idx + self._label_id_offset]['name']
        try:
          category_name = unicode(category_name, 'utf-8')
        except TypeError:
          pass
        category_name = unicodedata.normalize('NFKD', category_name).encode(
            'ascii', 'ignore')
        self._metric_names.append(
            self._metric_prefix + 'Precision@{}IOU/PerformanceByCategory/{}'.format(
                self._matching_iou_threshold, category_name))
        self._metric_names.append(
            self._metric_prefix + 'Recall@{}IOU/PerformanceByCategory/{}'.format(
                self._matching_iou_threshold, category_name))
        self._metric_names.append(
            self._metric_prefix + 'F@{}IOU/PerformanceByCategory/{}'.format(
                self._matching_iou_threshold, category_name))

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      A dictionary of metrics with the following fields -

      1. summary_metrics:
        'Precision@<matching_iou_threshold>IOU': for all class
        'Recall@<matching_iou_threshold>IOU': for all class
        'F@<matching_iou_threshold>IOU': for all class

      2. per_category_f: category specific results with keys of the form
        'PerformanceByCategory/
        f@<matching_iou_threshold>IOU/category'.
    """

    if (self._evaluation.num_gt_instances_per_class == 0).any():
      logging.warn(
          'The following classes have no ground truth examples: %s',
          np.squeeze(np.argwhere(self._evaluation.num_gt_instances_per_class == 0)) +
          self._evaluation.label_id_offset)

    if self._evaluation.use_weighted_mean_ap:
      all_scores = np.array([], dtype=float)
      all_tp_fp_labels = np.array([], dtype=bool)

    allclass_num_gt = 0
    allclass_num_tp = 0
    allclass_num_fp = 0

    for class_index in range(self._evaluation.num_class):
      allclass_num_gt += self._evaluation.num_gt_instances_per_class[class_index]

      if not self._evaluation.scores_per_class[class_index]:
        scores = np.array([], dtype=float)
        tp_fp_labels = np.array([], dtype=float)
      else:
        scores = np.concatenate(self._evaluation.scores_per_class[class_index])
        tp_fp_labels = np.concatenate(self._evaluation.tp_fp_labels_per_class[class_index])

      sorted_indices = np.argsort(scores)
      sorted_indices = sorted_indices[::-1]
      scores = scores[sorted_indices]
      n_detected = 0
      for index, score in enumerate(scores):
        if score > 0.5: 
          n_detected += 1
        else:
          break
      scores = scores[0:n_detected]
      tp_fp_labels = tp_fp_labels[sorted_indices][0:n_detected]

      tp_labels = tp_fp_labels.astype(float)
      fp_labels = (tp_fp_labels <= 0).astype(float)

      if len(scores) > 0:
        num_tp = np.cumsum(tp_labels)[-1]
        num_fp = np.cumsum(fp_labels)[-1]
      else:
        num_tp = 0
        num_fp = 0

      allclass_num_tp += num_tp
      allclass_num_fp += num_fp

      if self._evaluation.num_gt_instances_per_class[class_index] == 0:
        precision = 0
        recall = num_fp
        f = 0

      else:
        precision, recall = metrics.compute_precision_recall(
          scores, tp_fp_labels, self._evaluation.num_gt_instances_per_class[class_index])

        if len(precision) > 0:
          precision = precision[-1]
          recall = recall[-1]
          f = get_f(precision, recall)
        else:
          precision = 0
          recall = 0
          f = 0

      self._evaluation.precisions_per_class[class_index] = precision
      self._evaluation.recalls_per_class[class_index] = recall
      self.fs_per_class[class_index] = f

    allclass_precision = allclass_num_tp / (allclass_num_tp + allclass_num_fp)
    allclass_recall = allclass_num_tp / allclass_num_gt
    allclass_f = get_f(allclass_precision, allclass_recall)

    my_metrics = {
            self._metric_names[0]: allclass_precision,
            self._metric_names[1]: allclass_recall,
            self._metric_names[2]: allclass_f
            }

    category_index = label_map_util.create_category_index(self._categories)

    for idx in range(self._evaluation.num_class):
      if idx + self._label_id_offset in category_index:
        category_name = category_index[idx + self._label_id_offset]['name']
        try:
          category_name = unicode(category_name, 'utf-8')
        except TypeError:
          pass
        category_name = unicodedata.normalize(
            'NFKD', category_name).encode('ascii', 'ignore')

        display_name = ('Precision@{}IOU/PerformanceByCategory/{}'.format(
                self._matching_iou_threshold, category_name))
        my_metrics[display_name] = self._evaluation.precisions_per_class[idx]

        display_name = ('Recall@{}IOU/PerformanceByCategory/{}'.format(
                self._matching_iou_threshold, category_name))
        my_metrics[display_name] = self._evaluation.recalls_per_class[idx]

        display_name = ('F@{}IOU/PerformanceByCategory/{}'.format(
                self._matching_iou_threshold, category_name))
        my_metrics[display_name] = self.fs_per_class[idx]

    if self.b_export:
        self.dump_detections_to_json_file(self.export_path)

    return my_metrics

  def add_single_detected_image_info(self, image_id, detections_dict):
    out = super(MyEvaluator, self).add_single_detected_image_info(image_id, detections_dict)

    # if not standard_fields.InputDataFields.key in self.detection_boxes_list:
    #   self.detection_boxes_list[standard_fields.InputDataFields.key] = []
    # self.detection_boxes_list[standard_fields.InputDataFields.key].append(image_id)
    # tf.logging.warning(image_id)

    for key, value in detections_dict.items():
        if key in (standard_fields.InputDataFields.groundtruth_boxes, standard_fields.InputDataFields.groundtruth_classes):
            continue
        if not key in self.detection_boxes_list:
          self.detection_boxes_list[key] = []
        self.detection_boxes_list[key].append(value.tolist())

    return out

  def dump_detections_to_json_file(self, json_output_path):
    """Saves the detections into json_output_path

    Args:
      json_output_path: String containing the output file's path. It can be also
        None. In that case nothing will be written to the output file.
    """
    if json_output_path and json_output_path is not None:
      with tf.gfile.GFile(json_output_path, 'w') as fid:
        tf.logging.info('Dumping detections to output json file.')
        json_utils.Dump(
            obj=self.detection_boxes_list, fid=fid, float_digits=4, indent=2)
