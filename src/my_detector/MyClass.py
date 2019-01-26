import math
import tensorflow as tf
import re
import pprint
import numpy as np
import os
from distutils.version import StrictVersion
from PIL import Image
import json
import logging
import io
import base64
from tqdm import tqdm
from datetime import datetime
from dateutil.tz import tzlocal
import glob
import hashlib

from matplotlib import pyplot as plt

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import my_metrics
from object_detection.utils.per_image_evaluation import PerImageEvaluation
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops

# Variables
local_tz = tzlocal()
output_collections = {
    'num_detections': 'num_detections',
    'detection_scores' : 'detection_scores',
    'detection_boxes': 'detection_boxes',
    'detection_classes': 'detection_classes',
    'filenames': 'filenames',
    'source_id': 'source_id'
    }

# Helper functions
def create_example(xml_tree, full_path):
    root = xml_tree
    folder_name = root.find('folder').text
    image_name = root.find('filename').text
    file_name = image_name.encode('utf8')
    size=root.find('size')
    width = int(size[0].text)
    height = int(size[1].text)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if root.find('object'):
        for member in root.findall('object'):
            class_name = member.find('name').text
            classes_text.append(class_name.encode('utf8'))
            bndbox = member.find('bndbox')
            xmin.append(float(bndbox.find('xmin').text) / width)
            ymin.append(float(bndbox.find('ymin').text) / height)
            xmax.append(float(bndbox.find('xmax').text) / width)
            ymax.append(float(bndbox.find('ymax').text) / height)

            classes.append(int(class_name))
            truncated.append(int(0))
            difficult_obj.append(int(0))
            poses.append('Unspecified'.encode('utf8'))

    # Read corresponding images (JPEGImages folder)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
       raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # Create TFRecord
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    })) 
    return example  

def get_timestamp(dt):
    return dt.strftime('%m_%d_%Y_%H_%M_%S')

def load_image_from_tf_example(example):
    image_bytes = io.BytesIO(example.features.feature["image/encoded"].bytes_list.value[0])
    image = Image.open(image_bytes)
    return image, image_bytes

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

class MyObjectDetector:
    PROJECT_BASE_DIR = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    INPUT_BASE_DIR = os.path.join(PROJECT_BASE_DIR, 'input', 'train')
    TRAIN_BASE_DIR = os.path.join(PROJECT_BASE_DIR, 'training')
    PATH_TO_LABELS = os.path.join(TRAIN_BASE_DIR, 'road_label_map.pbtxt')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    num_classes = len(category_index)
    label_offset = 1

    def __init__(self, model_dir_name):
        self.MODEL_DIR = os.path.join(self.TRAIN_BASE_DIR, model_dir_name)
        self.MODEL_NAME = os.path.join(self.MODEL_DIR, 'export')

        self.PATH_TO_FROZEN_GRAPH = os.path.join(self.MODEL_NAME, 'frozen_inference_graph.pb')

        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
          raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

    def set_directory(self, project_base_dir):
        self.PROJECT_BASE_DIR = project_base_dir
        self.TRAIN_BASE_DIR = os.path.join(self.PROJECT_BASE_DIR, 'training')
        self.PATH_TO_LABELS = os.path.join(self.TRAIN_BASE_DIR, 'road_label_map.pbtxt')
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)

    def get_evaluation_dict(self, 
            result_filename, 
            test_filenames, 
            b_load_original_image=False,
            i_start=0, 
            num_draw=1):

        with open(result_filename) as fp:
            detections_dict = json.load(fp)

        if num_draw < 0:
            num_draw = len(detections_dict['source_id'])

        eval_dict = {
                'key': np.array(detections_dict['source_id'][i_start:num_draw]),
                'detection_scores': np.array(detections_dict['detection_scores'][i_start:num_draw]),
                'detection_classes': np.array(detections_dict['detection_classes'][i_start:num_draw]) + self.label_offset,
                'detection_boxes': np.array(detections_dict['detection_boxes'][i_start:num_draw]),
                'num_detections': np.array(detections_dict['num_detections'][i_start:num_draw]),
                }

        groundtruth_source_ids= []
        groundtruth_images = []
        groundtruth_image_sizes = []
        groundtruth_boxes = []
        groundtruth_classes = []

        for test_filename in test_filenames:
            record_strings = []
            for string_record in tf.python_io.tf_record_iterator(test_filename):
                record_strings.append(string_record)

            for string_record in tqdm(record_strings, desc='Load {} : '.format(os.path.basename(test_filename))):
                json_data = {}

                example = tf.train.Example()
                example.ParseFromString(string_record)

                image, image_bytes = load_image_from_tf_example(example)
                groundtruth_images.append(image)

                height = example.features.feature['image/height'].int64_list.value[0]
                width = example.features.feature['image/width'].int64_list.value[0]
                groundtruth_image_sizes.append((height, width))
                
                source_id = io.BytesIO(example.features.feature["image/source_id"].bytes_list.value[0])
                groundtruth_source_ids.append(source_id.getvalue().decode('utf-8'))

                groundtruth_class = np.array(example.features.feature['image/object/class/label'].int64_list.value, dtype=np.int32)
                groundtruth_classes.append(groundtruth_class)
                
                num_box = len(groundtruth_class)

                groundtruth_box = np.empty(shape=(num_box, 4))

                for i_box in range(num_box):
                    box = np.array([
                        example.features.feature['image/object/bbox/ymin'].float_list.value[i_box],
                        example.features.feature['image/object/bbox/xmin'].float_list.value[i_box],
                        example.features.feature['image/object/bbox/ymax'].float_list.value[i_box],
                        example.features.feature['image/object/bbox/xmax'].float_list.value[i_box]
                        ])
                    groundtruth_box[i_box] = box

                groundtruth_boxes.append(groundtruth_box)

        eval_dict['original_image'] = [0]*num_draw
        eval_dict['original_image_spatial_shape'] = [0]*num_draw
        eval_dict['true_image_shape'] = [0]*num_draw
        eval_dict['groundtruth_classes'] = [0]*num_draw
        eval_dict['groundtruth_boxes'] = [0]*num_draw
        for index, key in enumerate(tqdm(eval_dict['key'], desc='load groundtruth : ')):
            groundtruth_index = groundtruth_source_ids.index(key)

            eval_dict['original_image_spatial_shape'][index] = groundtruth_image_sizes[groundtruth_index]
            eval_dict['groundtruth_classes'][index] = groundtruth_classes[groundtruth_index]
            eval_dict['groundtruth_boxes'][index] = groundtruth_boxes[groundtruth_index]

            if b_load_original_image:
                image = groundtruth_images[groundtruth_index]
                image_np = load_image_into_numpy_array(image)
                eval_dict['original_image'][index] = image_np
                eval_dict['true_image_shape'][index] = np.array(image_np.shape).astype(np.int32)

        eval_dict['original_image'] = np.array(eval_dict['original_image'])
        eval_dict['original_image_spatial_shape'] = np.array(eval_dict['original_image_spatial_shape'])
        eval_dict['true_image_shape'] = np.array(eval_dict['true_image_shape'])
        eval_dict['groundtruth_classes'] = np.array(eval_dict['groundtruth_classes'])
        eval_dict['groundtruth_boxes'] = np.array(eval_dict['groundtruth_boxes'])

        # update class variables
        self.test_filenames = test_filenames
        self.result_filename = result_filename
        self.eval_dict = eval_dict

    def process_nms(self,
            score_threshold=0.5,
            nms_iou_threshold=0.3,
            nms_max_output_boxes=50):

        num_image = len(self.eval_dict['key'])
        original_num_detected = self.eval_dict['detection_scores'].shape[1]
        
        for index in tqdm(range(num_image), desc='image : '):
            scores = self.eval_dict['detection_scores'][index]
            classes = self.eval_dict['detection_classes'][index]
            boxes = self.eval_dict['detection_boxes'][index]

            processed_scores = []
            processed_classes = []
            processed_boxes = []

            for index_class in range(self.num_classes):
                indices_per_class = np.argwhere(classes==(index_class+self.label_offset)).squeeze()

                scores_per_class = scores[indices_per_class]
                boxes_per_class = boxes[indices_per_class]
                
                box_list = np_box_list.BoxList(boxes_per_class)
                box_list.add_field('scores', scores_per_class) 

                nms_boxes = np_box_list_ops.non_max_suppression(
                        box_list,
                        max_output_size=10000,
                        iou_threshold=nms_iou_threshold,
                        score_threshold=score_threshold)

                num_boxes = nms_boxes.num_boxes()

                processed_scores.extend(nms_boxes.get_field('scores'))
                processed_classes.extend(np.full(num_boxes, index_class + self.label_offset))

                bndbox = nms_boxes.get_coordinates()
                for index_box in range(num_boxes):
                    processed_boxes.append(
                            [bndbox[0][index_box], bndbox[1][index_box], bndbox[2][index_box], bndbox[3][index_box]])

            processed_scores = np.array(processed_scores)
            processed_classes = np.array(processed_classes)
            processed_boxes = np.array(processed_boxes)

            num_detected = processed_scores.shape[0]

            processed_scores = np.append(processed_scores, np.zeros(original_num_detected - num_detected))
            processed_classes = np.append(processed_classes, np.zeros(original_num_detected - num_detected))
            processed_boxes = np.append(processed_boxes, np.full((original_num_detected - num_detected, 4), 0), axis=0)

            self.eval_dict['detection_scores'][index] = np.array(processed_scores)
            self.eval_dict['detection_classes'][index] = np.array(processed_classes)
            self.eval_dict['detection_boxes'][index] = np.array(processed_boxes)

    def calculate_metrics(self,
            score_threshold=0.5,
            matching_iou_threshold=0.5,
            nms_iou_threshold=0.3,
            nms_max_output_boxes=50):

        per_image_evaluation = PerImageEvaluation(
            num_groundtruth_classes=self.num_classes,
            matching_iou_threshold=matching_iou_threshold,
            nms_iou_threshold=nms_iou_threshold,
            nms_max_output_boxes=nms_max_output_boxes)

        eval_dict = self.eval_dict
        num_image = eval_dict['key'].shape[0]

        keys = []
        num_gts = [] # shape=(N_file, N_class)
        num_tps = [] # shape=(N_file, N_class)
        num_fps = [] # shape=(N_file, N_class)

        with_fps = [[] for i in range(self.num_classes)]
        with_fns = [[] for i in range(self.num_classes)]
        
        for index in tqdm(range(num_image), desc='calculate metrics : '):
            key = eval_dict['key'][index]

            num_gt_boxes = len(eval_dict['groundtruth_classes'][index])
        
            scores, tp_fp_labels, is_class_correctly_detected_in_image = per_image_evaluation.compute_object_detection_metrics(
                detected_boxes=eval_dict['detection_boxes'][index],
                detected_scores=eval_dict['detection_scores'][index],
                detected_class_labels=eval_dict['detection_classes'][index] - self.label_offset,
                groundtruth_boxes=eval_dict['groundtruth_boxes'][index],
                groundtruth_class_labels=eval_dict['groundtruth_classes'][index] - self.label_offset,
                groundtruth_is_difficult_list=np.full(num_gt_boxes, False),
                groundtruth_is_group_of_list=np.full(num_gt_boxes, False),
                detected_masks=None, groundtruth_masks=None)

            keys.append(key)
            num_gts.append([0] * self.num_classes)
            num_tps.append([0] * self.num_classes)
            num_fps.append([0] * self.num_classes)
        
            for class_index in range(self.num_classes):
                num_gt = int(np.sum(eval_dict['groundtruth_classes'][index] == class_index + self.label_offset))
                num_gts[index][class_index] = num_gt

                scores_per_class = scores[class_index]
                tp_fp_labels_per_class = tp_fp_labels[class_index]
        
                sorted_indices = np.argsort(scores_per_class)
                sorted_indices = sorted_indices[::-1]
                scores_per_class = scores_per_class[sorted_indices]
                n_detected = 0
                for score in scores_per_class:
                  if score > score_threshold: 
                    n_detected += 1
                  else:
                    break
                scores_per_class = scores_per_class[0:n_detected]
                tp_fp_labels_per_class = tp_fp_labels_per_class[sorted_indices][0:n_detected]
        
                tp_labels = tp_fp_labels_per_class.astype(int)
                fp_labels = (tp_fp_labels_per_class <= 0).astype(int)
        
                if len(scores_per_class) > 0:
                  num_tp = int(np.cumsum(tp_labels)[-1])
                  num_fp = int(np.cumsum(fp_labels)[-1])
                else:
                  num_tp = 0
                  num_fp = 0

                num_tps[index][class_index] = num_tp
                num_fps[index][class_index] = num_fp

                if num_fp > 0:
                    with_fps[class_index].append(key)
                if num_gt > num_tp:
                    with_fns[class_index].append(key)
        
        metric_dict = {
                'key': keys,
                'num_gt': num_gts,
                'num_tp': num_tps,
                'num_fp': num_fps}

        metric_dict_classification = {
                'with_fp': with_fps,
                'with_fn': with_fns}

        with open(self.result_filename + '.metric', 'w') as fp:
            fp.write(json.dumps(metric_dict) + '\n')
            fp.write(json.dumps(metric_dict_classification) + '\n')

    def filter_evaluation_dict(self,
            indices=[0]):

        for key, valuelist in self.eval_dict.items():
            self.eval_dict[key] = valuelist[indices]
        
    def draw_detections_with_groundtruth(self,
            score_threshold=0.5):

        import matplotlib
        matplotlib.use('TkAGG')

        tf.enable_eager_execution()

        num_image = len(self.eval_dict['key'])

        eval_vis = vis_util.VisualizeSingleFrameDetections(
               self.category_index,
               max_examples_to_draw=num_image,
               max_boxes_to_draw=20,
               min_score_thresh=score_threshold,
               use_normalized_coordinates=True,
               summary_name_prefix='Detections_Left_Groundtruth_Right')
        
        images_with_groundtruth = eval_vis.images_from_evaluation_dict(self.eval_dict)

        for index, image in enumerate(images_with_groundtruth):
            images_with_groundtruth[index] = image.numpy()[0]
            plt.imshow(images_with_groundtruth[index])    
            plt.show()

    def convert_prediction_to_xml(self, 
            score_thres=0.5,
            nms_iou_threshold=0.3,
            nms_max_output_boxes=50):

        import xml.etree.ElementTree as ET
        import xml.dom.minidom as md

        tf.enable_eager_execution()

        prediction_xml_root = ET.Element('annotations')
        num_image = len(self.eval_dict['key'])

        for index in tqdm(range(num_image), desc='image : '):
            prediction_xml_thisfile = ET.SubElement(prediction_xml_root, 'annotation')
            prediction_xml_filename = ET.SubElement(prediction_xml_thisfile, 'filename')
            prediction_xml_filename.text = self.eval_dict['key'][index]

            scores = self.eval_dict['detection_scores'][index]
            classes = self.eval_dict['detection_classes'][index]
            boxes = self.eval_dict['detection_boxes'][index]

            size_y, size_x = self.eval_dict['original_image_spatial_shape'][index]

            for index_class in range(self.num_classes):
                indices_per_class = np.argwhere(classes==(index_class+self.label_offset)).squeeze()

                scores_per_class = scores[indices_per_class]
                boxes_per_class = boxes[indices_per_class]
                
                box_list = np_box_list.BoxList(boxes_per_class)
                box_list.add_field('scores', scores_per_class) 

                nms_boxes = np_box_list_ops.non_max_suppression(
                        box_list,
                        max_output_size=10000,
                        iou_threshold=nms_iou_threshold,
                        score_threshold=score_thres)

                num_boxes = nms_boxes.num_boxes()
                bndbox = nms_boxes.get_coordinates()

                for index_box in range(num_boxes):
                    prediction_xml_object = ET.SubElement(prediction_xml_thisfile, 'object')

                    name = str(index_class + self.label_offset)
                    pose = 'Unspecified'
                    truncated = str(0) 
                    difficult = str(0)
                    ymin = str(int(bndbox[0][index_box]*size_y))
                    xmin = str(int(bndbox[1][index_box]*size_x))  
                    ymax = str(int(bndbox[2][index_box]*size_y))
                    xmax = str(int(bndbox[3][index_box]*size_x))

                    prediction_xml_name = ET.SubElement(prediction_xml_object, 'name')
                    prediction_xml_name.text = name
        
                    prediction_xml_pose = ET.SubElement(prediction_xml_object, 'pose')
                    prediction_xml_pose.text = pose
        
                    prediction_xml_truncated = ET.SubElement(prediction_xml_object, 'truncated')
                    prediction_xml_truncated.text = truncated
        
                    prediction_xml_difficult = ET.SubElement(prediction_xml_object, 'difficult')
                    prediction_xml_difficult.text = difficult
        
                    prediction_xml_bndbox = ET.SubElement(prediction_xml_object, 'bndbox')
                    prediction_xml_xmin = ET.SubElement(prediction_xml_bndbox, 'xmin')
                    prediction_xml_xmin.text = xmin
                    prediction_xml_ymin = ET.SubElement(prediction_xml_bndbox, 'ymin')
                    prediction_xml_ymin.text = ymin 
                    prediction_xml_xmax = ET.SubElement(prediction_xml_bndbox, 'xmax')
                    prediction_xml_xmax.text = xmax
                    prediction_xml_ymax = ET.SubElement(prediction_xml_bndbox, 'ymax')
                    prediction_xml_ymax.text = ymax 

        xml_filename = re.sub('.json$', '.xml', self.result_filename)
        with open(xml_filename, 'w') as fp_xml:
            xml_string = ET.tostring(prediction_xml_root, encoding="unicode")
            document = md.parseString(xml_string)
            fp_xml.write(document.toprettyxml(indent='  '))
        
    def export_train_eval_data(self, num_location=7, random_seed=42, ratio_eval=None, num_eval=None):
        import xml.etree.ElementTree as ET
        import random
        
        if ratio_eval == None and num_eval == None:
            print('error')
            return

        random.seed(random_seed)

        cnt_train = 0
        cnt_val = 0

        for i_location in range(1, num_location + 1):
            LocationDirectory = os.path.join(self.INPUT_BASE_DIR, 'location{}'.format(i_location))

            SearchDirectory = self.INPUT_BASE_DIR
            SearchDirectory = os.path.join(SearchDirectory, 'location{}'.format(i_location))
            SearchDirectory = os.path.join(SearchDirectory, 'labels')
            SearchDirectory = os.path.join(SearchDirectory, '*.xml')
            filename = glob.glob(SearchDirectory)

            filename.sort()

            random.shuffle(filename)
            num_examples = len(filename)

            if ratio_eval != None:
                num_train_by_ratio = int(num_examples * (1 - ratio_eval))
                num_train = num_train_by_ratio
            if num_eval != None:
                num_train_by_num = num_examples - num_eval
                num_train = num_train_by_num

            if ratio_eval != None and num_eval != None:
                num_train = num_train_by_ratio if num_train_by_ratio < num_train_by_num else num_train_by_num
            
            train_examples = filename[:num_train]
            val_examples = filename[num_train:]

            print('location-{}'.format(i_location))
            print('  {} training and {} validation examples.'.format(len(train_examples), len(val_examples)))

            writer_train = tf.python_io.TFRecordWriter(os.path.join(self.INPUT_BASE_DIR, 'train.record-{0:05d}-of-{1:05d}'.format(i_location - 1, num_location)))
            for ReadFile in train_examples:
                xmltree = ET.parse(ReadFile).getroot()
            
                folder_name = xmltree.find('folder').text
                image_name = xmltree.find('filename').text
                full_path = self.INPUT_BASE_DIR
                full_path = os.path.join(full_path, folder_name)  #provide the path of images directory
                full_path = os.path.join(full_path, 'images')  #provide the path of images directory
                full_path = os.path.join(full_path, image_name)  #provide the file name
            
                writer_train.write(create_example(xmltree, full_path).SerializeToString())
                cnt_train += 1
            writer_train.close()

            writer_val = tf.python_io.TFRecordWriter(os.path.join(self.INPUT_BASE_DIR, '../', 'val', 'val.record-{0:05d}-of-{1:05d}'.format(i_location - 1, num_location)))
            for ReadFile in val_examples:
                xmltree = ET.parse(ReadFile).getroot()

                folder_name = xmltree.find('folder').text
                image_name = xmltree.find('filename').text
                full_path = self.INPUT_BASE_DIR
                full_path = os.path.join(full_path, folder_name)  #provide the path of images directory
                full_path = os.path.join(full_path, 'images')  #provide the path of images directory
                full_path = os.path.join(full_path, image_name)  #provide the file name

                writer_val.write(create_example(xmltree, full_path).SerializeToString())
                cnt_val += 1
            writer_val.close()

        print('Created {} traning examples'.format(cnt_train))
        print('Created {} eval examples'.format(cnt_val))
