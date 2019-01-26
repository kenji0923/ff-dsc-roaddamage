from MyClass import MyObjectDetector
import os
import glob
import tensorflow as tf

PROJECT_BASE_DIR = MyObjectDetector.PROJECT_BASE_DIR
TEST_DIR = os.path.join(PROJECT_BASE_DIR, 'input', 'test')
VAL_DIR = os.path.join(PROJECT_BASE_DIR, 'input', 'val')
TRAIN_DIR = os.path.join(PROJECT_BASE_DIR, 'input', 'train')

detector = MyObjectDetector('model_dir_2')
MODEL_DIR = detector.MODEL_DIR

def convert_prediction_to_xml():
    # # model_dir
    # result_filenames = os.path.join(MODEL_DIR, 'test_detections_01_26_2019_02_43_09.json')

    # model_dir_2
    result_filenames = os.path.join(MODEL_DIR, 'test_detections_12_28_2018_10_21_10.json')

    test_filenames = glob.glob(os.path.join(TEST_DIR, 'test.record-00000-of-00001'))

    detector.get_evaluation_dict(
            result_filename=result_filenames,
            test_filenames=test_filenames,
            b_load_original_image=False,
            num_draw=-1)

    detector.convert_prediction_to_xml(
            score_thres=0.5,
            nms_iou_threshold=0.3,
            nms_max_output_boxes=50)

def export_train_eval_data():
    detector.export_train_eval_data(num_location=7, random_seed=42, ratio_eval=0.1)

def calculate_metrics_of_train_eval_data():
    result_filename = os.path.join(MODEL_DIR, 'test_detections_01_02_2019_04_44_55.json')

    test_filenames = glob.glob(os.path.join(TRAIN_DIR, 'train.record-*'))
    test_filenames += glob.glob(os.path.join(VAL_DIR, 'val.record-*'))

    detector.get_evaluation_dict(
            result_filename=result_filename,
            test_filenames=test_filenames,
            b_load_original_image=False,
            num_draw=-1)

    detector.calculate_metrics(
            score_threshold=0.5,
            matching_iou_threshold=0.5,
            nms_iou_threshold=0.3,
            nms_max_output_boxes=50)

def draw_detections_with_groundtruth():
    result_filename = os.path.join(MODEL_DIR, 'test_detections_01_02_2019_04_44_55.json')

    test_filenames = glob.glob(os.path.join(TRAIN_DIR, 'train.record-*'))
    test_filenames += glob.glob(os.path.join(VAL_DIR, 'val.record-*'))

    detector.get_evaluation_dict(
            result_filename=result_filename,
            test_filenames=test_filenames,
            b_load_original_image=True,
            num_draw=10)

    # detector.filter_evaluation_dict(
    #         indices=[0])

    score_threshold = 0.5

    detector.process_nms(
            score_threshold=score_threshold,
            nms_iou_threshold=0.3,
            nms_max_output_boxes=50)

    detector.draw_detections_with_groundtruth(
            score_threshold=score_threshold)

def main():
    # export_train_eval_data()

    # draw_detections_with_groundtruth()

    # draw_test_result()

    convert_prediction_to_xml()

if __name__ == "__main__":
    main()
