#!/bin/sh

RUNTIME_VERSION=1.10

GCSDIR=fujifilm-dsc-3

CURRENT_DIR=$(cd .;pwd)
BASE_DIR=$(cd $(dirname ${BASH_SOURCE:-$0})/../;pwd)
MYDETECTOR_DIR=$(cd ${BASE_DIR}/src/my_detector;pwd)

USE_MODEL_DIR=model_dir_2

LOCAL_TRAINING_DIR=${BASE_DIR}/training
LOCAL_MODEL_DIR=${LOCAL_TRAINING_DIR}/${USE_MODEL_DIR}
LOCAL_INPUT_DIR=${BASE_DIR}/input
LOCAL_VALDATA_DIR=${LOCAL_INPUT_DIR}/val
LOCAL_TESTDATA_DIR=${LOCAL_INPUT_DIR}/test

source ${BASE_DIR}/../bin/activate

export PYTHONPATH=${MYDETECTOR_DIR}:${MYDETECTOR_DIR}/slim

function activate_python () {
  if [ "$1" = 2 ];then
    source ${BASE_DIR}/../venv_python2/bin/activate
  else
    source ${BASE_DIR}/../bin/activate
  fi
}

function now_timestamp () {
  echo `date +%m_%d_%Y_%H_%M_%S`
}

function remove_dir_if_exists () {
  dir_name=$1

  if [ -d ${dir_name} ]; then
    echo "Remove ${dir_name}"
    read -p " y? " answer 
    if [ ${answer} = "y" ]; then
      rm -rf ${dir_name}
      echo "${dir_name} has been removed."
    fi
  fi
}

function upload_input () {
  gsutil -m cp ${LOCAL_INPUT_DIR}/train/train.record* gs://${GCSDIR}/input/train/
  gsutil -m cp ${LOCAL_INPUT_DIR}/val/val.record* gs://${GCSDIR}/input/val/
}

function train () {
  SAMPLE_1_OF_N_EVAL_EXAMPLES=1
  PIPELINE_CONFIG_PATH=`find ${LOCAL_MODEL_DIR} -depth 1 -name *.config`

  if [ "$1" = "local" ]; then
    PIPELINE_CONFIG_PATH_LOCAL=${PIPELINE_CONFIG_PATH}.local
    sed -e "s;gs://${GCSDIR}/input;${LOCAL_INPUT_DIR};g" \
        -e "s;gs://${GCSDIR};${LOCAL_TRAINING_DIR};g" \
        ${PIPELINE_CONFIG_PATH} > $PIPELINE_CONFIG_PATH_LOCAL
    NUM_TRAIN_STEPS=10000

    cd $MYDETECTOR_DIR

    deactivate
    activate_python 2

    python object_detection/model_main.py \
      --pipeline_config_path=${PIPELINE_CONFIG_PATH_LOCAL} \
      --model_dir=${LOCAL_MODEL_DIR} \
      --num_train_steps=${NUM_TRAIN_STEPS} \
      --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
      --alsologtostderr

    cd $CURRENT_DIR

    deactivate
    activate_python

    rm $PIPELINE_CONFIG_PATH_LOCAL
  elif [ "$1" = "remote" ];then
    NUM_TRAIN_STEPS=10000

    cd $MYDETECTOR_DIR

    gsutil -m cp ${PIPELINE_CONFIG_PATH} gs://${GCSDIR}/${USE_MODEL_DIR}/
    REMOTE_PIPELINE_CONFIG_PATH=gs://${GCSDIR}/${USE_MODEL_DIR}/`basename ${PIPELINE_CONFIG_PATH}`

    deactivate
    activate_python 2

    python setup.py sdist
    rm -rf object_detection.egg-info
    DIST=dist

    gcloud ml-engine jobs submit training `whoami`_object_detection_road_`date +%m_%d_%Y_%H_%M_%S` \
      --runtime-version ${RUNTIME_VERSION} \
      --job-dir gs://${GCSDIR}/${USE_MODEL_DIR} \
      --package-path $MYDETECTOR_DIR \
      --packages ${DIST}/object_detection-0.1.tar.gz,${DIST}/slim-0.1.tar.gz,${DIST}/pycocotools-2.0.tar.gz \
      --module-name object_detection.model_main \
      --region asia-east1 \
      --config cloud/train.yml \
      --stream-logs \
      -- \
      --model_dir=gs://${GCSDIR}/${USE_MODEL_DIR} \
      --pipeline_config_path=${REMOTE_PIPELINE_CONFIG_PATH} \
      --num_train_steps=${NUM_TRAIN_STEPS} \
      --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES

    deactivate
    activate_python

    cd $CURRENT_DIR
  fi
}

function evaluate () {
  PIPELINE_CONFIG_PATH=`find ${LOCAL_MODEL_DIR} -depth 1 -name *.config.eval`

  SAMPLE_1_OF_N_EVAL_EXAMPLES=1

  if [ "$1" = "local" ]; then
    PIPELINE_CONFIG_PATH_LOCAL=${PIPELINE_CONFIG_PATH}.local
    sed -e "s;gs://${GCSDIR}/input;${LOCAL_INPUT_DIR};g" ${PIPELINE_CONFIG_PATH} > $PIPELINE_CONFIG_PATH_LOCAL

    cd $MYDETECTOR_DIR

    deactivate
    activate_python 2

    python object_detection/model_main.py \
      --checkpoint_dir ${LOCAL_MODEL_DIR} \
      --run_once="True" \
      --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
      --model_dir=${LOCAL_MODEL_DIR} \
      --pipeline_config_path=${PIPELINE_CONFIG_PATH_LOCAL}

    cd $CURRENT_DIR

    deactivate
    activate_python

    rm $PIPELINE_CONFIG_PATH_LOCAL
  elif [ "$1" = "remote" ];then
    gsutil -m cp ${PIPELINE_CONFIG_PATH} gs://${GCSDIR}/${USE_MODEL_DIR}/
    PIPELINE_CONFIG_PATH_REMOTE=gs://${GCSDIR}/${USE_MODEL_DIR}/`basename ${PIPELINE_CONFIG_PATH}`

    SAMPLE_1_OF_N_EVAL_EXAMPLES=1
    CHECKPOINT_DIR=gs://${GCSDIR}/${USE_MODEL_DIR}
    MODEL_DIR=gs://${GCSDIR}/${USE_MODEL_DIR}
    PIPELINE_CONFIG_PATH=$PIPELINE_CONFIG_PATH_REMOTE

    cd $MYDETECTOR_DIR

    deactivate
    activate_python 2
    
    python setup.py sdist
    rm -rf object_detection.egg-info
    DIST=dist

    gcloud ml-engine jobs submit training `whoami`_test_object_detection_road_`date +%m_%d_%Y_%H_%M_%S` \
      --runtime-version ${RUNTIME_VERSION} \
      --job-dir gs://${GCSDIR}/${USE_MODEL_DIR} \
      --package-path $MYDETECTOR_DIR \
      --packages ${DIST}/object_detection-0.1.tar.gz,${DIST}/slim-0.1.tar.gz,${DIST}/pycocotools-2.0.tar.gz \
      --module-name object_detection.model_main \
      --region asia-east1 \
      --config cloud/predict.yml \
      --stream-logs \
      -- \
      --checkpoint_dir=${CHECKPOINT_DIR} \
      --run_once="True" \
      --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
      --model_dir=${MODEL_DIR} \
      --pipeline_config_path=${PIPELINE_CONFIG_PATH}
    
    cd $CURRENT_DIR
    
    deactivate
    activate_python
  fi
}

function predict () {
  PIPELINE_CONFIG_PATH=`find ${LOCAL_MODEL_DIR} -depth 1 -name *.config.predict`

  if [ "$1" = "local" ]; then
    PIPELINE_CONFIG_PATH_LOCAL=${PIPELINE_CONFIG_PATH}.local
    sed -e "s;gs://${GCSDIR}/input;${LOCAL_INPUT_DIR};g" \
      ${PIPELINE_CONFIG_PATH} > $PIPELINE_CONFIG_PATH_LOCAL

    PIPELINE_CONFIG_PATH_REMOTE=gs://${GCSDIR}/${USE_MODEL_DIR}/`basename ${PIPELINE_CONFIG_PATH}`

    SAMPLE_1_OF_N_EVAL_EXAMPLES=600

    CHECKPOINT_DIR=${LOCAL_MODEL_DIR}
    MODEL_DIR=${LOCAL_MODEL_DIR}
    PIPELINE_CONFIG_PATH=${PIPELINE_CONFIG_PATH_LOCAL}

    cd $MYDETECTOR_DIR
    
    deactivate
    activate_python 2
    
    python object_detection/model_main.py \
      --checkpoint_dir=${CHECKPOINT_DIR} \
      --run_once="True" \
      --is_predict="True" \
      --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
      --model_dir=${MODEL_DIR} \
      --pipeline_config_path=${PIPELINE_CONFIG_PATH}
    
    cd $CURRENT_DIR
    
    deactivate
    activate_python
    
    rm $PIPELINE_CONFIG_PATH_LOCAL
  elif [ "$1" = "remote" ];then
    gsutil -m cp ${PIPELINE_CONFIG_PATH} gs://${GCSDIR}/${USE_MODEL_DIR}/
    PIPELINE_CONFIG_PATH_REMOTE=gs://${GCSDIR}/${USE_MODEL_DIR}/`basename ${PIPELINE_CONFIG_PATH}`

    SAMPLE_1_OF_N_EVAL_EXAMPLES=1
    CHECKPOINT_DIR=gs://${GCSDIR}/${USE_MODEL_DIR}
    MODEL_DIR=gs://${GCSDIR}/${USE_MODEL_DIR}
    PIPELINE_CONFIG_PATH=$PIPELINE_CONFIG_PATH_REMOTE

    cd $MYDETECTOR_DIR

    deactivate
    activate_python 2
    
    python setup.py sdist
    rm -rf object_detection.egg-info
    DIST=dist

    gcloud ml-engine jobs submit training `whoami`_test_object_detection_road_`date +%m_%d_%Y_%H_%M_%S` \
      --runtime-version ${RUNTIME_VERSION} \
      --job-dir gs://${GCSDIR}/${USE_MODEL_DIR} \
      --package-path $MYDETECTOR_DIR \
      --packages ${DIST}/object_detection-0.1.tar.gz,${DIST}/slim-0.1.tar.gz,${DIST}/pycocotools-2.0.tar.gz \
      --module-name object_detection.model_main \
      --region asia-east1 \
      --config cloud/predict.yml \
      --stream-logs \
      -- \
      --checkpoint_dir=${CHECKPOINT_DIR} \
      --run_once="True" \
      --is_predict="True" \
      --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
      --model_dir=${MODEL_DIR} \
      --pipeline_config_path=${PIPELINE_CONFIG_PATH}
    
    cd $CURRENT_DIR
    
    deactivate
    activate_python
  fi
}

# upload_input
# train local
# evaluate local
predict local
