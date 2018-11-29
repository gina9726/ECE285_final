DATA_PATH="/gpu2_data/gina/data"
TRAIN_JSON=${DATA_PATH}/vqa_raw_train.json
TEST_JSON=${DATA_PATH}/vqa_raw_test.json
NUM_ANS=1000

echo "python prepro.py --input_train_json ${TRAIN_JSON} --input_test_json ${TEST_JSON} --num_ans ${NUM_ANS}"
python prepro.py --input_train_json ${TRAIN_JSON} --input_test_json ${TEST_JSON} --num_ans ${NUM_ANS}

