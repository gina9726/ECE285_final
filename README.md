# ECE285_final
ECE 285 final project

### Requirements
The code is written in Python 2 and requires pytorch(0.3.0).
check pytorch version:
````
import torch
print(torch.__version__)
````
Notice: using pytorch 0.3.1 with cuda 9 will cause error.
## VQA: Visual Question Answering
### Data Preprocessing (from [VQA_LSTM_CNN](https://github.com/GT-Vision-Lab/VQA_LSTM_CNN))
```
$ ./prepro.sh
```
*prepro.py* is to preprocess training and testing question-image-answer pairs. It builds a dictionary (*word_to_ix.json*) of the words in questions, and encodes each word to a index. Also, it maps each answer into an index (the mapping is stored in *ix_to_ans.json*).

*prepro.sh* is just a script for running prepro.py.

outputfile:
* word_to_ix.json: dictionary of questions, mapping each word to a index.

* ix_to_ans.json: mapping each index to an answer. Note that these indices are different from those in the dictionary of questions.

* data_prepro.h5: preprocessed data. There are many fields in the file, which are listed as follows
 - "ques_train", "ques_length_train", "answers", "question_id_train", "img_pos_train"
 - "ques_test", "ques_length_test", "question_id_test", "img_pos_test"
To know the content and the format of each field, please refer to the code and comments in *prepro.py*.

* data_prepro.json: preprocessed data, storing the two mappings and the path of unique images.
 - "ix_to_word", "ix_to_ans"
 - "unique_img_train", "unique_img_test"
----------------------------------------------
### Extract Image (from [san-vqa-tensorflow](https://github.com/TingAnChien/san-vqa-tensorflow))
To get the image feature:
````
python prepro_img.py
````
Here we use caffe to extract fc7 feature from vgg19 CNN model. After feature Extraction, we save those image into two sets: unique-img-train and unique-img test.

*prepro_img.py* extracts image features into features. Here we use *caffe* to extract fc7 feature from vgg19 CNN model.

outputfile:
* data_img_fc7.h5: fc7 features extracted from vgg19.

### Training and Testing
```
$ python main.py --lr ${learning rate} --phase ${'train', 'valid' or 'test'} --model ${model_name} --save_model ${model_name} &> logs/${model_name}.log

> --lr: starting learning rate
> --phase: choose which phase(training, testing or validation)
> --model: start with the chosen model(skip this if train for the first time)
> --param_only: False to restore both params and opt states
                True to restore only params
> --save_model: save the model as model name

```
you can also run .sh file to train or test the code:
````
chmod +x ${train.sh, valid.sh, test.sh}
./${train.sh, valid.sh, test.sh}
````
Note that the code of our modified version of model is named with "_mask".

### Draw learning rate and accuracy curve
````
python draw_curve.py logs/${your doc. name}.log
````
### Evaluation (from [VQA](https://github.com/GT-Vision-Lab/VQA?fbclid=IwAR2xhKtJ7-J5G6NdKwATm3yT_b64rZuN13xGoj7UKRkGDoqel3a22aSEghs))
* An evaluation code is inside Evaluation file. Simply input an image and ask a question, it will provide an answer.
* Needed datasets: word_to_id.json, data_prepro.json, data_prepro.h5, data_img_fc7.h5, path of trained model, input question and image.
 
