# ECE285_final
ECE 285 final project

## VQA: Visual Question Answering
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
*prepro_img.py* extracts image features into features. Here we use *caffe* to extract fc7 feature from vgg19 CNN model.

outputfile:
* data_img_fc7.h5: fc7 features extracted from vgg19.

The file is too large we split it into several npy files.
0 ~ 16515 are saved to img_fc7_train_part1.npy
16515 ~ 33030 are saved to img_fc7_train_part2.npy
33030 ~ 49545 are saved to img_fc7_train_part3.npy
49545 ~ 66060 are saved to img_fc7_train_part4.npy
66060 ~ 82575 are saved to img_fc7_train_part5.npy
testing image features are saved to img_fc7_test.npy

### Training and Testing
```
$ python main.py --lr ${learning rate} --phase ${'train', 'valid' or 'test'} --model ${model_name} --save_model ${model_name} &> logs/${model_name}.log

> --lr: starting learning rate
> --phase: choose which phase(training, testing or validation)
> --model: start with the chosen model
> --save_model: save the model as model name
