from torch.utils.data.dataset import Dataset
import numpy as np
import json
import h5py

class vqaDataset(Dataset):
    
    def __init__(self, dataset, phase):

        with open(dataset['data_info']) as data_file:
            data = json.load(data_file)
            self.ix_to_word = data['ix_to_word']
            self.ix_to_ans = data['ix_to_ans']
            self.dict_size = len(self.ix_to_word.keys())
            self.num_ans = len(self.ix_to_ans.keys())

        with h5py.File(dataset['questions'], 'r') as hf:
            temp = hf.get('ques_length_%s' % phase)
            self.len_ques = np.array(temp)

            temp = hf.get('ques_%s' % phase)
            # make the word idx start from 0
            self.question = np.array(temp)-1
            self.question = self.question.astype(np.int64)
            self.max_len = self.question.shape[1]
            self.ques_mask = np.zeros([len(self.question), self.max_len])
            index = np.linspace(0, len(self.question)-1, len(self.question)).astype(np.int64)
            self.ques_mask[index, self.len_ques-1] = 1

            temp = hf.get('img_pos_%s' % phase)
            self.img_list = np.array(temp)

            temp = hf.get('question_id_%s' % phase)
            self.ques_ix = np.array(temp)

            # make the answer idx start from 0
            temp = hf.get('answers')
            self.answer = np.array(temp)-1
            self.answer = self.answer.astype(np.int64)

        with h5py.File(dataset['img_feat'], 'r') as hf:
            temp = hf.get('images_%s' % phase)
            self.img_feat = np.array(temp).astype(np.float64)

            # normalize image feature
            self.img_dim = self.img_feat.shape[1]
            temp = np.sqrt(np.sum(np.multiply(self.img_feat, self.img_feat), axis=1))
            self.img_feat = np.divide(self.img_feat, np.transpose(np.tile(temp, (self.img_dim, 1))))

    def __getitem__(self, index):
        img_feat = self.img_feat[self.img_list[index]]
        ques = self.question[index]
        ques_mask = self.ques_mask[index]
        ans = self.answer[index]
        return img_feat, ques, ques_mask, ans

    def __len__(self):
        return len(self.question)


