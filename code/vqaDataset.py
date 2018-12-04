from torch.utils.data.dataset import Dataset
import numpy as np
import json
import h5py
import pdb

class vqaDataset(Dataset):
    
    def __init__(self, dataset, phase):

        with open(dataset['data_info']) as data_file:
            data = json.load(data_file)
            self.ix_to_word = data['ix_to_word']
            self.ix_to_ans = data['ix_to_ans']
            self.dict_size = len(self.ix_to_word.keys())+1
            self.num_ans = len(self.ix_to_ans.keys())

        with h5py.File(dataset['questions'], 'r') as hf:
            temp = hf.get('ques_length_%s' % phase)
            self.len_ques = np.array(temp).astype(np.int64)

            temp = hf.get('ques_%s' % phase)
            question = np.array(temp).astype(np.int64)
            self.max_len = question.shape[1]
            # right align questions
            N = len(self.len_ques)
            self.question = np.zeros([N, self.max_len])
            for i in range(N):
                self.question[i][-self.len_ques[i]:] = question[i][:self.len_ques[i]]

            temp = hf.get('img_pos_%s' % phase)
            self.img_list = np.array(temp)

            temp = hf.get('question_id_%s' % phase)
            self.ques_ix = np.array(temp).astype(np.int64)

            # make the answer idx start from 0
            temp = hf.get('answers')
            self.answer = np.array(temp).astype(np.int64)-1

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
        ques_ix = self.ques_ix[index]
        ans = self.answer[index]
        return img_feat, ques, ques_ix, ans

    def __len__(self):
        return len(self.question)


