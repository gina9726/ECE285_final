import torch
from torch.autograd import Variable
from torch import nn
import pdb

class vqaModel(nn.Module):
    def __init__(self, dict_size, ques_emb_size, rnn_hidden_size, img_feat_dim, emb_size, output_size):
        super(vqaModel, self).__init__()

        self.ques_emb = nn.Embedding(dict_size, ques_emb_size, padding_idx=0)

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(
            input_size=ques_emb_size,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.state_dp = nn.Dropout(p=0.5)
        self.state_emb = nn.Linear(4 * rnn_hidden_size, emb_size)

        self.ans_mask = nn.Linear(4 * rnn_hidden_size, output_size)

        self.img_dp = nn.Dropout(p=0.5)
        self.img_emb = nn.Linear(img_feat_dim, emb_size)

        self.score1_dp = nn.Dropout(p=0.5)
        self.score1_emb = nn.Linear(emb_size, output_size)

        self.score2_dp = nn.Dropout(p=0.5)
        self.score2_emb = nn.Linear(output_size, output_size)

    def forward(self, img_feat, ques):
        # question embedding lookup
        ques_emb = torch.tanh(self.ques_emb(ques))

        batch_size = img_feat.size(0)
        h_0 = Variable(torch.zeros([2, batch_size, self.rnn_hidden_size])).cuda()
        c_0 = Variable(torch.zeros([2, batch_size, self.rnn_hidden_size])).cuda()

        # rnn: encode questions
        r_out, (h_n, h_c) = self.rnn(ques_emb, (h_0, c_0))
        h_nf = h_n.permute(1, 0, 2).contiguous().view(-1, 2 * self.rnn_hidden_size)
        h_cf = h_c.permute(1, 0, 2).contiguous().view(-1, 2 * self.rnn_hidden_size)
        state = torch.cat([h_nf, h_cf], dim=1)

        # multimodal fusion (question and image)
        state_dp = self.state_dp(state)
        state_emb = torch.tanh(self.state_emb(state_dp))

        ans_mask = torch.sigmoid(self.ans_mask(state_dp))

        img_dp = self.img_dp(img_feat)
        img_emb = torch.tanh(self.img_emb(img_dp))

        score1_dp = self.score1_dp(torch.mul(state_emb, img_emb))
        score1_emb = torch.tanh(self.score1_emb(score1_dp))

        score2_dp = self.score2_dp(score1_emb)
        score2_emb = self.score2_emb(score2_dp)
        output = torch.mul(score2_emb, ans_mask)

        return output, ans_mask

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            own_state[name].copy_(param)

