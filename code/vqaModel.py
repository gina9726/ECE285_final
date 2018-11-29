import torch
from torch import nn
import torch.nn.functional as F

class vqaModel(nn.Module):
    def __init__(self, dict_size, ques_emb_size, rnn_hidden_size, img_feat_dim, emb_size, output_size):
        super(vqaModel, self).__init__()

        self.ques_emb = nn.Embedding(dict_size, ques_emb_size)

        self.rnn = nn.LSTM(
            input_size=ques_emb_size,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        self.state_dp = nn.Dropout(p=0.5)
        self.state_emb = nn.Linear(rnn_hidden_size, emb_size)

        self.img_dp = nn.Dropout(p=0.5)
        self.img_emb = nn.Linear(img_feat_dim, emb_size)

        self.score_dp = nn.Dropout(p=0.5)
        self.score_emb = nn.Linear(emb_size, output_size)


    def forward(self, img_feat, ques, ques_mask):
        # question embedding lookup
        ques_emb = F.tanh(self.ques_emb(ques))

        # rnn: encode questions
        # ques_shape = (batch, time_step, input_size)
        r_out, (h_n, h_c) = self.rnn(ques_emb, None)
        state = torch.sum(torch.mul(r_out, ques_mask), dim=1)

        # multimodal fusion (question and image)
        state_dp = self.state_dp(state)
        state_emb = F.tanh(self.state_emb(state_dp))

        img_dp = self.img_dp(img_feat)
        img_emb = F.tanh(self.img_emb(img_dp))

        score_dp = self.score_dp(torch.mul(state_emb, img_emb))
        output = self.score_emb(score_dp)

        return output

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            own_state[name].copy_(param)

