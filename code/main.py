from __future__ import print_function
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import numpy as np
import argparse
import json
import pdb
import os
from vqaDataset import vqaDataset
from vqaModel import vqaModel

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0003,
                   help='base learning rate')
parser.add_argument('--phase', type=str, default='train',
                   help='phase')
parser.add_argument('--model', type=str, default='',
                   help='pretrained model')
parser.add_argument('--param_only', type=bool, default=False,
                   help='[False]: restore both params and opt states; [True]: restore only params')
parser.add_argument('--save_model', type=str, default='tmp',
                   help='save model path')

args = parser.parse_args()
print(args)

### config
# data
dataset = {}
dataset['data_info'] = '/home/gina/mlip/ECE285_final/data/data_prepro.json'
dataset['questions'] = '/home/gina/mlip/ECE285_final/data/data_prepro.h5'
dataset['img_feat'] = '/home/gina/mlip/data/data_img_fc7.h5'
# learning
base_lr = 0.0003
epoch = 60001
batch_size = 300
disp_ep = 50
save_ep = 1000
# model
ques_emb_size = 300
rnn_hidden_size = 512
emb_size = 1024
output_size = 1000
###

dtype = torch.FloatTensor
loss_func = nn.CrossEntropyLoss()

###
# Training the vqa model. (optional: restore parameters and optimizer states from a checkpoint).
# - model_path: model (checkpoint) to restore.
# - param_only: [True] for resotoring parameters;
#               [False] for restoring both parameters and optimizer states (e.g. lr).
###

def train(model_path='', param_only=False):

    # set learning rate and the path to save model
    save_dir = args.save_model
    if not os.path.exists('checkpoint/{}'.format(save_dir)):
        os.makedirs('checkpoint/{}'.format(save_dir))
    lr = base_lr
    if args.lr:
        lr = args.lr

    # create dataset, dataloader, model and optimizer
    dset_train = vqaDataset(dataset, 'train')
    sampler = SubsetRandomSampler(list(range(0, dset_train.__len__())))
    train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, sampler=sampler, shuffle=False)
    #train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True)

    vqa = vqaModel(
        dset_train.dict_size,
        ques_emb_size,
        rnn_hidden_size,
        dset_train.img_dim,
        emb_size,
        output_size
    ).cuda()

    optimizer = torch.optim.Adam(vqa.parameters(), lr=lr)

    # restore model parameters (and optimizer states)
    start_ep = 0
    if model_path != '':
        checkpoint = torch.load(model_path)
        vqa.load_state_dict(checkpoint['vqa_state'])
        print('Recover weights from {}'.format(model_path))
        if not param_only:
            start_ep = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['opt_state'])
            lr = checkpoint['opt_state']['param_groups'][0]['lr']
            optimizer.param_groups[0].update({'initial_lr': lr})
    print('Start training from epoch-{}, base_lr={}'.format(start_ep, lr))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.99)

    for ep in range(start_ep, epoch):
        for step, value in enumerate(train_loader):
            if step != 0:
                continue
            batch_imfeat = Variable(value[0].type(dtype), requires_grad=False).cuda()
            batch_ques = Variable(value[1].type(torch.LongTensor), requires_grad=False).cuda()
            batch_ans = Variable(value[3].type(torch.LongTensor), requires_grad=False).cuda()

            output = vqa(batch_imfeat, batch_ques)
            loss = loss_func(output, batch_ans)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if ep % disp_ep == 0:
            pred = torch.max(output, dim=1)[1]
            accuracy = torch.eq(pred, batch_ans).cpu().data.numpy().sum() / float(batch_size) * 100
            print('-----------------------------------------')
            print('Epoch: {}'.format(ep))
            print('current lr = %g' % optimizer.state_dict()['param_groups'][0]['lr'])
            print('Classification Loss: %.4f' % (loss.data[0]))
            print('Train Accuracy: %.2f' % accuracy+'%')
            print('-----------------------------------------')

        # save checkpoint
        if ep % save_ep == 0:
            torch.save({
                'epoch': ep,
                'vqa_state': vqa.state_dict(),
                'opt_state': optimizer.state_dict()},
                'checkpoint/{}/ep-{}.pt'.format(save_dir, ep)
            )
            print('Saving checkpoint/{}/ep-{}.pt'.format(save_dir, ep))

###
# Testing the vqa model.
# - model_path: model (checkpoint) to restore.
# - dset: dataset for testing (training/testing).
###

def test(model_path='checkpoint/tmp/ep-0.pt', dset='test'):

    # create dataset, dataloader, model
    dset_test = vqaDataset(dataset, dset)
    test_loader = DataLoader(dataset=dset_test, batch_size=batch_size, shuffle=False)

    vqa = vqaModel(
        dset_test.dict_size,
        ques_emb_size,
        rnn_hidden_size,
        dset_test.img_dim,
        emb_size,
        output_size
    ).cuda()

    # restore model parameters
    checkpoint = torch.load(model_path)
    vqa.load_state_dict(checkpoint['vqa_state'])
    print('Testing model: {}'.format(model_path))

    result = []
    for step, value in enumerate(test_loader):
        batch_imfeat = Variable(value[0].type(dtype), requires_grad=False).cuda()
        batch_ques = Variable(value[1].type(torch.LongTensor), requires_grad=False).cuda()
        batch_ques_ix = value[2].numpy()

        output = vqa(batch_imfeat, batch_ques)
        pred = torch.max(output, dim=1)[1].cpu().data.numpy()

        for i in xrange(len(pred)):
            ans = dset_test.ix_to_ans[str(pred[i]+1)]
            result.append({u'answer': ans, u'question_id': str(batch_ques_ix[i])})

    print('Testing done.')
    my_list = list(result)
    dd = json.dump(my_list,open(model_path.replace('.pt', '_%s.json' % dset), 'w'))

def main():
    model = ''
    if args.model:
        model = args.model
    if args.phase == 'train':
        train(model, args.param_only)
    elif args.phase == 'valid':
        test(model, 'train')
    else:
        test(model, 'test')

main()
