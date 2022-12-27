import itertools
import os
import torch
from option import Options
from networks import *

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

def get_options():
    opt = Options().initialize()
    return opt

opt = get_options()

model_gnn = torch.nn.DataParallel(model_gnn(
    in_dim=opt.in_dim,
    hidden_dim=opt.hidden_dim,
    out_dim=opt.out_dim,
    dim=opt.ROIs,
)).cuda()

G_AB =  torch.nn.DataParallel(GeneratorAB(
    input_size=opt.ROIs,
    hidden1=opt.hidden1,
    hidden2=opt.hidden2,
    hidden3=opt.hidden3,
    output_size=opt.hidden4,
    dropout=opt.dropout,
    batch_size=opt.BATCH_SIZE
)).cuda()

G_BA =  torch.nn.DataParallel(GeneratorBA(
    input_size=opt.ROIs,
    hidden1=opt.hidden1,
    hidden2=opt.hidden2,
    hidden3=opt.hidden3,
    output_size=opt.hidden4,
    dropout=opt.dropout,
    batch_size=opt.BATCH_SIZE,
    class_nums=opt.class_nums
)).cuda()

D_A = torch.nn.DataParallel(DiscriminatorA(
    input_size=opt.ROIs,
    hidden1=opt.hidden1,
    hidden2=opt.hidden2,
    hidden3=opt.hidden3,
    output_size=1,
    dropout=opt.dropout,
    batch_size=opt.BATCH_SIZE,
    class_nums=opt.class_nums
)).cuda()

D_B = torch.nn.DataParallel(DiscriminatorB(
    input_size=opt.ROIs,
    hidden1=opt.hidden1,
    hidden2=opt.hidden2,
    hidden3=opt.hidden3,
    output_size=1,
    dropout=opt.dropout,
    batch_size=opt.BATCH_SIZE,
)).cuda()

criterionIdt = torch.nn.L1Loss().cuda()
criterionCycle = torch.nn.L1Loss().cuda()
criterionGEN = torch.nn.L1Loss().cuda()

optimizer_G = torch.optim.SGD(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr_G,momentum=opt.momentum)
optimizer_D = torch.optim.SGD(itertools.chain(D_A.parameters(), D_B.parameters()), lr=opt.lr_D,momentum=opt.momentum)
optimizer_M = torch.optim.Adam(filter(lambda p: p.requires_grad, model_gnn.parameters()), lr=opt.lr_M,weight_decay=opt.weight_decay)