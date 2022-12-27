import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import math
from torch.nn import functional as F
from option import Options
from torch.nn import init

opt = Options().initialize()

class GraphConvolution(nn.Module):
   """
   Graph Convolutional Network
   """

   def __init__(self, in_features, out_features, bias=True):
      super(GraphConvolution, self).__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.weight = Parameter(torch.FloatTensor(in_features, out_features))
      if bias:
         self.bias = Parameter(torch.FloatTensor(out_features))
      else:
         self.register_parameter('bias', None)
      self.reset_parameters()

   def reset_parameters(self):
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)
      if self.bias is not None:
         self.bias.data.uniform_(-stdv, stdv)

   def forward(self, input,adj):
      support = torch.matmul(input, self.weight)
      output = torch.matmul(adj.float(), support)
      if self.bias is not None:
         return output + self.bias.cuda()
      else:
         return output

class GeneratorAB(nn.Module):
    """
       Encoder-Decoder network.
       """

    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, dropout, batch_size):
        super(GeneratorAB, self).__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_size = output_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.weight_std = False
        self.in_place = True
        self.gcn1_p = GraphConvolution(self.input_size, self.hidden1)
        self.gcn2_p = GraphConvolution(self.hidden1, self.hidden2)
        self.gcn3_p = GraphConvolution(self.hidden2, self.hidden3)
        self.gcn4_p = GraphConvolution(self.hidden3, self.output_size)
        self.gcn1_n = GraphConvolution(self.input_size, self.hidden1)
        self.gcn2_n = GraphConvolution(self.hidden1, self.hidden2)
        self.gcn3_n = GraphConvolution(self.hidden2, self.hidden3)
        self.gcn4_n = GraphConvolution(self.hidden3, self.output_size)
        self.init_weights = True
        self.sigmoid_output = nn.Sequential(
            nn.LogSigmoid())
        # self.GAP = nn.AdaptiveAvgPool1d(1)
        # self.controller = nn.Conv1d(opt.hidden3,
        #                             self.output_size * self.output_size + self.output_size, 1, bias=False)
        if self.init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def DCL(self, conv3, conv4):
        x_conv = conv3.permute(0, 2, 1).contiguous()
        x_feat = self.GAP(x_conv)
        params = self.controller(x_feat)
        params.squeeze_(-1)
        N_, H_, W_ = conv4.size()
        head_inputs = conv4.view(1, -1, H_)  # 变化成一个维度进行组卷积
        weight_nums, bias_nums = [], []
        # weight_nums.append(8 * 8)
        weight_nums.append(self.output_size * self.output_size)
        # bias_nums.append(8)
        bias_nums.append(self.output_size)
        # weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)
        weights, biases = self.parse_dynamic_params(params, self.output_size, weight_nums, bias_nums)
        logits = self.heads_forward(head_inputs, weights, biases, N_)
        logits = logits.reshape(N_, H_, -1)
        return logits

    def heads_forward(self, features, weights, biases, num_insts):
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias = b, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(-1, channels, 1,)
                bias_splits[l] = bias_splits[l].reshape(-1)
            else:
                weight_splits[l] = weight_splits[l].reshape(-1, channels, 1,)
                bias_splits[l] = bias_splits[l].reshape(-1)

        return weight_splits, bias_splits

    def normalize(self, adj):
        one_batch = []
        for batchsize in adj:
            temp_norm = []
            for reduce in batchsize:
                reduce = reduce + torch.eye(reduce.shape[0]).cuda()
                rowsum = reduce.sum(1)
                degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
                degree_mat_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
                adj_normalized = torch.mm(torch.mm(degree_mat_inv_sqrt, reduce), degree_mat_sqrt)
                temp_norm.append(adj_normalized.unsqueeze(0))
            temp_norm_ = torch.cat(temp_norm, 0)
            one_batch.append(temp_norm_.unsqueeze(0))
        one_batch = torch.cat(one_batch, 0)
        return one_batch

    def forward(self, adj):
        X = torch.eye(self.input_size).cuda()
        adj = self.normalize(adj)
        A = torch.transpose(adj, 1, 0)
        feature_p = A[0]
        feature_n = A[1]

        p_conv1 = nn.LeakyReLU(0.2, inplace=True)(self.gcn1_p(X, feature_p))
        p_conv2 = nn.LeakyReLU(0.2, inplace=True)(self.gcn2_p(p_conv1, feature_p))
        p_conv3 = nn.LeakyReLU(0.2, inplace=True)(self.gcn3_p(p_conv2, feature_p))
        p_conv4 = self.gcn4_p(p_conv3, feature_p)
        n_conv1 = nn.LeakyReLU(0.2, inplace=True)(self.gcn1_n(X, feature_n))
        n_conv2 = nn.LeakyReLU(0.2, inplace=True)(self.gcn2_n(n_conv1, feature_n))
        n_conv3 = nn.LeakyReLU(0.2, inplace=True)(self.gcn3_n(n_conv2, feature_n))
        n_conv4 = self.gcn4_n(n_conv3, feature_n)
        logits_p = self.DCL(p_conv3, p_conv4)
        logits_n = self.DCL(n_conv3, n_conv4)

        conv_concat = torch.cat([logits_p, logits_n], -1).reshape([-1, 2, self.input_size, self.output_size])
        # conv_concat = torch.cat([p_conv4, n_conv4], -1).reshape([-1, 2, self.input_size,self.output_size])
        x = F.dropout(conv_concat, p=0.5)
        x_t = x.permute(0, 1, 3, 2)
        x_mm = torch.matmul(x, x_t)
        out1 = self.sigmoid_output(x_mm) * (-1)
        return out1

class GeneratorBA(nn.Module):
    """
       Encoder-Decoder network.
       """

    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, dropout, batch_size, class_nums):
        super(GeneratorBA, self).__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_size = output_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.class_nums= class_nums
        self.weight_std = False
        self.in_place = True
        self.gcn1_p = GraphConvolution(self.input_size, self.hidden1)
        self.gcn2_p = GraphConvolution(self.hidden1, self.hidden2)
        self.gcn3_p = GraphConvolution(self.hidden2, self.hidden3)
        self.gcn4_p = GraphConvolution(self.hidden3, self.output_size)
        self.gcn1_n = GraphConvolution(self.input_size, self.hidden1)
        self.gcn2_n = GraphConvolution(self.hidden1, self.hidden2)
        self.gcn3_n = GraphConvolution(self.hidden2, self.hidden3)
        self.gcn4_n = GraphConvolution(self.hidden3, self.output_size)
        self.init_weights = True

        self.sigmoid_output = nn.Sequential(nn.LogSigmoid())
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.controller = nn.Conv1d(32 + self.class_nums, 72, 1, bias=False)

        if self.init_weights:
            self._initialize_weights()

    def heads_forward(self, features, weights, biases, num_insts):
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias = b, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(-1, channels, 1,)
                bias_splits[l] = bias_splits[l].reshape(-1)
            else:
                weight_splits[l] = weight_splits[l].reshape(-1, channels, 1,)
                bias_splits[l] = bias_splits[l].reshape(-1)

        return weight_splits, bias_splits

    def normalize(self, adj):
        one_batch = []
        for batchsize in adj:
            temp_norm = []
            for reduce in batchsize:
                reduce = reduce + torch.eye(reduce.shape[0]).cuda()
                rowsum = reduce.sum(1)
                degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
                degree_mat_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
                adj_normalized = torch.mm(torch.mm(degree_mat_inv_sqrt, reduce), degree_mat_sqrt)
                temp_norm.append(adj_normalized.unsqueeze(0))
            temp_norm_ = torch.cat(temp_norm, 0)
            one_batch.append(temp_norm_.unsqueeze(0))
        one_batch = torch.cat(one_batch, 0)
        return one_batch

    def DCL(self, conv3, task_encoding, conv4):
        print('task_encoding.shape: ',task_encoding.shape)#torch.Size([32, 16, 1])
        x_conv = conv3.permute(0, 2, 1).contiguous()
        print('x_conv.shape: ',x_conv.shape)#torch.Size([32, 32, 200])
        x_feat = self.GAP(x_conv)
        print('x_feat.shape: ',x_feat.shape)#torch.Size([32, 32, 1])
        x_cond = torch.cat([x_feat, task_encoding], 1)
        print('x_cond.shape: ',x_cond.shape)#torch.Size([32, 48, 1])
        params = self.controller(x_cond)
        print('parames.shape: ',params.shape)#torch.Size([32, 72, 1])
        params.squeeze_(-1)
        N_, H_, W_ = conv4.size()
        head_inputs = conv4.view(1, -1, H_)  # 变化成一个维度进行组卷积
        print('head_inputs.shape: ',head_inputs.shape)#head_inputs.shape:  torch.Size([1, 256, 200])
        weight_nums, bias_nums = [], []
        weight_nums.append(self.output_size * self.output_size)
        bias_nums.append(self.output_size)
        weights, biases = self.parse_dynamic_params(params, self.output_size, weight_nums, bias_nums)
        logits = self.heads_forward(head_inputs, weights, biases, N_)
        print('logits.shape: ',logits.shape)#logits.shape:  torch.Size([1, 256, 200])
        logits = logits.reshape(N_, H_, -1)
        return logits

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 16))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()

    def forward(self, adj,task_id):
        X = torch.eye(self.input_size).cuda()
        adj = self.normalize(adj)
        A = torch.transpose(adj, 1, 0)
        feature_p = A[0]
        feature_n = A[1]

        p_conv1 = nn.LeakyReLU(0.2, inplace=True)(self.gcn1_p(X, feature_p))
        p_conv2 = nn.LeakyReLU(0.2, inplace=True)(self.gcn2_p(p_conv1, feature_p))
        p_conv3 = nn.LeakyReLU(0.2, inplace=True)(self.gcn3_p(p_conv2, feature_p))
        p_conv4 = self.gcn4_p(p_conv3, feature_p)
        n_conv1 = nn.LeakyReLU(0.2, inplace=True)(self.gcn1_n(X, feature_n))
        n_conv2 = nn.LeakyReLU(0.2, inplace=True)(self.gcn2_n(n_conv1, feature_n))
        n_conv3 = nn.LeakyReLU(0.2, inplace=True)(self.gcn3_n(n_conv2, feature_n))
        n_conv4 = self.gcn4_n(n_conv3, feature_n)

        task_encoding = self.encoding_task(task_id)
        task_encoding.unsqueeze_(2)
        logits_p = self.DCL(p_conv3,task_encoding,p_conv4)
        logits_n = self.DCL(n_conv3, task_encoding, n_conv4)
        conv_concat = torch.cat([logits_p, logits_n], -1).reshape([-1, 2, self.input_size,self.output_size])

        x = F.dropout(conv_concat, p=0.5)
        x_t = x.permute(0, 1, 3, 2)
        x_mm = torch.matmul(x, x_t)
        out1 = self.sigmoid_output(x_mm) * (-1)
        return out1


class DiscriminatorA(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, dropout, batch_size, class_nums):
        super(DiscriminatorA, self).__init__()

        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_size = output_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.class_nums = class_nums
        self.gcn1_p = GraphConvolution(self.input_size, self.hidden1)
        self.gcn2_p = GraphConvolution(self.hidden1, self.hidden2)
        self.gcn3_p = GraphConvolution(self.hidden2, self.hidden3)
        self.gcn4_p = GraphConvolution(self.hidden3, self.output_size)
        self.gcn1_n = GraphConvolution(self.input_size, self.hidden1)
        self.gcn2_n = GraphConvolution(self.hidden1, self.hidden2)
        self.gcn3_n = GraphConvolution(self.hidden2, self.hidden3)
        self.gcn4_n = GraphConvolution(self.hidden3, self.output_size)
        self.attr_vec = None
        self.lin1 = nn.Sequential(nn.Linear(2 * self.input_size * self.output_size, 1))
        self.lin2 = nn.Sequential(
            nn.Linear(2 * self.input_size * self.output_size, self.class_nums),
            nn.Softmax()
        )

    def normalize(self, adj):
        one_batch = []
        for batchsize in adj:
            temp_norm = []
            for reduce in batchsize:
                reduce = reduce + torch.eye(reduce.shape[0]).cuda()
                rowsum = reduce.sum(1)
                degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
                degree_mat_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
                adj_normalized = torch.mm(torch.mm(degree_mat_inv_sqrt, reduce), degree_mat_sqrt)
                temp_norm.append(adj_normalized.unsqueeze(0))
            temp_norm_ = torch.cat(temp_norm, 0)
            one_batch.append(temp_norm_.unsqueeze(0))
        one_batch = torch.cat(one_batch, 0)
        return one_batch

    def forward(self, adj):
        X = torch.eye(self.input_size).cuda()
        adj = self.normalize(adj)
        A = torch.transpose(adj, 1, 0)
        feature_p = A[0]
        feature_n = A[1]

        p_conv1 = nn.LeakyReLU(0.2, inplace=True)(self.gcn1_p(X, feature_p))
        p_conv2 = nn.LeakyReLU(0.2, inplace=True)(self.gcn2_p(p_conv1, feature_p))
        p_conv3 = nn.LeakyReLU(0.2, inplace=True)(self.gcn3_p(p_conv2, feature_p))
        p_conv4 = self.gcn4_p(p_conv3, feature_p)
        n_conv1 = nn.LeakyReLU(0.2, inplace=True)(self.gcn1_n(X, feature_n))
        n_conv2 = nn.LeakyReLU(0.2, inplace=True)(self.gcn2_n(n_conv1, feature_n))
        n_conv3 = nn.LeakyReLU(0.2, inplace=True)(self.gcn3_n(n_conv2, feature_n))
        n_conv4 = self.gcn4_n(n_conv3, feature_n)
        conv_concat = torch.cat([p_conv4, n_conv4], -1).reshape([-1, 2 * self.input_size * self.output_size])
        out1 = self.lin1(conv_concat)
        out2 = self.lin2(conv_concat)
        return out1,out2


class DiscriminatorB(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3,output_size, dropout, batch_size):
        super(DiscriminatorB, self).__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_size = output_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.gcn1_p = GraphConvolution(self.input_size, self.hidden1)
        self.gcn2_p = GraphConvolution(self.hidden1, self.hidden2)
        self.gcn3_p = GraphConvolution(self.hidden2, self.hidden3)
        self.gcn4_p = GraphConvolution(self.hidden3, self.output_size)
        self.gcn1_n = GraphConvolution(self.input_size, self.hidden1)
        self.gcn2_n = GraphConvolution(self.hidden1, self.hidden2)
        self.gcn3_n = GraphConvolution(self.hidden2, self.hidden3)
        self.gcn4_n = GraphConvolution(self.hidden3, self.output_size)
        self.attr_vec = None
        self.lin1 = nn.Sequential(nn.Linear(2 * self.input_size * self.output_size, self.output_size))

    def normalize(self, adj):
        one_batch = []
        for batchsize in adj:
            temp_norm = []
            for reduce in batchsize:
                reduce = reduce + torch.eye(reduce.shape[0]).cuda()
                rowsum = reduce.sum(1)
                degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
                degree_mat_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
                adj_normalized = torch.mm(torch.mm(degree_mat_inv_sqrt, reduce), degree_mat_sqrt)
                temp_norm.append(adj_normalized.unsqueeze(0))
            temp_norm_ = torch.cat(temp_norm, 0)
            one_batch.append(temp_norm_.unsqueeze(0))
        one_batch = torch.cat(one_batch, 0)
        return one_batch

    def forward(self, adj):
        X = torch.eye(self.input_size).cuda()
        adj = self.normalize(adj)
        A = torch.transpose(adj, 1, 0)
        feature_p = A[0]
        feature_n = A[1]

        p_conv1 = nn.LeakyReLU(0.2, inplace=True)(self.gcn1_p(X, feature_p))
        p_conv2 = nn.LeakyReLU(0.2, inplace=True)(self.gcn2_p(p_conv1, feature_p))
        p_conv3 = nn.LeakyReLU(0.2, inplace=True)(self.gcn3_p(p_conv2, feature_p))
        p_conv4 = self.gcn4_p(p_conv3, feature_p)
        n_conv1 = nn.LeakyReLU(0.2, inplace=True)(self.gcn1_n(X, feature_n))
        n_conv2 = nn.LeakyReLU(0.2, inplace=True)(self.gcn2_n(n_conv1, feature_n))
        n_conv3 = nn.LeakyReLU(0.2, inplace=True)(self.gcn3_n(n_conv2, feature_n))
        n_conv4 = self.gcn4_n(n_conv3, feature_n)
        conv_concat = torch.cat([p_conv4, n_conv4], -1).reshape([-1, 2 * self.input_size * self.output_size])
        out1 = self.lin1(conv_concat)
        return out1

class GCN_gnn(nn.Module):
    def __init__(self, in_dim, out_dim, neg_penalty):
        super(GCN_gnn, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neg_penalty = neg_penalty
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.c = 0.85
        self.losses = []

    def forward(self, x, adj):
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()
        if x is None:
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        I_cAXW = eye + self.c * AXW
        y_relu = torch.nn.functional.relu(I_cAXW)
        temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        col_mean = temp.repeat([1, feature_dim, 1])
        y_norm = torch.divide(y_relu, col_mean)
        output = torch.nn.functional.softplus(y_norm)
        if self.neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(self.neg_penalty),
                                      torch.sum(torch.nn.functional.relu(1e-6 - self.kernel)))
            self.losses.append(neg_loss)
        return output

class model_gnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,dim):
        super(model_gnn, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dim = dim
        self.gcn1_p = GCN_gnn(in_dim, hidden_dim, 0.2)
        self.gcn2_p = GCN_gnn(in_dim, hidden_dim, 0.2)
        self.gcn1_n = GCN_gnn(in_dim, hidden_dim, 0.2)
        self.gcn2_n = GCN_gnn(in_dim, hidden_dim, 0.2)
        self.kernel_p = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n = nn.Parameter(torch.FloatTensor(dim, in_dim))
        self.lin1 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2 = nn.Linear(16, self.out_dim)
        self.losses = []
        self.reset_weigths()
        self.sigmoid_output = nn.Sequential(
            nn.Sigmoid()
        )

    def dim_reduce(self, adj_matrix, num_reduce,
                   ortho_penalty, variance_penalty, neg_penalty, kernel):
        kernel_p = torch.nn.functional.relu(kernel)
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(
            torch.tensordot(kernel_p, AF, [[0], [1]]),  # num_reduce*batch*num_reduce
            1, 0)  # num_reduce*batch*num_reduce*num_reduce
        kernel_p_tran = kernel_p.transpose(-1, -2)  # num_reduce * column_dim
        gram_matrix = torch.matmul(kernel_p_tran, kernel_p)
        diag_elements = gram_matrix.diag()

        if ortho_penalty != 0:
            ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
            ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
            self.losses.append(ortho_loss)

        if variance_penalty != 0:
            variance = diag_elements.var()
            variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
            self.losses.append(variance_loss)

        if neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                      torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
            self.losses.append(neg_loss)
        self.losses.append(0.05 * torch.sum(torch.abs(kernel_p)))
        return reduced_adj_matrix

    def reset_weigths(self):
        """reset weights
            """
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, A):
        A = torch.transpose(A, 1, 0)
        s_feature_p = A[0]
        s_feature_n = A[1]

        p_reduce = self.dim_reduce(s_feature_p, self.in_dim, 0.2, 0.3, 0.1, self.kernel_p)
        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        n_reduce = self.dim_reduce(s_feature_n, self.in_dim, 0.2, 0.5, 0.1, self.kernel_n)
        n_conv1 = self.gcn1_n(None, n_reduce)
        n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        conv_concat = torch.cat([p_conv2, n_conv2], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        output = self.lin2(self.lin1(conv_concat))
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        return output, loss