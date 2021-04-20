import torch
from torch import nn
import math
import torch.nn.functional as F

class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.05, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        norm1 = self.memory_v1.pow(2).sum(1, keepdim=True).pow(1. / 2)
        self.memory_v1 = self.memory_v1.div(norm1)
        norm2 = self.memory_v2.pow(2).sum(1, keepdim=True).pow(1. / 2)
        self.memory_v2 = self.memory_v2.div(norm2)

    def forward(self, v1, v2, y, idx=None):
        '''
        Args:
            v1: the feature of student network, size [batch_size, s_dim]
            v2: the feature of teacher network, size [batch_size, t_dim]
            y: the indices of these positive samples in the dataset, size [batch_size]
            idx: the indices of negative samples, size [batch_size, nce_k]
        '''
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        
        # since there are learnable parameters in embedding layer in teacher side, 
        # similar to crd, we also calculate the crcd over the teacher and the student side

        # compute anchor relation over the student side
        # choose anchor from teacher
        anchor_v2 = torch.index_select(self.memory_v2, 0, y.view(-1)).detach() # na, f
        # choose teacher feature for contrastive loss
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize) # b, (1+k) f
        # anchor-student relation
        anchor_relation = v1.unsqueeze(1) - anchor_v2.unsqueeze(0) + 1e-6 # b, na, f
        anchor_relation = anchor_relation.view(batchSize*batchSize, inputSize) #b*na, f
        anchor_relation = F.normalize(anchor_relation, p=2, dim=1)
        # anchor-teacher relation
        weight_v2_relation = weight_v2.unsqueeze(1) - anchor_v2.unsqueeze(1).unsqueeze(0) + 1e-6 # b, na, k+1, f
        weight_v2_relation = weight_v2_relation.view(batchSize*batchSize, self.K+1, inputSize) #b*na, K+1, f
        weight_v2_relation = F.normalize(weight_v2_relation, p=2, dim=2)
        out_v1 = torch.bmm(weight_v2_relation, anchor_relation.view(batchSize*batchSize, inputSize, 1)) # b*ba, K+1, f
        out_v1 = torch.exp(torch.div(out_v1, T))
        
        # compute anchor relation over the student side
        # choose anchor from student
        anchor_v1 = torch.index_select(self.memory_v1, 0, y.view(-1)).detach() # na, f
        # choose student features for contrastive loss
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize) # b, (1+k) f
        # anchor-teacher relation
        anchor_relation = v2.unsqueeze(1) - anchor_v1.unsqueeze(0) + 1e-6 # b, na, f
        anchor_relation = anchor_relation.view(batchSize*batchSize, inputSize) #b*na, f
        anchor_relation = F.normalize(anchor_relation, p=2, dim=1)
        # anchor-student relation
        weight_v1_relation = weight_v1.unsqueeze(1) - anchor_v1.unsqueeze(1).unsqueeze(0) + 1e-6 # b, na, k+1, f
        weight_v1_relation = weight_v1_relation.view(batchSize*batchSize, self.K+1, inputSize) #b*na, K+1, f
        weight_v1_relation = F.normalize(weight_v1_relation, p=2, dim=2)
        out_v2 = torch.bmm(weight_v1_relation, anchor_relation.view(batchSize*batchSize, inputSize, 1)) # b*ba, K+1, 1
        out_v2 = torch.exp(torch.div(out_v2, T))
        

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2

class ContrastMemory_queue(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory_queue, self).__init__()
        self.nLem = K # 
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams) # 
        self.multinomial.cuda()
        self.K = K
        self.T = T
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(K, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(K, inputSize).mul_(2 * stdv).add_(-stdv))
        norm1 = self.memory_v1.pow(2).sum(1, keepdim=True).pow(1. / 2)
        self.memory_v1 = self.memory_v1.div(norm1)
        norm2 = self.memory_v2.pow(2).sum(1, keepdim=True).pow(1. / 2)
        self.memory_v2 = self.memory_v2.div(norm2)

        self.outputSize = self.memory_v1.size(0)  # the size of train dataset
        self.inputSize = self.memory_v1.size(1)  # the feature len
        self.ptr = 0

    def forward(self, v1, v2, y, idx=None):
        batchSize = v1.size(0)
        assert self.outputSize > batchSize

        # index
        y = (torch.arange(batchSize).cuda(device=v1.device) + self.ptr) % self.outputSize

        self.ptr = (self.ptr + batchSize)% self.outputSize

       
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1)) # bs, f
            l_pos.mul_(self.momentum)
            l_pos.add_(torch.mul(v1, 1 - self.momentum)) 
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5) #
            updated_v1 = l_pos.div(l_norm) # 
            self.memory_v1.index_copy_(0, y, updated_v1) # 

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(self.momentum)
            ab_pos.add_(torch.mul(v2, 1 - self.momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        weight_v1 = self.memory_v1.detach() # K, f
        simi_v1  = torch.mm(v2, torch.t(weight_v1)) # bs, K
        simi_v2_own = torch.gather(simi_v1, dim=1, index=y.unsqueeze(dim=1)) #
        out_v2 = torch.cat([simi_v2_own, simi_v1 ], dim=1) #[bs, K+1]
        out_v2 = torch.exp(torch.div(out_v2, self.T)) # 
        # sample
        weight_v2 = self.memory_v2.detach()
        simi_v2 = torch.mm(v1, torch.t(weight_v2))
        simi_v1_own = torch.gather(simi_v2, dim=1, index=y.unsqueeze(dim=1))
        out_v1 = torch.cat([simi_v1_own, simi_v2], dim=1)
        out_v1 = torch.exp(torch.div(out_v1, self.T))  
        return out_v1, out_v2

class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj