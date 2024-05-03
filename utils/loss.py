import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
    def forward(self,logits):
        n,c,h,w = logits.shape
        loss_all = []
        for i in range(0,h-1):
            loss_all.append(logits[:,:,i,:] - logits[:,:,i+1,:])
        #loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss,torch.zeros_like(loss))



class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()
    def forward(self, x):
        n,dim,num_rows,num_cols = x.shape
        x = torch.nn.functional.softmax(x[:,:dim-1,:,:],dim=1)
        embedding = torch.Tensor(np.arange(dim-1)).float().to(x.device).view(1,-1,1,1)
        pos = torch.sum(x*embedding,dim = 1)

        diff_list1 = []
# modify to consider which similarity loss to use
        if SIM_DIST_LOSS == "dtw_loss":
            for i in range(0,num_rows // 2):
                diff_list1.append(dtw_distance(pos[:,i,:] , pos[:,i+1,:]))
# 
        elif SIM_DIST_LOSS == "orig_sim_loss":
            for i in range(0,num_rows // 2):
                diff_list1.append(pos[:,i,:] - pos[:,i+1,:])
        else: 
            print("Check the similarity/distance Loss function")
            raise NotImplementedError            

        loss = 0
        for i in range(len(diff_list1)-1):
            loss += self.l1(diff_list1[i],diff_list1[i+1])
        loss /= len(diff_list1) - 1
        return loss
    
    # Dynamic Time Warping (DTW) loss 
    def dtw_distance(s1, s2):
        """
        Calculate the Dynamic Time Warping (DTW) distance between two sequences.
        s1, s2: Sequences of predictions to compare.
        """
        # Initialize cost matrix
        n, m = len(s1), len(s2)
        cost = np.zeros((n, m))
        
        # Fill the cost matrix
        for i in range(n):
            for j in range(m):
                cost[i, j] = abs(s1[i] - s2[j]) ** 2
        
        # Initialize accumulated cost matrix
        accumulated_cost = np.zeros((n, m))
        
        # Fill the accumulated cost matrix
        accumulated_cost[0, 0] = cost[0, 0]
        for i in range(1, n):
            accumulated_cost[i, 0] = accumulated_cost[i - 1, 0] + cost[i, 0]
        for j in range(1, m):
            accumulated_cost[0, j] = accumulated_cost[0, j - 1] + cost[0, j]
        
        # Calculate the accumulated cost
        for i in range(1, n):
            for j in range(1, m):
                accumulated_cost[i, j] = cost[i, j] + min(accumulated_cost[i - 1, j], 
                                                        accumulated_cost[i, j - 1], 
                                                        accumulated_cost[i - 1, j - 1])
        
        # Return the DTW distance
        return np.sqrt(accumulated_cost[-1, -1])
    



