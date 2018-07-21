import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CRF(nn.Module):
    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        # Matrix of transition parameters. Entry i,j is the score of transitioning from j to i. We have 2 additional TAGs which are START_TAG and STOP_TAG
        self.transitions = nn.Parameter(torch.rand(self.tagset_size, self.tagset_size))

        # Score of transitions from START_TAG to other tag
        self.transitions_start = nn.Parameter(torch.rand(self.tagset_size))
        # Score of transitions from other tag to STOP_TAG
        self.transitions_stop = nn.Parameter(torch.rand(self.tagset_size))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(0)
        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        if use_cuda:
            forward_var = forward_var.cuda()

        # Iterate through the timestep
        alpha = []
        for i, feat in enumerate(feats):
            if i == 0: # the first timestep, transfer from START_TAG to other stag
                forward_var = forward_var + self.transitions_start.view(1,-1) + feat.view(1,-1)
            else:
                alphas_t = [] # The forward variables at this timestep
                for next_tag in xrange(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of the previous tag
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                    # the ith entry of trans_score is the score of transitioning to next_tag from i
                    trans_score = self.transitions[next_tag].view(1, -1)
                    # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                    # next_tag_var = forward_var + trans_score + emit_score
                    # The forward variable for this tag is log-sum-exp of all the scores.
                    alphas_t.append(log_sum_exp(forward_var + trans_score + emit_score))
                forward_var = torch.cat(alphas_t).view(1, -1)
            alpha.append(forward_var)

        # after last timestep, transition to STOP_TAG
        terminal_var = forward_var + self.transitions_stop.view(1, -1)

        return log_sum_exp(terminal_var), torch.cat(alpha , 0)

    def _viterbi_decode(self, feats):
        backpointers = []      
        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(0.)
 
        # forward_var at step i holds the viterbi variables for step i-1 
        forward_var = autograd.Variable(init_vvars)
        if use_cuda:
            forward_var = forward_var.cuda()

        # Iterate through the timestep
        for i, feat in enumerate(feats):
            if i == 0:  # the first timestep, transfer from START_TAG to other stag. There is no need to reored the backpointers for first time step, since the first tag must be START_TAG
                forward_var = forward_var + self.transitions_start.view(1, -1) + feat.view(1,-1)
            else:
                bptrs_t = [] # holds the backpointers for this step
                viterbivars_t = [] # holds the viterbi variables for this step
                for next_tag in xrange(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the previous step plus the score of transitioning from tag i to next_tag.
                    # We don't include the emission scores here because the max does not depend on them (we add them in below)
                    next_tag_var = forward_var + self.transitions[next_tag].view(1, -1)
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0,best_tag_id])
                # Now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

        # after last timestep, transition to STOP_TAG
        terminal_var = forward_var + self.transitions_stop.view(1, -1)
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0, best_tag_id]
        
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_path.reverse()
        return path_score, best_path

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        if use_cuda:
            score = score.cuda()
        
        for i, feat in enumerate(feats):
            if i == 0:
                score = score + self.transitions_start[tags[i]] + feat[tags[i]]
            else:
                score = score + self.transitions[tags[i], tags[i-1]] + feat[tags[i]]
        score = score + self.transitions_stop[tags[-1]]
        return score

    def _get_neg_log_likilihood_loss(self, feats, tags):
        # nonegative log likelihood
        forward_score, _ = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score