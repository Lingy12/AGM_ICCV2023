import torch

def Accuracy(logits,target):
    logits = logits.detach().cpu()
    target = target.detach().cpu()

    preds = logits.argmax(dim=-1)
    assert preds.shape == target.shape
    correct = torch.sum(preds==target)
    total = torch.numel(target)
    if total == 0:
        return 1
    else:
        return correct / total
    
def EmoScore(logits, target):
    # logit is of shape (batchsize, 6) same as target
    logits = logits.detach().cpu()
    target = target.detach().cpu().long()
    
    # print(target)
    preds = torch.where(logits > 0.5, 1, 0).long()
    
    n_xor_res = (~torch.bitwise_xor(preds.bool(), target.bool())).long()
    correct_emo_count = n_xor_res.sum(dim=1)

    emo_score = correct_emo_count / 6
    return torch.mean(emo_score)
    