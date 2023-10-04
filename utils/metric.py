import torch
from sklearn.metrics import classification_report
import numpy as np

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

def AccuracyEval(logits, target):
    # for evaluation
    logits = logits.detach().cpu()
    target = target.detach().cpu()

    preds = logits.argmax(dim=-1)
    
    assert preds.shape == target.shape
    print(classification_report(y_pred=preds, y_true=target)) # print classification report

def EmoScoreEval(logits, target):
    logits = logits.detach().cpu()
    target = target.detach().cpu()
    
    preds = torch.where(logits > 0.5, 1, 0).long()
    
    emo_class_labels = ["Happy", "Sad", "Anger", "Disgust", "Surprise", "Fear"]
    res_dict = {emo: {"pred":[], "target": []} for emo in emo_class_labels}
    
    for i in range(len(logits)):
        for j in range(len(logits[0])):
            res_dict[emo_class_labels[j]]["pred"].append(int(preds[i][j]))
            res_dict[emo_class_labels[j]]["target"].append(int(target[i][j]))
            
    for emo in emo_class_labels:
        print('Classification result for ' + emo)
        print(classification_report(y_true=np.array(res_dict[emo]['target']), y_pred=np.array(res_dict[emo]['pred'])))
