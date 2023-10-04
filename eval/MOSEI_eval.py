from sklearn.metrics import classification_report

import warnings
import time
import torch
import os
import math
import torch.nn as nn
import numpy as np
from utils.metric import AccuracyEval, EmoScoreEval
from tasks.MOSEI_task import Mosei_Task
from utils.function_tools import get_device,set_seed

def get_text_audio_score_sent(out_t, out_a, ans, transform):
    score_text = 0.
    score_audio = 0.
    for k in range(out_t.size(0)):
                # print(softmax(out_t))
                if torch.isinf(torch.log(transform(out_t)[k][ans[k]])) or transform(out_t)[k][ans[k]] < 1e-8:
                    score_text += - torch.log(torch.tensor(1e-8,dtype=out_t.dtype,device=out_t.device))
                else:
                    score_text += - torch.log(transform(out_t)[k][ans[k]])
                    
                if torch.isinf(torch.log(transform(out_a)[k][ans[k]])) or transform(out_a)[k][ans[k]] < 1e-8:
                    score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                else:
                    score_audio += - torch.log(transform(out_a)[k][ans[k]])
    return score_text, score_audio
    
def get_text_audio_score_emo(out_t, out_a, ans, transform):
    score_text = 0.
    score_audio = 0.
    
    prob_t, prob_a = transform(out_t), transform(out_a)
     
    for k in range(out_t.size(0)):
        t_score, a_score = torch.dot(prob_t[k], ans[k]), torch.dot(prob_a[k], ans[k]) # activate the presenting emotion
        if torch.isinf(torch.log(t_score)) or t_score < 1e-8:
            score_text += - torch.log(torch.tensor(1e-8,dtype=out_t.dtype,device=out_t.device))
        else:
            score_text += - torch.log(t_score)

        if torch.isinf(torch.log(a_score)) or a_score < 1e-8:
            score_audio += - torch.log(torch.tensor(1e-8,dtype=out_t.dtype,device=out_t.device))
        else:
            score_audio += - torch.log(a_score)
    return score_text, score_audio

def validate(model,validate_dataloader,cfgs,device):
    softmax = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    if cfgs.task == 'emotion':
        eval_func = EmoScoreEval
    else:
        eval_func = AccuracyEval
    pred_lst = []
    target_lst = []
    with torch.no_grad():
        if cfgs.use_mgpu:
            model.module.eval()
        else:
            model.eval()
        if cfgs.modality == "Audio":
            validate_audio_acc = 0.
            validate_score_a = 0.
        elif cfgs.modality == "Text":
            validate_text_acc = 0.
            validate_score_t = 0.
        elif cfgs.modality == "Multimodal":
            model.mode = "eval"
            validate_acc = 0.
            validate_text_acc = 0.
            validate_audio_acc = 0.
            validate_score_t = 0.
            validate_score_a = 0.
        total_batch = len(validate_dataloader)
        start_time = time.time()
        for step,(id,x,y,z,ans) in enumerate(validate_dataloader):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            ans = ans.to(device)
            if cfgs.modality == "Audio":
                # out_a = model.net(x,y,pad_x = True,pad_y = False)
                # pred_a = softmax(out_a)
                # audio_accuracy = (pred_a,ans)
                # validate_audio_acc += audio_accuracy.item() / total_batch

                # score_audio = 0.
                # for k in range(out_a.size(0)):   
                #     if torch.isinf(torch.log(softmax(out_a)[k][ans[k]])) or softmax(out_a)[k][ans[k]] < 1e-8:
                #         score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.devcie))
                #     else:
                #         score_audio += - torch.log(softmax(out_a)[k][ans[k]])
                # score_audio = score_audio / out_a.size(0)
                # validate_score_a = validate_score_a * step / (step + 1) + score_audio.item() / (step + 1)
                print('Audio eval not supported. ')
            elif cfgs.modality == "Text":
                # out_t = model.net(x,y,pad_x = False,pad_y = True)
                # pred_t = softmax(out_t)
                # text_accuracy = eval_func(pred_t,ans)
                # validate_text_acc += text_accuracy.item() / total_batch

                # score_text = 0.
                # for k in range(out_t.size(0)):
                #     if torch.isinf(torch.log(softmax(out_t)[k][ans[k]])) or softmax(out_t)[k][ans[k]] < 1e-8:
                #         score_text += - torch.log(torch.tensor(1e-8,dtype=out_t.dtype,device=out_t.device))
                #     else:
                #         score_text += - torch.log(softmax(out_t)[k][ans[k]])
                # score_text = score_text / out_t.size(0)
                # validate_score_t = validate_score_t * step / (step + 1) + score_text.item() / (step + 1)
                print('Text eval not supported')
            elif cfgs.modality == "Multimodal":
                out_t,out_a,C,out = model(x,y,z)
                pred = softmax(out)
                # pred_t = softmax(out_t)
                # pred_a = softmax(out_a)
                pred_lst.append(pred.cpu())
                target_lst.append(ans.cpu())
                
                if torch.isnan(out_t).any() or torch.isnan(out_a).any():
                    raise ValueError
        preds = torch.concat(pred_lst)
        target = torch.concat(target_lst)
        eval_func(preds, target)


def MOSEI_eval(cfgs):
    warnings.filterwarnings('ignore')
    set_seed(cfgs.random_seed)
    ts = time.strftime('%Y_%m_%d %H:%M:%S',time.localtime())
    print('Evaluation start at {}'.format(ts))
    
    model_dir = os.path.join('models', cfgs.expt_name)
    if cfgs.use_mgpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.gpu_ids
        gpu_ids = list(map(int,cfgs.gpu_ids.split(",")))
        device = get_device(cfgs.device)
    else:
        device = get_device(cfgs.device)
        
    task =Mosei_Task(cfgs)
    test_dataloader = task.test_dataloader

    model = task.model
    model.to(device)
    
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best.pt')))
    print('loaded best model')
    if cfgs.use_mgpu:
        model = torch.nn.DataParallel(model,device_ids=gpu_ids)
        model.to(device)
    
    validate(model,test_dataloader,cfgs,device)
