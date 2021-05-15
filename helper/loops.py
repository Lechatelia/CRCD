from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy
import torch.nn.functional as F


def adjust_mimic_loss_weight(cur_iter, total_iter, alpha=1, factor=2,adjust_loss_type="linear_decrease2"):
    iter_third = total_iter / 3
    if adjust_loss_type == "constant":
        mimic_loss_weights = 1.0
        task_loss_weight = 1.0
    elif adjust_loss_type == '240':
        if cur_iter < total_iter*24.0/25:
            mimic_loss_weights = 1.0
            task_loss_weight = 1.0
        else:
            mimic_loss_weights = 0.0
            task_loss_weight = 1.0
    elif adjust_loss_type == "linear_decrease":
        if cur_iter < iter_third:
            mimic_loss_weights = 1.0
            task_loss_weight = 0.0
        elif cur_iter < iter_third*2:
            mimic_loss_weights = 2 - cur_iter/iter_third
            task_loss_weight = 1-mimic_loss_weights
        else:
            mimic_loss_weights =0.0
            task_loss_weight = 1.0
    elif adjust_loss_type == "linear_decrease2":
        task_loss_weight = 1.0
        if cur_iter < iter_third:
            mimic_loss_weights = 1.0
        elif cur_iter < iter_third*2:
            mimic_loss_weights = 2.0 - cur_iter/iter_third
        else:
            mimic_loss_weights = 0.0

    elif adjust_loss_type == "exp_increase":
        mimic_loss_weights = alpha * (factor ** (cur_iter // iter_third))
    else:
        raise NotImplementedError

    return task_loss_weight, mimic_loss_weights

def gard_cos_sim(grad_t, grad_s):
    grad_t = grad_t.view(grad_t.size(0), -1)
    grad_s = grad_s.view(grad_s.size(0), -1)
    # return F.mse_loss(grad_s, grad_t)
    return 1 - F.cosine_similarity(grad_t, grad_s, dim=1).mean()


def get_gradnh_loss(feat_t, feat_s, loss_cls_teacher, loss_cls_student,  criterion_grad, index=None, contrast_idx=None):
    gm_loss = []
    teacher_grads = torch.autograd.grad([loss_cls_teacher],
                                       [feat_t],
                                       create_graph=True,
                                       only_inputs=True)
    student_grads = torch.autograd.grad([loss_cls_student],
                                       [feat_s],
                                       create_graph=True,
                                       only_inputs=True)
    for teacher_grad, student_grad in zip(teacher_grads, student_grads):
        if index is not None:
            gm_loss.append(criterion_grad(student_grad, teacher_grad, index, contrast_idx))
        else:
            gm_loss.append(criterion_grad(student_grad, teacher_grad))
    return sum(gm_loss)


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    # module_list[-1].train()

   
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    criterion_kd_grad = criterion_list[3]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_kd = AverageMeter()
    losses_grad = AverageMeter()
    losses_mimic = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):

        input, target, index, contrast_idx = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        # with torch.no_grad():
        feat_t, logit_t = model_t(input, is_feat=True, preact=preact)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_cls_teacher = criterion_cls(logit_t, target)
        loss_div = criterion_div(logit_s, logit_t)

        if opt.with_grad:
            if opt.grad_kd_type == 'mse':
                loss_gm = 10 * get_grad_loss(feat_t[-1], feat_s[-1], loss_cls_teacher, loss_cls, gard_cos_sim) * opt.grad_kd_weight
            else: #  ['crd', 'crcd',']
                loss_gm = get_grad_loss(feat_t[-1], feat_s[-1], loss_cls_teacher, loss_cls, criterion_kd_grad, index, contrast_idx) * opt.grad_kd_weight
        else:
            loss_gm = torch.tensor(0.0).to(loss_div.device)
        # list(self.layer3[0].conv2.parameters())
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = torch.tensor(0.0).to(loss_div.device)
        
        elif opt.distill == 'crd' or 'crcd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        else:
            raise NotImplementedError(opt.distill)

        # it is critical to adjust loss weight dynamically
        task_loss_weight, mimic_loss_weights = adjust_mimic_loss_weight((epoch-1)*len(train_loader)+idx,opt.epochs*len(train_loader), adjust_loss_type='240')
        
        loss = opt.gamma * loss_cls * task_loss_weight + opt.alpha * loss_div*mimic_loss_weights + opt.beta * loss_kd*mimic_loss_weights
        # loss = opt.gamma * loss_cls  + opt.alpha * loss_div + opt.beta * loss_kd
        if opt.with_grad:
            loss = loss + opt.alpha * loss_div* loss_gm

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_kd.update(loss_div.item(), input.size(0))
        losses_grad.update(loss_gm.item(), input.size(0))
        losses_mimic.update(loss_kd.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_t.zero_grad()
        optimizer.zero_grad()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                  'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                  'Loss_kd {loss_kd.val:.2f} ({loss_kd.avg:.2f})\t'
                  'Loss_grad {loss_grad.val:.2f} ({loss_grad.avg:.2f})\t'
                  'Loss_mimic {loss_mimic.val:.2f} ({loss_mimic.avg:.2f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Acc@5 {top5.val:.2f} ({top5.avg:.2f})\t'
                  'w1:{w1:.2f} w2:{w2:.2f}'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_kd= losses_kd, loss_mimic=losses_mimic, loss_grad=losses_grad,
                top1=top1, top5=top5,
                w1=task_loss_weight, w2=mimic_loss_weights))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg
