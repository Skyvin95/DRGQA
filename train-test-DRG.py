import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from DRGQAnet import DRGQA
from IQAPerformance import IQAPerformance
from regression import regression
from regression import func1
from datasets.Dataset_processing import Dataset
# from thop import profile
# from thop import clever_format

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_batch_size = 8
exp_id = 0
EPOCH = 150
LEARNING_RATE = 0.0001
train_ratio = 0.8
test_ratio = 0.2
INIT_PRETRAINED_BOOL = 1

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

if (torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_log_file = r'/home/DRGQA-main/train-test-DRG.log'
logging.basicConfig(
    level=logging.INFO,
    format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename=loss_log_file,
    filemode='a')

model_name = 'DRGQA_checkpoint.pkl'

net = DRGQA()
net.to(device)

if os.path.exists(model_name):
    net.load_state_dict(model_name)

# train-test setting
dataset = Dataset()
train_size = int(train_ratio * len(dataset)) 
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
logging.info('train_dataset:{}'.format(len(train_dataset)))
logging.info('test_dataset:{}'.format(len(test_dataset)))

def test(net, epoch):
    net.eval()
    Score = []
    mos = []
    test_loader = torch.utils.data.DataLoader(test_dataset)

    net.eval()
    for batch_index, data in enumerate(test_loader):
        raw_img_t, enhanced_img_t, tex_t, color_t, mos_t = data
        raw_img_t = raw_img_t.cuda().float()
        enhanced_img_t = enhanced_img_t.cuda().float()
        tex_t = tex_t.cuda().float()
        color_t = color_t.cuda().float()
        mos_t = mos_t.cuda().float()

        S = net(enhanced_img_t, tex_t, color_t)
        Score.append(S.squeeze().cpu().detach().float())
        mos.append(mos_t.squeeze().cpu().detach().float())

    preds, fit_paras = regression(func1, np.array(Score).astype(float), np.array(mos).astype(float))
    srocc, krocc, plcc, rmse, mae, outlier_ratio = IQAPerformance(preds, mos)

    logging.info('srocc: {}'.format(srocc))
    logging.info('krocc: {}'.format(krocc))
    logging.info('plcc: {}'.format(plcc))
    logging.info('rmse: {}'.format(rmse))
    logging.info('mae: {}'.format(mae))

    plt.ion()
    plt.scatter(np.array(Score), np.array(mos), s=20, marker='*', alpha=0.8)
    x = np.arange(np.min(Score), np.max(Score), 0.1)
    plt.xlabel('Objective scores')
    plt.ylabel('MOS')
    plt.plot(x, func1(x, *fit_paras), "red")
    plt.savefig('/home/DRGQA-main/Figures/' + str(epoch) + '.png')
    plt.show()
    plt.pause(2)
    plt.close()

    ## Model parameter calculation
    # flops, params = profile(net, inputs=(enhanced_img_t, tex_t, color_t))
    # logging.info('*******************************')
    # logging.info('flops:{}\t params:{}'.format(flops, params))
    # Flops, Params = clever_format([flops, params], '%.3f')
    # logging.info('Flops:{}\t Params:{}'.format(Flops, Params))
    # logging.info('*******************************')


def Q_loss(labels, preds):
    loss = torch.nn.functional.l1_loss(preds, labels)
    return loss


def train(net, EPOCH):
    Writer = SummaryWriter('logs/')

    paras = net.parameters()
    optim_IQA = torch.optim.AdamW(params=paras, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_IQA, milestones=[100], gamma=0.1)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)

    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            raw_img, enhanced_img, tex, color, mos = data
            raw_img = raw_img.cuda().float()
            enhanced_img = enhanced_img.cuda().float()
            tex = tex.cuda().float()
            color = color.cuda().float()
            mos = mos.cuda().float()

            net.train()
            mos_pred = net(enhanced_img, tex, color)
            loss = Q_loss(mos.squeeze(), mos_pred.squeeze())

            optim_IQA.zero_grad()
            loss.mean().backward()
            optim_IQA.step()

            Writer.add_scalar('DRGQA_Net_training_loss', loss.mean().cpu().data.numpy(), global_step=i)

            if i % 50 == 0:
                logging.info('Epoch: {}\t Iteration: {}\t training loss: {}'.format(epoch, i, loss.mean().cpu().data.numpy()))

        scheduler.step()

        if epoch % 50 == 0:
            torch.save(net.state_dict(), 'DRGQA_checkpoint_epoch_' + str(epoch) + '.pkl')

        if epoch % 50 == 0 or epoch == 149:
            test(net, epoch)

    Writer.export_scalars_to_json("./all_scalars.json") 
    torch.save(net.state_dict(), os.path.join('DRGQA_model.pkl'))
    Writer.close()


if __name__ == "__main__":
        train(net, EPOCH)
