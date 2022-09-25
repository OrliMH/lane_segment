from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss, SoftDiceLoss
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet, UpConvResNetUNet
from config import Config
import math
from torch.optim.lr_scheduler import _LRScheduler

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device_list = [0]
train_net = 'deeplabv3p' # 'unet'
# nets['deeplabv3p']:DeeplabVePlus
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}


class CosineAnnealingLR_with_Restart(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in AdamW:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. The original pytorch
    implementation only implements the cosine annealing part of SGDR,
    I added my own implementation of the restarts part.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Increase T_max by a factor of T_mult
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        model (pytorch model): The model to save.
        out_dir (str): Directory to save snapshots
        take_snapshot (bool): Whether to save snapshots at every restart

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mult, model, out_dir, take_snapshot, eta_min=0, eta_max=[1e-2], last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.current_epoch = last_epoch

        self.model = model
        self.out_dir = out_dir
        self.take_snapshot = take_snapshot
        self.base_lrs = eta_max 
        self.lr_history = []

        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2
                   for base_lr in self.base_lrs]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        ## restart
        if self.current_epoch == self.Te:
            print("restart at epoch {:03d}".format(self.last_epoch + 1))

            if self.take_snapshot:
                torch.save({
                    'epoch': self.T_max,
                    'state_dict': self.model.state_dict()
                }, self.out_dir + "Weight/" + 'snapshot_e_{:03d}.pth.tar'.format(self.T_max))

            ## reset epochs since the last reset
            self.current_epoch = 0

            ## reset the next goal
            self.Te = int(self.Te * self.T_mult)
            self.T_max = self.T_max + self.Te

def loss_func(predict, target, nbclasses, epoch):
    ''' can modify or add losses ''
    '''
    # print("inside loss_func:")
    # print("dtype(predict):{}".format(predict.dtype))
    # print("type(target):{}".format(target.dtype))
    ce_loss = MySoftmaxCrossEntropyLoss(nbclasses=nbclasses)(predict, target)
    dice_loss = SoftDiceLoss(nbclasses)(predict, target)
    # loss = (ce_loss + dice_loss)/2.
    loss = ce_loss + dice_loss
    return loss


def train_epoch(net, epoch, dataLoader, optimizer, trainF, config):
    net.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask'] # mask: n, h, w 
        mask = mask.type(torch.LongTensor)
        # print("after type transfer mask.dtype:{}".format(mask.dtype)) # torch.int32
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        optimizer.zero_grad()
        # cbrp-cbrp-
        out = net(image) # n, c, h, w
        # print("before loss_func mask.dtype:{}".format(mask.dtype)) # torch.int32
        mask_loss = loss_func(out, mask, config.NUM_CLASSES, epoch)
        # after type transfer mask.dtype:torch.int64
        # before loss_func mask.dtype:torch.int64
        # inside loss_func:
        # dtype(predict):torch.float32
        # type(target):torch.int64  
        total_mask_loss += mask_loss.item()
        # D(loss)/D(w)
        mask_loss.backward()
        # w = w - lr * delta_w
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def test(net, epoch, dataLoader, testF, config, cur_max_iou, lr):
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i:0 for i in range(9)}, "TA":{i:0 for i in range(9)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        mask = mask.type(torch.LongTensor)
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        out = net(image)
        mask_loss = loss_func(out, mask, config.NUM_CLASSES, epoch)
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
    testF.write("Epoch:{} \n".format(epoch))
    testF.write("lr:{} \n".format(lr))
    miou = 0
    for i in range(9):
        iou_i = result["TP"][i]/result["TA"][i]
        result_string = "{}: {:.4f} \n".format(i, iou_i)
        print(result_string)
        testF.write(result_string)
        miou += iou_i
    miou /= 9
    miou_string = "{}: {:.4f} \n".format('miou', miou)
    print(miou_string)
    if miou > cur_max_iou:
        cur_max_iou = miou
        save_pth = os.path.join(os.getcwd(), config.SAVE_PATH, "global_max_miou.pth.tar".format(epoch))
        # torch.save(net.state_dict(), save_pth)
        testF.write('epoch ' + str(epoch) + ' cur_max_iou model: ' + str(cur_max_iou) + '\n')
        testF.write("mask loss is {:.4f} \n".format(total_mask_loss / len(dataLoader)))
        testF.flush()
    testF.write(miou_string)
    testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    testF.flush()
    return cur_max_iou

def main():
    lane_config = Config()
    if os.path.exists(lane_config.SAVE_PATH):
        shutil.rmtree(lane_config.SAVE_PATH)
    os.makedirs(lane_config.SAVE_PATH, exist_ok=True)
    trainF = open(os.path.join(lane_config.SAVE_PATH, "train_log.csv"), 'w')
    testF = open(os.path.join(lane_config.SAVE_PATH, "val_log.csv"), 'w')
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LaneDataset("disk3/lane_segment/lane_segmentation_step_lr/data_list/train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))

    train_data_batch = DataLoader(train_dataset, batch_size=4*len(device_list), shuffle=True, drop_last=True, **kwargs)
    val_dataset = LaneDataset("disk3/lane_segment/lane_segmentation_step_lr/data_list/val.csv", transform=transforms.Compose([ToTensor()]))

    val_data_batch = DataLoader(val_dataset, batch_size=4*len(device_list), shuffle=False, drop_last=False, **kwargs)
    trainF.write("batch_size:4")
    trainF.flush()
    net = nets[train_net](lane_config)
    if torch.cuda.is_available():
        net = net.cuda(device=device_list[0])
        net = torch.nn.DataParallel(net, device_ids=device_list)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                              momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lane_config.BASE_LR, betas=lane_config.BETA, eps = lane_config.EPS, weight_decay=lane_config.WEIGHT_DECAY, amsgrad=True)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lane_config.step_size, gamma=0.1, last_epoch=- 1, verbose=False)
    adamW_CosineAnneal = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=lane_config.CYCLE_INTER,
                                          T_mult=2,
                                          model=net,
                                          out_dir='cosine_annealing_snapshot',
                                          take_snapshot=False,
                                          eta_min=1e-5,
                                          eta_max=[1e-1])
    cur_max_iou = -1
    for epoch in range(lane_config.EPOCHS):
        adamW_CosineAnneal.step()
        lr = optimizer.param_groups[0]['lr']
        train_epoch(net, epoch, train_data_batch, optimizer, trainF, lane_config)
        cur_max_iou = test(net, epoch, val_data_batch, testF, lane_config, cur_max_iou, lr)
        # scheduler.step() 
    trainF.close()
    testF.close()
    
if __name__ == "__main__":
    main()


