#!/usr/bin/python

from __future__ import division
import sys
import scipy
from net3d_class import AHNet
import imageio
import tqdm
from torch.utils.data import DataLoader
from time import time
# from ah3_dataset_oct_patch import AHNETdataset
from ah3_dataset_oct import AHNETdataset,save_image_tensor2cv2,img2d,my_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision
from visulize import get_logger
import torch
from torch.autograd import Variable
writer = SummaryWriter()
# img_train = './test_label1.txt'
# img_valid = './test_label1.txt'
img_train = './train_label.txt'
img_valid = './valid_label.txt'
file_train = './train'
file_valid = './valid'
input_D = 32
input_H = 192
input_W = 384
phase = "train"
epoches = 150
lr = 0.0003  # learning rate (constant)
batch_size = 1
weight_decay = 2e-5
momentum = 0.9
max_iters = 92 * epoches
pin_memory = True
num_workers = 0


# 3D AHNET model
model = AHNet()
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
flag_patch = 0
# loading train / valid datasets
training_dataset = AHNETdataset(file_train, img_train, input_W, input_H, input_D, phase)
valid_dataset = AHNETdataset(file_valid, img_valid, input_W, input_H, input_D, phase)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=pin_memory)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory)
# MSEloss
criterion = torch.nn.BCELoss(reduce = None)

# stochastic gradient descent
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
#                             weight_decay=weight_decay)

optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
#

start = time()
for epoch in range(epoches):
    running_loss = 0.0
    print('tqdm.tqdm(enumerate(data_loader))')

    # use model to train
    model.train()
    n = 0
    for i, (images, labels_group) in tqdm.tqdm(enumerate(training_dataloader)):
        # verify utilization of CUDA
        if torch.cuda.is_available():
            images = [Variable(image.cuda()) for image in images]
            labels_group = [ labels.cuda() for labels in labels_group]
        else:
            images = [Variable(image) for image in images]
            labels_group = [torch.tensor(labels_group,dtype=torch.float)]
        losses = []
        optimizer.zero_grad()
        for img, labels in zip(images, labels_group):
            outputs = model(img)
            label = labels.view(1).float()
            loss = criterion(outputs.cuda(), label.cuda())
            writer.add_graph(model=model, input_to_model=img)
            loss.backward()
            optimizer.step()
            n += 1
            running_loss += loss.item()
            img1 = img.clone().detach()
            writer.add_image("Training img0 ", (img2d(img1)), dataformats="HW")
    running_loss = running_loss / n


    model.eval()
    # cross validation
    valid_loss = 0
    y = 0
    for i, (images, labels_group) in tqdm.tqdm(enumerate(valid_dataloader)):

        if torch.cuda.is_available():
            images = [Variable(image.cuda()) for image in images]
            labels_group = [ labels.cuda() for labels in labels_group]
        else:
            images = [Variable(image) for image in images]
            labels_group = [labels for labels in labels_group]
        losses = []
        for img, labels in zip(images, labels_group):
            with torch.no_grad() :
                outputs = model(img)
                label = labels.view((1)).float()
                Test_loss = criterion(outputs.cuda(), label.cuda())
            valid_loss += Test_loss
            y = y + 1
    valid_loss /= y
    optimizer.zero_grad()
    writer.add_scalar("valid Loss/epoch", valid_loss.item(), epoch + 1)
    writer.add_scalar("train Loss/epoch", running_loss, epoch + 1)
    writer.flush()
    # save information in log file
    # logger.info('Epoch:[{}/{}]\t loss={:.5f}\t Test loss={:.5f}'.format(epoch, loss, est_loss ))
    print("Epoch [%d] Loss: %.4f" % (epoch + 1, valid_loss))
    print("Epoch [%d] Loss: %.4f" % (epoch + 1, running_loss))
    if (epoch + 1) % 1 == 0:
    # save the parameters per 20 epoches
        torch.save(model.state_dict(), "./pth/fcn-%d.pth" % (epoch + 1))
writer.close()
torch.save(model.state_dict(), "./pth/fcn-deconv.pth")
