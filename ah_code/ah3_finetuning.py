#!/usr/bin/python

from __future__ import division

from net3d_class import AHNet
from net2d import MCFCN
import tqdm
from torch.utils.data import DataLoader
from time import time
from ah3_dataset_oct import AHNETdataset,save_image_tensor2cv2,img2d,my_collate
# from ah3_dataset_oct_patch import AHNETdataset
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.autograd import Variable



writer = SummaryWriter()
# img_train = './test_label.txt'
# img_valid = './test_label.txt'
img_train = './train_label.txt'
img_valid = './valid_label.txt'
file_train = './train'
file_valid = './train'
input_D = 32
input_H = 192
input_W = 384
phase = "train"
epoches = 100

lr = 0.0002  # learning rate (constant)
batch_size = 1
weight_decay = 2e-5
momentum = 0.9
max_iters = 92 * epoches
pin_memory = True
num_workers = 0
pretrain_net = MCFCN()
pth = './pth/ah2d2.pth' # pretrained model name
checkpoint = torch.load(pth, map_location=torch.device('cpu'))
pretrain_net.load_state_dict(checkpoint)
flag_patch = 0
model = AHNet()
model.copy_from(pretrain_net)
# 3D AHNET model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print('ft with size %f * %f *%f , batch size : %f with flag_patch %f '%(input_W,input_H,input_D,batch_size,flag_patch))

# loading train / valid datasets
training_dataset = AHNETdataset(file_train, img_train, input_W, input_H, input_D, phase)
valid_dataset = AHNETdataset(file_valid, img_valid, input_W, input_H, input_D, phase)

if flag_patch:
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                     pin_memory=pin_memory,collate_fn=my_collate)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory,collate_fn=my_collate)
else:
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                     pin_memory=pin_memory)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory)
# BCEloss
criterion = torch.nn.BCELoss()

# stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
#                             weight_decay=weight_decay)
#
start = time()
for epoch in range(epoches):
    running_loss = 0.0
    print('tqdm.tqdm(enumerate(data_loader))')
    # use model to train
    model.train()
    running_loss = 0
    n = 0
    losses = []
    for i, (images, labels_group) in tqdm.tqdm(enumerate(training_dataloader)):
        # verify utilization of CUDA
        if torch.cuda.is_available():
            images = [Variable(image.cuda()) for image in images]
            labels_group = [ labels.cuda() for labels in labels_group]
        else:
            images = [Variable(image) for image in images]
            labels_group = [torch.tensor(labels_group,dtype=torch.float)]

        for img, labels in zip(images, labels_group):
            optimizer.zero_grad()
            outputs = model(img)
            label = labels.view(1).float()
            if torch.cuda.is_available():
                loss = criterion(outputs.cuda(), label.cuda())
            else:
                loss = criterion(outputs, label)
            loss.backward() # backward
            optimizer.step()
            losses.append(loss)
            running_loss += loss
            n += 1
    running_loss = running_loss / n
    optimizer.zero_grad()
    writer.add_graph(model=model, input_to_model=img)
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
        for img, labels in zip(images, labels_group):
            with torch.no_grad():
                outputs = model(img)
                label = labels.view((1)).float()
                if torch.cuda.is_available():
                    Test_loss = criterion(outputs.cuda(), label.cuda())
                else:
                    Test_loss = criterion(outputs, label)
            valid_loss += Test_loss
            y = y + 1
    valid_loss /= y
    optimizer.zero_grad()
    writer.add_scalar("ft valid Loss/epoch", valid_loss.item(), epoch + 1)
    writer.add_scalar("ft train Loss/epoch", running_loss.item(), epoch + 1)
    writer.flush()
    print("Epoch [%d] Valid Loss: %.4f" % (epoch + 1, valid_loss))
    print("Epoch [%d] Loss: %.4f" % (epoch + 1, running_loss))
    if (epoch + 1) % 1 == 0:
        # save the parameters per epoch
        torch.save(model.state_dict(), "./pth/ft-deconv-%d.pth" % (epoch + 1))
    print( "Epoch finished")
writer.close()
torch.save(model.state_dict(), "./pth/ft-deconv.pth")

