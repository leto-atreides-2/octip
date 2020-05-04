from __future__ import division
import torch
from torch.autograd import Variable
from net3d import AHNet
import tqdm
from torch.utils.data import DataLoader
from time import time
from dataset import AHNETdataset
from visulize import save_image_tensor
from torch.utils.tensorboard import SummaryWriter
import torchvision
from visulize import get_logger

logger = get_logger('./train.log')
writer = SummaryWriter('./TBwriter')

img_train = './train//txt_label.txt'
img_valid = './valid//txt_label.txt'
file_train = './train'
file_valid = './valid'
input_D = 32
input_H = 192
input_W = 192
phase = "train"
epoches = 100
lr = 0.0001  # learning rate (constant)
batch_size = 1
weight_decay = 2e-5
momentum = 0.9
weight = torch.ones(22)
weight[21] = 0
max_iters = 92 * epoches
pin_memory = True
num_workers = 0

# 3D AHNET model
model = AHNet()

# loading train / valid datasets
training_dataset = AHNETdataset(file_train, img_train, input_W, input_H, input_D, phase)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=pin_memory)
valid_dataset = AHNETdataset(file_valid, img_valid, input_W, input_H, input_D, phase)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)

# MSEloss
criterion = torch.nn.MSELoss(reduction='mean')

# stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)

start = time()
for epoch in range(epoches):
    running_loss = 0.0
    print('tqdm.tqdm(enumerate(data_loader))')

    # use model to train
    model.train()
    for i, (images, labels_group) in tqdm.tqdm(enumerate(training_dataloader)):
        # verify utilization of CUDA
        if torch.cuda.is_available():
            images = [Variable(image.cuda()) for image in images]
            labels_group = [labels for labels in labels_group]
        else:
            images = [Variable(image) for image in images]
            labels_group = [labels for labels in labels_group]

        optimizer.zero_grad()
        losses = []

        for img, labels in zip(images, labels_group):
            # calculate losses
            outputs = model(img)
            n = 0
            for output, label in zip(outputs, labels):
                output = torch.sigmoid(output.squeeze())
                losses.append(criterion(output, label.squeeze()))
                # losses = losses + criterion(output, label.squeeze()).tolist()
                n += 1

        loss = sum(losses) / n

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    # cross validation
    Test_loss = 0
    for i, (images, labels_group) in tqdm.tqdm(enumerate(valid_dataloader)):
        if torch.cuda.is_available():
            images = [Variable(image.cuda()) for image in images]
            labels_group = [labels for labels in labels_group]
        else:
            images = [Variable(image) for image in images]
            labels_group = [labels for labels in labels_group]
        optimizer.zero_grad()
        losses = []
        for img, labels in zip(images, labels_group):
            outputs = model(img)
            for output, label in zip(outputs, labels):
                output = torch.sigmoid(output.squeeze())
                Test_loss = criterion(output, label.squeeze())
                losses.append(Test_loss)
    # using tensorboard  to record Loss
    writer.add_scalar('_Train/Loss_',loss.item() ,epoch)

    writer.add_scalar('_Train/Loss_', Test_loss.item(), epoch)
    writer.flush()
    # grid = torchvision.utils.make_grid(images)
    # writer.add_image('Dataset input grid ' ,grid ,global_step= 0)


    # save information in log file
    # logger.info('Epoch:[{}/{}]\t loss={:.5f}\t Test loss={:.5f}'.format(epoch, loss, Test_loss ))

    print("Epoch [%d] Loss: %.4f" % (epoch + 1, running_loss / i))
    if (epoch + 1) % 5 == 0:
        #save the parameters per 20 epoches

        torch.save(model.state_dict(), "/home/stage/PycharmProjects/test_3d_ah/pth/fcn-deconv-%d.pth" % (epoch + 1))
    writer.close()
torch.save(model.state_dict(), "./pth/fcn-deconv.pth")
