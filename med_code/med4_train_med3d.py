'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''
import sys
from med3d.setting import parse_opts
from model_med3d import generate_model
# from med3d.dataset_oct_med3d_patch import AHNETdataset
from med3d.dataset_oct_med3d import AHNETdataset
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from med3d.utils.logger import log
import imageio
from scipy import ndimage
import os
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def my_collate(batch):
    nb_patch = 1

    for item in batch:
        if nb_patch < len(item[0]):
            [h, x, y, z] = item[0].shape
            patch_data = item[0].unfold(0, nb_patch, nb_patch)
            patch_label = item[1]*np.zeros((h//nb_patch,nb_patch))
            patch_label =torch.tensor(patch_label)
            # patch_label = patch_label.unfold(0, nb_patch, nb_patch)
            patch_data = patch_data.permute(0, 4, 1, 2, 3).contiguous().view(-1, nb_patch, x, y, z)
            patch_data = patch_data.unsqueeze(2)
            return [patch_data, patch_label]
        else:
            patch_data = item[0]
            patch_label = item[1]
            return [patch_data.unsqueeze(0), patch_label.unsqueeze(0)]
    return batch


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = img2d(input_tensor)
    imageio.imwrite(filename, input_tensor)


def img2d(input_tensor: torch.Tensor):
    if input_tensor.ndim == 3:
        input_tensor = input_tensor[0, :, :]
    elif input_tensor.ndim == 2:
        input_tensor = input_tensor
    elif input_tensor.ndim == 4:
        input_tensor = input_tensor[0, 0, :, :]
    elif input_tensor.ndim == 5:
        input_tensor = input_tensor[0, 0, 2, :, :]
    return input_tensor


def creat_txtfile(output_path, file_list):
    with open(output_path, 'w') as f:
        for list in file_list:
            print(list)
            f.write(str(list) + '\n')


def train(data_loader, data_loader_valid, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    test_contenu = []
    test_contenu.append('epoch' + ' ' + 'loss')
    batches_per_epoch = len(data_loader)
    batches_per_epoch_valid = len(data_loader_valid)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    log.info('size is {}*{}*{} , lr is {}'.format(sets.input_D, sets.input_H, sets.input_W ,0.0001))
    loss_class = nn.BCELoss()


    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        loss_class = loss_class.cuda()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        log.info('lr = {}'.format(scheduler.get_lr()))
        losses = 0
        flag_patch = 0

        model.train()
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_mask = batch_data
            np_patch = volumes.shape[0]
            if not sets.no_cuda:
                volumes = volumes.cuda()

            # n = 0
            # for item in volumes:
            #     n = n + 1
            #     save_image_tensor2cv2(item, "./img_ah/med_%s_%s_img.png" % (epoch, n))
            optimizer.zero_grad()
            out_mask = model(volumes)
            new_out_mask = out_mask.type(torch.DoubleTensor)
            label_mask = label_mask.unsqueeze(1)
            if not sets.no_cuda:
                label_mask = label_mask.cuda()
            output_p = torch.tensor(sum(new_out_mask) / len(new_out_mask), requires_grad=True)
            # output_p = torch.tensor(max(new_out_mask), requires_grad=True)
            label_mask = torch.tensor(sum(label_mask) / len(new_out_mask), dtype=torch.double)
            # label_mask = label_mask.view(1)
            # calculating loss
            loss_value_clas = loss_class(output_p, label_mask)
            loss = loss_value_clas
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses = losses + loss_value_clas
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, losses.item(), loss_value_clas.item(), avg_batch_time))
        model.eval()
        losses_valid = 0
        loss_value_clas_valid = []
        for batch_id_valid, batch_data_valid in enumerate(data_loader_valid):
            batch_id_sp_valid = epoch * batches_per_epoch_valid
            volumes_valid, label_mask_valid = batch_data_valid
            if not sets.no_cuda:
                volumes_valid = volumes_valid.cuda()
            optimizer.zero_grad()
            with torch.no_grad():
                out_mask_valid = model(volumes_valid)
                new_out_mask_valid = out_mask_valid.type(torch.DoubleTensor)
                label_mask_valid = label_mask_valid.unsqueeze(1)
                if not sets.no_cuda:
                    label_mask_valid = label_mask_valid.cuda()
                output_p_valid = torch.tensor(sum(new_out_mask_valid) / len(new_out_mask_valid), requires_grad=True)
                # output_p = torch.tensor(max(new_out_mask), requires_grad=True)
                label_mask_valid = torch.tensor(sum(label_mask_valid) / len(new_out_mask_valid), dtype=torch.double)
                # label_mask = label_mask.view(1)
                # calculating loss
                loss_value_clas_valid = loss_class(output_p_valid, label_mask_valid)
                losses_valid = losses_valid + loss_value_clas_valid
                writer.add_graph(model=model, input_to_model=volumes_valid)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_clas.item(), ))

        writer.add_scalar("train loss /Epoch", losses / batches_per_epoch, epoch)
        writer.add_scalar("valid loss /Epoch", losses_valid / batches_per_epoch_valid, epoch)
        test_contenu.append('%f  %f  %f' % (epoch, losses / batches_per_epoch, losses_valid / batches_per_epoch_valid))
        writer.flush()
        label = 'no_0_e4_32_224_448_sans_patch'
        torch.save(model.state_dict(), "./pth/med-pre-%s-%d.pth" % (label, epoch + 1))
    output_path = '/data_GPU/wenpei/script/octip/result_med3d.txt'
    creat_txtfile(test_contenu, output_path)

    writer.close()

    print('Finished training')
    if sets.ci_test:
        exit()

if __name__ == '__main__':

    print(torch.version.cuda)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # settting
    sets = parse_opts()

    sets.n_epochs = 50
    sets.no_cuda = False
    sets.pretrain_path = ''
    sets.num_workers = 0
    sets.model_depth = 50
    sets.resnet_shortcut = 'A'
    sets.input_D = 32
    sets.input_H = 224
    sets.input_W = 448
    sets.learning_rate = 0.0005
    sets.batch_size = 1
    sets.n_seg_classes = 2
    sets.save_intervals = 1

    img_train = './train_label.txt'
    img_valid = './valid_label.txt'
    # img_valid = './test_label1.txt'
    # img_train = './test_label1.txt'
    file_train = './train'
    file_valid = './valid'
    phase = "train"
 # learning rate (constant)
    weight_decay = 2e-5
    pin_memory = True
    num_workers = 0

    # getting model
    sets.no_cuda = torch.cuda.is_available()
    torch.manual_seed(sets.manual_seed)
    mymodel, parameters = generate_model(sets)
    print(mymodel)
    # optimizer

    params = [{'params': parameters, 'lr': sets.learning_rate}]

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=sets.learning_rate, weight_decay=2e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # train from resume
    # sets.resume_path = "./pth/med-deconv-1.pth"
    # sets.resume_path = "./trails/resnet_50_epoch_199_batch_0.pth.tar"
    # #
    # if os.path.isfile(sets.resume_path):
    #     print("=> loading checkpoint '{}'".format(sets.resume_path))
    #     checkpoint = torch.load(sets.resume_path)
    #     model_dict = mymodel.state_dict()
    #     a = {k[7:]: v for k, v in checkpoint['state_dict'].items() }
    #     state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if k[7:] in model_dict and model_dict[k[7:] ].shape == v.shape}
    #     model_dict.update(state_dict)
    #     mymodel.load_state_dict(model_dict)

    # # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True
    training_dataset = AHNETdataset(file_train, img_train, sets.input_W, sets.input_H, sets.input_D, phase)
    valid_dataset = AHNETdataset(file_valid, img_valid, sets.input_W, sets.input_H, sets.input_D, phase)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers,pin_memory=sets.pin_memory)
    data_loader_valid = DataLoader(valid_dataset, batch_size=sets.batch_size, shuffle=True,num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    train(data_loader, data_loader_valid, mymodel, optimizer, scheduler, total_epochs=sets.n_epochs,
          save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets)
