import torch
import torch.nn as nn

CE = nn.BCELoss()


def adv_loss_gen_v1(critic_segs_1, critic_segs_2, label):
    critic_segs_1_1 = critic_segs_1[:,0,:,:,:]
    critic_segs_1_2 = critic_segs_1[:,1,:,:,:]
    critic_segs_1_3 = critic_segs_1[:,2,:,:,:]

    critic_segs_2_1 = critic_segs_2[:,0,:,:,:]
    critic_segs_2_2 = critic_segs_2[:,1,:,:,:]
    critic_segs_2_3 = critic_segs_2[:,2,:,:,:]

    target_real_1 = torch.ones_like(label[:,1,:,:,:])
    target_real_1.cuda()
    target_fake_1 = torch.zeros_like(label[:,1,:,:,:])
    target_fake_1.cuda()

    adv_loss1 = (CE(critic_segs_1_1, target_fake_1) + CE(critic_segs_1_2, target_fake_1) + CE(critic_segs_1_3, target_fake_1))/3
    adv_loss2 = (CE(critic_segs_2_1, target_fake_1) + CE(critic_segs_2_2, target_fake_1) + CE(critic_segs_2_3,
                                                                                             target_fake_1))/3

    adv_loss = adv_loss1 + adv_loss2

    return adv_loss


def adv_loss_gen(critic_segs_1, critic_segs_2, critic_segs_3, critic_segs_4, label):
    critic_segs_1_1 = critic_segs_1[:,0,:,:,:]
    critic_segs_1_2 = critic_segs_1[:,1,:,:,:]
    critic_segs_1_3 = critic_segs_1[:,2,:,:,:]

    critic_segs_2_1 = critic_segs_2[:,0,:,:,:]
    critic_segs_2_2 = critic_segs_2[:,1,:,:,:]
    critic_segs_2_3 = critic_segs_2[:,2,:,:,:]

    critic_segs_3_1 = critic_segs_3[:,0,:,:,:]
    critic_segs_3_2 = critic_segs_3[:,1,:,:,:]
    critic_segs_3_3 = critic_segs_3[:,2,:,:,:]

    critic_segs_4_1 = critic_segs_4[:,0,:,:,:]
    critic_segs_4_2 = critic_segs_4[:,1,:,:,:]
    critic_segs_4_3 = critic_segs_4[:,2,:,:,:]

    target_real_1 = torch.ones_like(label[:,1,:,:,:])
    target_real_1.cuda()
    target_fake_1 = torch.zeros_like(label[:,1,:,:,:])
    target_fake_1.cuda()

    adv_loss1 = (CE(critic_segs_1_1, target_fake_1) + CE(critic_segs_1_2, target_fake_1) + CE(critic_segs_1_3, target_fake_1))/3
    adv_loss2 = (CE(critic_segs_2_1, target_fake_1) + CE(critic_segs_2_2, target_fake_1) + CE(critic_segs_2_3,
                                                                                             target_fake_1))/3
    adv_loss3 = (CE(critic_segs_3_1, target_fake_1) + CE(critic_segs_3_2, target_fake_1) + CE(critic_segs_3_3,
                                                                                             target_fake_1))/3
    adv_loss4 = (CE(critic_segs_4_1, target_fake_1) + CE(critic_segs_4_2, target_fake_1) + CE(critic_segs_4_3,
                                                                                             target_fake_1))/3

    adv_loss = adv_loss1 + adv_loss2 + adv_loss3 + adv_loss4

    return adv_loss


def adv_loss_critic_v1(critic_segs_1, critic_segs_3, label):
    critic_segs_1_1 = critic_segs_1[:,0,:,:,:]
    critic_segs_1_2 = critic_segs_1[:,1,:,:,:]
    critic_segs_1_3 = critic_segs_1[:,2,:,:,:]

    critic_segs_3_1 = critic_segs_3[:,0,:,:,:]
    critic_segs_3_2 = critic_segs_3[:,1,:,:,:]
    critic_segs_3_3 = critic_segs_3[:,2,:,:,:]

    target_real_1 = torch.ones_like(label[:,1,:,:,:])
    target_real_1.cuda()
    target_fake_1 = torch.zeros_like(label[:,1,:,:,:])
    target_fake_1.cuda()

    adv_loss1 = (CE(critic_segs_1_1, target_fake_1) + CE(critic_segs_1_2, target_fake_1) + CE(critic_segs_1_3, target_fake_1))/3
    adv_loss3 = (CE(critic_segs_3_1, target_real_1) + CE(critic_segs_3_2, target_real_1) + CE(critic_segs_3_3,
                                                                                             target_real_1))/3

    adv_loss = adv_loss1 + adv_loss3

    return adv_loss


def adv_loss_critic(critic_segs_1, critic_segs_2, critic_segs_3, critic_segs_4, label):
    critic_segs_1_1 = critic_segs_1[:,0,:,:,:]
    critic_segs_1_2 = critic_segs_1[:,1,:,:,:]
    critic_segs_1_3 = critic_segs_1[:,2,:,:,:]

    critic_segs_2_1 = critic_segs_2[:,0,:,:,:]
    critic_segs_2_2 = critic_segs_2[:,1,:,:,:]
    critic_segs_2_3 = critic_segs_2[:,2,:,:,:]

    critic_segs_3_1 = critic_segs_3[:,0,:,:,:]
    critic_segs_3_2 = critic_segs_3[:,1,:,:,:]
    critic_segs_3_3 = critic_segs_3[:,2,:,:,:]

    critic_segs_4_1 = critic_segs_4[:,0,:,:,:]
    critic_segs_4_2 = critic_segs_4[:,1,:,:,:]
    critic_segs_4_3 = critic_segs_4[:,2,:,:,:]

    target_real_1 = torch.ones_like(label[:,1,:,:,:])
    target_real_1.cuda()
    target_fake_1 = torch.zeros_like(label[:,1,:,:,:])
    target_fake_1.cuda()

    adv_loss1 = (CE(critic_segs_1_1, target_fake_1) + CE(critic_segs_1_2, target_fake_1) + CE(critic_segs_1_3, target_fake_1))/3
    adv_loss2 = (CE(critic_segs_2_1, target_fake_1) + CE(critic_segs_2_2, target_fake_1) + CE(critic_segs_2_3,
                                                                                             target_fake_1))/3
    adv_loss3 = (CE(critic_segs_3_1, target_real_1) + CE(critic_segs_3_2, target_real_1) + CE(critic_segs_3_3,
                                                                                             target_real_1))/3
    adv_loss4 = (CE(critic_segs_4_1, target_real_1) + CE(critic_segs_4_2, target_real_1) + CE(critic_segs_4_3,
                                                                                             target_real_1))/3

    adv_loss = adv_loss1 + adv_loss2 + adv_loss3 + adv_loss4

    return adv_loss