from cProfile import label
from glob import escape
from attr import asdict
import torch
import os
import numpy as np
import pickle
from tqdm import tqdm
from utils.Observables import *
from utils.resnet_vis import *
from scipy.linalg import logm
import scipy
import sys
import os
import random
import pdb
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
from PIL import Image 
from data_loader import RGBDDataset, demo_playback


LENGTH = 30


class DynamicsPredictionDataset(Dataset):
   def __init__(self, demo, img_path):
       self.demo = demo
       self.img_path = img_path

   def __len__(self):
       return len(self.demo)

   def __getitem__(self, index):
       path = self.demo[index]
       current_idx = random.randint(0, len(path)-(LENGTH+2))
       robot_current = np.array(path[current_idx]['handpos'])
       count = path[current_idx]['count']
       img_path = os.path.join(self.img_path, str(count)+".npy")
       rgbd = np.load(img_path)
       rgbd = np.transpose(rgbd, (2, 0, 1))
       robot_next = [np.array(path[current_idx+i]['handpos']) for i in range(1,LENGTH+1)]
       return rgbd, robot_current, robot_next


def generate_feature(model, generate_feature_data_path, img_path, device="cuda"):
    batch_size = 32
    generate_feature_data = []
    model.eval()
    train_dataset = RGBDDataset(img_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    error = 0
    count = 0
    vis_imgs = []
    with torch.no_grad():
        for batch_num, (rgbds, indexs) in enumerate(train_dataloader):
            rgbds = rgbds.float().to(device)
            outputs_feat = model(rgbds)
            outputs = outputs_feat.detach().cpu().numpy() #outputs_pos
            for index, output, rgbd in zip(indexs, outputs, rgbds):
                generate_feature_data.append({index:output})
            del rgbds
            del indexs   
    with open(generate_feature_data_path, 'wb') as handle:
        pickle.dump(generate_feature_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_koopman(Training_data, num_hand, num_obj, koopman_save_path):
   Koopman = DraftedObservable(num_hand, num_obj)
   num_obs = Koopman.compute_observable(num_hand, num_obj)
   A = np.zeros((num_obs, num_obs)) 
   G = np.zeros((num_obs, num_obs))
   ## loop to collect data
   print("Drafted koopman training starts!\n")
   for k in tqdm(range(len(Training_data))):
       hand_OriState = Training_data[k][0]['handpos']
       obj_OriState = Training_data[k][0]['rgbd_feature']
       assert len(obj_OriState) == num_obj
       assert len(hand_OriState) == num_hand
       z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
       for t in range(len(Training_data[k]) - 1):
           hand_OriState = Training_data[k][t + 1]['handpos']
           obj_OriState = Training_data[k][t+1]['rgbd_feature']
           z_t_1 = Koopman.z(hand_OriState, obj_OriState)
           A += np.outer(z_t_1, z_t)
           G += np.outer(z_t, z_t)
           z_t = z_t_1
   M = len(Training_data) * (len(Training_data[0]) - 1)
   A /= M
   G /= M
   koopman_operator = np.dot(A, scipy.linalg.pinv(G))
   cont_koopman_operator = koopman_operator
   np.save(koopman_save_path, cont_koopman_operator)
   print(f"Koopman matrix is saved!\n")

def main():
    num_hand = 6#7 #8 # 3 x,y,z + 3 theta_x, theta_y, theta_z + 1 gripper
    num_obj = 8 #8
    dynamics_batch_num = 4#8
    learningRate = 1e-4 #1e-5  #1e-4 for SL 1e-5 for K based error finetune?
    device = "cuda" if torch.cuda.is_available() else "cpu"

    demo_data_path = f"./data/joint_states.pickle"
    generate_feature_data_path = f'./data/generate_feature.pickle'
    img_path = f"./data/image_data"
    koopman_save_path = f"./result/koopmanMatrix.npy"

    # training parameters
    resnet_model = ResNet18_freq(feat_dim = num_obj)
    resnet_model = resnet_model.float()
    resnet_model.eval()
    resnet_model = resnet_model.to(device)

    min_loss = 100
    loss = torch.nn.L1Loss() 
    generate_feature(resnet_model, generate_feature_data_path, img_path, device=device)
    Training_data = demo_playback(demo_data_path, generate_feature_data_path)

    # Resnet Dynamics Training dataset
    dynamics_train_dataset = DynamicsPredictionDataset(Training_data, img_path)
    dynamics_train_dataloader = DataLoader(dynamics_train_dataset, batch_size=dynamics_batch_num, shuffle=True, num_workers=0)

    Koopman = DraftedObservable(num_hand, num_obj)
    train_koopman(Training_data, num_hand, num_obj, koopman_save_path)
    cont_koopman_operator = np.load(koopman_save_path) # matrix_file
    cont_koopman_operator = torch.from_numpy(cont_koopman_operator).to(device)

    # Train
    for epoch in range(3001):
        resnet_model.train() 
        if (epoch > 0 and epoch % 10 == 0): #50
            learningRate *= 0.9
        optimizer_feature = torch.optim.Adam(resnet_model.parameters(), lr=learningRate)
        epoch_loss = resnet_train(resnet_model, Koopman, cont_koopman_operator, dynamics_train_dataloader, loss, optimizer_feature, device, num_hand, num_obj)
        print(f"epoch {epoch} loss_avg {epoch_loss}, learningRate {learningRate}")

        if (epoch >= 50 and epoch % 50 == 0):
            # Update K
            generate_feature(resnet_model, generate_feature_data_path, img_path, device=device)
            Training_data = demo_playback(demo_data_path, generate_feature_data_path)
            train_koopman(Training_data, num_hand, num_obj, koopman_save_path)
            cont_koopman_operator = np.load(koopman_save_path) # matrix_file
            # Eval
        #     if (epoch >= 50):
        #         input_val = input("Press continue...")
        #         if input_val == "c":
        #             with torch.no_grad():
        #                 #pass
        #                 Pick_eval(resnet_model, Koopman, cont_koopman_operator, num_hand, num_obj, device)
        #                 #torch.save({'model_state_dict': resnet_model.state_dict()}, "./model/door_full_resnet")
            cont_koopman_operator = torch.from_numpy(cont_koopman_operator).to(device)
        #     torch.save({'model_state_dict': resnet_model.state_dict()}, "./model/door_full_resnet")


def resnet_train(resnet_model, Koopman, cont_koopman_operator, data_loader, loss, optimizer_feature, device, num_hand, num_obj, require_grad = True, dynamics_error = False):
    ErrorInOriginalRobot = 0
    epoch_loss = 0
    pos_loss = 0
    ori_loss = 0
    gripper_loss = 0

    for batch_num, (rgbds, robot_currents, robot_nexts) in enumerate(data_loader):
        optimizer_feature.zero_grad()
        robot_currents = robot_currents.float().to(device)
        rgbds = rgbds.float().to(device)
        pred_feats = resnet_model(rgbds)
        batch_size = len(robot_currents)
        for i in range(batch_size):
            hand_OriState = robot_currents[i]
            obj_OriState = pred_feats[i]
            z_t_computed = Koopman.z_torch(hand_OriState, obj_OriState).to(device)
            for t in range(len(robot_nexts)):
                robot_next = robot_nexts[t][i].float().to(device)
                z_t_1_computed = torch.matmul(cont_koopman_operator.float(), z_t_computed.float())
                loss_value = loss(z_t_1_computed[:num_hand], robot_next)
                ErrorInOriginalRobot += loss_value
                z_t_computed = z_t_1_computed
                epoch_loss += loss_value.item()
        ErrorInOriginalRobot *= 0.05 # weights
        ErrorInOriginalRobot.backward()
        optimizer_feature.step()
        ErrorInOriginalRobot = 0
        optimizer_feature.zero_grad()
    epoch_loss = epoch_loss/len(data_loader)
    return epoch_loss

if __name__ == '__main__':
    main()