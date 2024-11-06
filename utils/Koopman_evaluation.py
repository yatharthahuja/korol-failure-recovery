"""
Define the functions that are used to test the learned koopman matrix
"""
from glob import escape
from attr import asdict
import numpy as np
import time
from tqdm import tqdm
from utils.gym_env import GymEnv
from utils.quatmath import euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.resnet import *
import torch
import pdb
import torch.nn as nn  
import os
from torch.utils.data import Dataset, DataLoader
import pickle 
from PIL import Image
class RGBDDataset(Dataset):
    def __init__(self, demo_path, img_path):
        # self.data is useless
        with open(demo_path, 'rb') as f:
            self.data = pickle.load(f)
        self.img_path = img_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = os.path.join(self.img_path, "rgbd_"+str(index)+".npy") 
        #"/home/hongyic/3D_Learning/Kodex/Door/Data/rgbd_"+str(index)+".npy"
        rgbd = np.load(path)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        return rgbd, self.data[index]['objpos'], self.data[index]['gt_feat'], index
        # gt_feature = self.data[index]['gt_feat']
        # gt_pos = self.data[index]['objpos']
        # return rgbd, gt_pos, gt_feature, index

# demo_path = "/home/hongyic/3D_Learning/Kodex/Door/Data/door_full_objpos_feature.pickle"
# img_path = "/home/hongyic/3D_Learning/Kodex/Door/Data"
def generate_feature(model, feature_path, demo_path, img_path, device="cuda"):
    batch_num = 32
    Training_data = []
    model.eval()
    train_dataset = RGBDDataset(demo_path, img_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_num, shuffle=False, num_workers=4)
    error = 0
    with torch.no_grad():
        for batch_num, (rgbds, gt_pos, _, indexs) in enumerate(train_dataloader):
            rgbds = rgbds.float().to(device)
            gt_pos = gt_pos.float().to(device)

            outputs_pos, outputs_feat = model(rgbds)
            error += torch.sum(torch.abs(outputs_pos - gt_pos))
            outputs = outputs_feat.detach().cpu().numpy() #outputs_pos
            for index, output, rgbd in zip(indexs, outputs, rgbds):
                Training_data.append({index:output})
                # if (index == 0):
                #     return rgbd.cpu().numpy()
            #torch.cuda.empty_cache()
            del rgbds
            del indexs
            # if (batch_num%300==0 and batch_num > 0):
            #     print(batch_num, outputs[:16], error)
    # Door/feature_door_demo_full.pickle
    print(f"obj pos prediction error {error/len(train_dataloader)}")
    with open(feature_path, 'wb') as handle:
        pickle.dump(Training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_env_rgbd(e):
    img_obs, depth_obs = e.env.mj_render()
    depth_obs = depth_obs[...,np.newaxis]
    img_obs = (img_obs.astype(np.uint8) - 128.0) / 128
    rgbd = np.concatenate((img_obs, depth_obs),axis=2)
    rgbd = np.transpose(rgbd, (2, 0, 1))
    rgbd = rgbd[np.newaxis, ...]
    return rgbd

def koopman_evaluation(koopman_object, koopman_matrix, Test_data, num_hand, num_obj):
    '''
    Input: Koopman object (Drafted, MLP, GNN) for observable lifting
           Learned koopman matrix
           Testing data
           Velocity flag
    '''
    e = GymEnv("door-v0")
    e.reset()
    ErrorInLifted = np.zeros(koopman_object.compute_observable(num_hand, num_obj))
    ErrorInOriginal = np.zeros(num_hand + num_obj) 
    ErrorInOriginalObj = np.zeros(num_obj)
    ErrorInOriginalRobot = np.zeros(num_hand)
    #print(f"koopman_matrix {koopman_matrix[2 * num_hand: 2 * num_hand + num_obj,2 * num_hand: 2 * num_hand + num_obj]}")
    for k in tqdm(range(len(Test_data))):
        # e.set_env_state(Test_data[k][0]['init'])
        # get_env_rgbd(e)
        hand_OriState = Test_data[k][0]['handpos']

        implict_objpos = Test_data[k][0]['rgbd_feature']
        #np.append(np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel'])), Test_data[k][0]['nail_goal'])
        #Test_data[k][0]['rgbd_feature']
        #Test_data[k][0]['rgbd_feature']
        #obj_embedder(torch.from_numpy(Test_data[k][0]['objpos'])).detach().numpy()
        #implict_objpos_init = obj_embedder(torch.from_numpy(Test_data[k][0]['handle_init'])).detach().numpy()
        #Test_data[k][0]['handle_init_feature']
        #obj_embedder(torch.from_numpy(Test_data[k][0]['handle_init'])).detach().numpy()
        obj_OriState = implict_objpos #np.append(implict_objpos, np.append(Test_data[k][0]['objvel'], implict_objpos_init))
        #pdb.set_trace()
        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        z_t_computed = z_t
        for t in range(len(Test_data[k]) - 1):
            hand_OriState = Test_data[k][t + 1]['handpos']
            implict_objpos_t = Test_data[k][t+1]['rgbd_feature']
            #np.append(np.append(Test_data[k][t + 1]['objpos'], np.append(Test_data[k][t + 1]['objorient'], Test_data[k][t + 1]['objvel'])), Test_data[k][t + 1]['nail_goal'])
            #Test_data[k][t+1]['rgbd_feature']
            #Test_data[k][t+1]['rgbd_feature']
            #obj_embedder(torch.from_numpy(Test_data[k][t + 1]['objpos'])).detach().numpy()
            #implict_objpos_init_t = obj_embedder(torch.from_numpy(Test_data[k][t + 1]['handle_init'])).detach().numpy()
            #Test_data[k][t+1]['handle_init_feature']
            #obj_embedder(torch.from_numpy(Test_data[k][t + 1]['handle_init'])).detach().numpy()
            obj_OriState = implict_objpos_t #np.append(implict_objpos_t, implict_objpos_init_t)
            #np.append(implict_objpos_t, np.append(Test_data[k][t + 1]['objvel'], implict_objpos_init_t))
            z_t_1 = koopman_object.z(hand_OriState, obj_OriState) # states in lifted space at next time step (extracted from data)
            z_t_1_computed = np.dot(koopman_matrix, z_t_computed)
            x_t_1 = np.append(z_t_1[:num_hand], z_t_1[2 * num_hand: 2 * num_hand + num_obj])  # observation functions: hand_state, hand_state^2, object_state, object_state^2
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])  
            ErrorInLifted += np.abs(z_t_1 - z_t_1_computed)  # if using np.square, we will get weird results.
            ErrorInOriginal += np.abs(x_t_1 - x_t_1_computed)
            ErrorInOriginalObj += np.abs(x_t_1[num_hand:] - x_t_1_computed[num_hand:])
            ErrorInOriginalRobot += np.abs(x_t_1[:num_hand] - x_t_1_computed[:num_hand])
            #print(f"x_t_1[num_hand:] {x_t_1[num_hand:]}, x_t_1_computed[num_hand:] {x_t_1_computed[num_hand:]}")
            z_t = z_t_1
            z_t_computed = z_t_1_computed
    M = len(Test_data) * (len(Test_data[0]) - 1)
    ErrorInLifted /= M
    ErrorInOriginal /= M
    ErrorInOriginalObj /= M
    ErrorInOriginalRobot /= M
    return ErrorInLifted, ErrorInOriginal, ErrorInOriginalObj, ErrorInOriginalRobot

def koopman_policy_control(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize, obj_embedder, folder_name, use_obj_embedder=False, use_resnet=False, resnet_model=None, device='cuda:1'):

    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])  # palm_pos - handle_pos
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_list_sim = []
    success_rate = str()
    print(f"len(Test_data) {len(Test_data)}")
    if use_obj_embedder:
        obj_embedder = obj_embedder.to('cpu')
    for k in range(len(Test_data)):#tqdm(range(len(Test_data))): #len(Test_data)
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']

        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)

        rgb, depth = e.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        if use_obj_embedder:
            #pdb.set_trace()
            implict_objpos = obj_embedder(torch.from_numpy(Test_data[k][0]['objpos']).float()).numpy()
        elif use_resnet:
            #pdb.set_trace()
            _, implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device)) #Test_data[k][0]['rgbd_feature']
            implict_objpos = implict_objpos[0].cpu().numpy()
        else:
            implict_objpos = Test_data[k][0]['rgbd_feature']
        obj_OriState = implict_objpos
        
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            #pdb.set_trace()
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                obj_pos_world = x_t_1_computed[56:59]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  
            hand_pos_desired = hand_pos  # control frequency
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            # if Visualize:
            #     #if (t%5 == 0):
            #     img_obs, depth_obs = e.env.mj_render()
            #     from PIL import Image 
            #     img_obs = Image.fromarray(img_obs)
            #     img_obs.save(f"door_opening_{k}_{t}.png")
                #plt.show(block = False)
                #e.render()
                #print(depth_obs)
                #
                #e.env.sim.render(width = 64, height = 64, depth=True)
                #img_obs = e.env.viewer.read_pixels(1376, 752, depth=False, segmentation=True)#[1]
                
                # try:
                #     img_obs = e.env.viewer.read_pixels(1376, 752, depth=False)
                # except:
                #     e.env.mj_viewer_setup()
                # img_obs = e.env.viewer.read_pixels(1376, 752, depth=False)
                # # # # save image
                
                #img_obs.show()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.env.get_full_obs_visualization()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            #hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[4:] - hand_pos[4:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:4] - hand_pos[:4]))  
            # instead of defining the position error of the nail, we define it as the difference between the hammer tool and the nail
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t + 1]['observation'])) # palm-handle
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            #print(f"success in {k}", end="")
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    # success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    # print("Success rate (RL) = %f" % (1))

    if use_obj_embedder:
        obj_embedder = obj_embedder.to(device)
    if error_type == 'demo':
        return len(success_list_sim) / len(Test_data) #[hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return len(success_list_sim) / len(Test_data) #[obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     


def koopman_policy_control_reorientation_single_task(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type):
    print("Begin to compute the simulation errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Joint_NN_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_RL = []
    fall_list_sim = []
    fall_list_koopman = []
    fall_list_RL = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
    for k in range(len(Test_data)): #tqdm()
        gif = []
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_koopman = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = Test_data[k][0]['rgbd_feature']
        #np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        # init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'] + Test_data[k][0]['final_handpos'], np.zeros(6))
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        #pdb.set_trace()
        e.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict
        #Joint_NN_values[:, 0, k] = e.get_obs()[:24]
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39])
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            Computed_joint_values[:, t + 1, k] = hand_pos
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            #hand_pos = hand_pos[6:]
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            #Computed_torque_values[:, t, k] = NN_output
            #Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            # if Visualize:
            #     e.env.mj_render()
            rgb, depth = e.env.mj_render()
            img_obs = Image.fromarray(rgb)
            gif.append(img_obs)

            
            #img_obs = Image.fromarray(rgb)
            #img_obs.save(f"reorientation_{i}_{t}.png")
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']  # in demo data, it is defined in a new coordinate 
            gt_obj_orien_world_frame = ori_transform_inverse(gt_obj_orient, e.get_obs()[36:39])
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:24]
            #Joint_NN_values[:, t + 1, k] = hand_pos
            obj_pos = e.get_obs()[24:27]
            # Test_data[k][*]['desired_ori'] is the same value
            obj_orient = ori_transform(e.get_obs()[33:36], Test_data[k][t]['desired_ori'])  # change the coordinate
            obj_vel = e.get_obs()[27:33]          
            obj_obs = e.get_obs()
            # compute the errors
            #hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            hand_pos_PID_error[t, k] = np.mean(np.abs(hand_pos_desired - hand_pos))  
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            demo_ori_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            orien_similarity_RL = np.dot(gt_obj_orien_world_frame, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            dist_RL = np.linalg.norm(gt_obj_pos - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
            success_count_RL[t] = 1 if (orien_similarity_RL > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
            gif[0].save(f'/home/hongyic/3D_Learning/Kodex/Reorientation/fail/{k}.gif', save_all=True,optimize=False, append_images=gif[1:], loop=0)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
                success_list_sim.append(1)
                gif[0].save(f'/home/hongyic/3D_Learning/Kodex/Reorientation/success/{k}.gif', save_all=True,optimize=False, append_images=gif[1:], loop=0)
            else:
                gif[0].save(f'/home/hongyic/3D_Learning/Kodex/Reorientation/fail/{k}.gif', save_all=True,optimize=False, append_images=gif[1:], loop=0)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
        if np.abs(gt_obj_pos - desired_pos)[2] > 0.15:
            fall_list_RL.append(1)
        else:
            if sum(success_count_RL) > success_threshold:
                success_list_RL.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    # print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    # success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    # print("Success rate (RL) = %f" % (len(success_list_RL) / len(Test_data)))
    # print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    # success_rate += "Throw out rate (sim) = %f\n" % (len(fall_list_sim) / len(Test_data))
    # print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    # success_rate += "Throw out rate (koopman) = %f\n" % (len(fall_list_koopman) / len(Test_data))
    # print("Throw out rate (RL) = %f" % (len(fall_list_RL) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error, hand_pos_PID_error, Computed_torque_values, Demo_torque_values, Computed_joint_values, Joint_NN_values, success_rate]
    else:
        return [obj_ori_error, obj_ori_error_koopman, demo_ori_error, success_rate]    


def koopman_policy_control_reorientation(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type):
    print("Begin to compute the simulation errors!")
    hand_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_vel_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_ori_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_ori_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Joint_NN_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_RL = []
    fall_list_sim = []
    fall_list_koopman = []
    fall_list_RL = []
    success_rate = str()
    # e.set_env_state(path['init_state_dict'])
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_koopman = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = Test_data[k][0]['rgbd_feature']
        #np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        # init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'] + Test_data[k][0]['final_handpos'], np.zeros(6))
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        # For hand control test, we can set the initial pos of pen to be (0.15, 0, 0, 0, 0, 0), so that there is no contact.
        # init_state_dict['qpos'][num_handpos] = 0.15  
        init_state_dict['qvel'] = np.append(Test_data[k][0]['handvel'], np.zeros(6))
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        #pdb.set_trace()
        e.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict
        #Joint_NN_values[:, 0, k] = e.get_obs()[:24]
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39])
                    obj_pos_world = x_t_1_computed[24:27]
                else:
                    obj_ori_world = ori_transform_inverse(x_t_1_computed[51:54], e.get_obs()[36:39]) 
                    obj_pos_world = x_t_1_computed[48:51]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            Computed_joint_values[:, t + 1, k] = hand_pos
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos = hand_pos[6:]
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:24] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            #Computed_torque_values[:, t, k] = NN_output
            #Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            # if Visualize:
            #     e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            gt_obj_pos = Test_data[k][t + 1]['objpos']
            gt_obj_orient = Test_data[k][t + 1]['objorient']  # in demo data, it is defined in a new coordinate 
            gt_obj_orien_world_frame = ori_transform_inverse(gt_obj_orient, e.get_obs()[36:39])
            gt_obj_vel = Test_data[k][t + 1]['objvel']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:24]
            #Joint_NN_values[:, t + 1, k] = hand_pos
            obj_pos = e.get_obs()[24:27]
            # Test_data[k][*]['desired_ori'] is the same value
            obj_orient = ori_transform(e.get_obs()[33:36], Test_data[k][t]['desired_ori'])  # change the coordinate
            obj_vel = e.get_obs()[27:33]          
            obj_obs = e.get_obs()
            # compute the errors
            #hand_pos_error[t, k] = np.mean(np.abs(gt_hand_pos - hand_pos))  # probably not return the mean value?
            hand_pos_PID_error[t, k] = np.mean(np.abs(hand_pos_desired - hand_pos))  
            obj_pos_error[t, k] = np.mean(np.abs(gt_obj_pos - obj_pos) + np.abs(gt_obj_orient - obj_orient))
            obj_vel_error[t, k] = np.mean(np.abs(gt_obj_vel - obj_vel))
            obj_ori_error[t, k] = np.mean(np.abs(obj_obs[42:45]))  # obj_orien-desired_orien (goal error)
            obj_ori_error_koopman[t, k] = np.mean(np.abs(obj_ori_world - obj_obs[36:39]))  # goal error in koopman
            demo_ori_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            # compute the success rate
            orien_similarity_sim = np.dot(obj_obs[33:36], obj_obs[36:39])
            orien_similarity_koopman = np.dot(obj_ori_world, obj_obs[36:39])
            orien_similarity_RL = np.dot(gt_obj_orien_world_frame, obj_obs[36:39])
            dist = np.linalg.norm(obj_obs[39:42])
            desired_pos = -obj_obs[39:42]+obj_obs[24:27]
            dist_koopman = np.linalg.norm(obj_pos_world - desired_pos)
            dist_RL = np.linalg.norm(gt_obj_pos - desired_pos)
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
            success_count_koopman[t] = 1 if (orien_similarity_koopman > 0.90) else 0
            success_count_RL[t] = 1 if (orien_similarity_RL > 0.90) else 0
        if np.abs(obj_obs[39:42])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
                print(f"success in {k}")
                success_list_sim.append(1)
        if np.abs(obj_pos_world - desired_pos)[2] > 0.15:
            fall_list_koopman.append(1)
        else:
            if sum(success_count_koopman) > success_threshold:
                success_list_koopman.append(1)
        if np.abs(gt_obj_pos - desired_pos)[2] > 0.15:
            fall_list_RL.append(1)
        else:
            if sum(success_count_RL) > success_threshold:
                success_list_RL.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    # success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    # print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    # success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    # print("Success rate (RL) = %f" % (len(success_list_RL) / len(Test_data)))
    # print("Throw out rate (sim) = %f" % (len(fall_list_sim) / len(Test_data)))
    # success_rate += "Throw out rate (sim) = %f\n" % (len(fall_list_sim) / len(Test_data))
    # print("Throw out rate (koopman) = %f" % (len(fall_list_koopman) / len(Test_data)))
    # success_rate += "Throw out rate (koopman) = %f\n" % (len(fall_list_koopman) / len(Test_data))
    # print("Throw out rate (RL) = %f" % (len(fall_list_RL) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error, hand_pos_PID_error, Computed_torque_values, Demo_torque_values, Computed_joint_values, Joint_NN_values, success_rate]
    else:
        return [obj_ori_error, obj_ori_error_koopman, demo_ori_error, success_rate]    

def koopman_policy_control_relocate(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_demo = []
    success_rate = str()
    for k in range(len(Test_data)): #tqdm(
        success_count_sim = np.zeros(len(Test_data[k]) - 1)
        success_count_koopman = np.zeros(len(Test_data[k]) - 1)
        success_count_RL = np.zeros(len(Test_data[k]) - 1)
        num_handpos = len(Test_data[k][0]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = Test_data[k][0]['rgbd_feature']
        #np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                if not Velocity:
                    obj_pos_world = x_t_1_computed[30:33]
                    # this is the obj_pos in the new frame (converged object trajecotry), as modified in training:
                    # tmp['objpos'] = objpos - obs[45:48] # converged object position (obs[45:48] -> target position)
                    # this is also the error of the object relocation (current position - goal position)
                    # to accutally obtain the position in the world frame:
                    # obj_pos_world += init_state_dict['target_pos']
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[30:33]
                else:
                    obj_pos_world = x_t_1_computed[60:63]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['hand_qpos'] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            Computed_torque_values[:, t, k] = NN_output
            Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            # if Visualize:
            #     e.env.mj_render()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[6:] - hand_pos[6:]))  # probably not return the mean value?
            hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[6:] - hand_pos[6:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:6] - hand_pos[:6]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:6] - hand_pos[:6]))  
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[36:39]))  # obj_pos - tar_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_pos_world))  # goal error in koopman
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t]['observation']))
            if np.linalg.norm(obj_obs[36:39]) < 0.1:
                success_count_sim[t] = 1
            if np.linalg.norm(obj_pos_world) < 0.1:
                success_count_koopman[t] = 1
            if np.linalg.norm(Test_data[k][t]['observation']) < 0.1:
                success_count_RL[t] = 1
        if sum(success_count_sim) > success_threshold:
            #print(f"success in {k}")
            success_list_sim.append(1)
        if sum(success_count_koopman) > success_threshold:
            success_list_koopman.append(1)
        if sum(success_count_RL) > success_threshold:
            success_list_demo.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    # success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    # print("Success rate (koopman) = %f" % (len(success_list_koopman) / len(Test_data)))
    # success_rate += "Success rate (koopman) = %f\n" % (len(success_list_koopman) / len(Test_data))
    # print("Success rate (RL) = %f" % (len(success_list_demo) / len(Test_data)))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     


def koopman_policy_control_hammer(env_name, controller, koopman_object, koopman_matrix, Test_data, num_hand, num_obj, koopmanoption, error_type):
    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])  # tool_pos - nail_pos
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([26, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))): #len(Test_data)
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = Test_data[k][0]['rgbd_feature']
        #np.append(np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel'])), Test_data[k][0]['nail_goal'])
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['board_pos'] = Test_data[k][0]['init']['board_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                obj_pos_world = x_t_1_computed[52:55]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                obj_pos_world = x_t_1_computed[52:55]
            
            #hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos = x_t_1_computed[:num_handpos]
            # NAZA
            hand_pos = np.concatenate((hand_pos[3:5], hand_pos[6:]))
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:26] #[:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            Computed_torque_values[:, t, k] = NN_output
            #Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            # if True:
            #     #if (t%5 == 0):
            #     img_obs, depth_obs = e.env.mj_render()
            #     from PIL import Image 
            #     img_obs = Image.fromarray(img_obs)
            #     img_obs.save(f"hammer_opening_{k}_{t}.png")
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.get_obs()[:26]#[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            # hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[2:] - hand_pos[2:]))  # probably not return the mean value?
            # hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[2:] - hand_pos[2:]))  
            # hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:2] - hand_pos[:2]))  # probably not return the mean value?
            # hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:2] - hand_pos[:2]))  
            # # instead of defining the position error of the nail, we define it as the difference between the hammer tool and the nail
            # obj_pos_error[t, k] = np.mean(np.abs(obj_obs[49:52]))  # tool_pos - target_pos (hammer getting closer to the nail)
            # obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_pos_world - obj_obs[42:45]))  # obj_obs[42:45] -> current nail pos
            # demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t + 1]['observation']))
            current_nail_pos = obj_obs[42:45]
            target_nail_pos = Test_data[k][t]['nail_goal'] #obj_obs[46:49]
            # print("target_nail_pos:", target_nail_pos)
            dist = np.linalg.norm(current_nail_pos - target_nail_pos)
            #print(f"current_nail_pos {current_nail_pos}, target_nail_pos {target_nail_pos}")
        if dist < 0.01:
            print(f"success in {k}")
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (1))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     



def koopman_policy_control_door(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize, obj_embedder, folder_name):

    print("Begin to compute the simulation errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_PID_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])  # palm_pos - handle_pos
    demo_pos_error = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    Demo_torque_values = np.zeros([num_hand, len(Test_data[0]) - 1, len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    #e.generate_unseen_data_door(10)
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    print(f"len(Test_data) {len(Test_data)}")
    
   
    # resnet_model = torch.load(os.path.join(folder_name, "Door/door_resnet.pt"))#
    # resnet_model.eval()
    # resnet_init_model = resnet_model #torch.load(os.path.join(folder_name, "Door/door_init_resnet.pt")) # currently, same model gives better result.
    # resnet_init_model = torch.load("/home/hychen/KODex/door_init_resnet.pt")
    # resnet_init_model.eval()
    for k in tqdm(range(len(Test_data))): #len(Test_data)
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']

        rgb, depth = e.env.mj_render()
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        #implict_objpos = resnet_model(torch.from_numpy(rgbd)).detach().numpy()[0]
        implict_objpos = Test_data[k][0]['rgbd_feature']
        #obj_embedder(torch.from_numpy(Test_data[k][0]['objpos'])).detach().numpy()
        #print(f"predicted objpos {implict_objpos}, gt {implict_objpos_gt}")
        #implict_objpos_init = obj_embedder(torch.from_numpy(Test_data[k][0]['handle_init'])).detach().numpy()
        #resnet_init_model(torch.from_numpy(rgbd)).detach().numpy()[0]
        #obj_embedder(torch.from_numpy(Test_data[k][0]['handle_init'])).detach().numpy()    
        obj_OriState = implict_objpos
        #np.append(implict_objpos, implict_objpos_init)
        #np.append(implict_objpos, np.append(Test_data[k][0]['objvel'], implict_objpos_init))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        #pdb.set_trace()
        # len(Test_data[k]) - 1
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                obj_pos_world = x_t_1_computed[56:59]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos = hand_pos[2:] # NAZA
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos  # control frequency
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            #Computed_torque_values[:, t, k] = NN_output
            #Demo_torque_values[:, t, k] = Test_data[k][t]['action']
            #print(f"NN_output {NN_output}")
            #random_action = np.random.uniform(low=-2, high=2, size=(len(NN_output),))
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            # if Visualize:
            #     #if (t%5 == 0):
            #     img_obs, depth_obs = e.env.mj_render()
            #     from PIL import Image 
            #     img_obs = Image.fromarray(img_obs)
            #     img_obs.save(f"door_opening_{k}_{t}.png")
                #plt.show(block = False)
                #e.render()
                #print(depth_obs)
                #
                #e.env.sim.render(width = 64, height = 64, depth=True)
                #img_obs = e.env.viewer.read_pixels(1376, 752, depth=False, segmentation=True)#[1]
                
                # try:
                #     img_obs = e.env.viewer.read_pixels(1376, 752, depth=False)
                # except:
                #     e.env.mj_viewer_setup()
                # img_obs = e.env.viewer.read_pixels(1376, 752, depth=False)
                # # # # save image
                
                #img_obs.show()
            # ground-truth state values (obtained from RL in simulator)
            # gt_hand_pos = Test_data[k][t + 1]['handpos'] + Test_data[k][0]['final_handpos']
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts (in simulator)
            hand_pos = e.env.get_full_obs_visualization()[:num_handpos]     
            obj_obs = e.get_obs()
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            #hand_pos_PID_error_handJoint[t, k] = np.mean(np.abs(hand_pos_desired[4:] - hand_pos[4:]))  
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            hand_pos_PID_error_base[t, k] = np.mean(np.abs(hand_pos_desired[:4] - hand_pos[:4]))  
            # instead of defining the position error of the nail, we define it as the difference between the hammer tool and the nail
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            #pdb.set_trace()
            #obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            demo_pos_error[t, k] = np.mean(np.abs(Test_data[k][t + 1]['observation'])) # palm-handle
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            print(f"success in {k}")
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (1))
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_PID_error_handJoint, hand_pos_error_base, hand_pos_PID_error_base, Computed_torque_values, Demo_torque_values, success_rate]
    else:
        return [obj_pos_error, obj_pos_error_koopman, demo_pos_error, success_rate]     

def koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, Visualize):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 70 # selected time steps that is enough to finish the task with good performance
    obj_pos_error = np.zeros([horizon - 1, len(Test_data)])
    obj_pos_error_koopman = np.zeros([horizon - 1, len(Test_data)])
    e.reset()
    init_state_dict = dict()
    # e.set_env_state(path['init_state_dict'])
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k]['handpos'])
        if Velocity:
            hand_OriState = np.append(Test_data[k]['handpos'], Test_data[k]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k]['handpos']
        obj_OriState = np.append(Test_data[k]['objpos'], np.append(Test_data[k]['objvel'], Test_data[k]['handle_init']))
        init_state_dict['qpos'] = Test_data[k]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]
                else:
                    obj_pos_world = x_t_1_computed[56:59]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            if Visualize:
                e.env.mj_render()
            # calculated state values using Koopman rollouts (in simulator)      
            obj_obs = e.get_obs()
            # compute the errors
            obj_pos_error[t, k] = np.mean(np.abs(obj_obs[35:38]))  # palm_pos - handle_pos
            obj_pos_error_koopman[t, k] = np.mean(np.abs(obj_obs[29:32] - obj_pos_world))  # obj_obs[29:32] -> palm position
            current_hinge_pos = obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return obj_pos_error, obj_pos_error_koopman, success_rate
    
def koopman_error_visualization(env_name, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, error_type, Visualize):

    print("Begin to compute the koopman rollout errors!")
    hand_pos_error_handJoint = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    hand_pos_error_base = np.zeros([len(Test_data[0]) - 1, len(Test_data)])
    Computed_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    Demo_joint_values = np.zeros([num_hand, len(Test_data[0]), len(Test_data)])
    e = GymEnv(env_name)
    e.reset()
    state_dict = {}
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_handvel = len(Test_data[k][0]['handvel'])
        if Velocity:
            hand_OriState = np.append(Test_data[k][0]['handpos'], Test_data[k][0]['handvel'])  # at initial states
        else:
            hand_OriState = Test_data[k][0]['handpos']
        Demo_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        Computed_joint_values[:, 0, k] = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        else:
            input_state = np.append(hand_OriState, obj_OriState)
            input_state = torch.from_numpy(input_state)
            z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(len(Test_data[k]) - 1):
            e.KoopmanVisualize(seed = None, state_dict = state_dict) # visualize the demo data without any actions (more like a visualization)
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            else:
                x_t_1_computed = koopman_object.z_inverse(torch.from_numpy(z_t_1_computed)).numpy()
            # ground-truth state values
            gt_hand_pos = Test_data[k][t + 1]['handpos']
            # calculated state values using Koopman rollouts vb
            hand_pos = x_t_1_computed[:num_handpos]
            z_t = z_t_1_computed
            # print(x_t_1_computed[-3:]) # almost keep constant as the Test_data[k][0]['handle_init']
            if Velocity:
                hand_vel = x_t_1_computed[num_handpos:num_handpos + num_handvel]
            else:
                hand_vel = np.zeros(num_handvel)
            # compute the errors
            hand_pos_error_handJoint[t, k] = np.mean(np.abs(gt_hand_pos[4:] - hand_pos[4:]))  # probably not return the mean value?
            hand_pos_error_base[t, k] = np.mean(np.abs(gt_hand_pos[:4] - hand_pos[:4]))  # probably not return the mean value?
            Computed_joint_values[:, t + 1, k] = hand_pos
            Demo_joint_values[:, t + 1, k] = gt_hand_pos
            state_dict['qpos'] = np.append(hand_pos, np.zeros([2]))
            state_dict['qvel'] = np.zeros([30])
            if Visualize:  # Since the PID controller works well, we do not need to visualize the trajectories any more
                e.env.mj_render()
    if error_type == 'demo':
        return [hand_pos_error_handJoint, hand_pos_error_base, Computed_joint_values, Demo_joint_values]

