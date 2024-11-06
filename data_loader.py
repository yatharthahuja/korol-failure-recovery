import rosbag
import pdb 
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
from tqdm import tqdm

class RGBDDataset(Dataset):
   def __init__(self, img_path):
       self.img_path = img_path
       self.all_files = os.listdir(img_path)

   def __len__(self):
       return len(self.all_files)

   def __getitem__(self, index):
       path = os.path.join(self.img_path, str(index)+".npy")
       rgbd = np.load(path)
       rgbd = np.transpose(rgbd, (2, 0, 1))
       return rgbd, index



def demo_playback(demo_paths, generate_feature_data_path):
    Training_data = []
    with open(generate_feature_data_path, 'rb') as handle:
        feature_data = pickle.load(handle)
  
    print("Begin loading demo data!")
    count = 0
    # demo_1 = pickle.load(open("./demo1/joint_states.pickle", 'rb'))
    # demo_2 = pickle.load(open("./demo4/joint_states.pickle", 'rb'))
    # merged_data = [demo_1] + [demo_2]
    # with open("./data/joint_states.pickle", 'wb') as output_file:
    #     pickle.dump(merged_data, output_file)
    # pdb.set_trace()
    demos = pickle.load(open(demo_paths, 'rb'))
    sample_index = np.arange(len(demos)) #np.arange(len(demos))
    for t in tqdm(sample_index):
        path = demos[t]
        path_data = []
        #print(f"t {t}, len {len(path)}")
        for i in range(len(path)): #len(path)
            tmp = dict()
            # NAZA
            tmp['handpos'] = path[i]
            dict_value = feature_data[count].values()
            feature = list(dict_value)[0]
            tmp['rgbd_feature'] = feature
            tmp['count'] = count
            count += 1
            path_data.append(tmp)
        Training_data.append(path_data)
        #pdb.set_trace()
    return Training_data


