import bagpy
import rosbag
from bagpy import bagreader
import pandas as pd
import numpy as np
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
import pickle


def read_rosbag_image(demo_number):

    demo_name = f'demo{demo_number}'
    bag_file = f'./{demo_name}.bag'
    data_dir = f'./{demo_name}'

    bag = rosbag.Bag(bag_file, 'r')
    bridge = CvBridge()

    topic_rgb = '/camera/color/image_raw'
    topic_depth = '/camera/depth/image_rect_raw'
    topic_joint_states = '/yk_destroyer/joint_states'

    img_count = 0
    saved_images = 0
    data_dir = f'./{demo_name}/images/color'
    for topic, msg, t in bag.read_messages(topics=[topic_rgb]):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

            # cv2.imshow("Image", cv_image)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()

            # save every 10th image
            if img_count % 10 == 0:
                cv2.imwrite(f'./{data_dir}/{img_count}.png', cv_image)
                saved_images += 1

            img_count += 1
        
        except CvBridgeError as e:
            print(e)

    img_count = 0
    data_dir = f'./{demo_name}/images/depth'
    for topic, msg, t in bag.read_messages(topics=[topic_depth]):
        try:
            # cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            # Convert the ROS image message to an OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Convert the depth image to a numpy array
            depth_array = np.array(cv_image, dtype=np.float32)

            # cv2.imshow("Image", cv_image)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()

            # save every 10th image
            if img_count % 10 == 0:
                cv2.imwrite(f'./{data_dir}/{img_count}.png', depth_array)

            img_count += 1

        except CvBridgeError as e:
            print(e)

    # combine rgb and depth images into a single image and save each as npy
    data_dir = f'./{demo_name}/images/combined'
    
    for i in range(saved_images):
        rgb_img = cv2.imread(f'./{demo_name}/images/color/{i*10}.png')
        # normalize the rgb image to 0-1
        rgb_img = rgb_img / 255.0
        
        depth_img = cv2.imread(f'./{demo_name}/images/depth/{i*10}.png', cv2.IMREAD_UNCHANGED)
        # normalize the depth image to 0-1
        depth_img = depth_img / np.max(depth_img)        

        combined_img = np.dstack((rgb_img, depth_img))
        np.save(f'./{data_dir}/{i*10}.npy', combined_img)

    # read the joint states into a pandas dataframe
    joint_states = []
    for topic, msg, t in bag.read_messages(topics=[topic_joint_states]):
        joint_states.append(msg.position)
    
    if joint_states == []:
        print("No joint states found in the rosbag file")
        return
    
    indices = np.linspace(0, len(joint_states)-1, saved_images, dtype=int)
    joint_states = [joint_states[i] for i in indices]
    with open(f'./{demo_name}/joint_states.pickle', 'wb') as f:
        pickle.dump(joint_states, f)

    bag.close()
    

if __name__ == "__main__":
    read_rosbag_image(2)