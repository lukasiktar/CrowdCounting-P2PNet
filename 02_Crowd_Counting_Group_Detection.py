import os
import time
import cv2
import warnings
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from engine import *
from models import build_model
from scipy.spatial import ConvexHull


warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    return parser

def match_clusters(curr: dict,
                   prev: dict,
                   max_cluster_centroid_diff_dist: int=50,
                   max_cluster_area_diff: float=0.05
                   ):
    """match_clusters function performs matching the current clusters with the tracked/previous clusters by comparing the cluster_centroid_diff_distance between the centroids and the area differences."""
    cluster_centroid_diff_dist = np.linalg.norm(curr["centroid"] - prev["centroid"])

    cluster_area_diff = abs(curr["area"] - prev["area"]) / max(curr["area"], prev["area"])

    return (cluster_centroid_diff_dist < max_cluster_centroid_diff_dist) and (cluster_area_diff < max_cluster_area_diff)

def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def detect_groups(image : cv2.Mat ,points : list, tracked_clusters : dict):
    """Function that detects small group between the detected people.
     It defines the radius around detected points and analyses the group formation.
     Args:
        image - original image
        points - points represents the detected people
        tracked_clusters - tracked groups through time
        
    Returns:
        img - map that represents the detected groups
        tracket_clusters - tracked groups through time"""
    try:
        #DBSCAN initialization
        db = DBSCAN(eps=35.3, min_samples=15).fit(points)
        #Assigned labels to the points - each label presents the group (-1 == not in any group)
        labels = db.labels_
        next_label=0
        label_points_list=[]
        #Initialzation of clusters in current frame
        current_clusters=[]
        #Sorting the points to the clusters/label groups
        label_groups = {}
        for label in np.unique(labels):
            label_groups[label] = points[labels == label]

        #Calculating the centroid and area for each specific cluster/label group
        for label_group, points in label_groups.items():
            label_centroid=points.mean(axis=0)
            if label_group != -1:
                if len(points) < 10:
                    empty_img = np.ones_like(image) * 255
                    return tracked_clusters
            
                hull=ConvexHull(points)
                label_area = hull.volume
                label_points = points[hull.vertices]
            
                # create a **new list per cluster**
                label_points_list = [label_points]
                print(f"label:{label_group}, points: {len(points)}, label_centroid:{label_centroid},label_area: {label_area}")
                #Storing the detected clusters to the current cluster list (each cluster stored as dict)
                current_clusters.append({"label_group":label_group,
                                        "label_points_list": label_points_list,
                                        "centroid": label_centroid,
                                        "area": label_area,
                                        "points": points})


        #First iteration - there is no tracked clusters 
        if tracked_clusters is None:
            tracked_clusters = {}
            next_label = 0

            for curr in current_clusters:
                if curr["label_group"] == -1:
                    continue

                #Store the clusters into tracked
                tracked_clusters[next_label] = {
                    "centroid": curr["centroid"],
                    "area": curr["area"],
                    "points": curr["points"],
                    "label_points_list" : curr["label_points_list"],
                    "color": random_color()
                }

                # overwrite DBSCAN label with persistent label
                curr["label_group"] = next_label
                next_label += 1
        else:
            new_tracked = {}
            used_tracked_labels = set()

            for curr in current_clusters:
                if curr["label_group"] == -1:
                    continue

                matched_label = None 

                for label, prev in tracked_clusters.items():
                    if label in used_tracked_labels:
                        continue

                    if match_clusters(curr, prev, max_cluster_centroid_diff_dist=50, max_cluster_area_diff=850.0):
                        matched_label = label
                        break

                if matched_label is not None:
                    #Inherit cluster
                    curr["label_group"] = matched_label
                    new_tracked[matched_label] = {
                        "centroid": curr["centroid"],
                        "area": curr["area"],
                        "points": curr["points"],
                        "label_points_list" : curr["label_points_list"],
                        "color": tracked_clusters[matched_label]["color"]
                    }
                    used_tracked_labels.add(matched_label)
                else:
                    #New cluster
                    curr["label_group"] = next_label
                    new_tracked[next_label] = {
                        "centroid": curr["centroid"],
                        "area": curr["area"],
                        "points": curr["points"],
                        "label_points_list" : curr["label_points_list"],
                        "color": random_color()
                    }
                    next_label += 1

        tracked_clusters = new_tracked

        return tracked_clusters
    except:
        tracked_clusters = {}
        return tracked_clusters




def main(args, debug=False):
    torch.cuda.reset_peak_memory_stats()
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(f"Starting arguments: {args}")
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your video path here
    #video_path = "./02_Original_testing_videos/Marathon_02.mp4"
    #video_path = "./02_Original_testing_videos/Event_01.mp4"
    video_path = "./02_Original_testing_videos/Demonstrations_day_01.mp4"
    #video_path = "./02_Original_testing_videos/Crowd_day_01.mp4"
    counter=0
    cap = cv2.VideoCapture(video_path)

    # Get input video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = 2*int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output path
    out_path = "./01_Crowd_Counting/CrowdCounting-P2PNet/vis/Marathon_output.mp4"

    # Define codec (mp4 recommended)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter object
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    tracked_clusters = {}
    next_label = 0

    while True:
        success, img_raw = cap.read()
        if not success:
            break

        counter+=1

        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # round the size
        height , width,  _ = img_raw.shape
        new_width = width // 256 * 256
        new_height = height // 256 * 256
        img_raw = cv2.resize(img_raw, (new_width, new_height))
        start = time.time()

        # pre-proccessing
        img = transform(img_raw)
        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        threshold = 0.3
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()
        predict_cnt = int((outputs_scores > threshold).sum())

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        # draw the predictions
        size = 2
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        number_of_detected_people= len(points)


        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak memory used: {peak_mem:.2f} MB")
        end = time.time()

        print(f"Inference time : {end-start} s")


    
        tracked_clusters=detect_groups(img_raw, points, tracked_clusters)
        for cid, cluster in tracked_clusters.items():
            color = cluster["color"]
            for polygons in cluster["label_points_list"]:
                contour = polygons.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_to_draw, [contour], True, color, 2)
        
        #stacked_detection_map=np.hstack((img_to_draw,crowd_map))

        cv2.imshow("Crowd Counting", img_to_draw)
        cv2.waitKey(1)
        #Store the video
        #stacked_detection_map = cv2.cvtColor(stacked_detection_map, cv2.COLOR_RGB2BGR)
        stacked_detection_map = cv2.resize(img_to_draw, (width*2, height))

        out.write(stacked_detection_map)
    



    cap.release()
    out.release()
        # save the visualized image
        #cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(counter)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)