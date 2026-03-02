import os
import warnings
import argparse
import time
import cv2
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from engine import *
from models import build_model

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
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
    video_path = "./01_Crowd_Counting/CrowdCounting-P2PNet/vis/PeopleWalking.mp4"
    counter=0
    cap = cv2.VideoCapture(video_path)

    # Get input video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output path
    out_path = "./01_Crowd_Counting/CrowdCounting-P2PNet/vis/output_people_counting.mp4"

    # Define codec (mp4 recommended)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter object
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

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

        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]
        # draw the predictions
        size = 2
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        number_of_detected_people= len(points)
        img_to_draw = cv2.putText(img_to_draw, f"Detected people: {predict_cnt}", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak memory used: {peak_mem:.2f} MB")
        end = time.time()

        print(f"Inference time : {end-start} s")

        cv2.imshow("Crowd Couinting", img_to_draw)
        cv2.waitKey(1)


        #Store the video
        img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_RGB2BGR)
        img_to_draw = cv2.resize(img_to_draw, (width, height))

        out.write(img_to_draw)

    cap.release()
    out.release()
        # save the visualized image
        #cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(counter)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)