import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

import warnings
warnings.filterwarnings('ignore')

from data.datagen import JIGSAWS_tasks, image_features_save_path


class ResNet50Features(torch.nn.Module):
    def __init__(self):
        super(ResNet50Features, self).__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50 = torch.nn.Sequential(*(list(self.resnet50.children())[:-1]))
        for p in self.resnet50.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        return x


def extract_features_from_video(video_path, model, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 16
    features = []

    with torch.no_grad():
        model.to(device)
        model.eval()

        for i in range(0, frame_count, batch_size):
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    print(f'reading "{video_path}" occurs some error ')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = transforms.ToTensor()(frame)
                frame_tensor = transforms.Resize((224, 224))(frame_tensor)
                frame_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame_tensor)
                frames.append(frame_tensor)

            if len(frames) == 0:
                break

            batch = torch.stack(frames).to(device)
            feature_vectors = model(batch).cpu().numpy()
            features.extend(feature_vectors)

    cap.release()
    return np.array(features)

def save_features_to_file(features, features_path):
    np.save(features_path, features)

if __name__ == "__main__":
    # Load ResNet-50 model
    model = ResNet50Features()
    
    if not os.path.exists(image_features_save_path):
        os.mkdir(image_features_save_path)

    i = 0
    # for task in JIGSAWS_tasks:
    for task in JIGSAWS_tasks:

        video_paths = list()

        task_features_save_path = os.path.join(image_features_save_path, task)
        print(f"task_features_save_path:{task_features_save_path}")
        if not os.path.exists(task_features_save_path):
            os.mkdir(task_features_save_path)
        pwdPath = os.getcwd()      # '~/chenting/PredictionTrajectory/multimodalTransformer/data'
        root_path = os.path.join(os.path.dirname(os.path.dirname(pwdPath)), "compassSurgicalActivityRecognition", "Datasets", "dV")
        # print(f"the  curDir:{pwdPath}")
        # print(f"the parent path of curDir:{os.path.dirname(pwdPath)}")
        video_folder_path = os.path.join(root_path, task, 'video')
        print(f"debug video_folder_path: {video_folder_path}")
        for file_name in os.listdir(video_folder_path):
            video_path = os.path.join(video_folder_path, file_name)

            if ('right' in video_path.lower()) or (not os.path.exists(video_path.lower().replace('left', 'right'))):
                video_paths.append(video_path)
                
        for video_path in video_paths:
            print("Processing video:", video_path)
            features = extract_features_from_video(video_path, model)
            features_path = os.path.join(task_features_save_path, os.path.basename(video_path)[:-4] + '.npy')
            save_features_to_file(features, features_path)



