import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import timm

try:
    from lightglue import LightGlue, SuperPoint
except ImportError:
    print("Error: lightglue library not found.")
    exit()

DATASET_ROOT = "./data"
MODEL_PATH_COLOR = "./checkpoints/color_model.pth"
MODEL_PATH_STRUCT = "./checkpoints/struct_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 12
MAX_KEYPOINTS = 1024
GEOMETRY_THRESH = 3.0
MIN_INLIERS = 10
DEBUG_DIR = "./debug_errors_pure_geometry"
os.makedirs(DEBUG_DIR, exist_ok=True)

class OrientationAwareContextModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch_square = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                                           nn.BatchNorm2d(in_channels), nn.ReLU(True))
        self.branch_horizontal = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 7), padding=(0, 3), groups=in_channels),
            nn.BatchNorm2d(in_channels), nn.ReLU(True))
        self.branch_vertical = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0), groups=in_channels),
            nn.BatchNorm2d(in_channels), nn.ReLU(True))
        self.fusion = nn.Sequential(nn.Conv2d(in_channels * 3, in_channels, 1), nn.BatchNorm2d(in_channels),
                                    nn.ReLU(True))

    def forward(self, x):
        return x + self.fusion(
            torch.cat([self.branch_square(x), self.branch_horizontal(x), self.branch_vertical(x)], dim=1))

class StudentModel(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.features = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=0,
                                          global_pool='')
        in_ch = 1280
        self.oacm = OrientationAwareContextModule(in_ch)
        self.projector = nn.Sequential(
            nn.Linear(in_ch, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.oacm(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        feat = self.projector(x)
        return F.normalize(feat, p=2, dim=1)

class UniversitySystemEnsemble:
    def __init__(self, data_root):
        self.device = DEVICE
        self.sat_dir = os.path.join(data_root, "satellite")
        self.drone_dir = os.path.join(data_root, "drone")

        self.model_color = StudentModel().to(self.device)
        self._load_weights(self.model_color, MODEL_PATH_COLOR, "Color-Texture Model")

        self.model_struct = StudentModel().to(self.device)
        self._load_weights(self.model_struct, MODEL_PATH_STRUCT, "Spatial-Structure Model")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.local_extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)
        self.gallery = []
        self.gallery_mean = None
        self._build_gallery()

    def _load_weights(self, model, path, name):
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(
                {k.replace('module.', ''): v for k, v in state_dict.items() if 'classifier' not in k}, strict=False)
            model.eval()
            print(f"Success: {name} weights loaded.")
        else:
            print(f"Error: Cannot find {name} weights at {path}")
            exit()

    @torch.no_grad()
    def _extract_super_vector(self, image_tensor):
        vec_c = self.model_color(image_tensor)
        vec_s = self.model_struct(image_tensor)
        super_vec = F.normalize(torch.cat([vec_c, vec_s], dim=1), p=2, dim=1)
        return super_vec

    def _build_gallery(self):
        print("\nBuilding satellite gallery...")
        id_feature_map = {}

        for root, _, files in os.walk(self.sat_dir):
            id_name = os.path.basename(root)
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.tif')):
                    path = os.path.join(root, file)
                    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                    if img is None: continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        super_vec = self._extract_super_vector(tensor)
                        local_feats = self.local_extractor.extract(
                            torch.from_numpy(rgb.transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(self.device))

                    if id_name not in id_feature_map:
                        id_feature_map[id_name] = {'vecs': [], 'locals': [], 'paths': []}
                    id_feature_map[id_name]['vecs'].append(super_vec)
                    id_feature_map[id_name]['locals'].append(local_feats)
                    id_feature_map[id_name]['paths'].append(path)

        all_proto_vecs = []
        for id_name, data in id_feature_map.items():
            stacked_vecs = torch.cat(data['vecs'], dim=0)
            proto_vec = F.normalize(torch.mean(stacked_vecs, dim=0, keepdim=True), p=2, dim=1)
            self.gallery.append({
                'id': id_name, 'path': data['paths'][0],
                'super_vec': proto_vec, 'local_feats': data['locals'][0]
            })
            all_proto_vecs.append(proto_vec)

        self.gallery_vecs = torch.cat(all_proto_vecs, dim=0)
        self.gallery_mean = torch.mean(self.gallery_vecs, dim=0, keepdim=True)
        self.gallery_vecs = F.normalize(self.gallery_vecs - self.gallery_mean, p=2, dim=1)
        print(f"Gallery built. Total unique IDs: {len(self.gallery)}")

    def run_test(self):
        print("\nStarting multi-metric evaluation...")
        stats = {"global_r1": 0, "global_r12": 0, "final_r1": 0}
        total = 0
        total_time = 0.0

        for folder in os.scandir(self.drone_dir):
            if not folder.is_dir(): continue
            true_id = folder.name.lower()

            for img_file in os.scandir(folder.path):
                img_bgr = cv2.imdecode(np.fromfile(img_file.path, dtype=np.uint8), -1)
                if img_bgr is None: continue
                total += 1
                start_time = time.time()

                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    drone_tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                    super_vec_q = self._extract_super_vector(drone_tensor)
                    super_vec_q = F.normalize(super_vec_q - self.gallery_mean, p=2, dim=1)

                    local_feats_q = self.local_extractor.extract(
                        torch.from_numpy(rgb.transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(self.device))

                scores = torch.mm(super_vec_q, self.gallery_vecs.t()).squeeze(0)
                _, top_indices = torch.topk(scores, k=min(TOP_K, len(self.gallery)))

                global_top1_idx = int(top_indices[0])
                if self.gallery[global_top1_idx]['id'].lower() == true_id:
                    stats["global_r1"] += 1

                top_ids = [self.gallery[int(i)]['id'].lower() for i in top_indices]
                if true_id in top_ids:
                    stats["global_r12"] += 1

                best_match_idx = int(top_indices[0])
                final_pred_id = self.gallery[best_match_idx]['id'].lower()
                if final_pred_id == true_id:
                    stats["final_r1"] += 1

                latency = (time.time() - start_time) * 1000
                total_time += latency
                status = "PASS" if final_pred_id == true_id else "FAIL"
                print(f"[{status}] {true_id} -> {final_pred_id} ({latency:.1f}ms)")

        if total > 0:
            print("\n" + "=" * 45)
            print(f"Evaluation Results (Total Samples: {total})")
            print("-" * 45)
            print(f"1. Global Retrieval R@1:  {stats['global_r1'] / total * 100:.2f}%")
            print(f"2. Global Retrieval R@12: {stats['global_r12'] / total * 100:.2f}%")
            print(f"3. Final System R@1:      {stats['final_r1'] / total * 100:.2f}%")
            print("=" * 45)

if __name__ == "__main__":
    system = UniversitySystemEnsemble(DATASET_ROOT)
    system.run_test()