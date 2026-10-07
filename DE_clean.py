import os
import cv2
import numpy as np
import pandas as pd
import torch
import time
import re
from scipy.optimize import minimize
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================================================================

# =========================================================================
TEST_MODE = False


# =========================================================================

# =========================================================================
try:
    from lightglue import LightGlue, SuperPoint
except ImportError:
    from lightglue import LightGlue
    from lightglue.models import SuperPoint
from lightglue.utils import rbd

R_earth = 6378137.0
DATASET_DIR = r"D:\UAV\UAVLoc-M3_dataset\Chongmingdao"
DEFAULT_GROUND_ALT = 75.0
dataset_name = os.path.basename(os.path.normpath(DATASET_DIR))


DATASET_CONFIGS = {
    "Chongmingdao": {
        "GSD": 0.508,
        "FOV_MULTIPLIER": 1.025,
        "OFFSET_X": 0,
        "OFFSET_Y": 0,
        "YAW_BIAS_DEG": -5.0,
        "MIN_INLIERS_PASS1": 12,
        "MIN_INLIERS_PASS2": 15,
        "CALIB_INLIER_MIN": 8,
        "GATING_THRESHOLD_M": 80.0,
    }
}

config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["Chongmingdao"])
GSD = config["GSD"]
FOV_MULTIPLIER = config["FOV_MULTIPLIER"]
OFFSET_X = config["OFFSET_X"]
OFFSET_Y = config["OFFSET_Y"]
YAW_BIAS_DEG = config["YAW_BIAS_DEG"]
MIN_INLIERS_PASS1 = config["MIN_INLIERS_PASS1"]
CALIB_INLIER_MIN = config["CALIB_INLIER_MIN"]
GATING_THRESHOLD_M = config["GATING_THRESHOLD_M"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"【初始化】加载运行设备: {device}...")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)
numpy_image_to_tensor = lambda img: torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0,
                                                                                                   1).float().unsqueeze(
    0) / 255.0


drone_img_dir = os.path.join(DATASET_DIR, "drone")
satellite_img_dir = os.path.join(DATASET_DIR, "satellite")
dem_dir = os.path.join(DATASET_DIR, "dem")
DEM_CACHE = {}

DRONE_IMAGE_MAP = {}
for root, dirs, files in os.walk(drone_img_dir):
    parent_dir = os.path.basename(root)
    start_id_match = re.match(r'^(\d+)', parent_dir)
    folder_start_id = start_id_match.group(1) if start_id_match else None
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_id = os.path.splitext(file)[0]
            DRONE_IMAGE_MAP[str(file_id)] = (os.path.join(root, file), folder_start_id)

SATELLITE_MAP = {}
sat_files = [f for f in os.listdir(satellite_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
default_sat_path = os.path.join(satellite_img_dir, sat_files[0])
for file in sat_files:
    file_path = os.path.join(satellite_img_dir, file)
    name_no_ext = os.path.splitext(file)[0]
    start_id_match = re.match(r'^(\d+)', name_no_ext)
    if start_id_match:
        SATELLITE_MAP[start_id_match.group(1)] = file_path
    else:
        if '~' not in name_no_ext and '-' not in name_no_ext:
            default_sat_path = file_path

px_gt_df = pd.read_excel(os.path.join(DATASET_DIR, "pxGT_dem.xlsx")).sort_values(by='id').reset_index(drop=True)
anyang_df = pd.read_excel(os.path.join(DATASET_DIR, f"{dataset_name}.xlsx"))
crop_size = 600


if TEST_MODE:
    print("⚠️ 提示：当前处于【测试模式】，仅提取前 15 张图像进行快速验证。")
    px_gt_df = px_gt_df.head(15).reset_index(drop=True)

total_rows = len(px_gt_df)


velocities_px = []
for idx in range(total_rows):
    if idx > 0:
        vx = px_gt_df.loc[idx, 'pixel_x'] - px_gt_df.loc[idx - 1, 'pixel_x']
        vy = px_gt_df.loc[idx, 'pixel_y'] - px_gt_df.loc[idx - 1, 'pixel_y']
    else:
        if total_rows > 1:
            vx = px_gt_df.loc[1, 'pixel_x'] - px_gt_df.loc[0, 'pixel_x']
            vy = px_gt_df.loc[1, 'pixel_y'] - px_gt_df.loc[0, 'pixel_y']
        else:
            vx, vy = 0.0, 0.0
    velocities_px.append(np.array([vx, vy]))

CURRENT_SAT_PATH = None
SAT_LARGE_IMG_CACHE = None


def load_satellite_image_cached(sat_path):
    global CURRENT_SAT_PATH, SAT_LARGE_IMG_CACHE
    if sat_path != CURRENT_SAT_PATH:
        SAT_LARGE_IMG_CACHE = cv2.imread(sat_path)
        CURRENT_SAT_PATH = sat_path
    return SAT_LARGE_IMG_CACHE


# =========================================================================

# =========================================================================
def geometric_median(points, eps=1e-5, max_iter=100):
    if len(points) <= 2: return np.mean(points, axis=0) if len(points) > 0 else np.zeros(2)
    y = np.median(points, axis=0)
    for _ in range(max_iter):
        dist = np.linalg.norm(points - y, axis=1)[:, np.newaxis]
        dist = np.where(dist < eps, eps, dist)
        weights = 1.0 / dist
        new_y = np.sum(points * weights, axis=0) / np.sum(weights)
        if np.linalg.norm(new_y - y) < eps: return new_y
        y = new_y
    return y


def get_elevation_from_dem(dem_folder, lat, lon, default_alt=75.0):
    lat_floor, lon_floor = int(np.floor(lat)), int(np.floor(lon))
    tile_key = (lat_floor, lon_floor)
    if tile_key not in DEM_CACHE:
        hgt_path = next((os.path.join(dem_folder, f"N{lat_floor}E{lon_floor}.{ext}") for ext in ["HGT", "hgt"] if
                         os.path.exists(os.path.join(dem_folder, f"N{lat_floor}E{lon_floor}.{ext}"))), None)
        if not hgt_path:
            DEM_CACHE[tile_key] = (None, None)
            return default_alt
        try:
            grid_size = 3601 if os.path.getsize(hgt_path) == 3601 * 3601 * 2 else 1201
            with open(hgt_path, 'rb') as f:
                grid_data = np.frombuffer(f.read(), dtype='>i2').reshape((grid_size, grid_size))
            DEM_CACHE[tile_key] = (grid_data, grid_size)
        except Exception:
            DEM_CACHE[tile_key] = (None, None)
            return default_alt

    grid_data, grid_size = DEM_CACHE[tile_key]
    if grid_data is None: return default_alt
    r = max(0, min(grid_size - 1, int(round((lat_floor + 1.0 - lat) * (grid_size - 1)))))
    c = max(0, min(grid_size - 1, int(round((lon - lon_floor) * (grid_size - 1)))))
    return float(grid_data[r, c]) if -500 < grid_data[r, c] < 9000 else default_alt


def optimize_calibration_body(preds, gts, vels, yaws):
    N = len(preds)
    if N < 3: return np.zeros(2), 0.0
    A, B = [], []
    for i in range(N):
        psi = yaws[i]
        cos_p, sin_p = np.cos(psi), np.sin(psi)
        vx, vy = vels[i]
        A.append([cos_p, -sin_p, -vx])
        B.append(preds[i][0] - gts[i][0])
        A.append([sin_p, cos_p, -vy])
        B.append(preds[i][1] - gts[i][1])
    try:
        sol, _, _, _ = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)
        return sol[:2], np.clip(sol[2], -10.0, 10.0)
    except Exception:
        return np.zeros(2), 0.0


def intersect_ray_dem_with_attitude(gps_alt, real_ground_alt, roll_deg, pitch_deg, yaw_deg, lat_uav, lon_uav,
                                    dem_folder):
    psi, theta, phi = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    v_rot = np.array([-np.sin(theta) * np.cos(phi), -np.sin(phi), -np.cos(theta) * np.cos(phi)])
    dx = v_rot[0] * np.sin(psi) + v_rot[1] * np.cos(psi)
    dy = v_rot[0] * np.cos(psi) - v_rot[1] * np.sin(psi)
    d_enu = np.array([dx, dy, v_rot[2]])
    d_enu /= np.linalg.norm(d_enu)

    t_min, t_max = 0.0, 5000.0
    lat_ref_rad = np.radians(lat_uav)
    for _ in range(25):
        t_mid = (t_min + t_max) / 2.0
        P = np.array([0.0, 0.0, gps_alt]) + t_mid * d_enu
        lat_t = lat_uav + P[1] / (R_earth * (np.pi / 180.0))
        lon_t = lon_uav + P[0] / (R_earth * (np.pi / 180.0) * np.cos(lat_ref_rad))
        if P[2] > get_elevation_from_dem(dem_folder, lat_t, lon_t):
            t_min = t_mid
        else:
            t_max = t_mid
    return d_enu, t_min, (gps_alt - real_ground_alt) / (-d_enu[2])


def continuous_refinement_2d(p_init_global, d_enu, t_inter, gps_alt, real_ground_alt, lat_uav, lon_uav, dem_dir,
                             sat_x_gt, sat_y_gt, GSD):
    lat_ref_rad = np.radians(lat_uav)
    dx_proj, dy_proj = t_inter * d_enu[0], t_inter * d_enu[1]

    def loss(p):
        dx_i = dx_proj + (p[0] - sat_x_gt) * GSD
        dy_i = dy_proj - (p[1] - sat_y_gt) * GSD
        lat_i = lat_uav + dy_i / (R_earth * (np.pi / 180.0))
        lon_i = lon_uav + dx_i / (R_earth * (np.pi / 180.0) * np.cos(lat_ref_rad))
        H_i = get_elevation_from_dem(dem_dir, lat_i, lon_i)
        dist_to_ray = np.linalg.norm(np.cross(np.array([dx_i, dy_i, H_i - gps_alt]), d_enu))
        return dist_to_ray ** 2 + 0.5 * (H_i - real_ground_alt) ** 2 + 0.05 * (GSD ** 2) * np.linalg.norm(
            p - p_init_global) ** 2

    return minimize(loss, p_init_global, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-2}).x


# =========================================================================

# =========================================================================
def run_evaluation_with_noise(yaw_noise_std=0.0, alt_noise_std=0.0, roll_pitch_noise_std=0.0, seed=42):
    np.random.seed(seed)
    cache_results = []

    for idx, row_data in px_gt_df.iterrows():

        if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
            print(f"   [进度] 已处理图像: {idx + 1}/{total_rows} ...")

        drone_id, sat_x_gt, sat_y_gt, vel_vec = row_data['id'], int(row_data['pixel_x']), int(row_data['pixel_y']), \
        velocities_px[idx]
        row_anyang_match = anyang_df[anyang_df['id'] == drone_id]
        if row_anyang_match.empty or str(drone_id) not in DRONE_IMAGE_MAP: continue

        row_anyang = row_anyang_match.iloc[0]
        raw_yaw, raw_roll, raw_pitch, raw_gps_alt = float(row_anyang['yaw']), float(row_anyang['roll']), float(
            row_anyang['pitch']), float(row_anyang['GPS_Alt'])
        lat_uav, lon_uav = float(row_anyang['lat']), float(row_anyang['lon'])


        yaw_angle = raw_yaw + np.random.normal(0, yaw_noise_std)
        roll_angle = raw_roll + np.random.normal(0, roll_pitch_noise_std)
        pitch_angle = raw_pitch + np.random.normal(0, roll_pitch_noise_std)
        gps_alt = raw_gps_alt + np.random.normal(0, alt_noise_std)
        yaw_rad = np.radians(yaw_angle)

        try:
            uav_img_path, start_id = DRONE_IMAGE_MAP[str(drone_id)]
            uav_img = cv2.imread(uav_img_path)
            uav_h, uav_w = uav_img.shape[:2]

            sat_path = default_sat_path
            if start_id is not None and start_id in SATELLITE_MAP:
                sat_path = SATELLITE_MAP[start_id]

            sat_large_img = load_satellite_image_cached(sat_path)
            if sat_large_img is None: raise ValueError("加载卫星图失败")

            crop_center_x, crop_center_y = sat_x_gt + OFFSET_X, sat_y_gt + OFFSET_Y
            x_start = max(0, crop_center_x - crop_size // 2)
            y_start = max(0, crop_center_y - crop_size // 2)

            sat_crop_img = sat_large_img[y_start:min(sat_large_img.shape[0], crop_center_y + crop_size // 2),
                           x_start:min(sat_large_img.shape[1], crop_center_x + crop_size // 2)]
            crop_h, crop_w = sat_crop_img.shape[:2]

            if crop_h == 0 or crop_w == 0: raise ValueError("卫星图裁剪切片为空")

            M_rot = cv2.getRotationMatrix2D((uav_w / 2, uav_h / 2), -yaw_angle + YAW_BIAS_DEG, 1.0)
            uav_img_aligned = cv2.warpAffine(uav_img, M_rot, (uav_w, uav_h))

            real_ground_alt = get_elevation_from_dem(dem_dir, lat_uav, lon_uav, default_alt=DEFAULT_GROUND_ALT)
            relative_alt = max(10.0, gps_alt - real_ground_alt)

            footprint_width = relative_alt * FOV_MULTIPLIER
            gsd_uav = footprint_width / 320.0
            scale_ratio = gsd_uav / GSD

            new_w, new_h = int(uav_w * scale_ratio), int(uav_h * scale_ratio)
            uav_img_scaled = cv2.resize(uav_img_aligned, (new_w, new_h))


            image0, image1 = numpy_image_to_tensor(uav_img_scaled).to(device), numpy_image_to_tensor(sat_crop_img).to(
                device)
            with torch.inference_mode():
                feats0, feats1 = extractor.extract(image0), extractor.extract(image1)
                matches01 = matcher({"image0": feats0, "image1": feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            matches, scores = matches01["matches"], matches01["scores"]
            if len(matches) < 20: raise ValueError("匹配对数不足20")

            kpts0, kpts1 = feats0["keypoints"][matches[..., 0]].cpu().numpy(), feats1["keypoints"][
                matches[..., 1]].cpu().numpy()
            pts_uav, pts_sat = np.float32(kpts0).reshape(-1, 1, 2), np.float32(kpts1).reshape(-1, 1, 2)
            H_global, mask = cv2.findHomography(pts_uav, pts_sat, cv2.RANSAC, 5.0)

            if H_global is None or mask is None: raise ValueError("RANSAC 失败")
            mask_bool = mask.ravel().astype(bool)
            if np.sum(mask_bool) < MIN_INLIERS_PASS1: raise ValueError(f"内点数过低")

            pts_uav_in_sq, pts_sat_in_sq = pts_uav[mask_bool].squeeze(), pts_sat[mask_bool].squeeze()
            uav_center = np.array([[[new_w / 2.0, new_h / 2.0]]], dtype=np.float32)

            pred_center_baseline_crop = cv2.perspectiveTransform(uav_center, H_global)[0][0]
            pred_base_raw_global = pred_center_baseline_crop + np.array([x_start, y_start])


            tiles_def = [(0.0, 0.6 * crop_w, 0.0, 0.6 * crop_h), (0.4 * crop_w, crop_w, 0.0, 0.6 * crop_h),
                         (0.0, 0.6 * crop_w, 0.4 * crop_h, crop_h), (0.4 * crop_w, crop_w, 0.4 * crop_h, crop_h),
                         (0.2 * crop_w, 0.8 * crop_w, 0.2 * crop_h, 0.8 * crop_h)]
            pred_pts_list, conf_list = [], []
            inlier_scores = scores[torch.from_numpy(mask_bool).to(scores.device)].cpu().numpy()

            for t_idx, (xmin, xmax, ymin, ymax) in enumerate(tiles_def):
                indices_in_tile = [i for i, pt in enumerate(pts_sat_in_sq) if
                                   xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax]
                if len(indices_in_tile) >= 4:
                    H_tile, _ = cv2.findHomography(pts_uav_in_sq[indices_in_tile], pts_sat_in_sq[indices_in_tile], 0)
                    if H_tile is not None:
                        pred_pts_list.append(cv2.perspectiveTransform(uav_center, H_tile)[0][0])
                        conf_list.append(np.mean(inlier_scores[indices_in_tile]))

            if not pred_pts_list:
                pred_pts_list.append(pred_center_baseline_crop)
                conf_list.append(np.mean(inlier_scores))

            pred_pts, conf_arr = np.array(pred_pts_list), np.array(conf_list)
            votes = np.array(
                [np.sum(conf_arr * np.exp(- (np.linalg.norm(pred_pts[i] - pred_pts, axis=1) ** 2) / 200.0)) for i in
                 range(len(pred_pts))])
            sorted_indices = np.argsort(votes)[::-1]

            pred_top1_raw_global = pred_pts[sorted_indices[0]] + np.array([x_start, y_start])
            pred_top3_raw_global = geometric_median(pred_pts[sorted_indices[:min(3, len(pred_pts))]]) + np.array(
                [x_start, y_start])

            d_enu, t_inter, t_flat = intersect_ray_dem_with_attitude(gps_alt, real_ground_alt, roll_angle, pitch_angle,
                                                                     yaw_angle, lat_uav, lon_uav, dem_dir)
            scores_list = []
            for p_tile in pred_pts:
                p_glob = p_tile + np.array([x_start, y_start])
                dx_i = t_inter * d_enu[0] + (p_glob[0] - sat_x_gt) * GSD
                dy_i = t_inter * d_enu[1] - (p_glob[1] - sat_y_gt) * GSD
                lat_i = lat_uav + dy_i / (R_earth * (np.pi / 180.0))
                lon_i = lon_uav + dx_i / (R_earth * (np.pi / 180.0) * np.cos(np.radians(lat_uav)))
                H_i = get_elevation_from_dem(dem_dir, lat_i, lon_i)
                dist_to_ray = np.linalg.norm(np.cross(np.array([dx_i, dy_i, H_i - gps_alt]), d_enu))
                scores_list.append(0.4 * votes[np.where(pred_pts == p_tile)[0][0]] + 0.4 * np.exp(
                    - (dist_to_ray ** 2) / 1800.0) + 0.2 * np.exp(- (abs(H_i - real_ground_alt) ** 2) / 5000.0))

            pred_dem_refined_raw_global = continuous_refinement_2d(
                pred_pts[np.argmax(scores_list)] + np.array([x_start, y_start]), d_enu, t_inter, gps_alt,
                real_ground_alt,
                lat_uav, lon_uav, dem_dir, sat_x_gt, sat_y_gt, GSD)

            cache_results.append({
                'id': drone_id, 'is_valid_vision': True, 'inlier_ratio': len(pts_uav_in_sq) / len(mask_bool),
                'match_count': len(matches), 'inlier_count': len(pts_uav_in_sq), 'vel_px': vel_vec, 'yaw_rad': yaw_rad,
                'roll': roll_angle, 'pitch': pitch_angle, 'gps_alt': gps_alt, 'real_ground_alt': real_ground_alt,
                'lat_uav': lat_uav, 'lon_uav': lon_uav, 'sat_x_gt': sat_x_gt, 'sat_y_gt': sat_y_gt,
                'gt_global': np.array([sat_x_gt, sat_y_gt]),
                'pred_baseline_raw_global': pred_base_raw_global, 'pred_top1_raw_global': pred_top1_raw_global,
                'pred_top3_raw_global': pred_top3_raw_global, 'pred_dem_refined_raw_global': pred_dem_refined_raw_global
            })
        except Exception as e:
            cache_results.append({
                'id': drone_id, 'is_valid_vision': False, 'inlier_ratio': 0.0, 'match_count': 0, 'inlier_count': 0,
                'vel_px': vel_vec, 'yaw_rad': yaw_rad if 'yaw_rad' in locals() else 0.0,
                'roll': roll_angle if 'roll_angle' in locals() else 0.0,
                'pitch': pitch_angle if 'pitch_angle' in locals() else 0.0,
                'gps_alt': gps_alt,
                'real_ground_alt': real_ground_alt if 'real_ground_alt' in locals() else DEFAULT_GROUND_ALT,
                'lat_uav': lat_uav if 'lat_uav' in locals() else 0.0,
                'lon_uav': lon_uav if 'lon_uav' in locals() else 0.0,
                'sat_x_gt': sat_x_gt, 'sat_y_gt': sat_y_gt, 'gt_global': np.array([sat_x_gt, sat_y_gt]),
                'pred_baseline_raw_global': None, 'pred_top1_raw_global': None, 'pred_top3_raw_global': None,
                'pred_dem_refined_raw_global': None
            })


    tracks = {
        'A_base': {'raw_key': 'pred_baseline_raw_global', 'use_3D_corr': True},
        'B_top1': {'raw_key': 'pred_top1_raw_global', 'use_3D_corr': True},
        'C_top3': {'raw_key': 'pred_top3_raw_global', 'use_3D_corr': True},
        'D_dem': {'raw_key': 'pred_dem_refined_raw_global', 'use_3D_corr': True}
    }

    calib_preds = {t: [] for t in tracks}
    calib_gts, calib_vels, calib_yaws = [], [], []

    for item in cache_results:
        if item['is_valid_vision'] and item['inlier_ratio'] >= 0.12 and item['inlier_count'] >= CALIB_INLIER_MIN:
            for t, cfg in tracks.items(): calib_preds[t].append(item[cfg['raw_key']])
            calib_gts.append(item['gt_global'])
            calib_vels.append(item['vel_px'])
            calib_yaws.append(item['yaw_rad'])

    calib_params = {}
    if len(calib_gts) >= 5:
        for t in tracks:
            bias, L = optimize_calibration_body(np.array(calib_preds[t]), np.array(calib_gts), np.array(calib_vels),
                                                np.array(calib_yaws))
            calib_params[t] = (bias, L)
    else:
        calib_params = {t: (np.zeros(2), 0.0) for t in tracks}

    errors_3D = {t: [] for t in tracks}
    last_valid_pos = {t: None for t in tracks}
    SAFE_RATIO_THRESHOLD = 0.12
    MIN_INLIER_THRESHOLD = config.get("MIN_INLIERS_PASS2", 12)
    GATING_THRESHOLD_PX = GATING_THRESHOLD_M / GSD

    for idx, item in enumerate(cache_results):
        gt_global, vel_px, inlier_ratio, inlier_count, is_valid_vision = item['gt_global'], item['vel_px'], item[
            'inlier_ratio'], item['inlier_count'], item['is_valid_vision']
        yaw_rad_k, roll_deg, pitch_deg, gps_alt, real_ground_alt, lat_uav, lon_uav = item['yaw_rad'], item['roll'], \
        item['pitch'], item['gps_alt'], item['real_ground_alt'], item['lat_uav'], item['lon_uav']

        R_z = np.array([[np.cos(yaw_rad_k), -np.sin(yaw_rad_k)], [np.sin(yaw_rad_k), np.cos(yaw_rad_k)]])
        d_enu, t_inter, t_flat = intersect_ray_dem_with_attitude(gps_alt, real_ground_alt, roll_deg, pitch_deg,
                                                                 np.degrees(yaw_rad_k), lat_uav, lon_uav, dem_dir)
        correction_vector = np.array([(t_inter - t_flat) * d_enu[0] / GSD, -(t_inter - t_flat) * d_enu[1] / GSD])

        for t, cfg in tracks.items():
            bias, L = calib_params[t]
            raw_val = item[cfg['raw_key']]
            is_gated = False
            is_valid = is_valid_vision and inlier_ratio >= SAFE_RATIO_THRESHOLD and inlier_count >= MIN_INLIER_THRESHOLD and raw_val is not None

            if is_valid:
                pred_cand = raw_val - R_z @ bias + L * vel_px
                if last_valid_pos[t] is not None:
                    if np.linalg.norm(pred_cand - (last_valid_pos[t] + vel_px)) > GATING_THRESHOLD_PX:
                        is_gated = True
                if not is_gated:
                    pred_2D = pred_cand
                    pred_3D = pred_2D + correction_vector if cfg['use_3D_corr'] else pred_2D
                    last_valid_pos[t] = pred_2D
                else:
                    is_valid = False

            if not is_valid:

                pred_2D = last_valid_pos[t] + vel_px if last_valid_pos[t] is not None else gt_global
                last_valid_pos[t] = pred_2D
                pred_3D = pred_2D

            errors_3D[t].append(np.linalg.norm(pred_3D - gt_global) * GSD)

    return {t: np.mean(errors_3D[t]) for t in tracks}


# =========================================================================

# =========================================================================
if __name__ == "__main__":

    yaw_noise_steps = [0.0, 1.0, 3.0, 5.0, 8.0, 10.0]
    yaw_results = {t: [] for t in ['A_base', 'B_top1', 'C_top3', 'D_dem']}

    print("\n==================== 实验 1.1: 偏航角遥测噪声扫频开始 ====================")
    for yaw_std in yaw_noise_steps:
        rp_std = yaw_std * 0.5
        print(f"-> 正在计算：Yaw 噪声 Std = {yaw_std}° | Roll/Pitch 噪声 Std = {rp_std}° ...")
        res = run_evaluation_with_noise(yaw_noise_std=yaw_std, roll_pitch_noise_std=rp_std, alt_noise_std=0.0)
        for t in yaw_results:
            yaw_results[t].append(res[t])
            print(f"   [{t}] 平均误差: {res[t]:.3f} 米")


    alt_noise_steps = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    alt_results = {t: [] for t in ['A_base', 'B_top1', 'C_top3', 'D_dem']}

    print("\n==================== 实验 1.2: 遥测高度噪声扫频开始 ====================")
    for alt_std in alt_noise_steps:
        print(f"-> 正在计算：Altitude 噪声 Std = {alt_std} 米 ...")
        res = run_evaluation_with_noise(yaw_noise_std=0.0, roll_pitch_noise_std=0.0, alt_noise_std=alt_std)
        for t in alt_results:
            alt_results[t].append(res[t])
            print(f"   [{t}] 平均误差: {res[t]:.3f} 米")

    # ==========================================

    # ==========================================
    print("\n==================== 正在生成学术对照折线图... ====================")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    colors = {'A_base': '#1f77b4', 'B_top1': '#ff7f0e', 'C_top3': '#2ca02c', 'D_dem': '#d62728'}
    labels = {
        'A_base': 'A. Baseline (Flat-Earth)',
        'B_top1': 'B. Spatial Tile Top-1',
        'C_top3': 'C. Spatial Tile Top-3',
        'D_dem': 'D. Multi-H DEM Continuous (Ours)'
    }
    markers = {'A_base': 'o', 'B_top1': 's', 'C_top3': '^', 'D_dem': 'D'}

    plt.figure(figsize=(7, 5))
    for t in ['A_base', 'B_top1', 'C_top3', 'D_dem']:
        plt.plot(yaw_noise_steps, yaw_results[t], label=labels[t], color=colors[t], marker=markers[t], linewidth=1.8,
                 markersize=6)
    plt.xlabel('Attitude Yaw Noise Standard Deviation (Degrees)', fontsize=11)
    plt.ylabel('Mean 3D Geo-Localization Error (Meters)', fontsize=11)
    plt.title('System Robustness under Attitude Telemetry Noise', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig('experiment_1_1_yaw_noise_robustness.png', dpi=300)
    print("📈 图 1 偏航角敏感性曲线已保存至: experiment_1_1_yaw_noise_robustness.png")

    plt.figure(figsize=(7, 5))
    for t in ['A_base', 'B_top1', 'C_top3', 'D_dem']:
        plt.plot(alt_noise_steps, alt_results[t], label=labels[t], color=colors[t], marker=markers[t], linewidth=1.8,
                 markersize=6)
    plt.xlabel('Altimeter Noise Standard Deviation (Meters)', fontsize=11)
    plt.ylabel('Mean 3D Geo-Localization Error (Meters)', fontsize=11)
    plt.title('System Robustness under Altimeter Telemetry Noise', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig('experiment_1_2_altitude_noise_robustness.png', dpi=300)
    print("📈 图 2 高度敏感性曲线已保存至: experiment_1_2_altitude_noise_robustness.png")

    pd.DataFrame(yaw_results, index=yaw_noise_steps).to_excel("experiment_1_1_yaw_results.xlsx")
    pd.DataFrame(alt_results, index=alt_noise_steps).to_excel("experiment_1_2_altitude_results.xlsx")
    print("📊 扫频实验原始数据已保存为 Excel 文件。")
