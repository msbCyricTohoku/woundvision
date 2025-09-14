import os
import cv2
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

#scipy import -- nice way to check
try:
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linregress = None

SKIMAGE_AVAILABLE = False
try:
    from skimage.feature import graycomatrix, graycoprops
    from skimage import img_as_ubyte
    from skimage.color import rgb2gray
    SKIMAGE_AVAILABLE = True
except ImportError:
    graycomatrix = None
    graycoprops = None

#detectron2 import
DETECTRON2_AVAILABLE = False
try:
    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    pass

#config setup
def setup_cfg(model_weights, score_thresh=0.5, log_queue=None):
    def log_message(msg):
        if log_queue:
            log_queue.put(msg)
        else:
            print(msg)

    cfg = get_cfg()
    config_file_path = "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    
    try:
        import detectron2.model_zoo
        try:
            config_file_path = detectron2.model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            log_message(f"Found config via model_zoo.get_config_file: {config_file_path}")
        except Exception as e_zoo:
            log_message(f"ERROR: Config file not found via model_zoo: {e_zoo}. Critical error.")
            raise FileNotFoundError("Detectron2 config file '.../mask_rcnn_R_50_FPN_3x.yaml' not found.") from e_zoo
    except Exception as e_outer_config:
        log_message(f"ERROR: Problem resolving Detectron2 config path: {e_outer_config}")
        raise

    if not os.path.exists(config_file_path):
        log_message(f"ERROR: Detectron2 config file '{config_file_path}' not found.")
        raise FileNotFoundError(f"Detectron2 config file '{config_file_path}' not found.")

    try:
        log_message(f"Attempting to load config from: {config_file_path}")
        cfg.merge_from_file(config_file_path)
    except Exception as e:
        log_message(f"ERROR: Config path failed: {e}.")
        raise RuntimeError(f"Failed to load Detectron2 config: {e}") from e

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"
    log_message(f"Using {cfg.MODEL.DEVICE} device.")
    cfg.freeze()
    return cfg


def calculate_and_save_histogram(image, mask, output_path, bins=64, log_queue=None):
    def local_log(msg):
        if log_queue: log_queue.put(msg)
        else: print(msg)

    hist_fig = None
    try:
        if mask.dtype != np.uint8: mask_uint8 = mask.astype(np.uint8) * 255
        else: mask_uint8 = mask
        if np.sum(mask_uint8) == 0: return False
        colors = ('b', 'g', 'r')
        hist_fig = plt.figure(figsize=(6, 4))
        plt.title('Color Histogram (Wound Area)')
        plt.xlabel("Bins")
        plt.ylabel("# Pixels")
        has_data = False
        for i, color in enumerate(colors):
            try:
                hist = cv2.calcHist([image], [i], mask_uint8, [bins], [0, 256])
                if hist is not None and len(hist) > 0:
                    plt.plot(hist, color=color)
                    has_data = True
            except cv2.error as e:
                local_log(f"WARN: Error calculating histogram for channel {color}: {e}")
        if has_data:
            plt.xlim([0, bins])
            plt.grid(True, axis='y', linestyle=':')
            try:
                plt.savefig(output_path)
                return True
            except Exception as e:
                local_log(f"WARN: Error saving histogram plot {output_path}: {e}")
                return False
        else:
            local_log("WARN: No valid histogram data to plot.")
            return False
    finally:
        if hist_fig:
            plt.close(hist_fig)

def get_dominant_colors(image, mask, k=3, log_queue=None):
    def local_log(msg):
        if log_queue: log_queue.put(msg)
        else: print(msg)
    if mask.dtype == bool: mask_bool = mask
    else: mask_bool = mask.astype(bool)
    pixels = image[mask_bool]
    if pixels.shape[0] < k:
        if pixels.shape[0] > 0:
            return [tuple(map(int, np.mean(pixels, axis=0)))]
        return None

    pixels = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_centers = centers[sorted_indices]
        return [tuple(map(int, color)) for color in sorted_centers]
    except cv2.error as e:
        local_log(f"WARN: K-Means error: {e}. Returning average color.")
        if pixels.shape[0] > 0:
            return [tuple(map(int, np.mean(pixels, axis=0)))]
        return None

def draw_color_swatches(colors, swatch_size=50, spacing=10):
    if not colors: return None
    num_colors = len(colors)
    width = (swatch_size + spacing) * num_colors + spacing
    height = swatch_size + 2 * spacing
    swatch_img = np.zeros((height, width, 3), dtype=np.uint8) + 255
    for i, color in enumerate(colors):
        start_x = spacing + i * (swatch_size + spacing)
        bgr_color = tuple(map(int, color))
        cv2.rectangle(swatch_img, (start_x, spacing), (start_x + swatch_size, spacing + swatch_size), bgr_color, -1)
        cv2.rectangle(swatch_img, (start_x, spacing), (start_x + swatch_size, spacing + swatch_size), (0,0,0), 1)
    return swatch_img

def plot_pseudo_3d_wound(image, mask, output_path, elevation=5, stride_div=50, log_queue=None):
    """Generates a 3D plot with the same X-Y dimensions as the original image."""
    plot_fig = None
    try:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool): return

        h_orig, w_orig = image.shape[:2]
        stride = max(1, min(h_orig, w_orig) // stride_div)

        #here we create meshgrid for the entire image
        x = np.arange(w_orig)
        y = np.arange(h_orig)
        X, Y = np.meshgrid(x, y)
        
        #Z plane for the entire image, with elevation only at the wound
        Z = np.zeros_like(X, dtype=float)
        Z[mask_bool] = elevation
        
        #create an RGBA color map for the surface
        #wound area is colored, non-wound area is transparent
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        alpha_channel = np.zeros(image.shape[:2], dtype=np.uint8)
        alpha_channel[mask_bool] = 255 #this make wound area opaque
        rgba_image[:, :, 3] = alpha_channel
        
        plot_fig = plt.figure(figsize=(12, 9))
        ax = plot_fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, facecolors=rgba_image/255.0, linewidth=0, antialiased=True, shade=False)

        ax.set_title(f'Pseudo-3D Visualization (Full Image Context)', fontsize=12)
        ax.set_xlabel('X pixels'); ax.set_ylabel('Y pixels'); ax.set_zlabel('Artificial Height')
        ax.set_zlim(0, elevation + 2)
        ax.view_init(elev=30, azim=60)
        ax.invert_yaxis()
        plt.savefig(output_path, dpi=150)
    except Exception as e:
        if log_queue: log_queue.put(f"WARN: 3D Plot Error: {e}")
    finally:
        if plot_fig: plt.close(plot_fig)


def estimate_tissue_percentages(image, mask, log_queue=None):
    mask_bool = mask.astype(bool)
    default_pct = {'Granulation': 0.0, 'Slough': 0.0, 'Necrotic': 0.0, 'Other': 0.0}
    if not np.any(mask_bool): return default_pct, None

    coords = np.argwhere(mask_bool)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    wound_pixels_hsv = hsv_image[mask_bool]
    total_pixels = wound_pixels_hsv.shape[0]
    if total_pixels == 0: return default_pct, None

    necrotic_mask = (wound_pixels_hsv[:, 2] < 55)
    slough_mask = ((wound_pixels_hsv[:, 0] >= 20) & (wound_pixels_hsv[:, 0] < 45)) & (wound_pixels_hsv[:, 1] > 40) & (wound_pixels_hsv[:, 2] >= 55) & (~necrotic_mask)
    granulation_mask = (((wound_pixels_hsv[:, 0] >= 160) | (wound_pixels_hsv[:, 0] < 15)) & (wound_pixels_hsv[:, 1] > 50) & (wound_pixels_hsv[:, 2] >= 55)) & (~necrotic_mask) & (~slough_mask)
    other_mask = ~(necrotic_mask | slough_mask | granulation_mask)

    counts = {'Necrotic': np.sum(necrotic_mask), 'Slough': np.sum(slough_mask), 'Granulation': np.sum(granulation_mask), 'Other': np.sum(other_mask)}
    percentages = {k: float(round((v / total_pixels) * 100, 1)) for k, v in counts.items()}

    vis_mask_img = np.zeros_like(image)
    colors_bgr = {'Necrotic': (0, 0, 0), 'Slough': (0, 220, 220), 'Granulation': (0, 0, 200), 'Other': (128, 128, 128)}
    masks_1d = {'Necrotic': necrotic_mask, 'Slough': slough_mask, 'Granulation': granulation_mask, 'Other': other_mask}

    for tissue_type, mask_1d_values in masks_1d.items():
        tissue_coords = coords[mask_1d_values]
        if len(tissue_coords) > 0:
            vis_mask_img[tissue_coords[:, 0], tissue_coords[:, 1]] = colors_bgr[tissue_type]

    return percentages, vis_mask_img

def analyze_periwound(image, mask, dilation_pixels=15, log_queue=None):
    def local_log(msg):
        if log_queue: log_queue.put(msg)
        else: print(msg)
    mask_uint8 = mask.astype(np.uint8) * 255
    if not np.any(mask): return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_pixels, dilation_pixels))
    try:
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
        periwound_mask = cv2.subtract(dilated_mask, mask_uint8)
        periwound_mask_bool = periwound_mask > 0
        if not np.any(periwound_mask_bool): return None
        mean_bgr = np.mean(image[periwound_mask_bool], axis=0)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv_image[periwound_mask_bool], axis=0)
        return {'Mean_BGR': tuple(map(float, mean_bgr.round(1))), 'Mean_HSV': tuple(map(float, mean_hsv.round(1)))}
    except cv2.error as e:
        local_log(f"WARN: Periwound analysis error: {e}")
        return None

def calculate_texture_features(image, mask, log_queue=None):
    def local_log(msg):
        if log_queue: log_queue.put(msg)
        else: print(msg)
    if not SKIMAGE_AVAILABLE: return None
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool): return None
    try:
        gray_image_uint8 = img_as_ubyte(rgb2gray(image))
        coords = np.argwhere(mask_bool)
        if coords.shape[0] < 2: return None
        rows, cols = coords.T
        rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
        gray_cropped = gray_image_uint8[rmin:rmax+1, cmin:cmax+1]
        mask_cropped = mask_bool[rmin:rmax+1, cmin:cmax+1]
        
        masked_gray_pixels_cropped = gray_cropped[mask_cropped]
        if masked_gray_pixels_cropped.size == 0 or np.std(masked_gray_pixels_cropped) < 1e-6:
             return {'contrast': 0.0, 'dissimilarity': 0.0, 'homogeneity': 1.0, 'energy': 1.0, 'correlation': 1.0, 'ASM': 1.0}

        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_cropped, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        texture_features = {prop: np.mean(graycoprops(glcm, prop)) for prop in properties}
        return {k: float(round(v, 4)) for k, v in texture_features.items()}
    except Exception as e:
        local_log(f"WARN: Texture calculation error: {e}")
        return None

def calculate_sharpness(image, mask, log_queue=None):
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool): return None
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = np.argwhere(mask_bool)
        if coords.shape[0] < 9: return None
        rows, cols = coords.T
        rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
        gray_cropped = gray_image[rmin:rmax+1, cmin:cmax+1]
        if gray_cropped.size < 9: return None
        return float(round(cv2.Laplacian(gray_cropped, cv2.CV_64F).var(), 2))
    except Exception as e:
        if log_queue: log_queue.put(f"WARN: Sharpness calculation error: {e}")
        return None

def calculate_fractal_dimension_dbc(image, mask, output_plot_path=None, log_queue=None):
    def local_log(msg):
        if log_queue: log_queue.put(msg)
        else: print(msg)
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool): return None
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = np.argwhere(mask_bool)
        if coords.shape[0] < 4: return None
        rows, cols = coords.T
        rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
        h, w = rmax - rmin + 1, cmax - cmin + 1
        if h < 2 or w < 2: return None
        
        gray_cropped = gray_image[rmin:rmax+1, cmin:cmax+1]
        mask_cropped = mask_bool[rmin:rmax+1, cmin:cmax+1]
        
        if np.std(gray_cropped[mask_cropped]) < 1e-6: return 2.0

        max_box_limit = max(4, min(h, w) // 3)
        if max_box_limit < 2: return None
        box_sizes = sorted(list(set(np.round(np.geomspace(2, max_box_limit, num=8)).astype(int))))
        if not box_sizes: return None

        counts, valid_box_sizes = [], []
        for r in box_sizes:
            N_r = 0
            for j in range(0, h, r):
                for i in range(0, w, r):
                    block_mask = mask_cropped[j:min(j + r, h), i:min(i + r, w)]
                    if np.any(block_mask):
                        masked_block_pixels = gray_cropped[j:min(j + r, h), i:min(i + r, w)][block_mask]
                        if masked_block_pixels.size > 0:
                            intensity_range = int(np.max(masked_block_pixels)) - int(np.min(masked_block_pixels))
                            N_r += 1 if intensity_range == 0 else math.ceil(intensity_range / r)
            if N_r > 0:
                counts.append(N_r)
                valid_box_sizes.append(r)

        if len(counts) < 3: return None
        
        log_counts = np.log(np.array(counts))
        log_inv_sizes = np.log(1.0 / np.array(valid_box_sizes))
        
        if SCIPY_AVAILABLE:
            slope, intercept, r_value, _, _ = linregress(log_inv_sizes, log_counts)
            r_squared = r_value**2
            if r_squared < 0.90: local_log(f"  WARN: Low R^2 ({r_squared:.2f}) for FD fit.")
        else:
            slope, intercept = np.polyfit(log_inv_sizes, log_counts, 1)
            r_squared = None
        fractal_dim = slope

        if output_plot_path:
            plot_fig_fd, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(log_inv_sizes, log_counts, label='Data points', color='blue')
            ax.plot(log_inv_sizes, slope * log_inv_sizes + intercept, label=f'Fit (FD = {fractal_dim:.3f})', color='red', linestyle='--')
            title = f'Fractal Dimension (DBC) Log-Log Plot\nFD = {fractal_dim:.3f}'
            if r_squared is not None: title += f', R² = {r_squared:.3f}'
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('log(1 / Box Size r)'); ax.set_ylabel('log(Box Count N_r)'); ax.legend(); ax.grid(True)
            plt.tight_layout(); plt.savefig(output_plot_path); plt.close(plot_fig_fd)
        
        return float(round(fractal_dim, 3)) if not np.isnan(fractal_dim) else None
    except Exception as e:
        local_log(f"WARN: FD calculation error: {e}")
        return None


def create_composite_visualization(original_image, mask, instance_data, histogram_path, dominant_colors_swatch_img, tissue_vis_img, output_path, base_name_no_ext, instance_id, pixels_per_cm=None, log_queue=None):
    def local_log(msg):
        if log_queue: log_queue.put(msg)
        else: print(msg)
    comp_fig = None
    try:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool): return

        def format_cm(value, unit):
            return f"{float(value):.2f} {unit}" if value is not None else "N/A"

        coords = np.argwhere(mask_bool)
        rows, cols = coords.T
        rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
        padding = 20
        h_img, w_img = original_image.shape[:2]
        crop_rmin, crop_rmax = max(0, rmin - padding), min(h_img, rmax + 1 + padding)
        crop_cmin, crop_cmax = max(0, cmin - padding), min(w_img, cmax + 1 + padding)

        img_cropped = original_image[crop_rmin:crop_rmax, crop_cmin:crop_cmax]
        mask_cropped = mask_bool[crop_rmin:crop_rmax, crop_cmin:crop_cmax]
        tissue_vis_cropped = tissue_vis_img[crop_rmin:crop_rmax, crop_cmin:crop_cmax] if tissue_vis_img is not None else None

        comp_fig, ax = plt.subplots(3, 2, figsize=(14, 13))
        comp_fig.suptitle(f"Wound Analysis Summary: {base_name_no_ext} - Instance {instance_id}", fontsize=16)

        #overlay plot
        ax[0, 0].imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
        overlay = np.zeros((*mask_cropped.shape, 4), dtype=np.float32)
        overlay[mask_cropped] = [1.0, 0.0, 0.0, 0.4]
        ax[0, 0].imshow(overlay)
        ax[0, 0].set_title("Detected Wound Area (Cropped)"); ax[0, 0].axis('off')

        #isolated wound plot
        masked_region_only = img_cropped.copy(); masked_region_only[~mask_cropped] = 255
        ax[0, 1].imshow(cv2.cvtColor(masked_region_only, cv2.COLOR_BGR2RGB))
        ax[0, 1].set_title("Isolated Wound Shape"); ax[0, 1].axis('off')

        #histogram
        if histogram_path and os.path.exists(histogram_path):
            hist_img = cv2.imread(histogram_path)
            ax[1, 0].imshow(cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB))
        else:
            ax[1, 0].text(0.5, 0.5, "Histogram N/A", ha='center', va='center')
        ax[1, 0].set_title("Color Histogram"); ax[1, 0].axis('off')

        #dominant colors
        if dominant_colors_swatch_img is not None:
            ax[1, 1].imshow(cv2.cvtColor(dominant_colors_swatch_img, cv2.COLOR_BGR2RGB))
        else:
            ax[1, 1].text(0.5, 0.5, "Dominant Colors N/A", ha='center', va='center')
        ax[1, 1].set_title("Dominant Colors (BGR)"); ax[1, 1].axis('off')
        
        # Plot 2,0: Tissue Estimate
        if tissue_vis_cropped is not None:
             ax[2, 0].imshow(cv2.cvtColor(tissue_vis_cropped, cv2.COLOR_BGR2RGB))
             legend_text = 'Legend: ■ Necrotic (Black)  ■ Slough (Yellow)  ■ Granulation (Red)'
             ax[2, 0].text(0.5, -0.05, legend_text, transform=ax[2, 0].transAxes, fontsize=8, ha='center')
        else:
             ax[2, 0].text(0.5, 0.5, "Tissue Estimate N/A", ha='center', va='center')
        ax[2, 0].set_title("Heuristic Tissue Estimate"); ax[2, 0].axis('off')

        #metrics text in the plot
        metrics_text = "Key Metrics:\n\n"
        metrics_text += f" Area: {instance_data.get('Area_pixels2', 'N/A')} px² ({format_cm(instance_data.get('Area_cm2'), 'cm²')})\n"
        metrics_text += f" Perimeter: {instance_data.get('Perimeter_pixels', 0.0):.1f} px ({format_cm(instance_data.get('Perimeter_cm'), 'cm')})\n"
        metrics_text += f" Circularity: {instance_data.get('Circularity', 0.0):.3f}\n\n"
        tp = instance_data.get("TissuePercentages", {})
        metrics_text += "Tissue (%):\n"
        metrics_text += f"  Granulation: {tp.get('Granulation', 0.0):.1f}%\n"
        metrics_text += f"  Slough: {tp.get('Slough', 0.0):.1f}%\n"
        metrics_text += f"  Necrotic: {tp.get('Necrotic', 0.0):.1f}%\n\n"
        sh = instance_data.get("Sharpness")
        metrics_text += f"Sharpness (LapVar): {sh:.2f}\n" if sh is not None else "Sharpness: N/A\n"
        fd = instance_data.get("FractalDimension")
        metrics_text += f"Fractal Dim (DBC): {fd:.3f}\n" if fd is not None else "Fractal Dim: N/A\n"
        ax[2, 1].text(0.05, 0.95, metrics_text, ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))
        ax[2, 1].set_title("Calculated Metrics"); ax[2, 1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_path)

    except Exception as e:
        local_log(f"WARN: Failed to create composite viz: {e}")
    finally:
        if comp_fig: plt.close(comp_fig)


def simulate_healing_stages_erosion(initial_mask_uint8, num_stages=5, iterations_per_stage=2, log_queue=None):
    if initial_mask_uint8 is None or num_stages < 1: return []
    current_mask = initial_mask_uint8.copy()
    stage_masks = [current_mask.copy()]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(num_stages - 1):
        if np.sum(current_mask) == 0:
            stage_masks.extend([current_mask.copy()] * (num_stages - len(stage_masks)))
            break
        eroded_mask = cv2.erode(current_mask, kernel, iterations=iterations_per_stage)
        stage_masks.append(eroded_mask)
        current_mask = eroded_mask
    return stage_masks

def simulate_healing_stages_distance_transform(initial_mask_uint8, num_stages=5, shrink_factor=0.6, log_queue=None):
    if initial_mask_uint8 is None or num_stages < 1: return []
    current_mask = initial_mask_uint8.copy()
    if np.sum(current_mask) == 0: return [current_mask.copy()] * num_stages
    try:
        dist_transform = cv2.distanceTransform(current_mask, cv2.DIST_L2, 5)
        _, max_dist_val, _, _ = cv2.minMaxLoc(dist_transform, mask=current_mask)
        if max_dist_val <= 0: return [current_mask.copy()] * num_stages
        thresholds = np.linspace(0, max_dist_val * shrink_factor, num_stages)
        stage_masks = [cv2.bitwise_and(((dist_transform > t) * 255).astype(np.uint8), current_mask) for t in thresholds]
        return stage_masks
    except Exception as e:
        if log_queue: log_queue.put(f"WARN: Distance transform sim error: {e}")
        return []


def plot_healing_stages(original_image, stage_masks, output_path, base_name_no_ext, instance_id, method_name, log_queue=None):
    sim_plot_fig = None
    try:
        num_stages = len(stage_masks)
        if num_stages == 0: return
        colors_bgr = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
        ncols = min(num_stages, 5)
        nrows = math.ceil(num_stages / ncols)
        sim_plot_fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2), squeeze=False)
        axes = axes.flatten()
        sim_plot_fig.suptitle(f'Simulated Healing Stages ({method_name})\n{base_name_no_ext} - Instance {instance_id}', fontsize=12)

        for i, mask_stage in enumerate(stage_masks):
            ax = axes[i]
            stage_img_display = original_image.copy()
            if np.any(mask_stage):
                contours, _ = cv2.findContours(mask_stage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                thickness = max(2, int(original_image.shape[1] / 250))
                cv2.drawContours(stage_img_display, contours, -1, colors_bgr[i % len(colors_bgr)], thickness)
            
            ax.imshow(cv2.cvtColor(stage_img_display, cv2.COLOR_BGR2RGB))
            title = f"Original (Stage 0)" if i == 0 else f"Stage {i}"
            if i > 0 and np.sum(stage_masks[0]) > 0:
                reduction = 100 * (np.sum(stage_masks[0]) - np.sum(mask_stage)) / np.sum(stage_masks[0])
                title += f"\n(~{reduction:.0f}% Area Reduction)"
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        
        for j in range(num_stages, len(axes)): axes[j].axis('off')
        plt.tight_layout(rect=[0, 0.02, 1, 0.93]); plt.savefig(output_path)
    except Exception as e:
        if log_queue: log_queue.put(f"WARN: Failed to plot healing stages: {e}")
    finally:
        if sim_plot_fig: plt.close(sim_plot_fig)


def estimate_wound_depth(image, mask, log_queue=None):
    try:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool): return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient = cv2.Laplacian(gray, cv2.CV_64F)
        mean_depth = np.mean(gradient[mask_bool])
        std_depth = np.std(gradient[mask_bool])
        return {'MeanDepth': round(mean_depth, 2), 'StdDepth': round(std_depth, 2)}
    except Exception as e:
        if log_queue: log_queue.put(f"WARN: Depth Estimation Error: {e}")
        return None


def assess_margin_redness(image, mask, dilation=10, log_queue=None):
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1)
        ring = cv2.subtract(dilated, mask.astype(np.uint8) * 255)
        ring_mask = ring.astype(bool)
        mean_red = np.mean(image[ring_mask, 2])  #red channel
        return round(mean_red, 2)
    except Exception as e:
        if log_queue: log_queue.put(f"WARN: Margin Redness error: {e}")
        return None



def compute_wound_severity_score(metrics):
    try:
        score = 0
        score += min(metrics.get("Area_pixels2", 0) / 10000, 1.0) * 25
        score += min(metrics.get("TissuePercentages", {}).get("Necrotic", 0) / 50, 1.0) * 25
        score += max(1.0 - metrics.get("Circularity", 0), 0) * 20
        score += min(metrics.get("FractalDimension", 2.0) / 3.0, 1.0) * 15
        score += min(metrics.get("Sharpness", 0) / 1000, 1.0) * 15
        return round(score, 2)  #tot out of 100
    except Exception:
        return None



def run_inference_on_folder(image_folder, output_folder, predictor, dataset_metadata,
                            pixels_per_cm=None, log_queue=None, progress_queue=None):
    def local_log(msg):
        if log_queue:
            log_queue.put(msg)
        else:
            print(msg)

    os.makedirs(output_folder, exist_ok=True)
    subfolder_keys = ["histograms", "masked_regions", "dominant_colors", "pseudo_3d_plots", "tissue_estimates", "periwound_analysis", "composite_visualizations", "fractal_dimension_plots", "healing_simulations_erosion", "healing_simulations_distance"]
    folder_paths = {name: os.path.join(output_folder, name) for name in subfolder_keys}
    for path_val in folder_paths.values(): os.makedirs(path_val, exist_ok=True)

    metrics_file_path = os.path.join(output_folder, "wound_metrics.csv")
    all_metrics_data = []

    #csv header -- may need to revise if need more intermed. res. 
    with open(metrics_file_path, 'w') as f_metrics:
        header = ("ImageName,InstanceID,Area_pixels2,Perimeter_pixels,Circularity,Area_cm2,Perimeter_cm,"
                  "DominantColor1_BGR,Tissue_Granulation_%,Sharpness_LaplacianVar,FractalDimension_DBC,"
                  "WoundSeverityScore,MarginRednessIndex,MeanDepth_Laplacian\n")
        f_metrics.write(header)

    local_log("Searching for images in upload folder...")
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    image_files.sort()

    num_images = len(image_files)
    if num_images == 0:
        local_log("CRITICAL: No valid images (.png, .jpg, .jpeg, .bmp, .tif) found in the upload folder.")
        local_log("Analysis cannot continue.")
        if progress_queue: progress_queue.put(100)
        return

    local_log(f"Found {num_images} image(s) to process.")

    for img_idx, img_path in enumerate(image_files):
        if progress_queue: progress_queue.put(int(((img_idx) / num_images) * 100))
        
        img = cv2.imread(img_path)
        if img is None: 
            local_log(f"WARN: Could not read image {img_path}. Skipping.")
            continue
            
        base_name = os.path.basename(img_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        local_log(f"\nProcessing {img_idx+1}/{num_images}: {base_name}...")

        try:
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")
        except Exception as e:
            local_log(f"ERROR during prediction for {base_name}: {e}. Skipping.")
            continue

        v = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
        out_vis_initial = v.draw_instance_predictions(instances)
        result_img_vis = out_vis_initial.get_image()[:, :, ::-1].copy()

        if not instances.has("pred_masks") or len(instances.pred_masks) == 0:
            local_log(f"  No wound instances detected in {base_name}.")
            vis_out_path = os.path.join(output_folder, f"{base_name_no_ext}_prediction_no_detection.png")
            cv2.imwrite(vis_out_path, result_img_vis)
            continue

        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
        local_log(f"  Detected {len(masks)} instances.")

        for i, mask_instance in enumerate(masks):
            local_log(f"  -- Analyzing Instance {i} --")
            instance_data = {"ImageName": base_name, "InstanceID": i}
            mask_uint8 = mask_instance.astype(np.uint8) * 255
            
            instance_paths = {key: os.path.join(folder_paths[folder_key], f"{base_name_no_ext}_inst{i}_{key}.png") for key, folder_key in zip(subfolder_keys, subfolder_keys)}
            instance_paths['csv_metrics'] = metrics_file_path

            area_pixels = int(cv2.countNonZero(mask_uint8))
            instance_data["Area_pixels2"] = area_pixels
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                perimeter_pixels = float(cv2.arcLength(main_contour, True))
                instance_data["Perimeter_pixels"] = round(perimeter_pixels, 1)
                instance_data["Circularity"] = (4 * np.pi * area_pixels) / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
                if pixels_per_cm and pixels_per_cm > 0:
                    instance_data["Area_cm2"] = round(area_pixels / (pixels_per_cm ** 2), 2)
                    instance_data["Perimeter_cm"] = round(perimeter_pixels / pixels_per_cm, 2)
            
            masked_img_region = cv2.bitwise_and(img, img, mask=mask_uint8)
            cv2.imwrite(instance_paths['masked_regions'], masked_img_region)
            
            hist_path_for_composite = instance_paths['histograms'] if calculate_and_save_histogram(img, mask_uint8, instance_paths['histograms'], log_queue=log_queue) else None
            dominant_colors = get_dominant_colors(img, mask_instance, log_queue=log_queue)
            swatch_img = draw_color_swatches(dominant_colors)
            if swatch_img is not None: cv2.imwrite(instance_paths['dominant_colors'], swatch_img)
            
            tissue_percentages, tissue_vis_img = estimate_tissue_percentages(img, mask_instance, log_queue=log_queue)
            instance_data["TissuePercentages"] = tissue_percentages
            if tissue_vis_img is not None: cv2.imwrite(instance_paths['tissue_estimates'], tissue_vis_img)

            instance_data["Sharpness"] = calculate_sharpness(img, mask_instance, log_queue=log_queue)
            instance_data["FractalDimension"] = calculate_fractal_dimension_dbc(img, mask_instance, instance_paths['fractal_dimension_plots'], log_queue=log_queue)
            

            depth_data = estimate_wound_depth(img, mask_instance, log_queue=log_queue)
            if depth_data:
                instance_data["MeanDepth"] = depth_data["MeanDepth"]
                instance_data["StdDepth"] = depth_data["StdDepth"]

            redness = assess_margin_redness(img, mask_instance, log_queue=log_queue)
            instance_data["MarginRednessIndex"] = redness
            
            severity_score = compute_wound_severity_score(instance_data)
            instance_data["WoundSeverityScore"] = severity_score


            plot_pseudo_3d_wound(img, mask_instance, instance_paths['pseudo_3d_plots'], log_queue=log_queue)
            
            sim_stages_erosion = simulate_healing_stages_erosion(mask_uint8, log_queue=log_queue)
            if sim_stages_erosion: plot_healing_stages(img, sim_stages_erosion, instance_paths['healing_simulations_erosion'], base_name_no_ext, i, "Erosion", log_queue)
            sim_stages_dist = simulate_healing_stages_distance_transform(mask_uint8, log_queue=log_queue)
            if sim_stages_dist: plot_healing_stages(img, sim_stages_dist, instance_paths['healing_simulations_distance'], base_name_no_ext, i, "Distance Transform", log_queue)
            
            create_composite_visualization(img, mask_instance, instance_data, hist_path_for_composite, swatch_img, tissue_vis_img, instance_paths['composite_visualizations'], base_name_no_ext, i, pixels_per_cm, log_queue)
            
            if boxes is not None and i < len(boxes):
                x1, y1, _, _ = boxes[i].astype(int)

                text = f"Inst {i}: Area={instance_data.get('Area_pixels2', 'N/A')}px | Severity: {instance_data.get('WoundSeverityScore', 'N/A')}"
                cv2.putText(result_img_vis, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            all_metrics_data.append(instance_data)

            with open(metrics_file_path, 'a') as f:
                row = (f"{base_name},{i},{instance_data.get('Area_pixels2','')},{instance_data.get('Perimeter_pixels','')},"
                       f"{instance_data.get('Circularity',''):.3f},{instance_data.get('Area_cm2','')},{instance_data.get('Perimeter_cm','')},"
                       f"\"{dominant_colors[0] if dominant_colors else ''}\",{tissue_percentages.get('Granulation','')},"
                       f"{instance_data.get('Sharpness','')},{instance_data.get('FractalDimension','')},"
                       f"{instance_data.get('WoundSeverityScore', '')},{instance_data.get('MarginRednessIndex', '')},"
                       f"{instance_data.get('MeanDepth', '')}\n")
                f.write(row)

        vis_out_path = os.path.join(output_folder, f"{base_name_no_ext}_prediction_with_metrics.png")
        cv2.imwrite(vis_out_path, result_img_vis)

    if len(all_metrics_data) > 1:
        local_log("\nGenerating Area trend plot...")
        areas_inst0 = [d['Area_pixels2'] for d in all_metrics_data if d.get("InstanceID") == 0 and 'Area_pixels2' in d]
        names_inst0 = [d['ImageName'] for d in all_metrics_data if d.get("InstanceID") == 0 and 'Area_pixels2' in d]
        if len(areas_inst0) > 1:
            trend_plot_fig, ax_trend = plt.subplots(figsize=(10, 6))
            ax_trend.plot(range(len(areas_inst0)), areas_inst0, marker='o', linestyle='-')
            ax_trend.set_xticks(range(len(areas_inst0)))
            ax_trend.set_xticklabels([os.path.splitext(n)[0] for n in names_inst0], rotation=45, ha="right")
            ax_trend.set_xlabel("Image"); ax_trend.set_ylabel("Wound Area (pixels²)"); ax_trend.set_title("Wound Area Trend (Instance 0)")
            ax_trend.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "area_trend_plot_instance0.png"))
            plt.close(trend_plot_fig)
            
    if progress_queue: progress_queue.put(100)
