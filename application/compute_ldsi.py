import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


class LDSIVisualizer:
    def __init__(self, json_path: str, output_dir: str):
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cmap = cm.get_cmap('RdYlGn_r')
        self.weights = {
            'mean_b': 0.098,
            'bbox_width': 0.095,
            'bbox_height': 0.095,
            'bbox_x': 0.092,
            'bbox_y': 0.092,
            'area': 0.090,
            'mean_r': 0.087,
            'curliness': 0.086,
            'perimeter': 0.082,
            'mean_g': 0.080,
            'thickness': 0.056,
            'curvature': 0.047
        }
        self.load_data()

    def load_data(self) -> None:
        if self.json_path.exists():
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def compute_all_ldsi(self) -> None:
        instances = self.data.get('leaf_instances', [])
        if not instances:
            return

        feature_arrays = {k: [] for k in self.weights.keys()}

        for inst in instances:
            for k in self.weights.keys():
                feature_arrays[k].append(inst.get(k, 0.0))

        mins = {k: np.min(v) for k, v in feature_arrays.items()}
        maxs = {k: np.max(v) for k, v in feature_arrays.items()}

        for inst in instances:
            ldsi = 0.0
            for k, w in self.weights.items():
                val = inst.get(k, 0.0)
                denom = maxs[k] - mins[k]
                norm_val = (val - mins[k]) / denom if denom > 0 else 0.0
                ldsi += w * norm_val

            inst['ldsi'] = float(np.clip(ldsi, 0.0, 1.0))
            _ = inst.get('Countor Variation', 0.0)

    def create_figure_17_visualization(self, image_rgb: np.ndarray) -> str:
        instances = self.data.get('leaf_instances', [])
        if not instances:
            return ""

        self.compute_all_ldsi()
        ldsi_values = [inst['ldsi'] for inst in instances]

        fig = plt.figure(figsize=(20, 5))
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.3, 1.5], wspace=0.3)

        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.imshow(image_rgb)
        ax_a.axis('off')

        ax_b = fig.add_subplot(gs[0, 1])
        mask_img = np.ones_like(image_rgb) * 255
        for inst in instances:
            if 'polygon' in inst:
                pts = np.array(inst['polygon'], dtype=np.int32).reshape((-1, 1, 2))
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.fillPoly(mask_img, [pts], color)
                cv2.polylines(mask_img, [pts], True, (0, 0, 0), 1)
        ax_b.imshow(mask_img)
        ax_b.axis('off')

        ax_c = fig.add_subplot(gs[0, 2])
        ldsi_img = np.ones_like(image_rgb) * 255
        for inst in instances:
            if 'polygon' in inst:
                pts = np.array(inst['polygon'], dtype=np.int32).reshape((-1, 1, 2))
                color = self.cmap(inst['ldsi'])[:3]
                color_rgb = tuple(int(c * 255) for c in color)
                cv2.fillPoly(ldsi_img, [pts], color_rgb)
        ax_c.imshow(ldsi_img)
        ax_c.axis('off')

        ax_d = fig.add_subplot(gs[0, 3])
        gradient = np.linspace(1, 0, 256).reshape(-1, 1)
        gradient = np.hstack((gradient, gradient))
        ax_d.imshow(gradient, aspect='auto', cmap=self.cmap)
        ax_d.set_xticks([])
        ax_d.set_yticks([0, 128, 255])
        ax_d.set_yticklabels(['High Stress', 'Medium', 'Healthy'])
        ax_d.yaxis.tick_right()

        ax_e = fig.add_subplot(gs[0, 4])
        sorted_ldsi = np.sort(ldsi_values)
        x_idx = np.arange(len(sorted_ldsi))
        ax_e.scatter(x_idx, sorted_ldsi, c=sorted_ldsi, cmap=self.cmap, edgecolors='black', s=40)
        ax_e.set_ylim(-0.05, 1.05)
        ax_e.set_xlim(-1, len(sorted_ldsi))
        ax_e.set_ylabel("LDSI Score")
        ax_e.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        output_path = self.output_dir / f"Figure17_LDSI_Visualization_{self.json_path.stem}.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)


if __name__ == "__main__":
    json_input_path = "data_numbers/train/1_complete_ldsi.json"
    output_directory = "data_numbers/output"

    dummy_img = np.full((500, 530, 3), 255, dtype=np.uint8)
    visualizer = LDSIVisualizer(json_input_path, output_directory)
    visualizer.create_figure_17_visualization(dummy_img)