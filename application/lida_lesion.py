import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


class MultiscaleLesionExtractor:
    def __init__(self, scales: Tuple[float, ...] = (1.0, 0.5), kernel_size: int = 5):
        self.scales = scales
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    def _extract_scale_lesion(self, lab_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid_pixels = lab_image[mask > 0]
        if len(valid_pixels) == 0:
            return np.zeros(mask.shape, dtype=np.uint8)

        ab_channels = np.float32(valid_pixels[:, 1:])

        _, labels, centers = cv2.kmeans(
            ab_channels, 2, None, self.kmeans_criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        disease_cluster_idx = int(np.argmax(centers[:, 0]))
        disease_center = centers[disease_cluster_idx]

        distances = np.linalg.norm(ab_channels - disease_center, axis=1)

        dist_map = np.zeros(mask.shape, dtype=np.float32)
        dist_map[mask > 0] = distances

        dist_map_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        valid_dist = dist_map_norm[mask > 0]

        _, binary_thresh = cv2.threshold(
            valid_dist, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        scale_mask = np.zeros(mask.shape, dtype=np.uint8)
        scale_mask[mask > 0] = binary_thresh.flatten()

        return scale_mask

    def process(self, image_rgb: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
        original_h, original_w = image_rgb.shape[:2]
        lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

        fused_lesion_mask = np.zeros((original_h, original_w), dtype=np.uint8)

        for scale in self.scales:
            curr_w = int(original_w * scale)
            curr_h = int(original_h * scale)

            if scale == 1.0:
                scaled_lab = lab_image
                scaled_mask = leaf_mask
            else:
                scaled_lab = cv2.resize(lab_image, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR)
                scaled_mask = cv2.resize(leaf_mask, (curr_w, curr_h), interpolation=cv2.INTER_NEAREST)

            scale_lesion = self._extract_scale_lesion(scaled_lab, scaled_mask)

            if scale != 1.0:
                scale_lesion = cv2.resize(scale_lesion, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            fused_lesion_mask = cv2.bitwise_or(fused_lesion_mask, scale_lesion)

        final_lesion_mask = cv2.morphologyEx(fused_lesion_mask, cv2.MORPH_CLOSE, self.kernel)

        return final_lesion_mask


class LIDAVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, image_rgb: np.ndarray, leaf_mask: np.ndarray, lesion_mask: np.ndarray, image_id: str):
        total_leaf_pixels = np.count_nonzero(leaf_mask)
        total_lesion_pixels = np.count_nonzero(lesion_mask)
        healthy_pixels = total_leaf_pixels - total_lesion_pixels

        if total_leaf_pixels > 0:
            lesion_ratio = (total_lesion_pixels / total_leaf_pixels) * 100
            healthy_ratio = 100.0 - lesion_ratio
        else:
            lesion_ratio = 0.0
            healthy_ratio = 0.0

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_rgb)
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim == 2:
                ax1.plot(cnt[:, 0], cnt[:, 1], color="lime", linewidth=2)
        ax1.set_title("Input Leaf Instances", fontsize=14)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(lesion_mask, cmap='gray')
        ax2.set_title("LIDA Extracted Lesion Mask", fontsize=14)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        overlay = image_rgb.copy()
        overlay[lesion_mask > 0] = [255, 0, 0]
        ax3.imshow(overlay)
        ax3.set_title("Disease Overlay Map", fontsize=14)
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[1, 0])
        if total_leaf_pixels > 0:
            ax4.pie(
                [healthy_ratio, lesion_ratio],
                labels=['Healthy Area', 'Lesion Area'],
                colors=['#2ca02c', '#d62728'],
                autopct='%1.2f%%',
                startangle=90,
                textprops={'fontsize': 12}
            )
        ax4.set_title("Lesion Area Proportion", fontsize=14)

        ax5 = fig.add_subplot(gs[1, 1:3])
        ax5.axis("off")
        table_data = [
            ["Metric", "Value"],
            ["Total Leaf Area (Pixels)", f"{total_leaf_pixels:,}"],
            ["Total Lesion Area (Pixels)", f"{total_lesion_pixels:,}"],
            ["Lesion Proportion (%)", f"{lesion_ratio:.2f}%"],
            ["Visual Assessment", "Stressed" if lesion_ratio > 5.0 else "Healthy"]
        ]
        table = ax5.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            bbox=[0.1, 0.1, 0.8, 0.8]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        ax5.set_title("Quantitative Pathological Metrics", fontsize=14)

        output_path = self.output_dir / f"LIDA_Analysis_{image_id}.png"
        plt.suptitle(f"Leaf Information and Disease Analysis (LIDA) - {image_id}", fontsize=18, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


class LIDAProcessor:
    def __init__(self, output_dir: str = "lida_reports"):
        self.lesion_extractor = MultiscaleLesionExtractor()
        self.visualizer = LIDAVisualizer(output_dir)

    def analyze_image(self, image_rgb: np.ndarray, instances_masks: List[np.ndarray], image_id: str) -> List[
        np.ndarray]:
        lesion_results = []
        combined_lesion = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        combined_leaf = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        for instance_mask in instances_masks:
            lesion_mask = self.lesion_extractor.process(image_rgb, instance_mask)
            lesion_results.append(lesion_mask)

            combined_lesion = cv2.bitwise_or(combined_lesion, lesion_mask)
            combined_leaf = cv2.bitwise_or(combined_leaf, instance_mask)

        self.visualizer.generate_report(image_rgb, combined_leaf, combined_lesion, image_id)
        return lesion_results


if __name__ == "__main__":
    dummy_image = np.full((600, 800, 3), 100, dtype=np.uint8)
    dummy_image[200:400, 300:500] = [120, 150, 50]
    dummy_image[250:300, 350:400] = [200, 100, 50]

    dummy_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.circle(dummy_mask, (400, 300), 100, 255, -1)

    lida_processor = LIDAProcessor(output_dir="./lida_output")
    lida_processor.analyze_image(dummy_image, [dummy_mask], "sample_test_001")