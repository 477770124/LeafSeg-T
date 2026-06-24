import cv2
import numpy as np
import pandas as pd
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any


class PhenotypeFeatureExtractor:
    def __init__(self, reference_thickness: float = 1.0):
        self.href = reference_thickness

    def extract_geometry(self, contour: np.ndarray) -> Dict[str, float]:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)

        circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        compactness = area / (w * h) if (w * h) > 0 else 0.0

        equivalent_volume = area * compactness * 1.2
        thickness_index = equivalent_volume / (area * self.href) if area > 0 else 0.0

        complex_contour = np.empty(contour.shape[0], dtype=complex)
        complex_contour.real = contour[:, 0, 0]
        complex_contour.imag = contour[:, 0, 1]
        fourier_result = np.fft.fft(complex_contour)
        fourier_descriptors = np.abs(fourier_result)
        countor_variation = float(np.std(fourier_descriptors[1:15])) if len(fourier_descriptors) > 15 else 0.0

        return {
            "area": area,
            "perimeter": perimeter,
            "bbox_width": float(w),
            "bbox_height": float(h),
            "bbox_x": float(x + w / 2),
            "bbox_y": float(y + h / 2),
            "curliness": 1.0 - circularity,
            "curvature": circularity,
            "thickness": thickness_index,
            "Countor Variation": countor_variation
        }

    def extract_spectral(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        mean_val = cv2.mean(image, mask=mask)[:3]
        return {
            "mean_r": float(mean_val[0]),
            "mean_g": float(mean_val[1]),
            "mean_b": float(mean_val[2])
        }


class LDSICalculator:
    def __init__(self):
        self.entropy_weights = {
            "mean_b": 0.098,
            "bbox_width": 0.095,
            "bbox_height": 0.095,
            "bbox_x": 0.092,
            "bbox_y": 0.092,
            "area": 0.090,
            "mean_r": 0.087,
            "curliness": 0.086,
            "perimeter": 0.082,
            "mean_g": 0.080,
            "thickness": 0.056,
            "curvature": 0.047
        }

    def compute_ldsi(self, features: Dict[str, float], normalization_bounds: Dict[str, Tuple[float, float]]) -> float:
        ldsi_score = 0.0
        for feature_name, weight in self.entropy_weights.items():
            val = features.get(feature_name, 0.0)
            bounds = normalization_bounds.get(feature_name, (0.0, 1.0))
            min_val, max_val = bounds

            if max_val > min_val:
                norm_val = (val - min_val) / (max_val - min_val)
            else:
                norm_val = 0.0

            ldsi_score += weight * norm_val

        return float(np.clip(ldsi_score, 0.0, 1.0))


class LeafAnalysisPipeline:
    def __init__(self, json_dir: str, image_dir: str):
        self.json_dir = Path(json_dir)
        self.image_dir = Path(image_dir)
        self.feature_extractor = PhenotypeFeatureExtractor()
        self.ldsi_calculator = LDSICalculator()

    def process_dataset(self) -> pd.DataFrame:
        dataset_results = []

        for json_path in self.json_dir.glob('*.json'):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_name = data.get('imagePath', json_path.stem + '.png')
            image_path = self.image_dir / image_name

            if not image_path.exists():
                continue

            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            for idx, shape in enumerate(data.get('shapes', [])):
                if shape['shape_type'] != 'polygon':
                    continue

                points = np.array(shape['points'], dtype=np.int32)
                contour = points.reshape(-1, 1, 2)

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)

                geom_features = self.feature_extractor.extract_geometry(contour)
                spec_features = self.feature_extractor.extract_spectral(image_rgb, mask)

                combined_features = {**geom_features, **spec_features}
                combined_features['leaf_id'] = f"{json_path.stem}_{idx}"
                dataset_results.append(combined_features)

        df = pd.DataFrame(dataset_results)

        if not df.empty:
            norm_bounds = {col: (df[col].min(), df[col].max()) for col in self.ldsi_calculator.entropy_weights.keys() if
                           col in df.columns}
            df['LDSI'] = df.apply(lambda row: self.ldsi_calculator.compute_ldsi(row.to_dict(), norm_bounds), axis=1)

        return df


if __name__ == "__main__":
    pipeline = LeafAnalysisPipeline(json_dir="data/annotations", image_dir="data/images")
    results_df = pipeline.process_dataset()
    results_df.to_csv("phenotype_ldsi_results.csv", index=False)