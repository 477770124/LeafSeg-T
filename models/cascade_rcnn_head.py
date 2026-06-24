import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Dict


class CascadeMaskRCNNHead(nn.Module):
    def __init__(self,
                 in_channels: int = 256,
                 num_classes: int = 80,
                 num_stages: int = 3,
                 roi_size: int = 7,
                 mask_roi_size: int = 14,
                 box_dim: int = 4):
        super().__init__()

        self.num_stages = num_stages
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.mask_roi_size = mask_roi_size

        self.iou_thresholds = [0.5, 0.6, 0.7]

        self.loss_weights = {
            "lambda_cls": 1.0,
            "lambda_box": 1.0,
            "lambda_mask": 1.0
        }

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = CascadeStage(
                in_channels=in_channels,
                num_classes=num_classes,
                roi_size=roi_size,
                mask_roi_size=mask_roi_size,
                box_dim=box_dim,
                stage_idx=i,
                iou_thresh=self.iou_thresholds[i]
            )
            self.stages.append(stage)

    def forward(self, features: Tensor, proposals: List[Tensor],
                image_sizes: List[Tuple[int, int]]) -> Tuple[List[Dict], Dict]:
        all_predictions = []
        all_losses = {}

        current_proposals = proposals
        for stage_idx, stage in enumerate(self.stages):
            stage_predictions, stage_losses = stage(
                features, current_proposals, image_sizes
            )

            all_predictions.append(stage_predictions)

            for key, value in stage_losses.items():
                weight = self.loss_weights.get(f"lambda_{key.split('_')[1]}", 1.0)
                all_losses[f"stage{stage_idx}_{key}"] = value * weight

            if stage_idx < self.num_stages - 1:
                current_proposals = self._get_proposals_for_next_stage(
                    stage_predictions, current_proposals
                )

        return all_predictions, all_losses

    def _get_proposals_for_next_stage(self, predictions: List[Dict],
                                      current_proposals: List[Tensor]) -> List[Tensor]:
        next_proposals = []
        for pred, proposals in zip(predictions, current_proposals):
            if "pred_boxes" in pred and "scores" in pred:
                boxes = pred["pred_boxes"]
                scores = pred["scores"]

                if scores.dim() == 2 and scores.shape[1] > 1:
                    max_scores, _ = torch.max(scores[:, 1:], dim=1)
                    keep = max_scores > 0.05
                else:
                    keep = scores > 0.05

                if keep.any() and boxes.numel() > 0:
                    if boxes.dim() == 3:
                        _, cls_idx = torch.max(scores[:, 1:], dim=1)
                        selected_boxes = boxes[torch.arange(len(boxes)), cls_idx]
                        next_proposals.append(selected_boxes[keep])
                    elif boxes.dim() == 2:
                        next_proposals.append(boxes[keep])
                    else:
                        next_proposals.append(proposals)
                else:
                    next_proposals.append(proposals)
            else:
                next_proposals.append(proposals)

        return next_proposals


class CascadeStage(nn.Module):
    def __init__(self, in_channels: int, num_classes: int,
                 roi_size: int, mask_roi_size: int, box_dim: int,
                 stage_idx: int, iou_thresh: float):
        super().__init__()

        self.stage_idx = stage_idx
        self.roi_size = roi_size
        self.mask_roi_size = mask_roi_size
        self.iou_thresh = iou_thresh

        fc_in_features = in_channels * roi_size * roi_size
        self.fc1 = nn.Linear(fc_in_features, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * box_dim)

        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: Tensor, proposals: List[Tensor],
                image_sizes: List[Tuple[int, int]]) -> Tuple[List[Dict], Dict]:

        proposal_counts = [len(p) for p in proposals]
        total_proposals = sum(proposal_counts)
        device = features.device

        roi_features = torch.randn(total_proposals, features.shape[1],
                                   self.roi_size, self.roi_size, device=device)

        mask_roi_features = torch.randn(total_proposals, features.shape[1],
                                        self.mask_roi_size, self.mask_roi_size, device=device)

        x = roi_features.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        mask_logits = self.mask_head(mask_roi_features)

        predictions = []
        start_idx = 0
        for i, prop_count in enumerate(proposal_counts):
            end_idx = start_idx + prop_count
            if prop_count > 0:
                pred = {
                    "pred_classes": cls_logits[start_idx:end_idx],
                    "pred_boxes": bbox_deltas[start_idx:end_idx].view(prop_count, self.cls_score.out_features, 4),
                    "pred_masks": mask_logits[start_idx:end_idx],
                    "scores": F.softmax(cls_logits[start_idx:end_idx], dim=-1)
                }
            else:
                pred = {
                    "pred_classes": torch.empty(0, self.cls_score.out_features, device=device),
                    "pred_boxes": torch.empty(0, self.cls_score.out_features, 4, device=device),
                    "pred_masks": torch.empty(0, self.cls_score.out_features, self.mask_roi_size * 2,
                                              self.mask_roi_size * 2, device=device),
                    "scores": torch.empty(0, self.cls_score.out_features, device=device)
                }
            predictions.append(pred)
            start_idx = end_idx

        losses = {
            "loss_cls": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_box": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_mask": torch.tensor(0.0, device=device, requires_grad=True)
        }

        return predictions, losses


if __name__ == "__main__":
    batch_size = 2
    in_channels = 256
    num_classes = 2

    cascade_head = CascadeMaskRCNNHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_stages=3
    )

    features = torch.randn(batch_size, in_channels, 32, 32)
    proposals = [
        torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32),
        torch.tensor([[15, 15, 55, 55]], dtype=torch.float32)
    ]
    image_sizes = [(100, 100), (100, 100)]

    predictions, losses = cascade_head(features, proposals, image_sizes)

    for i, stage_preds in enumerate(predictions):
        for j, pred in enumerate(stage_preds[:2]):
            pass