"""MMDetection panel inference on one prepared RGB image.

The caller provides the exact config and checkpoint used during training.
This adapter does not tile large images or merge overlapping detections.
"""
from pathlib import Path
from modules.panel.contracts import PanelDetections


def prepare_inference_config(config):
    """Keep test transforms, but do not request ground-truth annotations."""
    if config.get('test_pipeline') is not None:
        config.test_pipeline = [step for step in config.test_pipeline if step.get('type') != 'LoadAnnotations']
    loader = config.get('test_dataloader')
    if loader and loader.get('dataset') and loader.dataset.get('pipeline') is not None:
        loader.dataset.pipeline = [step for step in loader.dataset.pipeline if step.get('type') != 'LoadAnnotations']
    return config


class RTMDetPanelModel:
    def __init__(self, config_path: str, checkpoint_path: str, *, device: str = 'cpu', threshold: float = 0.25):
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be in [0, 1]')
        config_file, checkpoint_file = Path(config_path), Path(checkpoint_path)
        if not config_file.is_file() or not checkpoint_file.is_file():
            raise FileNotFoundError('panel model config or checkpoint is missing')
        try:
            import torch
            from mmengine import Config
            from mmdet.apis import init_detector, inference_detector
        except ImportError as exc:
            raise RuntimeError('PyTorch, MMEngine and MMDetection are required') from exc
        config = prepare_inference_config(Config.fromfile(str(config_file)))
        config.model.backbone.init_cfg = None
        config.model.test_cfg.score_thr = min(0.05, threshold)
        config.model.test_cfg.max_per_img = 100
        self.model = init_detector(config, checkpoint=None, palette='random', device=device)
        checkpoint = torch.load(str(checkpoint_file), map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        if 'metainfo' in config and 'classes' in config.metainfo:
            self.model.dataset_meta = {**getattr(self.model, 'dataset_meta', {}), 'classes': tuple(config.metainfo['classes'])}
        self.inference_detector = inference_detector
        self.threshold = threshold
        self.version = checkpoint_file.name

    def predict(self, image) -> PanelDetections:
        import numpy as np
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('panel input must be an H x W x 3 uint8 RGB array')
        instances = self.inference_detector(self.model, image[:, :, ::-1].copy()).pred_instances
        scores = instances.scores.detach().cpu().numpy()
        keep = scores >= self.threshold
        if not hasattr(instances, 'masks'):
            raise RuntimeError('panel model did not return masks')
        masks = instances.masks.detach().cpu().numpy()[keep].astype(bool)
        if masks.shape[1:] != image.shape[:2]:
            raise RuntimeError('panel masks do not match input pixel grid')
        return PanelDetections(masks=masks, model_version=self.version, scores=scores[keep],
                               labels=instances.labels.detach().cpu().numpy()[keep],
                               boxes_xyxy=instances.bboxes.detach().cpu().numpy()[keep])
