from torch import cuda
import numpy as np
from enum import Enum
import yaml
import cv2
from threading import RLock

class SegmentMode(Enum):
    POINT = 0
    BBOX = 1



class AirbotSegment():
    def __init__(self) -> None:
        with open("configs/config_file.yaml", "r") as file:
            config_path = yaml.safe_load(file)["Path"]
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        self.model_name = config["AirbotSegment"]["model_name"] 
        model_type = config["AirbotSegment"][self.model_name]["model_type"]
        checkpoint = config["AirbotSegment"][self.model_name]["checkpoint"][model_type]
        self.device = config["AirbotSegment"]["device"]
        if not cuda.is_available():
            print("\33[31m Cuda not available, set SegmentAnything device to cpu! \33[0m")
            self.device = "cpu"
        
        if self.model_name == "SAM":
            from segment_anything_fast import sam_model_registry, SamPredictor
            model = sam_model_registry[model_type](checkpoint=checkpoint)
            model.to(self.device)
            self.model = SamPredictor(model)
        elif self.model_name == "MobileSAM":
            import warnings

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            from mobile_sam import sam_model_registry, SamPredictor

            model = sam_model_registry[model_type](checkpoint=checkpoint)
            model.to(self.device)
            model.eval()
            self.model = SamPredictor(model)
            
        elif self.model_name == "FastSAM":
            from ultralytics import FastSAM
            self.model = FastSAM(checkpoint)
        
        self.mode = SegmentMode.POINT
        self.input_points = []
        self.input_labels = []
        self.bboxs = []
        self.multi_output = False
        self._lock = RLock()

    def add_point(self, x: int, y: int, is_positive: bool):
        with self._lock:
            if is_positive:
                self.input_points.append([x, y])
                self.input_labels.append(1)
            else:
                self.input_points.append([x, y])
                self.input_labels.append(0)
    
    def add_bbox(self, bbox):
        with self._lock:
            self.bboxs.append(bbox)
    
    def clear_prompt(self):
        with self._lock:
            if self.mode == SegmentMode.BBOX:
                self.bboxs = []
            elif self.mode == SegmentMode.POINT:
                self.input_points = []
                self.input_labels = []

    def inference_bbox(self, image, bbox):
        """Atomically infer one box without changing another workflow's prompt."""
        with self._lock:
            previous_mode = self.mode
            try:
                self.mode = SegmentMode.BBOX
                self.bboxs = [bbox]
                return self.inference(image)
            finally:
                self.bboxs = []
                self.mode = previous_mode

    def inference(self, image):
        with self._lock:
            return self._inference_locked(image)

    def _inference_locked(self, image):
        if (self.mode == SegmentMode.BBOX and not self.bboxs) or \
           (self.mode == SegmentMode.POINT and
            (not self.input_points or not self.input_labels)):
            return None
        
        mask = None
        
        # 基于不同模型和分割模式的处理逻辑
        if self.model_name == "SAM" or self.model_name == "MobileSAM":
            self.model.set_image(image)
            if self.mode == SegmentMode.BBOX:
                masks, _, _ = self.model.predict(
                    box=np.array(self.bboxs),
                    multimask_output=self.multi_output,
                )
                mask = masks[0]
            elif self.mode == SegmentMode.POINT:
                masks, _, _ = self.model.predict(
                    point_coords=np.array(self.input_points),
                    point_labels=np.array(self.input_labels),
                    multimask_output=self.multi_output,
                )
                mask = masks[0]
        
        elif self.model_name == "FastSAM":
            predict_params = {"source": image, "device": self.device, "imgsz": 1280}
            
            if self.mode == SegmentMode.BBOX:
                predict_params["bboxes"] = self.bboxs
            elif self.mode == SegmentMode.POINT:
                predict_params["points"] = self.input_points
                predict_params["labels"] = self.input_labels
                
            result = self.model.predict(**predict_params)
            
            if result and len(result) > 0:
                raw_masks = result[0].masks.data.cpu().numpy()
                scores = result[0].boxes.conf.cpu().numpy()
                
                if len(raw_masks) > 0:
                    max_score_idx = np.argmax(scores)
                    raw_mask = raw_masks[max_score_idx]
                    
                    mask_resized = cv2.resize(raw_mask, (image.shape[1], image.shape[0]))
                    mask = mask_resized.astype(bool)
        
        # BBOX模式下清除提示
        if self.mode == SegmentMode.BBOX:
            self.clear_prompt()
            
        return mask
