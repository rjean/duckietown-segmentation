from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import GenericMask, _create_text_labels
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import BoxMode

import numpy as np

class DuckieVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)
    #    Visualizer.__init__(self, img_rgb, metadata, scale, instance_mode)
    def toto(self):
        return True
    
    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.
        """
        #boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        boxes=None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        #labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        labels=None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        colors = []
        alpha=0.8
        for c in classes:
            color=(0,1,1,1) #Green
            if c==0:
                color= (1,1,0,1) #Yellow Line
            if c==1:
                color= (0.8,0.8,1,1) #White Line
            if c==4:
                color= (1,0,0,1) #Red line
            if c==2:
                color= (0,1,0,1) #Obstacle
            colors.append(color)
        
        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
    
    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]

            labels = [x["category_id"] for x in annos]
            colors = []
            alpha=0.8
            for c in labels:
                color=(0,1,1,1) #Green
                if c==0:
                    color= (1,1,0,1) #Yellow Line
                if c==1:
                    color= (0.8,0.8,1,1) #White Line
                if c==4:
                    color= (1,0,0,1) #Red line
                if c==2:
                    color= (0,1,0,1) #Obstacle
                colors.append(color)
            names = self.metadata.get("thing_classes", None)
            #if names:
            #    labels = [names[i] for i in labels]
            #labels = [
            #    "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
            #    for i, a in zip(labels, annos)
            #]
            labels=None
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype="uint8")
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)

        pan_seg = dic.get("pan_seg", None)
        if pan_seg is None and "pan_seg_file_name" in dic:
            assert "segments_info" in dic
            with PathManager.open(dic["pan_seg_file_name"], "rb") as f:
                pan_seg = Image.open(f)
                pan_seg = np.asarray(pan_seg)
                from panopticapi.utils import rgb2id

                pan_seg = rgb2id(pan_seg)
            segments_info = dic["segments_info"]
        if pan_seg is not None:
            pan_seg = torch.Tensor(pan_seg)
            self.draw_panoptic_seg_predictions(pan_seg, segments_info, area_threshold=0, alpha=0.5)
        return self.output