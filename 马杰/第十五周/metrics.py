import numpy as np
from pycocotools.cocoeval import COCOeval

class COCOEvaluator:
    def __init__(self, coco_gt):
        self.coco_gt = coco_gt
        self.results = []
        
    def update(self, predictions):
        """更新预测结果"""
        for image_id, boxes, scores, labels, masks in predictions:
            if boxes.numel() == 0:
                continue
            # 确保数据类型转换
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy()
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                result = {
                    'image_id': image_id,
                    'category_id': label,
                    'bbox': box.tolist(),
                    'score': score.item(),
                    'segmentation': self._encode_mask(mask)
                }
                self.results.append(result)
    
    def evaluate(self):
        """计算评估指标"""
        if not self.results:
            return {
                'AP': 0, 'AP50': 0, 'AP75': 0,
                'APs': 0, 'APm': 0, 'APl': 0,
                'mask_AP': 0, 'mask_AP50': 0, 'mask_AP75': 0
            }
            
        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 获取结果
        stats = {
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
            'APs': coco_eval.stats[3],
            'APm': coco_eval.stats[4],
            'APl': coco_eval.stats[5]
        }
        
        # Mask评估
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        stats.update({
            'mask_AP': coco_eval.stats[0],
            'mask_AP50': coco_eval.stats[1],
            'mask_AP75': coco_eval.stats[2]
        })
        
        return stats
        
    def _encode_mask(self, mask):
        """将mask编码为COCO格式"""
        from pycocotools import mask as maskUtils
        mask = np.asfortranarray(mask)
        rle = maskUtils.encode(mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle 