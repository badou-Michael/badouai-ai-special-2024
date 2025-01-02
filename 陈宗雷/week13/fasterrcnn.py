# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Name: fasterrcnn
# Time:  09:19
# Author: chungrae
# Description:
from pathlib import Path
from typing import MutableMapping, List

import torch
import torchvision
from PIL import ImageDraw, Image, ImageFile
from torch import Tensor
from torchvision import transforms

from torchvision.models.detection import fasterrcnn_resnet50_fpn

class FasterRCNN:
    def __init__(self, img_path: Path):
        self.img_path = img_path
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        model.to(self.device)
        self.model = model


    @staticmethod
    def process_image(image: ImageFile) -> Tensor:
        transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)



    def predict(self, image: ImageFile) -> List[MutableMapping[str, Tensor]]:

        image = image.convert("RGB")
        image_tensor = self.process_image(image).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            return outputs

    @staticmethod
    def show_results(image: ImageFile, predictions: List[MutableMapping[str, Tensor]]) -> None:
        boxes = predictions[0]["boxes"].cpu().numpy()
        labels = predictions[0]["labels"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()
        draw = ImageDraw.Draw(image)
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                top, left, bottom, right = box
                draw.rectangle((left, top, right, bottom), outline="red", width=2)
                draw.text((top, left - 10), str(label), fill="red")
        image.show()


    def run(self):
        image = Image.open(self.img_path)
        prediction = self.predict(image)
        self.show_results(image, prediction)
        image.show()


if __name__ == "__main__":
    FasterRCNN(Path("./street.jpg")).run()
