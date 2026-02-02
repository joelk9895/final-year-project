import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os

class YOLOInference:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load model
        try:
            self.session = ort.InferenceSession(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
            
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        
        # Handle dynamic shapes (which might be strings like 'height', 'width')
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        if not isinstance(self.input_height, int):
            self.input_height = 640
        if not isinstance(self.input_width, int):
            self.input_width = 640
            
        print(f"Model Input: {self.input_names}, Shape: {self.input_shape} (Using {self.input_width}x{self.input_height})")

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        print(f"Model Output: {self.output_names}")

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]
        
        # Resize
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        
        # Scale 0-255 to 0-1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        
        return input_tensor

    def postprocess(self, outputs):
        predictions = np.squeeze(outputs[0]).T
        
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thres, :]
        scores = scores[scores > self.conf_thres]
        
        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        boxes = predictions[:, :4]
        
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        
        # Center x, Center y, w, h -> x1, y1, x2, y2
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        
        return boxes[indices], scores[indices], class_ids[indices]

    def run_inference(self, image):
        input_tensor = self.preprocess(image)
        
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        
        boxes, scores, class_ids = self.postprocess(outputs)
        return boxes, scores, class_ids

    def inference(self, img_path):
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return

        image = cv2.imread(img_path)
        boxes, scores, class_ids = self.run_inference(image)
        
        for (box, score, class_id) in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = f"Class {class_id}: {score:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        output_path = "output_" + os.path.basename(img_path)
        cv2.imwrite(output_path, image)
        print(f"Inference completed. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="license-plate-finetune-v1l-int8-dynamic.onnx", help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    yolo = YOLOInference(args.model)
    yolo.inference(args.image)
