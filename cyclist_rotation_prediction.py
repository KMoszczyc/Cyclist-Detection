import torch
import cv2
import argparse
import time
from torch_utils import (
    load_efficientnet_model, preprocess, read_classes
)



# Construct the argumet parser to parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/image_1.jpg',
                    help='path to the input image')
parser.add_argument('-d', '--device', default='cpu',
                    help='computation device to use',
                    choices=['cpu', 'cuda'])
args = vars(parser.parse_args())

# Set the computation device.
DEVICE = 'cpu'
# Initialize the model.
model = load_efficientnet_model()
# Load the ImageNet class names.
categories = read_classes()
# Initialize the image transforms.
transform = preprocess()
print(f"Computation device: {DEVICE}")

image_path = 'data/rotation_prediction_test/car.jpg'
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Apply transforms to the input image.
input_tensor = transform(image)
# Add the batch dimension.
input_batch = input_tensor.unsqueeze(0)
# Move the input tensor and model to the computation device.
input_batch = input_batch.to(DEVICE)
model.to(DEVICE)

with torch.no_grad():
    start_time = time.time()
    output = model(input_batch)
    end_time = time.time()
# Get the softmax probabilities.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# Check the top 5 categories that are predicted.
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    cv2.putText(image, f"{top5_prob[i].item()*100:.3f}%", (15, (i+1)*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{categories[top5_catid[i]]}", (160, (i+1)*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    print(categories[top5_catid[i]], top5_prob[i].item())
cv2.imshow('Result', image)
cv2.waitKey(0)
# Define the outfile file name.
save_name = f"outputs/{args['input'].split('/')[-1].split('.')[0]}_{DEVICE}.jpg"
cv2.imwrite(save_name, image)
print(f"Forward pass time: {(end_time-start_time):.3f} seconds")