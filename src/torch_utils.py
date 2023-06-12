import torch
import time
import matplotlib.pyplot as plt
from torchvision import transforms
plt.style.use('ggplot')

def load_efficientnet_model():
    """
    Load the pre-trained EfficientNetB0 model.
    """
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model

def load_resnet50_model():
    """
    Load the pre-trained ResNet50 model.
    """
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def preprocess():
    """
    Define the transform for the input image/frames.
    Resize, crop, convert to tensor, and apply ImageNet normalization stats.
    """
    transform =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),])
    return transform
def read_classes():
    """
    Load the ImageNet class names.
    """
    with open("../data/rotation_prediction_test/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def run_through_model(model, input_batch, n_runs):
    """
    Forward passes the image tensor batch through the model
    `n_runs` number if times. Prints the average milliseconds taken
    and returns a list containing all the forward pass times.
    """
    total_time = 0
    time_list = []
    for i in range(n_runs):
        print(f"Run: {i+1}", end='\r')
        with torch.no_grad():
            start_time = time.time()
            output = model(input_batch)
            end_time = time.time()
            time_list.append((end_time-start_time)*1000)
            total_time += (end_time - start_time)*1000
    print(f"{total_time/n_runs:.3f} milliseconds\n")
    return time_list
def plot_time_vs_iter(model_names, time_lists, device):
    """
    Plots the iteration vs time graph for given model.
    :param model_name: List of strings, name of both the models.
    :param time_list: List of lists, containing time take for each iteration
        for each model.
    :param device: Computation device.
    """
    colors = ['green', 'red']
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(model_names):
        plt.plot(
            time_lists[i], color=colors[i], linestyle='-',
            label=f"time taken (ms) {name}"
        )
    plt.xlabel('Iterations')
    plt.ylabel('Time Taken (ms)')
    plt.legend()
    plt.savefig(f"outputs/time_vs_iterations_{device}.png")
    plt.show()
    plt.close()