import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def classify_using_pytorch(img_path):  # chk if valid image path

    model = torchvision.models.squeezenet1_1(pretrained=False)
    model.load_state_dict(torch.load("model/squeezenet1_1.pth"))

    input_image = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    # Read the categories
    with open("utils/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        image_class = categories[top5_catid[i]]
        # print(categories[top5_catid[i]], top5_prob[i].item())
    return image_class


"""
# trying this function to make sure this works
img_path = "test_imgs/30.jpg"
img_class = classify_using_pytorch(img_path)
print(img_class)
"""
