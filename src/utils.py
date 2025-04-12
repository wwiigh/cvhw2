from torchvision import transforms
import random


def random_resize(img):
    size = random.choice([320, 480, 512, 640])
    return transforms.Resize((size, size))(img)


transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
])


transform_val = transforms.Compose([
            transforms.ToTensor(),
])
