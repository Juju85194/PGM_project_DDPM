from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# define image transformations (e.g. using torchvision)
transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

def get_dataset(dataset_name, image_size):
    # load dataset from the hub
    dataset = load_dataset(dataset_name)

    # define function
    def transforms_fn(examples):
       examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
       del examples["image"]

       return examples

    transformed_dataset = dataset.with_transform(transforms_fn).remove_columns("label")
    return transformed_dataset

def get_dataloader(dataset_name, image_size, batch_size):
    transformed_dataset = get_dataset(dataset_name, image_size)
    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

    return dataloader