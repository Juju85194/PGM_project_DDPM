import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import torch
import matplotlib.animation as animation

# Image plotting
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

def save_gif(save_path, samples, image_size, channels):
    fig = plt.figure()
    ims = []
    for i in range(len(samples)):
        im = plt.imshow(samples[i][0].reshape(image_size, image_size, channels), cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save(save_path)
    plt.show()

def get_reverse_transform(image_size):
    reverse_transform = Compose([
         Lambda(lambda t: (t + 1) / 2),
         Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
         Lambda(lambda t: t * 255.),
         Lambda(lambda t: t.numpy().astype(np.uint8)),
         ToPILImage(),
    ])
    return reverse_transform

def get_transform(image_size):
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
        Lambda(lambda t: (t * 2) - 1),

    ])
    return transform