import augmentations
import matplotlib.pyplot as plt
import tensorflow as tf


def visualize_augmentations(image_index, num_images):
    images = (
        tf.stack(
            [tf.keras.datasets.cifar10.load_data()[0][0][image_index]] * num_images,
            axis=0,
        )
        / 255
    )
    augmented_images = zip(
        images,
        augmentations.RandomGaussianNoise(stddev=0.03)(images),
        augmentations.RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3))(images),
        augmentations.RandomColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
        )(images),
        augmentations.RandomColorAffine(brightness=0.3, jitter=0.1)(images),
    )

    row_titles = [
        "Original:",
        "RandomGaussianNoise:",
        "RandomResizedCrop:",
        "RandomColorJitter:",
        "RandomColorAffine:",
    ]
    plt.figure(figsize=(num_images * 2.5, len(row_titles) * 2.5), dpi=100)
    for column, image_row in enumerate(augmented_images):
        for row, image in enumerate(image_row):
            plt.subplot(len(row_titles), num_images, row * num_images + column + 1)
            plt.imshow(image)
            if column == 0:
                plt.title(row_titles[row], loc="left")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"augmentations_{image_index}.png")


if __name__ == "__main__":
    visualize_augmentations(image_index=13, num_images=8)
    visualize_augmentations(image_index=30, num_images=8)
    visualize_augmentations(image_index=66, num_images=8)
