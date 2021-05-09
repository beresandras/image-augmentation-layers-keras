import random
import math
import tensorflow as tf

from tensorflow.keras import layers


# applies additive normal-distributed pixel noise to the image
class RandomGaussianNoise(layers.Layer):
    def __init__(self, stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, images, training=True):
        if training:
            images = tf.clip_by_value(
                images + tf.random.normal(tf.shape(images), stddev=self.stddev), 0, 1
            )
        return images


# the implementation of RandomResizedCrop and RandomColorJitter follow the torchvision library:
# https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
# https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
# however these augmentations:
# -run on batches of images
# -run on gpu
# -can be part of a model

# crops and resizes part of the image to the original resolutions
class RandomResizedCrop(layers.Layer):
    def __init__(self, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3), **kwargs):
        super().__init__(**kwargs)
        # area-range of the cropped part: (min area, max area), uniform sampling
        self.scale = scale
        # aspect-ratio-range of the cropped part: (log min ratio, log max ratio), log-uniform sampling
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # independently sampled scales and ratios for every image in the batch
            random_scales = tf.random.uniform(
                (batch_size,), self.scale[0], self.scale[1]
            )
            random_ratios = tf.exp(
                tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
            )

            # corresponding height and widths, clipped to fit in the image
            new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
            new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)

            # random anchors for the crop bounding boxes
            height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
            width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

            # assemble bounding boxes and crop
            bounding_boxes = tf.stack(
                [
                    height_offsets,
                    width_offsets,
                    height_offsets + new_heights,
                    width_offsets + new_widths,
                ],
                axis=1,
            )
            images = tf.image.crop_and_resize(
                images, bounding_boxes, tf.range(batch_size), (self.height, self.width)
            )

        return images


# distorts the color distibutions of images
class RandomColorJitter(layers.Layer):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, **kwargs):
        super().__init__(**kwargs)

        # color jitter ranges: (min jitter strength, max jitter strength)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # list of applicable color augmentations
        self.color_augmentations = [
            self.random_brightness,
            self.random_contrast,
            self.random_saturation,
            self.random_hue,
        ]

        # the tf.image.random_[brightness, contrast, saturation, hue] operations
        # cannot be used here, as they transform a batch of images in the same way

    def blend(self, images_1, images_2, ratios):
        # linear interpolation between two images, with values clipped to the valid range
        return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

    def random_brightness(self, images):
        # random interpolation/extrapolation between the image and darkness
        return self.blend(
            images,
            0,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.brightness, 1 + self.brightness
            ),
        )

    def random_contrast(self, images):
        # random interpolation/extrapolation between the image and its mean intensity value
        mean = tf.reduce_mean(
            tf.image.rgb_to_grayscale(images), axis=(1, 2), keepdims=True
        )
        return self.blend(
            images,
            mean,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.contrast, 1 + self.contrast
            ),
        )

    def random_saturation(self, images):
        # random interpolation/extrapolation between the image and its grayscale counterpart
        return self.blend(
            images,
            tf.image.rgb_to_grayscale(images),
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.saturation, 1 + self.saturation
            ),
        )

    def random_hue(self, images):
        # random shift in hue in hsv colorspace
        images = tf.image.rgb_to_hsv(images)
        images += tf.random.uniform(
            (tf.shape(images)[0], 1, 1, 3), (-self.hue, 0, 0), (self.hue, 0, 0)
        )
        # tf.math.floormod(images, 1.0) should be used here, however in introduces artifacts
        images = tf.where(images < 0.0, images + 1.0, images)
        images = tf.where(images > 1.0, images - 1.0, images)
        images = tf.image.hsv_to_rgb(images)
        return images

    def call(self, images, training=True):
        if training:
            # applies color augmentations in random order
            for color_augmentation in random.sample(self.color_augmentations, 4):
                images = color_augmentation(images)
        return images


# distorts the color distibutions of images
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0.2, saturation=0.2, hue=0.1, jitter=0.1, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.saturation = saturation
        self.hue = hue
        self.jitter = jitter

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # jitter: mixes colors across channels by applying a random affine transformation in color space
            color_mixers = tf.eye(3, batch_shape=[batch_size, 1]) + tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            # hue: applies a random rotation transformation around the main diagonal in color space
            rotation_angles = (
                tf.random.uniform(
                    (batch_size, 1, 1, 1), minval=-self.hue, maxval=self.hue
                )
                * math.pi
            )
            color_rotators = (
                tf.cos(rotation_angles) * tf.eye(3, batch_shape=[batch_size, 1])
                + tf.linalg.cross(
                    tf.sin(rotation_angles)
                    / tf.sqrt(3.0)
                    * tf.ones((batch_size, 1, 3, 3)),
                    tf.eye(3, batch_shape=[batch_size, 1]),
                )
                + (1 - tf.cos(rotation_angles)) / 3
            )

            # saturation: applies a random stretching transformation along the main diagonal in color space
            saturation_factors = tf.exp(
                tf.random.uniform(
                    (batch_size, 1, 1, 1),
                    minval=-self.saturation,
                    maxval=self.saturation,
                )
            )
            color_stretchers = (
                saturation_factors * tf.eye(3, batch_shape=[batch_size, 1])
                + (1 - saturation_factors) / 3
            )

            # brightness: applies a random scaling transformation in color space
            color_scalers = tf.exp(
                tf.random.uniform(
                    (batch_size, 1, 1, 1),
                    minval=-self.brightness,
                    maxval=self.brightness,
                )
            )

            images = tf.clip_by_value(
                images
                @ color_mixers
                @ color_rotators
                @ color_stretchers
                * color_scalers,
                0,
                1,
            )
        return images
