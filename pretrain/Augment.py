# -- coding: utf-8 --
import random
import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import ImageOps


class Augment:
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

    def _augment_pretext(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        x = self._resize(x)
        x = self._crop(x, shape)
        x = self._resize(x)
        if self.args.channel == 3:
            x = self._random_color_jitter(x, p=.8)
            x = self._random_grayscale(x, p=.2)
            x = self._random_gaussian_blur(x, p=.5)
        x = self._random_hflip(x)
        x = self._standardize(x)
        return x

    def _augment_lincls(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        x = self._resize(x)
        x = self.center_crop(x, shape[0], shape[1], 0.85)
        if self.mode == 'train':
            if self.args.channel == 3:
                x = self._random_color_jitter(x, p=.8)
                x = self._random_grayscale(x, p=.2)
                x = self._random_gaussian_blur(x, p=.5)
            x = self._random_hflip(x)
        x = self._standardize(x)
        return x

    def _standardize(self, x):
        x = tf.cast(x, tf.float32)
        x/=255.0
        #x=(x -[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
        return x

    def _crop(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_size=shape,
            bounding_boxes=coord,
            area_range=(.2, 1.),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        offset_height, offset_width, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        x = tf.slice(x, [offset_height, offset_width, 0], [target_height, target_width, -1])
        return x

    def _compute_crop_shape(
            self, image_height, image_width, aspect_ratio, crop_proportion):
        """Compute aspect ratio-preserving shape for central crop.
        The resulting shape retains `crop_proportion` along one side and a proportion
        less than or equal to `crop_proportion` along the other side.
        Args:
          image_height: Height of image to be cropped.
          image_width: Width of image to be cropped.
          aspect_ratio: Desired aspect ratio (width / height) of output.
          crop_proportion: Proportion of image to retain along the less-cropped side.
        Returns:
          crop_height: Height of image after cropping.
          crop_width: Width of image after cropping.
        """
        image_width_float = tf.cast(image_width, tf.float32)
        image_height_float = tf.cast(image_height, tf.float32)

        def _requested_aspect_ratio_wider_than_image():
            crop_height = tf.cast(
                tf.math.rint(crop_proportion / aspect_ratio * image_width_float),
                tf.int32)
            crop_width = tf.cast(
                tf.math.rint(crop_proportion * image_width_float), tf.int32)
            return crop_height, crop_width

        def _image_wider_than_requested_aspect_ratio():
            crop_height = tf.cast(
                tf.math.rint(crop_proportion * image_height_float), tf.int32)
            crop_width = tf.cast(
                tf.math.rint(crop_proportion * aspect_ratio * image_height_float),
                tf.int32)
            return crop_height, crop_width

        return tf.cond(
            aspect_ratio > image_width_float / image_height_float,
            _requested_aspect_ratio_wider_than_image,
            _image_wider_than_requested_aspect_ratio)

    def center_crop(self, image, height, width, crop_proportion):
        """Crops to center of image and rescales to desired size.
        Args:
          image: Image Tensor to crop.
          height: Height of image to be cropped.
          width: Width of image to be cropped.
          crop_proportion: Proportion of image to retain along the less-cropped side.
        Returns:
          A `height` x `width` x channels Tensor holding a central crop of `image`.
        """
        shape = tf.shape(image)
        image_height = shape[0]
        image_width = shape[1]
        crop_height, crop_width = self._compute_crop_shape(
            image_height, image_width, width / height, crop_proportion)
        offset_height = ((image_height - crop_height) + 1) // 2
        offset_width = ((image_width - crop_width) + 1) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, crop_height, crop_width)

        image = self._resize(image, [height, width])

        return image

    def _resize(self, x):
        x = tf.image.resize(x, (self.args.img_size, self.args.img_size))
        x = tf.saturate_cast(x, tf.uint8)
        return x


    def _color_jitter(self, x, _jitter_idx=[0, 1, 2, 3]):
        random.shuffle(_jitter_idx)
        _jitter_list = [
            self._brightness,
            self._contrast,
            self._saturation,
            self._hue]
        for idx in _jitter_idx:
            x = _jitter_list[idx](x)
        return x

    def _random_color_jitter(self, x, p=.8):
        if tf.less(tf.random.uniform([]), p):
            x = self._color_jitter(x)
        return x

    def _brightness(self, x, brightness=0.4):
        ''' Brightness in torchvision is implemented about multiplying the factor to image,
            but tensorflow.image is just implemented about adding the factor to image.

        In tensorflow.image.adjust_brightness,
            For regular images, `delta` should be in the range `[0,1)`,
            as it is added to the image in floating point representation,
            where pixel values are in the `[0,1)` range.

        adjusted = math_ops.add(
            flt_image, math_ops.cast(delta, flt_image.dtype), name=name)

        However in torchvision docs,
        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.

        In torchvision.transforms.functional_tensor,
            return _blend(img, torch.zeros_like(img), brightness_factor)
            where _blend
                return brightness * img1
        '''
        # x = tf.image.random_brightness(x, max_delta=self.args.brightness)
        x = tf.cast(x, tf.float32)
        delta = tf.random.uniform(
            shape=[],
            minval=1 - brightness,
            maxval=1 + brightness,
            dtype=tf.float32)

        x *= delta
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _contrast(self, x, contrast=0.4):
        x = tf.image.random_contrast(x, lower=max(0, 1 - contrast), upper=1 + contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _saturation(self, x, saturation=0.4):
        x = tf.image.random_saturation(x, lower=max(0, 1 - saturation), upper=1 + saturation)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _hue(self, x, hue=0.1):
        x = tf.image.random_hue(x, max_delta=hue)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _grayscale(self, x):
        return tf.image.rgb_to_grayscale(x)  # after expand_dims

    def _random_grayscale(self, x, p=.2):
        if tf.less(tf.random.uniform([]), p):
            x = self._grayscale(x)
            x = tf.tile(x, [1, 1, 3])
        return x

    def _random_hflip(self, x):
        x=tf.image.random_flip_left_right(x)
        return tf.image.random_flip_up_down(x)

    def _get_gaussian_kernel(self, sigma, filter_shape=3):
        x = tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1)
        x = tf.cast(x ** 2, sigma.dtype)
        x = tf.nn.softmax(-x / (2.0 * (sigma ** 2)))
        return x

    def _get_gaussian_kernel_2d(self, gaussian_filter_x, gaussian_filter_y):
        gaussian_kernel = tf.matmul(gaussian_filter_x, gaussian_filter_y)
        return gaussian_kernel

    def _random_gaussian_blur(self, x, p=.5):
        if tf.less(tf.random.uniform([]), p):
            sigma = tf.random.uniform([], .1, 2.)
            filter_shape = 3
            x = tf.expand_dims(x, axis=0)
            x = tf.cast(x, tf.float32)
            channels = tf.shape(x)[-1]

            gaussian_kernel_x = self._get_gaussian_kernel(sigma, filter_shape)
            gaussian_kernel_x = gaussian_kernel_x[None, :]

            gaussian_kernel_y = self._get_gaussian_kernel(sigma, filter_shape)
            gaussian_kernel_y = gaussian_kernel_y[:, None]

            gaussian_kernel_2d = self._get_gaussian_kernel_2d(gaussian_kernel_y, gaussian_kernel_x)
            gaussian_kernel_2d = gaussian_kernel_2d[:, :, None, None]
            gaussian_kernel_2d = tf.tile(gaussian_kernel_2d, [1, 1, channels, 1])

            x = tf.nn.depthwise_conv2d(x, gaussian_kernel_2d, (1, 1, 1, 1), "SAME")
            x = tf.squeeze(x)
            x = tf.saturate_cast(x, tf.uint8)
        return x

    def random_resize_crop(self, image, min_scale, max_scale, crop_size):
        # Conditional resizing
        if crop_size == 224:
            image_shape = 260
            image = tf.image.resize(image, (image_shape, image_shape))
        else:
            image_shape = 120
            image = tf.image.resize(image, (image_shape, image_shape))
        # Get the crop size for given min and max scale
        size = tf.random.uniform(shape=(1,), minval=min_scale * image_shape,
                                 maxval=max_scale * image_shape, dtype=tf.float32)
        size = tf.cast(size, tf.int32)[0]
        # Get the crop from the image
        crop = tf.image.random_crop(image, (size, size, self.args.channel))
        crop_resize = self._resize(crop)

        return crop_resize

    def scale_image(self, image):
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    def tie_together(self, image, min_scale, max_scale, crop_size):
        # Retrieve the image features
        image = image['image']
        # Scale the pixel values
        image = self.scale_image(image)
        # Random resized crops
        image = self.random_resize_crop(image, min_scale,
                                        max_scale, crop_size)
        # Color distortions & Gaussian blur
        if self.args.channel == 3:
            image = self._random_color_jitter(image)
            if self.args.dataset == 'imagenet':
                image = self._random_gaussian_blur(image)

        return image

    def get_multires_dataset(self, dataset,
                             size_crops,
                             num_crops,
                             min_scale,
                             max_scale,
                             options=None):
        loaders = tuple()
        for i, num_crop in enumerate(num_crops):
            for _ in range(num_crop):
                loader = (
                    dataset
                        .shuffle(1024)
                        .map(lambda x: self.tie_together(x, min_scale[i],
                                                         max_scale[i], size_crops[i]),
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
                )
                if options != None:
                    loader = loader.with_options(options)
                loaders += (loader,)
        trainloaders_zipped = tf.data.Dataset.zip(loaders)

        # Final trainloader
        loaders_zipped = (
            trainloaders_zipped
                .batch(self.args.batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
        return loaders_zipped

    def equalize(self, img, shape):
        # calculate histogram
        #
        # hists = np.histogram(img, bins=256)[0]
        # m, n = shape[:2]
        # # caculate cdf
        # hists_cumsum = np.cumsum(hists)
        # const_a = 256 / (m * n)
        # hists_cdf = (const_a * hists_cumsum).astype("uint8")
        #
        # # mapping
        # img_eq = hists_cdf[img]
        img_eq = cv.equalizeHist(img)
        return img_eq


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    """Blurs the given image with separable convolution.
    Args:
      image: Tensor of shape [height, width, channels] and dtype float to blur.
      kernel_size: Integer Tensor for the size of the blur kernel. This is should
        be an odd number. If it is an even number, the actual kernel size will be
        size + 1.
      sigma: Sigma value for gaussian operator.
      padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
    Returns:
      A Tensor representing the blurred image.
    """
    radius = tf.cast(kernel_size / 2, dtype=tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
    blur_filter = tf.exp(-tf.pow(x, 2.0) /
                         (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        # Tensorflow requires batched input to convolutions, which we can fake with
        # an extra dimension.
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


def random_apply(func, x, p):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)


def random_blur(image, height, width, p=1.0):
    """Randomly blur an image.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      p: probability of applying this transformation.
    Returns:
      A preprocessed image `Tensor`.
    """
    del width

    def _transform(image):
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        return gaussian_blur(
            image, kernel_size=height // 10, sigma=sigma, padding='SAME')

    return random_apply(_transform, p=p, x=image)


def batch_random_blur(images_list, height, width, blur_probability=0.5):
    """Apply efficient batch data transformations.
    Args:
      images_list: a list of image tensors.
      height: the height of image.
      width: the width of image.
      blur_probability: the probaility to apply the blur operator.
    Returns:
      Preprocessed feature list.
    """

    def generate_selector(p, bsz):
        shape = [bsz, 1, 1, 1]
        selector = tf.cast(
            tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p),
            tf.float32)
        return selector

    new_images_list = []
    for images in images_list:
        images_new = random_blur(images, height, width, p=1.)
        selector = generate_selector(blur_probability, tf.shape(images)[0])
        images = images_new * selector + images * (1 - selector)
        images = tf.clip_by_value(images, 0., 1.)
        new_images_list.append(images)

    return new_images_list


def equalize(img):
    # calculate histogram
    img_eq = cv.equalizeHist(img)
    return img_eq


if __name__ == '__main__':
    from gl import Config

    image_path = r'G:\superwang\ck+\processed\0\S010_002_00000001.png'
    image = cv.imread(image_path)
    print(image.shape)
    # image = np.expand_dims(image, -1)
    cv.imshow('a', image)
    cv.waitKey(0)
    g = Augment(Config())._standardize(image)
    # g = Augment(Config())._crop(image,(256,256,3))
    print(g)
    cv.imshow('g', g.numpy())
    cv.waitKey(0)
