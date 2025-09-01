import math
import os
from abc import abstractmethod
from io import BytesIO

# import arcade
import cv2
import numpy as np
import skimage as sk
from PIL import Image as PILImage
from PIL import ImageEnhance, ImageOps
# from dataloader.corrupted.image_based.PythonShaders.ShaderToy import Shadertoy
from scipy.ndimage.interpolation import map_coordinates
from skimage.color.colorconv import rgb2hsv, hsv2rgb
from skimage.filters import gaussian
import numpy as np
from scipy.ndimage import gaussian_filter

from constants import CorruptionConstants as const
from dataloader.corrupted.corruption_utils import disk, MotionImage, clipped_zoom, plasma_fractal


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    width, height = x.size
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(height - c[1], c[1], -1):
            for w in range(width - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    width, height = x.size
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())


    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (height, width):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    width, height = x.size
    map_size = 2 ** math.ceil(math.log(max(width, height), 2))
    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:height, :width][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    # idx = np.random.randint(5)

    width, height = x.size
    root_path = os.path.dirname(__file__)
    frost_filter_path = os.path.join(root_path, "frost/frostlarge.jpg")
    frost = cv2.imread(frost_filter_path)
    # randomly crop and convert to rgb
    frost = frost[:height, :width][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    width, height = x.size
    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(height, width,
                                                                                          1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return np.array(x)


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    width, height = x.size
    x = x.resize((int(width * c), int(height * c)), PILImage.BOX)
    x = x.resize((width, height), PILImage.BOX)

    return np.array(x)


def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),  # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def sun_glare(x, severity=1):
    c = [1, 0.9, 0.8, 0.7, 0.6][severity - 1]
    # idx = np.random.randint(5)
    x = x.copy()
    width, height = x.size
    cwd = os.path.dirname(__file__)
    sunglare_path = os.path.join(cwd, "corruption_filter/sunglare.png")
    sunglare = PILImage.open(sunglare_path).convert('RGBA')
    margin = (1 - c) / 2
    # crop the middle part of the sunglare with ratio
    sunglare = sunglare.crop((sunglare.size[0] * margin, sunglare.size[1] * margin, sunglare.size[0] * (1 - margin),
                              sunglare.size[1] * (1 - margin)))
    # Resize the sunglare to the size of the image
    sunglare = sunglare.resize((width, height), PILImage.LANCZOS)
    x.paste(sunglare, (0, 0), sunglare)

    return np.array(x)


# def waterdrop(x, severity=1):
#     c = ["rain1.glsl", "rain2.glsl", "rain3.glsl", "rain4.glsl", "rain5.glsl"][severity - 1]
#     # Setup Window
#     IMAGE_SIZE = x.size
#     window = arcade.open_window(*IMAGE_SIZE, window_title="Waterdrop")
#
#     root_path = os.path.dirname(__file__)
#     noise_image_path = os.path.join(root_path, "PythonShaders/noise.png")
#     noise = PILImage.open(noise_image_path)
#     noise = noise.resize(IMAGE_SIZE)
#     noise = noise.rotate(180)
#     noise = ImageOps.mirror(noise)
#     noise = np.asarray(noise, dtype=np.uint8)
#
#     # Setup Shader:
#     shader = Shadertoy.create_from_file(window.get_size(), "PythonShaders/shaders/" + c)
#     shader.channel_1 = to_texture(noise, window.ctx)
#
#     # Load Images
#     x = x.rotate(180)
#     x = ImageOps.mirror(x)
#     x = np.asarray(x, dtype=np.uint8)
#
#     # Set shader data
#     shader.channel_0 = to_texture(x, window.ctx)
#
#     # Render image
#     random_time = 1 * 10
#     shader.render(time=random_time)
#
#     # Save image
#     result = arcade.get_image(0, 0, IMAGE_SIZE[0], IMAGE_SIZE[1])
#     result = np.array(result)
#
#     arcade.close_window()
#
#     return result[:, :, :3]


def wildfire_smoke(x, severity=1):
    c = [1, 0.9, 0.8, 0.7, 0.6][severity - 1]
    # idx = np.random.randint(5)
    x = x.copy()
    width, height = x.size

    cwd = os.path.dirname(__file__)
    wildfire_smoke_filter_path = os.path.join(cwd, "corruption_filter/smoke.png")
    smoke = PILImage.open(wildfire_smoke_filter_path).convert('RGBA')
    # crop the bottom part of the sunglare
    smoke = smoke.crop((0, smoke.size[1] * (1 - c), smoke.size[0], smoke.size[1]))
    smoke = smoke.resize((width, height), PILImage.LANCZOS)
    x.paste(smoke, (0, 0), smoke)

    return np.array(x)


def dust(x, severity=1):
    c = [1, 0.95, 0.9, 0.8, 0.7][severity - 1]
    x = x.copy()
    width, height = x.size
    cwd = os.path.dirname(__file__)
    dust_filter_path = os.path.join(cwd, "corruption_filter/dust.png")
    dust = PILImage.open(dust_filter_path).convert('RGBA')
    dust = dust.crop((0, 0, dust.size[0] * c, dust.size[1] * c))
    dust = dust.resize((width, height), PILImage.LANCZOS)
    x.paste(dust, (0, 0), dust)

    return np.array(x)


def rain(x, severity=1):
    c = [0.2, 0.4, 0.6, 0.8, 1][severity - 1]
    x = x.copy()
    width, height = x.size
    cwd = os.path.dirname(__file__)
    rain_filter_path = os.path.join(cwd, "corruption_filter/rain.png")
    rain = PILImage.open(rain_filter_path).convert('RGBA')
    # enhance the rain
    rain = ImageEnhance.Brightness(rain).enhance(c)
    # crop the bottom part of the sunglare
    rain = rain.resize((width, height), PILImage.LANCZOS)
    x.paste(rain, (0, 0), rain)

    return np.array(x)


def no_corruption(x, severity=1):
    return np.array(x)


corruption_name_to_function = {
    const.NO_CORRUPTION: no_corruption,
    const.GAUSSIAN_NOISE: gaussian_noise,
    const.SHOT_NOISE: shot_noise,
    const.IMPULSE_NOISE: impulse_noise,
    const.SPECKLE_NOISE: speckle_noise,
    const.GAUSSIAN_BLUR: gaussian_blur,
    const.GLASS_BLUR: glass_blur,
    const.DEFOCUS_BLUR: defocus_blur,
    const.MOTION_BLUR: motion_blur,
    const.ZOOM_BLUR: zoom_blur,
    const.FOG: fog,
    const.FROST: frost,
    const.SNOW: snow,
    const.SPATTER: spatter,
    const.CONTRAST: contrast,
    const.BRIGHTNESS: brightness,
    const.SATURATE: saturate,
    const.JPEG_COMPRESSION: jpeg_compression,
    const.PIXELATE: pixelate,
    const.ELASTIC_TRANSFORM: elastic_transform,
    const.SUN_GLARE: sun_glare,
    const.WILDFIRE_SMOKE: wildfire_smoke,
    const.DUST: dust,
    const.RAIN: rain
}


class BaseImageCorruptionGenerator:

    def __init__(self):
        self._corruption_name_to_function = corruption_name_to_function

        self._corruptions_list = list(self._corruption_name_to_function.keys())
        self._no_corruption_threshold = 0.7

    @abstractmethod
    def fetch_corruption_transforms(self, num_frames):
        pass


class FixedDatasetCorruptionGenerator(BaseImageCorruptionGenerator):

    def __init__(self, dataset_corruption_type, corruption_severity_level):
        super().__init__()
        self._dataset_corruption_type = dataset_corruption_type
        self._corruption_severity_level = corruption_severity_level

    def fetch_corruption_transforms(self, num_frames):
        if self._dataset_corruption_type is None:
            corruption_index = np.random.randint(0, len(self._corruptions_list))
            self._dataset_corruption_type = self._corruptions_list[corruption_index]

        corruption_transform_list = []
        for _ in range(num_frames):
            # Flip a coin to decide whether to apply the fixed corruption
            corruption_transform = self._dataset_corruption_type if np.random.rand() < self._no_corruption_threshold else const.NO_CORRUPTION
            corruption_transform_list.append(self._corruption_name_to_function[corruption_transform])

        return corruption_transform_list


class FixedVideoCorruptionGenerator(BaseImageCorruptionGenerator):

    def __init__(self, corruption_severity_level):
        super().__init__()
        self._corruption_severity_level = corruption_severity_level

    def fetch_corruption_transforms(self, num_frames):
        corruption_index = np.random.randint(0, len(self._corruptions_list))
        video_corruption_type = self._corruptions_list[corruption_index]

        corruption_transform_list = []
        for _ in range(num_frames):
            # Flip a coin to decide whether to apply the fixed corruption
            corruption_transform = video_corruption_type if np.random.rand() < 0.5 else const.NO_CORRUPTION
            corruption_transform_list.append(self._corruption_name_to_function[corruption_transform])

        return corruption_transform_list


class MixedVideoCorruptionGenerator(BaseImageCorruptionGenerator):

    def __init__(self, corruption_severity_level):
        super().__init__()
        self._corruption_severity_level = corruption_severity_level

    def fetch_corruption_transforms(self, num_frames):
        corruption_transform_list = []
        for _ in range(num_frames):
            # Flip a coin to decide whether to apply the fixed corruption
            corruption_index = np.random.randint(0, len(self._corruptions_list))
            corruption_transform = self._corruptions_list[corruption_index]
            corruption_transform_list.append(self._corruption_name_to_function[corruption_transform])

        return corruption_transform_list


class StandardGenerator(BaseImageCorruptionGenerator):

    def __init__(self, corruption_severity_level):
        super().__init__()
        self._corruption_severity_level = 0

    def fetch_corruption_transforms(self, num_frames):
        corruption_transform_list = []
        for _ in range(num_frames):
            corruption_transform = const.NO_CORRUPTION
            corruption_transform_list.append(self._corruption_name_to_function[corruption_transform])

        return corruption_transform_list
