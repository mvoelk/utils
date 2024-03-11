
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class AugmentationUtility(object):
    def __init__(self,
            saturation_var=0.25,
            brightness_var=0.25,
            contrast_var=0.5,
            lighting_std=0.5,
            blur_kernel_sizes=(3,5),
            color_std=12.75):

        self.__dict__.update(locals())

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var + (1 - self.saturation_var)
        rgb = rgb * alpha + (1 - alpha) * gs[:,:,None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var + (1 - self.saturation_var)
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var + (1 - self.contrast_var)
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img = img + noise
        return np.clip(img, 0, 255)

    def noise(self, img):
        img_size = img.shape[:2]
        scale = np.random.randint(8)
        noise = np.array(np.random.exponential(scale, img_size), dtype=np.int) * np.random.randint(-1,2, size=img_size)
        #noise = np.array(np.random.normal(0, scale, img_size), dtype=np.int)
        noise = np.repeat(noise[:,:,None], 3, axis=2)
        img = img + noise
        return np.clip(img, 0, 255)

    def blur(self, img):
        k = np.random.choice(self.blur_kernel_sizes)
        img = cv2.GaussianBlur(img,(k,k),0)
        return np.clip(img, 0, 255)

    def color(self, img):
        # color shift
        #img = img + np.random.randint(48, size=(3)) - 24
        img = img + np.random.randn(3) * self.color_std
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img):
        img = img[:,::-1]
        y[:,(0,2)] = 1 - y[:,(2,0)]
        return img, y
    
    def vertical_flip(self, img, y):
        img = img[::-1,:]
        y[:,(1,3)] = 1 - y[:,(3,1)]
        return img, y

    def augment(self, img):
        for f in random.sample([
            self.noise,
            self.lighting,
            self.brightness,
            self.contrast,
            self.saturation,
            self.blur,
            self.color,
        ], 3): img = f(img)
        return img


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003].
    
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis",
        in Proc. of the International Conference on Document Analysis 
        and Recognition, 2003.
       
    Related code: https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    if len(image.shape) > 2:
        c = image.shape[2]
        distored_image = [map_coordinates(image[:,:,i], indices, order=1, mode='reflect') for i in range(c)]
        distored_image = np.concatenate(distored_image, axis=1)
    else:
        distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    
    return distored_image.reshape(image.shape)


def random_batch_mask1(shape=(128,32,32,3), n=4, p=0.95):
    batch_size = shape[0]
    size = shape[1]
    batch_mask = []
    for j in range(batch_size):
        mask = np.ones((size,size), 'uint8')
        s = size
        for i in range(n):
            s = s // 2
            r = size // s
            m = np.random.binomial(1, p, size=(s,s))
            m = np.repeat(np.repeat(m, r, axis=1), r, axis=0)
            m = np.roll(np.roll(m, np.random.randint(size), axis=1), np.random.randint(size), axis=0)
            mask = np.minimum(mask, m)
        mask = np.repeat(mask[...,None], repeats=shape[3], axis=-1)
        batch_mask.append(mask)
    return np.float32(batch_mask)

random_batch_mask = random_batch_mask1

def random_batch_mask2(shape=(128,32,32,3)):
    batch_size = shape[0]
    size = shape[1]
    batch_mask = []
    for j in range(batch_size):
        s = size
        m = np.random.binomial(1, 30/(s**2), size=(s,s))
        m[s//4:3*s//4,s//4:3*s//4] = 1
        m = m.astype('uint8')
        k = s//4 + random.randint(0, s//16)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        m = cv2.dilate(m, kernel, iterations=1)
        #k2 = max(1,s//32); k2 = 3
        #kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k2,k2))
        #m = cv2.dilate(m, kernel2, iterations=1)
        #m = cv2.erode(m, kernel2, iterations=1)
        mask = m
        mask = np.repeat(mask[...,None], repeats=shape[3], axis=-1)
        batch_mask.append(mask)
    return np.float32(batch_mask)

def plot_masks(masks):
    plt.figure(figsize=(16,8))
    for i in range(8):
        plt.subplot(181+i)
        plt.imshow(masks[i,:,:,0], cmap='gray')
    plt.show()

def blur_batch_mask(masks, kernel_sizes=[1,3,5,7,9,11]):
    masks_new = []
    for m in masks:
        k = np.random.choice(kernel_sizes)
        masks_new.append(cv2.GaussianBlur(m,(k,k),0))
    return np.float32(masks_new)





def random_box(img_shape):
    h, w = img_shape
    while True:
        x_min, x_max = np.sort(np.random.randint(0, w, 2))
        if x_min < x_max:
            break
    while True:
        y_min, y_max = np.sort(np.random.randint(0, h, 2))
        if y_min < y_max:
            break
    return x_min, y_min, x_max, y_max

def poly_mask(img_shape):
    x_min, y_min, x_max, y_max = random_box(img_shape)
    n = np.random.randint(3, 6)
    x = np.random.randint(x_min, x_max, (1,n))
    y = np.random.randint(y_min, y_max, (1,n))
    pts = np.stack([x,y], axis=-1)
    m = np.zeros(img_shape, np.uint8)
    m = cv2.fillPoly(m, pts, 1)
    return m

def ellipses_mask(img_shape, r_max=5):
    h, w = img_shape
    kx = 1 + np.random.randint(0, int(r_max)) * 2
    ky = 1 + np.random.randint(0, int(r_max)) * 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kx,ky))
    x_min, y_min, x_max, y_max = random_box((h,w))
    #x_min, y_min, x_max, y_max = random_box((h-ky,w-kx))
    #x_min, y_min, x_max, y_max = x_min+kx//2, y_min+ky//2, x_max+kx//2, y_max+ky//2
    dx, dy = x_max-x_min, y_max-y_min
    m = np.zeros(img_shape, np.uint8)
    #m[y_min:y_max,x_min:x_max] = np.uint8(np.random.binomial(1, 30/(dx*dy)*np.random.uniform(), size=(dy,dx)))
    m[y_min:y_max,x_min:x_max] = np.uint8(np.random.binomial(1, np.clip(30/(dx*dy)*np.random.uniform(), 0, 1), size=(dy,dx)))
    m = cv2.dilate(m, kernel, iterations=1)
    return m


def random_corruption_mask(img_shape):
    h, w = img_shape
    
    def noise_mask(img_shape, p_max=1.0):
        m = np.random.binomial(1, np.random.uniform(0, p_max), size=img_shape)
        return m
    
    m1 = poly_mask(img_shape)
    m2 = poly_mask(img_shape)
    m3 = noise_mask(img_shape)
    img1 = np.logical_or(m1, np.logical_and(m2, m3))
    img = img1

    m1 = ellipses_mask(img_shape, r_max=h//20)
    m2 = ellipses_mask(img_shape, r_max=h//10)
    m3 = noise_mask(img_shape)
    img2 = np.logical_or(m1, np.logical_and(m2, m3))
    img = np.logical_or(img, img2)
    
    m4 = noise_mask(img_shape, 0.05)
    img = np.logical_or(img, m4)
    
    return np.logical_not(img)



def checkerboard_mask(size=256, s=4, batch_size=None):
    assert size % (2*s) == 0
    m = np.tile(np.repeat(np.repeat([[1,0],[0,1]], s, axis=0), s, axis=1), (size//(2*s),size//(2*s)))
    if batch_size is not None:
        m = np.repeat(m[None,:,:,None], batch_size, axis=0)
    return np.float32(m)

def noise_mask(size=256, p=0.5, batch_size=None):
    m = np.random.binomial(1, p, (size,size))
    if batch_size is not None:
        m = np.repeat(m[None,:,:,None], batch_size, axis=0)
    return np.float32(m)

