import numpy as np
import matplotlib.pyplot as plt

def initializeCanvas(H, W):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ratio = H / np.pi
    return canvas, ratio

def generateMesh(shape, FOV_H=np.pi/4, FOV_W=np.pi/3):
    H, W, _ = shape
    center_H = H / 2.0
    center_W = W / 2.0

    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    xx, yy = np.meshgrid(x, y, sparse=False)
    xx = xx - center_W
    yy = yy - center_H

    # Angle covered per pixel
    theta_unit = FOV_W / W
    phi_unit = FOV_H / H

    theta = xx * theta_unit
    phi = yy * phi_unit + np.pi / 2

    # Compute the spherical coordinates of each pixel
    x_ca, y_ca, z_ca = sphericalToCartesian(theta, phi)

    return x_ca, y_ca, z_ca

def sphericalToCartesian(theta, phi):
    rho = 1
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return x, y, z

def cartesianToSpherical(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / rho)
    theta = np.arctan2(y, x)
    return phi, theta

def transformPixels(x, y, z, Rot):
    H, W = x.shape

    # Construct all columns for coordinates
    coords = np.zeros((3, H, W)) 
    coords[0, :, :] = x
    coords[1, :, :] = y
    coords[2, :, :] = z
    coords_reshaped = coords.reshape((3, H * W))

    # Transform the pixels
    coords_reshaped = np.dot(Rot, coords_reshaped)

    # Reshape back
    x_prime = coords_reshaped[0, :]
    y_prime = coords_reshaped[1, :]
    z_prime = coords_reshaped[2, :]

    return x_prime, y_prime, z_prime

def copyPixels(img, canvas, x, y):
    H, W, _ = img.shape

    # Flip the image horizontally, viewing it from inside the sphere
    img = img[:, ::-1, :]

    # Construct coordinates for indexing
    rows = np.array(y.reshape(H * W), dtype=np.int32)
    cols = np.array(x.reshape(H * W), dtype=np.int32)

    ch_1 = np.zeros(H * W, dtype=np.int32)
    ch_2 = np.ones(H * W, dtype=np.int32)
    ch_3 = np.ones(H * W, dtype=np.int32) * 2

    img_ch_1 = img[:, :, 0].reshape(H * W)
    img_ch_2 = img[:, :, 1].reshape(H * W)
    img_ch_3 = img[:, :, 2].reshape(H * W)

    canvas[rows, cols, ch_1] = img_ch_1
    canvas[rows, cols, ch_2] = img_ch_2
    canvas[rows, cols, ch_3] = img_ch_3

    return canvas

def stitchImage(img, R, canvas, ratio):
    # Generate XYZ mesh of a spherical image
    x_ca, y_ca, z_ca = generateMesh(img.shape)

    # Transform the pixels to given orientation
    x_prime, y_prime, z_prime = transformPixels(x_ca, y_ca, z_ca, R)

    # Convert back the cartesian coordinates to spherical
    phi_y, theta_x = cartesianToSpherical(x_prime, y_prime, z_prime)
    
    # Shift theta to range [0, 2*pi]
    theta_x += np.pi

    # Transform the angles to pixel coordinates
    x_final = np.array(theta_x * ratio, dtype=np.int32)
    y_final = np.array(phi_y * ratio, dtype=np.int32)

    canvas = copyPixels(img=img, canvas=canvas, x=x_final, y=y_final)

    return canvas

def plotPanorama(vicd, camd, save_path):
    # Record valid number of images captured by camera
    canvas, ratio = initializeCanvas(camd["cam"][0].shape[0], camd["cam"][0].shape[0] * 2)
    cam_ts = len(camd["ts"][0])
    vic_ts = len(vicd["ts"][0])

    j = 0
    for i in range(cam_ts):
        cam_time = camd["ts"][0][i]
        vic_time = vicd["ts"][0][j]

        # find the nearest time stamp
        while vic_time < cam_time:
            j += 1
            if j >= vic_ts:
                j -= 1
                vic_time = vicd["ts"][0][j]
                break

            vic_time = vicd["ts"][0][j]

        img = camd["cam"][:, :, :, i]
        rotation = vicd["rots"][:, :, j]

        canvas = stitchImage(img, rotation, canvas, ratio)

    plt.imsave(save_path, canvas)

