import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl


All_All_lines = []
def draw_lane_lines2(image):


    # Greyscale image
    if image is None:

        return

    greyscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imshape = greyscaled_image.shape

    # Gaussian Blur
    blurred_grey_image = gaussian_blur(greyscaled_image)


        # Canny edge detection
    edges_image = canny(blurred_grey_image)

    # Mask edges image
    border = 0
    vertices = np.array([[(0, imshape[0]), (465, 320), (475, 320),
                          (imshape[1], imshape[0])]], dtype=np.int32)
    edges_image_with_mask = region_selection(edges_image)
    # Hough lines



    hough_transform_lines = HoughLines(edges_image_with_mask)



    # Combine lines image with original image
    left_line, right_line = lane_lines(image=image, lines=hough_transform_lines)
    all_lines = (left_line, right_line)
    All_All_lines.append(all_lines)
    final_pic = draw_lane_lines(image=image, lines=all_lines)



    return final_pic

def grayscale(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey
def gaussian_blur(image):
    blur = cv2.blur(image, (5, 5))
    return blur
def canny(image):
    edge = cv2.Canny(image,50,150)

    return edge
def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    mask = np.zeros_like(image)
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #We could have used fixed numbers as the vertices of the polygon,
    #but they will not be applicable to images with different dimesnions.
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def HoughLines(image):
    rho = 1              #Distance resolution of the accumulator in pixels.
    theta = np.pi/180    #Angle resolution of the accumulator in radians.
    threshold = 20       #Only lines that are greater than threshold will be returned.
    minLineLength = 20   #Line segments shorter than that are rejected.
    maxLineGap = 300     #Maximum allowed gap between points on the same line to link them
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)
def draw_lines(image, lines, color = [255, 0, 0], thickness = 2):
    """
    Draw lines onto the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            lines: The lines we want to draw.
            color (Default = red): Line color.
            thickness (Default = 2): Line thickness.
    """
    image = np.copy(image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        # print("X1", x1, "y1", y1, "x2", x2, "y2", y2)
        # cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
#Start of the video processing code

def video(path):
    imgs = []
    final_imgs = []
    vidcap = cv2.VideoCapture(path)
    fps = int(vidcap.get(5))
    print(fps)

    success = True
    while success:
        success, image = vidcap.read()
        imgs.append(image)
    print(len(imgs))

    for img in imgs:
        final_img = draw_lane_lines2(img)
        final_imgs.append(final_img)
    # height, width, layers = final_imgs[1].shape
    video_name = "Output_Video.avi"
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    final_image_array = np.asarray(final_imgs)
    height, width, layers = final_image_array[1].shape
    height = int(height)
    width = int(width)


    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))


    for i in range(1, len(final_image_array)):
        if final_image_array[i] is None:
            return
        #Final Frame img
        imggggg = final_image_array[i]

        video.write(imggggg)





    cv2.destroyAllWindows()
    video.release()
    # video = cv2.VideoWriter('video2.avi', -1, 1, (width, height))
    # for j in range(0, 5):
    #     video.write(final_imgs[j])
    #     print("Testtttttt")
    # cv2.destroyAllWindows()
    # video.release()






video("/Users/dylanmashini/PycharmProjects/Self Driving/video.mp4")






