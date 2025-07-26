import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from math import ceil

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area
def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img

def nothing(x):
    pass

def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src

def get_available_frames():
    frames_file = 'frames.txt'
    frames = []
    if os.path.exists(frames_file):
        with open(frames_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    name, width, height = parts
                    frames.append({
                        'name': name,
                        'width': int(width),
                        'height': int(height)
                    })
    else:
        # Default frames if file doesn't exist
        frames = [
            {'name': 'Small', 'width': 300, 'height': 400},
            {'name': 'Medium', 'width': 500, 'height': 700},
            {'name': 'Large', 'width': 800, 'height': 1000},
        ]
    return frames

def get_fitting_frames(painting_width, painting_height):
    frames = get_available_frames()
    fitting = []
    for frame in frames:
        if painting_width <= frame['width'] and painting_height <= frame['height']:
            fitting.append(frame)
    return fitting

def get_lshape_frames():
    frames = [
        {'name': 'gold_classic', 'img_path': 'frames/gold_classic.png'},
        {'name': 'slim_gold', 'img_path': 'frames/slim_gold.png'},
        {'name': 'grey_modern', 'img_path': 'frames/grey_modern.png'}
    ]
    return frames

def extract_frame_content(image_path):
    """Extract frame content by removing external black background while preserving frame details"""
    from PIL import Image, ImageOps
    import numpy as np
    
    # Load image
    img = Image.open(image_path).convert('RGBA')
    # Convert to numpy array for processing
    img_array = np.array(img)
    
    # Create a mask for external black pixels
    # We'll use a more sophisticated approach to identify the actual frame
    is_black = np.all(img_array[:, :, :3] < 30, axis=2)
    
    # Find the frame boundaries by scanning from edges
    def find_content_boundary(arr, reverse=False):
        if reverse:
            arr = arr[::-1]
        for i, row in enumerate(arr):
            if not np.all(row):  # If not all pixels in row are black
                return len(arr) - i if reverse else i
        return 0
    
    # Scan from all four sides
    top = find_content_boundary(is_black)
    bottom = find_content_boundary(is_black[::-1, :], reverse=True)
    left = find_content_boundary(is_black.T)
    right = find_content_boundary(is_black.T[::-1, :], reverse=True)
    
    # Crop to content
    frame_content = img.crop((left, top, right, bottom))
    return frame_content

def create_depth_map(frame_profile, width, height):
    """Create a depth map for 3D frame rendering"""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    # Create base depth map
    depth = np.zeros((height, width), dtype=np.float32)
    
    # Frame profile parameters
    frame_width = max(min(min(width, height) // 12, 120), 60)
    bevel_width = frame_width // 3
    frame_depth = frame_width // 4
    
    # Create frame edges with proper beveling
    for i in range(frame_width):
        # Calculate depth based on frame profile
        if i < bevel_width:
            # Outer bevel
            d = (i / bevel_width) * frame_depth
        elif i > frame_width - bevel_width:
            # Inner bevel
            d = ((frame_width - i) / bevel_width) * frame_depth
        else:
            # Flat surface
            d = frame_depth
        depth[i, :] = d  # Top
        depth[-i-1, :] = d  # Bottom
        depth[:, i] = d  # Left
        depth[:, -i-1] = d  # Right
    
    # Smooth transitions
    depth = gaussian_filter(depth, sigma=2)
    return depth

def apply_lighting(img, depth_map, light_angle=45):
    """Apply 3D lighting effects using the depth map"""
    import numpy as np
    
    # Convert image to float32 for calculations
    img_float = np.array(img, dtype=np.float32) / 255.0
    
    # Calculate surface normals from depth map
    gy, gx = np.gradient(depth_map)
    normals = np.dstack((-gx, -gy, np.ones_like(depth_map)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    
    # Light direction (adjustable)
    light = np.array([
        np.cos(np.radians(light_angle)),
        np.sin(np.radians(light_angle)),
        1.0
    ])
    light /= np.linalg.norm(light)
    
    # Calculate diffuse lighting
    diffuse = np.maximum(0, np.sum(normals * light, axis=2))
    diffuse = diffuse[..., np.newaxis]
    
    # Calculate specular highlights
    reflection = 2.0 * diffuse * normals
    reflection = reflection - light
    specular = np.maximum(0, reflection[:,:,2]) ** 20
    specular = specular[..., np.newaxis]
    
    # Combine lighting effects
    lit_img = img_float * (0.7 + 0.3 * diffuse + 0.2 * specular)
    lit_img = np.clip(lit_img * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(lit_img)

def autocrop_lcorner(img):
    import numpy as np
    from PIL import Image
    img_np = np.array(img)
    # Create mask for non-black (frame) pixels
    mask = np.any(img_np[:, :, :3] > 30, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img  # fallback: nothing to crop
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    cropped = img.crop((x0, y0, x1, y1))
    return cropped


def white_to_transparent(img, threshold=240):
    import numpy as np
    img = img.convert('RGBA')
    arr = np.array(img)
    # Create mask for near-white pixels
    mask = np.all(arr[:, :, :3] > threshold, axis=2)
    arr[mask, 3] = 0  # Set alpha to 0 for white
    return Image.fromarray(arr)


def get_frame_width_from_lcorner(lcorner_img):
    import numpy as np
    arr = np.array(lcorner_img)
    alpha = arr[:, :, 3]
    # Find the first row/col from the edge where alpha=0 (transparent)
    # We'll use the top and left edges for the width
    def first_transparent_edge(a):
        for i in range(a.shape[0]):
            if np.all(a[i, :] == 0):
                return i
        return a.shape[0]  # fallback: no transparency
    # Top edge
    top = first_transparent_edge(alpha)
    # Left edge
    left = first_transparent_edge(alpha.T)
    # Use the minimum (in case of slight asymmetry)
    return min(top, left)


def find_inner_edge(lcorner_img):
    import numpy as np
    arr = np.array(lcorner_img)
    alpha = arr[:, :, 3]
    size = min(alpha.shape[0], alpha.shape[1])
    for i in range(size):
        if alpha[i, i] > 0:
            return i
    return 0


def composite_frame_around_painting(painting_path, frame_name, output_path, painting_width_in=None, painting_height_in=None):
    from PIL import Image
    import numpy as np
    import os

    painting = Image.open(painting_path).convert('RGBA')
    pw, ph = painting.size

    # Use a fixed frame width for a visually slim frame
    frame_width = 40  # pixels

    # Frame asset path
    base_path = 'static/frames/'
    lcorner_path = os.path.join(base_path, f'{frame_name}.png')
    lcorner_img = Image.open(lcorner_path).convert('RGBA')
    lcorner_img = white_to_transparent(lcorner_img)
    lcorner_img = autocrop_lcorner(lcorner_img)
    lw, lh = lcorner_img.size

    # Crop L-corner and sides to frame_width (preserve detail)
    l_corner = lcorner_img.crop((0, 0, frame_width, frame_width))
    side_top = lcorner_img.crop((frame_width, 0, lw, frame_width))
    side_left = lcorner_img.crop((0, frame_width, frame_width, lh))

    # Output canvas (frame + painting)
    out_w, out_h = pw + 2 * frame_width, ph + 2 * frame_width
    out = Image.new('RGBA', (out_w, out_h), (0, 0, 0, 0))

    # Place painting
    out.paste(painting, (frame_width, frame_width), painting)

    # Place corners
    out.paste(l_corner, (0, 0), l_corner)
    l_corner_tr = l_corner.transpose(Image.FLIP_LEFT_RIGHT)
    out.paste(l_corner_tr, (out_w - frame_width, 0), l_corner_tr)
    l_corner_bl = l_corner.transpose(Image.FLIP_TOP_BOTTOM)
    out.paste(l_corner_bl, (0, out_h - frame_width), l_corner_bl)
    l_corner_br = l_corner.transpose(Image.ROTATE_180)
    out.paste(l_corner_br, (out_w - frame_width, out_h - frame_width), l_corner_br)

    # Tile sides at native resolution
    if side_top.width > 0 and side_top.height > 0:
        for x in range(frame_width, out_w - frame_width, side_top.width):
            region = side_top
            if x + side_top.width > out_w - frame_width:
                region = side_top.crop((0, 0, out_w - frame_width - x, frame_width))
            out.paste(region, (x, 0), region)
            region_b = region.transpose(Image.FLIP_TOP_BOTTOM)
            out.paste(region_b, (x, out_h - frame_width), region_b)
    if side_left.width > 0 and side_left.height > 0:
        for y in range(frame_width, out_h - frame_width, side_left.height):
            region = side_left
            if y + side_left.height > out_h - frame_width:
                region = side_left.crop((0, 0, frame_width, out_h - frame_width - y))
            out.paste(region, (0, y), region)
            region_r = region.transpose(Image.FLIP_LEFT_RIGHT)
            out.paste(region_r, (out_w - frame_width, y), region_r)

    # Save as PNG
    if not output_path.lower().endswith('.png'):
        output_path = output_path.rsplit('.', 1)[0] + '.png'
    out.save(output_path, format='PNG', quality=100, optimize=False)