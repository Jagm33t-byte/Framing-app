import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, redirect, url_for , make_response
import utils
from utils import get_fitting_frames, get_lshape_frames

UPLOAD_FOLDER = 'uploads'
CROPPED_FOLDER = 'cropped'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    filename = file.filename
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    cropped_filename = 'cropped_' + filename
    cropped_path = os.path.join(CROPPED_FOLDER, cropped_filename)

    # Try auto-detect and get corners
    img = cv2.imread(upload_path)
    heightImg, widthImg = img.shape[:2]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = (200, 200)
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxArea = utils.biggestContour(contours)
    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        corners = biggest.reshape(4, 2).tolist()
    else:
        # Default to image corners if detection fails
        corners = [[0, 0], [widthImg, 0], [widthImg, heightImg], [0, heightImg]]

    return render_template('manual_crop.html', filename=filename, corners=corners)

@app.route('/manual_crop')
def manual_crop():
    filename = request.args.get('filename')
    # If corners are passed as query params, parse them, else default to image corners
    img = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
    heightImg, widthImg = img.shape[:2]
    corners = request.args.get('corners')
    if corners:
        import ast
        corners = ast.literal_eval(corners)
    else:
        corners = [[0, 0], [widthImg, 0], [widthImg, heightImg], [0, heightImg]]
    return render_template('manual_crop.html', filename=filename, corners=corners)



@app.route('/save_manual_crop', methods=['POST'])
def save_manual_crop():
    filename = request.form['filename']
    x = int(float(request.form['x']))
    y = int(float(request.form['y']))
    w = int(float(request.form['width']))
    h = int(float(request.form['height']))

    img_path = os.path.join(UPLOAD_FOLDER, filename)
    img = cv2.imread(img_path)
    cropped = img[y:y+h, x:x+w]

    cropped_filename = 'cropped_' + filename
    cropped_path = os.path.join(CROPPED_FOLDER, cropped_filename)
    cv2.imwrite(cropped_path, cropped)

    # Get dimensions of cropped image
    cropped_img = cv2.imread(cropped_path)
    height, width = cropped_img.shape[:2]
    fitting_frames = get_fitting_frames(width, height)

    return render_template('select_frame.html', filename=cropped_filename, width=width, height=height, frames=fitting_frames)


@app.route('/preview_crop', methods=['POST'])
def preview_crop():
    filename = request.form['filename']
    x = int(float(request.form['x']))
    y = int(float(request.form['y']))
    w = int(float(request.form['width']))
    h = int(float(request.form['height']))

    img_path = os.path.join(UPLOAD_FOLDER, filename)
    img = cv2.imread(img_path)
    cropped = img[y:y+h, x:x+w]

    cropped_filename = 'cropped_' + filename
    cropped_path = os.path.join(CROPPED_FOLDER, cropped_filename)
    cv2.imwrite(cropped_path, cropped)

    return render_template('preview.html', filename=cropped_filename, orig_filename=filename, x=x, y=y, w=w, h=h)


@app.route('/save_polygon_crop', methods=['POST'])
def save_polygon_crop():
    filename = request.form['filename']
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    img = cv2.imread(img_path)

    h, w = img.shape[:2]

    pts = []
    for i in range(4):
        x = float(request.form['x'+str(i)])
        y = float(request.form['y'+str(i)])
        pts.append([x, y])
    pts = np.array(pts, dtype='float32')

    # Order points: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    cropped_filename = 'cropped_' + filename
    cropped_path = os.path.join(CROPPED_FOLDER, cropped_filename)
    cv2.imwrite(cropped_path, warped)

    return redirect(url_for('show_final', filename=cropped_filename))


@app.route('/ask_dimensions', methods=['POST'])
def ask_dimensions():
    filename = request.form['filename']
    return render_template('ask_dimensions.html', filename=filename)

# Add a new route for frame selection after dimensions
@app.route('/frame_select', methods=['POST'])
def frame_select():
    filename = request.form['filename']
    width = float(request.form['width'])
    height = float(request.form['height'])
    width_unit = request.form['width_unit']
    height_unit = request.form['height_unit']
    if width_unit == 'feet':
        width *= 12
    if height_unit == 'feet':
        height *= 12
    lshape_frames = get_lshape_frames()
    return render_template('frame_select.html', filename=filename, width=width, height=height, frames=lshape_frames)

# Add a new route for frame preview after selection
@app.route('/frame_preview', methods=['POST'])
def frame_preview():
    filename = request.form['filename']
    width = request.form['width']
    height = request.form['height']
    frame_name = request.form['frame']
    lshape_frames = get_lshape_frames()
    selected_frame = next((f for f in lshape_frames if f['name'] == frame_name), lshape_frames[0])
    
    # Clean up any existing preview images for this file
    preview_path = f'static/frames/preview_{filename}.png'
    if os.path.exists(preview_path):
        os.remove(preview_path)
        
    from utils import composite_frame_around_painting
    painting_path = f'cropped/{filename}'
    composite_frame_around_painting(painting_path, frame_name, preview_path)
    
    # Add cache busting parameter
    preview_url = f'{preview_path}?t={int(os.path.getmtime(preview_path))}'
    
    return render_template('frame_preview.html', filename=filename, width=width, height=height, frame=selected_frame, preview_path=preview_url)

@app.route('/select_frame', methods=['POST'])
def select_frame():
    filename = request.form['filename']
    width = float(request.form['width'])
    height = float(request.form['height'])
    width_unit = request.form['width_unit']
    height_unit = request.form['height_unit']
    # Convert to inches if needed
    if width_unit == 'feet':
        width *= 12
    if height_unit == 'feet':
        height *= 12
    # Swap width and height for correct display
    lshape_frames = get_lshape_frames()
    return render_template('select_frame.html', filename=filename, width=width, height=height, frames=lshape_frames)

@app.route('/show_final', methods=['GET', 'POST'])
def show_final():
    if request.method == 'POST':
        filename = request.form.get('filename')
        frame = request.form.get('frame')
        return f'''
        <p>✅ Final cropped image:</p>
        <img src="/cropped/{filename}" width="300"><br>
        <p>Selected Frame: {frame}</p>
        '''
    else:
        filename = request.args.get('filename')
        return f'''
        <p>✅ Final cropped image:</p>
        <img src="/cropped/{filename}" width="300">
        '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/cropped/<filename>')
def cropped_file(filename):
    response = make_response(send_from_directory(CROPPED_FOLDER, filename))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def detect_and_crop(image_path, output_path):
    import utils
    img = cv2.imread(image_path)
    heightImg, widthImg = img.shape[:2]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = (200, 200)
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxArea = utils.biggestContour(contours)
    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        pts1 = np.float32(biggest.reshape(4, 2))
        pts2 = np.float32([[0, 0], [widthImg, 0], [widthImg, heightImg], [0, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
        cv2.imwrite(output_path, imgWarpColored)
        return True
    return False

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

if __name__ == '__main__':
    app.run(debug=True)
