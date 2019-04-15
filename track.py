import numpy as np
import cv2
import glob, os, sys, time, datetime

# TODO corners save when offline processing
ONLINE = True
CALIBRATE = False
RELATIVE_DESTINATION_PATH = str(datetime.date.today()) + '_distance/'
FPS = 60
THRESHOLD_WALL_VS_FLOOR = 80
THRESHOLD_ANIMAL_VS_FLOOR = 70
HD = 1280, 640
BGR_COLOR = {'red': (0,0,255),
             'green': (127,255,0),
             'blue': (255,127,0),
             'yellow': (0,127,255),
             'black': (0,0,0),
             'white': (255,255,255)}
WAIT_DELAY = 1

perspectiveMatrix = dict()
croppingPolygon = np.array([[0,0]])
croppingPolygons = dict()
tetragons = []
name = ""

RENEW_TETRAGON = True

def counterclockwiseSort(tetragon):
    tetragon = sorted(tetragon, key=lambda e: e[0])
    tetragon[0:2] = sorted(tetragon[0:2], key=lambda e: e[1])
    tetragon[2:4] = sorted(tetragon[2:4], key=lambda e: e[1], reverse=True)
    return tetragon

# TODO pointlike tetragon moving instead drawing it by clicking
# mouse callback function for drawing a cropping polygon
def drawFloorCrop(event, x, y, flags, params):
    global perspectiveMatrix, name, RENEW_TETRAGON
    imgCroppingPolygon = np.zeros_like(params['imgFloorCorners'])
    if event == cv2.EVENT_RBUTTONUP:
        cv2.destroyWindow(f'Floor Corners for {name}')
    if len(params['croppingPolygons'][name]) > 4 and event == cv2.EVENT_LBUTTONUP:
        RENEW_TETRAGON = True
        h = params['imgFloorCorners'].shape[0]
        # delete 5th extra vertex of the floor cropping tetragon
        params['croppingPolygons'][name] = np.delete(params['croppingPolygons'][name], -1, 0)
        params['croppingPolygons'][name] = params['croppingPolygons'][name] - [h,0]
        
        # Sort cropping tetragon vertices counter-clockwise starting with top left
        params['croppingPolygons'][name] = counterclockwiseSort(params['croppingPolygons'][name])
        # Get the matrix of perspective transformation
        params['croppingPolygons'][name] = np.reshape(params['croppingPolygons'][name], (4,2))
        tetragonVertices = np.float32(params['croppingPolygons'][name])
        tetragonVerticesUpd = np.float32([[0,0], [0,h], [h,h], [h,0]])
        perspectiveMatrix[name] = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON:
            params['croppingPolygons'][name] = np.array([[0,0]])
            RENEW_TETRAGON = False
        if len(params['croppingPolygons'][name]) == 1:
            params['croppingPolygons'][name][0] = [x,y]
        params['croppingPolygons'][name] = np.append(params['croppingPolygons'][name], [[x,y]], axis=0)
    if event == cv2.EVENT_MOUSEMOVE and not (len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON):
        params['croppingPolygons'][name][-1] = [x,y]
        if len(params['croppingPolygons'][name]) > 1:
            cv2.fillPoly(
                imgCroppingPolygon,
                [np.reshape(
                    params['croppingPolygons'][name],
                    (len(params['croppingPolygons'][name]),2)
                )],
                BGR_COLOR['green'], cv2.LINE_AA)
            imgCroppingPolygon = cv2.addWeighted(params['imgFloorCorners'], 1.0, imgCroppingPolygon, 0.5, 0.)
            cv2.imshow(f'Floor Corners for {name}', imgCroppingPolygon)

def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return np.abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

def floorCrop(filename):
    global perspectiveMatrix, tetragons, name, croppingPolygons
    name = os.path.splitext(filename)[0]
    cap = cv2.VideoCapture(filename)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while not frame.any():
        ret, frame = cap.read()

    frame = frame[:, w-h : w]

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernelSize = (5,5)
    frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
    retval, mask = cv2.threshold(frameBlur, THRESHOLD_WALL_VS_FLOOR, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    tetragons = []
    HALF_AREA = 0.5 * h * h
    for contour in contours:
        contourPerimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)

        # If the contour is convex tetragon
        # and its area is above a half of total frame area,
        # then it's most likely the floor
        if len(contour) == 4 and cv2.contourArea(contour) > HALF_AREA:
            contour = contour.reshape(-1, 2)
            max_cos = np.max([angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
            if max_cos < 0.3:
                tetragons.append(contour)
    frameGray = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)
    imgSquare = np.zeros_like(frameGray)
    cv2.fillPoly(imgSquare, tetragons, BGR_COLOR['red'], cv2.LINE_AA)
    # cv2.add(frameGray, imgSquare / 2, frameGray)
    cv2.drawContours(frameGray, tetragons, -1, BGR_COLOR['red'], 2, cv2.LINE_AA)

    if len(tetragons) > 0:
        tetragonVertices = tetragons[0]
    else:
        tetragonVertices = np.float32([[0,0], [0,h], [h,h], [h,0]])
    # Sort the cropping tetragon vertices according to the following order:
    # [left,top], [left,bottom], [right,bottom], [right,top]
    tetragonVertices = counterclockwiseSort(tetragonVertices)
    croppingPolygons[name] = tetragonVertices
    tetragonVertices = np.float32(tetragonVertices)
    tetragonVerticesUpd = np.float32([[0,0], [0,h], [h,h], [h,0]])
    perspectiveMatrix[name] = cv2.getPerspectiveTransform(np.float32(croppingPolygons[name]), tetragonVerticesUpd)
    frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (h,h))
    
    imgFloorCorners = np.hstack([frame, frameGray])
    cv2.imshow(f'Floor Corners for {name}', imgFloorCorners)
    cv2.setMouseCallback(
        f'Floor Corners for {name}',
        drawFloorCrop,
        {'imgFloorCorners': imgFloorCorners, 'croppingPolygons': croppingPolygons},
        )
    k = cv2.waitKey(0)
    if k == 27:
        sys.exit()
    cv2.destroyWindow(f'Floor Corners for {name}')
    return tetragonVertices, perspectiveMatrix[name]

def trace(filename):
    global perspectiveMatrix, croppingPolygons, tetragons, name, WAIT_DELAY
    # croppingPolygons[name] = np.array([[0,0]])
    name = os.path.splitext(filename)[0]
    cap = cv2.VideoCapture(filename)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while not frame.any():
        ret, frame = cap.read()

    background = frame.copy()
    i_frame = 1
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame is not None:
        ret, frame = cap.read()
        if frame is None:
            break
        background = cv2.addWeighted(frame, 0.5 - i_frame / n_frames, background, 0.5 + i_frame / n_frames, 0)
        i_frame += 1
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()

    frame = frame[:, w-h : w]
    
    # floorCrop(filename)

    video = cv2.VideoWriter(f'{RELATIVE_DESTINATION_PATH}timing/{name}_trace.avi',
        cv2.VideoWriter_fourcc(*'X264'),
        FPS, HD, cv2.INTER_LINEAR)
    imgTrack = np.zeros_like(frame)

    start = time.time()
    distance = _x = _y = 0
    while frame is not None:
        ret, frame = cap.read()
        
        if frame is None:   # not logical
            break
        frameColor = frame[:, w-h : w].copy()
        frame = cv2.subtract(frame, background)

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        frame = frame[:, w-h : w]
        if len(croppingPolygons[name]) == 4:
            cv2.drawContours(frameColor, [np.reshape(croppingPolygons[name], (4,2))], -1, BGR_COLOR['red'], 2, cv2.LINE_AA)
        else:
            cv2.drawContours(frameColor, tetragons, -1, BGR_COLOR['red'], 2, cv2.LINE_AA)

        frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (h,h))

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernelSize = (25,25)
        frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
        _, thresh = cv2.threshold(frameBlur, THRESHOLD_ANIMAL_VS_FLOOR, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) < 1:   # TODO more pythonic way of the check
            continue

        # Find a contour with the biggest area (animal most likely)
        contour = contours[np.argmax(list(map(cv2.contourArea, contours)))]
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        if _x == 0 and _y == 0:
            _x = x
            _y = y
        distance += np.sqrt(((x - _x) / float(h))**2 + ((y - _y) / float(h))**2)

        if ONLINE:
            # Draw the most acute angles of the contour (tail/muzzle/paws of the animal)
            hull = cv2.convexHull(contour)
            imgPoints = np.zeros(frame.shape,np.uint8)
            for i in range(2, len(hull) - 2):
                if np.dot(hull[i][0] - hull[i-2][0], hull[i][0] - hull[i+2][0]) > 0:
                    imgPoints = cv2.circle(imgPoints, (hull[i][0][0],hull[i][0][1]), 5, BGR_COLOR['yellow'], -1, cv2.LINE_AA)

            # Draw a contour and a centroid of the animal
            cv2.drawContours(imgPoints, [contour], 0, BGR_COLOR['green'], 2, cv2.LINE_AA)
            imgPoints = cv2.circle(imgPoints, (x,y), 5, BGR_COLOR['black'], -1)
            
            # Draw a track of the animal
            # imgTrack = cv2.add(np.zeros_like(imgTrack), cv2.line(imgTrack, (x,y), (_x,_y),
                # (255, 127, int(cap.get(cv2.CAP_PROP_POS_AVI_RATIO) * 255)), 1, cv2.LINE_AA))
            imgTrack = cv2.addWeighted(np.zeros_like(imgTrack), 0.85, cv2.line(imgTrack, (x,y), (_x,_y),
                (255, 127, int(cap.get(cv2.CAP_PROP_POS_AVI_RATIO) * 255)), 1, cv2.LINE_AA), 0.98, 0.)
            imgContour = cv2.add(imgPoints, imgTrack)

            frame = cv2.bitwise_and(frame, frame, mask=thresh)
            frame = cv2.addWeighted(frame, 0.4, imgContour, 1.0, 0.)

            cv2.putText(frame, 'Distance ' + str('%.2f' % distance),
                (190,420), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['white'])
            cv2.putText(frame, 'Time ' + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)),
                (200,450), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['white'])
            cv2.circle(frame, (x,y), 5, BGR_COLOR['black'], -1, cv2.LINE_AA)
            layout = np.hstack((frame, frameColor))

            cv2.imshow(f'Open Field Trace of {name}', layout)
            video.write(cv2.resize(layout, HD))

            k = cv2.waitKey(WAIT_DELAY) & 0xff
            if k == 27:
                break
            if k == 32:
                if WAIT_DELAY == 1:
                    WAIT_DELAY = 0  # pause
                else:
                    WAIT_DELAY = 1  # play as fast as possible
        _x = x
        _y = y
    cv2.destroyAllWindows()
    cap.release()

    if ONLINE:
        video.release()
        cv2.imwrite(RELATIVE_DESTINATION_PATH + 'traces/' + name + '_[distance]=%.2f' % distance +
            '_[time]=%.1fs' % t + '.png', cv2.resize(imgTrack, (max(HD), max(HD))))

    print(filename + '\tdistance %.2f\t' % distance + 'processing/real time %.1f' % float(time.time() - start) + '/%.1f s' % t)
    file.write(name + ',%.2f' % distance + ',%.1f\n' % t)
    file.close()

if len(sys.argv) > 1 and '--online' in sys.argv:
    ONLINE = True
if not os.path.exists(RELATIVE_DESTINATION_PATH + 'traces'):
    os.makedirs(RELATIVE_DESTINATION_PATH + 'traces')
if not os.path.exists(RELATIVE_DESTINATION_PATH + 'timing'):
    os.makedirs(RELATIVE_DESTINATION_PATH + 'timing')
file = open(RELATIVE_DESTINATION_PATH + 'distances.csv', 'w')
file.write('animal,distance [unit of the box side],run time [seconds]\n')
file.close()

for filename in glob.glob('*.avi'):
    floorCrop(filename)
for filename in glob.glob('*.avi'):
    file = open(RELATIVE_DESTINATION_PATH + 'distances.csv', 'a')
    trace(filename)
