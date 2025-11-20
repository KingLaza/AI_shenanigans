import cv2 as cv
from cv2 import aruco
import glob
import numpy as np

BOARD_SIZE = (5, 7)
MARKER_SIZE = 2
MARKER_SEPARATION = 0.4
aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
aruco_params = aruco.DetectorParameters()
board = aruco.GridBoard(BOARD_SIZE, MARKER_SIZE, MARKER_SEPARATION, aruco_dictionary)

def calibrate(files: str):
    h = 0
    w = 0
    counter = []
    corner_lista = []
    id_lista = []
    images = glob.glob(files)
    for (i, img) in enumerate(images):
        image = cv.imread(str(img))
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            img_gray, aruco_dictionary, parameters=aruco_params)
        if i == 0:
            h, w = image.shape[:2]
            corner_lista = corners
            id_lista = ids
        else:
            corner_lista = np.vstack((corner_lista, corners))
            id_lista = np.vstack((id_lista, ids))
        counter.append(len(ids))

    counter = np.array(counter)
    _, mtx, dist, _, _ = aruco.calibrateCameraAruco(
        corner_lista, id_lista, counter, board, (w, h), None, None)

    ncm, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return (mtx, dist, ncm, roi)


def video(cap, camera_matrix, dist_coefficients, camera_matrix_new, roi):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.array(frame)
        frame_remapped_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, bad_img_points = aruco.detectMarkers(
            frame_remapped_gray, aruco_dictionary, parameters=aruco_params)
        aruco.refineDetectedMarkers(
            frame_remapped_gray, board, corners, ids, bad_img_points)

        im_with_aruco_board = frame

        if type(ids) is np.ndarray and ids.any() != None:
            im_with_aruco_board = aruco.drawDetectedMarkers(
                frame, corners, ids, (0, 255, 0))
            rot_vec = ()
            trans_vec = ()
            retval, rot_vec, trans_vec = aruco.estimatePoseBoard(
                corners, ids, board, camera_matrix, dist_coefficients, rot_vec, trans_vec)
            if retval != 0:
                im_with_aruco_board = cv.drawFrameAxes(
                    im_with_aruco_board, camera_matrix, dist_coefficients, rot_vec, trans_vec, 5)

        undistorted = cv.undistort(im_with_aruco_board,
                             camera_matrix, dist, None, camera_matrix_new)

        width = int(undistorted.shape[1] * 0.6)
        height = int(undistorted.shape[0] * 0.6)
        dim = (width, height)
        undistorted = cv.resize(undistorted, dim, interpolation=cv.INTER_AREA)

        cv.imshow("Video ", undistorted)
        if cv.waitKey(2) & 0xFF == ord('c'):
            break
    return

if __name__ == '__main__':
    mtx, dist, nmc, roi = calibrate('resources/*.jpg')
    cap = cv.VideoCapture("resources/Aruco_board.mp4")
    video(cap, mtx, dist, nmc, roi)
    cap.release()
    cv.destroyAllWindows()

