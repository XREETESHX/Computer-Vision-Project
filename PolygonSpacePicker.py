import cv2
import pickle
import numpy as np

IMG_PATH = 'carParkImg.png'
SAVE_FILE = 'polygons'
POINTS_PER_POLYGON = 4

try:
    with open(SAVE_FILE, 'rb') as f:
        polygons = pickle.load(f)
except:
        polygons = []

current_pts = []


def draw():
    img = cv2.imread(IMG_PATH)
    if img is None:
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(img, 'Image not found: ' + IMG_PATH, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    overlay = img.copy()

    for i, poly in enumerate(polygons):
        pts = np.array(poly, np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
        cv2.fillPoly(overlay, [pts], color=(255, 0, 255, 50))
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        cv2.putText(overlay, str(i + 1), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i, p in enumerate(current_pts):
        cv2.circle(overlay, p, 4, (0, 255, 255), -1)
        if i > 0:
            cv2.line(overlay, current_pts[i - 1], current_pts[i], (0, 255, 255), 2)

    if len(current_pts) == POINTS_PER_POLYGON:
        pts = np.array(current_pts, np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.imshow('Image', overlay)


def save():
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(polygons, f)


def point_in_poly(x, y, poly):
    """Check if point is inside polygon"""
    pts = np.array(poly, np.int32)
    return cv2.pointPolygonTest(pts, (x, y), False) >= 0


def mouse_callback(event, x, y, flags, param):
    global current_pts, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_pts) < POINTS_PER_POLYGON:
            current_pts.append((x, y))
            if len(current_pts) == POINTS_PER_POLYGON:
                polygons.append(current_pts.copy())
                current_pts = []
                save()

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, poly in enumerate(polygons):
            if point_in_poly(x, y, poly):
                polygons.pop(i)
                save()
                break


def main():
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    print('Controls:')
    print('  Left click: add point (4 points per polygon)')
    print('  Right click: delete polygon')
    print('  u: undo last point')
    print('  r: reset current polygon')
    print('  s: save')
    print('  q or Esc: quit')

    while True:
        draw()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('u'):
            if current_pts:
                current_pts.pop()
        elif key == ord('r'):
            current_pts = []
        elif key == ord('s'):
            save()
            print('Saved', len(polygons), 'polygons to', SAVE_FILE)
        elif key == ord('q') or key == 27:
            save()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
