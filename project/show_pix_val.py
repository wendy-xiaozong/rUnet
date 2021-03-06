import cv2


def movemouse(event, x, y, flags, param):
    global img
    img2 = img.copy()
 
    if event==cv2.EVENT_MOUSEMOVE:
         font = cv2.FONT_HERSHEY_SIMPLEX
         message = '{}'.format(img2[y, x])
         cv2.putText(img2, message, (int(w/2.5),int(h/16)),
                     font, 0.5, (255,255,255), 1)
         cv2.circle(img2, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow('image', img2)
 
 
def main():
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", movemouse)
    cv2.waitKey()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    img = cv2.imread('sea.jpg')
    img_size = img.shape
    h, w = img_size[0:2]
    main()
