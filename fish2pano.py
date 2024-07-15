import cv2
import numpy as np
import math
PI = 3.14159265358979323846

def fish2pano(img):
    width = img.shape[0] //2

    Panorama = np.zeros((width*2, width*4, 3), np.uint8)
    pano_x = np.zeros((width, width*4, 1), np.float32)
    pano_y = np.zeros((width, width*4, 1), np.float32)
    for i in range(width):
        for j in range(width*4):
            radius = width -i
            x = width + int(radius * np.cos(2*PI*j/4/width))
            y = width - int(radius * np.sin(2*PI*j/4/width))
            
            if x < img.shape[0] and y<img.shape[1]:
                Panorama[i+width, j] = img[x, y]
                pano_x[i, j] = x
                pano_y[i, j] = y
                
    return Panorama, pano_x, pano_y


def main():
    video_path = "/home/kist/Downloads/abc.mp4"
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("/home/kist/Videos/pano.mp4", fourcc, 30, (3072, 768))


    ret, frame = cap.read()
    frame = cv2.resize(frame, dsize = (1536, 1536))

    pano, pano_x, pano_y = fish2pano(frame)
    cv2.imshow('y', pano)
    while(True):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, dsize = (1536, 1536))
            re_frame = cv2.remap(frame, pano_x, pano_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

            cv2.imshow("pano", re_frame)
            frame = cv2.resize(frame, dsize = (600, 600))
            cv2.imshow("x", frame)

            video_writer.write(re_frame)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break 
        else :
            break
    video_writer.release()
    
    
if __name__ == "__main__":
    main()