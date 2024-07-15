from fish2pano import fish2pano
from Equirec2Perspec import GetPerspective, fish2persp
import cv2
import os
import tqdm
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, required=True
)

args = parser.parse_args()

def main(video_path):
    """_summary_
    """
    cap = cv2.VideoCapture(video_path)
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # while(True):
    #     ret, frame = cap.read()
    #     frame = frame[:, 250:-212, :]
    #     print(frame.shape)
    #     frame = cv2.resize(frame, dsize=(1550, 1550))

    #     cv2.imshow("frame", frame)
    #     cv2.waitKey(1)
    ret, frame = cap.read()

    # frame = frame[:, :, :]
    # frame = cv2.resize(frame,  dsize=(1536, 1536))
    # fisheye image to panorama image
    # frame = cv2.resize(frame, (1280, 1280))
    _, pano_x, pano_y = fish2pano(frame)
    lst_remap = []
    for theta in [0, 90, 180, 270]:
        x_remap_mat, y_remap_mat = fish2persp(frame, 80, theta, -50, 600, 600, pano_x, pano_y)
        lst_remap.append([x_remap_mat, y_remap_mat])

    # pickle.dump([pano_x, pano_y], "/remap_matrix/fish2persp")
    # pickle.dump(lst_remap, "/remap_matrix/fish2persp")
    
    # lst_video_writer = []
    # for i in range(len(lst_remap)):
    #     lst_video_writer.append(cv2.VideoWriter(
    #         os.path.join(save_path, f"persp_{i}.avi"), cv2.VideoWriter_fourcc(*"DIVX"), 30, (720,720)))
    # pano_writer = cv2.VideoWriter(os.path.join(save_path, f"pano.avi"), cv2.VideoWriter_fourcc(*"DIVX"), 30, (pano_x.shape[1], pano_x.shape[0]))
    
    for _ in range(int(video_length)) :
        ret, frame = cap.read()
        
        if ret:
            # cv2.imshow(f"real", frame)
            # frame = frame[:, 256:-256, :]

            # frame = cv2.resize(frame,  dsize=(1536, 1536))
            # cv2.imshow(f"real", frame)
            # frame = cv2.resize(frame, (1280, 1280))

            pano = cv2.remap(frame, pano_x, pano_y, interpolation=cv2.INTER_LINEAR, borderMode = cv2.BORDER_WRAP)
            # pano_writer.write(pano)
            cv2.imshow(f"pano",pano)

            for i, remap_mat in enumerate(lst_remap):
                persp = cv2.remap(frame, remap_mat[0], remap_mat[1], interpolation=cv2.INTER_LINEAR, borderMode = cv2.BORDER_WRAP)
                # lst_video_writer[i].write(persp)
                cv2.imshow(f"persp_{i}",persp)
            cv2.waitKey(1)
    # pano_writer.release()
    # for i in range(len(lst_video_writer)):
    #     lst_video_writer[i].release()
        
        
if __name__ == "__main__":
    # save_path = "/home/kist/Videos"
    main(args.video)

