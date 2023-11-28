import torch
import uvd
import os
import cv2
from tqdm import tqdm

if __name__ == "__main__":

    # note that the video cannot be too long, a 40-second video works

    video_filename = "/home/weiyu/Research/kdm/UVD/scripts/examples/rgb_static_0_1000.mp4"
    save_dir = "/home/weiyu/Research/kdm/UVD/scripts/examples/segments"

    # liv and dinov2 have been tested
    preprocessor_name = "dinov2"

    video_name = os.path.basename(video_filename).replace(".mp4", "")
    segment_dir = os.path.join(save_dir, video_name, preprocessor_name)

    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)

    # (N sub-goals, *video frame shape)
    subgoal_idxs, frames = uvd.get_uvd_subgoals(
        video_filename,   # video filename or (L, *video frame shape) video numpy array
        preprocessor_name=preprocessor_name,    # Literal["vip", "r3m", "liv", "clip", "vc1", "dinov2"]
        device="cuda" if torch.cuda.is_available() else "cpu",  # device for loading frozen preprocessor
        return_indices=True,   # True if only want the list of subgoal timesteps
    )

    subgoals = [frames[idx] for idx in subgoal_idxs]

    for subgoal in subgoals:
        print(subgoal.shape)

    prev_subgoal_idx = 0
    for si, subgoal_idx in tqdm(enumerate(subgoal_idxs)):

        imgs = frames[prev_subgoal_idx:subgoal_idx]

        # cv2.namedWindow("segment")
        # for img in imgs:
        #     cv2.imshow("segment", img)
        #     # milliseconds: 30 fps = 33.3 ms
        #     waitKey = (cv2.waitKey(33) & 0xFF)
        #     if waitKey == ord('q'):  # if Q pressed you could do something else with other keypress
        #         print("closing video and exiting")
        #         cv2.destroyWindow("segment")
        #         break
        # key = cv2.waitKey()
        # if key == ord("q"):
        #     cv2.destroyAllWindows()

        segment_name = f"subgoal_{si}_{prev_subgoal_idx}_{subgoal_idx}"
        video_filename = os.path.join(segment_dir, f"{segment_name}.mp4")
        print(f"Saving segment to {video_filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_filename, fourcc, 30.0, (224, 224))

        for image in imgs:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()

        prev_subgoal_idx = subgoal_idx

