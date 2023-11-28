import torch
import uvd

if __name__ == "__main__":
    # (N sub-goals, *video frame shape)
    subgoal_idxs, frames = uvd.get_uvd_subgoals(
        "/home/weiyu/Research/kdm/UVD/scripts/examples/microwave-bottom_burner-light_switch-slide_cabinet.mp4",   # video filename or (L, *video frame shape) video numpy array
        preprocessor_name="vip",    # Literal["vip", "r3m", "liv", "clip", "vc1", "dinov2"]
        device="cuda" if torch.cuda.is_available() else "cpu",  # device for loading frozen preprocessor
        return_indices=True,   # True if only want the list of subgoal timesteps
    )

    subgoals = [frames[idx] for idx in subgoal_idxs]

    for subgoal in subgoals:
        print(subgoal.shape)