import os
import cv2
import numpy as np


class EpisodeVideoRecorder:
    """Record one episode as an mp4 by stitching the camera color images each frame.

    Fed the same obs dict the sim publishes over ZMQ:
        obs["camera_observations"]["color_images"][cam_name] -> HxWx3 uint8 RGB
    Output: the mp4 at the video_path passed in, e.g.
    logs/task_videos/<task>_<ts>/loop<N>.mp4 (logs/ is git-ignored).
    Recording runs whenever start_recording()/add_frame() are called by the sim side.
    """

    def __init__(self, video_path, fps=30):
        self.video_path = video_path
        self.fps = fps
        self.video_writer = None
        self.episode_dir = None
        self.stop_flag = False

    def start_recording(self):
        """Prepare the output directory for this episode (writer is lazily created on first frame)."""
        self.stop_flag = False
        self.episode_dir = os.path.dirname(self.video_path)
        os.makedirs(self.episode_dir, exist_ok=True)
        print(f"[video] recording episode to {self.episode_dir}")
        self.video_writer = None

    def add_frame(self, obs):
        """Append one frame built from all camera color images in obs."""
        if self.stop_flag:
            return
        if obs is None or "camera_observations" not in obs:
            return
        try:
            color_images = obs["camera_observations"]["color_images"]

            # Collect all camera images (kept sorted for a stable layout).
            cam_images = []
            for cam_name in sorted(color_images.keys()):
                img = color_images[cam_name]
                if len(img.shape) == 3 and img.shape[2] == 3:
                    cam_images.append((cam_name, img))

            if not cam_images:
                return

            # Layout: 1 -> as-is, 2 -> side by side, 3-4 -> 2x2 grid (blank fills 3rd/4th).
            if len(cam_images) == 1:
                combined_img = cam_images[0][1]
            elif len(cam_images) == 2:
                h = min(img.shape[0] for _, img in cam_images)
                images = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for _, img in cam_images]
                combined_img = np.hstack(images)
            else:
                n = min(4, len(cam_images))
                h = min(cam_images[i][1].shape[0] for i in range(n))
                w = min(cam_images[i][1].shape[1] for i in range(n))
                images = [cv2.resize(cam_images[i][1], (w, h)) for i in range(n)]
                while len(images) < 4:
                    images.append(np.zeros_like(images[0]))
                top_row = np.hstack([images[0], images[1]])
                bottom_row = np.hstack([images[2], images[3]])
                combined_img = np.vstack([top_row, bottom_row])

            # Lazily create the writer once the frame size is known.
            if self.video_writer is None:
                height, width = combined_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (width, height))
                print(f"[video] writer created: {self.video_path} ({width}x{height})")

            # obs images are RGB; OpenCV writer expects BGR.
            combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
            self.video_writer.write(combined_img_bgr)

        except Exception as e:
            print(f"[video] error adding frame: {e}")

    def stop_recording(self):
        """Release the writer and finalize the mp4."""
        self.stop_flag = True
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            if self.video_path:
                print(f"[video] saved: {self.video_path}")
        else:
            print(f"[video] episode had no frames recorded")

