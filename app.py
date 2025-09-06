import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import tempfile
import os
import subprocess
from magic_shield import MagicShield

shield = MagicShield()


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.writer = None
        self.frame_size = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output_file = None
        self.output_path = None
        self.latest_frame = None
        self.recording = False  # recording state

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = shield.process(img)

        self.latest_frame = img.copy()

        if self.recording:
            # Initialize writer when recording starts
            if self.writer is None:
                self.output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                self.output_path = self.output_file.name
                h, w = img.shape[:2]
                self.frame_size = (w, h)
                self.writer = cv2.VideoWriter(
                    self.output_path, self.fourcc, 20.0, self.frame_size
                )

            if self.writer:
                self.writer.write(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def release(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False


def convert_to_playable_mp4(input_path: str) -> str | None:
    """Re-encode video using ffmpeg to ensure browser compatibility."""
    fixed_path = input_path.replace(".mp4", "_fixed.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        fixed_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0 or not os.path.exists(fixed_path):
        st.error(f"⚠️ Failed to convert video. FFmpeg output:\n{result.stderr.decode()}")
        return None

    return fixed_path


st.title("🌀 Doctor Strange Magic Shield (Web Demo)")

ctx = webrtc_streamer(key="magic-shield", video_processor_factory=VideoProcessor)

# Snapshot button
if st.button("📸 Capture Snapshot"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        frame_bgr = ctx.video_processor.latest_frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".png", frame_bgr)

        st.success("✅ Snapshot captured!")
        st.image(frame_rgb, caption="Captured Snapshot", width="stretch")

        st.download_button(
            label="⬇️ Download Snapshot",
            data=buffer.tobytes(),
            file_name="snapshot.png",
            mime="image/png"
        )
    else:
        st.warning("⚠️ No frame available yet.")

# Recording controls
if ctx.video_processor:
    if st.button("🔴 Start Recording") and not ctx.video_processor.recording:
        ctx.video_processor.recording = True
        st.info("🎥 Recording started...")

    if st.button("🛑 Stop & Save Recording") and ctx.video_processor.recording:
        ctx.video_processor.release()
        if ctx.video_processor.output_path and os.path.exists(ctx.video_processor.output_path):
            fixed_video = convert_to_playable_mp4(ctx.video_processor.output_path)
            if fixed_video:
                st.success("✅ Video saved and converted!")
                with open(fixed_video, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Recorded Video",
                        data=file,
                        file_name="output.mp4",
                        mime="video/mp4"
                    )
