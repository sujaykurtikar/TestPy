import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from magic_shield import MagicShield

shield = MagicShield()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = shield.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🌀 Doctor Strange Magic Shield (Web Demo)")
webrtc_streamer(key="magic-shield", video_processor_factory=VideoProcessor)
