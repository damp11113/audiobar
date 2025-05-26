import cv2
import numpy as np
import wave
from pyogg import OpusDecoder
from tqdm import tqdm


def bytes_similarity(bytes1, bytes2):
    """Calculate similarity percentage between two byte sequences"""
    if len(bytes1) != len(bytes2):
        return 0.0

    if len(bytes1) == 0:
        return 100.0

    # Count matching bytes
    matching_bytes = sum(b1 == b2 for b1, b2 in zip(bytes1, bytes2))
    return (matching_bytes / len(bytes1)) * 100.0


input_video = r"C:\Users\sansw\Desktop\Sequence 01.mp4"
output_wav = r"C:\Users\sansw\Desktop\decoded_audio4.wav"
sample_rate = 48000
num_channels = 2
downscale_factor = 4
similarity_threshold = 90.0  # Skip frames that are 90% or more similar

# Open video file
video = cv2.VideoCapture(input_video)

# Get video properties
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize Opus decoder
opus_decoder = OpusDecoder()
opus_decoder.set_sampling_frequency(sample_rate)
opus_decoder.set_channels(num_channels)

# Initialize WAV output
out_wav = wave.open(output_wav, 'wb')
out_wav.setnchannels(num_channels)
out_wav.setsampwidth(2)  # 16-bit audio
out_wav.setframerate(sample_rate)
last_frame = b""
last_opus_bytes = None  # Store the previous encoded opus bytes for comparison

scaled_resolution = (int(width / downscale_factor), int(height / downscale_factor))

with tqdm(total=total_frames, desc='Decoding frames') as pbar:
    while True:
        ret, frame = video.read()
        if not ret:
            break

        downscaled_frame = cv2.resize(frame, scaled_resolution, interpolation=cv2.INTER_NEAREST)

        # Convert frame to grayscale and threshold to get binary values
        gray = cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2GRAY)
        bits = (gray > 127).flatten()

        # Convert bits back to bytes
        opus_bytes = np.packbits(bits.astype(bool)).tobytes()

        # Check similarity with previous frame
        if last_opus_bytes is not None:
            similarity = bytes_similarity(opus_bytes, last_opus_bytes)

            if similarity >= similarity_threshold:
                print(f"Frame skipped! Similarity: {similarity:.1f}%")
                pbar.update(1)
                continue

        # Store current opus bytes for next comparison
        last_opus_bytes = opus_bytes

        # Decode audio
        try:
            audio_data = opus_decoder.decode(memoryview(bytearray(opus_bytes)))
            last_frame = audio_data
        except:
            audio_data = last_frame
            print("Frame corrupted! Using last frame")

        if audio_data:
            # Convert to 16-bit PCM and write to WAV
            audio_pcm = np.frombuffer(audio_data, dtype=np.int16)
            out_wav.writeframes(audio_pcm.tobytes())

        pbar.update(1)

# Clean up
video.release()
out_wav.close()