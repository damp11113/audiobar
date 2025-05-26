import numpy as np
import cv2
from pyogg import OpusBufferedEncoder
import wave
from tqdm import tqdm
import math

def find_best_resolution(total_pixels):
    # Find all possible resolutions
    possible_resolutions = []

    for width in range(1, int(math.sqrt(total_pixels)) + 1):
        if total_pixels % width == 0:
            height = total_pixels // width
            possible_resolutions.append((width, height))

    # Print all possible resolutions
    print("Possible resolutions:")
    for idx, (width, height) in enumerate(possible_resolutions, start=1):
        print(f"{idx}: {width} x {height}")

    # Ask the user for the preferred aspect
    orientation = input("Would you prefer the resolution to be more 'horizontal' or 'vertical'? (Enter 'h' for horizontal, 'v' for vertical): ").strip().lower()

    # Adjust resolutions based on the user preference
    if orientation == 'h':
        # Swap width and height for horizontal preference
        possible_resolutions = [(height, width) for (width, height) in possible_resolutions]
    elif orientation == 'v':
        # No change for vertical, default order
        pass
    else:
        print("Invalid input, proceeding with default order.")

    # Ask the user to select a resolution
    try:
        user_choice = int(input(f"Please choose a resolution (1-{len(possible_resolutions)}): "))
        if 1 <= user_choice <= len(possible_resolutions):
            return possible_resolutions[user_choice - 1]
        else:
            print("Invalid choice, returning the most square-like resolution.")
            return possible_resolutions[0]
    except ValueError:
        print("Invalid input, returning the most square-like resolution.")
        return possible_resolutions[0]

input_wav = r"C:\Users\sansw\Desktop\Где мой дом.wav"
output_video = r"C:\Users\sansw\Desktop\encoded_opus2.avi"
bitrate = 128000
frame_size = 60
fps = 25
upscale_factor = 4

# Open input file
in_wav = wave.open(input_wav, "rb")
sample_rate = in_wav.getframerate()
num_channels = in_wav.getnchannels()
num_frames = in_wav.getnframes()

# Initialize Opus encoder
opus_encoder = OpusBufferedEncoder()
opus_encoder.set_application("audio")
opus_encoder.set_sampling_frequency(sample_rate)
opus_encoder.set_channels(num_channels)
opus_encoder.set_bitrates(bitrate)
opus_encoder.set_compresion_complex(10)
opus_encoder.set_bitrate_mode("CBR")
opus_encoder.set_frame_size(frame_size)

# Calculate parameters
chunk_size = int((frame_size / 1000) * sample_rate)
bits_per_frame = int(bitrate * (frame_size / 1000))
resolution = find_best_resolution(bits_per_frame)

upscaled_resolution = (resolution[0] * upscale_factor, resolution[1] * upscale_factor)

# Initialize video writer
video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, upscaled_resolution)

try:
    with tqdm(total=num_frames, unit='frame', desc='Processing chunks') as pbar:
        while True:
            # Read audio chunks
            frames = in_wav.readframes(chunk_size)
            if not frames:
                break

            # Process audio
            frames = np.frombuffer(frames, dtype=np.int16)
            opus_bytes = opus_encoder.buffered_encode(memoryview(bytearray(frames)), flush=True)[0][0].tobytes()

            # Convert to bits and create frame
            bits = np.unpackbits(np.frombuffer(opus_bytes, dtype=np.uint8))

            # Pad or truncate bits to match frame size
            bits_needed = resolution[0] * resolution[1]
            if len(bits) < bits_needed:
                bits = np.pad(bits, (0, bits_needed - len(bits)))
            else:
                bits = bits[:bits_needed]
            # Reshape bits to match frame dimensions
            frame = bits.reshape(resolution[1], resolution[0]) * 255

            frame = cv2.resize(frame, upscaled_resolution, interpolation=cv2.INTER_NEAREST)

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Write frame
            video_writer.write(frame)
            pbar.update(chunk_size)
finally:
    # Clean up
    video_writer.release()
    in_wav.close()
