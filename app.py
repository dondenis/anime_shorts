import os
import uuid
import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename

from transformers import pipeline
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from moviepy.video.fx.all import crop

# ----------------------------------
# Configuration
# ----------------------------------
UPLOAD_FOLDER = "uploads"
VIDEO_FOLDER = "videos"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Video configuration
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
IMAGE_DURATION = 7.5  # seconds per image, so 4 x 7.5 = 30 sec
FPS = 24

# TTS model configuration (using a recent open-source TTS model from ESPnet)
TTS_MODEL = "microsoft/speecht5_tts"

# Ensure directories exist.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Initialize the Flask application.
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "your-secret-key"  # Change to a secure random key for production.

# ----------------------------------
# Helper Functions
# ----------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_script(anime, season, character, custom_prompt):
    """
    Fallback: Generate a basic story script using string formatting.
    """
    script = (
        f"In the captivating world of {anime}, during season {season}, "
        f"we follow the journey of {character}. {custom_prompt} "
        "Experience the heartfelt battles and magical moments unfold as destiny calls."
    )
    return script

def generate_llm_script(anime, season, character, custom_prompt):
    """
    Generates a 30-second long story script using a text-generation LLM.
    The prompt instructs the model to create an engaging narrative suitable for a 30-second anime short video.
    """
    prompt_text = (
        f"Generate a compelling, creative story script for a 30-second anime short video. "
        f"The story should be engaging and approximately 30 seconds long when read aloud. "
        f"Include the following details: Anime: {anime}, Season: {season}, Character: {character}. "
        f"Additional context: {custom_prompt}."
    )
    # Use a text-generation model; here we use 'gpt2-medium'. Adjust max_length as needed.
    generator = pipeline("text-generation", model="gpt2-medium")
    result = generator(prompt_text, max_length=150, num_return_sequences=1)
    generated_text = result[0]['generated_text']
    # Remove the prompt_text from the beginning if present
    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text):].strip()
    return generated_text

def generate_voiceover(voice_script, output_audio):
    """
    Uses the Hugging Face TTS pipeline to generate a voice-over audio file from the provided voice_script.
    """
    tts = pipeline("text-to-speech", model=TTS_MODEL, framework="pt")
    output = tts(voice_script)
    audio_array = output["array"]
    sampling_rate = output["sampling_rate"]
    sf.write(output_audio, np.array(audio_array), sampling_rate)
    return output_audio

def create_pan_clip(image_path, duration=IMAGE_DURATION, resolution=(VIDEO_WIDTH, VIDEO_HEIGHT)):
    """
    Creates an ImageClip with a horizontal pan effect.
    The image is assumed to be larger than the target resolution.
    """
    clip = ImageClip(image_path)
    iw, ih = clip.size

    if iw < resolution[0] or ih < resolution[1]:
        raise ValueError(f"Image {image_path} is smaller than the required resolution {resolution}")

    # Define horizontal pan from left (0) to right (max shift) over the clip's duration.
    x1 = lambda t: int((iw - resolution[0]) * (t / duration))
    x2 = lambda t: int((iw - resolution[0]) * (t / duration)) + resolution[0]
    y1 = int((ih - resolution[1]) / 2)
    y2 = y1 + resolution[1]

    pan_clip = clip.fx(crop, x1=x1, x2=x2, y1=y1, y2=y2)
    pan_clip = pan_clip.set_duration(duration)
    return pan_clip

def create_video(image_paths, audio_file, output_video):
    """
    Assembles the 4 images (with pan effects) and audio into a single video.
    """
    clips = []
    for path in image_paths:
        clip = create_pan_clip(path)
        clips.append(clip)
    video_clip = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_file)
    video_with_audio = video_clip.set_audio(audio.set_duration(video_clip.duration))
    video_with_audio.write_videofile(output_video, fps=FPS)
    return output_video

# ----------------------------------
# Flask Routes
# ----------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check for the file part.
        if "images" not in request.files:
            flash("No file part in the request")
            return redirect(request.url)
        
        files = request.files.getlist("images")
        
        # Ensure exactly 4 images are uploaded.
        if len(files) != 4:
            flash("Please upload exactly 4 images.")
            return redirect(request.url)
        
        # Validate and save files.
        saved_image_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
                file.save(file_path)
                saved_image_paths.append(file_path)
            else:
                flash("One or more files are not valid images.")
                return redirect(request.url)
        
        # Get form data for story generation.
        anime = request.form.get("anime", "Unknown Anime")
        season = request.form.get("season", "Unknown Season")
        character = request.form.get("character", "Unknown Character")
        prompt = request.form.get("prompt", "")
        
        # Generate the script using the LLM-based function.
        script_text = generate_llm_script(anime, season, character, prompt)
        
        # Generate voice-over audio.
        audio_filename = f"{uuid.uuid4().hex}_voice.wav"
        audio_output_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        generate_voiceover(script_text, audio_output_path)
        
        # Create video file.
        video_filename = f"{uuid.uuid4().hex}_anime_short.mp4"
        video_output_path = os.path.join(VIDEO_FOLDER, video_filename)
        try:
            create_video(saved_image_paths, audio_output_path, video_output_path)
        except Exception as e:
            flash(f"Error while creating video: {str(e)}")
            return redirect(request.url)
        
        # Once the video is ready, serve it for download.
        return send_file(video_output_path, as_attachment=True)
    
    return render_template("index.html")

# ----------------------------------
# Run the Application
# ----------------------------------
if __name__ == "__main__":
    app.run(debug=True)