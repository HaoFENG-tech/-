Offline Voice Recognition Tool Open Source Sharing
I. Introduction
This project is a tool that integrates voice extraction, speaker separation, and high-performance offline speech-to-text functionality. Written in Python, it combines several powerful open-source libraries such as faster-whisper, demucs, pyannote.audio, and speechbrain. It can process audio files and output transcription results with speaker information. Additionally, the project provides a simple and user-friendly GUI interface for easy operation.
II. Project Features
Voice Extraction: Supports professional voice/accompaniment separation using the Demucs model. It also provides the Librosa HPSS method (not recommended).
Speaker Separation: Uses pyannote.audio to separate speakers and identify the time periods of different speakers in the audio.
Speech Recognition: Utilizes faster-whisper for high-performance offline speech-to-text conversion, supporting multiple languages.
GUI Interface: Offers a simple graphical user interface that allows users to select audio files, set parameters, and view recognition results.
III. Code Structure
Main Classes and Functions
VoiceExtractor Class: Integrates voice extraction, speaker separation, and speech recognition functions.
__init__: Initializes and loads the faster-whisper model and the speaker separation model.
extract_vocals_hpss: Extracts vocals using Librosa's HPSS method.
extract_vocals_demucs: Performs professional voice extraction using the Demucs model.
perform_speaker_diarization: Executes speaker separation.
extract_speaker_segments: Extracts audio segments of each speaker based on the speaker separation results.
speech_to_text_with_speakers: Performs speaker separation and speech recognition, returning transcription results with speaker information.
speech_to_text: Converts an audio file to text using the local faster-whisper model.
save_transcript_with_speakers Function: Saves the recognition text with speaker information to a file.
AudioRecorder Class: An audio recording class that supports starting and stopping recording.
VoiceRecognitionApp Class: The GUI interface of the voice recognition application, providing functions such as model settings, audio input, and processing mode selection.
IV. Usage Instructions
1. Environment Configuration
Before running the code, some necessary configurations are required:

python
运行
# 1. 【Mandatory】Set HTTP/HTTPS proxy
# !!! Please replace "http://127.0.0.1:7890" with your own proxy server address and port !!!
# This step is crucial for your Python script to connect to Hugging Face.
# If you don't need a proxy, you can comment out these two lines.
proxy_url = "http://127.0.0.1:7899"  # <-- Enter your proxy address here
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

# 2. 【Optional but Recommended】Comment out or delete the mirror settings because we are now accessing the official site directly through the proxy.
# If the proxy access is not smooth, you can uncomment the following line and comment out the above proxy settings to try the mirror.
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# 3. 【Mandatory】Use your token directly here to ensure the program can pass the verification.
# !!! Please replace "hf_xxxxxxxx..." with your own Hugging Face access token !!!
self.diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"  # <-- Enter your Token here
)
2. Install Dependencies
Run the following command to install the required libraries:

bash
pip install librosa soundfile torch faster-whisper demucs pyannote.audio speechbrain pyaudio tkinter
3. Run the Program
Run the following command in the terminal to start the GUI interface:

bash
python Voice.py
4. GUI Operations
Model Settings: Select the size of the faster-whisper model and the recognition language.
Audio Input: You can choose an existing audio file or use the recording function to record audio.
Processing Modes: Supports four processing modes, including speech recognition only, speaker separation + recognition, Demucs voice extraction + speaker separation + recognition, and HPSS voice extraction + speaker separation + recognition.
Start Processing: Click the button to start processing the audio file.
Save Results: Save the recognition results to a file.
V. Notes
Network Connection: The models need to be downloaded on the first run. Please ensure a stable network connection.
Hugging Face Token: A valid Hugging Face access token is required to use the speaker separation function.
Dependency Installation: Make sure all the required libraries are installed correctly.
VI. Summary
This project provides a powerful offline voice recognition solution that supports multiple functions and processing modes. Through a simple GUI interface, users can easily process audio files and obtain accurate transcription results. We hope this project will be helpful to you!