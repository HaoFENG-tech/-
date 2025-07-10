离线语音识别工具开源分享
一、简介
本项目是一个集成了人声提取、说话人分离和高性能离线语音转文字功能的工具。它使用 Python 编写，结合了多个强大的开源库，如 faster-whisper、demucs、pyannote.audio 和 speechbrain，可以对音频文件进行处理，并输出带有说话人信息的转录结果。此外，项目还提供了一个简单易用的 GUI 界面，方便用户操作。
二、项目功能
人声提取：支持使用 Demucs 模型进行专业的人声 / 伴奏分离，也提供了 Librosa 的 HPSS 方法（不推荐）。
说话人分离：使用 pyannote.audio 进行说话人分离，识别音频中不同说话人的时间段。
语音识别：使用 faster-whisper 进行高性能离线语音转文字，支持多种语言。
GUI 界面：提供一个简单的图形用户界面，方便用户选择音频文件、设置参数和查看识别结果。
三、代码结构
主要类和函数
VoiceExtractor 类：集成了人声提取、说话人分离和语音识别功能。
__init__：初始化并加载 faster-whisper 模型和说话人分离模型。
extract_vocals_hpss：使用 Librosa 的 HPSS 方法提取人声。
extract_vocals_demucs：使用 Demucs 模型进行专业的人声提取。
perform_speaker_diarization：执行说话人分离。
extract_speaker_segments：根据说话人分离结果提取各个说话人的音频段。
speech_to_text_with_speakers：使用说话人分离和语音识别，返回带有说话人信息的转录结果。
speech_to_text：使用本地 faster-whisper 模型将音频文件转换为文字。
save_transcript_with_speakers 函数：将带有说话人信息的识别文本保存到文件中。
AudioRecorder 类：音频录制类，支持开始和停止录音。
VoiceRecognitionApp 类：语音识别应用的 GUI 界面，提供了模型设置、音频输入、处理模式选择等功能。
四、使用方法
1. 环境配置
在运行代码之前，需要进行一些必要的配置：

python
运行
# 1. 【强制】设置HTTP/HTTPS代理
# !!! 请务必将 "http://127.0.0.1:7890" 替换成您自己的代理服务器地址和端口 !!!
# 这一步是让你的Python脚本能连接到Hugging Face的关键。
# 如果您不需要代理，可以注释掉这两行。
proxy_url = "http://127.0.0.1:7899"  # <-- 在这里填入你的代理地址
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

# 2. 【可选但推荐】注释掉或删除镜像设置，因为我们现在通过代理直接访问官方。
# 如果代理访问不畅，可以取消注释下面这行，并注释掉上面的代理设置，尝试镜像。
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# 3. 【强制】在这里直接使用你的令牌，确保程序能验证通过。
# !!! 请务必将 "hf_xxxxxxxx..." 替换为您自己的Hugging Face访问令牌 !!!
self.diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"  # <-- 在这里填入你的Token
)
2. 安装依赖
运行以下命令安装所需的库：

bash
pip install librosa soundfile torch faster-whisper demucs pyannote.audio speechbrain pyaudio tkinter
3. 运行程序
在终端中运行以下命令启动 GUI 界面：

bash
python Voice.py
4. GUI 操作
模型设置：选择 faster-whisper 模型的大小和识别语言。
音频输入：可以选择已有的音频文件，也可以使用录音功能录制音频。
处理模式：支持四种处理模式，包括仅语音识别、说话人分离 + 识别、Demucs 人声提取 + 说话人分离 + 识别、HPSS 人声提取 + 说话人分离 + 识别。
开始处理：点击按钮开始处理音频文件。
保存结果：将识别结果保存到文件中。
五、注意事项
网络连接：首次运行需要下载模型，请确保网络连接正常。
Hugging Face 令牌：使用说话人分离功能时，需要提供有效的 Hugging Face 访问令牌。
依赖安装：请确保所有依赖库都已正确安装。
六、总结
本项目提供了一个强大的离线语音识别解决方案，支持多种功能和处理模式。通过简单的 GUI 界面，用户可以方便地对音频文件进行处理，并获得准确的转录结果。希望这个项目能对大家有所帮助！
