import os
import librosa
import soundfile as sf
import tempfile
import torch
from faster_whisper import WhisperModel
from typing import Optional, List, Dict, Tuple
import numpy as np
from datetime import timedelta
import threading
import wave
import pyaudio
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import time

# ===================== 修改部分 START =====================

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

# ===================== 修改部分 END =======================


# 尝试导入 demucs
try:
    from demucs.separate import separate_sources
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# 尝试导入 pyannote.audio 用于说话人分离
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

print("PYANNOTE_AVAILABLE", PYANNOTE_AVAILABLE)
# 尝试导入 speechbrain 用于说话人识别
try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
print("SPEECHBRAIN_AVAILABLE", SPEECHBRAIN_AVAILABLE)


class VoiceExtractor:
    """
    集成了人声提取、说话人分离和使用 faster-whisper 进行高性能离线语音转文字的工具类。
    """
    def __init__(self, model_size: str = "base"):
        """
        初始化并加载 faster-whisper 模型和说话人分离模型。
        """
        # 初始化 Whisper 模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"正在加载 Whisper 模型 '{model_size}'... (设备: {device}, 计算类型: {compute_type})")
        print("首次运行需要下载模型，请耐心等待")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("faster-whisper 模型加载成功。")
        except Exception as e:
            print(f"错误：加载 faster-whisper 模型失败 - {e}")
            print("请检查网络连接或相关依赖 (torch, faster-whisper, CTranslate2) 是否正确安装。")
            raise
        
        # 初始化说话人分离模型
        self.diarization_pipeline = None
        self.speaker_recognition = None
        
        if PYANNOTE_AVAILABLE:
            try:
                print("正在加载说话人分离模型...")
                # ===================== 修改部分 START =====================
                # 3. 【强制】在这里直接使用你的令牌，确保程序能验证通过。
                # !!! 请务必将 "hf_xxxxxxxx..." 替换为您自己的Hugging Face访问令牌 !!!
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token="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"  # <-- 在这里填入你的Token
                )
                # ===================== 修改部分 END =======================
                print("说话人分离模型加载成功。")
            except Exception as e:
                print(f"警告：说话人分离模型加载失败 - {e}")
                print("请确保已申请 Hugging Face 访问令牌并设置 use_auth_token 参数。")
                self.diarization_pipeline = None
        
        if SPEECHBRAIN_AVAILABLE:
            try:
                print("正在加载说话人识别模型...")
                self.speaker_recognition = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb"
                )
                print("说话人识别模型加载成功。")
            except Exception as e:
                print(f"警告：说话人识别模型加载失败 - {e}")
                self.speaker_recognition = None

    def extract_vocals_hpss(self, audio_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        【不推荐】使用 Librosa 的 HPSS (谐波-打击乐分离) 提取人声。
        """
        print("警告：正在使用 HPSS 方法提取人声，此方法可能严重影响识别准确率。")
        print("开始提取人声（谐波部分）...")
        try:
            y, sr_rate = librosa.load(audio_path, sr=None)
            if y.ndim > 1:
                y = librosa.to_mono(y)
            y_harmonic, _ = librosa.effects.hpss(y)
            
            if output_path is None:
                temp_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                output_path = temp_f.name
                temp_f.close()

            sf.write(output_path, y_harmonic, sr_rate)
            print(f"人声（谐波部分）已保存到: {output_path}")
            return output_path
        except Exception as e:
            print(f"错误：HPSS 人声提取失败 - {e}")
            return None

    def extract_vocals_demucs(self, audio_path: str, output_dir: str) -> Optional[str]:
        """
        【推荐】使用 Demucs 模型进行专业的人声/伴奏分离。
        """
        if not DEMUCS_AVAILABLE:
            print("错误：Demucs 库未安装。请运行 'pip install demucs' 来启用此功能。")
            return None
        
        print("开始使用 Demucs 进行专业人声提取（这可能需要一些时间）...")
        try:
            separated_sources = separate_sources(
                [audio_path],
                out_dir=output_dir,
                model="htdemucs_ft",
            )
            
            source_basename = os.path.splitext(os.path.basename(audio_path))[0]
            vocals_path = os.path.join(output_dir, "htdemucs_ft", source_basename, "vocals.wav")

            if os.path.exists(vocals_path):
                print(f"专业人声提取成功，文件已保存到: {vocals_path}")
                return vocals_path
            else:
                print(f"错误：Demucs 处理完成，但未找到预期的人声文件: {vocals_path}")
                return None
        except Exception as e:
            print(f"错误：Demucs 人声提取失败 - {e}")
            return None

    def perform_speaker_diarization(self, audio_path: str) -> Optional[Annotation]:
        """
        执行说话人分离，识别音频中不同说话人的时间段。
        
        Args:
            audio_path (str): 音频文件路径
            
        Returns:
            Optional[Annotation]: 说话人分离结果
        """
        if not self.diarization_pipeline:
            print("错误：说话人分离模型未加载。请检查 pyannote.audio 安装和访问令牌设置。")
            return None
        
        print("正在执行说话人分离...")
        try:
            diarization = self.diarization_pipeline(audio_path)
            
            print(f"检测到 {len(diarization.labels())} 个不同的说话人")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"说话人 {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")
            
            return diarization
        except Exception as e:
            print(f"错误：说话人分离失败 - {e}")
            return None

    def extract_speaker_segments(self, audio_path: str, diarization: Annotation) -> Dict[str, List[Tuple[float, float, str]]]:
        """
        根据说话人分离结果提取各个说话人的音频段。
        
        Args:
            audio_path (str): 原始音频文件路径
            diarization (Annotation): 说话人分离结果
            
        Returns:
            Dict[str, List[Tuple[float, float, str]]]: 每个说话人的时间段和对应的音频文件路径
        """
        print("正在提取各说话人的音频段...")
        
        # 加载原始音频
        y, sr = librosa.load(audio_path, sr=None)
        
        speaker_segments = {}
        temp_dir = tempfile.mkdtemp()
        
        try:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_sample = int(turn.start * sr)
                end_sample = int(turn.end * sr)
                
                # 提取该时间段的音频
                segment_audio = y[start_sample:end_sample]
                
                # 保存为临时文件
                segment_filename = f"speaker_{speaker}_{turn.start:.2f}_{turn.end:.2f}.wav"
                segment_path = os.path.join(temp_dir, segment_filename)
                sf.write(segment_path, segment_audio, sr)
                
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                
                speaker_segments[speaker].append((turn.start, turn.end, segment_path))
                
        except Exception as e:
            print(f"错误：提取说话人音频段失败 - {e}")
            return {}
        
        return speaker_segments

    def speech_to_text_with_speakers(self, audio_path: str, language: str = 'zh', beam_size: int = 5) -> Optional[Dict]:
        """
        使用说话人分离和语音识别，返回带有说话人信息的转录结果。
        
        Args:
            audio_path (str): 音频文件路径
            language (str): 指定识别语言的ISO 639-1代码
            beam_size (int): Beam search 的大小
            
        Returns:
            Optional[Dict]: 包含说话人信息的识别结果
        """
        # 执行说话人分离
        diarization = self.perform_speaker_diarization(audio_path)
        if not diarization:
            print("说话人分离失败，回退到普通语音识别模式...")
            text = self.speech_to_text(audio_path, language, beam_size)
            return {"full_text": text, "speakers": None}
        
        # 提取说话人音频段
        speaker_segments = self.extract_speaker_segments(audio_path, diarization)
        if not speaker_segments:
            print("提取说话人音频段失败，回退到普通语音识别模式...")
            text = self.speech_to_text(audio_path, language, beam_size)
            return {"full_text": text, "speakers": None}
        
        # 对每个说话人的音频段进行语音识别
        results = {
            "speakers": {},
            "timeline": [],
            "full_text": ""
        }
        
        all_segments = []
        
        print("\n开始对各说话人进行语音识别...")
        for speaker, segments in speaker_segments.items():
            print(f"\n=== 处理说话人 {speaker} ===")
            speaker_texts = []
            
            for start_time, end_time, segment_path in segments:
                try:
                    # 对该段音频进行语音识别
                    segments_iter, info = self.model.transcribe(segment_path, language=language, beam_size=beam_size)
                    
                    segment_text = ""
                    for segment in segments_iter:
                        segment_text += segment.text
                    
                    if segment_text.strip():
                        speaker_texts.append(segment_text.strip())
                        all_segments.append({
                            "speaker": speaker,
                            "start": start_time,
                            "end": end_time,
                            "text": segment_text.strip()
                        })
                        
                        print(f"[{start_time:.2f}s - {end_time:.2f}s] {speaker}: {segment_text.strip()}")
                    
                except Exception as e:
                    print(f"错误：识别说话人 {speaker} 的音频段失败 - {e}")
                    continue
            
            results["speakers"][speaker] = speaker_texts
        
        # 按时间顺序排序所有段落
        all_segments.sort(key=lambda x: x["start"])
        results["timeline"] = all_segments
        
        # 生成完整文本
        full_text_parts = []
        for segment in all_segments:
            full_text_parts.append(f"[{segment['speaker']}] {segment['text']}")
        
        results["full_text"] = "\n".join(full_text_parts)
        
        return results

    def speech_to_text(self, audio_path: str, language: str = 'zh', beam_size: int = 5) -> Optional[str]:
        """
        使用本地 faster-whisper 模型将音频文件转换为文字。
        """
        print(f"正在使用 faster-whisper 进行离线识别: {audio_path}")
        if not os.path.exists(audio_path):
            print(f"错误：文件不存在 - {audio_path}")
            return None
            
        try:
            segments, info = self.model.transcribe(audio_path, language=language, beam_size=beam_size)
            
            print(f"检测到语言 '{info.language}'，置信度: {info.language_probability:.2f}")
            print(f"音频时长: {info.duration:.2f} 秒")
            
            full_text = []
            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                full_text.append(segment.text)
            
            recognized_text = "\n".join(full_text)
            print("\n识别完成。")
            return recognized_text
            
        except Exception as e:
            print(f"错误：faster-whisper 识别失败 - {e}")
            return None


def save_transcript_with_speakers(results: Dict, original_audio_path: str):
    """将带有说话人信息的识别文本保存到文件中"""
    if not results or not results.get("full_text"):
        print("没有可保存的文本。")
        return
    
    base_name = os.path.splitext(os.path.basename(original_audio_path))[0]
    output_dir = os.path.dirname(original_audio_path)
    
    # 保存完整转录文本
    full_transcript_path = os.path.join(output_dir, f"{base_name}_transcript_with_speakers.txt")
    with open(full_transcript_path, 'w', encoding='utf-8') as f:
        f.write("=== 完整转录结果（按时间顺序）===\n\n")
        f.write(results["full_text"])
        
        if results.get("speakers"):
            f.write("\n\n=== 按说话人分组 ===\n\n")
            for speaker, texts in results["speakers"].items():
                f.write(f"说话人 {speaker}:\n")
                for i, text in enumerate(texts, 1):
                    f.write(f"  {i}. {text}\n")
                f.write("\n")
        
        if results.get("timeline"):
            f.write("\n=== 详细时间轴 ===\n\n")
            for segment in results["timeline"]:
                f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n")
    
    print(f"\n结果已保存到: {full_transcript_path}")


class AudioRecorder:
    """音频录制类"""
    def __init__(self):
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.thread = None
        
    def start_recording(self, filename="recording.wav", channels=1, rate=44100, chunk=1024):
        """开始录音"""
        self.recording = True
        self.filename = filename
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.frames = []
        
        # 启动录音线程
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        
    def _record(self):
        """录音线程内部实现"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        while self.recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
            
        self.stream.stop_stream()
        self.stream.close()
        
        # 保存录音文件
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
    def stop_recording(self):
        """停止录音"""
        self.recording = False
        if self.thread:
            self.thread.join()
        return self.filename


class VoiceRecognitionApp(tk.Tk):
    """语音识别应用的GUI界面"""
    def __init__(self):
        super().__init__()
        self.title("离线语音识别工具")
        self.geometry("900x700")  # 增大窗口尺寸
        self.recorder = AudioRecorder()
        self.voice_extractor = None
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self, padding=(10, 5))
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板（设置区域）
        left_frame = ttk.Frame(main_frame, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # --- 模型设置区域 ---
        model_frame = ttk.LabelFrame(left_frame, text="模型设置", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # 模型选择
        ttk.Label(model_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.model_var = tk.StringVar(value="base")
        models = ["tiny", "base", "small", "medium", "large-v2"]
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=models, state="readonly", width=10)
        model_combo.grid(row=0, column=1, sticky=tk.W, pady=3)
        
        # 语言选择
        ttk.Label(model_frame, text="识别语言:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.lang_var = tk.StringVar(value="zh")
        lang_combo = ttk.Combobox(model_frame, textvariable=self.lang_var, values=["zh", "en", "ja", "auto"], width=10)
        lang_combo.grid(row=1, column=1, sticky=tk.W, pady=3)
        
        # --- 音频输入区域 ---
        input_frame = ttk.LabelFrame(left_frame, text="音频输入", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        
        # 文件选择
        file_row = ttk.Frame(input_frame)
        file_row.pack(fill=tk.X, pady=3)
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(file_row, textvariable=self.file_var)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        browse_btn = ttk.Button(file_row, text="浏览", width=6, command=self.browse_file)
        browse_btn.pack(side=tk.RIGHT)
        
        # 录音控制
        rec_row = ttk.Frame(input_frame)
        rec_row.pack(fill=tk.X, pady=3)
        self.rec_btn = ttk.Button(rec_row, text="开始录音", width=10, command=self.toggle_recording)
        self.rec_btn.pack(side=tk.LEFT)
        
        self.rec_status = tk.StringVar(value="")
        rec_status = ttk.Label(rec_row, textvariable=self.rec_status, width=10)
        rec_status.pack(side=tk.LEFT, padx=5)
        
        self.rec_time = tk.StringVar(value="00:00")
        rec_time = ttk.Label(rec_row, textvariable=self.rec_time, width=8)
        rec_time.pack(side=tk.RIGHT)
        
        # --- 处理模式区域 ---
        mode_frame = ttk.LabelFrame(left_frame, text="处理模式", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)

        # 定义处理模式选项
        modes = [
            ("仅语音识别", "1"),
            ("说话人分离+识别", "2"),
            ("Demucs人声提取+说话人分离+识别", "3"),
            ("HPSS人声提取+说话人分离+识别", "4"),
        ]
        self.mode_var = tk.StringVar(value="1")

        # 垂直排列选项
        for text, value in modes:
            frame = ttk.Frame(mode_frame)
            frame.pack(fill=tk.X, pady=2)
            rb = ttk.Radiobutton(frame, text=text, variable=self.mode_var, value=value)
            rb.pack(side=tk.LEFT, anchor=tk.W)
        
        # --- 按钮区域 ---
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        process_btn = ttk.Button(btn_frame, text="开始处理", command=self.process_audio, width=12)
        process_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        save_btn = ttk.Button(btn_frame, text="保存结果", command=self.save_results, width=12)
        save_btn.pack(side=tk.RIGHT)
        
        # ====== 重点修改：大幅放大识别结果区域 ======
        # 创建结果框架 - 占据主窗口右侧空间
        result_frame = ttk.LabelFrame(
            main_frame, 
            text="识别结果", 
            padding=10
        )
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建更大的文本区域
        self.result_text = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD,
            font=("Microsoft YaHei", 12),  # 增大字体
            padx=15,
            pady=15,
            height=30  # 增加行数
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 添加分割线视觉提示
        ttk.Separator(main_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # --- 状态栏 ---
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def browse_file(self):
        """浏览选择音频文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("音频文件", "*.wav *.mp3 *.flac"), ("所有文件", "*.*")]
        )
        if file_path:
            self.file_var.set(file_path)
            self.status_var.set(f"已选择文件: {os.path.basename(file_path)}")
    
    def toggle_recording(self):
        """开始/停止录音"""
        if not self.recorder.recording:
            # 开始录音
            if not self.file_var.get():
                self.file_var.set("recording.wav")
            
            self.recorder.start_recording(self.file_var.get())
            self.rec_btn.config(text="停止录音")
            self.rec_status.set("录音中...")
            self.record_start_time = time.time()
            self.update_recording_time()
        else:
            # 停止录音
            self.recorder.stop_recording()
            self.rec_btn.config(text="开始录音")
            self.rec_status.set("录音已保存")
            self.status_var.set(f"录音已保存: {self.file_var.get()}")
    
    def update_recording_time(self):
        """更新录音时间显示"""
        if self.recorder.recording:
            elapsed = int(time.time() - self.record_start_time)
            mins, secs = divmod(elapsed, 60)
            self.rec_time.set(f"{mins:02d}:{secs:02d}")
            self.after(1000, self.update_recording_time)
    
    def process_audio(self):
        """处理音频"""
        audio_file = self.file_var.get()
        if not os.path.exists(audio_file):
            messagebox.showerror("错误", "请先选择或录制音频文件")
            return
        
        if not self.voice_extractor:
            try:
                self.status_var.set("正在加载模型...")
                self.update()
                self.voice_extractor = VoiceExtractor(model_size=self.model_var.get())
            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
                self.status_var.set("模型加载失败")
                return
        
        # 在后台线程中处理
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "处理中，请稍候...\n")
        self.status_var.set("处理中...")
        
        thread = threading.Thread(target=self._process_audio, args=(audio_file,))
        thread.daemon = True
        thread.start()
    
    def _process_audio(self, audio_file):
        """后台处理音频"""
        try:
            mode = self.mode_var.get()
            lang = self.lang_var.get()
            
            if mode == "1":
                # 直接识别
                text = self.voice_extractor.speech_to_text(audio_file, language=lang)
                result = {"full_text": text, "speakers": None}
            elif mode == "2":
                # 说话人分离+识别
                result = self.voice_extractor.speech_to_text_with_speakers(audio_file, language=lang)
            elif mode == "3":
                # Demucs提取人声+说话人分离+识别
                output_dir = os.path.dirname(audio_file)
                extracted_path = self.voice_extractor.extract_vocals_demucs(audio_file, output_dir)
                if extracted_path:
                    result = self.voice_extractor.speech_to_text_with_speakers(extracted_path, language=lang)
                else:
                    result = None
            elif mode == "4":
                # HPSS提取人声+说话人分离+识别
                vocals_filename = os.path.splitext(os.path.basename(audio_file))[0] + '_vocals_hpss.wav'
                vocals_path = os.path.join(os.path.dirname(audio_file), vocals_filename)
                extracted_path = self.voice_extractor.extract_vocals_hpss(audio_file, vocals_path)
                if extracted_path:
                    result = self.voice_extractor.speech_to_text_with_speakers(extracted_path, language=lang)
                else:
                    result = None
            
            # 更新UI显示结果
            self.after(0, self.show_results, result, audio_file)
            self.status_var.set("处理完成")
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("错误", f"处理失败: {str(e)}"))
            self.status_var.set("处理失败")
    
    def show_results(self, result, audio_file):
        """在UI中显示结果"""
        if not result or not result.get("full_text"):
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "处理未产生任何结果。")
            return
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "=== 完整识别结果 ===\n\n")
        self.result_text.insert(tk.END, result["full_text"])
        
        if result.get("speakers"):
            self.result_text.insert(tk.END, "\n\n=== 按说话人分组 ===\n\n")
            for speaker, texts in result["speakers"].items():
                self.result_text.insert(tk.END, f"说话人 {speaker}:\n")
                for i, text in enumerate(texts, 1):
                    self.result_text.insert(tk.END, f"  {i}. {text}\n")
                self.result_text.insert(tk.END, "\n")
        
        self.result = result
        self.audio_file = audio_file
        self.status_var.set("处理完成")
    
    def save_results(self):
        """保存结果到文件"""
        if hasattr(self, 'result') and hasattr(self, 'audio_file'):
            save_transcript_with_speakers(self.result, self.audio_file)
            messagebox.showinfo("保存成功", f"结果已保存到: {os.path.dirname(self.audio_file)}")
        else:
            messagebox.showwarning("警告", "没有可保存的结果")


if __name__ == "__main__":
    # 创建并运行UI应用
    app = VoiceRecognitionApp()
    app.mainloop()