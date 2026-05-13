"""
Video ReEncoder - 视频批量压缩工具
Copyright (C) 2024 Edgar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import ffmpeg
from tqdm import tqdm


@dataclass
class SampleSegment:
    """采样片段信息"""
    start_time: float  # 开始时间（秒）
    duration: float    # 持续时间（秒）
    
@dataclass
class SampleResult:
    """采样结果"""
    segment: SampleSegment
    original_size: int   # 原始片段大小（字节）
    compressed_size: int # 压缩后大小（字节）
    compression_ratio: float  # 压缩率 = compressed / original
    success: bool        # 是否成功
    error_msg: str = ""  # 错误信息


class CRFSampleEstimator:
    """CRF多点采样预估器
    
    用于在正式转码前，对视频部分片段进行真实CRF编码采样，
    预估目标CRF下的大致压缩收益。
    """
    
    def __init__(self, crf: int, codec: str = 'h264', use_gpu: bool = True,
                 min_benefit_threshold: float = 0.10, max_sample_count: int = 5,
                 sample_duration_base: float = 20.0):
        """
        初始化采样预估器
        
        Args:
            crf: 目标CRF值
            codec: 编码器类型
            use_gpu: 是否使用GPU加速
            min_benefit_threshold: 最小收益阈值（默认10%），低于此值则跳过
            max_sample_count: 最大采样点数（默认5个）
            sample_duration_base: 基础采样时长（默认20秒）
        """
        self.crf = crf
        self.codec = codec
        self.use_gpu = use_gpu
        self.min_benefit_threshold = min_benefit_threshold
        self.max_sample_count = max_sample_count
        self.sample_duration_base = sample_duration_base
        self.ffmpeg_dir = Path(__file__).parent / 'ffmpeg_bin'
    
    def calculate_sample_plan(self, duration: float) -> List[SampleSegment]:
        """根据视频时长计算采样计划
        
        Args:
            duration: 视频总时长（秒）
            
        Returns:
            采样片段列表
        """
        segments = []
        
        if duration <= 0:
            return segments
        
        # 短视频特殊处理：小于30秒的视频不采样，直接转码
        if duration < 30:
            return segments
        
        # 动态调整采样策略
        if duration < 60:
            # 30-60秒：单点采样，时长10秒
            sample_count = 1
            sample_duration = min(10.0, duration * 0.3)
        elif duration < 180:
            # 1-3分钟：2点采样，时长15秒
            sample_count = 2
            sample_duration = min(15.0, duration * 0.2)
        elif duration < 600:
            # 3-10分钟：3点采样，时长20秒
            sample_count = 3
            sample_duration = min(20.0, duration * 0.15)
        elif duration < 1800:
            # 10-30分钟：4点采样，时长25秒
            sample_count = 4
            sample_duration = min(25.0, duration * 0.1)
        else:
            # 30分钟以上：5点采样，时长30秒
            sample_count = min(self.max_sample_count, 5)
            sample_duration = min(30.0, duration * 0.05)
        
        # 计算采样时间点（均匀分布，避免边缘）
        margin = sample_duration * 0.5  # 距离边缘的缓冲
        usable_duration = duration - 2 * margin
        
        if usable_duration < sample_duration:
            # 可用时长不足，只采中间一点
            start_time = (duration - sample_duration) / 2
            segments.append(SampleSegment(start_time=start_time, duration=sample_duration))
        else:
            # 均匀分布采样点
            for i in range(sample_count):
                if sample_count == 1:
                    # 单点：放在中间
                    position = 0.5
                else:
                    # 多点：均匀分布
                    position = (i + 1) / (sample_count + 1)
                
                start_time = margin + position * usable_duration - sample_duration / 2
                
                # 确保不越界
                start_time = max(0, min(start_time, duration - sample_duration))
                
                segments.append(SampleSegment(start_time=start_time, duration=sample_duration))
        
        return segments
    
    def encode_sample_segment(self, video_path: Path, segment: SampleSegment, 
                              output_path: Path) -> Tuple[int, int, bool, str]:
        """编码单个采样片段
        
        Args:
            video_path: 输入视频路径
            segment: 采样片段信息
            output_path: 输出文件路径
            
        Returns:
            (original_size, compressed_size, success, error_msg)
        """
        try:
            # 获取编码器
            codec_name = self._get_codec_name()
            
            # 构建FFmpeg命令进行片段采样编码
            cmd = [
                str(self.ffmpeg_dir / 'ffmpeg.exe'),
                '-ss', str(segment.start_time),  # 开始时间
                '-i', str(video_path),
                '-t', str(segment.duration),     # 持续时间
                '-c:v', codec_name,
            ]
            
            # 添加CRF参数
            if 'nvenc' in codec_name:
                cmd.extend(['-cq', str(self.crf)])
            elif 'qsv' in codec_name:
                cmd.extend(['-global_quality', str(self.crf)])
            elif 'amf' in codec_name:
                cmd.extend(['-qp_i', str(self.crf), '-qp_p', str(self.crf), '-qp_b', str(self.crf)])
            else:
                cmd.extend(['-crf', str(self.crf)])
            
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '128K',
                '-pix_fmt', 'yuv420p',
                '-loglevel', 'error',
                '-y',
                str(output_path)
            ])
            
            # 执行编码
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120  # 2分钟超时
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')[:200]
                return 0, 0, False, f"FFmpeg错误: {error_msg}"
            
            # 获取文件大小
            if not output_path.exists():
                return 0, 0, False, "输出文件未生成"
            
            compressed_size = output_path.stat().st_size
            
            # 估算原始片段大小（按时间比例）
            full_size = video_path.stat().st_size
            probe = ffmpeg.probe(str(video_path), show_entries='format=duration')
            full_duration = float(probe.get('format', {}).get('duration', segment.duration))
            
            if full_duration > 0:
                original_size = int(full_size * (segment.duration / full_duration))
            else:
                original_size = compressed_size  # 无法估算，假设相同
            
            return original_size, compressed_size, True, ""
            
        except subprocess.TimeoutExpired:
            return 0, 0, False, "采样编码超时"
        except Exception as e:
            return 0, 0, False, str(e)
    
    def _get_codec_name(self) -> str:
        """获取编码器名称"""
        if self.use_gpu:
            if self.codec == 'av1':
                return 'av1_nvenc'  # 简化，实际应该检测
            elif self.codec == 'hevc':
                return 'hevc_nvenc'
            else:
                return 'h264_nvenc'
        else:
            if self.codec == 'av1':
                return 'libaom-av1'
            elif self.codec == 'hevc':
                return 'libx265'
            else:
                return 'libx264'
    
    def estimate_compression_benefit(self, video_path: Path, duration: float) -> Optional[dict]:
        """预估压缩收益
        
        Args:
            video_path: 视频文件路径
            duration: 视频时长（秒）
            
        Returns:
            预估结果字典，包含：
            - estimated_ratio: 预估压缩率
            - estimated_benefit: 预估收益百分比
            - should_skip: 是否应该跳过
            - sample_count: 实际采样数
            - samples: 采样结果列表
            如果无法预估则返回None
        """
        # 计算采样计划
        segments = self.calculate_sample_plan(duration)
        
        # 如果没有采样片段（短视频），返回None表示直接转码
        if not segments:
            return None
        
        print(f"  🔍 开始CRF采样预估（{len(segments)}个采样点）...")
        
        samples = []
        total_original = 0
        total_compressed = 0
        successful_samples = 0
        
        temp_dir = video_path.parent / '.temp_samples'
        temp_dir.mkdir(exist_ok=True)
        
        try:
            for i, segment in enumerate(segments, 1):
                temp_output = temp_dir / f"sample_{video_path.stem}_{i}.mp4"
                
                print(f"    采样 {i}/{len(segments)}: {segment.start_time:.1f}s ~ {segment.start_time + segment.duration:.1f}s", end='')
                
                original_size, compressed_size, success, error_msg = self.encode_sample_segment(
                    video_path, segment, temp_output
                )
                
                if success and original_size > 0:
                    ratio = compressed_size / original_size if original_size > 0 else 1.0
                    benefit = (1 - ratio) * 100
                    
                    total_original += original_size
                    total_compressed += compressed_size
                    successful_samples += 1
                    
                    sample_result = SampleResult(
                        segment=segment,
                        original_size=original_size,
                        compressed_size=compressed_size,
                        compression_ratio=ratio,
                        success=True
                    )
                    samples.append(sample_result)
                    
                    print(f" | 压缩率: {ratio:.2f} (收益: {benefit:.1f}%)")
                else:
                    print(f" | ✗ 失败: {error_msg}")
                    sample_result = SampleResult(
                        segment=segment,
                        original_size=0,
                        compressed_size=0,
                        compression_ratio=1.0,
                        success=False,
                        error_msg=error_msg
                    )
                    samples.append(sample_result)
                
                # 清理临时文件
                if temp_output.exists():
                    temp_output.unlink()
        
        finally:
            # 清理临时目录
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        # 如果没有成功的采样，返回None
        if successful_samples == 0:
            print(f"  ⚠ 所有采样均失败，跳过预估")
            return None
        
        # 计算整体压缩率
        overall_ratio = total_compressed / total_original if total_original > 0 else 1.0
        overall_benefit = (1 - overall_ratio) * 100
        
        # 检查采样结果的离散程度
        if len(samples) > 1:
            ratios = [s.compression_ratio for s in samples if s.success]
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                variance = sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)
                std_dev = variance ** 0.5
                
                # 如果标准差过大，说明视频内容变化大，预估可能不准确
                if std_dev > 0.15:  # 标准差超过15%
                    print(f"  ⚠ 采样结果差异较大 (σ={std_dev:.2f})，预估仅供参考")
        
        should_skip = overall_benefit < self.min_benefit_threshold * 100
        
        result = {
            'estimated_ratio': overall_ratio,
            'estimated_benefit': overall_benefit,
            'should_skip': should_skip,
            'sample_count': successful_samples,
            'samples': samples,
            'total_original_size': total_original,
            'total_compressed_size': total_compressed
        }
        
        if should_skip:
            print(f"  ⏭ 预估收益过低 ({overall_benefit:.1f}%)，将跳过转码")
        else:
            print(f"  ✓ 预估收益: {overall_benefit:.1f}% (压缩率: {overall_ratio:.2f})")
        
        return result


class VideoReEncoder:
    """视频重新编码器，用于批量压缩视频码率"""
    
    VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.wmv', '.m4v'}
    FFMPEG_DIR = Path(__file__).parent / 'ffmpeg_bin'
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None, 
                 target_bitrate: str = '1000K', recursive: bool = False, 
                 use_gpu: bool = True, codec: str = 'h264',
                 copy_skipped: bool = False, crf: Optional[int] = None,
                 min_benefit_threshold: float = -10.0, max_sample_count: int = 5):
        """
        初始化编码器
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径，如果为 None 则生成在源文件同目录下
            target_bitrate: 目标视频码率，如 '1000K', '2M' 等（CRF模式下可选）
            recursive: 是否递归处理子目录
            use_gpu: 是否使用 GPU 硬件加速（默认 True）
            codec: 视频编码器类型 ('h264', 'hevc', 或 'av1')，默认 'h264'
            copy_skipped: 是否将跳过的视频复制到输出目录（默认 False）
            crf: CRF值（恒定质量因子），范围0-51，值越小质量越高。如果设置则优先使用CRF模式
            min_benefit_threshold: CRF采样预估的最小收益阈值（百分比，默认-10%，允许小幅增大以尝试压缩）
            max_sample_count: CRF采样预估的最大采样点数
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.target_bitrate = target_bitrate
        self.recursive = recursive
        self.use_gpu = use_gpu
        self.codec = codec.lower()
        self.copy_skipped = copy_skipped
        self.crf = crf
        
        # 如果使用CRF模式，验证CRF值范围
        if self.crf is not None:
            if not (0 <= self.crf <= 51):
                raise ValueError(f"CRF值必须在0-51范围内，当前值：{self.crf}")
            print(f"✓ 使用CRF模式，CRF值：{self.crf}")
        else:
            print(f"✓ 使用固定码率模式，目标码率：{self.target_bitrate}")
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在：{input_dir}")
        
        self.gpu_encoder = None
        self._ensure_ffmpeg()
        if self.use_gpu:
            self._detect_gpu_encoder()
        
        # 初始化CRF采样预估器（仅在CRF模式下）
        self.sample_estimator = None
        if self.crf is not None:
            self.sample_estimator = CRFSampleEstimator(
                crf=self.crf,
                codec=self.codec,
                use_gpu=self.use_gpu,
                min_benefit_threshold=min_benefit_threshold / 100.0,  # 转换为小数
                max_sample_count=max_sample_count,
                sample_duration_base=20.0
            )
    
    def _download_ffmpeg(self):
        """下载便携版 FFmpeg"""
        print("正在下载 FFmpeg...")
        
        try:
            import urllib.request
            import ssl
            
            # 创建 SSL 不验证上下文（用于下载）
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Windows 平台的 FFmpeg 下载链接（使用 gyan.dev 的构建版本）
            ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            
            self.FFMPEG_DIR.mkdir(exist_ok=True)
            zip_path = self.FFMPEG_DIR / "ffmpeg.zip"
            
            # 下载文件
            with urllib.request.urlopen(ffmpeg_url, context=ssl_context) as response:
                total_size = int(response.getheader('Content-Length', 0))
                downloaded = 0
                
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        f.write(chunk)
                        
                        # 显示下载进度
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r下载进度：{percent:.1f}%", end='', flush=True)
            
            print("\n正在解压 FFmpeg...")
            
            # 解压文件
            with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
                # 找到包含 exe 文件的目录
                for name in zip_ref.namelist():
                    if 'bin/ffmpeg.exe' in name:
                        base_dir = name.split('/')[0]
                        break
                
                # 提取 ffmpeg.exe、ffprobe.exe 等文件
                for name in zip_ref.namelist():
                    if name.startswith(f"{base_dir}/bin/") and name.endswith('.exe'):
                        zip_ref.extract(name, str(self.FFMPEG_DIR))
                        # 移动到 FFMPEG_DIR 根目录
                        extracted_path = self.FFMPEG_DIR / name
                        final_path = self.FFMPEG_DIR / Path(name).name
                        if extracted_path != final_path:
                            shutil.move(str(extracted_path), str(final_path))
                
                # 清理多余目录
                for item in self.FFMPEG_DIR.iterdir():
                    if item.is_dir() and item.name != '__pycache__':
                        shutil.rmtree(item)
            
            # 删除压缩包
            zip_path.unlink()
            
            print("✓ FFmpeg 下载完成")
            
        except Exception as e:
            print(f"\n✗ FFmpeg 下载失败：{e}")
            raise RuntimeError(f"无法下载 FFmpeg: {e}")
    
    def _ensure_ffmpeg(self):
        """确保 FFmpeg 可用，如果不可用则下载"""
        ffmpeg_exe = self.FFMPEG_DIR / 'ffmpeg.exe'
        ffprobe_exe = self.FFMPEG_DIR / 'ffprobe.exe'
        
        if not ffmpeg_exe.exists() or not ffprobe_exe.exists():
            print("未检测到 FFmpeg，正在下载便携版...")
            self._download_ffmpeg()
        
        # 将 FFmpeg 添加到 PATH
        ffmpeg_path = str(self.FFMPEG_DIR)
        if ffmpeg_path not in os.environ['PATH']:
            os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']
        
        # 验证 FFmpeg 是否可用
        try:
            result = subprocess.run(
                [str(ffmpeg_exe), '-version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg 验证失败")
            print(f"✓ FFmpeg 已就绪：{ffmpeg_exe}")
        except Exception as e:
            print(f"FFmpeg 验证失败：{e}")
            self._download_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查 ffmpeg 二进制文件是否可用"""
        try:
            probe = ffmpeg.probe('test.mp4')
        except Exception as e:
            if 'test.mp4' in str(e) or 'No such file' in str(e):
                pass
            else:
                raise RuntimeError("ffmpeg 不可用，请确保已安装 ffmpeg-python 和 ffmpeg 二进制文件")
    
    def find_video_files(self) -> List[Path]:
        """查找所有视频文件"""
        video_files = []
        
        if self.recursive:
            pattern = '**/*'
        else:
            pattern = '*'
        
        for file_path in self.input_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                video_files.append(file_path)
        
        return sorted(video_files)
    
    def get_audio_bitrate(self, video_path: Path) -> str:
        """获取视频文件的音频码率"""
        try:
            probe = ffmpeg.probe(str(video_path), select_streams='a:0', show_entries='stream=bit_rate')
            streams = probe.get('streams', [])
            
            if streams and len(streams) > 0:
                bit_rate = streams[0].get('bit_rate')
                if bit_rate and bit_rate != 'N/A':
                    return f'{int(bit_rate)}'
            
            return '128K'
        except Exception:
            return '128K'
    
    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）"""
        try:
            # 方法 1：从 format 获取时长
            probe = ffmpeg.probe(str(video_path), show_entries='format=duration')
            duration = float(probe.get('format', {}).get('duration', 0))
            
            if duration > 0 and duration < 86400:  # 合理范围：小于 24 小时
                return duration
            
            # 方法 2：从视频流获取时长
            probe = ffmpeg.probe(str(video_path))
            streams = probe.get('streams', [])
            for stream in streams:
                if stream.get('codec_type') == 'video':
                    # 尝试多个时长字段
                    duration = float(stream.get('duration', 0))
                    if duration <= 0:
                        duration = float(stream.get('tags', {}).get('DURATION', 0))
                    if duration <= 0:
                        duration = float(stream.get('tags', {}).get('_STATISTICS_WRITING_DATE_UTC', 0))
                    
                    if duration > 0 and duration < 86400:
                        return duration
            
            # 方法 3：如果仍然无效，返回一个合理的默认值
            if duration > 86400 or duration <= 0:
                print(f"  ⚠ 警告：检测到的视频时长 {duration:.1f}s 可能不正确")
                return 1.0
            
            return duration
            
        except Exception as e:
            print(f"  ⚠ 无法获取视频时长：{e}")
            return 1.0
    
    def _detect_gpu_encoder(self):
        """检测可用的 GPU 编码器"""
        print("\n正在检测 GPU 编码器...")
        
        try:
            ffmpeg_exe = str(self.FFMPEG_DIR / 'ffmpeg.exe')
            
            # 获取所有编码器列表
            result = subprocess.run(
                [ffmpeg_exe, '-encoders'],
                capture_output=True,
                timeout=10
            )
            
            # 使用 errors='ignore' 或 errors='replace' 来处理编码问题
            try:
                output = result.stdout.decode('utf-8', errors='ignore') + result.stderr.decode('utf-8', errors='ignore')
            except Exception:
                output = result.stdout.decode('latin-1', errors='ignore') + result.stderr.decode('latin-1', errors='ignore')
            
            # 根据选择的编解码器检测 GPU 编码器
            if self.codec == 'av1':
                # AV1 编码器优先级
                gpu_encoders = [
                    ('av1_nvenc', 'NVIDIA NVENC AV1'),       # RTX 40 系列支持
                    ('av1_qsv', 'Intel QSV AV1'),            # Arc 及更新显卡支持
                    ('av1_amf', 'AMD AMF AV1'),              # RDNA3 及更新显卡支持
                    ('libaom-av1', 'CPU libaom-av1'),        # CPU 编码（参考）
                    ('svt-av1', 'CPU SVT-AV1')               # CPU 快速编码（参考）
                ]
                encoder_type = 'AV1'
            elif self.codec == 'hevc':
                # HEVC/H.265 编码器优先级
                gpu_encoders = [
                    ('nvenc_hevc', 'NVIDIA NVENC HEVC'),
                    ('hevc_nvenc', 'NVIDIA NVENC HEVC (旧版)'),
                    ('hevc_qsv', 'Intel QSV HEVC'),
                    ('hevc_amf', 'AMD AMF HEVC'),
                    ('hevc_vaapi', 'Intel VAAPI HEVC'),
                    ('hevc_videotoolbox', 'Apple VideoToolbox HEVC')
                ]
                encoder_type = 'HEVC (H.265)'
            else:
                # H.264 编码器优先级
                gpu_encoders = [
                    ('nvenc_h264', 'NVIDIA NVENC'),
                    ('h264_nvenc', 'NVIDIA NVENC (旧版)'),
                    ('h264_qsv', 'Intel QSV'),
                    ('h264_amf', 'AMD AMF'),
                    ('h264_vaapi', 'Intel VAAPI'),
                    ('h264_videotoolbox', 'Apple VideoToolbox')
                ]
                encoder_type = 'H.264'
            
            for encoder_name, gpu_name in gpu_encoders:
                if encoder_name in output:
                    self.gpu_encoder = encoder_name
                    print(f"✓ 检测到 GPU 编码器：{gpu_name} ({encoder_name})")
                    return
            
            # 如果没有 GPU 编码器，使用 CPU 编码器
            if self.codec == 'av1':
                # 优先使用 SVT-AV1（更快），其次使用 libaom-av1
                if 'svt-av1' in output:
                    self.gpu_encoder = 'svt-av1'
                    print(f"⚠ 未检测到 GPU AV1 编码器，将使用 CPU SVT-AV1 编码")
                elif 'libaom-av1' in output:
                    self.gpu_encoder = 'libaom-av1'
                    print(f"⚠ 未检测到 GPU AV1 编码器，将使用 CPU libaom-av1 编码")
                else:
                    print(f"⚠ 未检测到可用的 AV1 编码器，请确保 FFmpeg 支持 AV1 编码")
                    self.gpu_encoder = None
            else:
                print(f"⚠ 未检测到可用的 {encoder_type} GPU 编码器，将使用 CPU 编码")
                self.gpu_encoder = None
            
        except Exception as e:
            print(f"✗ GPU 检测失败：{e}，将使用 CPU 编码")
            self.gpu_encoder = None
    
    def get_video_codec(self):
        """获取视频编码器名称"""
        if self.use_gpu and self.gpu_encoder:
            # 检查是否是 GPU 编码器
            if any(gpu in self.gpu_encoder for gpu in ['nvenc', 'qsv', 'amf', 'vaapi', 'videotoolbox']):
                return self.gpu_encoder
        
        # CPU 编码器
        if self.codec == 'av1':
            # 优先返回 SVT-AV1（更快），其次 libaom-av1
            return 'svt-av1' if self.gpu_encoder == 'svt-av1' else 'libaom-av1'
        elif self.codec == 'hevc':
            return 'libx265'
        return 'libx264'
    
    def get_encode_params(self):
        """获取编码参数"""
        codec = self.get_video_codec()
        
        base_params = {
            'c:v': codec,
            'c:a': 'aac',
            'strict': 'experimental',
            'y': None
        }
        
        # 如果使用CRF模式，设置CRF参数
        if self.crf is not None:
            base_params['crf'] = str(self.crf)
        else:
            # 否则使用固定码率
            base_params['b:v'] = self.target_bitrate
        
        # 根据编码器类型调整参数
        if self.use_gpu and codec:
            if 'nvenc' in codec:
                # NVIDIA NVENC 特定参数
                base_params['preset'] = 'p7'  # 高质量预设（NVENC最高质量）
                base_params['tune'] = 'hq'    # 高质量调优
                
                # 如果使用CRF模式，设置CQ值（NVENC使用-cq而非-crf）
                if self.crf is not None:
                    base_params['rc'] = 'vbr'     # 可变码率
                    base_params['cq'] = str(self.crf)  # 使用用户设置的CRF值作为CQ
                else:
                    base_params['rc'] = 'vbr'     # 可变码率
                    # AV1 特定优化
                    if 'av1' in codec:
                        base_params['cq'] = '25'  # AV1 质量参数
                        base_params['temporal-aq'] = '1'
                        base_params['zerorefdelay'] = '1'
                    # HEVC 特定优化
                    elif 'hevc' in codec:
                        base_params['cq'] = '23'
                        base_params['temporal-aq'] = '1'
                    # H.264 特定优化
                    else:
                        base_params['cq'] = '21'
                    
            elif 'qsv' in codec:
                # Intel QSV 特定参数
                base_params['preset'] = 'veryslow'  # CRF模式使用最慢预设以获得最佳压缩率
                base_params['lookahead'] = '1'
                
                # 如果使用CRF模式，设置Global Quality
                if self.crf is not None:
                    base_params['global_quality'] = str(self.crf)
                
                # AV1 特定优化
                if 'av1' in codec:
                    base_params['low_power'] = 'off'
                    base_params['adaptive_i'] = '1'
                # HEVC 特定优化
                elif 'hevc' in codec:
                    base_params['low_power'] = 'off'
                    
            elif 'amf' in codec:
                # AMD AMF 特定参数
                base_params['quality'] = 'quality'
                base_params['preusage'] = 'quality'
                
                # 如果使用CRF模式，设置QP值
                if self.crf is not None:
                    base_params['qp_i'] = str(self.crf)
                    base_params['qp_p'] = str(self.crf)
                    base_params['qp_b'] = str(self.crf)
                
                # AV1 特定优化
                if 'av1' in codec:
                    base_params['en_preenc'] = '1'
                    base_params['frame_pacing'] = 'balanced'
                # HEVC 特定优化
                elif 'hevc' in codec:
                    base_params['en_preenc'] = '1'
        else:
            # CPU 编码器参数
            if codec == 'libaom-av1':
                # AOM-AV1 参数（慢但质量好）
                if self.crf is not None:
                    # CRF模式：使用最慢预设以获得最佳压缩率
                    base_params['cpu-used'] = '0'  # 最慢但质量最好（0-8，0最慢）
                    base_params['crf'] = str(self.crf)
                else:
                    base_params['cpu-used'] = '4'  # 速度/质量平衡
                base_params['auto-alt-ref'] = '1'
                base_params['enable-cdef'] = '1'
                base_params['enable-restoration'] = '1'
                
            elif codec == 'svt-av1':
                # SVT-AV1 参数
                if self.crf is not None:
                    # CRF模式：使用最慢预设以获得最佳压缩率
                    base_params['preset'] = '0'  # 0-13，0最慢质量最好
                    base_params['crf'] = str(self.crf)
                else:
                    base_params['preset'] = '8'  # 默认速度
                    base_params['crf'] = '30'
                base_params['tile-columns'] = '4'
                base_params['tile-rows'] = '2'
                
            elif codec == 'libx265':
                # x265 特定参数
                if self.crf is not None:
                    # CRF模式：使用veryslow预设以获得最佳压缩率
                    base_params['preset'] = 'veryslow'
                    base_params['crf'] = str(self.crf)
                else:
                    base_params['preset'] = 'medium'
                    base_params['crf'] = '28'
                base_params['x265-params'] = 'aq-mode=2:aq-strength=1.0'
                
            elif codec == 'libx264':
                # x264 特定参数
                if self.crf is not None:
                    # CRF模式：使用veryslow预设以获得最佳压缩率
                    base_params['preset'] = 'veryslow'
                    base_params['crf'] = str(self.crf)
                else:
                    base_params['preset'] = 'medium'
                    base_params['crf'] = '23'
        
        return base_params
    
    def get_video_bitrate(self, video_path: Path) -> Optional[int]:
        """获取视频文件的视频码率（bps）"""
        try:
            probe = ffmpeg.probe(str(video_path), select_streams='v:0', show_entries='stream=bit_rate')
            streams = probe.get('streams', [])
            
            if streams and len(streams) > 0:
                bit_rate = streams[0].get('bit_rate')
                if bit_rate and bit_rate != 'N/A':
                    return int(bit_rate)
            
            # 如果流级别没有码率信息，尝试从文件级别获取
            probe = ffmpeg.probe(str(video_path), show_entries='format=bit_rate')
            format_info = probe.get('format', {})
            bit_rate = format_info.get('bit_rate')
            if bit_rate and bit_rate != 'N/A':
                return int(bit_rate)
            
            return None
        except Exception:
            return None
    
    def parse_bitrate_to_bps(self, bitrate_str: str) -> int:
        """将码率字符串转换为 bps"""
        bitrate_str = bitrate_str.upper().strip()
        
        if bitrate_str.endswith('M'):
            return int(float(bitrate_str[:-1]) * 1_000_000)
        elif bitrate_str.endswith('K'):
            return int(float(bitrate_str[:-1]) * 1_000)
        else:
            # 假设是纯数字，单位为 bps
            return int(bitrate_str)
    
    def encode_video(self, input_path: Path, output_path: Path) -> bool:
        """
        编码单个视频文件
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            
        Returns:
            bool: 编码是否成功
        """
        # 检查输出文件是否已存在，如果存在则跳过
        if output_path.exists():
            print(f"\n正在处理：{input_path.name}")
            print(f"  ✓ 跳过：输出文件已存在 {output_path.name}")
            return True
        
        print(f"\n正在处理：{input_path.name}")
        codec_name = self.get_video_codec()
        codec_display = f"{'GPU - ' if self.use_gpu and any(g in codec_name for g in ['nvenc', 'qsv', 'amf']) else 'CPU - '}{codec_name}"
        print(f"  使用编码器：{codec_display}")
        
        # 显示编码格式
        if self.codec == 'av1':
            format_display = 'AV1 (最新一代)'
        elif self.codec == 'hevc':
            format_display = 'HEVC (H.265)'
        else:
            format_display = 'H.264'
        print(f"  编码格式：{format_display}")
        
        # 显示码率控制模式
        if self.crf is not None:
            print(f"  码率控制：CRF模式 (CRF={self.crf})")
            
            # CRF模式下，先进行采样预估
            if self.sample_estimator:
                duration = self.get_video_duration(input_path)
                estimate_result = self.sample_estimator.estimate_compression_benefit(input_path, duration)
                
                if estimate_result is not None:
                    # 有采样结果，根据预估决定是否跳过
                    if estimate_result['should_skip']:
                        print(f"  ⏭ 跳过：预估压缩收益过低 ({estimate_result['estimated_benefit']:.1f}%)")
                        
                        # 如果启用了复制跳过文件功能
                        if self.copy_skipped:
                            print(f"  📋 复制原文件到输出目录...")
                            try:
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(str(input_path), str(output_path))
                                print(f"  ✓ 复制完成：{output_path.name}")
                                return True
                            except Exception as e:
                                print(f"  ✗ 复制失败：{e}")
                                return False
                        else:
                            return True
                    else:
                        print(f"  ✓ 预估值得转码，继续处理")
                else:
                    # 短视频或无法预估，直接转码
                    print(f"  ℹ 短视频或无法预估，直接进行转码")
        else:
            print(f"  码率控制：固定码率模式")
            print(f"  目标视频码率：{self.target_bitrate}")
        
        # 获取原始视频码率（用于信息显示）
        original_bitrate = self.get_video_bitrate(input_path)
        
        # 在固定码率模式下检查原始码率
        if self.crf is None and original_bitrate:
            target_bitrate_bps = self.parse_bitrate_to_bps(self.target_bitrate)
            print(f"  原始视频码率：{original_bitrate:,} bps ({original_bitrate/1000:.0f}K)")
            print(f"  目标视频码率：{target_bitrate_bps:,} bps ({target_bitrate_bps/1000:.0f}K)")
            
            # 检查是否需要重新编码
            if original_bitrate <= target_bitrate_bps:
                print(f"  ✓ 跳过：原始码率已低于目标码率，无需处理")
                
                # 如果启用了复制跳过文件功能
                if self.copy_skipped:
                    print(f"  📋 复制原文件到输出目录...")
                    try:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(input_path), str(output_path))
                        print(f"  ✓ 复制完成：{output_path.name}")
                        return True
                    except Exception as e:
                        print(f"  ✗ 复制失败：{e}")
                        return False
                else:
                    return True
        
        audio_bitrate = self.get_audio_bitrate(input_path)
        print(f"  检测到音频码率：{audio_bitrate}")
        
        temp_output = output_path.with_suffix('.temp.mp4')
        
        try:
            # 获取视频时长
            duration = self.get_video_duration(input_path)
            print(f"  视频总时长：{duration:.1f}s")
            
            # 保存原始的 ffmpeg 二进制路径
            original_ffmpeg = os.environ.get('FFMPEG_BINARY')
            
            # 设置 FFMPEG_BINARY 环境变量指向本地 FFmpeg
            os.environ['FFMPEG_BINARY'] = str(self.FFMPEG_DIR / 'ffmpeg.exe')
            
            # 构建 FFmpeg 命令
            cmd = [
                str(self.FFMPEG_DIR / 'ffmpeg.exe'),
                '-i', str(input_path),
                '-c:v', self.get_video_codec(),
            ]
            
            # 根据模式和编码器类型添加码率控制参数
            codec_name = self.get_video_codec()
            
            if self.crf is not None:
                # CRF模式：不同编码器使用不同的参数名
                if 'nvenc' in codec_name:
                    # NVIDIA GPU编码器使用 -cq
                    cmd.extend(['-cq', str(self.crf)])
                elif 'qsv' in codec_name:
                    # Intel QSV使用 -global_quality
                    cmd.extend(['-global_quality', str(self.crf)])
                elif 'amf' in codec_name:
                    # AMD AMF使用 -qp_i -qp_p -qp_b
                    cmd.extend(['-qp_i', str(self.crf), '-qp_p', str(self.crf), '-qp_b', str(self.crf)])
                else:
                    # CPU编码器使用 -crf
                    cmd.extend(['-crf', str(self.crf)])
            else:
                # 固定码率模式
                cmd.extend(['-b:v', self.target_bitrate])
            
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', audio_bitrate,
                '-strict', 'experimental',
                '-pix_fmt', 'yuv420p',
                '-progress', 'pipe:1',
                '-loglevel', 'quiet',
                '-stats_period', '0.5',
                '-y',
                str(temp_output)
            ])
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            # 解析进度
            time_pattern = re.compile(r'out_time_ms=(\d+)')
            
            with tqdm(total=duration if duration > 0 and duration < 86400 else None, 
                     desc='  处理进度', 
                     unit='s',
                     disable=(duration <= 0 or duration >= 86400),
                     bar_format='{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}, {rate_fmt}]' if (duration > 0 and duration < 86400) else '{l_bar}{bar}| {elapsed} [{rate_fmt}]') as pbar:
                
                # 逐行读取 stdout（进度信息）
                while True:
                    line = process.stdout.readline()
                    
                    if not line and process.poll() is not None:
                        break
                    
                    if line:
                        try:
                            line_str = line.decode('utf-8', errors='ignore')
                        except Exception:
                            line_str = line.decode('latin-1', errors='ignore')
                        
                        # 匹配 out_time_ms
                        match = time_pattern.search(line_str)
                        if match:
                            # out_time_ms 是微秒数
                            time_ms = int(match.group(1))
                            current_time = time_ms / 1_000_000  # 转换为秒
                            
                            # 只在合理范围内更新进度
                            if duration > 0 and duration < 86400 and current_time < duration * 1.1:
                                pbar.n = current_time
                                pbar.refresh()
                
                # 等待进程结束
                process.wait()
                
                # 恢复原始环境变量
                if original_ffmpeg:
                    os.environ['FFMPEG_BINARY'] = original_ffmpeg
                else:
                    os.environ.pop('FFMPEG_BINARY', None)
                
                if process.returncode != 0:
                    error_bytes = process.stderr.read()
                    try:
                        error_output = error_bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        error_output = error_bytes.decode('latin-1', errors='ignore')
                    raise RuntimeError(f"FFmpeg 错误：{error_output[:500]}")
            
            # CRF模式下，检查压缩后的文件大小
            if self.crf is not None and temp_output.exists():
                original_size = input_path.stat().st_size
                compressed_size = temp_output.stat().st_size
                
                original_size_mb = original_size / (1024 * 1024)
                compressed_size_mb = compressed_size / (1024 * 1024)
                size_ratio = (compressed_size - original_size) / original_size * 100
                
                print(f"  原始文件大小：{original_size_mb:.2f} MB")
                print(f"  压缩文件大小：{compressed_size_mb:.2f} MB")
                
                # 如果压缩后文件更大，使用原文件
                if compressed_size > original_size:
                    print(f"  ⚠ 警告：压缩后文件体积增大了 {size_ratio:.1f}%")
                    print(f"  📋 使用原文件（删除压缩文件）...")
                    
                    # 删除临时压缩文件
                    temp_output.unlink()
                    
                    # 复制原文件到输出位置
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(input_path), str(output_path))
                    
                    print(f"  ✓ 已拷贝原文件：{output_path.name}")
                    print(f"  ✓ 编码完成：{output_path.name}（未压缩）")
                    return True
                else:
                    # 压缩成功，显示体积减少
                    size_reduction = (1 - compressed_size / original_size) * 100
                    print(f"  体积减少：{size_reduction:.1f}%")
            
            # 正常情况：重命名临时文件为最终输出文件
            temp_output.rename(output_path)
            
            print(f"  ✓ 编码完成：{output_path.name}")
            return True
            
        except KeyboardInterrupt:
            print(f"\n  ⚠ 用户取消操作")
            # 删除临时文件和输出文件
            if temp_output.exists():
                temp_output.unlink()
                print(f"  🗑 已删除临时文件：{temp_output.name}")
            if output_path.exists():
                output_path.unlink()
                print(f"  🗑 已删除输出文件：{output_path.name}")
            raise  # 重新抛出异常，让上层处理
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ 编码失败：{error_msg}")
            if temp_output.exists():
                temp_output.unlink()
            return False
    
    def process(self):
        """批量处理所有视频文件"""
        video_files = self.find_video_files()
        
        if not video_files:
            print(f"在目录 '{self.input_dir}' 下未找到视频文件")
            return
        
        print(f"找到 {len(video_files)} 个视频文件")
        print("=" * 60)
        
        success_count = 0
        fail_count = 0
        
        try:
            for i, video_path in enumerate(video_files, 1):
                print(f"\n[{i}/{len(video_files)}]")
                
                if self.output_dir:
                    relative_path = video_path.relative_to(self.input_dir)
                    output_path = self.output_dir / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = video_path.parent / f"{video_path.stem}_compressed{video_path.suffix}"
                
                if self.encode_video(video_path, output_path):
                    success_count += 1
                else:
                    fail_count += 1
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print(f"⚠ 用户中断操作")
            print(f"已处理：{success_count} 个成功，{fail_count} 个失败")
            print("程序已退出")
            return
        
        print("\n" + "=" * 60)
        print(f"处理完成！")
        print(f"  成功：{success_count} 个")
        print(f"  失败：{fail_count} 个")


def main():
    parser = argparse.ArgumentParser(
        description='批量压缩视频码率工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py -i ./videos -b 1000K
  python main.py -i ./videos -o ./output -b 2M -r
  python main.py --input-dir ./videos --bitrate 1500K
  python main.py -i ./videos -b 1000K --cpu  # 强制使用 CPU 编码
  python main.py -i ./videos -b 800K --codec hevc  # 使用 HEVC 编码
  python main.py -i ./videos -b 600K --codec av1   # 使用 AV1 编码（推荐）
  python main.py -i ./videos -b 1000K -o ./output --copy-skipped  # 复制跳过的视频
  python main.py -i ./videos --crf 23  # 使用 CRF 模式，CRF值为23
  python main.py -i ./videos --crf 28 --codec hevc  # HEVC + CRF 28
  python main.py -i ./videos --crf 30 --codec av1   # AV1 + CRF 30
  python main.py -i ./videos --crf 23 --min-benefit 15  # 最小收益15%才转码
  python main.py -i ./videos --crf 23 --max-samples 3   # 最多3个采样点
        """
    )
    
    parser.add_argument('-i', '--input-dir', required=True, 
                       help='输入视频目录路径')
    parser.add_argument('-o', '--output-dir', default=None,
                       help='输出目录路径（默认保存在源文件同目录，添加_compressed 后缀）')
    parser.add_argument('-b', '--bitrate', default='1000K',
                       help='目标视频码率（默认：1000K），例如：500K, 1M, 2M 等。如果指定了--crf则此参数可选')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='是否递归处理子目录')
    parser.add_argument('--cpu', action='store_true',
                       help='强制使用 CPU 编码，不使用 GPU 加速')
    parser.add_argument('--codec', choices=['h264', 'hevc', 'av1'], default='h264',
                       help='视频编码格式（默认：h264）。选项：h264（兼容性好）、hevc（高效）、av1（最新最高效）')
    parser.add_argument('--copy-skipped', '-c', action='store_true',
                       help='将因码率低于目标而跳过的视频复制到输出目录（默认不复制）')
    parser.add_argument('--crf', type=int, default=None,
                       help='CRF值（恒定质量因子），范围0-51，值越小质量越高。推荐使用：H.264(18-28), HEVC(20-30), AV1(25-35)。如果设置则优先使用CRF模式而非固定码率')
    parser.add_argument('--min-benefit', type=float, default=-10.0,
                       help='CRF采样预估的最小收益阈值（百分比，默认-10%）。负值表示允许文件略微增大以尝试压缩，实际压缩后会检测文件大小，只有更小时才采用')
    parser.add_argument('--max-samples', type=int, default=5,
                       help='CRF采样预估的最大采样点数（默认5个）')
    
    args = parser.parse_args()
    
    try:
        encoder = VideoReEncoder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_bitrate=args.bitrate,
            recursive=args.recursive,
            use_gpu=not args.cpu,
            codec=args.codec,
            copy_skipped=args.copy_skipped,
            crf=args.crf,
            min_benefit_threshold=args.min_benefit,
            max_sample_count=args.max_samples
        )
        encoder.process()
    except Exception as e:
        print(f"错误：{e}")
        exit(1)


if __name__ == '__main__':
    main()
