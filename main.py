import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
import ffmpeg


class VideoReEncoder:
    """视频重新编码器，用于批量压缩视频码率"""
    
    VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.wmv', '.m4v'}
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None, 
                 target_bitrate: str = '1000K', recursive: bool = False):
        """
        初始化编码器
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径，如果为 None 则生成在源文件同目录下
            target_bitrate: 目标视频码率，如 '1000K', '2M' 等
            recursive: 是否递归处理子目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.target_bitrate = target_bitrate
        self.recursive = recursive
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在：{input_dir}")
        
        self._check_ffmpeg()
    
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
    
    def encode_video(self, input_path: Path, output_path: Path) -> bool:
        """
        编码单个视频文件
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            
        Returns:
            bool: 编码是否成功
        """
        print(f"\n正在处理：{input_path.name}")
        print(f"  目标视频码率：{self.target_bitrate}")
        
        audio_bitrate = self.get_audio_bitrate(input_path)
        print(f"  检测到音频码率：{audio_bitrate}")
        
        temp_output = output_path.with_suffix('.temp.mp4')
        
        try:
            (
                ffmpeg
                .input(str(input_path))
                .output(
                    str(temp_output),
                    **{'c:v': 'libx264', 'b:v': self.target_bitrate, 'c:a': 'aac', 
                       'b:a': audio_bitrate, 'strict': 'experimental', 'y': None}
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            temp_output.rename(output_path)
            print(f"  ✓ 编码完成：{output_path.name}")
            return True
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
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
        """
    )
    
    parser.add_argument('-i', '--input-dir', required=True, 
                       help='输入视频目录路径')
    parser.add_argument('-o', '--output-dir', default=None,
                       help='输出目录路径（默认保存在源文件同目录，添加_compressed 后缀）')
    parser.add_argument('-b', '--bitrate', default='1000K',
                       help='目标视频码率（默认：1000K），例如：500K, 1M, 2M 等')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='是否递归处理子目录')
    
    args = parser.parse_args()
    
    try:
        encoder = VideoReEncoder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_bitrate=args.bitrate,
            recursive=args.recursive
        )
        encoder.process()
    except Exception as e:
        print(f"错误：{e}")
        exit(1)


if __name__ == '__main__':
    main()
