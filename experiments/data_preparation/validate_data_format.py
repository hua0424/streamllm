#!/usr/bin/env python3
"""
数据格式验证工具
验证音频数据和转录文本是否符合实验要求的格式规范
"""

import json
import wave
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging


class DataFormatValidator:
    """数据格式验证器"""
    
    def __init__(self, data_dir: str = "experiments/data"):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "processed_audio"
        self.transcript_dir = self.data_dir / "transcripts"
        
        # 预期的长度分组
        self.expected_groups = ["short_5to10s", "medium_10to20s", "long_20plus"]
        
        # 音频格式要求
        self.audio_requirements = {
            "format": "WAV",
            "sample_rate": 16000,
            "sample_width": 2,  # 16bit = 2 bytes
            "channels": 1,      # mono
        }
        
        # 转录文本必需字段
        self.required_fields = ["audio_file", "duration", "text", "language", "speaker", "ground_truth"]
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def validate_directory_structure(self) -> Tuple[bool, List[str]]:
        """验证目录结构"""
        errors = []
        
        # 检查主目录
        if not self.data_dir.exists():
            errors.append(f"数据目录不存在: {self.data_dir}")
            return False, errors
            
        if not self.audio_dir.exists():
            errors.append(f"音频目录不存在: {self.audio_dir}")
            
        if not self.transcript_dir.exists():
            errors.append(f"转录目录不存在: {self.transcript_dir}")
            
        # 检查长度分组目录
        for group in self.expected_groups:
            audio_group_dir = self.audio_dir / group
            transcript_group_dir = self.transcript_dir / group
            
            if not audio_group_dir.exists():
                errors.append(f"音频分组目录不存在: {audio_group_dir}")
                
            if not transcript_group_dir.exists():
                errors.append(f"转录分组目录不存在: {transcript_group_dir}")
        
        return len(errors) == 0, errors
    
    def validate_audio_file(self, audio_path: Path) -> Tuple[bool, List[str]]:
        """验证单个音频文件格式"""
        errors = []
        
        if not audio_path.exists():
            errors.append(f"音频文件不存在: {audio_path}")
            return False, errors
            
        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                # 检查采样率
                if wav_file.getframerate() != self.audio_requirements["sample_rate"]:
                    errors.append(f"{audio_path}: 采样率错误 - 期望{self.audio_requirements['sample_rate']}Hz，实际{wav_file.getframerate()}Hz")
                
                # 检查位深度
                if wav_file.getsampwidth() != self.audio_requirements["sample_width"]:
                    errors.append(f"{audio_path}: 位深度错误 - 期望{self.audio_requirements['sample_width']*8}bit，实际{wav_file.getsampwidth()*8}bit")
                
                # 检查声道数
                if wav_file.getnchannels() != self.audio_requirements["channels"]:
                    errors.append(f"{audio_path}: 声道数错误 - 期望{self.audio_requirements['channels']}声道，实际{wav_file.getnchannels()}声道")
                    
                # 计算音频时长
                duration = wav_file.getnframes() / wav_file.getframerate()
                
                # 验证时长是否符合分组要求
                group_name = audio_path.parent.name
                if not self._validate_duration_for_group(duration, group_name):
                    errors.append(f"{audio_path}: 音频时长{duration:.1f}s不符合分组{group_name}的要求")
                    
        except Exception as e:
            errors.append(f"{audio_path}: 无法读取音频文件 - {str(e)}")
            
        return len(errors) == 0, errors
    
    def _validate_duration_for_group(self, duration: float, group_name: str) -> bool:
        """验证音频时长是否符合分组要求"""
        duration_ranges = {
            "short_5to10s": (5, 10),
            "medium_10to20s": (10, 20),
            "long_20plus": (20, float('inf'))
        }
        
        if group_name not in duration_ranges:
            return False
            
        min_dur, max_dur = duration_ranges[group_name]
        return min_dur <= duration <= max_dur
    
    def validate_transcript_file(self, transcript_path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
        """验证转录文件格式"""
        errors = []
        transcript_data = {}
        
        if not transcript_path.exists():
            errors.append(f"转录文件不存在: {transcript_path}")
            return False, errors, transcript_data
            
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
                
            # 检查必需字段
            for field in self.required_fields:
                if field not in transcript_data:
                    errors.append(f"{transcript_path}: 缺少必需字段 '{field}'")
                elif not transcript_data[field]:  # 检查字段是否为空
                    errors.append(f"{transcript_path}: 字段 '{field}' 不能为空")
            
            # 检查数据类型
            if "duration" in transcript_data:
                try:
                    float(transcript_data["duration"])
                except (ValueError, TypeError):
                    errors.append(f"{transcript_path}: 'duration' 字段必须是数字")
            
            # 检查语言字段值
            if "language" in transcript_data:
                valid_languages = ["zh", "en", "mixed"]
                if transcript_data["language"] not in valid_languages:
                    errors.append(f"{transcript_path}: 'language' 字段值必须是 {valid_languages} 之一")
                    
        except json.JSONDecodeError as e:
            errors.append(f"{transcript_path}: JSON格式错误 - {str(e)}")
        except Exception as e:
            errors.append(f"{transcript_path}: 读取文件错误 - {str(e)}")
            
        return len(errors) == 0, errors, transcript_data
    
    def validate_data_consistency(self, group_name: str) -> Tuple[bool, List[str]]:
        """验证数据一致性（音频文件与转录文件的对应关系）"""
        errors = []
        
        audio_group_dir = self.audio_dir / group_name
        transcript_group_dir = self.transcript_dir / group_name
        
        if not (audio_group_dir.exists() and transcript_group_dir.exists()):
            return True, []  # 如果目录不存在，在其他验证中会报错
        
        # 获取音频文件列表
        audio_files = {f.stem: f for f in audio_group_dir.glob("*.wav")}
        
        # 获取转录文件列表
        json_files = {f.stem: f for f in transcript_group_dir.glob("*.json")}
        txt_files = {f.stem: f for f in transcript_group_dir.glob("*.txt")}
        
        # 检查每个音频文件是否有对应的转录文件
        for audio_stem in audio_files:
            if audio_stem not in json_files:
                errors.append(f"音频文件 {audio_files[audio_stem]} 缺少对应的JSON转录文件")
            
            if audio_stem not in txt_files:
                errors.append(f"音频文件 {audio_files[audio_stem]} 缺少对应的TXT转录文件")
        
        # 检查转录文件中引用的音频文件是否存在
        for json_stem, json_file in json_files.items():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if "audio_file" in data:
                    expected_audio = Path(data["audio_file"]).stem
                    if expected_audio != json_stem:
                        errors.append(f"转录文件 {json_file} 中的 audio_file 字段与文件名不匹配")
                        
            except Exception as e:
                errors.append(f"验证转录文件 {json_file} 时出错: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_all(self) -> Dict[str, Any]:
        """执行全面验证"""
        self.logger.info("开始数据格式验证...")
        
        validation_results = {
            "overall_status": True,
            "directory_structure": {"status": True, "errors": []},
            "audio_validation": {"status": True, "errors": [], "files_checked": 0},
            "transcript_validation": {"status": True, "errors": [], "files_checked": 0},
            "consistency_validation": {"status": True, "errors": []},
            "summary": {"total_audio_files": 0, "total_transcript_files": 0, "total_errors": 0}
        }
        
        # 1. 验证目录结构
        self.logger.info("验证目录结构...")
        dir_status, dir_errors = self.validate_directory_structure()
        validation_results["directory_structure"]["status"] = dir_status
        validation_results["directory_structure"]["errors"] = dir_errors
        
        if not dir_status:
            self.logger.error("目录结构验证失败，跳过后续验证")
            validation_results["overall_status"] = False
            validation_results["summary"]["total_errors"] = len(dir_errors)
            return validation_results
        
        # 2. 验证每个分组的数据
        for group in self.expected_groups:
            self.logger.info(f"验证分组: {group}")
            
            # 验证音频文件
            audio_group_dir = self.audio_dir / group
            if audio_group_dir.exists():
                for audio_file in audio_group_dir.glob("*.wav"):
                    audio_status, audio_errors = self.validate_audio_file(audio_file)
                    validation_results["audio_validation"]["files_checked"] += 1
                    validation_results["summary"]["total_audio_files"] += 1
                    
                    if not audio_status:
                        validation_results["audio_validation"]["status"] = False
                        validation_results["audio_validation"]["errors"].extend(audio_errors)
            
            # 验证转录文件
            transcript_group_dir = self.transcript_dir / group
            if transcript_group_dir.exists():
                for transcript_file in transcript_group_dir.glob("*.json"):
                    trans_status, trans_errors, _ = self.validate_transcript_file(transcript_file)
                    validation_results["transcript_validation"]["files_checked"] += 1
                    validation_results["summary"]["total_transcript_files"] += 1
                    
                    if not trans_status:
                        validation_results["transcript_validation"]["status"] = False
                        validation_results["transcript_validation"]["errors"].extend(trans_errors)
            
            # 验证数据一致性
            cons_status, cons_errors = self.validate_data_consistency(group)
            if not cons_status:
                validation_results["consistency_validation"]["status"] = False
                validation_results["consistency_validation"]["errors"].extend(cons_errors)
        
        # 计算总体状态
        validation_results["overall_status"] = (
            validation_results["directory_structure"]["status"] and
            validation_results["audio_validation"]["status"] and
            validation_results["transcript_validation"]["status"] and
            validation_results["consistency_validation"]["status"]
        )
        
        # 计算总错误数
        total_errors = (
            len(validation_results["directory_structure"]["errors"]) +
            len(validation_results["audio_validation"]["errors"]) +
            len(validation_results["transcript_validation"]["errors"]) +
            len(validation_results["consistency_validation"]["errors"])
        )
        validation_results["summary"]["total_errors"] = total_errors
        
        return validation_results
    
    def print_validation_report(self, results: Dict[str, Any]):
        """打印验证报告"""
        print("\n" + "="*60)
        print("数据格式验证报告")
        print("="*60)
        
        # 总体状态
        status_symbol = "✅" if results["overall_status"] else "❌"
        print(f"\n总体状态: {status_symbol} {'通过' if results['overall_status'] else '失败'}")
        
        # 详细结果
        sections = [
            ("目录结构验证", "directory_structure"),
            ("音频文件验证", "audio_validation"),
            ("转录文件验证", "transcript_validation"),
            ("数据一致性验证", "consistency_validation")
        ]
        
        for section_name, section_key in sections:
            section_data = results[section_key]
            status_symbol = "✅" if section_data["status"] else "❌"
            print(f"\n{section_name}: {status_symbol}")
            
            if "files_checked" in section_data:
                print(f"  检查文件数: {section_data['files_checked']}")
            
            if section_data["errors"]:
                print(f"  错误数: {len(section_data['errors'])}")
                for error in section_data["errors"][:5]:  # 只显示前5个错误
                    print(f"    - {error}")
                if len(section_data["errors"]) > 5:
                    print(f"    ... 还有 {len(section_data['errors']) - 5} 个错误")
        
        # 统计信息
        print(f"\n统计信息:")
        print(f"  音频文件总数: {results['summary']['total_audio_files']}")
        print(f"  转录文件总数: {results['summary']['total_transcript_files']}")
        print(f"  错误总数: {results['summary']['total_errors']}")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="验证实验数据格式")
    parser.add_argument("--data-dir", default="experiments/data", 
                       help="数据目录路径 (默认: experiments/data)")
    parser.add_argument("--output", help="将验证结果保存到JSON文件")
    parser.add_argument("--quiet", action="store_true", help="仅显示错误信息")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # 创建验证器并执行验证
    validator = DataFormatValidator(args.data_dir)
    results = validator.validate_all()
    
    # 打印报告
    validator.print_validation_report(results)
    
    # 保存结果到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n验证结果已保存到: {args.output}")
    
    # 返回适当的退出码
    return 0 if results["overall_status"] else 1


if __name__ == "__main__":
    exit(main())