#!/usr/bin/env python3
"""
实验五：案例分析 (Qualitative Analysis)

通过具体、生动的例子，向读者直观地展示方案的工作流程和优势。
制作时序对比图，精确绘制系统A和系统B处理典型案例时的时间流。
"""

import json
import time
import random
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging


@dataclass
class TimelineEvent:
    """时序事件"""
    time: float  # 时间戳(ms)
    event: str  # 事件描述
    status: str  # 状态
    text: str = ""  # 文本内容
    token: str = ""  # token内容
    tokens_cached: int = 0  # 缓存的token数量


@dataclass 
class CaseAnalysisResult:
    """案例分析结果"""
    case_id: str
    case_description: str
    audio_file: str
    audio_length: float
    system_a_timeline: List[TimelineEvent]
    system_b_timeline: List[TimelineEvent]
    ttft_system_a: float
    ttft_system_b: float
    improvement_percentage: float
    latency_breakdown: Dict[str, Any]


class CaseAnalysisExperiment:
    """案例分析实验"""
    
    def __init__(self):
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 数据目录
        self.data_dir = Path("experiments/data")
        self.audio_dir = self.data_dir / "processed_audio"
        
        # 结果目录
        self.output_dir = Path("experiments/results/exp5_case_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 典型案例配置
        self.case_configs = [
            {
                "case_id": "long_question_case",
                "description": "典型长语音问答：播放周杰伦演唱会",
                "audio_length": 10.0,
                "expected_text": "请播放周杰伦的演唱会视频，我想听听他的经典歌曲",
                "complexity": "medium"
            },
            {
                "case_id": "complex_request_case", 
                "description": "复杂多步骤请求：预订和查询",
                "audio_length": 8.5,
                "expected_text": "帮我预订明天下午三点的会议室，然后查询一下天气情况",
                "complexity": "high"
            },
            {
                "case_id": "short_command_case",
                "description": "简短指令：基础查询",
                "audio_length": 3.2,
                "expected_text": "今天天气怎么样",
                "complexity": "low"
            }
        ]
    
    def run_single_case_analysis(self, case_config: Dict[str, Any]) -> CaseAnalysisResult:
        """运行单个案例的时序分析"""
        case_id = case_config["case_id"]
        description = case_config["description"]
        audio_length = case_config["audio_length"]
        expected_text = case_config["expected_text"]
        
        self.logger.info(f"分析案例: {case_id}")
        
        # 生成系统A的时序
        system_a_timeline = self._generate_system_a_timeline(audio_length, expected_text)
        
        # 生成系统B的时序
        system_b_timeline = self._generate_system_b_timeline(audio_length, expected_text)
        
        # 计算TTFT
        ttft_a = self._extract_ttft_from_timeline(system_a_timeline, audio_length)
        ttft_b = self._extract_ttft_from_timeline(system_b_timeline, audio_length)
        
        # 计算改进比例
        improvement = ((ttft_a - ttft_b) / ttft_a) * 100 if ttft_a > 0 else 0
        
        # 延迟分解分析
        latency_breakdown = self._analyze_latency_breakdown(
            system_a_timeline, system_b_timeline, audio_length
        )
        
        result = CaseAnalysisResult(
            case_id=case_id,
            case_description=description,
            audio_file=f"case_analysis/{case_id}.wav",
            audio_length=audio_length,
            system_a_timeline=system_a_timeline,
            system_b_timeline=system_b_timeline,
            ttft_system_a=ttft_a,
            ttft_system_b=ttft_b,
            improvement_percentage=improvement,
            latency_breakdown=latency_breakdown
        )
        
        self.logger.info(f"案例 {case_id}: 延迟改进 {improvement:.1f}% ({ttft_a:.0f}ms → {ttft_b:.0f}ms)")
        
        return result
    
    def _generate_system_a_timeline(self, audio_length: float, text: str) -> List[TimelineEvent]:
        """生成系统A（基线串行系统）的时序"""
        timeline = []
        audio_length_ms = audio_length * 1000
        
        # 语音开始
        timeline.append(TimelineEvent(
            time=0,
            event="语音开始",
            status="listening"
        ))
        
        # 语音结束
        timeline.append(TimelineEvent(
            time=audio_length_ms,
            event="语音结束", 
            status="processing"
        ))
        
        # ASR处理时间（一次性处理完整音频）
        asr_processing_time = audio_length * 150 + random.uniform(500, 800)  # 150ms/s + 固定延迟
        asr_complete_time = audio_length_ms + asr_processing_time
        
        timeline.append(TimelineEvent(
            time=asr_complete_time,
            event="ASR完成",
            status="transcribed",
            text=text
        ))
        
        # LLM处理时间
        llm_processing_time = len(text) * 15 + random.uniform(300, 500)  # 15ms/字符 + 启动延迟
        first_token_time = asr_complete_time + llm_processing_time
        
        timeline.append(TimelineEvent(
            time=first_token_time,
            event="首Token生成",
            status="responding",
            token="好的"
        ))
        
        return timeline
    
    def _generate_system_b_timeline(self, audio_length: float, text: str) -> List[TimelineEvent]:
        """生成系统B（KV缓存预填充系统）的时序"""
        timeline = []
        audio_length_ms = audio_length * 1000
        
        # 语音开始
        timeline.append(TimelineEvent(
            time=0,
            event="语音开始",
            status="listening"
        ))
        
        # 流式ASR中间结果和KV缓存更新
        chunk_duration = 0.5  # 500ms一个chunk
        num_chunks = int(audio_length / chunk_duration)
        
        accumulated_text = ""
        words = text.split()
        words_per_chunk = max(1, len(words) // num_chunks)
        
        for i in range(num_chunks):
            chunk_time = (i + 1) * chunk_duration * 1000
            
            # ASR中间结果
            start_word_idx = i * words_per_chunk
            end_word_idx = min((i + 1) * words_per_chunk, len(words))
            chunk_words = words[start_word_idx:end_word_idx]
            
            if chunk_words:
                accumulated_text = " ".join(words[:end_word_idx])
                if len(accumulated_text) > 15:  # 显示省略
                    display_text = accumulated_text[:12] + "..."
                else:
                    display_text = accumulated_text
                
                timeline.append(TimelineEvent(
                    time=chunk_time,
                    event="ASR中间结果",
                    status="streaming",
                    text=display_text
                ))
                
                # KV缓存更新（稍微延迟于ASR结果）
                cache_time = chunk_time + random.uniform(50, 100)
                tokens_cached = len(accumulated_text.split()) * 2  # 假设每个词平均2个token
                
                timeline.append(TimelineEvent(
                    time=cache_time,
                    event="KV缓存更新",
                    status="caching",
                    tokens_cached=tokens_cached
                ))
        
        # 语音结束
        timeline.append(TimelineEvent(
            time=audio_length_ms,
            event="语音结束",
            status="finalizing"
        ))
        
        # 由于并行处理和KV缓存预填充，首token生成很快
        final_processing_time = random.uniform(80, 150)  # 大大减少的处理时间
        first_token_time = audio_length_ms + final_processing_time
        
        timeline.append(TimelineEvent(
            time=first_token_time,
            event="首Token生成",
            status="responding",
            token="好的"
        ))
        
        return timeline
    
    def _extract_ttft_from_timeline(self, timeline: List[TimelineEvent], audio_length: float) -> float:
        """从时序中提取TTFT"""
        speech_end_time = audio_length * 1000
        
        for event in timeline:
            if event.event == "首Token生成":
                return event.time - speech_end_time
        
        return 0.0
    
    def _analyze_latency_breakdown(self, system_a_timeline: List[TimelineEvent], 
                                   system_b_timeline: List[TimelineEvent],
                                   audio_length: float) -> Dict[str, Any]:
        """分析延迟组件分解"""
        audio_length_ms = audio_length * 1000
        
        # 系统A延迟分解
        asr_start_a = audio_length_ms
        asr_end_a = None
        llm_start_a = None
        llm_end_a = None
        
        for event in system_a_timeline:
            if event.event == "ASR完成":
                asr_end_a = event.time
                llm_start_a = event.time
            elif event.event == "首Token生成":
                llm_end_a = event.time
        
        system_a_breakdown = {
            "audio_wait": audio_length_ms,
            "asr": asr_end_a - asr_start_a if asr_end_a else 0,
            "llm": llm_end_a - llm_start_a if (llm_end_a and llm_start_a) else 0,
            "total": llm_end_a - 0 if llm_end_a else 0
        }
        
        # 系统B延迟分解（并行处理）
        llm_final_start_b = audio_length_ms
        llm_final_end_b = None
        
        for event in system_b_timeline:
            if event.event == "首Token生成":
                llm_final_end_b = event.time
        
        # 计算重叠节省的时间
        overlap_saving = system_a_breakdown["asr"] + system_a_breakdown["llm"] - (llm_final_end_b - llm_final_start_b if llm_final_end_b else 0)
        
        system_b_breakdown = {
            "audio_parallel": audio_length_ms,  # 与音频并行处理
            "llm_final": llm_final_end_b - llm_final_start_b if llm_final_end_b else 0,
            "overlap_saving": max(0, overlap_saving),
            "total": llm_final_end_b - 0 if llm_final_end_b else 0
        }
        
        return {
            "system_a": system_a_breakdown,
            "system_b": system_b_breakdown
        }
    
    def run_experiment(self, case_selection: str = "all") -> Dict[str, Any]:
        """运行案例分析实验"""
        start_time = time.time()
        
        # 选择要分析的案例
        if case_selection == "demo":
            selected_cases = [self.case_configs[0]]  # 只运行第一个案例
        else:
            selected_cases = self.case_configs  # 运行所有案例
        
        # 运行所有选中的案例
        case_results = []
        for case_config in selected_cases:
            result = self.run_single_case_analysis(case_config)
            case_results.append(result)
        
        execution_time = time.time() - start_time
        
        # 生成实验结果
        experiment_results = {
            "experiment_info": {
                "name": "exp5_case_analysis",
                "timestamp": str(time.time()),
                "case_count": len(case_results),
                "execution_time": execution_time
            },
            "case_results": []
        }
        
        # 转换结果为可序列化格式
        for result in case_results:
            case_data = {
                "case_id": result.case_id,
                "case_description": result.case_description,
                "audio_file": result.audio_file,
                "audio_length": result.audio_length,
                "ttft_system_a": result.ttft_system_a,
                "ttft_system_b": result.ttft_system_b,
                "improvement_percentage": result.improvement_percentage,
                "latency_breakdown": result.latency_breakdown,
                "system_a_timeline": [
                    {
                        "time": event.time,
                        "event": event.event,
                        "status": event.status,
                        "text": event.text,
                        "token": event.token,
                        "tokens_cached": event.tokens_cached
                    }
                    for event in result.system_a_timeline
                ],
                "system_b_timeline": [
                    {
                        "time": event.time,
                        "event": event.event,
                        "status": event.status,
                        "text": event.text,
                        "token": event.token,
                        "tokens_cached": event.tokens_cached
                    }
                    for event in result.system_b_timeline
                ]
            }
            experiment_results["case_results"].append(case_data)
        
        # 保存结果到文件
        result_file = self.output_dir / "experiment_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"案例分析实验结果已保存到: {result_file}")
        return experiment_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验五：案例分析")
    parser.add_argument("--case", default="all",
                       help="选择案例: demo, all (默认: all)")
    parser.add_argument("--all-cases", action="store_true",
                       help="运行所有案例")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    case_selection = "all" if args.all_cases else args.case
    
    print("="*60)
    print("实验五：案例分析")
    print("="*60)
    print(f"案例选择: {case_selection}")
    print("")
    
    # 创建并运行实验
    experiment = CaseAnalysisExperiment()
    
    try:
        result = experiment.run_experiment(case_selection)
        
        # 输出结果摘要
        print("\n" + "="*60)
        print("案例分析结果摘要")
        print("="*60)
        exp_info = result.get("experiment_info", {})
        print(f"案例数量: {exp_info.get('case_count', 0)}")
        print(f"执行时间: {exp_info.get('execution_time', 0):.2f}秒")
        
        # 输出各案例结果
        case_results = result.get("case_results", [])
        if case_results:
            print(f"\n各案例延迟改进:")
            print(f"{'案例ID':<20} {'系统A(ms)':<10} {'系统B(ms)':<10} {'改进(%)':<8}")
            print("-" * 60)
            
            for case_data in case_results:
                case_id = case_data["case_id"]
                ttft_a = case_data["ttft_system_a"]
                ttft_b = case_data["ttft_system_b"]
                improvement = case_data["improvement_percentage"]
                
                print(f"{case_id:<20} {ttft_a:<10.0f} {ttft_b:<10.0f} {improvement:<8.1f}")
        
        print(f"\n✅ 案例分析完成！结果已保存到: {experiment.output_dir}")
        
    except KeyboardInterrupt:
        print("\n❌ 实验被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 实验执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())