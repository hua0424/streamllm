#!/usr/bin/env python3
"""
测试真实环境下的系统A
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from system_a_baseline import SystemA_BaselineSequential

def test_with_real_data():
    """使用真实数据测试系统A"""
    print("=" * 60)
    print("系统A真实环境测试")
    print("=" * 60)
    
    # 创建系统实例
    print("\n初始化系统A...")
    try:
        system = SystemA_BaselineSequential(
            asr_model_size="base",
            llm_model_name="Qwen/Qwen1.5-0.5B-Chat"
        )
        print("✅ 系统初始化成功")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    # 测试简单的短音频
    print("\n测试短音频样本...")
    test_audio = system.get_data_path(
        experiment_name="core_comparison",
        length_group="short",
        sample_id="sample_001"
    )
    
    # 检查文件是否存在
    if not Path(test_audio).exists():
        print(f"⚠️ 音频文件不存在: {test_audio}")
        print("请先运行 fill_experiments_data.py 准备数据")
        return
    
    # 加载元数据
    metadata = system.load_sample_metadata(
        experiment_name="core_comparison",
        length_group="short",
        sample_id="sample_001"
    )
    
    print(f"音频文件: {Path(test_audio).name}")
    print(f"预期文本: {metadata.get('text', '未知')}")
    print(f"语言: {metadata.get('language', '未知')}")
    print(f"时长: {metadata.get('duration', 0):.1f}秒")
    
    # 测试各个组件
    print("\n1. 测试音频时长获取...")
    try:
        duration = system.get_audio_duration(test_audio)
        print(f"   ✅ 音频时长: {duration:.2f}秒")
    except Exception as e:
        print(f"   ❌ 获取音频时长失败: {e}")
        return
    
    print("\n2. 测试ASR处理...")
    try:
        transcript, asr_timing = system.process_audio_complete(test_audio)
        print(f"   ✅ ASR转录: {transcript[:50]}...")
        print(f"   处理时间: {asr_timing['asr_processing_time']:.2f}秒")
    except Exception as e:
        print(f"   ❌ ASR处理失败: {e}")
        print("   提示: 请确保fast-whisper模型已正确安装")
        return
    
    print("\n3. 测试LLM推理...")
    try:
        first_token, llm_timing = system.process_llm_inference(transcript)
        print(f"   ✅ 首Token: {first_token}")
        print(f"   生成时间: {llm_timing['llm_processing_time']:.3f}秒")
    except Exception as e:
        print(f"   ❌ LLM处理失败: {e}")
        print("   提示: 请确保Qwen模型已正确安装并可访问")
        return
    
    print("\n4. 测试完整流水线...")
    try:
        result = system.process_complete_pipeline(test_audio, simulate_delay=False)
        
        print(f"   ✅ 完整处理成功")
        print(f"   TTFT: {result['performance_metrics']['ttft_ms']:.1f}ms")
        print(f"   总处理时间: {result['performance_metrics']['total_processing_time_ms']:.1f}ms")
        
        # 验证结果结构
        assert result['system_name'] == 'SystemA_BaselineSequential'
        assert not result['performance_metrics']['has_streaming_asr']
        assert not result['performance_metrics']['has_kv_cache']
        assert result['performance_metrics']['processing_type'] == 'sequential'
        
        print("\n✅ 所有测试通过！系统A已准备就绪，可以进行实验。")
        
    except Exception as e:
        print(f"   ❌ 完整流水线处理失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_with_real_data()