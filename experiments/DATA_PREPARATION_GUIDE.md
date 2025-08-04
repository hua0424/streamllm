# 语音对话系统实验数据准备指南

## 📋 数据需求概述

基于《级联式语音对话系统的延迟优化》论文的7个核心实验，我们需要准备**用户语音问题**和**LLM回复**的完整对话数据：

### 1. **核心数据类型**
- **用户语音问题**: 不同长度的中文语音问题/指令
- **对话上下文**: 每个语音问题的文本转录
- **期望回复**: LLM对用户问题的标准回复(用于验证系统正常工作)
- **对话场景**: 涵盖多种真实对话场景

### 2. **数据规模要求**
- **总样本数**: 约180个高质量语音问题 (每个长度组30个)
- **语音长度**: 3s, 5s, 10s, 15s, 20s, 30s (模拟真实用户问题长度分布)
- **说话人多样性**: 至少8个不同说话人 (男女各半)
- **对话场景**: 信息查询、任务指令、闲聊互动、知识问答等

### 3. **关键特征要求**
- **自然性**: 语音问题应该是自然的对话式表达，而非朗读
- **完整性**: 每个语音都是一个完整的问题或指令
- **多样性**: 涵盖不同语调、语速、表达方式
- **可回复性**: LLM能够对每个问题生成有意义的回复

## 🔍 推荐数据源

### 🎯 专门的语音对话数据集

#### **1. 中文语音对话数据集 (推荐)**

**MultiWOZ中文版**
- **特点**: 多轮任务导向对话，包含语音版本
- **规模**: 约1万个对话，10万轮次
- **场景**: 订票、订餐、查询天气等真实对话场景
- **获取**: 通过学术合作或付费获取

**SpeechOcean762中文对话数据**
- **特点**: 自然对话语音，包含情感和语调变化
- **规模**: 约5000小时对话语音
- **质量**: 专业录制，高质量标注
- **许可**: 学术研究可申请

#### **2. 开源语音问答数据集**

**Chinese Speech QA Dataset**
- **特点**: 中文语音问答配对
- **内容**: 日常问题+语音回答
- **获取**: GitHub开源项目
- **适用性**: 直接可用于对话系统实验

**Aidatatang_200zh语音数据**
- **规模**: 200小时中文语音
- **特点**: 包含自然对话片段
- **获取**: [OpenSLR](http://www.openslr.org/62/)
- **优势**: 免费开源，质量较好

### 🔄 文本对话数据集(需TTS转换) - 主要数据源

#### **1. DuConv对话数据集 (重点推荐)**
- **来源**: 百度发布的中文对话数据集
- **规模**: 29万轮对话，包含知识问答
- **特点**: 
  - 知识驱动的问答对话
  - 用户问题+助手回复的完整配对
  - 涵盖百科知识、生活常识等
- **获取**: [DuConv](https://ai.baidu.com/broad/introduction?dataset=duconv)
- **适用场景**: 信息查询、知识问答类对话

#### **2. LCCC中文对话数据集**
- **规模**: 1200万轮对话 (选择高质量子集)
- **特点**: 
  - 闲聊对话为主
  - 自然对话表达
  - 多样化话题
- **获取**: [GitHub](https://github.com/thu-coai/CDial-GPT)
- **筛选策略**: 选择短问题(3-30秒语音长度)且有明确回复的对话

#### **3. KdConv知识对话数据集**
- **特点**: 基于知识图谱的对话数据
- **规模**: 约4.5万个多轮对话
- **场景**: 电影、音乐、旅游等领域的问答
- **获取**: [GitHub](https://github.com/thu-coai/KdConv)
- **优势**: 问题明确，回复质量高

#### **4. Chinese-ChatBot-Corpus**
- **特点**: 多领域中文对话语料
- **内容**: 
  - 客服对话 (任务导向)
  - 闲聊对话 (开放域)
  - 问答对话 (知识型)
- **获取**: [GitHub](https://github.com/codemayq/chinese_chatbot_corpus)

#### **5. 自构建问题数据集**
基于常见对话场景手工设计:
- **日常查询**: "现在几点了？"、"今天天气怎么样？"
- **任务指令**: "帮我设置明天8点的闹钟"、"播放一首轻音乐"
- **知识问答**: "北京的人口有多少？"、"怎么做宫保鸡丁？"
- **情感交流**: "我今天心情不好"、"你觉得这个怎么样？"

## 📊 数据准备清单

### 阶段1: 对话数据收集 (第1-2天)

#### **优先级1: 文本对话数据收集**
- [ ] **DuConv数据集下载** (重点 - 29万轮知识问答)
  ```bash
  # 申请下载DuConv数据集
  # 筛选单轮问答对话
  python extract_qa_pairs.py --input duconv.json --output qa_pairs.json
  ```
- [ ] **LCCC数据集获取** (选择高质量子集)
  ```bash
  git clone https://github.com/thu-coai/CDial-GPT
  # 筛选适合的短问题
  python filter_short_questions.py --max_length 30
  ```
- [ ] **KdConv知识对话数据**
  ```bash
  git clone https://github.com/thu-coai/KdConv
  # 提取用户问题部分
  python extract_user_questions.py
  ```

#### **优先级2: 现有语音对话数据收集**
- [ ] **Aidatatang_200zh下载** (免费开源)
  ```bash
  wget http://www.openslr.org/resources/62/aidatatang_200zh.tgz
  # 从中筛选对话片段
  ```
- [ ] **寻找开源语音问答数据**
- [ ] **申请学术语音对话数据集访问权限**

#### **优先级3: 自构建问题数据集**
- [ ] **设计常见对话场景问题** (180个)
  - 信息查询类 (30个): "现在几点？"、"天气如何？"
  - 任务指令类 (30个): "设置闹钟"、"播放音乐"
  - 知识问答类 (60个): "北京人口？"、"如何做菜？"
  - 闲聊互动类 (30个): "今天心情"、"推荐电影"
  - 功能操作类 (30个): "打开应用"、"发送消息"
- [ ] **问题长度分布设计**
  - 3-5秒: 短问题 (30个)
  - 5-8秒: 中短问题 (30个)  
  - 8-12秒: 中等问题 (30个)
  - 12-18秒: 中长问题 (30个)
  - 18-25秒: 长问题 (30个)
  - 25-35秒: 很长问题 (30个)

### 阶段2: 问题筛选和文本处理 (第3-4天)

#### **对话数据筛选和预处理**
- [ ] **从DuConv提取合适的用户问题**
  ```python
  def extract_user_questions(duconv_data):
      questions = []
      for dialogue in duconv_data:
          for turn in dialogue['conversation']:
              if turn['role'] == 'user':
                  question = turn['utterance']
                  if is_suitable_question(question):  # 3-35秒语音长度
                      questions.append({
                          'text': question,
                          'estimated_duration': estimate_speech_duration(question),
                          'category': classify_question_type(question),
                          'expected_response': find_assistant_response(dialogue, turn)
                      })
      return questions
  ```

- [ ] **问题质量筛选标准**
  ```python
  def is_suitable_question(question):
      # 长度检查 (估算语音3-35秒)
      if not (10 <= len(question) <= 100):
          return False
      
      # 完整性检查 (完整问题或指令)
      if not (question.endswith('？') or question.endswith('?') or 
              is_command(question) or is_request(question)):
          return False
      
      # 可回复性检查
      if is_answerable_by_llm(question):
          return True
      
      return False
  ```

- [ ] **按长度和类型分组**
  ```python
  def categorize_questions(questions):
      categories = {
          'short_3_5s': [],     # 3-5秒问题
          'medium_5_10s': [],   # 5-10秒问题  
          'long_10_15s': [],    # 10-15秒问题
          'very_long_15s+': []  # 15秒以上问题
      }
      
      for q in questions:
          duration = q['estimated_duration']
          if duration <= 5:
              categories['short_3_5s'].append(q)
          elif duration <= 10:
              categories['medium_5_10s'].append(q)
          elif duration <= 15:
              categories['long_10_15s'].append(q)
          else:
              categories['very_long_15s+'].append(q)
      
      return categories
  ```

#### **创建实验数据结构**
- [ ] **标准化数据格式**
  ```json
  {
    "question_id": "q_001",
    "text": "请问今天北京的天气怎么样？",
    "estimated_duration": 4.2,
    "category": "information_query",
    "difficulty": "easy",
    "expected_response": "今天北京多云，气温18-25度，适宜出行。",
    "keywords": ["天气", "北京", "查询"],
    "speaker_requirements": {
      "gender": "any",
      "age": "adult",
      "accent": "standard"
    }
  }
  ```

- [ ] **验证LLM回复能力**
  ```python
  def test_llm_response(question_text):
      # 使用项目中的LLM测试是否能正常回复
      from src.llm.stream_llm_inference import StreamLLMInference
      
      llm = StreamLLMInference()
      response = llm.generate_response(question_text)
      
      # 检查回复质量
      if len(response) > 10 and not is_nonsense(response):
          return True, response
      return False, None
  ```

### 阶段3: 语音问题生成 (第5-6天)

#### **TTS工具选择 - 针对对话场景优化**

**推荐方案1: 微软Azure TTS (首选)**
- **优势**: 
  - 高质量自然语音
  - 多种中文音色 (男女老少)
  - 支持情感和语调调节
  - 适合对话场景
- **成本**: 每月免费500万字符 (约可生成50小时音频)
- **对话场景优化实现**:
  ```python
  import azure.cognitiveservices.speech as speechsdk
  
  def generate_question_speech(question_data, output_file):
      speech_config = speechsdk.SpeechConfig(
          subscription="your_key", 
          region="your_region"
      )
      
      # 根据问题类型选择合适音色和语调
      voice_config = select_voice_for_question(question_data)
      speech_config.speech_synthesis_voice_name = voice_config['voice']
      
      # 使用SSML控制语调和语速
      ssml_text = f"""
      <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
          <voice name="{voice_config['voice']}">
              <prosody rate="{voice_config['rate']}" pitch="{voice_config['pitch']}">
                  {question_data['text']}
              </prosody>
          </voice>
      </speak>
      """
      
      audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
      synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
      
      result = synthesizer.speak_ssml_async(ssml_text).get()
      return result.reason == speechsdk.ResultReason.SynthesizingSpeechCompleted
  
  def select_voice_for_question(question_data):
      """根据问题类型选择合适的语音配置"""
      voice_configs = {
          'information_query': {
              'voice': 'zh-CN-XiaohanNeural',  # 女声，清晰
              'rate': '1.0',
              'pitch': 'medium'
          },
          'task_command': {
              'voice': 'zh-CN-YunxiNeural',    # 男声，稳重
              'rate': '0.9',
              'pitch': 'low'
          },
          'casual_chat': {
              'voice': 'zh-CN-XiaoxiaoNeural', # 女声，活泼
              'rate': '1.1', 
              'pitch': 'high'
          },
          'knowledge_qa': {
              'voice': 'zh-CN-YunyangNeural',  # 男声，正式
              'rate': '0.95',
              'pitch': 'medium'
          }
      }
      return voice_configs.get(question_data['category'], voice_configs['information_query'])
  ```

**推荐方案2: PaddleSpeech (备选)**
- **优势**: 
  - 百度开源，中文效果好
  - 免费使用
  - 支持本地部署
- **对话场景配置**:
  ```python
  from paddlespeech.cli.tts import TTSExecutor
  
  def generate_natural_question_speech(text, output_path, speaker_id=0):
      tts = TTSExecutor()
      
      # 使用不同说话人ID模拟多样性
      tts(text=text, 
          output=output_path,
          am='fastspeech2_csmsc',
          am_config=None,
          am_ckpt=None,
          am_stat=None,
          spk_id=speaker_id,  # 0-99不同说话人
          lang='zh',
          device='gpu')
  ```

#### **批量语音问题生成流程**
- [ ] **问题文本预处理和验证**
  ```python
  def prepare_questions_for_tts(question_list):
      validated_questions = []
      for q in question_list:
          # 验证问题完整性
          if not is_complete_question(q['text']):
              continue
              
          # 估算语音时长
          estimated_duration = estimate_speech_duration(q['text'])
          if not (3 <= estimated_duration <= 35):
              continue
          
          # 测试LLM是否能回复
          can_reply, sample_response = test_llm_response(q['text'])
          if not can_reply:
              continue
              
          q['validated'] = True
          q['estimated_duration'] = estimated_duration
          q['sample_response'] = sample_response
          validated_questions.append(q)
      
      return validated_questions
  ```

- [ ] **按长度和场景分组生成**
  ```python
  def batch_generate_dialogue_audio(questions, output_dir):
      # 按长度分组
      length_groups = {
          'length_3s': [q for q in questions if 3 <= q['estimated_duration'] < 5],
          'length_5s': [q for q in questions if 5 <= q['estimated_duration'] < 8], 
          'length_10s': [q for q in questions if 8 <= q['estimated_duration'] < 12],
          'length_15s': [q for q in questions if 12 <= q['estimated_duration'] < 18],
          'length_20s': [q for q in questions if 18 <= q['estimated_duration'] < 25],
          'length_30s': [q for q in questions if 25 <= q['estimated_duration'] <= 35]
      }
      
      voice_rotation = [
          "zh-CN-XiaoxiaoNeural",  # 女声，活泼
          "zh-CN-YunxiNeural",     # 男声，稳重
          "zh-CN-YunyangNeural",   # 男声，正式
          "zh-CN-XiaohanNeural",   # 女声，清晰
          "zh-CN-XiaochenNeural",  # 女声，温和
          "zh-CN-YunjianNeural",   # 男声，亲切
          "zh-CN-XiaoruiNeural",   # 女声，专业
          "zh-CN-XiaoyuNeural"     # 女声，自然
      ]
      
      for length_group, group_questions in length_groups.items():
          group_dir = Path(output_dir) / length_group
          group_dir.mkdir(exist_ok=True)
          
          for i, question in enumerate(group_questions[:30]):  # 每组最多30个
              voice = voice_rotation[i % len(voice_rotation)]
              output_file = group_dir / f"question_{i:03d}.wav"
              
              # 生成语音
              success = generate_question_speech_with_voice(
                  question, output_file, voice
              )
              
              if success:
                  # 保存元数据
                  metadata_file = group_dir / f"question_{i:03d}.json"
                  save_question_metadata(question, metadata_file, {
                      'audio_file': str(output_file),
                      'voice_used': voice,
                      'actual_duration': get_audio_duration(output_file)
                  })
  ```

- [ ] **生成对话系统测试用的标准问题集**
  ```python
  def create_standard_dialogue_test_set():
      """创建涵盖各种对话场景的标准测试问题"""
      
      standard_questions = {
          # 信息查询类 (30个)
          'information_query': [
              "现在几点了？",
              "今天天气怎么样？", 
              "北京到上海有多远？",
              "明天是星期几？",
              "人民币对美元的汇率是多少？",
              # ... 更多问题
          ],
          
          # 任务指令类 (30个)  
          'task_command': [
              "请帮我设置明天早上8点的闹钟",
              "播放一首轻音乐",
              "把音量调到50%",
              "发送短信给张三说我晚点到",
              "提醒我下午3点开会",
              # ... 更多指令
          ],
          
          # 知识问答类 (60个)
          'knowledge_qa': [
              "什么是人工智能？",
              "中国有多少个省份？", 
              "怎么做宫保鸡丁？",
              "牛顿发现了什么定律？",
              "如何学好英语？",
              # ... 更多问答
          ],
          
          # 闲聊互动类 (30个)
          'casual_chat': [
              "我今天心情有点不好",
              "你觉得这个电影怎么样？",
              "推荐一本好书给我",
              "聊聊最近的新闻吧",
              "你有什么爱好吗？",
              # ... 更多闲聊
          ],
          
          # 功能操作类 (30个)
          'function_control': [
              "打开微信",
              "关闭蓝牙",
              "连接WiFi",
              "截个屏", 
              "清理手机垃圾",
              # ... 更多操作
          ]
      }
      
      return standard_questions
  ```

### 阶段4: 音频质量处理 (第7天)

#### **噪声版本生成**
- [ ] **添加背景噪声**
  ```python
  import librosa
  import numpy as np
  
  def add_noise(audio_path, noise_level_db):
      y, sr = librosa.load(audio_path, sr=16000)
      
      # 生成白噪声
      noise = np.random.normal(0, 1, len(y))
      
      # 计算信噪比
      signal_power = np.mean(y ** 2)
      noise_power = signal_power / (10 ** (noise_level_db / 10))
      noise = noise * np.sqrt(noise_power)
      
      # 添加噪声
      noisy_audio = y + noise
      return noisy_audio
  ```

- [ ] **生成不同SNR版本**
  - 30dB SNR (轻微噪声)
  - 20dB SNR (中等噪声)  
  - 10dB SNR (较强噪声)

#### **音频格式多样化**
- [ ] **生成不同采样率版本**
  ```bash
  # 8kHz版本 (电话质量)
  ffmpeg -i input.wav -ar 8000 output_8k.wav
  
  # 44.1kHz版本 (高质量)
  ffmpeg -i input.wav -ar 44100 output_44k.wav
  ```

- [ ] **压缩格式测试**
  ```bash
  # MP3压缩
  ffmpeg -i input.wav -b:a 128k output.mp3
  ```

### 阶段5: 对话数据组织和验证 (第8天)

#### **对话系统数据目录结构**
```
experiments/datasets/
├── dialogue_questions/        # 用户语音问题 (主要数据)
│   ├── length_3s/            # 3-5秒问题 (30个)
│   │   ├── question_001.wav
│   │   ├── question_001.json  # 元数据
│   │   └── ...
│   ├── length_5s/            # 5-8秒问题 (30个)
│   ├── length_10s/           # 8-12秒问题 (30个)
│   ├── length_15s/           # 12-18秒问题 (30个)
│   ├── length_20s/           # 18-25秒问题 (30个)
│   └── length_30s/           # 25-35秒问题 (30个)
├── quality_variants/         # 音频质量变体
│   ├── clean/               # 清晰版本 (基准)
│   ├── noise_30db/          # 30dB噪声环境
│   ├── noise_20db/          # 20dB噪声环境
│   └── noise_10db/          # 10dB噪声环境
├── format_variants/         # 格式变体 (用于鲁棒性测试)
│   ├── wav_16k/            # 16kHz WAV (标准)
│   ├── wav_8k/             # 8kHz WAV (电话质量)
│   ├── wav_44k/            # 44.1kHz WAV (高质量)
│   └── mp3_128k/           # 128k MP3 (压缩)
├── dialogue_context/        # 对话上下文和期望回复
│   ├── question_transcripts.json  # 问题转录文本
│   ├── expected_responses.json    # LLM期望回复
│   └── dialogue_scenarios.json   # 对话场景分类
├── metadata/               # 元数据
│   ├── question_catalog.json     # 问题目录
│   ├── speaker_profiles.json     # 说话人信息
│   ├── scenario_mapping.json     # 场景映射
│   └── validation_results.json   # 验证结果
├── test_splits/            # 实验数据分组
│   ├── length_impact_test.json      # 实验1: 长度影响
│   ├── model_comparison_test.json   # 实验2: 模型对比
│   ├── ablation_study_test.json     # 实验3: 消融实验
│   ├── quality_robustness_test.json # 实验5: 质量鲁棒性
│   └── concurrent_test.json         # 实验7: 并发测试
└── validation/             # 数据验证
    ├── llm_response_test.json       # LLM回复测试结果
    ├── asr_accuracy_test.json       # ASR识别准确率
    └── dialogue_quality_report.json # 对话质量报告
```

#### **对话数据元数据格式**
```json
// question_001.json 示例
{
    "question_id": "q_001",
    "audio_file": "question_001.wav", 
    "text": "请问今天北京的天气怎么样？",
    "category": "information_query",
    "subcategory": "weather_query",
    "estimated_duration": 4.2,
    "actual_duration": 4.1,
    "difficulty": "easy",
    "speaker_info": {
        "voice_model": "zh-CN-XiaohanNeural",
        "gender": "female",
        "age_group": "adult",
        "accent": "standard"
    },
    "expected_response": {
        "sample_response": "今天北京多云，气温18-25度，适宜出行。",
        "response_type": "informative",
        "llm_tested": true,
        "response_quality": "good"
    },
    "keywords": ["天气", "北京", "查询"],
    "validation": {
        "is_complete_question": true,
        "is_answerable": true,
        "asr_confidence": 0.95,
        "audio_quality": "excellent"
    },
    "experiment_usage": ["length_impact", "model_comparison", "quality_test"]
}
```

#### **对话数据验证脚本**
- [ ] **对话完整性检查**
  ```python
  def validate_dialogue_dataset():
      issues = []
      
      # 检查问题-回复配对完整性
      for question_file in get_all_question_files():
          metadata = load_question_metadata(question_file)
          
          # 检查音频文件存在性
          if not Path(metadata['audio_file']).exists():
              issues.append(f"Missing audio file: {metadata['audio_file']}")
          
          # 检查问题文本质量
          if not is_valid_question(metadata['text']):
              issues.append(f"Invalid question: {metadata['question_id']}")
          
          # 检查LLM回复测试
          if not metadata.get('expected_response', {}).get('llm_tested'):
              issues.append(f"LLM not tested: {metadata['question_id']}")
      
      # 检查长度分布平衡性
      length_distribution = analyze_question_length_distribution()
      for length_group, count in length_distribution.items():
          if count < 25:  # 每组至少25个问题
              issues.append(f"Insufficient questions in {length_group}: {count}")
      
      # 检查对话场景覆盖度
      scenario_coverage = analyze_scenario_coverage()
      required_scenarios = ['information_query', 'task_command', 'knowledge_qa', 
                           'casual_chat', 'function_control']
      for scenario in required_scenarios:
          if scenario_coverage.get(scenario, 0) < 10:
              issues.append(f"Insufficient {scenario} questions")
      
      return issues
  ```

- [ ] **对话质量评估**
  ```python
  def assess_dialogue_quality():
      quality_report = {
          'audio_quality': {},
          'llm_compatibility': {},
          'dialogue_naturalness': {},
          'scenario_balance': {}
      }
      
      for question_file in get_all_question_files():
          metadata = load_question_metadata(question_file)
          question_id = metadata['question_id']
          
          # 音频质量评估
          audio_quality = assess_single_audio_quality(metadata['audio_file'])
          quality_report['audio_quality'][question_id] = audio_quality
          
          # LLM兼容性测试
          llm_compatibility = test_llm_compatibility(metadata['text'])
          quality_report['llm_compatibility'][question_id] = llm_compatibility
          
          # 对话自然度评估
          naturalness = assess_question_naturalness(metadata['text'])
          quality_report['dialogue_naturalness'][question_id] = naturalness
      
      # 场景平衡性分析
      quality_report['scenario_balance'] = analyze_scenario_balance()
      
      return quality_report
  
  def test_llm_compatibility(question_text):
      """测试问题与LLM的兼容性"""
      try:
          from src.llm.stream_llm_inference import StreamLLMInference
          
          llm = StreamLLMInference()
          response = llm.generate_response(question_text)
          
          return {
              'can_respond': len(response) > 10,
              'response_quality': rate_response_quality(response),
              'response_time': measure_response_time(llm, question_text),
              'sample_response': response[:100] + '...' if len(response) > 100 else response
          }
      except Exception as e:
          return {'can_respond': False, 'error': str(e)}
  
  def assess_question_naturalness(question_text):
      """评估问题的自然度"""
      scores = {
          'is_complete': ends_with_question_mark_or_command(question_text),
          'has_natural_flow': has_natural_language_flow(question_text),
          'appropriate_length': 10 <= len(question_text) <= 100,
          'uses_colloquial_language': uses_natural_expressions(question_text),
          'clear_intent': has_clear_intent(question_text)
      }
      
      naturalness_score = sum(scores.values()) / len(scores)
      return {
          'score': naturalness_score,
          'details': scores,
          'recommendation': 'good' if naturalness_score > 0.8 else 'needs_improvement'
      }
  ```

## 🛠️ 数据处理工具脚本

### 主要处理脚本

#### **1. 音频分割脚本**
```python
# experiments/data_preparation/audio_segmentation.py
import librosa
import soundfile as sf
from pathlib import Path

def segment_audio_by_length(input_dir, output_dir, target_lengths):
    """按目标长度分割音频"""
    for target_length in target_lengths:
        output_subdir = Path(output_dir) / f"length_{target_length}s"
        output_subdir.mkdir(exist_ok=True)
        
        for audio_file in Path(input_dir).glob("*.wav"):
            segments = extract_segments(audio_file, target_length)
            for i, segment in enumerate(segments):
                output_file = output_subdir / f"{audio_file.stem}_{i:03d}.wav"
                sf.write(output_file, segment, 16000)
```

#### **2. TTS批量生成脚本**
```python
# experiments/data_preparation/batch_tts.py
from concurrent.futures import ThreadPoolExecutor
import json

def batch_generate_tts(text_data, output_dir, max_workers=4):
    """批量生成TTS音频"""
    
    def generate_single(item):
        text, output_path, voice_style = item
        return text_to_speech(text, output_path, voice_style)
    
    tasks = prepare_tts_tasks(text_data, output_dir)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(generate_single, tasks))
    
    return results
```

#### **3. 数据集构建脚本**
```python
# experiments/data_preparation/build_dataset.py
def build_experiment_dataset(config):
    """构建完整的实验数据集"""
    
    # 1. 收集和处理原始音频
    process_raw_audio(config['raw_audio_dir'])
    
    # 2. 生成TTS音频
    generate_tts_audio(config['text_data'])
    
    # 3. 创建质量变体
    create_quality_variants(config['base_audio_dir'])
    
    # 4. 组织目录结构
    organize_dataset_structure(config['output_dir'])
    
    # 5. 生成元数据
    generate_metadata(config['output_dir'])
    
    # 6. 验证数据集
    validation_report = validate_dataset(config['output_dir'])
    
    return validation_report
```

## 📈 对话数据质量保证

### 对话系统专用质量控制标准
- **问题完整性**: 每个语音都是完整的问题或指令
- **LLM兼容性**: 100%的问题能够获得LLM有意义回复
- **音频质量**: SNR > 20dB, 清晰度评分 > 4.0/5.0
- **时长准确性**: 实际语音时长与预估误差 < ±15%
- **对话自然度**: 自然度评分 > 0.8/1.0
- **说话人多样性**: 8种不同音色，男女比例均衡
- **场景覆盖**: 5大对话场景，每个场景至少30个问题

### 对话数据验收标准
- [ ] **数量要求**: 每个长度组至少25个高质量问题
- [ ] **质量要求**: 
  - [ ] 问题文本语法正确，表达自然
  - [ ] 所有问题经过LLM回复测试，回复质量良好
  - [ ] 音频清晰，无明显噪声或失真
- [ ] **格式要求**:
  - [ ] 音频格式统一 (16kHz, 单声道, WAV)
  - [ ] 元数据文件完整且JSON格式正确
  - [ ] 文件命名规范统一
- [ ] **实验适用性**:
  - [ ] 覆盖7个实验所需的全部测试场景
  - [ ] 数据分组清晰，便于不同实验使用
  - [ ] 通过自动化质量检测脚本验证

## 💰 成本估算

### TTS服务成本
- **Azure TTS**: 免费500万字符/月，约可生成50小时音频
- **Google Cloud TTS**: $4/100万字符
- **开源方案**: 仅硬件成本，约需要8-16小时处理时间

### 人工成本
- **数据收集**: 1-2天
- **质量检查**: 1天  
- **标注验证**: 0.5天

### 存储成本
- **原始数据**: 约5-10GB
- **处理后数据**: 约2-3GB
- **备份存储**: 建议云端备份

## ⏰ 时间规划

### 总体时间: 8-10天

1. **第1-2天**: 数据源调研和下载
2. **第3-4天**: 音频预处理和分类
3. **第5-6天**: TTS生成和质量处理
4. **第7天**: 音频质量变体生成
5. **第8天**: 数据组织和验证
6. **第9-10天**: 问题修复和最终验收

## 🚀 对话数据快速开始命令

```bash
# 1. 创建对话数据目录结构
mkdir -p experiments/datasets/{dialogue_questions,quality_variants,dialogue_context,metadata,test_splits,validation}
mkdir -p experiments/datasets/dialogue_questions/{length_3s,length_5s,length_10s,length_15s,length_20s,length_30s}

# 2. 下载和处理对话数据集
bash experiments/data_preparation/download_dialogue_datasets.sh

# 3. 从对话数据集提取问题
python experiments/data_preparation/extract_dialogue_questions.py \
    --duconv_path data/duconv.json \
    --lccc_path data/lccc.json \
    --output_dir experiments/datasets/dialogue_questions

# 4. 验证问题与LLM兼容性
python experiments/data_preparation/test_llm_compatibility.py \
    --questions_dir experiments/datasets/dialogue_questions \
    --llm_model "Qwen/Qwen1.5-0.5B-Chat"

# 5. 批量生成语音问题
python experiments/data_preparation/batch_generate_dialogue_audio.py \
    --questions_file experiments/datasets/dialogue_context/validated_questions.json \
    --output_dir experiments/datasets/dialogue_questions \
    --tts_service azure  # 或 paddlespeech

# 6. 生成质量变体 (噪声、格式变体)
python experiments/data_preparation/generate_audio_variants.py \
    --input_dir experiments/datasets/dialogue_questions \
    --output_dir experiments/datasets/quality_variants

# 7. 验证完整对话数据集
python experiments/data_preparation/validate_dialogue_dataset.py \
    --dataset_dir experiments/datasets

# 8. 为实验生成测试分组
python experiments/data_preparation/create_experiment_splits.py \
    --dataset_dir experiments/datasets \
    --output_dir experiments/datasets/test_splits
```

### 单步执行选项

```bash
# 仅生成标准问题集 (如果无法获取公开数据集)
python experiments/data_preparation/create_standard_questions.py \
    --output_file experiments/datasets/dialogue_context/standard_questions.json \
    --num_questions_per_category 30

# 仅测试现有音频与LLM兼容性
python experiments/data_preparation/test_existing_audio_llm_compatibility.py \
    --audio_dir data/processed_audio \
    --llm_model "Qwen/Qwen1.5-0.5B-Chat"

# 快速验证数据质量
python experiments/data_preparation/quick_quality_check.py \
    --dataset_dir experiments/datasets/dialogue_questions
```

## 📞 技术支持

### 遇到问题时的处理流程
1. 检查 `experiments/logs/data_preparation.log`
2. 参考 `experiments/data_preparation/troubleshooting.md`
3. 在项目Issues中搜索类似问题
4. 提交新的Issue包含详细错误信息

---

**注意**: 数据准备是实验成功的关键，建议提前2-3周开始准备，确保有充足时间处理各种意外情况。