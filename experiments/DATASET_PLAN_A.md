# 单轮对话数据集方案A - 详细获取指南

## 📋 方案概述

**目标**: 180个高质量单轮语音问题，分布如下：
- **WebQA事实问答**: 60个问题
- **CrossWOZ任务导向**: 40个问题  
- **LCCC日常闲聊**: 40个问题
- **JDDC客服场景**: 40个问题

## 🔗 数据集详细获取方式

### **1. WebQA事实问答数据集 (60个问题) ⭐⭐⭐⭐⭐**

#### **基本信息**
- **来源**: 百度发布的中文问答数据集
- **规模**: 42万个问答对
- **许可**: 学术研究免费使用
- **特点**: 
  - 单轮问答格式，问题简洁明确
  - 涵盖事实性问题、常识问题  
  - 问题长度分布合理(3-30秒语音)

#### **获取步骤**
```bash
# 1. GitHub下载
git clone https://github.com/WebQA-dataset/WebQA.git
cd WebQA

# 2. 数据文件位置
# me_train.ann.json - 训练集
# me_validation.ann.json - 验证集
# me_test.ann.json - 测试集

# 3. 查看数据格式
head -5 me_train.ann.json
```

#### **数据格式示例**
```json
{
    "qid": "train_0001",
    "question": "北京的人口有多少？",
    "answer": "截至2021年，北京市常住人口约2189万人。",
    "evidences": [
        {
            "evidence": "根据北京市统计局数据...",
            "url": "http://example.com"
        }
    ]
}
```

#### **筛选策略**
```python
def filter_webqa_questions(webqa_data):
    """从WebQA中筛选适合语音对话的问题"""
    suitable_questions = []
    
    for item in webqa_data:
        question = item['question']
        answer = item['answer']
        
        # 长度筛选 (估算3-30秒语音)
        if 5 <= len(question) <= 50:
            # 问题完整性检查
            if question.endswith('？') or question.endswith('?'):
                # 答案质量检查
                if len(answer) > 10 and len(answer) < 200:
                    suitable_questions.append({
                        'question': question,
                        'expected_answer': answer,
                        'category': 'factual_qa',
                        'source': 'webqa',
                        'qid': item['qid']
                    })
    
    return suitable_questions[:60]  # 取前60个
```

---

### **2. CrossWOZ任务导向对话数据 (40个问题) ⭐⭐⭐⭐**

#### **基本信息**
- **来源**: 清华大学发布的跨领域任务导向对话
- **规模**: 1万个对话，可提取单轮请求
- **许可**: MIT License，学术商用免费
- **特点**:
  - 包含酒店、餐厅、景点、地铁等场景
  - 单轮用户请求清晰明确
  - 中文表达自然，贴近真实使用

#### **获取步骤**
```bash
# 1. GitHub下载
git clone https://github.com/thu-coai/CrossWOZ.git
cd CrossWOZ

# 2. 数据文件
# data/crosswoz/train.json - 训练集 (8K对话)
# data/crosswoz/val.json - 验证集 (1K对话)  
# data/crosswoz/test.json - 测试集 (1K对话)

# 3. 查看数据结构
python -c "import json; print(json.load(open('data/crosswoz/train.json'))[:1])"
```

#### **数据提取脚本**
```python
def extract_crosswoz_requests(crosswoz_file):
    """从CrossWOZ提取用户任务请求"""
    import json
    
    with open(crosswoz_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    user_requests = []
    
    for dialogue_id, dialogue in data.items():
        messages = dialogue['messages']
        goal = dialogue.get('goal', {})
        
        for i, message in enumerate(messages):
            if message['role'] == 'usr':
                content = message['content']
                
                # 筛选任务导向的请求
                if is_task_oriented_request(content):
                    # 找到对应的系统回复
                    system_response = None
                    if i + 1 < len(messages) and messages[i + 1]['role'] == 'sys':
                        system_response = messages[i + 1]['content']
                    
                    user_requests.append({
                        'question': content,
                        'expected_response': system_response,
                        'domain': extract_domain_from_goal(goal),
                        'intent': extract_intent(content),
                        'category': 'task_oriented',
                        'source': 'crosswoz',
                        'dialogue_id': dialogue_id
                    })
    
    return user_requests[:40]  # 取前40个

def is_task_oriented_request(text):
    """判断是否为任务导向请求"""
    task_keywords = [
        '预订', '查询', '推荐', '帮我', '我想', '需要',
        '酒店', '餐厅', '景点', '地铁', '出行'
    ]
    return any(keyword in text for keyword in task_keywords)
```

---

### **3. LCCC日常闲聊数据 (40个问题) ⭐⭐⭐⭐**

#### **基本信息**
- **来源**: 清华大学LCCC数据集的高质量子集
- **规模**: 从1200万轮对话中选择单轮高质量对话
- **许可**: 学术研究免费，需申请
- **特点**:
  - 社交媒体真实对话
  - 表达自然，贴近口语
  - 话题多样化，情感丰富

#### **获取步骤**
```bash
# 1. GitHub下载代码
git clone https://github.com/thu-coai/CDial-GPT.git
cd CDial-GPT

# 2. 申请数据访问权限
# 访问: https://ai.tencent.com/ailab/nlp/dialogue/#datasets
# 填写申请表格，说明学术研究用途
# 通常1-3个工作日获得下载链接

# 3. 下载数据文件
# LCCC-base.json - 基础版本 (680万轮对话)
# LCCC-large.json - 完整版本 (1200万轮对话)
wget [申请获得的下载链接] -O LCCC-base.json
```

#### **数据筛选脚本**
```python
def filter_lccc_casual_chat(lccc_file):
    """从LCCC筛选适合的闲聊问题"""
    import json
    
    casual_questions = []
    
    with open(lccc_file, 'r', encoding='utf-8') as f:
        for line in f:
            dialogue = json.loads(line.strip())
            
            # 只处理单轮对话
            if len(dialogue) == 2:
                user_msg = dialogue[0]
                bot_msg = dialogue[1]
                
                if (is_suitable_casual_question(user_msg) and 
                    is_reasonable_response(bot_msg)):
                    casual_questions.append({
                        'question': user_msg,
                        'expected_response': bot_msg,
                        'category': 'casual_chat',
                        'source': 'lccc'
                    })
                
                if len(casual_questions) >= 40:
                    break
    
    return casual_questions

def is_suitable_casual_question(text):
    """判断是否为合适的闲聊问题"""
    # 长度检查
    if not (5 <= len(text) <= 50):
        return False
    
    # 内容过滤
    exclude_patterns = [
        r'http[s]?://',  # 包含链接
        r'@\w+',         # @用户名
        r'#\w+#',        # 话题标签
        r'[0-9]{11}',    # 手机号
    ]
    
    for pattern in exclude_patterns:
        if re.search(pattern, text):
            return False
    
    # 闲聊特征检查
    casual_indicators = [
        '怎么样', '觉得', '喜欢', '推荐', '今天', 
        '最近', '心情', '电影', '音乐', '书'
    ]
    
    return any(indicator in text for indicator in casual_indicators)
```

---

### **4. JDDC客服对话数据 (40个问题) ⭐⭐⭐⭐**

#### **基本信息**
- **来源**: 京东客服对话数据集
- **规模**: 约100万轮对话
- **许可**: 学术研究免费使用
- **特点**:
  - 真实客服场景对话
  - 用户问题明确具体
  - 适合任务导向对话测试

#### **获取步骤**
```bash
# 1. GitHub下载
git clone https://github.com/jd-aig/nlp_baai.git
cd nlp_baai/jddc2019_task1

# 2. 数据文件
# train.txt - 训练集
# dev.txt - 验证集
# test.txt - 测试集

# 3. 查看数据格式
head -10 train.txt
```

#### **数据格式示例**
```text
# train.txt 格式
1	我想查询订单状态	好的，请提供您的订单号
2	这个商品什么时候发货	您的订单将在24小时内发货
3	如何申请退货	您可以在订单详情页面申请退货
```

#### **提取脚本**
```python
def extract_jddc_questions(jddc_file):
    """从JDDC提取客服场景问题"""
    service_questions = []
    
    with open(jddc_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                session_id = parts[0]
                user_question = parts[1]
                service_response = parts[2]
                
                if is_valid_service_question(user_question):
                    service_questions.append({
                        'question': user_question,
                        'expected_response': service_response,
                        'category': 'customer_service',
                        'source': 'jddc',
                        'session_id': session_id
                    })
                
                if len(service_questions) >= 40:
                    break
    
    return service_questions

def is_valid_service_question(text):
    """判断是否为有效的客服问题"""
    # 长度检查
    if not (5 <= len(text) <= 60):
        return False
    
    # 客服场景关键词
    service_keywords = [
        '订单', '查询', '退货', '换货', '发货', '物流',
        '支付', '价格', '优惠', '客服', '投诉', '建议'
    ]
    
    return any(keyword in text for keyword in service_keywords)
```

---

## 🔧 人工构建补充数据策略

### **补充原则**
根据四个数据集筛选结果，针对性补充缺失的长度组或场景类型。

### **长度分布目标**
```python
target_distribution = {
    'length_3s': 30,   # 3-5秒: 简短问题
    'length_5s': 30,   # 5-8秒: 中短问题  
    'length_10s': 30,  # 8-12秒: 中等问题
    'length_15s': 30,  # 12-18秒: 中长问题
    'length_20s': 30,  # 18-25秒: 长问题
    'length_30s': 30   # 25-35秒: 很长问题
}
```

### **补充流程**
1. **分析筛选结果**: 统计四个数据集筛选后的长度分布
2. **识别缺失**: 找出不足25个问题的长度组
3. **场景平衡**: 确保每种对话场景都有充足覆盖
4. **人工构建**: 针对缺失部分设计问题

### **人工构建模板库**
```python
manual_construction_templates = {
    # 3-5秒短问题 (预期语音长度)
    'short_3_5s': {
        'factual': [
            "现在几点？", "今天星期几？", "天气如何？",
            "北京在哪里？", "一加一等于几？"
        ],
        'task': [
            "开灯", "关门", "播放音乐", "暂停播放", 
            "调大音量", "设置闹钟"
        ],
        'casual': [
            "你好", "再见", "谢谢你", "怎么样？",
            "还好吗？", "在忙什么？"
        ],
        'service': [
            "找客服", "要投诉", "申请退货", "查订单",
            "联系人工", "转接客服"
        ]
    },
    
    # 5-8秒中短问题
    'medium_5_8s': {
        'factual': [
            "北京今天天气怎么样？", "人民币汇率是多少？",
            "中国有多少个省份？", "明天是几号？"
        ],
        'task': [
            "帮我设置明天的闹钟", "播放一首轻音乐",
            "查一下明天的天气", "发消息给张三"
        ],
        'casual': [
            "最近过得怎么样？", "有什么好电影推荐？",
            "今天心情如何？", "周末有什么计划？"
        ],
        'service': [
            "我想查询订单状态", "这个产品怎么使用？",
            "想要申请退换货", "价格能便宜点吗？"
        ]
    },
    
    # 8-12秒中等问题
    'medium_10_12s': {
        'factual': [
            "请问人工智能的发展历史是什么样的？",
            "中国古代四大发明都包括哪些内容？",
            "全球变暖对环境有什么具体影响？"
        ],
        'task': [
            "帮我预订明天晚上七点的餐厅位置",
            "查一下从北京到上海的高铁时刻表",
            "设置每天早上八点的工作提醒"
        ],
        'casual': [
            "我最近工作压力很大，有什么建议吗？",
            "推荐一本适合年轻人阅读的好书",
            "想学习一门新的编程语言，该选哪个？"
        ],
        'service': [
            "我买的商品有质量问题想要申请退换货",
            "订单已经三天了为什么还没有发货？",
            "能不能详细介绍一下你们的售后服务？"
        ]
    },
    
    # 12-18秒中长问题
    'medium_long_15s': {
        'factual': [
            "能不能详细解释一下区块链技术的基本原理和它在现实生活中的应用场景？",
            "请介绍一下中国传统文化中的儒家思想对现代社会的影响和意义",
            "全球气候变化的主要原因是什么，我们个人能做些什么来应对这个问题？"
        ],
        'task': [
            "帮我制定一个合理的减肥计划，包括饮食搭配和运动安排，目标是三个月减重十斤",
            "我想学习摄影，请推荐一些适合初学者的相机设备和基础拍摄技巧",
            "计划下个月去日本旅游，能帮我安排一下东京和大阪的五日游行程吗？"
        ],
        'casual': [
            "我正在考虑换工作但又担心新环境，你觉得我应该如何权衡利弊做决定？",
            "最近对投资理财很感兴趣，作为新手应该从哪些方面开始学习？",
            "想要培养一个新的兴趣爱好来丰富业余生活，有什么好的建议吗？"
        ],
        'service': [
            "我在你们网站购买的笔记本电脑出现了系统问题，想了解一下保修政策和维修流程",
            "订购的商品与描述不符而且包装破损，希望能够获得满意的解决方案",
            "想要了解你们公司的会员制度和积分兑换规则，以及如何享受更多优惠"
        ]
    },
    
    # 18-25秒长问题
    'long_20s': {
        'factual': [
            "我想深入了解一下人工智能技术的发展历程，从早期的专家系统到现在的深度学习，以及未来可能的发展方向和对社会的影响",
            "能不能详细介绍一下中医养生的基本理论和实践方法，包括饮食调理、运动锻炼和作息规律等各个方面的具体建议",
            "请解释一下量子物理学的基本概念和原理，以及它在现代科技发展中的重要作用和未来的应用前景"
        ],
        'task': [
            "我想要开始学习编程，但是完全没有基础，能帮我制定一个详细的学习计划，包括语言选择、学习资源和实践项目的安排吗？",
            "计划和朋友们一起创业开一家咖啡店，需要考虑哪些因素，包括选址、装修、设备采购和人员招聘等各个方面",
            "想要改善家里的装修风格，预算大概十万元左右，希望能够得到一些实用的设计建议和材料选择的指导"
        ]
    },
    
    # 25-35秒很长问题
    'very_long_30s': {
        'factual': [
            "我最近在研究人工智能和机器学习的相关技术，想要深入了解一下深度学习和传统机器学习方法之间的本质区别，以及它们各自的优缺点、适用场景和在实际项目中的选择标准，能详细介绍一下吗？",
            "对于中国古代历史文化很感兴趣，特别想了解一下唐朝的政治制度、经济发展、文化艺术和对外交流等各个方面的情况，以及它对后世中国历史发展的深远影响和现代价值"
        ],
        'task': [
            "我正在考虑购买一套新房子，但是对于房地产市场的走势、贷款政策、税费计算和装修预算等方面都不太了解，希望能够得到一些专业的建议和详细的分析，帮助我做出明智的决定",
            "想要系统性地学习投资理财知识，从基础的理财概念到股票、基金、债券等各种投资工具的特点和风险，以及如何根据个人情况制定合适的投资策略和资产配置方案"
        ],
        'casual': [
            "我已经工作五年了，最近对自己的职业发展方向感到很迷茫，不知道是否应该继续在现在的行业深耕，还是转行尝试新的领域，也在考虑是否要继续深造提升学历，希望能够得到一些建议",
            "作为一个新手父母，在育儿方面有很多困惑和担心，比如如何平衡工作和家庭、怎样给孩子提供良好的教育环境、以及如何处理教育过程中遇到的各种问题和挑战"
        ]
    }
}
```

### **质量控制流程**
```python
def validate_manual_questions(questions):
    """验证人工构建问题的质量"""
    validation_results = []
    
    for q in questions:
        result = {
            'question': q['text'],
            'estimated_duration': estimate_speech_duration(q['text']),
            'is_complete': is_complete_question(q['text']),
            'is_natural': assess_naturalness(q['text']),
            'llm_compatible': test_llm_response(q['text']),
            'category_match': verify_category_match(q['text'], q['category'])
        }
        validation_results.append(result)
    
    return validation_results
```

## 📊 数据收集执行计划

### **第1天: 数据集下载**
- [ ] 下载WebQA数据集
- [ ] 下载CrossWOZ数据集  
- [ ] 申请LCCC数据集访问权限
- [ ] 下载JDDC数据集

### **第2天: 数据筛选**
- [ ] 运行WebQA筛选脚本，获得60个问题
- [ ] 运行CrossWOZ提取脚本，获得40个问题
- [ ] 运行JDDC提取脚本，获得40个问题
- [ ] 等待LCCC审批(可能需要1-3天)

### **第3天: 质量验证**
- [ ] 验证筛选出的问题质量
- [ ] 测试LLM兼容性
- [ ] 分析长度分布情况

### **第4天: 补充构建**
- [ ] 根据分析结果确定需要补充的长度组
- [ ] 人工构建缺失的问题
- [ ] 最终质量检查

### **第5天: 数据组织**
- [ ] 按长度组织最终数据集
- [ ] 生成元数据文件
- [ ] 验证数据集完整性

这个方案确保了数据的多样性、质量和实验适用性，同时提供了详细的获取和处理指导。