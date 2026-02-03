# MedRAX 项目架构详解

## 项目概述

**MedRAX** 是一个医学影像推理 Agent 框架，通过集成多个医学图像分析工具，利用大模型进行复杂的医学推理任务。整个系统采用**微服务 + 主控制器**的架构。

```
┌─────────────────────────────────────────────────────────────────┐
│                    主评估脚本 (medrax_eval)                      │
│                 处理 2500 医学 QA 样本的评估                     │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
                     ┌─────────────────────────┐
                     │  main.py 初始化 Agent   │
                     │  配置系统提示和工具列表  │
                     └─────────────────────────┘
                                 ↓
       ┌─────────────────────────────────────────────┬─────────────────────────────────────────────┐
       ↓                                             ↓                                             ↓
┌─────────────────┐                        ┌─────────────────┐                        ┌─────────────────┐
│  run_vllm.sh    │                        │run_chexagent.sh │                        │ run_maira2.sh   │
│                 │                        │                 │                        │                 │
│ vLLM 推理引擎   │                        │ CheXagent VQA   │                        │ MAIRA-2 短语    │
│ 主模型推理      │                        │ 工具服务器      │                        │ 定位工具服务器  │
└─────────────────┘                        └─────────────────┘                        └─────────────────┘
     端口 8000                                 端口 19101                               端口 19102
```

---

## 四个 SBATCH 脚本详解

### 1️⃣ `run_vllm.sbatch` - 主模型推理引擎

**作用**：启动 vLLM 推理服务器，提供 OpenAI 兼容的 API

```bash
#!/bin/bash
#SBATCH --job-name=vllm_qwen3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

export CUDA_VISIBLE_DEVICES=0

conda run -n medrax \
vllm serve /mnt/realccvl15/ychen646/llms/M3D-LaMed-Phi-3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name M3D-LaMed-Phi-3-4B \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**关键参数解读**：

| 参数 | 作用 |
|------|------|
| `--host 0.0.0.0` | 监听所有网络接口，支持远程调用 |
| `--port 8000` | 默认推理端口 |
| `--served-model-name` | 模型标识符（LangChain 会用这个名字调用） |
| `--gpu-memory-utilization 0.9` | 用 90% 的 GPU 显存，提高吞吐量 |
| `--max-model-len 8192` | 最大上下文长度（tokens） |
| `--enable-auto-tool-choice` | **关键**：启用自动工具选择，Agent 能调用工具 |
| `--tool-call-parser hermes` | 用 Hermes 格式解析工具调用 |

**工作流程**：
1. 启动时加载模型权重到 GPU
2. 创建 HTTP 服务器监听 8000 端口
3. 接收来自 `main.py` 的推理请求（JSON 格式）
4. 使用 vLLM 的高效批处理引擎推理
5. 返回推理结果（文本 + 工具调用信息）

---

### 2️⃣ `run_chexagent.sbatch` - 胸部 X 射线 VQA 工具

**作用**：启动 CheXagent 微服务，专门分析胸部 X 射线图像

```bash
#!/bin/bash
#SBATCH --job-name=chexagent
#SBATCH --gres=gpu:1
#SBATCH --mem=14G

export CUDA_VISIBLE_DEVICES=0

cd /home/ypan81/MedRAX

conda run -n medrax_chexagent_new \
python tool_servers/chexagent_server.py \
  --host 0.0.0.0 \
  --port 19101
```

**工作流程**：
1. FastAPI 框架启动 HTTP 服务
2. 在后台加载 CheXagent 模型（transformers）
3. 监听 19101 端口
4. 等待接收 `/vqa` POST 请求

**CheXagent 服务实现** (`tool_servers/chexagent_server.py`)：

```python
# 请求格式
class VQARequest(BaseModel):
    image_paths: List[str]  # X 射线图像路径
    prompt: str              # 问题或指令
    max_new_tokens: int      # 最长生成长度

# 推理函数
def run_chexagent_vqa(image_paths, prompt, max_new_tokens):
    # 1. 读取图像
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    # 2. 准备输入 (tokenize + prepare)
    inputs = processor(images=images, text=prompt, return_tensors="pt")
    
    # 3. 模型推理
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # 4. 解码输出
    response = processor.decode(outputs[0])
    
    return {"response": response}
```

**关键特性**：
- 支持多图输入
- 高度特化的医学图像分析能力
- 返回结构化响应（JSON）

---

### 3️⃣ `run_maira2.sbatch` - 短语定位工具

**作用**：启动 MAIRA-2 服务，进行胸部 X 射线中的短语定位（物体检测）

```bash
#!/bin/bash
#SBATCH --job-name=maira2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

export CUDA_VISIBLE_DEVICES=0
export MAIRA2_CACHE_DIR="/mnt/realccvl15/ychen646/llms/MedRAX_models"

cd /home/ypan81/MedRAX

conda run -n medrax_maira2_new \
python tool_servers/maira2_server.py \
  --host 0.0.0.0 \
  --port 19102
```

**工作流程**：
1. 启动 FastAPI 服务器
2. 加载 MAIRA-2 多模态模型（使用本地缓存避免 HF 认证问题）
3. 监听 19102 端口
4. 处理短语定位请求

**MAIRA-2 实现** (`tool_servers/maira2_server.py`)：

```python
# 请求格式
class GroundingRequest(BaseModel):
    image_path: str     # 单张 X 射线图像
    phrase: str         # 要定位的医学短语（如"肺部阴影"）
    max_new_tokens: int

# 推理流程
def phrase_grounding(image_path, phrase):
    # 1. 读取图像并预处理
    image = Image.open(image_path).convert("RGB")
    
    # 2. 构建输入提示（告诉模型要定位哪个短语）
    prompt = f"Locate the following in the chest X-ray: {phrase}"
    
    # 3. 准备多模态输入
    inputs = processor(images=[image], text=prompt, return_tensors="pt")
    
    # 4. 模型推理（输出坐标而非文本）
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # 5. 解析输出为边界框坐标
    boxes = parse_boxes(processor.decode(outputs[0]))
    
    return {"boxes": boxes}
```

**内存优化** (关键问题解决)：

原始 MAIRA-2 加载时会触发 OOM。解决方案：
```python
# ✅ 优化的加载方式
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",           # 智能设备分配，避免峰值内存
    low_cpu_mem_usage=True,      # 边加载边卸载权重
    torch_dtype=torch.float16,   # 权重用 fp16 节省显存
    cache_dir=CACHE_DIR,
)
```

---

### 4️⃣ `run_medrax_eval.sbatch` - 主评估脚本

**作用**：协调所有服务，进行 2500 样本的评估

```bash
#!/bin/bash
#SBATCH --job-name=medrax_eval
#SBATCH --gres=gpu:1
#SBATCH --mem=28G

# 指向三个微服务的 URL
export OPENAI_BASE_URL="http://ccvl35:8000/v1"
export CHEXAGENT_SERVER_URL="http://ccvl35:19101"
export MAIRA2_SERVER_URL="http://ccvl35:19102"

export OPENAI_MODEL="M3D-LaMed-Phi-3-4B"

cd /home/ypan81/MedRAX

conda run -n medrax_eval_new \
python medrax_agent_bench.py \
  --model M3D-LaMed-Phi-3-4B \
  --temperature 0.2 \
  --log-prefix qM3D-LaMed-Phi-3-4B-medrax-tools
```

**关键点**：
- 注意使用 `ccvl35:8000` 而非 `127.0.0.1:8000`（因为在不同计算节点）
- 环境变量 `OPENAI_MODEL` 必须和 vLLM 的 `served-model-name` 一致
- 结果保存为 JSON Lines 格式

---

## 数据流详解

### 完整推理流程

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. 评估脚本 medrax_agent_bench.py                                 │
│    读取 2500 个医学 QA 样本                                       │
│    提取：问题、选项、医学图像、答案                               │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ 2. 图像编码 (_encode_images_to_b64)                              │
│    把所有 DICOM/PNG 图像转为 Base64 字符串                         │
│    格式：["data:image/png;base64,xxx", ...]                      │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ 3. 构建 User Message (build_user_message_content)               │
│    包含：                                                         │
│    - 系统指令（"你是 CXR 推理 Agent"）                           │
│    - 所有图像的可用路径列表                                       │
│    - Base64 编码的图像（多模态输入）                              │
│    - 问题文本 + 选项                                             │
│    - 指令："只输出一个大写字母（A-F）"                           │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ 4. 调用 main.py 中的 Agent.invoke()                              │
│    初始化 LangGraph Agent：                                       │
│    - 模型：ChatOpenAI（使用 vLLM 后端）                          │
│    - 工具：8 个医学分析工具                                       │
│    - 系统提示：医学推理指导                                       │
└──────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│ 5. Agent 推理循环 (LangGraph)                                    │
│    [初始状态] → [模型推理] → [工具调用判断]                       │
│         ↓                                                         │
│    ┌──[是否需要工具]──┐                                          │
│    NO               YES                                          │
│    ↓                ↓                                            │
│  [结束]       [解析工具调用]                                      │
│              [并发调用 1-8 个工具]                                │
│                    ↓                                             │
│              [收集工具结果]                                       │
│                    ↓                                             │
│              [反馈给模型]                                         │
│                    ↓                                             │
│              [继续推理或结束]                                     │
└──────────────────────────────────────────────────────────────────┘
```

### 工具调用示例

**场景**：Agent 需要分析胸部 X 射线

```
Agent 输出：
{
  "tool_calls": [
    {
      "name": "ImageVisualizerTool",
      "args": {
        "image_paths": ["/path/to/cxr.png"],
        "description": "Display the chest X-ray"
      }
    },
    {
      "name": "chest_xray_expert",  // CheXagent
      "args": {
        "image_paths": ["/path/to/cxr.png"],
        "prompt": "What are the main findings in this chest X-ray?"
      }
    }
  ]
}
       ↓
[Agent 并发调用]
       ↓
┌────────────────────┬──────────────────────┐
│  Tool 1 执行       │  Tool 2 执行          │
│  ImageVisualizer   │  CheXagent VQA       │
│  返回：图像路径    │  返回：分析文本      │
└────────────────────┴──────────────────────┘
       ↓
[合并结果，反馈给 Agent]
       ↓
Agent 继续推理，输出最终答案
```

---

## 关键代码块解析

### A. Agent 初始化 (main.py)

```python
def initialize_agent(prompt_file, tools_to_use, ...):
    # 1. 加载所有可用工具的构造器字典
    all_tools = {
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "XRayVQATool": lambda: XRayVQATool(...),  # 远程调用 CheXagent
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(...),  # 远程调用 MAIRA-2
        # ... 其他工具
    }
    
    # 2. 选择性初始化工具（避免加载无用工具）
    tools_dict = {}
    for tool_name in tools_to_use:
        tools_dict[tool_name] = all_tools[tool_name]()
    
    # 3. 创建 LangChain 的 ChatOpenAI 对象
    model = ChatOpenAI(
        model="M3D-LaMed-Phi-3-4B",  # 必须和 vLLM 的 served-model-name 一致！
        temperature=0.2,
        base_url=os.getenv("OPENAI_BASE_URL"),  # http://ccvl35:8000/v1
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # 4. 构建 Agent（LangGraph）
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        system_prompt=prompt,  # 医学推理指导文本
        checkpointer=MemorySaver(),  # 记录推理过程
    )
    
    return agent, tools_dict
```

### B. 样本处理 (medrax_agent_bench.py)

```python
def main():
    # 1. 加载数据集（2500 个医学 QA 样本）
    dataset = load_dataset("medqa", split="validation")
    
    # 2. 初始化 Agent
    agent, tools_dict = build_agent()
    
    # 3. 逐样本处理
    results = []
    for sample in dataset:
        try:
            # 3.1 提取图像路径
            image_paths = _collect_image_paths(sample, image_root)
            
            # 3.2 编码为 Base64
            b64_images = _encode_images_to_b64(image_paths)
            
            # 3.3 构建 user message（多模态输入）
            content = _build_user_message_content(
                image_paths, b64_images, sample
            )
            
            # 3.4 调用 Agent
            response = agent.invoke(
                {"messages": [{"role": "user", "content": content}]},
                config={"recursion_limit": 25},  # 最多 25 步推理
            )
            
            # 3.5 提取工具轨迹和性能指标
            tool_trace = extract_tool_trace(response["messages"])
            trace_summary = summarize_tool_trace(tool_trace)
            
            # 3.6 评估答案
            model_answer = normalize_choice(response["messages"][-1].content)
            correct = (model_answer == sample["answer"])
            
            # 3.7 保存结果
            results.append({
                "sample_id": sample["id"],
                "correct": correct,
                "model_answer": model_answer,
                "ground_truth": sample["answer"],
                "status": "success",
                "num_tools_called": trace_summary["num_tools_called"],
                "tool_failures": trace_summary["tool_failures"],
            })
            
        except Exception as e:
            results.append({
                "sample_id": sample["id"],
                "status": "error",
                "error": str(e),
            })
    
    # 4. 保存为 JSON Lines
    with open(f"results_{timestamp}.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
```

### C. 远程工具调用 (remote_tools.py)

```python
class XRayVQATool(BaseTool):
    name = "chest_xray_expert"
    server_url = ""  # 从环境变量 CHEXAGENT_SERVER_URL 读取
    
    def __init__(self):
        base_url = os.environ.get("CHEXAGENT_SERVER_URL")
        if not base_url:
            raise ValueError("CHEXAGENT_SERVER_URL not set!")
        self.server_url = base_url.rstrip("/")
    
    def _run(self, image_paths: List[str], prompt: str, max_new_tokens: int = 512):
        # 组织请求体
        payload = {
            "image_paths": image_paths,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        
        # 发送 HTTP POST 请求到 CheXagent 服务器
        resp = requests.post(
            f"{self.server_url}/vqa",
            json=payload,
            timeout=600,
        )
        
        if resp.status_code != 200:
            return f"Error: {resp.status_code}"
        
        data = resp.json()
        return data.get("response", "")
```

---

## 系统架构的设计优势

### 1. **微服务隔离**
- 各工具独立运行，互不干扰
- 工具故障不会导致整个系统崩溃
- 可单独升级或替换工具

### 2. **资源高效利用**
- 每个服务可独立配置 GPU/内存
- CheXagent (3B) 只需 14GB，MAIRA-2 (7B) 只需 24GB
- vLLM 主模型可灵活配置

### 3. **易于扩展**
- 添加新工具只需：
  1. 编写工具服务器 (`tool_servers/xxx_server.py`)
  2. 编写远程工具类 (`remote_tools.py` 中新增类)
  3. 在 `main.py` 的 `all_tools` 字典中注册

### 4. **通用推理框架**
- Agent 框架通过 LangGraph 实现
- 支持任意 OpenAI 兼容的模型（vLLM、OpenAI API、本地 LLM 等）
- 可轻松替换主模型

---

## 数据流总结

```
样本 → 图像编码 → 多模态输入 → Agent 推理 ─┬─→ [直接回答]
                                    └─→ [工具调用] ─→ CheXagent
                                                   ├─→ MAIRA-2
                                                   └─→ 其他工具 → 结果汇总 → 最终答案

评估指标：
- 准确率 (Accuracy)：回答正确的比例
- 工具调用数 (num_tools_called)：平均用了多少工具
- 工具失败率：工具返回错误的比例
- 推理步数 (recursion depth)：平均需要多少推理步
```

---

## 模型切换指南

当要用不同的模型时（如 Qwen3-VL-8B）：

### 步骤 1: 修改 `run_vllm.sbatch`
```bash
# 改这两行
vllm serve /mnt/realccvl15/ychen646/llms/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b-instruct
```

### 步骤 2: 修改 `run_medrax_eval.sbatch`
```bash
export OPENAI_MODEL="qwen3-vl-8b-instruct"
# 对应的日志前缀
--log-prefix qwen3-vl-8b-medrax-tools
```

### 步骤 3: 如果改了 main.py 中的模型名，也要同步修改
```python
# main.py 第 87 行左右
model = ChatOpenAI(
    model="qwen3-vl-8b-instruct",  # 这里要和 vLLM 一致
    ...
)
