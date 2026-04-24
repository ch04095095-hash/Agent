# oellm_agent (Python, board-side)

基于 `oellm_runtime/example/oellm_multichat` 的轻量 Agent 封装。

## 1. 前置条件

先确保官方 demo 已经可运行：

1. `sh set_performance_mode.sh`
2. 配置 `LD_LIBRARY_PATH` 指向 `oellm_runtime/lib`
3. `oellm_multichat -c qwen_multichat_config.json` 能正常对话

> 注意：`qwen_multichat_config.json` 里的 `hbm_path / tokenizer_dir / template_path` 必须有效。

## 2. 文件说明

- `agent.py`: Agent 主程序
- `requirements.txt`: 依赖
- `kb/vector_kb.py`: 向量知识库核心（切块、向量化、混合检索）
- `kb/ingest_vector_kb.py`: 文件上传/入库脚本
- `kb/search_vector_kb.py`: 混合检索脚本

## 3. 运行

```bash
cd /mnt/ssd/Agent/oellm_agent
python3 agent.py \
  --runtime-dir /home/root/llm/D-Robotics_LLM_S100_1.0.0_SDK/oellm_runtime \
  --multichat-bin /home/root/llm/D-Robotics_LLM_S100_1.0.0_SDK/oellm_runtime/example/oellm_multichat/oellm_multichat \
  --multichat-cfg /home/root/llm/D-Robotics_LLM_S100_1.0.0_SDK/oellm_runtime/example/oellm_multichat/qwen_multichat_config.json \
  --workdir /home/root \
  --kb-collection coal_truck_kb \
  --kb-embed-model BAAI/bge-m3
```

启动后输入：
- 普通问题：直接问
- 退出：`quit`

## 4. 当前内置工具（带安全限制）

- `read_file(path)`：读取 `--workdir` 下文件
- `write_file(path, content)`：写文件到 `--workdir` 下
- `run_shell(command)`：仅允许白名单命令（`ls/pwd/echo/cat/head/tail/grep/wc`）
- `kb_ingest(paths, chunk_size, chunk_overlap)`：上传文件到向量知识库（支持目录递归）
- `kb_search(query, top_k, source_type, dense_k, bm25_k, w_dense, w_bm25)`：混合检索

## 5. 真实向量知识库（上传文件/切块/向量化/检索/混合排序）

### 5.1 安装依赖

```bash
cd /mnt/ssd/Agent/oellm_agent
pip install -r requirements.txt
```

### 5.2 本地知识库模式（FAISS + SQLite）

当前项目使用本地模式：

- 向量索引：FAISS（本地文件）
- 元数据：SQLite（`kb/data/kb.sqlite3`）
- 不依赖 Qdrant 服务

### 5.3 文档上传与入库

支持文件类型：`.txt/.md/.json/.pdf`

- `txt/md/pdf`：直接读全文
- `json`：优先读取 `content` 字段，同时识别 `doc_id/title/source_type/tags`

示例（上传目录）：

```bash
python3 kb/ingest_vector_kb.py kb/docs \
  --db kb/data/kb.sqlite3 \
  --collection coal_truck_kb \
  --embed-model BAAI/bge-small-zh-v1.5 \
  --chunk-size 800 \
  --chunk-overlap 120
```

示例（上传多个文件/目录混合）：

```bash
python3 kb/ingest_vector_kb.py kb/docs manuals/extra.pdf notes.md
```

### 5.4 检索与混合排序

当前采用：

- Dense 检索：FAISS 向量相似度（本地索引）
- Sparse 检索：BM25（本地 SQLite 元数据文本）
- Hybrid 融合：归一化后线性加权 `score = w_dense * dense + w_bm25 * bm25`

查询示例：

```bash
python3 kb/search_vector_kb.py "甲烷超限且水温升高怎么处置" \
  --db kb/data/kb.sqlite3 \
  --collection coal_truck_kb \
  --embed-model BAAI/bge-small-zh-v1.5 \
  --top-k 5 --dense-k 30 --bm25-k 30 \
  --w-dense 0.7 --w-bm25 0.3
```

按来源过滤：

```bash
python3 kb/search_vector_kb.py "冷却系统故障征兆" --source-type maintenance
```

### 5.5 你的“真实知识库”链路已覆盖

- 文件上传（多格式）
- 文本切块（可配 chunk size/overlap）
- 向量化（FastEmbed，默认 BAAI/bge-small-zh-v1.5）
- 向量库存储（本地 FAISS 索引）
- 检索（Dense + BM25）
- 混合排序（可调权重）

## 6. 常见问题

1. **启动卡住**
   - 通常是 `oellm_multichat` 未正常启动或模型路径配置错误。
2. **找不到库文件**
   - 确认 `oellm_runtime/lib` 在 `LD_LIBRARY_PATH` 中。
3. **工具调用不稳定**
   - 属于小模型 JSON 输出波动，可在 `SYSTEM_PROMPT` 继续加强约束。
4. **KB 查询没命中**
   - 先确认已执行 `build_kb.py`，再检查 `kb/docs/*.json` 内容是否包含查询关键词。
