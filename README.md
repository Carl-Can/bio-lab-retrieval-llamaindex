# 生物实验检索系统

面向实验笔记的单机检索问答。把 PDF、Markdown、Word、纯文本放进 `docs/`，用自然语言查出相关段落，并可由大模型基于引用生成回答。

默认检索是稠密向量加 BM25（中文分词）的混合检索，再用交叉编码器重排。知识库本身不在仓库里，克隆后需要自己放入文档并建索引。

## 检索链路

```
文档 → 切块（512 token，重叠 50）→ bge-m3 向量 + BM25
     → RRF 融合 → bge-reranker 重排 →（可选）LLM 生成
```

- 检索模式：`hybrid`（默认）、`dense`、`exact`（精确余弦，评测用，结果可复现）
- 重排：`bge`（默认）、`bge-onnx`、`jev`，或 `none`
- Embedding：默认本地 `BAAI/bge-m3`，也可改成 OpenAI
- 生成：OpenAI，或本地 vLLM（OpenAI 兼容接口）
- `docs/` 下的一级目录名会写成专题元数据，查询时可按专题过滤

## 目录

```
.
├── api/            # FastAPI
├── core/           # 解析、索引、混合检索、重排、grounding
├── config/         # 环境变量
├── frontend/       # React + Vite
├── scripts/        # 评测脚本
├── tests/
├── docs/           # 本地知识库，不提交
├── init_index.py   # 清库、去重、查看状态
└── .env.example
```

## 环境

- Python 3.10
- Node.js（只跑前端时需要）
- 本地 Embedding 默认走 GPU（`HF_DEVICE=cuda`）。没有 GPU 时在 `.env` 里改成 `cpu`
- 首次需要能下载 Hugging Face 模型；缓存就绪后可保持 `HF_LOCAL_FILES_ONLY=True`

## 运行

在仓库根目录：

```bash
cp .env.example .env
# 编辑 .env，至少填 OPENAI_API_KEY（用本地 vLLM 时改为 LLM_PROVIDER=vllm）

uv sync
uv run uvicorn api.main:app --reload
```

另开一个终端启动前端（http://localhost:3000，接口代理到 8000）：

```bash
cd frontend
npm install
npm run dev
```

把文献放进 `docs/`（可按专题分子目录），然后在首页点「初始化索引」，或调用 `POST /initialize`。这一步会写入专题、文件名等元数据。

`uv run python init_index.py` 用来清库、去掉重复向量、查看向量数量。

不用 uv 时，可以用 `pip install -r requirements.txt`，再用 `uvicorn api.main:app --reload` 启动。锁文件以 `uv.lock` 为准。

## 接口

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/stats` | 向量数量 |
| GET | `/topics` | 专题列表 |
| POST | `/initialize` | 从 `docs/` 建索引 |
| POST | `/upload` | 上传并写入索引（最大 10MB） |
| GET | `/query` | 检索；`generate=false` 只返回片段 |
| GET | `/documents` | 文档列表 |
| GET | `/docs/{doc_id}` | 查看切块 |
| DELETE | `/docs/{doc_id}` | 删除 |

`/query` 常用参数：`q`、`top_k`（1–10）、`retrieval`、`rerank`、`topic`、`generate`、`verify`（对生成结果做引用核验）。

`.env` 里设置了 `API_KEY` 之后，除健康检查外的请求需要带请求头 `X-API-Key`。留空则不鉴权。

## 测试

不加载模型的测试：

```bash
pytest -m "not heavy" -q
```

检索和答案质量评测需要本地模型和已建好的索引，见 `scripts/eval_retrieval.py`、`scripts/eval_answer_quality.py`。评测集和 `docs/` 一样留在本地，不随仓库发布。

## 配置

全部配置在 `.env`，模板是 `.env.example`。和检索质量直接相关的是 `RETRIEVAL_MODE`、`RERANK_PROVIDER`、`CHUNK_SIZE`、`HF_DEVICE`、`LLM_PROVIDER`。
