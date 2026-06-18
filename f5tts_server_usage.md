# F5-TTS Vietnamese HTTP Server — `f5tts_server.py`

## 依赖安装

```bash
pip install tornado
```

---

## 启动服务

```bash
# 最简启动（全部使用 infer.bat 默认参数）
python f5tts_server.py

# 自定义端口 / 地址
python f5tts_server.py --port 8080 --host 0.0.0.0

# 自定义默认参考音频与文本
python f5tts_server.py --ref_audio ref2.wav --ref_text "văn bản tham chiếu của bạn"

# 自定义语速
python f5tts_server.py --speed 1.2

# 查看所有参数
python f5tts_server.py --help
```

### 启动参数一览

| 参数 | 默认值                                    | 说明 |
|------|----------------------------------------|------|
| `--host` | `0.0.0.0`                              | 监听地址 |
| `--port` | `9000`                                 | 监听端口 |
| `--model` | `F5TTS_Base`                           | 模型名称（对应 infer.bat `--model`） |
| `--vocoder_name` | `vocos`                                | Vocoder 名称 |
| `--speed` | `1.0`                                  | 默认语速 |
| `--vocab_file` | `./models/vocab.txt`                   | 词表文件路径 |
| `--ckpt_file` | `./models/model_last.pt`               | 模型权重路径 |
| `--ref_audio` | `./ref.wav`                            | 默认参考音频路径 |
| `--ref_text` | `cả hai bên hãy cố gắng hiểu cho nhau` | 默认参考文本 |

---

## API 接口

### `GET /health` — 健康检查

```bash
curl http://localhost:9000/health
# {"status": "ok"}
```

---

### `POST /tts` — 语音合成

支持两种请求方式：

| 方式 | Content-Type | 参考音频来源 |
|------|-------------|-------------|
| 方式一 | `application/json` | 服务器本地路径 |
| 方式二 | `multipart/form-data` | 直接上传文件，或填服务器路径 |

响应头：

| 响应头 | 说明 |
|--------|------|
| `Content-Type` | `audio/wav` |
| `Content-Disposition` | `attachment; filename="tts_xxxx.wav"` |
| `X-Inference-Time` | 推理耗时，如 `5.231s` |
| `X-Task-Id` | 本次任务 ID |
| `X-Download-Url` | 可重复下载的路径，如 `/audio/tts_xxxx.wav` |

---

#### 方式一：application/json

**请求字段：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_text` | string | — | **必填**，待合成文本 |
| `ref_audio` | string | `--ref_audio` | 服务器本地参考音频路径 |
| `ref_text` | string | `--ref_text` | 参考音频对应文本 |
| `speed` | float | `1.0` | 语速 |
| `model` | string | `F5TTS_Base` | 模型名称 |
| `vocoder_name` | string | `vocos` | Vocoder 名称 |
| `vocab_file` | string | `./models/vocab.txt` | 词表路径 |
| `ckpt_file` | string | `./models/model_last.pt` | 权重路径 |

**curl 示例：**

```bash
# 最简调用（使用服务默认参考音频）
curl -X POST http://localhost:9000/tts \
  -H "Content-Type: application/json" \
  -d '{"gen_text": "xin chào, đây là dịch vụ tổng hợp giọng nói"}' \
  --output output.wav

# 指定参考音频路径 + 语速
curl -X POST http://localhost:9000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "gen_text": "mình muốn ra nước ngoài để tiếp xúc nhiều công ty lớn",
    "ref_audio": "ref2.wav",
    "ref_text": "văn bản tham chiếu",
    "speed": 1.2
  }' \
  --output output.wav
```

---

#### 方式二：multipart/form-data（支持文件上传）

**参考音频优先级（三选一）：**
1. 上传文件字段 `ref_audio`（最高）
2. 表单字段 `ref_audio_path`（服务器本地路径）
3. 服务启动时的 `--ref_audio` 默认音频（兜底）

**curl 示例：**

```bash
# 上传本地音频文件作为参考音频
curl -X POST http://localhost:9000/tts \
  -F "gen_text=xin chào, đây là dịch vụ tổng hợp giọng nói" \
  -F "ref_audio=@/path/to/my_voice.wav" \
  -F "ref_text=văn bản tham chiếu" \
  --output output.wav

# 不上传文件，使用服务器本地路径
curl -X POST http://localhost:9000/tts \
  -F "gen_text=xin chào" \
  -F "ref_audio_path=ref2.wav" \
  --output output.wav
```

---

### `GET /audio/<filename>` — 下载已合成文件

每次 `/tts` 请求成功后，合成结果会保存到 `outputs/` 目录，可通过此接口下载：

```bash
curl http://localhost:9000/audio/tts_abc123def456.wav --output output.wav
```

---

### `POST /files/upload` — 上传参考音频到缓存

按文件 MD5 去重存储到 `upload_files/` 目录，后续可通过 `server_path` 在 `/tts` 中使用：

```bash
curl -X POST http://localhost:9000/files/upload \
  -F "file=@my_voice.wav"
# {"hash": "d41d8cd98f00b204e9800998ecf8427e", "server_path": "upload_files/d41d....wav"}
```

---

### `GET /files/check?hash=<md5>` — 查询上传缓存

```bash
curl "http://localhost:9000/files/check?hash=d41d8cd98f00b204e9800998ecf8427e"
# {"exists": true,  "server_path": "upload_files/d41d....wav"}
# {"exists": false}
```

---

## Python 调用示例

### JSON 方式（最简）

```python
import requests

resp = requests.post(
    "http://localhost:9000/tts",
    json={"gen_text": "xin chào, đây là dịch vụ tổng hợp giọng nói"},
    timeout=120,
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
print("推理耗时:", resp.headers.get("X-Inference-Time"))
print("可重复下载:", resp.headers.get("X-Download-Url"))
```

### multipart 方式（上传参考音频）

```python
import requests

with open("my_voice.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:9000/tts",
        files={"ref_audio": ("my_voice.wav", f, "audio/wav")},
        data={
            "gen_text": "mình muốn ra nước ngoài để tiếp xúc nhiều công ty lớn",
            "ref_text": "văn bản tham chiếu",
            "speed": "1.0",
        },
        timeout=120,
    )

resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
print("推理耗时:", resp.headers.get("X-Inference-Time"))
```

### 先上传缓存，再复用

```python
import requests

# 1. 上传参考音频（按 MD5 去重，重复上传无副作用）
with open("my_voice.wav", "rb") as f:
    up = requests.post("http://localhost:9000/files/upload",
                       files={"file": f}, timeout=30)
up.raise_for_status()
server_path = up.json()["server_path"]

# 2. 使用缓存路径合成
resp = requests.post(
    "http://localhost:9000/tts",
    json={
        "gen_text": "xin chào",
        "ref_audio": server_path,
        "ref_text": "văn bản tham chiếu",
    },
    timeout=120,
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

