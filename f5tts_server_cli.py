"""
F5-TTS Vietnamese Tornado HTTP 语音合成服务（CLI 版）

功能说明:
    基于 Tornado 框架提供 HTTP 接口，底层通过调用
    `f5-tts_infer-cli` 命令行工具完成语音合成，
    无需在进程内加载模型，启动更轻量。

启动方式:
    python f5tts_server_cli.py [--port 9000] [--host 0.0.0.0]
                               [--ref_audio ref.wav] [--ref_text "..."]
                               [--speed 1.0] [--model F5TTS_Base]
                               [--vocab_file ./models/vocab.txt]
                               [--ckpt_file ./models/model_last.pt]

API 接口说明:
    GET  /health             健康检查
    POST /tts                语音合成（JSON 或 multipart/form-data）
    GET  /audio/<filename>   下载已合成文件
    POST /files/upload       上传参考音频到服务器缓存
    GET  /files/check?hash=  查询上传缓存是否存在
"""

import os
import sys
import uuid
import time
import hashlib
import logging
import argparse
import tempfile
import subprocess
import threading
import shutil
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tornado.ioloop
import tornado.web
import tornado.httpserver

current_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局变量
# ---------------------------------------------------------------------------
# 并发控制：同一时刻只允许一个推理任务（CLI 进程），避免显存溢出
infer_semaphore = threading.Semaphore(1)

# 合成结果输出目录
OUTPUTS_DIR = os.path.join(current_dir, "outputs")

# 上传文件缓存目录（按 MD5 哈希存储）
UPLOAD_FILES_DIR = os.path.join(current_dir, "upload_files")


# 默认推理参数（由命令行参数填充）
default_args = {
    "model":        "F5TTS_Base",
    "vocoder_name": "vocos",
    "speed":        1.0,
    "vocab_file":   "./models/vocab.txt",
    "ckpt_file":    "./models/model_last.pt",
    "ref_audio":    "./ref.wav",
    "ref_text":     "",
}


# ---------------------------------------------------------------------------
# CLI 推理核心
# ---------------------------------------------------------------------------

def run_infer_cli(
    gen_text,
    ref_audio,
    output_path,
    ref_text=None,
    speed=1.0,
    model="F5TTS_Base",
    vocoder_name="vocos",
    vocab_file="./models/vocab.txt",
    ckpt_file="./models/model_last.pt",
):
    """
    组装并执行 f5-tts_infer-cli 命令。
    成功时返回 (True, ""), 失败时返回 (False, stderr_message)。
    """
    cmd = [
        "f5-tts_infer-cli",
        "--model",        model,
        "--ref_audio",    ref_audio,
        "--gen_text",     gen_text,
        "--speed",        str(speed),
        "--vocoder_name", vocoder_name,
        "--vocab_file",   vocab_file,
        "--ckpt_file",    ckpt_file,
        "--output_file",  output_path,
    ]

    # ref_text 为空时不传参数，让 CLI 自动转录
    if ref_text:
        cmd += ["--ref_text", ref_text]

    logger.info("执行命令: %s", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["WANDB_CONSOLE"] = "off"
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=current_dir,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout.strip():
        logger.info("CLI stdout:\n%s", result.stdout.strip())
    if result.returncode != 0:
        logger.error("CLI stderr:\n%s", result.stderr.strip())
        return False, result.stderr.strip()

    return True, ""


# ---------------------------------------------------------------------------
# 请求处理器
# ---------------------------------------------------------------------------

class HealthHandler(tornado.web.RequestHandler):
    """GET /health — 健康检查"""

    def get(self):
        self.set_header("Content-Type", "application/json")
        self.finish({"status": "ok"})


class AudioDownloadHandler(tornado.web.RequestHandler):
    """GET /audio/<filename> — 下载已合成的音频文件"""

    def get(self, filename):
        filename = os.path.basename(filename)
        file_path = os.path.join(OUTPUTS_DIR, filename)
        if not os.path.exists(file_path):
            self.set_status(404)
            self.finish({"error": f"File not found: {filename}"})
            return
        with open(file_path, "rb") as f:
            data = f.read()
        self.set_header("Content-Type", "audio/wav")
        self.set_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.set_header("Content-Length", str(len(data)))
        self.finish(data)


class FileCheckHandler(tornado.web.RequestHandler):
    """GET /files/check?hash=<md5> — 查询上传缓存"""

    def get(self):
        file_hash = self.get_argument("hash", "").strip().lower()
        if not file_hash:
            self.set_status(400)
            self.finish({"error": "Missing 'hash' query parameter"})
            return

        os.makedirs(UPLOAD_FILES_DIR, exist_ok=True)
        for fname in os.listdir(UPLOAD_FILES_DIR):
            name_part = os.path.splitext(fname)[0]
            if name_part == file_hash:
                server_path = os.path.join(UPLOAD_FILES_DIR, fname)
                self.finish({"exists": True, "server_path": server_path})
                return

        self.finish({"exists": False})


class FileUploadHandler(tornado.web.RequestHandler):
    """POST /files/upload — 上传参考音频（MD5 去重缓存）"""

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self):
        self.set_status(204)
        self.finish()

    def post(self):
        files = self.request.files.get("file")
        if not files:
            self.set_status(400)
            self.finish({"error": "No file uploaded. Use field name 'file'."})
            return

        uploaded = files[0]
        body = uploaded["body"]
        original_filename = uploaded.get("filename", "audio.wav")
        _, ext = os.path.splitext(original_filename)
        ext = ext.lower() if ext else ".wav"

        file_hash = hashlib.md5(body).hexdigest()

        os.makedirs(UPLOAD_FILES_DIR, exist_ok=True)
        dest_path = os.path.join(UPLOAD_FILES_DIR, f"{file_hash}{ext}")

        if not os.path.exists(dest_path):
            with open(dest_path, "wb") as f:
                f.write(body)
            logger.info("已保存上传文件: %s (%d bytes)", dest_path, len(body))
        else:
            logger.info("文件已存在，跳过保存: %s", dest_path)

        self.finish({"hash": file_hash, "server_path": dest_path})


class TTSHandler(tornado.web.RequestHandler):
    """
    POST /tts — 语音合成接口

    支持两种请求方式：
      方式一：application/json         参考音频使用服务器本地路径
      方式二：multipart/form-data      支持直接上传参考音频文件
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self):
        self.set_status(204)
        self.finish()

    def post(self):
        import json as _json

        content_type = self.request.headers.get("Content-Type", "")
        temp_files_to_cleanup = []

        # ------------------------------------------------------------------ #
        # 1. 解析请求参数                                                       #
        # ------------------------------------------------------------------ #
        if "multipart/form-data" in content_type:
            # ── 方式二：multipart/form-data ──────────────────────────────── #
            params = {}
            for key, val_list in self.request.body_arguments.items():
                params[key] = val_list[0].decode("utf-8") if val_list else ""

            def save_uploaded_audio(field_name):
                """将上传的音频保存为临时文件，返回路径或 None。"""
                files = self.request.files.get(field_name)
                if not files:
                    return None
                uploaded = files[0]
                original_filename = uploaded.get("filename", "audio.wav")
                _, ext = os.path.splitext(original_filename)
                ext = ext.lower() if ext else ".wav"
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="upload_ref_")
                os.close(tmp_fd)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded["body"])
                logger.info(
                    "已保存上传音频 [%s] → %s (%d bytes)",
                    field_name, tmp_path, len(uploaded["body"]),
                )
                temp_files_to_cleanup.append(tmp_path)
                return tmp_path

            ref_audio_uploaded = save_uploaded_audio("ref_audio")

            # 参考音频优先级：上传文件 > ref_audio_path 字段 > 默认
            ref_audio = (
                ref_audio_uploaded
                or params.get("ref_audio_path", "").strip()
                or default_args["ref_audio"]
            )
            ref_text     = params.get("ref_text", "").strip() or None
            gen_text     = params.get("gen_text", "").strip()
            speed        = float(params.get("speed",        default_args["speed"]))
            model        = params.get("model",        default_args["model"]).strip() or default_args["model"]
            vocoder_name = params.get("vocoder_name", default_args["vocoder_name"]).strip() or default_args["vocoder_name"]
            vocab_file   = params.get("vocab_file",   default_args["vocab_file"]).strip() or default_args["vocab_file"]
            ckpt_file    = params.get("ckpt_file",    default_args["ckpt_file"]).strip() or default_args["ckpt_file"]

        else:
            # ── 方式一：application/json ─────────────────────────────────── #
            try:
                body = self.request.body
                if not body:
                    self.set_status(400)
                    self.finish({"error": "Empty request body"})
                    return
                params = _json.loads(body.decode("utf-8"))
            except Exception as e:
                self.set_status(400)
                self.finish({"error": f"Invalid JSON: {e}"})
                return

            ref_audio_uploaded = None
            ref_audio    = params.get("ref_audio",    default_args["ref_audio"])
            ref_text     = params.get("ref_text",     None) or None   # 空字符串也视为不传
            gen_text     = params.get("gen_text",     "").strip()
            speed        = float(params.get("speed",        default_args["speed"]))
            model        = params.get("model",        default_args["model"])
            vocoder_name = params.get("vocoder_name", default_args["vocoder_name"])
            vocab_file   = params.get("vocab_file",   default_args["vocab_file"])
            ckpt_file    = params.get("ckpt_file",    default_args["ckpt_file"])

        # ------------------------------------------------------------------ #
        # 2. 校验必填参数                                                       #
        # ------------------------------------------------------------------ #
        if not gen_text:
            self.set_status(400)
            self.finish({"error": "Field 'gen_text' is required and must not be empty"})
            return

        # ------------------------------------------------------------------ #
        # 3. 校验参考音频路径                                                   #
        # ------------------------------------------------------------------ #
        if not os.path.exists(ref_audio):
            self.set_status(400)
            self.finish({"error": f"Reference audio file not found: {ref_audio}"})
            return

        # ------------------------------------------------------------------ #
        # 4. 执行 f5-tts_infer-cli 推理（串行，避免显存溢出）                    #
        # ------------------------------------------------------------------ #
        # 尝试获取信号量（非阻塞），若正在推理则返回 429
        if not infer_semaphore.acquire(blocking=True, timeout=300):
            self.set_status(429)
            self.finish({"error": "服务器正忙，请稍后重试"})
            return

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        task_id      = uuid.uuid4().hex
        out_filename = f"tts_{task_id[:12]}.wav"
        output_path  = os.path.join(OUTPUTS_DIR, out_filename)

        try:
            logger.info(
                "TTS request | gen_text_len=%d | ref_audio=%s | speed=%.1f | "
                "ref_text=%s | uploaded=%s",
                len(gen_text), ref_audio, speed,
                repr(ref_text) if ref_text else "<auto>",
                ref_audio_uploaded is not None,
            )
            t0 = time.perf_counter()

            ok, err_msg = run_infer_cli(
                gen_text     = gen_text,
                ref_audio    = ref_audio,
                output_path  = output_path,
                ref_text     = ref_text,          # None 则 CLI 不加 --ref_text
                speed        = speed,
                model        = model,
                vocoder_name = vocoder_name,
                vocab_file   = vocab_file,
                ckpt_file    = ckpt_file,
            )

            if not ok:
                self.set_status(500)
                self.finish({"error": f"推理失败: {err_msg}"})
                return

            if not os.path.exists(output_path):
                self.set_status(500)
                self.finish({"error": "推理完成但输出文件不存在，请检查 CLI 日志"})
                return

            elapsed = time.perf_counter() - t0
            logger.info("TTS done in %.2fs → %s", elapsed, out_filename)

            with open(output_path, "rb") as f:
                wav_bytes = f.read()

            self.set_header("Content-Type", "audio/wav")
            self.set_header("Content-Disposition", f'attachment; filename="{out_filename}"')
            self.set_header("Content-Length", str(len(wav_bytes)))
            self.set_header("X-Inference-Time", f"{elapsed:.3f}s")
            self.set_header("X-Task-Id", task_id)
            self.set_header("X-Download-Url", f"/audio/{out_filename}")
            self.finish(wav_bytes)

        except Exception as e:
            logger.exception("TTS 请求处理异常")
            # 推理失败时清理输出文件
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            self.set_status(500)
            self.finish({"error": str(e)})

        finally:
            infer_semaphore.release()
            # 清理上传的临时文件
            for tmp in temp_files_to_cleanup:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                        logger.debug("已清理上传临时文件: %s", tmp)
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Tornado 应用工厂
# ---------------------------------------------------------------------------

def make_app():
    return tornado.web.Application(
        [
            (r"/tts",          TTSHandler),
            (r"/health",       HealthHandler),
            (r"/audio/(.+)",   AudioDownloadHandler),
            (r"/files/upload", FileUploadHandler),
            (r"/files/check",  FileCheckHandler),
        ],
        debug=False,
    )


# ---------------------------------------------------------------------------
# 程序入口
# ---------------------------------------------------------------------------

def main():
    global default_args

    parser = argparse.ArgumentParser(
        description="F5-TTS Vietnamese Tornado HTTP Server (CLI 版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",         type=str,   default="0.0.0.0",
                        help="服务监听地址")
    parser.add_argument("--port",         type=int,   default=9000,
                        help="服务监听端口")
    parser.add_argument("--model",        type=str,   default="F5TTS_Base",
                        help="模型名称（对应 configs/*.yaml）")
    parser.add_argument("--vocoder_name", type=str,   default="vocos",
                        help="Vocoder 名称")
    parser.add_argument("--speed",        type=float, default=1.0,
                        help="默认语速")
    parser.add_argument("--vocab_file",   type=str,   default="./models/vocab.txt",
                        help="词表文件路径")
    parser.add_argument("--ckpt_file",    type=str,   default="./models/model_last.pt",
                        help="模型权重文件路径")
    parser.add_argument("--ref_audio",    type=str,   default="./ref.wav",
                        help="默认参考音频路径")
    parser.add_argument("--ref_text",     type=str,   default="",
                        help="默认参考文本（留空则让 CLI 自动转录）")
    args = parser.parse_args()

    default_args.update({
        "model":        args.model,
        "vocoder_name": args.vocoder_name,
        "speed":        args.speed,
        "vocab_file":   args.vocab_file,
        "ckpt_file":    args.ckpt_file,
        "ref_audio":    args.ref_audio,
        "ref_text":     args.ref_text,
    })

    # 检查 f5-tts_infer-cli 是否可用
    cli_path = shutil.which("f5-tts_infer-cli")
    if cli_path is None:
        logger.error(
            "未找到 f5-tts_infer-cli，请确认已安装 f5-tts 并激活对应环境。"
        )
        sys.exit(1)
    logger.info("f5-tts_infer-cli 可用 ✓ (%s)", cli_path)

    if not os.path.exists(args.ref_audio):
        logger.warning("默认参考音频文件不存在: %s", args.ref_audio)
    if not os.path.exists(args.ckpt_file):
        logger.warning("模型权重文件不存在: %s", args.ckpt_file)
    if not os.path.exists(args.vocab_file):
        logger.warning("词表文件不存在: %s", args.vocab_file)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FILES_DIR, exist_ok=True)

    app = make_app()
    server = tornado.httpserver.HTTPServer(app, max_buffer_size=100 * 1024 * 1024)
    server.listen(args.port, address=args.host)
    logger.info(
        "F5-TTS Vietnamese HTTP 服务（CLI 版）已启动，监听地址: http://%s:%d",
        args.host, args.port,
    )
    logger.info(
        "可用接口:  POST /tts   GET /health   GET /audio/<file>   "
        "POST /files/upload   GET /files/check"
    )

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("收到退出信号，正在关闭服务...")
        tornado.ioloop.IOLoop.current().stop()


if __name__ == "__main__":
    main()

