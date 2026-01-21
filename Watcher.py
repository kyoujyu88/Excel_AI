import time
import os
import sys
import glob
from config import ConfigManager
from rag import RAGManager
from engine import AIEngine

# 文字化け対策（ExcelはShift-JISが好きなので対応します）
ENCODINGS = ['utf-8', 'cp932', 'shift_jis']

class AIWatcher:
    def __init__(self):
        # 現在のフォルダの場所
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 郵便ポストの場所（ひとつ上の階層の exchange_box）
        self.box_dir = os.path.join(os.path.dirname(self.base_dir), "exchange_box")
        
        # ファイルのパス定義
        self.req_file = os.path.join(self.box_dir, "request.txt")   # 質問（Excelから）
        self.res_file = os.path.join(self.box_dir, "response.txt")  # 回答（AIから）
        self.busy_file = os.path.join(self.box_dir, "writing.lock") # 書き込み中サイン

        # フォルダがなければ作る
        if not os.path.exists(self.box_dir):
            os.makedirs(self.box_dir)
            print(f"ポストを作りました: {self.box_dir}")

        # AIの準備（いつもの読み込み）
        print("だんご大家族を呼んでいます...（AI起動中）")
        self.config = ConfigManager(self.base_dir)
        self.rag = RAGManager(self.base_dir)
        self.engine = AIEngine(self.config)
        
        # モデル読み込み（設定ファイルの前回モデルを使用）
        self.load_ai_model()

    def load_ai_model(self):
        # last_model が空なら、ggufフォルダの最初のやつを使う
        model_name = self.config.params.get("last_model", "")
        if not model_name:
            gguf_files = glob.glob(os.path.join(self.base_dir, "gguf", "*.gguf"))
            if gguf_files:
                model_name = os.path.basename(gguf_files[0])
        
        if model_name:
            path = os.path.join(self.base_dir, "gguf", model_name)
            print(f"モデルを準備しています: {model_name}")
            ok, msg = self.engine.load_model(path)
            if not ok: print(f"エラー: {msg}")
            else: print("準備完了です！ ポストを見張ります。")
        else:
            print("エラー: モデルが見つかりません。")

    def read_text_safe(self, path):
        # どの文字コードでも読めるように頑張る関数
        for enc in ENCODINGS:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except: continue
        return ""

    def process_request(self):
        # 1. 質問を読み取る
        question = self.read_text_safe(self.req_file)
        if not question: return

        print(f"お手紙が届きました！: {question[:20]}...")

        # 2. ロックファイル作成（Excelに「今、考え中だよ」と伝える）
        with open(self.busy_file, "w") as f: f.write("BUSY")

        # 3. リクエストファイルを削除（二重処理防止）
        try: os.remove(self.req_file)
        except: pass

        # 4. RAG検索
        ctx, files = self.rag.get_context(question)
        rag_text = ""
        if files:
            print(f"  参照資料: {', '.join(files)}")
            rag_text = f"以下の【参照情報】を元に回答してください。\n\n【参照情報】\n{ctx}"
        else:
            rag_text = "親切に回答してください。"

        # 5. プロンプト作成（GUI版と同じロジック）
        sys_msg = self.config.get_system_prompt("normal")
        model_name = self.config.params.get("last_model", "").lower()
        prompt = ""

        if "gemma" in model_name:
            prompt = f"<start_of_turn>user\n{sys_msg}\n\n{rag_text}\n\n【質問】\n{question}<end_of_turn>\n<start_of_turn>model\n"
        elif "elyza" in model_name or "llama-3" in model_name:
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}\n{rag_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        else:
            prompt = f"{sys_msg}\n\n{rag_text}\n\nユーザー: {question}\nシステム:"

        # 6. 生成（ストリームじゃなく一気に待つ）
        print("  お返事を書いています...", end="", flush=True)
        full_response = ""
        stream = self.engine.generate(prompt)
        
        if stream:
            for out in stream:
                text = out['choices'][0]['text']
                full_response += text
        print(" 完了！")

        # 7. 回答をファイルに書き込む（UTF-8で保存）
        # ※ Excel側で読み込むときに文字コード変換させます
        try:
            with open(self.res_file, "w", encoding="utf-8") as f:
                f.write(full_response)
        except Exception as e:
            print(f"書き込みエラー: {e}")

        # 8. ロック解除（書き込み終了）
        if os.path.exists(self.busy_file):
            os.remove(self.busy_file)

    def run(self):
        print(f"[{self.box_dir}] を監視中...")
        print("終了するには Ctrl+C を押してください")
        
        while True:
            try:
                # request.txt があるかチェック
                if os.path.exists(self.req_file):
                    # ファイルが書き込み中でないか、少しだけ待ってから読む
                    time.sleep(0.5)
                    self.process_request()
                
                # 0.5秒休憩（CPUに優しく）
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n見張り番を終了します。お疲れ様でした！")
                break
            except Exception as e:
                print(f"エラー発生: {e}")
                time.sleep(1)

if __name__ == "__main__":
    watcher = AIWatcher()
    watcher.run()
