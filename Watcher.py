import time
import os
import sys
import glob
from config import ConfigManager
from rag import RAGManager
from engine import AIEngine

ENCODINGS = ['utf-8', 'cp932', 'shift_jis']

class AIWatcher:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # ポストの場所（共有フォルダ）
        self.box_dir = os.path.join(os.path.dirname(self.base_dir), "exchange_box")
        
        if not os.path.exists(self.box_dir):
            os.makedirs(self.box_dir)

        print("だんご大家族（マルチユーザー版）を起動します...")
        self.config = ConfigManager(self.base_dir)
        self.rag = RAGManager(self.base_dir)
        self.engine = AIEngine(self.config)
        self.load_ai_model()

    def load_ai_model(self):
        model_name = self.config.params.get("last_model", "")
        if not model_name:
            gguf_files = glob.glob(os.path.join(self.base_dir, "gguf", "*.gguf"))
            if gguf_files: model_name = os.path.basename(gguf_files[0])
        
        if model_name:
            print(f"モデル準備完了: {model_name}")
            self.engine.load_model(os.path.join(self.base_dir, "gguf", model_name))

    def read_text_safe(self, path):
        for enc in ENCODINGS:
            try:
                with open(path, "r", encoding=enc) as f: return f.read()
            except: continue
        return ""

    def process_one_file(self, req_path):
        # ファイル名からIDを取得 (req_XXXX.txt -> XXXX)
        filename = os.path.basename(req_path)
        unique_id = filename.replace("req_", "").replace(".txt", "")
        
        # 質問を読む
        question = self.read_text_safe(req_path)
        if not question: 
            try: os.remove(req_path) # 空なら消す
            except: pass
            return

        print(f"📩 受信[{unique_id}]: {question[:15]}...")

        # リクエストファイルを削除（受付完了）
        try: os.remove(req_path)
        except: pass

        # RAG検索 & プロンプト作成
        ctx, files = self.rag.get_context(question)
        if files:
            rag_text = f"以下の【参照情報】を事実に回答してください。\n\n【参照情報】\n{ctx}"
            print(f"   📖 参照: {len(files)}件")
        else:
            rag_text = "親切に回答してください。"

        sys_msg = self.config.get_system_prompt("normal")
        model_name = self.config.params.get("last_model", "").lower()
        
        # プロンプト組立
        if "gemma" in model_name:
            prompt = f"<start_of_turn>user\n{sys_msg}\n\n{rag_text}\n\n【質問】\n{question}<end_of_turn>\n<start_of_turn>model\n"
        elif "elyza" in model_name or "llama-3" in model_name:
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}\n{rag_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        else:
            prompt = f"{sys_msg}\n\n{rag_text}\n\nユーザー: {question}\nシステム:"

        # 生成
        print(f"   ✍️ 回答生成中...", end="", flush=True)
        full_response = ""
        stream = self.engine.generate(prompt)
        if stream:
            for out in stream:
                full_response += out['choices'][0]['text']
        print(" 完了")

        # 返信ファイル作成 (res_XXXX.txt)
        res_path = os.path.join(self.box_dir, f"res_{unique_id}.txt")
        try:
            with open(res_path, "w", encoding="utf-8") as f:
                f.write(full_response)
        except Exception as e:
            print(f"保存エラー: {e}")

    def run(self):
        print(f"監視開始: {self.box_dir}")
        print("終了は Ctrl+C")
        
        while True:
            try:
                # "req_" で始まるファイルを全部見つける
                req_files = glob.glob(os.path.join(self.box_dir, "req_*.txt"))
                
                # 古い順（作成順）に並べ替える＝順番待ちを守る
                req_files.sort(key=os.path.getctime)
                
                for req_path in req_files:
                    # 1件処理する
                    self.process_one_file(req_path)
                    # 連続処理でPCが熱くならないよう一瞬休む
                    time.sleep(0.1)
                
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    AIWatcher().run()
