import time
import os
import sys
import glob
import shutil
import csv
from datetime import datetime
from config import ConfigManager
from rag import RAGManager
from engine import AIEngine

ENCODINGS = ['utf-8', 'cp932', 'shift_jis']

class AIWatcher:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.box_dir = os.path.join(os.path.dirname(self.base_dir), "exchange_box")
        
        # 履歴ログ設定
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.log_file = os.path.join(self.log_dir, "history.csv")
        
        if not os.path.exists(self.box_dir): os.makedirs(self.box_dir)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="cp932", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["日時", "ユーザーID", "質問内容", "AI回答"])

        print("だんご大家族（一括生成対応版）を起動します...")
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

    def save_history(self, uid, question, answer):
        try:
            now_str = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            with open(self.log_file, "a", encoding="cp932", errors="replace", newline="") as f:
                writer = csv.writer(f)
                clean_q = question.replace("\n", " ").replace("\r", "")
                clean_a = answer.replace("\n", " ").replace("\r", "")
                writer.writerow([now_str, uid, clean_q, clean_a])
            print("   📒 履歴を記録しました")
        except Exception as e:
            print(f"   ⚠️ 履歴保存エラー: {e}")

    def process_one_file(self, req_path):
        filename = os.path.basename(req_path)
        unique_id = filename.replace("req_", "").replace(".txt", "")
        
        question = self.read_text_safe(req_path)
        if not question: 
            try: os.remove(req_path)
            except: pass
            return

        print(f"📩 受信[{unique_id}]: {question[:15]}...")

        try: os.remove(req_path)
        except: pass

        ctx, files = self.rag.get_context(question)
        rag_text = f"以下の情報を元に回答。\n{ctx}" if files else "親切に回答してください。"
        
        sys_msg = self.config.get_system_prompt("normal")
        model_name = self.config.params.get("last_model", "").lower()
        
        if "gemma" in model_name:
            prompt = f"<start_of_turn>user\n{sys_msg}\n\n{rag_text}\n\n【質問】\n{question}<end_of_turn>\n<start_of_turn>model\n"
        elif "elyza" in model_name or "llama-3" in model_name:
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}\n{rag_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        else:
            prompt = f"{sys_msg}\n\n{rag_text}\n\nユーザー: {question}\nシステム:"

        print(f"   ✍️ 回答生成中...", end="", flush=True)
        
        # -------------------------------------------------------
        # ★ここを修正しました！ (一発受け取り)
        # -------------------------------------------------------
        full_response = self.engine.generate(prompt)
        
        # 万が一エラーで None が返ってきた時の保険
        if full_response is None: 
            full_response = "（エラー：回答の生成に失敗しました）"
        
        print(" 完了")

        # 履歴保存
        self.save_history(unique_id, question, full_response)

        # 安全な書き込み処理
        final_path = os.path.join(self.box_dir, f"res_{unique_id}.txt")
        temp_path = os.path.join(self.box_dir, f"tmp_{unique_id}.txt")
        
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(full_response)
            shutil.move(temp_path, final_path)
        except Exception as e:
            print(f"保存エラー: {e}")
            if os.path.exists(temp_path): os.remove(temp_path)

    def run(self):
        print(f"監視開始: {self.box_dir}")
        print(f"履歴保存先: {self.log_file}")
        
        while True:
            try:
                req_files = glob.glob(os.path.join(self.box_dir, "req_*.txt"))
                req_files.sort(key=os.path.getctime)
                
                for req_path in req_files:
                    self.process_one_file(req_path)
                    time.sleep(0.1)
                
                time.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    AIWatcher().run()
