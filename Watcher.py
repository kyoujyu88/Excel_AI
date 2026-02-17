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

class AIWatcher:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ---------------------------------------------------------
        # ★SharePointなど、実際のフォルダパスに合わせてください
        # ---------------------------------------------------------
        # self.box_dir = r"\\SharePoint\Server\exchange_box"
        self.box_dir = os.path.join(os.path.dirname(self.base_dir), "exchange_box")
        
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.log_file = os.path.join(self.log_dir, "history.csv")
        
        if not os.path.exists(self.box_dir): os.makedirs(self.box_dir)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        # 履歴ファイルもShift-JIS(cp932)で統一
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="cp932", newline="", errors="replace") as f:
                writer = csv.writer(f)
                writer.writerow(["日時", "ユーザーID", "質問内容", "AI回答"])

        print("だんご大家族（プロンプト確認機能付き）を起動します...")
        self.config = ConfigManager(self.base_dir)
        self.rag = RAGManager(self.base_dir)
        self.engine = AIEngine(self.config)
        self.load_ai_model()

        # ---------------------------------------------------------
        # ★追加機能：現在のプロンプトを表示する
        # ---------------------------------------------------------
        # 現在使用するモード（基本はnormal）
        current_mode = "normal"
        sys_msg = self.config.get_system_prompt(current_mode)
        
        print("\n" + "="*60)
        print(f" 📝 現在のシステムプロンプト (モード: {current_mode})")
        print("="*60)
        print(sys_msg)
        print("="*60 + "\n")

    def load_ai_model(self):
        model_name = self.config.params.get("last_model", "")
        if not model_name:
            gguf_files = glob.glob(os.path.join(self.base_dir, "gguf", "*.gguf"))
            if gguf_files: model_name = os.path.basename(gguf_files[0])
        
        if model_name:
            print(f"モデル準備完了: {model_name}")
            self.engine.load_model(os.path.join(self.base_dir, "gguf", model_name))
        else:
            print("警告: モデルが見つかりません。")

    def process_one_file(self, req_path):
        filename = os.path.basename(req_path)
        unique_id = filename.replace("req_", "").replace(".txt", "")
        
        # ★読み込み：Shift-JIS (cp932) で強制的に読む
        question = ""
        try:
            with open(req_path, "r", encoding="cp932", errors="ignore") as f:
                question = f.read()
        except:
            pass

        if not question: 
            try: os.remove(req_path)
            except: pass
            return

        print(f"📩 受信[{unique_id}]: {question[:15]}...")

        try: os.remove(req_path)
        except: pass

        ctx, files = self.rag.get_context(question)
        rag_text = f"以下の情報を元に回答。\n{ctx}" if files else "親切に回答してください。"
        
        # プロンプト取得（normalモード固定）
        sys_msg = self.config.get_system_prompt("normal")
        model_name = self.config.params.get("last_model", "").lower()
        
        # プロンプト作成
        if "gemma" in model_name:
            prompt = f"<start_of_turn>user\n{sys_msg}\n\n{rag_text}\n\n【質問】\n{question}<end_of_turn>\n<start_of_turn>model\n"
        elif "elyza" in model_name or "llama-3" in model_name:
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}\n{rag_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        else:
            prompt = f"{sys_msg}\n\n{rag_text}\n\nユーザー: {question}\nシステム:"

        print(f"   ✍️ 回答生成中...", end="", flush=True)
        
        # ★生成：一括取得
        full_response = self.engine.generate(prompt)
        if full_response is None: 
            full_response = "（エラー：回答の生成に失敗しました）"
        elif isinstance(full_response, dict):
             full_response = full_response['choices'][0]['text']
        
        print(" 完了")

        # 履歴保存
        self.save_history(unique_id, question, full_response)

        # ★保存：Shift-JIS (cp932) で書き込む
        final_path = os.path.join(self.box_dir, f"res_{unique_id}.txt")
        temp_path = os.path.join(self.box_dir, f"tmp_{unique_id}.txt")
        
        try:
            with open(temp_path, "w", encoding="cp932", errors="replace") as f:
                f.write(full_response)
            shutil.move(temp_path, final_path)
        except Exception as e:
            print(f"保存エラー: {e}")
            if os.path.exists(temp_path): os.remove(temp_path)

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

    def run(self):
        print(f"監視開始: {self.box_dir}")
        print(f"履歴保存先: {self.log_file}")
        
        status_file = os.path.join(self.box_dir, "status.txt")
        last_heartbeat = 0
        
        while True:
            try:
                # ハートビート
                if time.time() - last_heartbeat > 5.0:
                    try:
                        with open(status_file, "w", encoding="cp932") as f:
                            f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " - READY")
                        last_heartbeat = time.time()
                    except: pass
                
                req_files = glob.glob(os.path.join(self.box_dir, "req_*.txt"))
                req_files.sort(key=os.path.getctime)
                
                for req_path in req_files:
                    self.process_one_file(req_path)
                    time.sleep(0.1)
                
                time.sleep(1.0)
                
            except KeyboardInterrupt:
                print("\n終了します。")
                if os.path.exists(status_file):
                    try: os.remove(status_file)
                    except: pass
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    AIWatcher().run()
