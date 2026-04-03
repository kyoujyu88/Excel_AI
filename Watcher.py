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
from pypdf import PdfReader  # ★ pypdf に変更しました！

class AIWatcher:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.box_dir = os.path.join(os.path.dirname(self.base_dir), "exchange_box")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.log_file = os.path.join(self.log_dir, "history.csv")
        
        if not os.path.exists(self.box_dir): os.makedirs(self.box_dir)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="cp932", newline="", errors="replace") as f:
                writer = csv.writer(f)
                writer.writerow(["日時", "ユーザーID", "質問内容", "AI回答"])

        print("だんご大家族（pypdf 査読対応版）を起動します...")
        self.cleanup_box(max_age_minutes=10)
        
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
        else:
            print("警告: モデルが見つかりません。")

    def cleanup_box(self, max_age_minutes=5):
        try:
            now = time.time()
            # txt と pdf の両方をお掃除対象にします
            files = glob.glob(os.path.join(self.box_dir, "*_*.txt")) + glob.glob(os.path.join(self.box_dir, "req_*.pdf"))
            for f in files:
                if "status.txt" in f: continue
                ctime = os.path.getctime(f)
                if now - ctime > (max_age_minutes * 60):
                    try: os.remove(f)
                    except: pass
        except: pass

    # =========================================================
    # ★ pypdf を使った査読（校正）専用の処理
    # =========================================================
    def process_pdf_file(self, pdf_path, unique_id):
        print(f"📄 PDF査読開始[{unique_id}]")
        full_report = f"【PDF査読結果：{unique_id}】\n\n"
        
        # 査読用の厳しいプロンプトを読み込む
        sys_msg = self.config.get_system_prompt("proofread") 
        model_name = self.config.params.get("last_model", "").lower()

        try:
            # pypdfでPDFを読み込む
            reader = PdfReader(pdf_path)
            
            for page_num, page in enumerate(reader.pages):
                # テキストを抽出
                text = page.extract_text()
                
                if not text or not text.strip():
                    continue # 空白ページや画像だけのページは飛ばします

                print(f"   📖 第{page_num+1}ページ目をチェック中...", end="", flush=True)
                
                # AIに渡す質問文を組み立てる
                question = f"【対象テキスト：第{page_num+1}ページ】\n{text.strip()}"
                
                if "gemma" in model_name:
                    prompt = f"<start_of_turn>user\n{sys_msg}\n\n{question}<end_of_turn>\n<start_of_turn>model\n"
                elif "elyza" in model_name or "llama-3" in model_name:
                    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                else:
                    prompt = f"{sys_msg}\n\nユーザー: {question}\nシステム:"

                # AIに考えさせる
                response = self.engine.generate(prompt)
                
                if isinstance(response, dict):
                    response = response['choices'][0]['text']
                if not response:
                    response = "（エラー：生成失敗）"
                
                # レポートの末尾に追記していく
                full_report += f"{response}\n\n-------------------------\n\n"
                print(" 完了")
            
            # 処理が終わったPDFファイルはシュレッダーにかけます
            os.remove(pdf_path) 

        except Exception as e:
            full_report = f"PDFの読み込み中にエラーが発生しました: {e}"
            print(f"PDFエラー: {e}")
            try: os.remove(pdf_path)
            except: pass

        # 最終的なレポートをテキストファイルとして返す
        self.save_and_move_result(unique_id, full_report)

    def process_one_file(self, req_path):
        filename = os.path.basename(req_path)
        unique_id = filename.replace("req_", "").replace(".txt", "").replace(".pdf", "")
        ext = os.path.splitext(filename)[1].lower()

        # もし投げ込まれたのがPDFなら、査読係に任せる
        if ext == ".pdf":
            self.process_pdf_file(req_path, unique_id)
            return

        # -----------------------------------------------------
        # 以下は通常のテキストチャット（RAG）の処理です
        # -----------------------------------------------------
        question = ""
        try:
            with open(req_path, "r", encoding="cp932", errors="ignore") as f:
                question = f.read()
        except: pass

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
        full_response = self.engine.generate(prompt)
        
        if isinstance(full_response, dict):
             full_response = full_response['choices'][0]['text']
        if not full_response: 
            full_response = "（エラー：回答の生成に失敗しました）"
        
        print(" 完了")
        self.save_history(unique_id, question, full_response)
        self.save_and_move_result(unique_id, full_response)

    def save_and_move_result(self, unique_id, text):
        final_path = os.path.join(self.box_dir, f"res_{unique_id}.txt")
        temp_path = os.path.join(self.box_dir, f"tmp_{unique_id}.txt")
        try:
            with open(temp_path, "w", encoding="cp932", errors="replace") as f:
                f.write(text)
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
        except: pass

    def run(self):
        print(f"監視開始: {self.box_dir}")
        status_file = os.path.join(self.box_dir, "status.txt")
        last_heartbeat = 0
        last_cleanup = 0 
        
        while True:
            try:
                now = time.time()
                if now - last_heartbeat > 5.0:
                    try:
                        with open(status_file, "w", encoding="cp932") as f:
                            f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " - READY")
                        last_heartbeat = now
                    except: pass
                
                if now - last_cleanup > 60.0:
                    self.cleanup_box(max_age_minutes=5)
                    last_cleanup = now

                req_files = glob.glob(os.path.join(self.box_dir, "req_*.txt")) + glob.glob(os.path.join(self.box_dir, "req_*.pdf"))
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
