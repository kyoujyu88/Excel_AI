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
from pypdf import PdfReader

class AIWatcher:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ---------------------------------------------------------
        # ★設定：共有フォルダのパス
        # 実際の運用環境に合わせて、SharePointの同期フォルダ等の正しいパスに書き換えてください。
        # ---------------------------------------------------------
        self.box_dir = os.path.join(os.path.dirname(self.base_dir), "exchange_box")
        
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.log_file = os.path.join(self.log_dir, "history.csv")
        
        if not os.path.exists(self.box_dir): os.makedirs(self.box_dir)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="cp932", newline="", errors="replace") as f:
                writer = csv.writer(f)
                writer.writerow(["日時", "ユーザーID", "質問内容", "AI回答"])

        print("だんご大家族（PDF査読・スマートリスト対応版）を起動します...")
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
            files = glob.glob(os.path.join(self.box_dir, "*_*.txt")) + glob.glob(os.path.join(self.box_dir, "req_*.pdf"))
            for f in files:
                if "status.txt" in f: continue
                ctime = os.path.getctime(f)
                if now - ctime > (max_age_minutes * 60):
                    try: os.remove(f)
                    except: pass
        except: pass

    # =========================================================
    # 📄 PDFファイルの査読（校正）処理
    # =========================================================
    def process_pdf_file(self, pdf_path, unique_id):
        print(f"\n📄 PDF査読開始 [{unique_id}]")
        full_report = f"【PDF査読結果】\n\n"
        
        # 査読プロンプトの読み込み
        sys_msg = self.config.get_system_prompt("proofread")
        
        # ---------------------------------------------------------
        # ★安全装置：もしproofread.txtが空っぽだったり読めなかった場合
        # ---------------------------------------------------------
        if not sys_msg or sys_msg.strip() == "":
            print("   ⚠️ proofread.txt が読み込めなかったため、予備のプロンプトを使用します！")
            sys_msg = "あなたは丁寧な校正アシスタントです。一般的な文法や自然さだけをチェックし、専門用語はスルーしてください。問題なければ「特になし」と答えてください。\n\n【報告形式】\n・[P.○] 【対象箇所】: 「～」\n  【理由】: ～\n  【修正案】: 「～」"

        # 確認用：今回使うプロンプトをコンソールに表示
        print("\n=== 🔍 今回AIに渡す指示書（プロンプト） ===")
        print(sys_msg)
        print("==========================================\n")

        model_name = self.config.params.get("last_model", "").lower()
        error_count = 0 # 指摘の数をカウントします

        try:
            reader = PdfReader(pdf_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                if not text or not text.strip():
                    continue # 空白ページはスキップ

                print(f"   📖 第{page_num+1}ページ目をチェック中...", end="", flush=True)
                
                # ページ数をAIに伝えるための設定
                question = f"【対象テキスト：第{page_num+1}ページ】\n{text.strip()}"
                
                # モデルに合わせたプロンプトの組み立て
                if "gemma" in model_name:
                    prompt = f"<start_of_turn>user\n{sys_msg}\n\n{question}<end_of_turn>\n<start_of_turn>model\n"
                elif "elyza" in model_name or "llama-3" in model_name:
                    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                else:
                    prompt = f"{sys_msg}\n\nユーザー: {question}\nシステム:"

                response = self.engine.generate(prompt)
                
                if isinstance(response, dict):
                    response = response['choices'][0]['text']
                if not response:
                    response = "（エラー：生成失敗）"
                
                # ---------------------------------------------------------
                # 「特になし」ならスルー、指摘があればリストに追加
                # ---------------------------------------------------------
                if "特になし" in response or response.strip() == "":
                    print(" 問題なし")
                else:
                    full_report += f"{response.strip()}\n\n"
                    error_count += 1
                    print(" ⚠️指摘あり")
            
            # 全部のページが完璧だった場合
            if error_count == 0:
                full_report += "指摘する箇所はありませんでした。素晴らしい文章です！\n"
            
            # 終わったPDFは削除します
            try:
                os.remove(pdf_path)
            except Exception as e:
                pass

        except Exception as e:
            full_report = f"PDFの読み込み中にエラーが発生しました: {e}"
            print(f"   ⚠️ PDFエラー: {e}")
            try: os.remove(pdf_path)
            except: pass

        self.save_and_move_result(unique_id, full_report)

    # =========================================================
    # 📝 通常のテキストファイル（RAGチャット）処理
    # =========================================================
    def process_one_file(self, req_path):
        filename = os.path.basename(req_path)
        unique_id = filename.replace("req_", "").replace(".txt", "").replace(".pdf", "")
        ext = os.path.splitext(filename)[1].lower()

        # PDFなら査読処理へ分岐
        if ext == ".pdf":
            self.process_pdf_file(req_path, unique_id)
            return

        question = ""
        try:
            with open(req_path, "r", encoding="cp932", errors="ignore") as f:
                question = f.read()
        except: pass

        if not question: 
            try: os.remove(req_path)
            except: pass
            return

        print(f"\n📩 受信[{unique_id}]: {question[:15]}...")
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

    # =========================================================
    # 💾 保存や記録の共通処理
    # =========================================================
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

    # =========================================================
    # メインループ
    # =========================================================
    def run(self):
        print(f"監視開始: {self.box_dir}")
        status_file = os.path.join(self.box_dir, "status.txt")
        last_heartbeat = 0
        last_cleanup = 0 
        
        while True:
            try:
                now = time.time()
                # 5秒に1回の「生きてるよ！」の合図
                if now - last_heartbeat > 5.0:
                    try:
                        with open(status_file, "w", encoding="cp932") as f:
                            f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " - READY")
                        last_heartbeat = now
                    except: pass
                
                # 60秒に1回のポストのお掃除
                if now - last_cleanup > 60.0:
                    self.cleanup_box(max_age_minutes=5)
                    last_cleanup = now

                # txt と pdf の両方を探します
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
