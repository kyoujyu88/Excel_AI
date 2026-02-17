import os
import glob
import pickle
import json
import numpy as np
import faiss
import shutil
import tempfile
from llama_cpp import Llama 

class RAGManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.knowledge_dir = os.path.join(base_dir, "knowledge")
        self.db_path = os.path.join(base_dir, "vector_db")
        
        self.config_path = os.path.join(base_dir, "config.json")
        self.model_path = ""
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    last_model = cfg.get("last_model", "")
                    if last_model:
                        self.model_path = os.path.join(base_dir, "gguf", last_model)
            except: pass
        
        if not self.model_path or not os.path.exists(self.model_path):
            ggufs = glob.glob(os.path.join(base_dir, "gguf", "*.gguf"))
            if ggufs: self.model_path = ggufs[0]
            else: self.model_path = ""

        if not os.path.exists(self.knowledge_dir): os.makedirs(self.knowledge_dir)
        if not os.path.exists(self.db_path): os.makedirs(self.db_path)

        self.index = None
        self.chunks = []
        self.embed_model = None 
        
        self.load_db()

    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            return "モデルファイルが見つかりません。config.jsonを確認してください。"

        if self.embed_model is None:
            m_name = os.path.basename(self.model_path)
            print(f"Embeddingモデル読込中: {m_name}")
            try:
                self.embed_model = Llama(
                    model_path=self.model_path,
                    embedding=True,
                    verbose=False,
                    n_ctx=2048,
                    n_threads=6,
                    n_gpu_layers=0
                )
            except Exception as e:
                return f"モデル読込エラー: {e}"
        return None

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm == 0: return vec
        return vec / norm

    def build_database(self, callback=None):
        def report(msg):
            print(msg)
            if callback: callback(msg)

        err = self._load_model()
        if err: 
            report(err)
            return err

        files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        if not files: return "知識ファイル(.txt)がありません"

        report(f"【検出ファイル一覧】")
        for f in files: report(f" - {os.path.basename(f)}")
        report("-" * 20)

        new_chunks = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    filename = os.path.basename(file_path)
                    
                    chunk_size = 600
                    overlap = 100
                    
                    for i in range(0, len(text), chunk_size - overlap):
                        chunk_text = text[i : i + chunk_size].strip()
                        if len(chunk_text) > 20:
                            new_chunks.append(f"【出典:{filename}】\n{chunk_text}")
            except: pass

        if not new_chunks: return "有効なテキストがありませんでした"

        embeddings = []
        report(f"ベクトル化開始 ({len(new_chunks)}件)...")
        
        for i, chunk in enumerate(new_chunks):
            try:
                vec = self.embed_model.create_embedding(chunk)
                raw_vec = vec['data'][0]['embedding']
                if isinstance(raw_vec[0], list): raw_vec = raw_vec[0]
                
                # 正規化しないでそのまま入れる（Faissに任せる）
                # Elyzaなどのモデルは値が大きいため、ここで下手にいじると情報が消える可能性がある
                embeddings.append(raw_vec)
                
            except Exception as e:
                report(f"Error chunk {i}: {e}")
            
            if (i+1) % 5 == 0: 
                report(f"進捗: {i+1}/{len(new_chunks)} 完了")

        if not embeddings: return "ベクトル化失敗"

        np_embeddings = np.array(embeddings, dtype='float32')
        dimension = np_embeddings.shape[1]

        # 内積(IP)検索
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np_embeddings)
        self.chunks = new_chunks

        if not os.path.exists(self.db_path): os.makedirs(self.db_path)
        
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".faiss")
            os.close(fd)
            faiss.write_index(self.index, temp_path)
            
            target_path = os.path.join(self.db_path, "index.faiss")
            if os.path.exists(target_path): os.remove(target_path)
            shutil.move(temp_path, target_path)
            
            with open(os.path.join(self.db_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks, f)
        except Exception as e:
            msg = f"保存エラー: {e}"
            report(msg)
            return msg

        final_msg = f"完了！ {len(new_chunks)}件処理しました。"
        report(final_msg)
        return final_msg

    # ----------------------------------------------------------------
    # ★超・強化版ハイブリッド検索
    # ----------------------------------------------------------------
    def get_context(self, query):
        if self.index is None or not self.chunks: return "", []
        err = self._load_model()
        if err: 
            print(f"RAG Error: {err}")
            return "", []

        try:
            # 1. ベクトル検索
            vec_res = self.embed_model.create_embedding(query)
            query_vec = vec_res['data'][0]['embedding']
            if isinstance(query_vec[0], list): query_vec = query_vec[0]
            
            np_query = np.array([query_vec], dtype='float32')
            
            # 多めに候補を取る
            search_k = 50
            if search_k > len(self.chunks): search_k = len(self.chunks)
            
            distances, indices = self.index.search(np_query, search_k)
            
            # 2. スコア調整（キーワード一致の爆上げ）
            # 質問文から、意味のありそうな文字だけを抽出（ひらがな1文字などはノイズになりやすいが、今回は単純化）
            # 重複を除いた文字セットを作る
            q_chars = list(set(query.replace(" ", "").replace("　", "")))
            
            scored_chunks = []
            
            print(f"\n--- スコア計算内訳 (Base -> Bonus) ---")
            
            for i, vector_score in zip(indices[0], distances[0]):
                if i < len(self.chunks) and i >= 0:
                    chunk = self.chunks[i]
                    
                    # ★ここを変更：単純に「含まれている文字数」をカウントする
                    match_count = 0
                    matched_chars = []
                    for char in q_chars:
                        if char in chunk:
                            match_count += 1
                            matched_chars.append(char)
                    
                    # ★ボーナス計算
                    # 1文字ヒットするごとに「500点」加算します。
                    # これなら 19559点 vs 19560点 の僅差を一撃でひっくり返せます。
                    bonus_score = match_count * 500.0
                    
                    final_score = vector_score + bonus_score
                    
                    # ログ出し（デバッグ用）
                    fname = chunk.split("【出典:")[1].split("】")[0]
                    if match_count > 0 and len(scored_chunks) < 10:
                        print(f"[{fname[:5]}...] Vec:{vector_score:.1f} + Bonus:{bonus_score:.0f} (Match:{''.join(matched_chars)})")

                    scored_chunks.append({
                        "chunk": chunk,
                        "score": final_score,
                        "fname": fname
                    })
            
            # 3. 並べ替え
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            
            # 4. 採用
            results = []
            source_files = []
            file_counts = {}
            
            print(f"\n--- 最終検索結果 (Top 6) ---")
            for item in scored_chunks:
                fname = item["fname"]
                count = file_counts.get(fname, 0)
                if count >= 3: continue 
                
                results.append(item["chunk"])
                if fname not in source_files: source_files.append(fname)
                file_counts[fname] = count + 1
                
                print(f"・Total: {item['score']:.1f} | {fname}")
                
                if len(results) >= 6: break
            print("--------------------------------\n")

            if results:
                context_text = "\n\n".join(results)
                formatted = f"\n\n### 🧠 知識データベース参照 ###\n{context_text}\n#############################\n"
                return formatted, source_files
        except Exception as e:
            print(f"検索エラー: {e}")
        
        return "", []

    def open_folder(self): os.startfile(self.knowledge_dir)
    def load_user_file(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
        except: return None
    def load_db(self):
        try:
            idx = os.path.join(self.db_path, "index.faiss")
            chk = os.path.join(self.db_path, "chunks.pkl")
            if os.path.exists(idx) and os.path.exists(chk):
                self.index = faiss.read_index(idx)
                with open(chk, "rb") as f: self.chunks = pickle.load(f)
                print("DB読込完了")
        except: pass
