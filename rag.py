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
        
        # 設定ファイルからモデルパスを取得
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
                    
                    # チャンクサイズ設定
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
                
                np_vec = np.array(raw_vec, dtype='float32')
                embeddings.append(self._normalize(np_vec))
                
            except Exception as e:
                report(f"Error chunk {i}: {e}")
            
            if (i+1) % 5 == 0: 
                report(f"進捗: {i+1}/{len(new_chunks)} 完了")

        if not embeddings: return "ベクトル化失敗"

        np_embeddings = np.array(embeddings)
        dimension = np_embeddings.shape[1]

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
    # ★ここが大改造ポイント！「ハイブリッド検索」
    # ----------------------------------------------------------------
    def get_context(self, query):
        if self.index is None or not self.chunks: return "", []
        err = self._load_model()
        if err: 
            print(f"RAG Error: {err}")
            return "", []

        try:
            # 1. まずベクトルで「意味が近いもの」を探す
            vec_res = self.embed_model.create_embedding(query)
            query_vec = vec_res['data'][0]['embedding']
            if isinstance(query_vec[0], list): query_vec = query_vec[0]
            
            np_query = np.array(query_vec, dtype='float32')
            np_query = self._normalize(np_query)
            if np_query.ndim == 1: np_query = np.expand_dims(np_query, axis=0)
            
            # 多めに候補を取ってくる（50件）
            search_k = 50
            if search_k > len(self.chunks): search_k = len(self.chunks)
            
            distances, indices = self.index.search(np_query, search_k)
            
            # 2. キーワード（文字）の一致度でボーナス点を与える
            # 日本語はスペースがないので、文字集合(Set)の重なり具合(Jaccard係数)で判定
            q_chars = set(query)
            
            scored_chunks = []
            
            for i, vector_score in zip(indices[0], distances[0]):
                if i < len(self.chunks) and i >= 0:
                    chunk = self.chunks[i]
                    
                    # 文字の一致度を計算 (0.0 ～ 1.0)
                    # 例: 「天皇」という文字がchunkにあればスコアが跳ね上がる
                    c_chars = set(chunk)
                    intersection = len(q_chars & c_chars)
                    union = len(q_chars | c_chars)
                    keyword_score = 0.0
                    if union > 0:
                        keyword_score = intersection / union
                    
                    # ★最終スコア = ベクトルスコア + (キーワードスコア × 重み)
                    # 重みを 0.5 に設定して、文字一致の影響力を強めます
                    final_score = vector_score + (keyword_score * 0.5)
                    
                    scored_chunks.append({
                        "chunk": chunk,
                        "score": final_score,
                        "fname": chunk.split("【出典:")[1].split("】")[0]
                    })
            
            # 3. 最終スコアが高い順に並べ替え
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            
            # 4. 上位を採用する（偏り防止付き）
            results = []
            source_files = []
            file_counts = {}
            
            print(f"\n--- 検索ヒット状況 (Hybrid Rank) ---")
            for item in scored_chunks:
                fname = item["fname"]
                count = file_counts.get(fname, 0)
                if count >= 3: continue # 同じファイルからは3つまで
                
                results.append(item["chunk"])
                if fname not in source_files: source_files.append(fname)
                file_counts[fname] = count + 1
                
                print(f"・Score: {item['score']:.4f} | {fname}")
                
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
