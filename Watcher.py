    # (前略... AIWatcherクラスの中)

    def run(self):
        print(f"監視開始: {self.box_dir}")
        print(f"履歴保存先: {self.log_file}")
        
        # ★追加：ステータスファイルの場所
        status_file = os.path.join(self.box_dir, "status.txt")
        last_heartbeat = 0
        
        while True:
            try:
                # ------------------------------------------------
                # ★追加機能：5秒に1回、生存報告をする（ハートビート）
                # ------------------------------------------------
                if time.time() - last_heartbeat > 5.0:
                    try:
                        # 中身は何でもいいですが、現在時刻を書いておきます
                        with open(status_file, "w", encoding="utf-8") as f:
                            f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " - READY")
                        last_heartbeat = time.time()
                    except:
                        pass # ファイルロックなどで失敗しても気にしない
                
                # --- (いつもの監視処理) ---
                req_files = glob.glob(os.path.join(self.box_dir, "req_*.txt"))
                req_files.sort(key=os.path.getctime)
                
                for req_path in req_files:
                    self.process_one_file(req_path)
                    time.sleep(0.1)
                
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                # ★終了時にステータスファイルを消すと親切
                if os.path.exists(status_file):
                    os.remove(status_file)
                print("\n終了します。")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
