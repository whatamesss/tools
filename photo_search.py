#!/usr/bin/env python3.13
import sys
import os
import torch
import json
import argparse
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QListWidgetItem, 
    QVBoxLayout, QWidget, QLineEdit, QLabel, QStatusBar, 
    QFileDialog, QMessageBox, QToolBar, QHBoxLayout, QPushButton, QMenu
)
from PyQt6.QtGui import QIcon, QPixmap, QImage, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl, QFile
from PyQt6.QtGui import QDesktopServices

# --- Backend Logic ---

class PhotoSearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Info] Loading CLIP model on {self.device}...")
        
        model_id = "openai/clip-vit-large-patch14"
        
        # SMART LOADING:
        try:
            print("[Info] Attempting offline load from cache...")
            self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=True, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_id, local_files_only=True)
            print("[Info] Model loaded successfully (Offline).")
        except OSError:
            print("-" * 40)
            print("[Warning] Model files not found in cache.")
            print("[Info] Connecting to Hugging Face Hub to download...")
            print("-" * 40)
            self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
            self.model = CLIPModel.from_pretrained(model_id)
            print("\n[Info] Download complete. Future runs will be offline.")
        
        if self.device == "cuda":
            self.model = self.model.half()
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.projection_dim
        
        self.cache_dir = Path.home() / ".cache" / "photo_search"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.cache_dir / "photo_index.json"
        self.embeddings_file = self.cache_dir / "photo_embeddings.pt"
        self.metadata_file = self.cache_dir / "photo_metadata.json"
        
        self.image_paths = []
        self.image_embeddings = None
        self.image_metadata = {}
        self.indexed_set = set()

    def load_index(self):
        if self.index_file.exists() and self.embeddings_file.exists():
            try:
                raw_paths = json.load(open(self.index_file, "r"))
                self.image_paths = raw_paths
                
                # FIX 1: PyTorch 2.x compatibility
                loaded_embeddings = torch.load(str(self.embeddings_file), map_location='cpu', weights_only=False)
                
                if loaded_embeddings.shape[1] != self.embedding_dim:
                    print(f"[Backend] Incompatible index detected. Re-index required.")
                    return False

                self.image_embeddings = loaded_embeddings
                self.indexed_set = set(raw_paths)
                
                if self.metadata_file.exists():
                    self.image_metadata = json.load(open(self.metadata_file, "r"))
                else:
                    self.image_metadata = {}
                
                for p in self.image_paths:
                    if p not in self.image_metadata:
                        self.image_metadata[p] = {"std_dev": 50.0, "edge_score": 50.0}

                print(f"[Backend] Loaded {len(self.image_paths)} image embeddings.")
                return True
            except Exception as e:
                print(f"[Backend] Error loading index: {e}")
        return False

    def _load_single_image(self, file_path):
        try:
            img = Image.open(file_path)
            img = ImageOps.exif_transpose(img)
            
            # OPTIMIZATION: Resize large images immediately to save RAM.
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
            img = img.convert("RGB")
            
            # Calculate metadata on a small thumbnail
            meta_img = img.copy()
            meta_img.thumbnail((512, 512))
            gray = np.array(meta_img.convert("L"))
            
            std_dev = float(np.std(gray))
            
            # FIX 2: Corrected NumPy slice for 'dy' to fix "No Photos" bug
            dx = np.abs(gray[:, 2:] - gray[:, :-2])
            dy = np.abs(gray[2:, :] - gray[:-2, :]) 
            edge_score = float(np.mean(dx) + np.mean(dy))
            
            return img, str(file_path.resolve()), {
                "std_dev": std_dev, 
                "edge_score": edge_score
            }
            
        except Exception:
            return None, None, None

    def index_photos(self, source_dir, progress_callback=None):
        source_path = Path(source_dir).resolve()
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        print(f"[Info] Scanning {source_path} for images...")
        
        files = []
        for root, dirs, filenames in os.walk(source_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in filenames:
                if Path(filename).suffix.lower() in extensions:
                    files.append(Path(root) / filename)
        
        if self.indexed_set:
            initial_count = len(files)
            files = [f for f in files if str(f.resolve()) not in self.indexed_set]
            skipped = initial_count - len(files)
            if skipped > 0:
                print(f"[Info] Skipped {skipped} already indexed images.")

        if not files:
            print("[Info] No new images to index.")
            return

        print(f"[Info] Processing {len(files)} new images...")

        batch_size = 48
        if self.device == "cuda":
            try:
                free_vram, total_vram = torch.cuda.mem_get_info()
                free_vram_mb = free_vram / (1024**2)
                estimated_size_per_image = 20
                target_usage = free_vram_mb * 0.85
                calculated_bs = int(target_usage / estimated_size_per_image)
                batch_size = max(32, min(512, calculated_bs))
                print(f"[Info] VRAM {free_vram_mb:.0f}MB. Batch size: {batch_size}")
            except Exception:
                pass

        new_embeddings = []
        new_paths = []
        new_metadata = []
        processed_count = 0

        max_workers = min(16, os.cpu_count() or 4) 
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._load_single_image, f): f for f in files}
            
            batch_images = []
            batch_map = []
            batch_meta = []

            for future in as_completed(future_to_file):
                img, path, meta = future.result()
                
                if img is None:
                    continue
                
                batch_images.append(img)
                batch_map.append(path)
                batch_meta.append(meta)

                if len(batch_images) >= batch_size:
                    self._process_batch(batch_images, batch_map, batch_meta, 
                                        new_embeddings, new_paths, new_metadata)
                    
                    processed_count += len(batch_images)
                    if progress_callback:
                        progress_callback(f"Indexing... {processed_count}/{len(files)} images")
                    
                    batch_images = []
                    batch_map = []
                    batch_meta = []

            if batch_images:
                self._process_batch(batch_images, batch_map, batch_meta, 
                                    new_embeddings, new_paths, new_metadata)
                processed_count += len(batch_images)
                if progress_callback:
                    progress_callback(f"Indexing... {processed_count}/{len(files)} images")

        if new_embeddings:
            if self.image_embeddings is not None:
                all_embeddings = torch.cat([self.image_embeddings, torch.cat(new_embeddings, dim=0)], dim=0)
                all_paths = self.image_paths + new_paths
            else:
                all_embeddings = torch.cat(new_embeddings, dim=0)
                all_paths = new_paths

            for i, p in enumerate(new_paths):
                self.image_metadata[p] = new_metadata[i]

            print(f"[Info] Saving index ({len(all_paths)} total embeddings)...")
            
            # Atomic save
            temp_emb = str(self.embeddings_file) + ".tmp"
            temp_idx = str(self.index_file) + ".tmp"
            temp_meta = str(self.metadata_file) + ".tmp"
            
            torch.save(all_embeddings, temp_emb)
            with open(temp_idx, "w") as f:
                json.dump(all_paths, f)
            with open(temp_meta, "w") as f:
                json.dump(self.image_metadata, f)
                
            os.replace(temp_emb, str(self.embeddings_file))
            os.replace(temp_idx, str(self.index_file))
            os.replace(temp_meta, str(self.metadata_file))

            self.image_embeddings = all_embeddings
            self.image_paths = all_paths
            self.indexed_set = set(self.image_paths)
            print("[Info] Indexing complete.")
        else:
            print("[Info] No valid images processed.")

    def _process_batch(self, batch_images, batch_map, batch_meta, 
                       new_embeddings, new_paths, new_metadata):
        inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda":
            inputs['pixel_values'] = inputs['pixel_values'].half()

        with torch.no_grad():
            raw_output = self.model.get_image_features(**inputs)
            
            # FIX 3: Transformers 5.x compatibility
            if hasattr(raw_output, 'pooler_output'):
                features = raw_output.pooler_output
            elif isinstance(raw_output, tuple):
                features = raw_output[0]
            else:
                features = raw_output
                
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            new_embeddings.append(features.cpu())
            new_paths.extend(batch_map)
            new_metadata.extend(batch_meta)

    def clean_database(self, progress_callback=None):
        if not self.image_paths:
            return 0

        valid_paths = []
        valid_indices = []
        removed_count = 0
        
        total = len(self.image_paths)
        
        for i, path in enumerate(self.image_paths):
            if os.path.exists(path):
                valid_paths.append(path)
                valid_indices.append(i)
            else:
                removed_count += 1
                if path in self.image_metadata:
                    del self.image_metadata[path]
            
            if progress_callback and i % 1000 == 0:
                progress_callback(f"Verifying... {i}/{total}")

        if removed_count > 0:
            print(f"[Info] Cleaning database: Removing {removed_count} missing files.")
            
            self.image_paths = valid_paths
            self.indexed_set = set(self.image_paths)
            
            if self.image_embeddings is not None:
                if valid_indices:
                    self.image_embeddings = self.image_embeddings[valid_indices]
                else:
                    self.image_embeddings = None
            
            if self.image_embeddings is not None:
                torch.save(self.image_embeddings, str(self.embeddings_file))
            elif self.embeddings_file.exists():
                 os.remove(self.embeddings_file)

            with open(self.index_file, "w") as f:
                json.dump(self.image_paths, f)
            with open(self.metadata_file, "w") as f:
                json.dump(self.image_metadata, f)
                
        return removed_count

    def mark_photo_deleted(self, file_path):
        if file_path in self.image_metadata:
            self.image_metadata[file_path]['deleted'] = True
        with open(self.metadata_file, "w") as f:
            json.dump(self.image_metadata, f)
        if file_path in self.indexed_set:
            self.indexed_set.remove(file_path)

    def search(self, query, top_k=50):
        if self.image_embeddings is None: return []
        
        search_text = f"a photo of a {query}" if len(query.split()) < 4 else query
            
        inputs = self.processor(text=[search_text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            raw_output = self.model.get_text_features(**inputs)
            
            # FIX 3: Transformers 5.x compatibility
            if hasattr(raw_output, 'pooler_output'):
                text_features = raw_output.pooler_output
            elif isinstance(raw_output, tuple):
                text_features = raw_output[0]
            else:
                text_features = raw_output

            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        image_embeddings_gpu = self.image_embeddings.to(self.device)
        if self.device == "cuda":
            image_embeddings_gpu = image_embeddings_gpu.half()

        similarities = (image_embeddings_gpu @ text_features.T).squeeze(1)
        
        values, indices = torch.topk(similarities, top_k * 10)
        
        results = []
        scores = values.cpu().numpy()
        idxs = indices.cpu().numpy()
        
        if len(scores) == 0: return []

        best_score = scores[0]
        hard_floor = 0.19
        dynamic_floor = best_score * 0.55
        effective_floor = max(hard_floor, dynamic_floor)
        
        for i in range(len(scores)):
            score = scores[i]
            idx = idxs[i]
            
            if score < effective_floor: continue
            
            file_path = self.image_paths[idx]
            meta = self.image_metadata.get(file_path, {})
            
            if meta.get('deleted', False): continue
            if self._is_garbage(meta): continue

            results.append({"file": file_path, "score": float(score)})
            if len(results) >= top_k: break
        
        return results

    def _is_garbage(self, meta):
        std_dev = meta.get("std_dev", 100.0)
        edge_score = meta.get("edge_score")
        if edge_score is None: return std_dev < 25.0 
        is_boring = (std_dev < 25.0 and edge_score < 15.0)
        is_blank = std_dev < 10.0
        return is_blank or is_boring

    def get_garbage_photos(self):
        results = []
        for path, meta in self.image_metadata.items():
            if not os.path.exists(path): continue
            if meta.get('deleted', False): continue
            if self._is_garbage(meta):
                results.append({"file": path, "score": 0.0, "is_garbage": True})
        return results

# --- Threading Workers ---

class IndexWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, searcher, directory):
        super().__init__()
        self.searcher = searcher
        self.directory = directory

    def run(self):
        self.searcher.index_photos(self.directory, self.progress.emit)
        self.finished.emit(0)

class SearchWorker(QThread):
    result = pyqtSignal(list)

    def __init__(self, searcher, query):
        super().__init__()
        self.searcher = searcher
        self.query = query

    def run(self):
        hits = self.searcher.search(self.query)
        self.result.emit(hits)

class GarbageWorker(QThread):
    result = pyqtSignal(list)

    def __init__(self, searcher):
        super().__init__()
        self.searcher = searcher

    def run(self):
        hits = self.searcher.get_garbage_photos()
        self.result.emit(hits)

class CleanWorker(QThread):
    finished = pyqtSignal(int)

    def __init__(self, searcher):
        super().__init__()
        self.searcher = searcher

    def run(self):
        removed = self.searcher.clean_database()
        self.finished.emit(removed)

class ThumbnailWorker(QThread):
    thumbnail_loaded = pyqtSignal(str, QImage)

    def __init__(self, files):
        super().__init__()
        self.files = files

    def run(self):
        for path in self.files:
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    img.thumbnail((200, 200))
                    
                    # Create QImage directly from PIL data
                    data = img.tobytes("raw", "RGB")
                    q_img = QImage(data, img.width, img.height, 3 * img.width, QImage.Format.Format_RGB888)
                    
                    # QImage created from raw bytes does not copy the data.
                    # We must copy it, otherwise 'data' goes out of scope and the image becomes garbage.
                    self.thumbnail_loaded.emit(path, q_img.copy())
            except Exception:
                pass

# --- GUI Application ---

class PhotoOrganizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CLIP Photo Organizer")
        self.resize(1200, 800)
        
        self.searcher = PhotoSearch()
        self.current_results = []
        self.thumb_worker = None
        
        self.setup_ui()
        self.setup_menu()
        
        if self.searcher.load_index():
            self.statusBar().showMessage(f"Ready. {len(self.searcher.indexed_set)} images indexed.")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type a description...")
        self.search_input.returnPressed.connect(self.start_search)
        
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.start_search)
        
        garbage_btn = QPushButton("Find Bad Photos")
        garbage_btn.clicked.connect(self.find_garbage)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_btn)
        search_layout.addWidget(garbage_btn)
        layout.addLayout(search_layout)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(200, 200))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        
        self.list_widget.itemDoubleClicked.connect(self.open_image)
        self.list_widget.itemClicked.connect(self.show_path_in_status)
        
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        
        layout.addWidget(self.list_widget)
        self.setStatusBar(QStatusBar())

    def setup_menu(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        
        index_action = QAction("Index Folder", self)
        index_action.triggered.connect(self.select_folder_to_index)
        toolbar.addAction(index_action)

        clean_action = QAction("Clean Database", self)
        clean_action.setToolTip("Remove entries for files that have been moved or deleted outside this app")
        clean_action.triggered.connect(self.clean_database)
        toolbar.addAction(clean_action)

    def clean_database(self):
        reply = QMessageBox.question(self, 'Clean Database', 
                                     'This will scan your database and remove entries for files that no longer exist on disk.\n\nContinue?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                     QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Cleaning database...")
            self.clean_worker = CleanWorker(self.searcher)
            self.clean_worker.finished.connect(self.clean_finished)
            self.clean_worker.start()

    def clean_finished(self, removed_count):
        self.statusBar().showMessage(f"Clean complete. Removed {removed_count} missing files.")
        self.searcher.load_index()

    def show_context_menu(self, position):
        item = self.list_widget.itemAt(position)
        if not item:
            return

        path = item.data(Qt.ItemDataRole.UserRole)
        
        menu = QMenu()
        
        open_folder_action = menu.addAction("Open in File Browser")
        open_folder_action.triggered.connect(lambda: self.open_in_explorer(path))
        
        delete_action = menu.addAction("Move to Trash")
        delete_action.triggered.connect(lambda: self.delete_photo_action(path, item))
        
        menu.exec(self.list_widget.viewport().mapToGlobal(position))

    def open_in_explorer(self, path):
        # Linux specific file browser opening
        try:
            if os.path.exists("/usr/bin/dolphin"):
                subprocess.Popen(['dolphin', '--select', path])
            elif os.path.exists("/usr/bin/nautilus"):
                subprocess.Popen(['nautilus', '--select', path])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
        except Exception:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))

    def delete_photo_action(self, path, item):
        reply = QMessageBox.question(self, 'Move to Trash', 
                                     f'Move this file to Trash?\n{path}',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                     QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            if QFile.moveToTrash(path):
                self.searcher.mark_photo_deleted(path)
                row = self.list_widget.row(item)
                self.list_widget.takeItem(row)
                self.statusBar().showMessage(f"Moved to Trash: {os.path.basename(path)}")
            else:
                QMessageBox.warning(self, "Error", "Could not move file to trash.")

    def select_folder_to_index(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if folder:
            self.statusBar().showMessage(f"Indexing {folder}...")
            self.worker = IndexWorker(self.searcher, folder)
            self.worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
            self.worker.finished.connect(self.indexing_finished)
            self.worker.start()

    def indexing_finished(self, count):
        self.statusBar().showMessage(f"Indexing complete. Total images: {len(self.searcher.indexed_set)}")

    def start_search(self):
        query = self.search_input.text()
        if not query:
            return
        
        self.list_widget.clear()
        self.statusBar().showMessage("Searching...")
        
        self.search_worker = SearchWorker(self.searcher, query)
        self.search_worker.result.connect(self.display_results)
        self.search_worker.start()

    def find_garbage(self):
        self.list_widget.clear()
        self.statusBar().showMessage("Scanning metadata for bad photos...")
        
        self.garbage_worker = GarbageWorker(self.searcher)
        self.garbage_worker.result.connect(self.display_results)
        self.garbage_worker.start()

    def display_results(self, hits):
        self.list_widget.clear()
        self.current_results = hits
        
        if not hits:
            self.statusBar().showMessage("No results found.")
            return

        self.statusBar().showMessage(f"Found {len(hits)} results. Loading thumbnails...")
        
        paths_to_load = []
        
        for hit in hits:
            path = hit['file']
            score = hit.get('score', 0.0)
            
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, path)
            
            # FIX: Set size hint immediately so items don't stack on top of each other
            item.setSizeHint(QSize(200, 200))
            
            tooltip = f"{os.path.basename(path)}\nScore: {score:.3f}"
            if hit.get('is_garbage'):
                tooltip += "\n[LOW QUALITY]"
            item.setToolTip(tooltip)
            
            self.list_widget.addItem(item)
            paths_to_load.append(path)

        # Start background thumbnail loading
        if self.thumb_worker and self.thumb_worker.isRunning():
            self.thumb_worker.terminate() # Stop previous loading if any
            
        self.thumb_worker = ThumbnailWorker(paths_to_load)
        self.thumb_worker.thumbnail_loaded.connect(self.update_thumbnail)
        self.thumb_worker.finished.connect(lambda: self.statusBar().showMessage("Ready"))
        self.thumb_worker.start()

    def update_thumbnail(self, path, q_img):
        # Convert QImage to QPixmap in the main thread
        pixmap = QPixmap.fromImage(q_img)
        
        # Fallback: linear scan (because findItems matches text, not data)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setIcon(QIcon(pixmap))
                return

    def show_path_in_status(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.statusBar().showMessage(path)

    def open_image(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if not os.path.exists(path):
            return

        if sys.platform == 'linux':
            try:
                subprocess.Popen(
                    ['kioclient', 'exec', path],
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                return
            except FileNotFoundError:
                pass
        
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

# --- Main Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Photo Search")
    parser.add_argument("--index", type=str, help="Directory to index")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--reindex", action="store_true", help="Clear index")
    parser.add_argument("--top", type=int, default=5, help="Results count")
    parser.add_argument("--find-garbage", action="store_true", help="Find low quality images")
    
    args, unknown = parser.parse_known_args()

    cache_path = Path.home() / ".cache" / "photo_search"
    index_p = cache_path / "photo_index.json"
    embed_p = cache_path / "photo_embeddings.pt"

    if args.index or args.search or args.find_garbage:
        searcher = PhotoSearch()
        
        if args.reindex:
            if index_p.exists(): index_p.unlink()
            if embed_p.exists(): embed_p.unlink()
            meta_p = cache_path / "photo_metadata.json"
            if meta_p.exists(): meta_p.unlink()
            print("[CLI] Cleared existing index.")
        
        searcher.load_index()

        if args.index:
            searcher.index_photos(args.index)
        elif args.find_garbage:
            hits = searcher.get_garbage_photos()
            print(f"\nFound {len(hits)} low quality images:")
            print("-" * 50)
            for h in hits:
                print(f"[Garbage] {h['file']}")
        elif args.search:
            hits = searcher.search(args.search, top_k=args.top)
            print(f"\nTop {args.top} results for '{args.search}':")
            print("-" * 50)
            for h in hits:
                print(f"[Score: {h['score']:.3f}] {h['file']}")

    else:
        app = QApplication(sys.argv)
        
        if args.reindex:
            if index_p.exists(): index_p.unlink()
            if embed_p.exists(): embed_p.unlink()
            meta_p = cache_path / "photo_metadata.json"
            if meta_p.exists(): meta_p.unlink()
            print("[GUI] Index cleared.")

        window = PhotoOrganizerWindow()
        window.show()
        sys.exit(app.exec())
