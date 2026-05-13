"""
Z+F FlexScanMask
Anonymization tool for Z+F fisheye images (cam_0 / cam_1).

Pipeline per image:
  1. Rotate 180 degrees
  2. SAM3 text-prompt search: person, license plate (on rotated image)
  3. Rotate image and SAM masks back to original orientation
  4. Load camera-specific hard mask (cam0_scanner_mask.png or cam1_scanner_mask.png)
  5. Combine hard mask + SAM masks
  6. Blur all combined regions on the original-orientation image
  7. Save
"""

import os
import sys
import queue
import threading
import time
from pathlib import Path
from tkinter import filedialog

# Ensure local venv is on sys.path when launched without activating it (e.g. from IDE or double-click)
_venv_site = Path(__file__).parent / "venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".jfif"}
SAM3_CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints" / "sam3"
HARD_MASK_CAM0_PATH  = Path(__file__).parent / "cam0_scanner_mask.png"
HARD_MASK_CAM1_PATH  = Path(__file__).parent / "cam1_scanner_mask.png"
AUTO_OUTPUT_NAME     = "FlexScanMask_Output"
APP_VERSION         = "1.0"

SAM3_CONFIDENCE    = 0.15
SAM3_MIN_MASK_PX   = 500
SAM3_PROMPTS       = ["person", "license plate"]
BLUR_KERNEL_BASE   = 101
BLUR_PASSES        = 3
QUANTIZE_STEP      = 8
PADDING_FRACTION   = 0.04

APPEARANCE = {
    "fg_color_primary":   "#1a1a2e",
    "fg_color_secondary": "#16213e",
    "accent":             "#0f3460",
    "highlight":          "#e94560",
    "text_primary":       "#eaeaea",
    "text_secondary":     "#a0a0b0",
    "success":            "#4ade80",
    "warning":            "#fbbf24",
    "error":              "#f87171",
}


# ──────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────

def pad_mask(mask: np.ndarray, pad: float = PADDING_FRACTION) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask
    h_ext = ys.max() - ys.min()
    w_ext = xs.max() - xs.min()
    k = max(3, int(min(h_ext, w_ext) * pad))
    k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def blur_region(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return image
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    roi = image[y_min:y_max + 1, x_min:x_max + 1].copy()
    if roi.size == 0:
        return image
    roi_h, roi_w = roi.shape[:2]
    k = max(BLUR_KERNEL_BASE, int(min(roi_h, roi_w) * 0.4))
    k = k if k % 2 == 1 else k + 1
    k = min(k, 301)
    blurred = roi
    for _ in range(BLUR_PASSES):
        blurred = cv2.GaussianBlur(blurred, (k, k), sigmaX=0, sigmaY=0)
    blurred = (blurred // QUANTIZE_STEP * QUANTIZE_STEP).astype(np.uint8)
    local_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    image[y_min:y_max + 1, x_min:x_max + 1][local_mask > 0] = blurred[local_mask > 0]
    return image


def load_hard_mask(path: Path, target_h: int, target_w: int) -> np.ndarray | None:
    """
    Load PNG mask (black=blur, white=keep).
    Returns binary uint8 mask (255=blur) resized to target dimensions.
    Returns None if file does not exist.
    """
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (target_h, target_w):
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (img < 128).astype(np.uint8) * 255


def rotate_mask_180(mask: np.ndarray) -> np.ndarray:
    """Rotate a binary mask 180 degrees."""
    return cv2.rotate(mask, cv2.ROTATE_180)


# ──────────────────────────────────────────────────────────────
# Processor
# ──────────────────────────────────────────────────────────────

class Processor:
    def __init__(self, msg_queue: queue.Queue):
        self.msg_queue   = msg_queue
        self.sam3_proc   = None
        self.sam3_loaded = False
        self._stop_event = threading.Event()

    def _emit(self, kind: str, **kwargs):
        self.msg_queue.put({"kind": kind, **kwargs})

    def _log(self, text: str):
        self._emit("log", text=text)

    def _progress(self, value: float, current: int = 0, total: int = 0):
        self._emit("progress", value=value, current=current, total=total)

    def _done(self, success: bool):
        self._emit("done", success=success)

    def stop(self):
        self._stop_event.set()

    # ── Load SAM3 ────────────────────────────────────────────
    def _load_sam3(self) -> bool:
        if self.sam3_loaded:
            return True
        try:
            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import sam3.perflib.fused as _fused_mod
            import sam3.model.vitdet as _vitdet_mod

            def _addmm_act_f32(activation, linear, mat1):
                x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
                return activation()(x)

            _fused_mod.addmm_act = _addmm_act_f32
            _vitdet_mod.addmm_act = _addmm_act_f32

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available():
                self._log("[WARN] CUDA not available - SAM3 runs on CPU (slow).")

            import sam3 as _sam3_pkg
            bpe_path = Path(_sam3_pkg.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            if not bpe_path.exists():
                bpe_path = Path(_sam3_pkg.__file__).parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

            self._log("[LOAD] Loading SAM3 ...")
            ckpt_dir  = SAM3_CHECKPOINT_DIR
            ckpt_path = None
            if ckpt_dir.exists():
                for candidate in ("sam3.pt", "model.safetensors"):
                    if (ckpt_dir / candidate).exists():
                        ckpt_path = str(ckpt_dir / candidate)
                        break

            if ckpt_path is None:
                self._log("  Checkpoint not local - loading from HuggingFace ...")

            model = build_sam3_image_model(
                checkpoint_path=ckpt_path,
                bpe_path=str(bpe_path) if bpe_path.exists() else None,
                device=device,
                load_from_HF=(ckpt_path is None),
            )
            model = model.float()
            self.sam3_proc = Sam3Processor(
                model,
                confidence_threshold=SAM3_CONFIDENCE,
                device=device,
            )
            self.sam3_loaded = True
            self._log("[OK] SAM3 loaded.")
            return True

        except ImportError as exc:
            self._log(f"[ERROR] Import failed: {exc}")
            self._log(f"        sys.path[0]: {sys.path[0] if sys.path else '?'}")
            venv_site = Path(__file__).parent / "venv" / "Lib" / "site-packages"
            self._log(f"        venv exists: {venv_site.exists()} ({venv_site})")
            self._log("        -> Run start.bat to fix.")
            return False
        except Exception as exc:
            self._log(f"[ERROR] SAM3: {exc}")
            return False

    # ── SAM3 segmentation on one image ───────────────────────
    def _run_sam3(self, cv_image: np.ndarray) -> list[np.ndarray]:
        import torch
        h, w = cv_image.shape[:2]
        masks: list[np.ndarray] = []

        rgb     = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        try:
            state = self.sam3_proc.set_image(pil_img)
        except Exception as exc:
            self._log(f"  [ERROR] SAM3 set_image: {exc}")
            return masks

        covered = np.zeros((h, w), dtype=np.uint8)

        for prompt in SAM3_PROMPTS:
            try:
                self.sam3_proc.reset_all_prompts(state)
                result = self.sam3_proc.set_text_prompt(prompt, state)
            except Exception as exc:
                self._log(f"  SAM3 prompt '{prompt}' error: {exc}")
                continue

            raw    = result.get("masks")
            scores = result.get("scores")

            if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
                self._log(f"  '{prompt}': 0 detections")
                continue

            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            found = 0
            for i, m in enumerate(raw):
                if m.ndim == 4:   m = m[0, 0]
                elif m.ndim == 3: m = m[0]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.float32), (w, h),
                                   interpolation=cv2.INTER_LINEAR)
                binary = (m > 0.5).astype(np.uint8) * 255
                px = int(np.count_nonzero(binary))
                if px < SAM3_MIN_MASK_PX:
                    continue

                inter = int(np.count_nonzero(np.bitwise_and(binary, covered)))
                uni   = int(np.count_nonzero(np.bitwise_or(binary, covered)))
                iou   = inter / uni if uni > 0 else 0.0
                if iou > 0.8:
                    continue

                score_str = f" score={scores[i]:.2f}" if scores is not None and i < len(scores) else ""
                self._log(f"  '{prompt}': mask {px:,} px{score_str}")
                padded = pad_mask(binary)
                masks.append(padded)
                covered = np.maximum(covered, padded)
                found += 1

            if found == 0:
                self._log(f"  '{prompt}': 0 detections")

        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return masks

    # ── Process one image ────────────────────────────────────
    def _process_image(
        self,
        img_path: Path,
        output_dir: Path,
    ) -> bool:
        t0 = time.perf_counter()
        self._log(f">> {img_path.name}")

        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            self._log(f"  [ERROR] Could not load image.")
            return False

        h, w = cv_img.shape[:2]

        # Step 1: Rotate 180 for SAM3 (person upright, scanner at bottom)
        cv_rotated = cv2.rotate(cv_img, cv2.ROTATE_180)

        # Step 2: SAM3 on rotated image
        self._log("  Running SAM3 ...")
        sam_masks_rotated = self._run_sam3(cv_rotated)

        # Step 3: Rotate SAM masks back to original orientation
        sam_masks = [rotate_mask_180(m) for m in sam_masks_rotated]
        self._log(f"  SAM3: {len(sam_masks)} mask(s)")

        # Step 4: Load camera-specific hard mask (original orientation)
        name_lower = img_path.name.lower()
        if name_lower.startswith("cam_0"):
            mask_path = HARD_MASK_CAM0_PATH
        elif name_lower.startswith("cam_1"):
            mask_path = HARD_MASK_CAM1_PATH
        else:
            mask_path = None

        all_masks: list[np.ndarray] = list(sam_masks)

        if mask_path is not None:
            hard = load_hard_mask(mask_path, h, w)
            if hard is not None:
                all_masks.append(hard)
                self._log(f"  Hard mask: {mask_path.name} ({int(np.count_nonzero(hard)):,} px)")
            else:
                self._log(f"  [WARN] Hard mask not found: {mask_path.name}")

        if not all_masks:
            self._log("  No regions to blur - saving original.")
        else:
            # Step 5: Combine all masks and blur on original-orientation image
            combined = np.zeros((h, w), dtype=np.uint8)
            for m in all_masks:
                combined = np.maximum(combined, m)
            cv_img = blur_region(cv_img, combined)
            self._log(f"  Blurred {int(np.count_nonzero(combined)):,} px total")

        # Step 6: Save
        ext = img_path.suffix.lower()
        out_name = img_path.stem + (ext if ext != ".jfif" else ".jpg")
        out_path = output_dir / out_name

        encode_params = []
        if ext in {".jpg", ".jpeg", ".jfif"}:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == ".png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

        cv2.imwrite(str(out_path), cv_img, encode_params)

        dt = time.perf_counter() - t0
        self._log(f"  Saved: {out_name} ({dt:.1f}s)")
        return True

    # ── Main run loop ────────────────────────────────────────
    def run(self, input_paths: list[Path], output_dir: Path):
        if not self._load_sam3():
            self._done(False)
            return

        # Log mask status once at start
        for path, label in [(HARD_MASK_CAM0_PATH, "cam_0"), (HARD_MASK_CAM1_PATH, "cam_1")]:
            if path.exists():
                self._log(f"[OK] Hard mask found: {path.name}")
            else:
                self._log(f"[WARN] Hard mask missing: {path.name}")

        total  = len(input_paths)
        failed = 0

        for idx, img_path in enumerate(input_paths):
            if self._stop_event.is_set():
                self._log("Processing cancelled.")
                break

            self._progress(idx / total, current=idx, total=total)
            ok = self._process_image(img_path, output_dir)
            if not ok:
                failed += 1

        self._progress(1.0, current=total, total=total)
        if failed > 0:
            self._log(f"Done. {failed}/{total} file(s) failed.")
        else:
            self._log(f"Done. All {total} file(s) processed.")
        self._done(failed == 0)


# ──────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class FlexScanMaskApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"Z+F FlexScanMask v{APP_VERSION}")
        self.geometry("860x680")
        self.resizable(True, True)
        self.configure(fg_color=APPEARANCE["fg_color_primary"])
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._input_paths:   list[Path] = []
        self._output_dir:    Path | None = None
        self._processor:     Processor | None = None
        self._proc_thread:   threading.Thread | None = None
        self._msg_queue:     queue.Queue = queue.Queue()

        self._build_ui()
        self.after(100, self._poll_queue)

    # ── UI construction ──────────────────────────────────────
    def _build_ui(self):
        A = APPEARANCE

        # Title
        title_frame = ctk.CTkFrame(self, fg_color=A["accent"], corner_radius=0)
        title_frame.pack(fill="x", pady=(0, 0))
        ctk.CTkLabel(
            title_frame,
            text=f"Z+F FlexScanMask  v{APP_VERSION}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=A["text_primary"],
        ).pack(pady=12)

        # Main container
        main = ctk.CTkFrame(self, fg_color=A["fg_color_secondary"], corner_radius=10)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # Input section
        ctk.CTkLabel(main, text="Input", font=ctk.CTkFont(weight="bold"),
                     text_color=A["text_secondary"]).pack(anchor="w", padx=16, pady=(12, 0))

        input_row = ctk.CTkFrame(main, fg_color="transparent")
        input_row.pack(fill="x", padx=16, pady=4)

        self.lbl_input = ctk.CTkLabel(
            input_row, text="No file or folder selected",
            text_color=A["text_secondary"], anchor="w",
        )
        self.lbl_input.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            input_row, text="Single image", width=120,
            fg_color=A["accent"], hover_color=A["highlight"],
            command=self._pick_file,
        ).pack(side="right", padx=(4, 0))

        ctk.CTkButton(
            input_row, text="Folder", width=90,
            fg_color=A["accent"], hover_color=A["highlight"],
            command=self._pick_folder,
        ).pack(side="right", padx=(4, 0))

        # Output section
        ctk.CTkLabel(main, text="Output", font=ctk.CTkFont(weight="bold"),
                     text_color=A["text_secondary"]).pack(anchor="w", padx=16, pady=(10, 0))

        out_row = ctk.CTkFrame(main, fg_color="transparent")
        out_row.pack(fill="x", padx=16, pady=4)

        self.lbl_output = ctk.CTkLabel(
            out_row, text="Auto (subfolder next to input)",
            text_color=A["text_secondary"], anchor="w",
        )
        self.lbl_output.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            out_row, text="Change", width=90,
            fg_color=A["accent"], hover_color=A["highlight"],
            command=self._pick_output,
        ).pack(side="right")

        # Hard mask status
        for mask_path, cam_label in [(HARD_MASK_CAM0_PATH, "cam_0"), (HARD_MASK_CAM1_PATH, "cam_1")]:
            color = A["success"] if mask_path.exists() else A["warning"]
            text  = f"Mask {cam_label}: {mask_path.name} found" if mask_path.exists() \
                    else f"Mask {cam_label}: {mask_path.name} NOT found"
            ctk.CTkLabel(main, text=text, text_color=color,
                         font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(3, 0))

        # Progress
        prog_frame = ctk.CTkFrame(main, fg_color="transparent")
        prog_frame.pack(fill="x", padx=16, pady=(12, 4))

        self.progress_bar = ctk.CTkProgressBar(prog_frame, height=14)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x")

        status_row = ctk.CTkFrame(main, fg_color="transparent")
        status_row.pack(fill="x", padx=16)
        self.lbl_status  = ctk.CTkLabel(status_row, text="Ready", text_color=A["text_secondary"])
        self.lbl_status.pack(side="left")
        self.lbl_counter = ctk.CTkLabel(status_row, text="", text_color=A["text_secondary"])
        self.lbl_counter.pack(side="right")

        # Buttons
        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(pady=10)

        self.btn_start = ctk.CTkButton(
            btn_row, text="Start", width=140, height=40,
            fg_color=A["highlight"], hover_color="#a02840",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start,
        )
        self.btn_start.pack(side="left", padx=6)

        self.btn_stop = ctk.CTkButton(
            btn_row, text="Stop", width=100, height=40,
            fg_color=A["accent"], hover_color=A["highlight"],
            state="disabled",
            command=self._stop,
        )
        self.btn_stop.pack(side="left", padx=6)

        # Log
        ctk.CTkLabel(main, text="Log", font=ctk.CTkFont(weight="bold"),
                     text_color=A["text_secondary"]).pack(anchor="w", padx=16, pady=(8, 0))
        self.log_box = ctk.CTkTextbox(
            main, height=260,
            fg_color=A["fg_color_primary"],
            text_color=A["text_primary"],
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(4, 12))
        self.log_box.configure(state="disabled")

    # ── File/folder pickers ──────────────────────────────────
    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp *.jfif"), ("All", "*.*")],
        )
        if path:
            p = Path(path)
            self._input_paths = [p]
            self._output_dir  = None
            self.lbl_input.configure(text=str(p), text_color=APPEARANCE["success"])
            self.lbl_output.configure(text="Auto", text_color=APPEARANCE["text_secondary"])

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if folder:
            p = Path(folder)
            imgs = sorted([
                f for f in p.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ])
            if not imgs:
                self.lbl_input.configure(text="No images found in folder", text_color=APPEARANCE["error"])
                return
            self._input_paths = imgs
            self._output_dir  = None
            self.lbl_input.configure(
                text=f"{p.name}/ ({len(imgs)} images)",
                text_color=APPEARANCE["success"],
            )
            self.lbl_output.configure(
                text=f"Auto: {p.name}/{AUTO_OUTPUT_NAME}/",
                text_color=APPEARANCE["text_secondary"],
            )

    def _pick_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self._output_dir = Path(folder)
            self.lbl_output.configure(text=str(self._output_dir), text_color=APPEARANCE["success"])

    # ── Start / stop ─────────────────────────────────────────
    def _start(self):
        if not self._input_paths:
            self._log_line("No input selected.")
            return

        if self._output_dir is None:
            parent = self._input_paths[0].parent
            self._output_dir = parent / AUTO_OUTPUT_NAME
            self.lbl_output.configure(
                text=str(self._output_dir),
                text_color=APPEARANCE["warning"],
            )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.progress_bar.set(0)
        self.lbl_status.configure(text="Processing ...", text_color=APPEARANCE["warning"])
        self.lbl_counter.configure(text="")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        self._processor  = Processor(self._msg_queue)
        self._proc_thread = threading.Thread(
            target=self._processor.run,
            args=(self._input_paths, self._output_dir),
            daemon=True,
        )
        self._proc_thread.start()

    def _stop(self):
        if self._processor:
            self._processor.stop()
        self.btn_stop.configure(state="disabled")
        self.lbl_status.configure(text="Cancelling ...", text_color=APPEARANCE["error"])

    # ── Queue polling ────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                kind = msg["kind"]
                if kind == "log":
                    self._log_line(msg["text"])
                elif kind == "progress":
                    self.progress_bar.set(msg["value"])
                    c, t = msg.get("current", 0), msg.get("total", 0)
                    if t > 0:
                        self.lbl_counter.configure(text=f"{c} / {t}")
                elif kind == "done":
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    if msg["success"]:
                        self.lbl_status.configure(text="Completed", text_color=APPEARANCE["success"])
                    else:
                        self.lbl_status.configure(text="Finished with errors", text_color=APPEARANCE["error"])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _log_line(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ── Shutdown ─────────────────────────────────────────────
    def _on_close(self):
        if self._processor:
            self._processor.stop()
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.destroy()
        os._exit(0)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FlexScanMaskApp()
    app.mainloop()