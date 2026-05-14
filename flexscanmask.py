"""
Z+F FlexScanMask
Anonymization tool for Z+F fisheye images (cam_0 / cam_1).

Pipeline per image:
  1. Rotate 180 degrees
  2. SAM3 text-prompt search: person, license plate (on rotated image)
  3. Rotate image and SAM masks back to original orientation
  4. Load camera-specific hard mask (mask/cam_0 or mask/cam_1)
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
MASK_DIR             = Path(__file__).parent / "mask"

# Suchregion-Masken: schwarz = hier soll SAM gezielt nach Scanner suchen
SEARCH_REGION_CAM0  = MASK_DIR / "mask_cam_0.png"
SEARCH_REGION_CAM1  = MASK_DIR / "mask_cam_1.png"

# Feste Fallback-Masken: werden am Ende mit SAM-Ergebnissen verschmolzen
FALLBACK_MASK_CAM0  = MASK_DIR / "cam0_scanner_mask.png"
FALLBACK_MASK_CAM1  = MASK_DIR / "cam1_scanner_mask.png"

AUTO_OUTPUT_NAME     = "FlexScanMask_Output"
APP_VERSION         = "1.1"

SAM3_CONFIDENCE    = 0.20
SAM3_MIN_MASK_PX   = 500

# Prompts pro Kamera
# global: auf dem ganzen Bild suchen (nach 180-Grad-Rotation)
# scanner: nur im Suchbereich der Suchregion-Maske suchen (auf Original-Orientierung)
SAM3_CAMERA_PROMPTS = {
    "cam_0": {
        "global": ["person", "license plate"],
        "scanner": ["blue device"],
    },
    "cam_1": {
        "global": ["person", "license plate"],
        "scanner": [],
    },
}
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


def dilate_mask_fixed(mask: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    kernel_size = max(3, kernel_size)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def fill_region_black(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image[mask > 0] = 0
    return image


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
    def __init__(self, msg_queue: queue.Queue, mode: str = "blur"):
        self.msg_queue   = msg_queue
        self.mode        = mode  # "blur" or "black"
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
    def _run_sam3(
        self,
        cv_image: np.ndarray,
        prompts: list[str],
        max_masks_per_prompt: int | None = None,
    ) -> list[np.ndarray]:
        import torch
        h, w = cv_image.shape[:2]
        masks: list[np.ndarray] = []

        rgb     = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        t_total = time.perf_counter()
        try:
            t_set_image = time.perf_counter()
            state = self.sam3_proc.set_image(pil_img)
            self._log(f"  SAM3 set_image: {time.perf_counter() - t_set_image:.1f}s")
        except Exception as exc:
            self._log(f"  [ERROR] SAM3 set_image: {exc}")
            return masks

        covered = np.zeros((h, w), dtype=np.uint8)

        for prompt in prompts:
            t_prompt = time.perf_counter()
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
            indices = range(len(raw))
            if scores is not None:
                indices = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)

            for i in indices:
                m = raw[i]
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
                if max_masks_per_prompt is not None and found >= max_masks_per_prompt:
                    break

            if found == 0:
                self._log(f"  '{prompt}': 0 detections")
            self._log(f"  '{prompt}' time: {time.perf_counter() - t_prompt:.1f}s")

        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._log(f"  SAM3 total: {time.perf_counter() - t_total:.1f}s")
        return masks

    # ── SAM3 segmentation restricted to a mask region ───────
    def _run_sam3_masked(
        self,
        cv_image: np.ndarray,
        prompts: list[str],
        hard_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """Run SAM3 only on the crop covered by hard_mask, then map masks back."""
        if hard_mask is None or not np.any(hard_mask):
            return []

        mask_region = dilate_mask_fixed(hard_mask, kernel_size=31)
        ys, xs = np.where(mask_region > 0)
        if len(ys) == 0:
            return []

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        crop = cv_image[y_min:y_max + 1, x_min:x_max + 1]
        self._log(f"  SAM3 scanner crop: {crop.shape[1]}x{crop.shape[0]} px")
        sam_masks = self._run_sam3(crop, prompts, max_masks_per_prompt=1)
        if not sam_masks:
            return []

        restricted = []
        for m in sam_masks:
            full_mask = np.zeros(hard_mask.shape, dtype=np.uint8)
            full_mask[y_min:y_max + 1, x_min:x_max + 1] = m
            clipped = cv2.bitwise_and(full_mask, mask_region)
            if np.count_nonzero(clipped) >= SAM3_MIN_MASK_PX:
                restricted.append(clipped)
            elif np.count_nonzero(full_mask) >= SAM3_MIN_MASK_PX:
                overlap = np.count_nonzero(np.bitwise_and(full_mask, hard_mask))
                if overlap > SAM3_MIN_MASK_PX // 2:
                    restricted.append(clipped)

        return restricted

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

        # Detect camera type
        name_lower = img_path.name.lower()
        cam_type = None
        for ct in ("cam_0", "cam_1"):
            if name_lower.startswith(ct):
                cam_type = ct
                break

        # Get camera-specific prompts
        cam_config = SAM3_CAMERA_PROMPTS.get(cam_type, {
            "global": ["person", "license plate"],
            "scanner": [],
        })

        all_masks: list[np.ndarray] = []

        # Step 1: Rotate 180 for SAM3 (person upright, scanner at bottom)
        cv_rotated = cv2.rotate(cv_img, cv2.ROTATE_180)

        # Step 2: SAM3 global prompts on rotated image
        if cam_config["global"]:
            t_global = time.perf_counter()
            self._log("  Running SAM3 (global) ...")
            sam_masks_rotated = self._run_sam3(cv_rotated, cam_config["global"])
            sam_masks = [rotate_mask_180(m) for m in sam_masks_rotated]
            self._log(f"  SAM3 global: {len(sam_masks)} mask(s) in {time.perf_counter() - t_global:.1f}s")
            all_masks.extend(sam_masks)

        # Step 3: Load search region mask (black = where SAM should look for scanner)
        search_region = None
        if cam_type == "cam_0":
            search_region = load_hard_mask(SEARCH_REGION_CAM0, h, w)
        elif cam_type == "cam_1":
            search_region = load_hard_mask(SEARCH_REGION_CAM1, h, w)

        # Step 4: SAM3 scanner search on original image (restricted to search region)
        if search_region is not None and cam_config["scanner"]:
            t_scanner = time.perf_counter()
            self._log(f"  Running SAM3 (scanner: {', '.join(cam_config['scanner'])}) ...")
            scanner_masks = self._run_sam3_masked(cv_img, cam_config["scanner"], search_region)
            self._log(f"  SAM3 scanner: {len(scanner_masks)} mask(s) in {time.perf_counter() - t_scanner:.1f}s")
            all_masks.extend(scanner_masks)

        # Step 5: Load fallback mask and merge
        fallback = None
        if cam_type == "cam_0":
            fallback = load_hard_mask(FALLBACK_MASK_CAM0, h, w)
        elif cam_type == "cam_1":
            fallback = load_hard_mask(FALLBACK_MASK_CAM1, h, w)
        if fallback is not None:
            all_masks.append(fallback)
            self._log(f"  Fallback mask: {cam_type} ({int(np.count_nonzero(fallback)):,} px)")

        if not all_masks:
            self._log("  No regions to anonymize - saving original.")
        else:
            # Step 6: Combine all masks and anonymize on original-orientation image
            combined = np.zeros((h, w), dtype=np.uint8)
            for m in all_masks:
                combined = np.maximum(combined, m)
            px = int(np.count_nonzero(combined))
            if self.mode == "black":
                t_anonymize = time.perf_counter()
                cv_img = fill_region_black(cv_img, combined)
                self._log(f"  Filled black {px:,} px total in {time.perf_counter() - t_anonymize:.1f}s")
            else:
                t_anonymize = time.perf_counter()
                cv_img = blur_region(cv_img, combined)
                self._log(f"  Blurred {px:,} px total in {time.perf_counter() - t_anonymize:.1f}s")

        # Step 7: Save
        ext = img_path.suffix.lower()
        out_name = img_path.stem + (ext if ext != ".jfif" else ".jpg")
        out_path = output_dir / out_name

        encode_params = []
        if ext in {".jpg", ".jpeg", ".jfif"}:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == ".png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

        t_save = time.perf_counter()
        cv2.imwrite(str(out_path), cv_img, encode_params)
        self._log(f"  Save time: {time.perf_counter() - t_save:.1f}s")

        dt = time.perf_counter() - t0
        self._log(f"  Saved: {out_name} ({dt:.1f}s)")
        return True

    # ── Main run loop ────────────────────────────────────────
    def run(self, input_paths: list[Path], output_dir: Path, mode: str = "blur", confidence: float = SAM3_CONFIDENCE):
        self.mode = mode
        if not self._load_sam3():
            self._done(False)
            return
        self.sam3_proc.confidence_threshold = confidence
        self._log(f"[INFO] SAM3 confidence: {confidence:.2f}")

        # Log mask status once at start
        for path, label in [
            (SEARCH_REGION_CAM0, "cam_0 search"),
            (FALLBACK_MASK_CAM0, "cam_0 fallback"),
            (SEARCH_REGION_CAM1, "cam_1 search"),
            (FALLBACK_MASK_CAM1, "cam_1 fallback"),
        ]:
            if path.exists():
                self._log(f"[OK] Mask found: {label} - {path.name}")
            else:
                self._log(f"[WARN] Mask missing: {label} - {path.name}")

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
        self._anon_mode:     str = "blur"
        self._sam_confidence: float = SAM3_CONFIDENCE

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

        # Mask status
        all_mask_paths = [
            SEARCH_REGION_CAM0, FALLBACK_MASK_CAM0,
            SEARCH_REGION_CAM1, FALLBACK_MASK_CAM1,
        ]
        found = sum(1 for p in all_mask_paths if p.exists())
        total = len(all_mask_paths)
        if found == total:
            ctk.CTkLabel(main, text=f"Masks: {found}/{total} found",
                         text_color=A["success"],
                         font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(3, 0))
        else:
            ctk.CTkLabel(main, text=f"Masks: {found}/{total} found",
                         text_color=A["warning"],
                         font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(3, 0))

        # Anonymization mode
        mode_frame = ctk.CTkFrame(main, fg_color="transparent")
        mode_frame.pack(fill="x", padx=16, pady=(10, 0))
        ctk.CTkLabel(
            mode_frame, text="Anonymization mode:",
            text_color=A["text_secondary"],
        ).pack(side="left", padx=(0, 10))
        self.seg_mode = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Blur", "Black fill"],
            command=self._on_mode_change,
            fg_color=A["accent"],
            selected_color=A["highlight"],
            selected_hover_color="#a02840",
            unselected_color=A["accent"],
            unselected_hover_color=A["fg_color_primary"],
            text_color=A["text_primary"],
        )
        self.seg_mode.set("Blur")
        self.seg_mode.pack(side="left")

        # SAM3 confidence slider
        conf_frame = ctk.CTkFrame(main, fg_color="transparent")
        conf_frame.pack(fill="x", padx=16, pady=(8, 0))
        ctk.CTkLabel(
            conf_frame, text="SAM confidence:",
            text_color=A["text_secondary"],
        ).pack(side="left", padx=(0, 10))
        self.lbl_confidence = ctk.CTkLabel(
            conf_frame, text=f"{SAM3_CONFIDENCE:.2f}",
            text_color=A["text_primary"], width=36,
        )
        self.lbl_confidence.pack(side="right")
        self.slider_confidence = ctk.CTkSlider(
            conf_frame,
            from_=0.05, to=0.90, number_of_steps=17,
            command=self._on_confidence_change,
            fg_color=A["accent"],
            progress_color=A["highlight"],
            button_color=A["highlight"],
            button_hover_color="#a02840",
        )
        self.slider_confidence.set(SAM3_CONFIDENCE)
        self.slider_confidence.pack(side="left", fill="x", expand=True)

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

    def _on_mode_change(self, value: str):
        self._anon_mode = "black" if value == "Black fill" else "blur"

    def _on_confidence_change(self, value: float):
        self._sam_confidence = round(value, 2)
        self.lbl_confidence.configure(text=f"{self._sam_confidence:.2f}")

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

        self._processor  = Processor(self._msg_queue, mode=self._anon_mode)
        self._proc_thread = threading.Thread(
            target=self._processor.run,
            args=(self._input_paths, self._output_dir, self._anon_mode, self._sam_confidence),
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
