"""
ABA Alphabet Live Tracer
========================
Scoring engine: pure OpenCV geometry — NO TensorFlow, NO model file needed.

How scoring works
-----------------
1. GIBBERISH / DOT GUARD  → if strokes are too sparse, too tiny, or cover
   almost no area  → score is capped at 0–15 %  (feels like 0 to the child)

2. SHAPE SCORE (Hu Moments)  → scale- & rotation-invariant shape descriptor
   comparing the child's drawing to a pre-rendered template.
   Naturally gives ≈50–70 % for any reasonable letter attempt.

3. COVERAGE BONUS  → did the child fill the canvas in roughly the right zone?
   Adds up to +20 % on top.

4. STROKE RICHNESS BONUS  → enough ink (not just one dot)? Adds up to +10 %.

Final score is blended and clamped to 0–100 %.
Pass threshold = 50 %.
"""

import tkinter as tk
import cv2
import os
import threading
import time
from datetime import datetime
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import messagebox
import webbrowser
import math

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CANVAS_SIZE   = 450
TEMPLATE_SIZE = 256
PASS_THRESHOLD = 0.50          # 50 % to pass
MIN_POINTS_FOR_REAL_ATTEMPT = 40   # below this → gibberish / dot guard kicks in
PEN_RADIUS    = 8
PEN_COLOR_HEX = "#1565C0"


# ─────────────────────────────────────────────────────────────────────────────
#  PRE-BUILD LETTER TEMPLATES  (white bg, thick black stroke)
# ─────────────────────────────────────────────────────────────────────────────
def _build_template(letter: str, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Returns a binary (0/255) numpy array – black letter on white."""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    font = None
    for cand in ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf",
                 "FreeSansBold.ttf", "LiberationSans-Bold.ttf",
                 "Verdana Bold.ttf", "verdanab.ttf"]:
        try:
            font = ImageFont.truetype(cand, int(size * 0.82))
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), letter, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((size - tw) // 2 - bb[0], (size - th) // 2 - bb[1]),
              letter, fill=0, font=font)
    arr = np.array(img, dtype=np.uint8)
    # Thicken the template so child's thick strokes match better
    kernel = np.ones((5, 5), np.uint8)
    ink = (arr < 128).astype(np.uint8) * 255
    ink = cv2.dilate(ink, kernel, iterations=2)
    return ink   # 255 = ink pixel, 0 = background


TEMPLATES: dict[str, np.ndarray] = {l: _build_template(l) for l in "ABCD"}


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def _render_strokes(points: list, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Render stroke list → binary numpy (255=ink, 0=bg), scaled to `size`."""
    if not points:
        return np.zeros((size, size), dtype=np.uint8)
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(img)
    r = PEN_RADIUS
    for (x, y) in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    _, binary = cv2.threshold(arr, 50, 255, cv2.THRESH_BINARY)
    return binary


def _hu_similarity(drawn_bin: np.ndarray, tmpl_bin: np.ndarray) -> float:
    """
    Compare two binary images using Hu Moments.
    Returns 0.0–1.0 (1.0 = identical shape).
    Hu moments are scale/rotation/translation invariant.
    """
    m1 = cv2.moments(drawn_bin)
    m2 = cv2.moments(tmpl_bin)
    h1 = cv2.HuMoments(m1).flatten()
    h2 = cv2.HuMoments(m2).flatten()

    # Log-scale comparison (standard approach)
    def log_hu(h):
        return np.array([
            -math.copysign(1, v) * math.log10(abs(v) + 1e-10)
            for v in h
        ])

    lh1, lh2 = log_hu(h1), log_hu(h2)
    diff = np.abs(lh1 - lh2)

    # Use first 5 moments (most discriminative)
    score = 1.0 / (1.0 + np.mean(diff[:5]) * 0.35)
    return float(np.clip(score, 0.0, 1.0))


def _bounding_box_ratio(points: list) -> float:
    """
    Returns 0–1: how well the drawing fills the canvas bounding box.
    A tiny dot cluster → near 0. A full letter → 0.4–0.9.
    """
    if len(points) < 5:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    area_ratio = (w * h) / (CANVAS_SIZE * CANVAS_SIZE)
    return float(np.clip(area_ratio * 6, 0.0, 1.0))   # scale so ~0.15 canvas area → 0.9


def _stroke_richness(points: list) -> float:
    """
    Returns 0–1 based on number of unique stroke points.
    40 pts → 0.5, 150 pts → 1.0, <10 pts → ~0.
    """
    n = len(set(points))   # deduplicate
    return float(np.clip(n / 150.0, 0.0, 1.0))


def score_drawing(points: list, letter: str) -> float:
    """
    Main scoring function. Returns 0.0–1.0.

    Gibberish / dot guard:
      - Fewer than MIN_POINTS_FOR_REAL_ATTEMPT unique points → capped at 0.12
      - Bounding box area < 3 % of canvas → capped at 0.10

    Otherwise:
      - 60 % weight: Hu-moment shape similarity vs template
      - 25 % weight: bounding-box spatial spread
      - 15 % weight: stroke richness (ink coverage)

    The blend is designed so any genuine letter attempt scores ≈ 0.50–0.75.
    """
    unique_pts = list(set(points))
    n = len(unique_pts)

    # ── GIBBERISH / DOT GUARD ─────────────────────────────────────────────
    if n < MIN_POINTS_FOR_REAL_ATTEMPT:
        # Very few points → dots or a tiny scribble
        return float(np.clip(n / MIN_POINTS_FOR_REAL_ATTEMPT * 0.12, 0.0, 0.12))

    bb_ratio = _bounding_box_ratio(unique_pts)
    if bb_ratio < 0.05:
        # Drawing is physically tiny — probably dots bunched together
        return 0.05

    # ── FULL SCORING ──────────────────────────────────────────────────────
    drawn_bin = _render_strokes(unique_pts, size=TEMPLATE_SIZE)
    tmpl_bin  = TEMPLATES[letter]

    hu_score  = _hu_similarity(drawn_bin, tmpl_bin)
    rich      = _stroke_richness(unique_pts)

    # Dilated-IoU as a secondary shape signal
    kernel = np.ones((11, 11), np.uint8)
    tmpl_dilated = cv2.dilate(tmpl_bin, kernel, iterations=2)
    inter = np.sum((drawn_bin > 0) & (tmpl_dilated > 0))
    union = np.sum((drawn_bin > 0) | (tmpl_dilated > 0))
    iou   = float(inter / union) if union > 0 else 0.0

    # Weighted blend
    raw = (0.50 * hu_score) + (0.25 * iou) + (0.15 * bb_ratio) + (0.10 * rich)

    # Generous floor: any real attempt (passes dot guard) gets ≥ 0.30
    raw = max(raw, 0.30)

    return float(np.clip(raw, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class ABALiveTracer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ABA Alphabet — Watch & Trace")
        self.root.geometry("1140x860")
        self.root.configure(bg="#0F172A")

        # TTS
        self.engine = None
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 130)
            except Exception:
                pass

        # Session state
        self.items   = list("ABCD")
        self.index   = 0
        self.traced  = False
        self.stroke_points: list[tuple[int, int]] = []
        self.session_results: list[dict] = []
        self.attempt_counts = {l: 0 for l in self.items}
        self.letter_start_time: float | None = None

        # Video
        self.video_demos = {
            "A": r"C:\Users\user\Downloads\videoplayback_XKUyRINk.mp4",
            "B": r"C:\Users\user\Downloads\b_k4L0awY5.mp4",
            "C": r"C:\Users\user\Downloads\how to write capital letter c - Le professeur (720p, h264).mp4",
            "D": r"C:\Users\user\Downloads\how to write capital letter d - Le professeur (720p, h264).mp4",
        }
        self.cap           = None
        self.video_running = False

        # Timer
        self.timer_seconds = 30
        self.current_time  = self.timer_seconds
        self.timer_job     = None

        self._build_ui()
        self.load_letter()

    # ── UI CONSTRUCTION ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top header bar ────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#1E293B", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="✏  ABA Alphabet Tracer",
                 font=("Georgia", 20, "bold"), bg="#1E293B", fg="#F1F5F9").pack(side="left", padx=24)
        self.timer_label = tk.Label(header, text="",
                                    font=("Courier New", 18, "bold"),
                                    bg="#1E293B", fg="#38BDF8")
        self.timer_label.pack(side="right", padx=24)

        # ── Progress pills ────────────────────────────────────────────────
        pill_row = tk.Frame(self.root, bg="#0F172A", pady=8)
        pill_row.pack(fill="x")
        self.pill_labels: dict[str, tk.Label] = {}
        for l in self.items:
            pill = tk.Label(pill_row, text=f" {l} ",
                            font=("Arial", 14, "bold"),
                            bg="#334155", fg="#94A3B8",
                            relief="flat", padx=12, pady=4)
            pill.pack(side="left", padx=6)
            self.pill_labels[l] = pill

        # ── Main two-panel area ───────────────────────────────────────────
        main = tk.Frame(self.root, bg="#0F172A")
        main.pack(expand=True, fill="both", padx=24, pady=8)

        # Left panel — teacher video
        left_wrap = tk.Frame(main, bg="#1E293B", bd=0,
                             highlightthickness=2, highlightbackground="#334155")
        left_wrap.pack(side="left", expand=True, fill="both", padx=(0, 12))

        tk.Label(left_wrap, text="👀  WATCH TEACHER",
                 font=("Arial", 13, "bold"), bg="#1E293B", fg="#F87171",
                 pady=8).pack()
        self.teacher_canvas = tk.Canvas(left_wrap,
                                        width=CANVAS_SIZE, height=CANVAS_SIZE,
                                        bg="#000000", highlightthickness=0)
        self.teacher_canvas.pack(padx=12, pady=(0, 8))
        tk.Button(left_wrap, text="↺  Replay",
                  font=("Arial", 12, "bold"),
                  bg="#F87171", fg="white", activebackground="#EF4444",
                  relief="flat", padx=16, pady=6,
                  command=self.play_video).pack(pady=(0, 12))

        # Right panel — student canvas
        right_wrap = tk.Frame(main, bg="#1E293B", bd=0,
                              highlightthickness=2, highlightbackground="#334155")
        right_wrap.pack(side="left", expand=True, fill="both", padx=(12, 0))

        tk.Label(right_wrap, text="🖊  YOUR TURN",
                 font=("Arial", 13, "bold"), bg="#1E293B", fg="#38BDF8",
                 pady=8).pack()

        self.letter_prompt = tk.Label(right_wrap, text="",
                                      font=("Georgia", 54, "bold"),
                                      bg="#1E293B", fg="#38BDF8")
        self.letter_prompt.pack()

        self.student_canvas = tk.Canvas(right_wrap,
                                        width=CANVAS_SIZE, height=CANVAS_SIZE,
                                        bg="#FFFFFF", highlightthickness=0,
                                        cursor="pencil")
        self.student_canvas.pack(padx=12, pady=(4, 6))
        self.student_canvas.bind("<B1-Motion>", self.paint)
        self.student_canvas.bind("<ButtonRelease-1>", self._pen_lifted)

        btn_row = tk.Frame(right_wrap, bg="#1E293B")
        btn_row.pack(pady=(0, 4))
        self.done_btn = tk.Button(btn_row, text="✅  I'm Done!",
                                  font=("Arial", 13, "bold"),
                                  bg="#0EA5E9", fg="white",
                                  activebackground="#0284C7",
                                  relief="flat", padx=20, pady=7,
                                  state="disabled",
                                  command=self._on_done_pressed)
        self.done_btn.pack(side="left", padx=6)
        tk.Button(btn_row, text="🗑  Clear",
                  font=("Arial", 12), bg="#334155", fg="#CBD5E1",
                  activebackground="#475569",
                  relief="flat", padx=14, pady=7,
                  command=self._clear_only).pack(side="left", padx=6)

        self.feedback_label = tk.Label(right_wrap, text="",
                                       font=("Arial", 12, "bold"),
                                       bg="#1E293B", fg="#F87171",
                                       wraplength=440)
        self.feedback_label.pack(pady=(2, 8))

        # ── Bottom status bar ─────────────────────────────────────────────
        self.status_bar = tk.Label(self.root,
                                   text="✅  Smart shape-scoring engine ready",
                                   font=("Arial", 10), bg="#1E293B", fg="#64748B",
                                   anchor="w", padx=16, pady=4)
        self.status_bar.pack(fill="x", side="bottom")

    # ── LETTER PILLS ────────────────────────────────────────────────────────
    def _update_pills(self, completed_letter: str | None = None, passed: bool = False):
        current = self.items[self.index] if self.index < len(self.items) else None
        for l, pill in self.pill_labels.items():
            if l == completed_letter:
                pill.config(bg="#22C55E" if passed else "#EF4444", fg="white")
            elif l == current:
                pill.config(bg="#0EA5E9", fg="white")
            else:
                pill.config(bg="#334155", fg="#94A3B8")

    # ── FEEDBACK ────────────────────────────────────────────────────────────
    def _show_feedback(self, text: str, color: str = "#F87171", ms: int = 2800):
        self.feedback_label.config(text=text, fg=color)
        self.root.after(ms, lambda: self.feedback_label.config(text=""))

    # ── VIDEO ────────────────────────────────────────────────────────────────
    def play_video(self):
        letter = self.items[self.index]
        path   = self.video_demos.get(letter, "")
        if not path or not os.path.exists(path):
            self.teacher_canvas.delete("all")
            self.teacher_canvas.create_text(
                CANVAS_SIZE // 2, CANVAS_SIZE // 2,
                text=f"No video\nfor  '{letter}'",
                font=("Arial", 26, "bold"), fill="#475569", justify="center")
            return
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.video_running = True
        self._stream_video()

    def _stream_video(self):
        if not self.video_running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release(); self.cap = None; self.video_running = False
            return
        frame = cv2.resize(frame, (CANVAS_SIZE, CANVAS_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.teacher_canvas.img = photo
        self.teacher_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.root.after(30, self._stream_video)

    # ── DRAWING ─────────────────────────────────────────────────────────────
    def paint(self, event):
        r = PEN_RADIUS
        self.student_canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            fill=PEN_COLOR_HEX, outline="")
        self.stroke_points.append((event.x, event.y))
        if not self.traced:
            self.traced = True
            self.done_btn.config(state="normal")

    def _pen_lifted(self, _event):
        # Brief visual pulse when pen lifts — shows ink count in status
        n = len(set(self.stroke_points))
        self.status_bar.config(text=f"📝  Ink points recorded: {n}   "
                                    f"(need ~{MIN_POINTS_FOR_REAL_ATTEMPT}+ for a full letter)")

    def _clear_only(self):
        """Clear the canvas but keep the timer running."""
        self.student_canvas.delete("all")
        self.stroke_points = []
        self.traced = False
        self.done_btn.config(state="disabled")
        self.status_bar.config(text="🗑  Canvas cleared — draw again!")

    # ── DONE / TIMER ────────────────────────────────────────────────────────
    def _on_done_pressed(self):
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        self.done_btn.config(state="disabled")
        self._evaluate_and_proceed()

    def start_timer(self):
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        self.current_time = self.timer_seconds
        self._tick()

    def _tick(self):
        secs = self.current_time
        color = "#38BDF8" if secs > 10 else ("#FB923C" if secs > 5 else "#F87171")
        self.timer_label.config(text=f"⏱  {secs:02d}s", fg=color)
        if secs > 0:
            self.current_time -= 1
            self.timer_job = self.root.after(1000, self._tick)
        else:
            self.timer_job = None
            if not self.traced:
                self._show_feedback(
                    f"⏰ Time's up! Please trace letter {self.items[self.index]}.",
                    "#FB923C", 2500)
                self.speak("Time's up. Please start tracing.")
                self._reset_canvas()
                self.start_timer()
            else:
                self.done_btn.config(state="disabled")
                self._evaluate_and_proceed()

    # ── CANVAS RESET ────────────────────────────────────────────────────────
    def _reset_canvas(self):
        self.student_canvas.delete("all")
        self.stroke_points = []
        self.traced = False
        self.done_btn.config(state="disabled")

    def load_letter(self):
        self.video_running = False
        if self.cap:
            self.cap.release(); self.cap = None
        self.teacher_canvas.delete("all")
        self._reset_canvas()
        self.feedback_label.config(text="")
        letter = self.items[self.index]
        self.letter_prompt.config(text=f"Trace:  {letter}")
        self._update_pills()
        self.letter_start_time = time.time()
        self.speak(letter)
        self.start_timer()
        self.root.after(500, self.play_video)
        self.status_bar.config(text=f"🔤  Now tracing letter  {letter}  —  draw on the white canvas!")

    # ── SCORING + EVALUATE ───────────────────────────────────────────────────
    def _evaluate_and_proceed(self):
        expected   = self.items[self.index]
        elapsed    = round(time.time() - self.letter_start_time) if self.letter_start_time else 0
        self.attempt_counts[expected] += 1
        attempt_num = self.attempt_counts[expected]

        sim    = score_drawing(self.stroke_points, expected)
        passed = sim >= PASS_THRESHOLD
        pct    = int(sim * 100)
        n_pts  = len(set(self.stroke_points))

        print(f"[Score] letter={expected}  attempt={attempt_num}  "
              f"pts={n_pts}  similarity={sim:.3f}  passed={passed}")

        self.session_results.append({
            "letter":       expected,
            "similarity":   sim,
            "passed":       passed,
            "attempt":      attempt_num,
            "time_seconds": elapsed,
            "ink_points":   n_pts,
        })

        self._update_pills(completed_letter=expected if passed else None, passed=passed)

        if passed:
            stars = "🌟" * min(3, max(1, (pct - 50) // 15 + 1))
            self._show_feedback(
                f"{stars} Amazing! You nailed letter {expected}!  ({pct}%)",
                "#4ADE80", 2200)
            self.speak(f"Amazing! You traced {expected} correctly!")
            self.root.after(2300, self._next_or_report)
        else:
            if n_pts < MIN_POINTS_FOR_REAL_ATTEMPT:
                msg = (f"Looks like just a dot or tiny scribble ({pct}%). "
                       f"Try tracing the whole letter {expected}!")
            else:
                msg = (f"Good try! Your {expected} scored {pct}%. "
                       f"Keep the shape bigger and clearer — you've got this!")
            self._show_feedback(msg, "#FB923C", 3500)
            self.speak(f"Try again. Trace the full letter {expected}.")
            self.root.after(3600, self._retry_letter)

    def _retry_letter(self):
        self._reset_canvas()
        self.letter_start_time = time.time()
        self.start_timer()
        self.status_bar.config(
            text=f"🔄  Try again — trace letter  {self.items[self.index]}  bigger and clearer!")

    # ── TTS ─────────────────────────────────────────────────────────────────
    def speak(self, text: str):
        if self.engine:
            threading.Thread(
                target=lambda: (self.engine.say(text), self.engine.runAndWait()),
                daemon=True).start()

    # ── NAVIGATION ──────────────────────────────────────────────────────────
    def _next_or_report(self):
        nxt = self.index + 1
        if nxt >= len(self.items):
            self.generate_html_report()
        else:
            self.index = nxt
            self.load_letter()

    # ── HTML REPORT ─────────────────────────────────────────────────────────
    def generate_html_report(self):
        self.speak("Great session! Opening your report now.")
        now          = datetime.now()
        session_date = now.strftime("%B %d, %Y")
        session_time = now.strftime("%I:%M %p")

        # Best attempt per letter
        letter_summaries: dict[str, dict] = {}
        for letter in self.items:
            attempts = [r for r in self.session_results if r["letter"] == letter]
            if not attempts:
                continue
            best       = max(attempts, key=lambda r: r["similarity"])
            passed_any = any(r["passed"] for r in attempts)
            total_time = sum(r["time_seconds"] for r in attempts)
            letter_summaries[letter] = {
                "attempts":   len(attempts),
                "passed":     passed_any,
                "acc_pct":    int(best["similarity"] * 100),
                "total_time": total_time,
                "ink_points": best["ink_points"],
            }

        total   = len(letter_summaries)
        passed  = sum(1 for s in letter_summaries.values() if s["passed"])
        overall = int((passed / total) * 100) if total else 0

        if overall >= 80:
            ov_emoji, ov_msg, ov_color = "🌟", "Fantastic work today!", "#22C55E"
        elif overall >= 50:
            ov_emoji, ov_msg, ov_color = "😊", "Good effort! Keep practising!", "#F59E0B"
        else:
            ov_emoji, ov_msg, ov_color = "💪", "Keep going — you're learning!", "#EF4444"

        needs_improvement = [l for l, s in letter_summaries.items() if not s["passed"]]

        if needs_improvement:
            ni_items = "".join(
                f'<li><strong>Letter {l}</strong> — best score was '
                f'<strong>{letter_summaries[l]["acc_pct"]}%</strong>. '
                f'Watch the video and trace "{l}" more fully.</li>'
                for l in needs_improvement
            )
            ni_html = f"""
      <div class="improve-box">
        <div class="improve-title">📌 Letters That Need More Practice</div>
        <ul class="improve-list">{ni_items}</ul>
      </div>"""
        else:
            ni_html = """
      <div class="improve-box passed">
        <div class="improve-title" style="color:#15803d;">🎉 Every Letter Passed!</div>
        <p style="color:#166534;margin-top:8px;font-weight:600;">
          Incredible — all four letters scored above 50%!
        </p>
      </div>"""

        cards_html = ""
        for letter in self.items:
            if letter not in letter_summaries:
                continue
            s = letter_summaries[letter]
            if s["passed"]:
                badge, b_bg        = "✅ Passed", "#22C55E"
                c_bg, bar_col      = "#F0FDF4", "#4ADE80"
                emoji, tip         = "⭐", "Great job — your tracing matched the letter shape well!"
            else:
                badge, b_bg        = "🔄 Keep Practising", "#F59E0B"
                c_bg, bar_col      = "#FFFBEB", "#FCD34D"
                emoji              = "📝"
                tip = (f"Your best attempt was {s['acc_pct']}%. "
                       f"Try to trace the full letter shape — bigger strokes help!")

            mins = s["total_time"] // 60
            secs = s["total_time"] % 60
            t_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            a_str = f"{s['attempts']} attempt{'s' if s['attempts'] != 1 else ''}"

            cards_html += f"""
      <div class="card" style="background:{c_bg};">
        <div class="letter-bubble">{letter}</div>
        <div class="card-body">
          <div class="card-title">{emoji} Letter {letter}</div>
          <span class="badge" style="background:{b_bg};">{badge}</span>
          <div class="bar-label">Shape similarity score</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{s['acc_pct']}%;background:{bar_col};"></div>
          </div>
          <div class="pct">{s['acc_pct']}%</div>
          <div class="meta">⏱ {t_str} &nbsp;|&nbsp; 🔁 {a_str} &nbsp;|&nbsp; 🖊 {s['ink_points']} ink pts</div>
          <div class="tip">💡 {tip}</div>
        </div>
      </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ABA Tracing Report</title>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Baloo+2:wght@700;800&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{
    font-family:'Nunito',sans-serif;
    background:linear-gradient(160deg,#0F172A 0%,#1E293B 60%,#0F172A 100%);
    min-height:100vh;padding:40px 20px;
  }}
  .page{{
    max-width:820px;margin:0 auto;
    background:#F8FAFC;border-radius:28px;
    box-shadow:0 32px 80px rgba(0,0,0,0.45);overflow:hidden;
  }}
  .header{{
    background:linear-gradient(135deg,#0EA5E9 0%,#6366F1 100%);
    padding:50px 44px 56px;text-align:center;position:relative;
  }}
  .header::after{{
    content:'';position:absolute;bottom:-24px;left:0;right:0;
    height:48px;background:#F8FAFC;
    border-radius:50% 50% 0 0 / 100% 100% 0 0;
  }}
  .header-emoji{{font-size:44px;margin-bottom:10px;}}
  .header h1{{
    font-family:'Baloo 2',cursive;font-size:36px;color:#fff;
    text-shadow:0 4px 16px rgba(0,0,0,0.3);
  }}
  .header .sub{{color:rgba(255,255,255,0.85);font-size:15px;font-weight:700;margin-top:6px;}}
  .date-chip{{
    display:inline-block;margin-top:14px;
    background:rgba(255,255,255,0.2);
    color:#fff;border-radius:20px;padding:6px 20px;
    font-size:13px;font-weight:700;
  }}
  .body{{padding:60px 44px 44px;}}
  .overall{{
    background:linear-gradient(135deg,#EFF6FF,#F5F3FF);
    border:2px solid #C7D2FE;border-radius:22px;
    padding:32px;text-align:center;margin-bottom:30px;
  }}
  .ov-emoji{{font-size:64px;}}
  .ov-label{{font-size:12px;color:#94A3B8;font-weight:800;
    text-transform:uppercase;letter-spacing:1.5px;margin:10px 0 4px;}}
  .ov-score{{font-family:'Baloo 2',cursive;font-size:56px;font-weight:800;color:{ov_color};}}
  .ov-msg{{font-size:18px;font-weight:800;color:#1E293B;margin-top:8px;}}
  .ov-sub{{font-size:14px;color:#64748B;font-weight:600;margin-top:6px;}}
  .improve-box{{
    background:#FFF7ED;border:2px solid #FCD34D;
    border-radius:18px;padding:24px 28px;margin-bottom:28px;
  }}
  .improve-box.passed{{background:#F0FDF4;border-color:#86EFAC;}}
  .improve-title{{
    font-family:'Baloo 2',cursive;font-size:17px;color:#B45309;margin-bottom:10px;
  }}
  .improve-list{{padding-left:20px;}}
  .improve-list li{{
    font-size:14px;color:#78350F;font-weight:600;
    margin-bottom:8px;line-height:1.7;
  }}
  .section-title{{
    font-family:'Baloo 2',cursive;font-size:22px;color:#3B82F6;margin-bottom:18px;
  }}
  .card{{
    display:flex;align-items:flex-start;gap:20px;
    border-radius:18px;padding:22px 24px;
    margin-bottom:16px;border:2px solid rgba(0,0,0,0.05);
  }}
  .letter-bubble{{
    width:72px;height:72px;border-radius:50%;flex-shrink:0;
    background:linear-gradient(135deg,#0EA5E9,#6366F1);
    color:#fff;font-family:'Baloo 2',cursive;
    font-size:36px;font-weight:800;
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 8px 20px rgba(14,165,233,0.4);
  }}
  .card-body{{flex:1;}}
  .card-title{{font-size:17px;font-weight:800;color:#1E293B;margin-bottom:6px;}}
  .badge{{
    display:inline-block;color:#fff;border-radius:10px;
    padding:3px 13px;font-size:12px;font-weight:700;margin-bottom:10px;
  }}
  .bar-label{{
    font-size:11px;color:#94A3B8;font-weight:700;
    text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;
  }}
  .bar-track{{
    background:#E2E8F0;border-radius:99px;height:12px;
    width:100%;overflow:hidden;margin-bottom:6px;
  }}
  .bar-fill{{height:100%;border-radius:99px;transition:width 0.6s ease;}}
  .pct{{font-size:26px;font-weight:900;color:#334155;}}
  .meta{{font-size:12px;color:#94A3B8;font-weight:600;margin-top:6px;}}
  .tip{{
    font-size:13px;color:#44403C;font-weight:600;margin-top:10px;
    background:rgba(0,0,0,0.04);border-radius:10px;
    padding:9px 13px;line-height:1.6;
  }}
  .footer{{
    text-align:center;padding:22px;
    background:#F1F5F9;color:#94A3B8;font-size:13px;font-weight:600;
  }}
  @media print{{
    body{{background:#fff;padding:0;}}
    .page{{box-shadow:none;border-radius:0;}}
  }}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <div class="header-emoji">🌈</div>
    <h1>ABA Alphabet Tracing Report</h1>
    <div class="sub">Letters A · B · C · D</div>
    <div class="date-chip">📅 {session_date} &nbsp;|&nbsp; 🕐 {session_time}</div>
  </div>

  <div class="body">
    <div class="overall">
      <div class="ov-emoji">{ov_emoji}</div>
      <div class="ov-label">Overall Score</div>
      <div class="ov-score">{overall}%</div>
      <div class="ov-msg">{ov_msg}</div>
      <div class="ov-sub">✅ {passed} out of {total} letters passed (50 % threshold)</div>
    </div>

    {ni_html}

    <div class="section-title">📝 Letter-by-Letter Breakdown</div>
    {cards_html}
  </div>

  <div class="footer">
    Generated by ABA Alphabet Tracer &nbsp;•&nbsp; {session_date}
  </div>
</div>
</body>
</html>"""

        desktop  = os.path.join(os.path.expanduser("~"), "Desktop")
        filename = f"ABA_Report_{now.strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(desktop, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report saved → {filepath}")
        webbrowser.open(f"file:///{filepath}")
        messagebox.showinfo(
            "Session Complete! 🎉",
            f"Great work today!\n\nReport saved to your Desktop:\n{filename}"
            "\n\nOpening in browser — you can print it from there!")
        self.root.quit()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    ABALiveTracer(root)
    root.mainloop()
