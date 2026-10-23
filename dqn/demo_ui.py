"""
Demo UI cho CarRacing DQN
Chạy: python demo_ui.py
"""

import sys
import os
import threading
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import queue

sys.path.insert(0, os.path.dirname(__file__))

CHECKPOINT  = "checkpoints/best.pth"
DISPLAY_W   = 420
DISPLAY_H   = 420

BG      = "#0A1628"
CARD    = "#0F2744"
NAVY    = "#0D2137"
TEAL    = "#0891B2"
TEAL_L  = "#22D3EE"
WHITE   = "#FFFFFF"
GRAY    = "#94A3B8"
GRAY_L  = "#E2E8F0"
GREEN   = "#10B981"
AMBER   = "#F59E0B"
RED     = "#EF4444"

ACTION_NAMES = ["Giữ nguyên", "Rẽ trái", "Rẽ phải", "Tăng tốc", "Phanh"]


class DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CarRacing DQN — Demo")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.env        = None
        self.agent      = None
        self.running    = False
        self.ready      = False
        self.paused     = False
        self.frame_q    = queue.Queue(maxsize=2)

        self.ep_num     = 0
        self.ep_step    = 0
        self.ep_reward  = 0.0
        self.ep_history = []   # [(ep, total_reward)]
        self.cur_action = "-"

        self._build_ui()
        self._load_async()

    # ──────────────────────────────────────────────────────────────────
    # BUILD UI
    # ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top accent bar
        tk.Frame(self.root, bg=TEAL, height=4).pack(fill="x")

        # Header
        hdr = tk.Frame(self.root, bg=BG, padx=20, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Self-Driving Car", font=("Arial", 20, "bold"),
                 bg=BG, fg=WHITE).pack(side="left")
        tk.Label(hdr, text="  DQN Demo", font=("Arial", 13),
                 bg=BG, fg=TEAL_L).pack(side="left")

        # Body
        body = tk.Frame(self.root, bg=BG, padx=20, pady=16)
        body.pack(fill="both")

        # ── LEFT: canvas ──────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", padx=(0,14))

        canvas_border = tk.Frame(left, bg=TEAL, padx=2, pady=2)
        canvas_border.pack()
        self.canvas = tk.Canvas(canvas_border, width=DISPLAY_W, height=DISPLAY_H,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack()

        # Status bar
        self.status_var = tk.StringVar(value="⏳  Đang tải model và môi trường...")
        tk.Frame(left, bg=NAVY, pady=6).pack(fill="x", pady=(6,0))
        status_frame = tk.Frame(left, bg=NAVY, pady=6)
        status_frame.pack(fill="x")
        tk.Label(status_frame, textvariable=self.status_var, font=("Arial", 11),
                 bg=NAVY, fg=TEAL_L, padx=12).pack()

        # Control buttons
        btn_frame = tk.Frame(left, bg=BG, pady=8)
        btn_frame.pack(fill="x")

        self.btn_pause = tk.Button(
            btn_frame, text="⏸  Tạm dừng", font=("Arial", 11, "bold"),
            bg=AMBER, fg=WHITE, activebackground="#D97706",
            relief="flat", cursor="hand2", pady=8,
            command=self._toggle_pause, state="disabled"
        )
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=(0,6))

        self.btn_stop = tk.Button(
            btn_frame, text="■  Dừng hẳn", font=("Arial", 11, "bold"),
            bg=RED, fg=WHITE, activebackground="#DC2626",
            relief="flat", cursor="hand2", pady=8,
            command=self._stop, state="disabled"
        )
        self.btn_stop.pack(side="left", fill="x", expand=True)

        # ── RIGHT: stats + history ─────────────────────────────────────
        right = tk.Frame(body, bg=BG, width=230)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        # Live stats
        tk.Label(right, text="LIVE", font=("Arial", 9, "bold"),
                 bg=BG, fg=GRAY).pack(anchor="w", pady=(0,4))

        self.stat_vars = {}
        items = [
            ("ep",      "Episode",        WHITE),
            ("step",    "Step",           WHITE),
            ("reward",  "Reward (step)",  AMBER),
            ("total",   "Total reward",   GREEN),
            ("action",  "Action",         TEAL_L),
        ]
        for key, label, color in items:
            card = tk.Frame(right, bg=CARD, padx=10, pady=6)
            card.pack(fill="x", pady=2)
            tk.Label(card, text=label, font=("Arial", 9),
                     bg=CARD, fg=GRAY).pack(anchor="w")
            var = tk.StringVar(value="—")
            tk.Label(card, textvariable=var, font=("Arial", 15, "bold"),
                     bg=CARD, fg=color).pack(anchor="w")
            self.stat_vars[key] = var

        # Episode history
        tk.Label(right, text="LỊCH SỬ EPISODE", font=("Arial", 9, "bold"),
                 bg=BG, fg=GRAY).pack(anchor="w", pady=(12,4))

        hist_frame = tk.Frame(right, bg=CARD, padx=8, pady=6)
        hist_frame.pack(fill="both", expand=True)

        # Header row
        hrow = tk.Frame(hist_frame, bg=CARD)
        hrow.pack(fill="x")
        tk.Label(hrow, text="Ep", width=4, font=("Arial", 9, "bold"),
                 bg=CARD, fg=GRAY, anchor="w").pack(side="left")
        tk.Label(hrow, text="Reward", font=("Arial", 9, "bold"),
                 bg=CARD, fg=GRAY, anchor="w").pack(side="left", padx=(8,0))
        tk.Label(hrow, text="Steps", font=("Arial", 9, "bold"),
                 bg=CARD, fg=GRAY, anchor="e").pack(side="right")

        tk.Frame(hist_frame, bg=TEAL, height=1).pack(fill="x", pady=4)

        # Scrollable listbox
        self.hist_list = tk.Listbox(
            hist_frame, bg=CARD, fg=GRAY_L, font=("Courier New", 10),
            selectbackground=NAVY, relief="flat", highlightthickness=0,
            height=8, activestyle="none"
        )
        self.hist_list.pack(fill="both", expand=True)

        # Best reward label
        self.best_var = tk.StringVar(value="Best: —")
        tk.Label(right, textvariable=self.best_var, font=("Arial", 10, "bold"),
                 bg=BG, fg=GREEN).pack(anchor="w", pady=(6,0))

        # Draw placeholder
        self._draw_placeholder("⏳ Đang tải...")
        self._update_ui()

    def _draw_placeholder(self, msg=""):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, DISPLAY_W, DISPLAY_H, fill="#000000")
        self.canvas.create_text(DISPLAY_W//2, DISPLAY_H//2 - 20,
                                text="🚗", font=("Arial", 48), fill=TEAL)
        if msg:
            self.canvas.create_text(DISPLAY_W//2, DISPLAY_H//2 + 40,
                                    text=msg, font=("Arial", 12), fill=GRAY)

    # ──────────────────────────────────────────────────────────────────
    # LOAD MODEL + ENV
    # ──────────────────────────────────────────────────────────────────
    def _load_async(self):
        def _load():
            try:
                from environment.carracing_env import CarRacingEnv
                from agents.dqn_agent import DQNAgent
                from configs.config import ENV_CONFIG, AGENT_CONFIG

                self.status_var.set("⏳  Khởi động môi trường...")
                cfg = {**ENV_CONFIG, "render_mode": "rgb_array"}
                self.env = CarRacingEnv(cfg)

                self.status_var.set("⏳  Tải model...")
                self.agent = DQNAgent(AGENT_CONFIG)

                if os.path.exists(CHECKPOINT):
                    self.agent.load(CHECKPOINT)
                    self.status_var.set("✅  Sẵn sàng — đang chạy tự động...")
                else:
                    self.status_var.set("⚠️  Không tìm thấy checkpoint!")
                    return

                self.ready = True
                # Tự động bắt đầu chạy luôn
                self.root.after(0, self._start_auto)

            except Exception as e:
                self.status_var.set(f"❌  Lỗi: {str(e)[:60]}")

        threading.Thread(target=_load, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────
    # DEMO LOOP — tự động chạy episode liên tục
    # ──────────────────────────────────────────────────────────────────
    def _start_auto(self):
        self.running = True
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        """Chạy episode liên tục cho đến khi bị dừng."""
        while self.running:
            self._run_one_episode()
            if self.running:
                time.sleep(0.5)  # nghỉ ngắn giữa các episode

    def _run_one_episode(self):
        self.ep_num    += 1
        self.ep_step    = 0
        self.ep_reward  = 0.0
        self._ep_total  = 0.0  
        ep_steps_total  = 0

        self.root.after(0, lambda ep=self.ep_num:
            self.status_var.set(f"▶  Episode {ep} đang chạy..."))

        state = self.env.reset()

        while self.running:
            # Tạm dừng
            while self.paused and self.running:
                time.sleep(0.1)

            action = self.agent.select_action(state, training=False)
            next_state, reward, done, _ = self.env.step(action)

            self.ep_step    = ep_steps_total + 1
            self.ep_reward  = reward
            self._ep_total += reward  
            self.cur_action = ACTION_NAMES[action]
            ep_steps_total += 1

            # Lấy frame để render
            raw = self.env._env.render()
            if raw is not None:
                try:
                    self.frame_q.put_nowait(raw)
                except queue.Full:
                    pass

            state = next_state
            time.sleep(0.025)

            if done:
                break

        # Episode kết thúc — lưu lịch sử
        total = sum([self.ep_reward]) if ep_steps_total == 1 else self.ep_reward
        # Tính tổng thật bằng cách lấy từ env trực tiếp
        self._record_episode(self.ep_num, ep_steps_total)

    def _record_episode(self, ep, steps):
        """Không có total reward riêng nên tính từ stat_vars."""
        # Đọc total từ label
        try:
            total = float(self.stat_vars["total"].get())
        except Exception:
            total = 0.0

        self.ep_history.append((ep, total, steps))

        # Update history list
        def _update():
            color = GREEN if total >= 0 else RED
            line = f"#{ep:>3}  {total:>+8.1f}  {steps:>5} steps"
            self.hist_list.insert(0, line)  # mới nhất ở trên
            self.hist_list.itemconfig(0, fg=GREEN if total >= 0 else "#F87171")

            # Best reward
            best = max(self.ep_history, key=lambda x: x[1])
            self.best_var.set(f"Best: Ep#{best[0]}  {best[1]:+.1f}")

            self.status_var.set(
                f"✅  Episode {ep} xong — reward {total:+.1f} | Ep tiếp theo đang chuẩn bị..."
            )
        self.root.after(0, _update)

    # ──────────────────────────────────────────────────────────────────
    # CONTROLS
    # ──────────────────────────────────────────────────────────────────
    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="▶  Tiếp tục")
            self.status_var.set("⏸  Đã tạm dừng")
        else:
            self.btn_pause.config(text="⏸  Tạm dừng")
            self.status_var.set("▶  Đang chạy tiếp...")

    def _stop(self):
        self.running = False
        self.paused  = False
        self.btn_pause.config(state="disabled", text="⏸  Tạm dừng")
        self.btn_stop.config(state="disabled")
        self.status_var.set("⏹  Đã dừng.")

    # ──────────────────────────────────────────────────────────────────
    # UI UPDATE LOOP (~30 FPS)
    # ──────────────────────────────────────────────────────────────────
    def _update_ui(self):
        # Update stats
        if self.running and not self.paused:
            self.stat_vars["ep"].set(str(self.ep_num))
            self.stat_vars["step"].set(str(self.ep_step))
            self.stat_vars["reward"].set(f"{self.ep_reward:+.2f}")
            self.stat_vars["action"].set(self.cur_action)

            # Tính total reward từ đầu episode
            self.stat_vars["total"].set(f"{self._ep_total:.1f}")

        # Nếu episode mới bắt đầu thì reset total
        if not hasattr(self, "_last_ep"):
            self._last_ep = 0
        if self.ep_num != self._last_ep:
            self._ep_total = 0.0
            self._last_ep = self.ep_num

        # Update game frame
        try:
            frame = self.frame_q.get_nowait()
            img = Image.fromarray(frame).resize(
                (DISPLAY_W, DISPLAY_H), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self.canvas._photo = photo
        except queue.Empty:
            pass

        self.root.after(33, self._update_ui)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("720x580")
    app = DemoApp(root)
    root.mainloop()