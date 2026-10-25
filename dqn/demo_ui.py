"""Interactive challenge demo. Run with .venv/Scripts/python.exe dqn/demo_ui.py."""
from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
BG, CARD, PANEL = "#0b1220", "#131f30", "#1b2b40"
TEXT, MUTED, ACCENT, AMBER, RED = "#edf3fa", "#98abc1", "#57dbc2", "#ffc575", "#ff7d87"
SCENES = {"Rừng cây": "forest", "Đô thị": "city", "Sa mạc": "desert", "Đường gốc": "classic"}
LEVELS = {"Thường": "normal", "Khó": "hard"}
ACTIONS = ["Thả trôi", "Rẽ trái", "Rẽ phải", "Tăng tốc", "Phanh"]
REASONS = {"wrecked": "Xe hỏng", "stuck": "Mắc kẹt / lạc đường", "time_limit": "Hết thời gian",
           "finished": "Hoàn thành", "out_of_bounds": "Ra ngoài bản đồ"}


@dataclass(frozen=True)
class Settings:
    scenario: str
    difficulty: str
    weather: str
    driver: str
    seed: int
    checkpoint: str


class Simulation(threading.Thread):
    """Own all Gym/PyTorch objects here; only the main thread touches Tk."""
    def __init__(self):
        super().__init__(daemon=True)
        self.commands = queue.Queue()
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.env = self.agent = None
        self.loaded_checkpoint = None
        self.settings = None
        self.paused = False
        self.action = 0
        self.episode = 0
        self.next_reset = None

    def _reset(self):
        self.seed = self.settings.seed + self.episode
        self.episode += 1
        self.state = self.env.reset(seed=self.seed)
        self.total, self.steps = 0., 0
        self.next_reset = None
        self.events.put(("reset", (self.env.map_data(), self.seed, self.episode)))
        self._frame(self.env.last_info, 0., 0)

    def _start_map(self, settings):
        from environment.carracing_env import CarRacingEnv
        from configs.config import ENV_CONFIG, AGENT_CONFIG
        self.events.put(("status", "Đang tạo môi trường..."))
        if self.env is not None:
            self.env.close()
            self.env = None
        self.settings = settings
        if settings.driver == "AI" and self.loaded_checkpoint != settings.checkpoint:
            self.events.put(("status", "Đang tải model..."))
            if not Path(settings.checkpoint).is_file():
                raise FileNotFoundError("Không tìm thấy checkpoint. Chọn file .pth hoặc chuyển sang Tự lái.")
            import torch
            from agents.dqn_agent import DQNAgent
            torch.set_num_threads(2)
            agent = DQNAgent(AGENT_CONFIG)
            agent.load(settings.checkpoint)
            self.agent, self.loaded_checkpoint = agent, settings.checkpoint
        self.env = CarRacingEnv({**ENV_CONFIG, "scenario": settings.scenario,
                                 "difficulty": settings.difficulty, "weather": settings.weather})
        self.paused, self.action, self.episode = False, 0, 0
        self._reset()
        self.events.put(("started", settings))

    def _frame(self, info, reward, action):
        payload = dict(info=info.copy(), reward=reward, total=self.total,
                       step=self.steps, episode=self.episode, seed=self.seed,
                       action=action, frame=self.env.render())
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(payload)

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    while True:
                        command, value = self.commands.get_nowait()
                        if command == "start":
                            self._start_map(value)
                        elif command == "pause":
                            self.paused = value
                        elif command == "action":
                            self.action = value
                except queue.Empty:
                    pass
                except Exception as error:
                    if self.env is not None:
                        self.env.close()
                    self.env = None
                    self.events.put(("error", str(error)))
                if self.env is None or self.paused:
                    self.stop_event.wait(.03)
                    continue
                if self.next_reset is not None:
                    if time.monotonic() >= self.next_reset:
                        self._reset()
                    else:
                        self.stop_event.wait(.03)
                        continue
                try:
                    action = self.agent.select_action(self.state, training=False) if self.settings.driver == "AI" else self.action
                    self.state, reward, done, info = self.env.step(action)
                    self.steps += 1
                    self.total += reward
                    self._frame(info, reward, action)
                    if done:
                        self.events.put(("finished", (self.seed, self.total, info)))
                        self.next_reset = time.monotonic() + 1.5
                    # Four simulation frames at 50 Hz => about real-time playback.
                    self.stop_event.wait(.06)
                except Exception as error:
                    self.env.close()
                    self.env = None
                    self.events.put(("error", str(error)))
        finally:
            if self.env is not None:
                self.env.close()


class DemoApp:
    def __init__(self, root):
        self.root = root
        root.title("VROOM — Driving Challenge")
        root.configure(bg=BG)
        root.geometry("1120x820")
        root.resizable(False, False)
        self.worker = Simulation()
        self.paused, self.started = False, False
        self.keys = set()
        self.map_transform = None
        self.car_dot = None
        self._build_ui()
        self.worker.start()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.bind("<KeyPress>", self._key_down)
        root.bind("<KeyRelease>", self._key_up)
        root.bind("<FocusOut>", self._focus_out)
        self._poll()
        root.after(150, self._start)

    def _label(self, parent, text="", *, color=TEXT, size=10, bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kwargs)

    def _button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, bg=ACCENT if primary else PANEL,
                         fg=BG if primary else TEXT, activebackground="#7ce6d2", relief="flat",
                         font=("Segoe UI", 10, "bold"), padx=16, pady=8, cursor="hand2")

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT,
                        arrowcolor=ACCENT, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL,
                        borderwidth=0, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        style.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT,
                        borderwidth=0, rowheight=25, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        tk.Frame(self.root, bg=ACCENT, height=3).pack(fill="x")
        header = tk.Frame(self.root, bg=BG, padx=24, pady=12)
        header.pack(fill="x")
        self._label(header, "VROOM", size=24, bold=True).pack(side="left")
        self._label(header, "  /  DRIVING CHALLENGE", color=ACCENT, size=12).pack(side="left", pady=(6, 0))
        self._label(header, "Vật cản thật · Va chạm · Độ bám đường", color=MUTED).pack(side="right", pady=(8, 0))
        controls = tk.Frame(self.root, bg=CARD, padx=16, pady=12)
        controls.pack(fill="x", padx=24)
        self.scene = tk.StringVar(value="Rừng cây")
        self.level = tk.StringVar(value="Thường")
        self.driver = tk.StringVar(value="AI")
        self.seed_var = tk.StringVar(value="42")
        self.rain = tk.BooleanVar(value=False)
        self.checkpoint = str(ROOT / "checkpoints" / "best.pth")
        for title, variable, values, width in [("BỐI CẢNH", self.scene, list(SCENES), 13),
                                               ("ĐỘ KHÓ", self.level, list(LEVELS), 9),
                                               ("ĐIỀU KHIỂN", self.driver, ["AI", "Tự lái"], 10)]:
            group = tk.Frame(controls, bg=CARD)
            group.pack(side="left", padx=(0, 16))
            self._label(group, title, color=MUTED, size=8, bold=True).pack(anchor="w", pady=(0, 4))
            ttk.Combobox(group, textvariable=variable, values=values, state="readonly", width=width).pack()
        group = tk.Frame(controls, bg=CARD)
        group.pack(side="left", padx=(0, 12))
        self._label(group, "MAP SEED", color=MUTED, size=8, bold=True).pack(anchor="w", pady=(0, 4))
        tk.Entry(group, textvariable=self.seed_var, width=9, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 11)).pack(ipady=6)
        tk.Checkbutton(controls, text="Mưa / đường trơn", variable=self.rain, bg=CARD, fg=TEXT,
                       selectcolor=PANEL, activebackground=CARD, activeforeground=TEXT,
                       font=("Segoe UI", 10)).pack(side="left", padx=4, pady=(16, 0))
        self._button(controls, "Chạy map", self._start, True).pack(side="right", pady=(12, 0))
        model_row = tk.Frame(self.root, bg=BG, padx=24, pady=8)
        model_row.pack(fill="x")
        self.model_label = self._label(model_row, "Model: best.pth · Model gốc chưa được huấn luyện né vật cản", color=AMBER, size=9)
        self.model_label.pack(side="left")
        tk.Button(model_row, text="Chọn model .pth", command=self._choose_model, bg=BG, fg=ACCENT,
                  relief="flat", font=("Segoe UI", 9), cursor="hand2").pack(side="right")
        body = tk.Frame(self.root, bg=BG, padx=24)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", anchor="n", padx=(0, 18))
        self.canvas = tk.Canvas(left, width=720, height=480, bg="#070d15", highlightthickness=1,
                                highlightbackground="#294258", takefocus=True)
        self.canvas.pack()
        self.canvas.create_text(360, 240, text="Đang tạo đường đua...", fill=MUTED, font=("Segoe UI", 15))
        self.image_item = None
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set())
        self.status = tk.StringVar(value="Đang khởi động...")
        tk.Label(left, textvariable=self.status, bg=CARD, fg=ACCENT, font=("Segoe UI", 10),
                 wraplength=690, justify="left", anchor="w", padx=12, pady=10).pack(fill="x", pady=(8, 0))
        buttons = tk.Frame(left, bg=BG, pady=10)
        buttons.pack(fill="x")
        self.pause_btn = self._button(buttons, "Tạm dừng", self._pause)
        self.pause_btn.pack(side="left")
        self._button(buttons, "Map tiếp theo", self._next).pack(side="left", padx=8)
        self._label(buttons, "↑ / W ga    ← → / A D rẽ    Space phanh", color=MUTED, size=9).pack(side="right")
        right = tk.Frame(body, bg=BG, width=310)
        right.pack(side="left", fill="both", expand=True)
        self._label(right, "TELEMETRY", color=MUTED, size=9, bold=True).pack(anchor="w", pady=(0, 6))
        cards = tk.Frame(right, bg=BG)
        cards.pack(fill="x")
        cards.columnconfigure((0, 1), weight=1)
        self.stats = {}
        for i, (key, title, color) in enumerate([("speed", "TỐC ĐỘ · km/h", TEXT), ("progress", "ĐƯỜNG ĐÃ ĐI", ACCENT),
                                                ("damage", "HƯ HẠI", RED), ("hits", "VA CHẠM", AMBER),
                                                ("reward", "TỔNG REWARD", TEXT), ("action", "HÀNH ĐỘNG", ACCENT)]):
            card = tk.Frame(cards, bg=CARD, padx=10, pady=6)
            card.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0, 6), pady=(0, 6))
            self._label(card, title, color=MUTED, size=8).pack(anchor="w")
            var = tk.StringVar(value="—")
            tk.Label(card, textvariable=var, bg=CARD, fg=color, font=("Segoe UI", 14, "bold")).pack(anchor="w")
            self.stats[key] = var
        self.map_label = self._label(right, "BẢN ĐỒ · chấm cam = vật cản", color=MUTED, size=9, bold=True)
        self.map_label.pack(anchor="w", pady=(8, 6))
        self.minimap = tk.Canvas(right, width=290, height=120, bg=CARD, highlightthickness=0)
        self.minimap.pack(fill="x")
        self.run_label = self._label(right, "Seed 42 · Episode 1", color=MUTED, size=9)
        self.run_label.pack(anchor="w", pady=6)
        self._label(right, "LƯỢT VỪA CHẠY", color=MUTED, size=9, bold=True).pack(anchor="w", pady=(6, 5))
        self.history = ttk.Treeview(right, columns=("seed", "reward", "hits"), show="headings", height=3)
        for key, title, width in [("seed", "Seed", 60), ("reward", "Reward", 95), ("hits", "Va chạm", 75)]:
            self.history.heading(key, text=title)
            self.history.column(key, width=width, anchor="center")
        self.history.pack(fill="x")
        self._label(right, "Cam: rào chắn · Xanh: xe đỗ\nNâu: thùng hàng · Mưa giảm độ bám\nVa chạm −80 điểm · Hỏng 100% kết thúc", color=MUTED, size=9,
                    justify="left").pack(anchor="w", pady=(10, 0))

    def _settings(self):
        seed = int(self.seed_var.get())
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed phải nằm trong khoảng 0 đến 4294967295.")
        return Settings(SCENES[self.scene.get()], LEVELS[self.level.get()],
                        "rain" if self.rain.get() else "dry", self.driver.get(), seed, self.checkpoint)

    def _start(self):
        try:
            settings = self._settings()
        except ValueError as error:
            self.status.set(str(error))
            return
        self.started, self.paused = False, False
        self.pause_btn.config(text="Tạm dừng")
        self.keys.clear()
        self.worker.commands.put(("start", settings))
        self.status.set("Đang tải map...")
        self.canvas.focus_set()

    def _choose_model(self):
        path = filedialog.askopenfilename(title="Chọn checkpoint", initialdir=ROOT / "checkpoints",
                                          filetypes=[("PyTorch checkpoint", "*.pth *.pt")])
        if path:
            self.checkpoint = path
            self.model_label.config(text=f"Model: {Path(path).name} · Chỉ chọn model cùng kiến trúc 5 hành động")
            self._start()

    def _pause(self):
        if self.started:
            self.paused = not self.paused
            self.worker.commands.put(("pause", self.paused))
            self.pause_btn.config(text="Tiếp tục" if self.paused else "Tạm dừng")
            self.status.set("Đã tạm dừng" if self.paused else "Đang chạy...")

    def _next(self):
        try:
            self.seed_var.set(str(int(self.seed_var.get()) + 1))
        except ValueError:
            self.seed_var.set("42")
        self._start()

    def _key_down(self, event):
        if event.widget.winfo_class() in ("Entry", "TCombobox"):
            return
        self.keys.add(event.keysym.lower())
        self._send_keys()

    def _key_up(self, event):
        self.keys.discard(event.keysym.lower())
        self._send_keys()

    def _focus_out(self, event):
        self.keys.clear()
        self.worker.commands.put(("action", 0))

    def _send_keys(self):
        keys = self.keys
        action = 4 if keys & {"space", "down", "s"} else 1 if keys & {"left", "a"} else 2 if keys & {"right", "d"} else 3 if keys & {"up", "w"} else 0
        self.worker.commands.put(("action", action))

    def _draw_map(self, data):
        self.minimap.delete("all")
        points = data["track"]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        scale = min(260 / max(max(xs) - min(xs), 1), 95 / max(max(ys) - min(ys), 1))
        mid_x, mid_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        self.map_transform = lambda p: (145 + (p[0] - mid_x) * scale, 60 - (p[1] - mid_y) * scale)
        coords = [v for p in points + [points[0]] for v in self.map_transform(p)]
        self.minimap.create_line(*coords, fill="#5d7288", width=5, smooth=True)
        for p in data["obstacles"]:
            x, y = self.map_transform(p)
            self.minimap.create_oval(x-2, y-2, x+2, y+2, fill=AMBER, outline="")
        self.car_dot = self.minimap.create_oval(0, 0, 6, 6, fill=ACCENT, outline=TEXT)

    def _poll(self):
        try:
            while True:
                event, payload = self.worker.events.get_nowait()
                if event == "status":
                    self.status.set(payload)
                elif event == "error":
                    self.started = False
                    self.status.set(f"Lỗi: {payload}")
                elif event == "started":
                    self.started = True
                    self.status.set("AI đang chạy · Model gốc có thể va chạm vì chưa học môi trường mới" if payload.driver == "AI" else "Tự lái · Click khung hình rồi dùng phím mũi tên / WASD")
                elif event == "reset":
                    data, seed, episode = payload
                    self._draw_map(data)
                    self.run_label.config(text=f"Seed {seed} · Episode {episode} · {len(data['obstacles'])} vật cản")
                    self.seed_var.set(str(seed))
                elif event == "finished":
                    seed, total, info = payload
                    self.history.insert("", 0, values=(seed, f"{total:+.1f}", info["collisions"]))
                    for item in self.history.get_children()[30:]:
                        self.history.delete(item)
                    reason = REASONS.get(info.get("end_reason"), "Kết thúc lượt")
                    self.status.set(f"{reason} · Reward {total:+.1f} · Đang chuẩn bị map tiếp theo...")
        except queue.Empty:
            pass
        try:
            payload = self.worker.frames.get_nowait()
            info = payload["info"]
            image = Image.fromarray(payload["frame"]).resize((720, 480), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            if self.image_item is None:
                self.canvas.delete("all")
                self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            else:
                self.canvas.itemconfigure(self.image_item, image=self.photo)
            for key, value in {"speed": f"{info['speed_kmh']:.0f}", "progress": f"{info['progress']:.0%}",
                               "damage": f"{info['damage']}%", "hits": str(info['collisions']),
                               "reward": f"{payload['total']:+.1f}", "action": ACTIONS[payload['action']]}.items():
                self.stats[key].set(value)
            if self.map_transform and self.car_dot is not None:
                x, y = self.map_transform(info["car_position"])
                self.minimap.coords(self.car_dot, x-3, y-3, x+3, y+3)
        except queue.Empty:
            pass
        self.root.after(33, self._poll)

    def _close(self):
        self.worker.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()
