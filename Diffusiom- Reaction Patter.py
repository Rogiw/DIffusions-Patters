import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

try:
    from scipy.signal import convolve2d
except ImportError as exc:
    raise SystemExit(
        "Este script precisa do scipy. Instale com: pip install scipy matplotlib numpy"
    ) from exc


KERNEL = np.array(
    [
        [0.05, 0.20, 0.05],
        [0.20, -1.00, 0.20],
        [0.05, 0.20, 0.05],
    ],
    dtype=float,
)

PRESETS = {
    "Coral": {"feed": 0.0545, "kill": 0.0620, "da": 0.20, "db": 0.10},
    "Spots": {"feed": 0.0350, "kill": 0.0650, "da": 0.20, "db": 0.10},
    "Worms": {"feed": 0.0460, "kill": 0.0630, "da": 0.20, "db": 0.10},
    "Maze": {"feed": 0.0290, "kill": 0.0570, "da": 0.20, "db": 0.10},
}


class ReactionDiffusionApp:
    def __init__(self, size=160, seed=4):
        self.size = size
        self.seed = seed
        self.running = True
        self.steps_per_frame = 6
        self.active_preset = "Coral"
        self.rng = np.random.default_rng(seed)

        self.da = PRESETS[self.active_preset]["da"]
        self.db = PRESETS[self.active_preset]["db"]
        self.feed = PRESETS[self.active_preset]["feed"]
        self.kill = PRESETS[self.active_preset]["kill"]

        self.a, self.b = self.make_initial_state()
        self.build_ui()

    def make_initial_state(self):
        a = np.ones((self.size, self.size), dtype=np.float64)
        b = np.zeros((self.size, self.size), dtype=np.float64)

        noise = self.rng.random((self.size, self.size))
        b[noise < 0.015] = 1.0
        a += 0.04 * self.rng.random((self.size, self.size))
        b += 0.04 * self.rng.random((self.size, self.size))

        return np.clip(a, 0.0, 1.0), np.clip(b, 0.0, 1.0)

    def step(self):
        laplace_a = convolve2d(self.a, KERNEL, mode="same", boundary="wrap")
        laplace_b = convolve2d(self.b, KERNEL, mode="same", boundary="wrap")
        reaction = self.a * self.b * self.b

        self.a += self.da * laplace_a - reaction + self.feed * (1.0 - self.a)
        self.b += self.db * laplace_b + reaction - (self.kill + self.feed) * self.b

        np.clip(self.a, 0.0, 1.0, out=self.a)
        np.clip(self.b, 0.0, 1.0, out=self.b)

    def build_ui(self):
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.manager.set_window_title("Reaction-Diffusion Interativo")
        self.fig.subplots_adjust(left=0.07, right=0.86, bottom=0.28, top=0.93)

        self.image = self.ax.imshow(self.b, cmap="magma", vmin=0, vmax=1, interpolation="bilinear")
        self.ax.set_title("Reaction-Diffusion Gray-Scott", fontsize=14)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        colorbar = self.fig.colorbar(self.image, ax=self.ax, fraction=0.045, pad=0.03)
        colorbar.set_label("Concentracao B")

        self.feed_slider = self.add_slider("Feed", 0.01, 0.08, self.feed, 0.22)
        self.kill_slider = self.add_slider("Kill", 0.035, 0.08, self.kill, 0.17)
        self.da_slider = self.add_slider("Difusao A", 0.01, 1.0, self.da, 0.12)
        self.db_slider = self.add_slider("Difusao B", 0.01, 1.0, self.db, 0.07)

        for slider in (self.feed_slider, self.kill_slider, self.da_slider, self.db_slider):
            slider.on_changed(self.on_slider_change)

        self.play_button = self.add_button("Pausar", [0.07, 0.015, 0.10, 0.035], self.toggle_running)
        self.reset_button = self.add_button("Reset", [0.19, 0.015, 0.10, 0.035], self.reset)

        self.preset_buttons = []
        x = 0.43
        for name in PRESETS:
            self.preset_buttons.append(
                self.add_button(name, [x, 0.015, 0.09, 0.035], lambda _event, preset=name: self.apply_preset(preset))
            )
            x += 0.10

        self.animation = FuncAnimation(self.fig, self.update, interval=25, blit=False, cache_frame_data=False)

    def add_slider(self, label, minimum, maximum, value, bottom):
        axis = self.fig.add_axes([0.18, bottom, 0.58, 0.025])
        return Slider(axis, label, minimum, maximum, valinit=value)

    def add_button(self, label, rect, callback):
        axis = self.fig.add_axes(rect)
        button = Button(axis, label)
        button.on_clicked(callback)
        return button

    def on_slider_change(self, _value):
        self.feed = self.feed_slider.val
        self.kill = self.kill_slider.val
        self.da = self.da_slider.val
        self.db = self.db_slider.val
        self.active_preset = "Manual"

    def apply_preset(self, preset):
        values = PRESETS[preset]
        self.active_preset = preset
        self.feed_slider.set_val(values["feed"])
        self.kill_slider.set_val(values["kill"])
        self.da_slider.set_val(values["da"])
        self.db_slider.set_val(values["db"])
        self.reset()

    def toggle_running(self, _event):
        self.running = not self.running
        self.play_button.label.set_text("Pausar" if self.running else "Rodar")

    def reset(self, _event=None):
        self.rng = np.random.default_rng(self.seed)
        self.a, self.b = self.make_initial_state()
        self.image.set_data(self.b)
        self.fig.canvas.draw_idle()

    def update(self, _frame):
        if self.running:
            for _ in range(self.steps_per_frame):
                self.step()
            self.image.set_data(self.b)
            self.ax.set_xlabel(
                f"{self.active_preset} | feed={self.feed:.4f} | kill={self.kill:.4f} | "
                f"Da={self.da:.3f} | Db={self.db:.3f}"
            )
        return (self.image,)

    def show(self):
        plt.show()


if __name__ == "__main__":
    ReactionDiffusionApp().show()
