# The Montezuma environment provides an interface to interact 
# with Atari 2600 games through Gymnasium, enabling object tracking 
# and analysis. This environment extends the functionality of 
# traditional Atari environments by incorporating different 
# object detection modes (RAM, vision, or both) and supports 
# enhanced observation spaces for advanced tasks like reinforcement learning.
from collections import deque
from itertools import chain
from game_object import ValueObject
import numpy as np
import gymnasium as gym
import cv2
import pygame
UPSCALE_FACTOR = 6
from vision import (
    detect_objects_vision,mark_bb, to_rgba, get_class_dict, get_max_objects,
    get_object_state_size, init_objects, draw_label, draw_arrow)


class Montezuma(gym.Env):
    """
    NOTE (simplified):
    - env_name fixed to "MontezumaRevenge-v5"
    - mode fixed to "vision"
    - obs_mode fixed to "obj"
    """

    def __init__(
        self,
        hud=False,
        render_mode=None,
        render_oc_overlay=False,
        buffer_window_size=4,
        *args,
        **kwargs,
    ):
        # --- fixed configuration (as requested) ---
        self.env_name = "MontezumaRevenge-v5"
        self.mode = "vision"
        self.obs_mode = "obj"

        # --- keep user-configurable where still meaningful ---
        self.hud = hud
        self.render_mode = render_mode
        self.render_oc_overlay = render_oc_overlay
        self.buffer_window_size = buffer_window_size

        # Determine game name (kept minimal & robust)
        self.game_name = self.env_name.split("/")[1].split("-")[0] if "ALE/" in self.env_name else self.env_name.split("-")[0]

        # Create base env (no NameNotFound fallback needed since fixed env_name)
        gym_render_mode = "rgb_array" if render_oc_overlay else render_mode
        self._env = gym.make(self.env_name, render_mode=gym_render_mode, *args, **kwargs)

        # --- object-centric (obj) only ---
        self.max_objects_per_cat = get_max_objects(self.game_name, self.hud)
        self._class_dict = get_class_dict(self.game_name)
        self._slots = [
            self._class_dict[c]() for c, n in self.max_objects_per_cat.items() for _ in range(n)
        ]
        self._ns_state = np.zeros(sum(len(o._nsrepr) for o in self._slots))
        self.ns_meaning = [f"{o.category} ({o._ns_meaning})" for o in self._slots]

        # observation space: (buffer_window_size, oc_state_size)
        self._env.observation_space = gym.spaces.Box(
            0, 255.0, (self.buffer_window_size, get_object_state_size(self.game_name, self.hud))
        )

        # --- buffers: only what we actually use ---
        # keep rgb stack only if you need rendering / explanations
        self.create_rgb_stack = True
        self.create_ns_stack = True
        self.create_dqn_stack = False

        self._state_buffer_rgb = deque([], maxlen=self.buffer_window_size) if self.create_rgb_stack else None
        self._state_buffer_ns = deque([], maxlen=self.buffer_window_size) if self.create_ns_stack else None
        self._state_buffer_dqn = None

        # action space + ALE
        self.action_space = self._env.action_space
        self._ale = self._env.unwrapped.ale
        self.ale = self._ale

        # inherit attributes from base env (kept, but minimal)
        for meth in dir(self._env):
            if meth not in dir(self):
                try:
                    setattr(self, meth, getattr(self._env, meth))
                except AttributeError:
                    pass

        # --- fixed to vision detection ---
        self.detect_objects = self._detect_objects_vision
        self.objects = init_objects(self.game_name, self.hud, vision=True)

        # rendering init flags
        self.rendering_initialized = False
        self.window: pygame.Surface = None
        self.clock: pygame.time.Clock = None

    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = self._env.step(*args, **kwargs)

        # vision detection (unchanged path)
        self.detect_objects()

        # fill stacks
        self._fill_buffer()

        # obj-only observation
        obs = np.array(self._state_buffer_ns)
        return obs, reward, truncated, terminated, info

    def _detect_objects_vision(self):
        """
        Detect objects using vision-based extraction.

        (视觉识别部分不简化：仍然使用 detect_objects_vision + getScreenRGB 的调用方式)
        """
        detect_objects_vision(
            self.objects,
            self._env.env.unwrapped.ale.getScreenRGB(),
            self.game_name,
            self.hud,
        )  # type: ignore


    def _reset_buffer(self):
        for _ in range(self.buffer_window_size):
            self._fill_buffer()

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)

        # re-init objects (vision fixed)
        self.objects = init_objects(self.game_name, self.hud, vision=True)

        self.detect_objects()
        self._reset_buffer()

        obs = np.array(self._state_buffer_ns)
        return obs, info

    def _fill_buffer(self):
        if self.create_rgb_stack:
            self._state_buffer_rgb.append(self.getScreenRGB())
        if self.create_ns_stack:
            self._state_buffer_ns.append(self.ns_state)

    def _initialize_rendering(self, sample_image):
        assert sample_image is not None
        pygame.init()
        if self.render_mode == "human":
            pygame.display.set_caption(self.game_name)

        self.image_size = (sample_image.shape[1], sample_image.shape[0])
        self.window_size = (sample_image.shape[1] * UPSCALE_FACTOR, sample_image.shape[0] * UPSCALE_FACTOR)
        self.label_font = pygame.font.SysFont("Pixel12x10", 16)

        if self.render_mode == "human":
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        else:
            self.window = pygame.Surface(self.window_size)

        self.rendering_initialized = True

    def render(self, image=None):
        if image is None:
            image = self._env.render()

        if not self.render_oc_overlay:
            if self.rendering_initialized:
                return image.swapaxes(0, 1).repeat(UPSCALE_FACTOR, axis=0).repeat(UPSCALE_FACTOR, axis=1)
            return image

        if not self.rendering_initialized:
            self._initialize_rendering(image)

        image = np.transpose(image, (1, 0, 2))
        image_surface = pygame.Surface(self.image_size)
        pygame.pixelcopy.array_to_surface(image_surface, image)
        upscaled_image = pygame.transform.scale(image_surface, self.window_size)
        self.window.blit(upscaled_image, (0, 0))

        overlay_surface = pygame.Surface(self.window_size)
        overlay_surface.set_colorkey((0, 0, 0))

        for game_object in self.objects:
            x, y = game_object.xy
            w, h = game_object.wh

            if x == np.nan:
                continue

            dx, dy = game_object.dx * UPSCALE_FACTOR, game_object.dy * UPSCALE_FACTOR
            x, y, w, h = x * UPSCALE_FACTOR, y * UPSCALE_FACTOR, w * UPSCALE_FACTOR, h * UPSCALE_FACTOR
            x_c, y_c = x + w // 2, y + h // 2

            pygame.draw.rect(overlay_surface, color=game_object.rgb, rect=(x, y, w, h), width=2)

            label = game_object.category
            if isinstance(game_object, ValueObject):
                label += f" ({game_object.value})"
            draw_label(self.window, label, position=(x, y + h + 4), font=self.label_font)

            if dx != 0 or dy != 0:
                draw_arrow(
                    overlay_surface,
                    start_pos=(float(x_c), float(y_c)),
                    end_pos=(x_c + 2 * dx, y_c + 2 * dy),
                    color=(100, 200, 255),
                    width=2,
                )

        self.window.blit(overlay_surface, (0, 0))

        if self.render_mode == "human":
            frameskip = self._env.unwrapped._frameskip if isinstance(self._env.unwrapped._frameskip, int) else 1
            self.clock.tick(60 // frameskip)
            pygame.display.flip()
            pygame.event.pump()
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def seed(self, seed, *args, **kwargs):
        self._env.seed(seed, *args, **kwargs)

    def getScreenRGB(self):
        return self._ale.getScreenRGB()

    @property
    def nb_actions(self):
        return self.action_space.n

    @property
    def get_rgb_state(self):
        return self._ale.getScreenRGB()

    def set_ram(self, target_ram_position, new_value):
        return self._env.unwrapped.ale.setRAM(target_ram_position, new_value)

    def get_ram(self):
        return self._ale.getRAM()

    def get_action_meanings(self):
        return self._env.env.env.get_action_meanings()

    def _get_obs(self):
        return self._env.env.env.unwrapped._get_obs()

    def detect_objects_both(self):
        # API compatibility
        self._detect_objects_vision()

    def _clone_state(self):
        return self._env.env.env.ale.cloneSystemState()

    def _restore_state(self, state):
        return self._env.env.env.ale.restoreSystemState(state)

    @property
    def ns_state(self):
        return list(chain.from_iterable([o._nsrepr for o in self.objects]))

    def render_explanations(self):
        rendered = np.zeros_like(self._state_buffer_rgb[0]).astype(float)
        coefs = [0.05, 0.1, 0.25, 0.6]
        for coef, state_i in zip(coefs, self._state_buffer_rgb):
            rendered += coef * state_i
        rendered = rendered.astype(int)

        for obj in self.objects:
            mark_bb(rendered, obj.xywh, color=obj.rgb)

        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        plt.imshow(rendered)
        rows, cells, colors = [], [], []
        columns = ["X, Y", "W, H", "R, G, B"]
        for obj in self.objects:
            rows.append(obj.category)
            cells.append([obj.xy, obj.wh, obj.rgb])
            colors.append(to_rgba(obj.rgb))

        t_height = 0.03 * len(rows)
        table = plt.table(
            cellText=cells,
            rowLabels=rows,
            rowColours=colors,
            colLabels=columns,
            colWidths=[.2, .2, .3],
            bbox=[0.1, 1.02, 0.8, t_height],
            loc="top",
        )
        table.set_fontsize(14)
        plt.subplots_adjust(top=0.8)
        plt.show()

    def aggregated_render(self, coefs=[0.05, 0.1, 0.25, 0.6]):
        rendered = np.zeros_like(self._state_buffer_rgb[0]).astype(float)
        for coef, state_i in zip(coefs, self._state_buffer_rgb):
            rendered += coef * state_i
        return rendered.astype(int)

    def get_keys_to_action(self):
        return self._env.unwrapped.get_keys_to_action()
