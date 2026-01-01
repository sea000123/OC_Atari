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
from visualize import (
    detect_objects_vision, get_class_dict, get_max_objects,
    get_object_state_size, init_objects, draw_label, draw_arrow)
from gymnasium.error import NameNotFound


class Montezuma(gym.Env):
    """
    固定配置：
    - env_name = "MontezumaRevenge-v5"
    - mode = "vision"
    - obs_mode = "obj"
    只保留视觉识别 + 对象状态堆叠，不包含 RAM / both / dqn / ori 相关逻辑
    """

    def __init__(self, env_name="ALE/MontezumaRevenge-v5", mode="vision", hud=False,
                 obs_mode="obj", render_mode=None, render_oc_overlay=False,
                 buffer_window_size=4, create_buffer_stacks=["obj"], *args, **kwargs):

        # 固定参数（即使外部传入别的，也强制使用固定值）
        env_name = "ALE/MontezumaRevenge-v5"
        mode = "vision"
        obs_mode = "obj"

        game_name = env_name.split("/")[1].split("-")[0].split("No")[0].split("Deterministic")[0] \
            if "ALE/" in env_name else env_name.split("-")[0].split("No")[0].split("Deterministic")[0]

        self.env_name = env_name
        self.game_name = game_name
        self.mode = mode
        self.obs_mode = obs_mode
        self.hud = hud

        gym_render_mode = "rgb_array" if render_oc_overlay else render_mode
        self.buffer_window_size = buffer_window_size

        try:
            self._env = gym.make(env_name, render_mode=gym_render_mode, *args, **kwargs)
        except NameNotFound:
            cenv_name = f"ALE/{env_name}-v5"
            self._env = gym.make(cenv_name, render_mode=gym_render_mode, *args, **kwargs)
            self.env_name = cenv_name

        # ========== obj-only observation space ==========
        self.max_objects_per_cat = get_max_objects(self.game_name, self.hud)
        self._class_dict = get_class_dict(self.game_name)
        self._slots = [self._class_dict[c]() for c, n in self.max_objects_per_cat.items() for _ in range(n)]
        self._ns_state = np.zeros(sum([len(o._nsrepr) for o in self._slots]))
        self.ns_meaning = [f"{o.category} ({o._ns_meaning})" for o in self._slots]

        self._env.observation_space = gym.spaces.Box(
            0, 255.0, (self.buffer_window_size, get_object_state_size(self.game_name, self.hud))
        )

        # rendering
        self.render_mode = render_mode
        self.render_oc_overlay = render_oc_overlay
        self.rendering_initialized = False

        # buffers（只保留 obj stack）
        self.create_ns_stack = True
        self._state_buffer_ns = deque([], maxlen=self.buffer_window_size)

        # action / ale
        self.action_space = self._env.action_space
        self._ale = self._env.unwrapped.ale
        self.ale = self._ale

        # inherit env attrs
        for meth in dir(self._env):
            if meth not in dir(self):
                try:
                    setattr(self, meth, getattr(self._env, meth))
                except AttributeError:
                    pass

        # ========== vision-only object detection ==========
        global init_objects
        self.detect_objects = self._detect_objects_vision
        self.objects = init_objects(self.game_name, self.hud, vision=True)

    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = self._env.step(*args, **kwargs)
        self.detect_objects()
        self._fill_buffer()
        obs = np.array(self._state_buffer_ns)
        return obs, reward, truncated, terminated, info

    # ====== 保留函数名，但 RAM 检测已删除：保留占位符避免外部报错 ======
    def _detect_objects_ram(self):
        raise NotImplementedError("RAM object detection removed: mode fixed to vision.")

    def _detect_objects_both(self):
        raise NotImplementedError("Both-mode removed: mode fixed to vision.")

    def _detect_objects_vision(self):
        """
        Detect objects using vision-based extraction.
        不简化该部分：保持原调用方式
        """
        detect_objects_vision(
            self.objects,
            self._env.env.unwrapped.ale.getScreenRGB(),
            self.game_name,
            self.hud
        )

    def _reset_buffer(self):
        for _ in range(self.buffer_window_size):
            self._fill_buffer()

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        self.objects = init_objects(self.game_name, self.hud, vision=True)
        self.detect_objects()
        self._reset_buffer()
        obs = np.array(self._state_buffer_ns)
        return obs, info

    def _fill_buffer(self):
        self._state_buffer_ns.append(self.ns_state)

    window: pygame.Surface = None
    clock: pygame.time.Clock = None

    def _initialize_rendering(self, sample_image):
        assert sample_image is not None
        pygame.init()
        if self.render_mode == "human":
            pygame.display.set_caption(self.game_name)
        self.image_size = (sample_image.shape[1], sample_image.shape[0])
        self.window_size = (
            sample_image.shape[1] * UPSCALE_FACTOR,
            sample_image.shape[0] * UPSCALE_FACTOR
        )
        self.label_font = pygame.font.SysFont('Pixel12x10', 16)
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

            pygame.draw.rect(
                overlay_surface,
                color=game_object.rgb,
                rect=(x, y, w, h),
                width=2
            )
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
                    width=2
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

