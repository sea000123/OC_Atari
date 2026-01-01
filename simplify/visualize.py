import sys
from termcolor import colored
import numpy as np
from game_object import NoObject
import pygame


def detect_objects_vision(objects, obs, game_name, hud):
    p_module = __name__.split('.')[:-1] + ['vision']  +[game_name.lower()]
    game_module = '.'.join(p_module)
    for obj in objects:  # saving the previsous positions
        if obj:
            obj._save_prev()
    try:
        mod = sys.modules[game_module]
        mod._detect_objects
    except KeyError:
        raise NotImplementedError(
            colored(f"Game module does not exist: {game_module}", "red"))
    except AttributeError:
        raise NotImplementedError(
            colored(f"_detect_objects not implemented for game: {game_name}", "red"))
    return mod._detect_objects(objects, obs, hud)

def mark_bb(image_array, bb, color=(255, 0, 0), surround=True):
    """
    Marks a bounding box on the image.

    :param image_array: The image to mark the point on
    :type image_array: RGB np.array
    :param bb: The bouding box of the detected object in (x, y, w, h) format
    :type bb: (int, int, int, int)
    :param color: The rgb values of the point.
    :type color: (int, int, int)
    :param surround: If ``True``, place the bouding box with an offset of 1 pixel to surround the object
    :type surround: bool
    """
    x, y, w, h = bb
    if surround:
        if x > 0:
            x, w = bb[0] - 1, bb[2] + 1
        else:
            x, w = bb[0], bb[2]
        if y > 0:
            y, h = bb[1] - 1, bb[3] + 1
        else:
            y, h = bb[1], bb[3]
    bottom = min(208, y + h)
    right = min(158, x + w)
    try:
        image_array[y:bottom + 1, x] = color
        image_array[y:bottom + 1, right] = color
        image_array[y, x:right + 1] = color
        image_array[bottom, x:right + 1] = color
    except IndexError:
        pass

def to_rgba(color):
    return np.concatenate([np.array(color)/255, [.7]])

def get_max_objects(game_name, hud):
    p_module = __name__.split('.')[:-1] + ['vision']  +[game_name.lower()]
    game_module = '.'.join(p_module)
    import importlib
    try:
        mod = importlib.import_module(game_module)
    except ModuleNotFoundError:
        raise KeyError(f"Game module does not exist: {game_module}")
    if hud:
        return mod.MAX_NB_OBJECTS_HUD
    return mod.MAX_NB_OBJECTS
    try:
        mod = sys.modules[game_module]
        if hud:
            return mod.MAX_NB_OBJECTS_HUD
        return mod.MAX_NB_OBJECTS
    except KeyError as err:
        raise KeyError(f"Game module does not exist: {game_module}")
    except AttributeError as err:
        raise AttributeError(
            f"MAX_NB_OBJECTS_HUD not implemented for game: {game_name}")

def get_class_dict(game_name):
    p_module = __name__.split('.')[:-1] + ['vision']  +[game_name.lower()]
    game_module = '.'.join(p_module)
    try:
        mod = sys.modules[game_module]
        classes = {}
        for name, number in mod.MAX_NB_OBJECTS_HUD.items():
            classes[name] = getattr(mod, name)
        return classes
    except KeyError as err:
        raise KeyError(f"Game module does not exist: {game_module}")
    except AttributeError as err:
        raise AttributeError(
            f"MAX_NB_OBJECTS_HUD not implemented for game: {game_name}")
    
# parses MAX_NB* dicts, returns default init list of objects
def instantiate_max_objects(game_name, max_obj_dict):
    objects = []
    p_module = __name__.split('.')[:-1] + ['vision']  +[game_name.lower()]
    game_module = '.'.join(p_module)
    try:
        mod = sys.modules[game_module]
    except KeyError as err:
        return []
    for k, v in max_obj_dict.items():
        for _ in range(0, v):
            objects.append(getattr(mod, k)())
    return objects

def get_object_state_size(game_name, hud):
    max_obj = get_max_objects(game_name, hud)
    iobjects = instantiate_max_objects(game_name, max_obj)
    nsrepr_tot = [o._nsrepr for o in iobjects]
    return sum(map(len, nsrepr_tot))

def use_vision_objects(objects, game_module):
    """
    replaces ram objects with their equivalent vision objects
    """
    game_module_vision = game_module.replace('ram', 'vision')
    mod = sys.modules[game_module_vision]
    for i, obj in enumerate(objects):
        if obj:  # skip None objects
            objects[i] = getattr(mod, objects[i].category)(*obj.xywh)
        else:
            objects[i] = NoObject()
    return objects

def init_objects(game_name, hud, vision=False):
    p_module = __name__.split('.')[:-1] + ['vision']  +[game_name.lower()]
    game_module = '.'.join(p_module)
    try:
        mod = sys.modules[game_module]
        if vision:
            return use_vision_objects(mod._init_objects_ram(hud), game_module)
        return mod._init_objects_ram(hud)
    except KeyError as err:
        raise KeyError(f"Game module does not exist: {game_module}")
    except AttributeError as err:
        raise AttributeError(
            f"init_objects not implemented for game: {game_name}")

ROT_MATRIX = np.array([[0, -1], [1, 0]])
def draw_arrow(surface: pygame.Surface, start_pos, end_pos,
               tip_length: int = 6, tip_width: int = 6, **kwargs):
    # start_pos: (float, float), end_pos: (float, float),
    start_pos = np.asarray(start_pos)
    end_pos = np.asarray(end_pos)

    # Arrow body
    pygame.draw.line(surface, start_pos=start_pos, end_pos=end_pos, **kwargs)

    # Arrow tip
    arrow_dir = end_pos - start_pos
    arrow_dir_norm = arrow_dir / np.linalg.norm(arrow_dir)
    tip_anchor = end_pos - tip_length * arrow_dir_norm

    left_tip_end = tip_anchor + tip_width / 2 * \
        np.matmul(ROT_MATRIX, arrow_dir_norm)
    right_tip_end = tip_anchor - tip_width / \
        2 * np.matmul(ROT_MATRIX, arrow_dir_norm)

    pygame.draw.line(surface, start_pos=left_tip_end,
                     end_pos=end_pos, **kwargs)
    pygame.draw.line(surface, start_pos=right_tip_end,
                     end_pos=end_pos, **kwargs)
    
def draw_label(surface: pygame.Surface, text: str, position, font: pygame.font.SysFont):
    """Renders a framed label text to a pygame surface.
    position: (int, int).
    """
    text = font.render(text, True, (255, 255, 255), None)
    text_rect = text.get_rect()

    frame_rect = text_rect.copy()
    frame_rect.topleft = position
    frame_rect.w += 5
    frame_rect.h += 6

    frame_surface = pygame.Surface((frame_rect.w, frame_rect.h))
    frame_surface.set_alpha(80)  # make transparent

    # Draw label background
    frame_surface.fill((0, 0, 0))
    surface.blit(frame_surface, position)

    # Draw text
    text_rect.topleft = position[0] + 3, position[1] + 3
    surface.blit(text, text_rect)
