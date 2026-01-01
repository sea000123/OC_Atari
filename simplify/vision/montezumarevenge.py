from game_object import GameObject
import cv2
import numpy as np

MAX_NB_OBJECTS = {'Player': 1, 'Key': 1, 'Amulet': 1, 'Sword': 1, 'Torch': 1, 'Ruby': 3, 'Skull': 2, 'Spider': 1, 'Snake': 2,
                  'Barrier': 2, 'Beam': 8, 'Rope': 2, 'Wall': 4, 'Ladder': 3, 'Platform': 7, 'Disappearing_Platform': 12, 'Conveyer_Belt': 2, 'Key_HUD': 4, 'Amulet_HUD': 1, 'Torch_HUD': 1, 'Sword_HUD': 2}
MAX_NB_OBJECTS_HUD = {'Player': 1, 'Skull': 2, 'Spider': 1, 'Snake': 2, 'Key': 1, 'Amulet': 1, 'Torch': 1, 'Sword': 1,
                      'Barrier': 2, 'Beam': 8, 'Rope': 2, 'Ruby': 3, 'Wall': 4, 'Ladder': 2, 'Platform': 7, 'Disappearing_Platform': 12, 'Conveyer_Belt': 2, 'Key_HUD': 4, 'Amulet_HUD': 1, 'Torch_HUD': 1, 'Sword_HUD': 2,
                      'Score': 6, 'Life': 5}
def assert_in(observed, expected, tol):
    """
    Asserts if the observed point is equal to the expected one with a given tolerance.
    True if ||observed - expected|| <= tol, with || the maximum over the two dimensions.

    :param observed: The observed value point (e.g. (x,y), (w,h))
    :type observed: (int, int)
    :param expected: The expected value point (also (x,y), (w,h))
    :type expected: (int, int)
    :param tol: A given tolerance.
    :type tol: int or (int, int)
    
    :return: True if points within the tolerance
    :rtype: bool
    """
    if type(tol) is int:
        tol = (tol, tol)
    return np.all([expected[i] + tol[i] >= observed[i] >= expected[i] - tol[i] for i in range(2)])

def iou(bb, gt_bb):
    """
    Computes the intersection over union between two bounding boxes. 
    |iou_image|

    :param bb: The bouding box of the detected object in (x, y, w, h) format
    :type bb: (int, int, int, int)
    :param gt_bb: The ground truth bouding box
    :type gt_bb: (int, int, int, int)
    """
    inner_width = min(bb[1] + bb[3], gt_bb[1] + gt_bb[3]) - max(bb[1], gt_bb[1])
    inner_height = min(bb[0] + bb[2], gt_bb[0] + gt_bb[2]) - max(bb[0], gt_bb[0])
    if inner_width < 0 or inner_height < 0:
        return 0
    # bb_height, bb_width = bb[1] - bb[0], bb[3] - bb[2]
    intersection = inner_height * inner_width
    return intersection / ((bb[3] * bb[2]) + (gt_bb[3] * gt_bb[2]) - intersection)


def _merge_close_contours_iter(contours, closing_dist):
    merged_contours = []
    one_merge = False   # at least one merge during last iteration
    while contours:
        x, y, w, h = contours.pop(0)
        merged = False
        for i, (mx, my, mw, mh) in enumerate(merged_contours):
            # Calculate distance between bounding boxes
            c1x, c1y = x + w / 2, y + h / 2
            c2x, c2y = mx + mw / 2, my + mh / 2
            dx, dy = abs(c1x - c2x), abs(c1y - c2y)
            dx, dy = max(0, dx - (w+mw)/2), max(0, dy - (h+mh)/2)
            distance = dx + dy # Manhattan distance

            if distance < closing_dist:
                # Merge the boxes
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x + w, mx + mw) - new_x
                new_h = max(y + h, my + mh) - new_y
                merged_contours[i] = (new_x, new_y, new_w, new_h)
                merged = True
                one_merge = True
                break
        if not merged:
            merged_contours.append((x, y, w, h))
    return merged_contours, one_merge


def merge_close_contours(contours, closing_dist):
    """
    Merges the close contours into one bounding box.

    :param contours: The list of bounding boxes to merge
    :type contours: list of (int, int, int, int)
    :param closing_dist: The closing distance, for the under which two (or more) instances are merged \
    into one bounding box.
    :type closing_dist: int

    :return: a list of tuple boxing boxes
    :rtype: list of (int, int, int)
    """
    merged_contours, one_merge = _merge_close_contours_iter(contours, closing_dist)
    while one_merge:
        merged_contours, one_merge = _merge_close_contours_iter(merged_contours, closing_dist)
    return merged_contours
def find_objects(image, color, size=None, tol_s=10,
                 position=None, tol_p=2, min_distance=10,
                 closing_active=True, closing_dist=3,
                 minx=0, miny=0, maxx=160, maxy=210):
    """
    Finds the single colored objects in the image.

    :param image: The image to mark the point on
    :type image: np.array
    :param color: The color of the object
    :type color: list of (int, int, int)
    :param size: presupposed size of the targeted object (to detect)
    :type size: int or (int, int)
    :param tol_s: tolerance on the presupposed size of the targeted object
    :type tol_s: int or (int, int)
    :param position: presupposed position of the targeted object (to detect)
    :type position: int or (int, int)
    :param tol_p: tolerance on the presupposed position of the targeted object
    :type tol_p: int or (int, int)
    :param min_distance: the minimum distance to an existing object to be considered a new object
    :type min_distance: int
    :param closing_active: If true, gathers in one bounding box the instances that are less than \
    `closing_dist` away.
    :type closing_active: bool
    :param closing_dist: The closing distance, for the under which two (or more) instances are merged \
    into one bounding box.
    :type closing_dist: int
    :param minx: minimum x position where the object can be located
    :type minx: int
    :param miny: minimum y position where the object can be located
    :type miny: int
    :param maxx: maximum x position where the object can be located
    :type maxx: int
    :param maxy: maximum y position where the object can be located
    :type maxy: int

    :return: a list of tuple boxing boxes
    :rtype: list of (int, int, int)
    """
    mask = cv2.inRange(image[miny:maxy, minx:maxx, :], np.array(color), np.array(color))
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 1)
    contours = [cv2.boundingRect(cnt) for cnt in contours]
    if closing_active and len(contours) > 1:
        contours = merge_close_contours(contours, closing_dist)
    detected = []
    for cnt in contours:
        x, y, w, h = cnt
        x, y = x + minx, y + miny  # compensing cuttoff
        if size:
            if not assert_in((w, h), size, tol_s):
                continue
        if position:
            if not assert_in((x, y), position, tol_p):
                continue
        if min_distance:
            too_close = False
            for det in detected:
                if iou(det, (x, y, w, h)) > 0.05:
                    too_close = True
                    break
            if too_close:
                continue
        detected.append((x, y, w, h))
    return detected

def find_mc_objects(image, colors, size=None, tol_s=10, position=None, tol_p=2, 
                    min_distance=10, closing_active=True, closing_dist=3,
                    minx=0, miny=0, maxx=160, maxy=210, all_colors=True):
    """
    Finds the multicolors objects in the image. 
        
    This functions is used to detect object in e.g. Atlantis (depicted bellow). 
    
    |atlantis_image|

    :param image: The image to mark the point on
    :type image: np.array
    :param colors: The colors of the object
    :type colors: list of (int, int, int)
    :param size: presupposed size of the targeted object (to detect)
    :type size: int or (int, int)
    :param tol_s: tolerance on the presupposed size of the targeted object
    :type tol_s: int or (int, int)
    :param size: presupposed size of the targeted object (to detect)
    :type size: int or (int, int)
    :param tol_s: tolerance on the presupposed size of the targeted object
    :type tol_s: int or (int, int)
    :param position: presupposed position of the targeted object (to detect)
    :type position: int or (int, int)
    :param tol_p: tolerance on the presupposed position of the targeted object
    :type tol_p: int or (int, int)
    :param min_distance: tolerance on the presupposed position of the targeted object
    :type min_distance: int
    :param closing_active: If true, gathers in one bounding box the instances that are less than \
    `closing_dist` 
    :type closing_active: bool
    :param closing_dist: The closing distance, for the under which two (or more) instances are merged \
    into one bounding box.
    :type closing_dist: int
    :param minx: minimum x position where the object can be located
    :type minx: int
    :param miny: minimum y position where the object can be located
    :type miny: int
    :param maxx: maximum x position where the object can be located
    :type maxx: int
    :param maxy: maximum y position where the object can be located
    :type maxy: int
    :param all_colors: If ``True``, only return the object if every given color in `colors` is present in the image
    :type all_colors: bool


    :return: a list of tuple boxing boxes
    :rtype: list of (int, int, int, int)
    """
    masks = [cv2.inRange(image[miny:maxy, minx:maxx, :],
                         np.array(color), np.array(color)) for color in colors]
    if all_colors: 
        for mask in masks:
            if mask.max() == 0: # if any color is missing from the whole image
                return []
    mask = sum(masks)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 1)
    contours = [cv2.boundingRect(cnt) for cnt in contours]
    if closing_active and len(contours) > 1:
        contours = merge_close_contours(contours, closing_dist)
    detected = []
    for cnt in contours:
        x, y, w, h = cnt
        x, y = x + minx, y + miny  # compensing cuttoff
        if size:
            if not assert_in((w, h), size, tol_s):
                continue
        if position:
            if not assert_in((x, y), position, tol_p):
                continue
        if min_distance:
            too_close = False
            for det in detected:
                if iou(det, (x, y, w, h)) > 0.05:
                    too_close = True
                    break
            if too_close:
                continue
        if all_colors: # all colors are present in this specific object
            all_contained = True
            for k in range(len(masks)):
                contained = False
                for i in range(w):
                    for j in range(h):
                        try:
                            if masks[k][y+j-miny][x+i-minx]:
                                contained = True
                                break
                        except:
                            continue

                    if contained:
                        break
                if not contained:
                    all_contained = False
                    break
            if all_contained:
                detected.append((x, y, w, h))
        else:
            detected.append((x, y, w, h))
    return detected


objects_colors = {'white': [236, 236, 236], 'yellow': [232, 204, 99], 'orange': [213, 130, 74],
                  'blue': [101, 111, 228], 'green': [92, 186, 92], 'yellow_2': [204, 216, 110],
                  'white_2': [214, 214, 214], 'white_3': [192, 192, 192],
                  'playercolors': [[228, 111, 111], [200, 72, 72], [210, 182, 86]],
                  'playercolors2': [[240, 128, 128], [200, 72, 72], [210, 182, 86]],
                  'lifecolors': [[200, 72, 72], [210, 182, 86]]}


class Player(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [228, 111, 111]


#  ---- enemies -----
class Skull(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [236, 236, 236]


class Spider(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [92, 186, 92]


class Snake(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [192, 192, 192]


#  ---- collectable objects -----
class Key(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [232, 204, 99]


class Amulet(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [210, 182, 86]


class Torch(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [204, 216, 110]


class Sword(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [214, 214, 214]


class Ruby(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [213, 130, 74]


#  ---- others -----
class Barrier(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [232, 204, 99]


class Beam(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [101, 111, 228]


class Rope(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [232, 204, 99]


#  ---- HUD -----
class Life(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [210, 182, 86]


class Score(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [236, 236, 236]


class Torch_HUD(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [232, 204, 99]


class Sword_HUD(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [232, 204, 99]


class Key_HUD(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [232, 204, 99]


class Amulet_HUD(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = [232, 204, 99]


def _detect_objects(objects, obs, hud=False):
    objects.clear()

    players = find_mc_objects(obs, objects_colors['playercolors'], miny=25)
    for bb in players:
        objects.append(Player(*bb))

    players2 = find_mc_objects(obs, objects_colors['playercolors2'], miny=25)
    for bb in players2:
        objects.append(Player(*bb))

    skull = find_objects(obs, objects_colors['white'], miny=25, size=(13, 7))
    for bb in skull:
        objects.append(Skull(*bb))

    spider = find_objects(obs, objects_colors['green'], miny=25)
    for bb in spider:
        objects.append(Spider(*bb))

    snek = find_objects(obs, objects_colors['white_3'], miny=25, size=(13, 7))
    for bb in snek:
        objects.append(Snake(*bb))

    torch = find_objects(
        obs, objects_colors['yellow_2'], size=(6, 13), miny=48)
    for bb in torch:
        objects.append(Torch(*bb))

    sword = find_objects(obs, objects_colors['white_2'], size=(7, 15), miny=48)
    for bb in sword:
        objects.append(Sword(*bb))

    rope = find_objects(obs, objects_colors['yellow'], size=(1, 39), tol_s=2)
    for bb in rope:
        objects.append(Rope(*bb))

    rope2 = find_objects(obs, objects_colors['yellow'], size=(1, 51), tol_s=4)
    for bb in rope2:
        objects.append(Rope(*bb))

    rope_w = find_objects(obs, objects_colors['white'], miny=25, size=(1, 25))
    for bb in rope_w:
        r = Rope(*bb)
        r.rgb = objects_colors['white']
        objects.append(r)

    key = find_objects(obs, objects_colors['yellow'], size=(7, 15), miny=48)
    for bb in key:
        objects.append(Key(*bb))

    amulet = find_mc_objects(
        obs, objects_colors['playercolors'], miny=25, size=(6, 15), tol_s=2)
    for bb in amulet:
        objects.append(Amulet(*bb))

    barrier = find_objects(
        obs, objects_colors['yellow'], size=(4, 37), tol_s=2)
    for bb in barrier:
        objects.append(Barrier(*bb))

    beam = find_objects(
        obs, objects_colors['blue'], minx=10, maxx=150, closing_dist=4)
    for bb in beam:
        objects.append(Beam(*bb))

    ruby = find_objects(obs, objects_colors['orange'], size=(7, 12), tol_s=2)
    for bb in ruby:
        objects.append(Ruby(*bb))

    if hud:

        torch_h = find_objects(obs, objects_colors['yellow'], size=(
            6, 13), tol_s=0, maxy=48, closing_dist=1)
        for bb in torch_h:
            objects.append(Torch_HUD(*bb))

        sword_h = find_objects(obs, objects_colors['yellow'], size=(
            6, 15), tol_s=0, maxy=48, closing_dist=1,)
        for bb in sword_h:
            objects.append(Sword_HUD(*bb))

        key_h = find_objects(obs, objects_colors['yellow'], size=(
            7, 15), tol_s=0, maxy=48, closing_dist=1)
        for bb in key_h:
            objects.append(Key_HUD(*bb))

        amulet_h = find_objects(obs, objects_colors['yellow'], size=(
            5, 15), tol_s=0, maxy=48, closing_dist=1)
        for bb in amulet_h:
            objects.append(Amulet_HUD(*bb))

        lifes = find_mc_objects(
            obs, objects_colors['lifecolors'], maxy=25, closing_dist=1)
        for bb in lifes:
            objects.append(Life(*bb))

        scores = find_objects(
            obs, objects_colors['white'], maxy=25, closing_dist=1)
        for bb in scores:
            objects.append(Score(*bb))
