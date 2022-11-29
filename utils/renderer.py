"""
Parts of the code are taken from from https://github.com/akanazawa/hmr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from skimage.transform import resize
from torchvision.utils import make_grid

from models.smpl import get_smpl_faces
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight
import logging
logger = logging.getLogger(__name__)
# Rotate the points by a specified angle.
#  https://github.com/classner/up/blob/master/up_tools/camera.py
def rotateY(points, angle):
    """Rotate all points in a 2D array around the y axis."""
    ry = np.array([
        [np.cos(angle),     0.,     np.sin(angle)],
        [0.,                1.,     0.           ],
        [-np.sin(angle),    0.,     np.cos(angle)]
    ])
    return np.dot(points, ry)

def rotateX( points, angle ):
    """Rotate all points in a 2D array around the x axis."""
    rx = np.array([
        [1.,    0.,                 0.           ],
        [0.,    np.cos(angle),     -np.sin(angle)],
        [0.,    np.sin(angle),     np.cos(angle) ]
    ])
    return np.dot(points, rx)

def rotateZ( points, angle ):
    """Rotate all points in a 2D array around the z axis."""
    rz = np.array([
        [np.cos(angle),     -np.sin(angle),     0. ],
        [np.sin(angle),     np.cos(angle),      0. ],
        [0.,                0.,                 1. ]
    ])
    return np.dot(points, rz)


def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    else:
        print('Unknown skeleton!!')

    for child in xrange(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image

def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = np.array([0, 0, 255])
    margin = 15
    start_x = 5
    start_y = margin
    for key in sorted(content.keys()):
        text = "%s: %.2g" % (key, content[key])
        cv2.putText(image, text, (start_x, start_y), 0, 0.45, black)
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image


def visualize_reconstruction(img, img_size, gt_kp, vertices, pred_kp, camera, renderer, color='pink', focal_length=1000,mpjpe=999,mpjpe_pa=999):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    gt_vis = gt_kp[:, 2].astype(bool)
    loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
    debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2], "kpl": loss,"MPJPE":mpjpe,"MPJPE-PA":mpjpe_pa}
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices, camera_t=camera_t,
                               img=img, use_bg=True,
                               focal_length=focal_length,
                               body_color=color)
    rend_img = draw_text(rend_img, debug_text)

    # Draw skeleton
    gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * img_size
    pred_joint = ((pred_kp + 1) * 0.5) * img_size
    img_with_gt = draw_skeleton( img, gt_joint, draw_edges=False, vis=gt_vis)
    skel_img = draw_skeleton(img_with_gt, pred_joint)

    combined = np.hstack([skel_img, rend_img])

    return combined

class Renderer(object):
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=800, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'pink': [.9, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353] }
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1.,1.,1.]

        self.renderer.set(v=vertices, f=faces,
                          vc=color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r


class OpenDRenderer:
    def __init__(self, resolution=(224, 224), ratio=1):
        self.resolution = (resolution[0] * ratio, resolution[1] * ratio)
        self.ratio = ratio
        self.focal_length = 5000.
        self.K = np.array([[self.focal_length, 0., self.resolution[1] / 2.],
                           [0., self.focal_length, self.resolution[0] / 2.],
                           [0., 0., 1.]])
        self.colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            'purple': np.array([0.5, 0.5, 0.7]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }
        self.renderer = ColoredRenderer()
        self.faces = get_smpl_faces()

    def reset_res(self, resolution):
        self.resolution = (resolution[0] * self.ratio, resolution[1] * self.ratio)
        self.K = np.array([[self.focal_length, 0., self.resolution[1] / 2.],
                           [0., self.focal_length, self.resolution[0] / 2.],
                           [0., 0., 1.]])

    def __call__(self, verts, faces=None, color=None, color_type='white', R=None, mesh_filename=None,
                 img=np.zeros((256, 256, 3)), cam=np.array([1, 0, 0]),
                 rgba=False, addlight=True):
        '''Render mesh using OpenDR
        verts: shape - (V, 3)
        faces: shape - (F, 3)
        img: shape - (224, 224, 3), range - [0, 255] (np.uint8)
        axis: rotate along with X/Y/Z axis (by angle)
        R: rotation matrix (used to manipulate verts) shape - [3, 3]
        Return:
            rendered img: shape - (224, 224, 3), range - [0, 255] (np.uint8)
        '''
        ## Create OpenDR renderer
        rn = self.renderer
        h, w = self.resolution
        K = self.K

        f = np.array([K[0, 0], K[1, 1]])
        c = np.array([K[0, 2], K[1, 2]])

        if faces == None:
            faces = self.faces
        if len(cam) == 4:
            t = np.array([cam[2], cam[3], 2 * K[0, 0] / (w * cam[0] + 1e-9)])
        elif len(cam) == 3:
            t = np.array([cam[1], cam[2], 2 * K[0, 0] / (w * cam[0] + 1e-9)])

        rn.camera = ProjectPoints(rt=np.array([0, 0, 0]), t=t, f=f, c=c, k=np.zeros(5))
        rn.frustum = {'near': 1., 'far': 1000., 'width': w, 'height': h}

        albedo = np.ones_like(verts) * .9

        if color is not None:
            color0 = np.array(color)
            color1 = np.array(color)
            color2 = np.array(color)
        elif color_type == 'white':
            color0 = np.array([1., 1., 1.])
            color1 = np.array([1., 1., 1.])
            color2 = np.array([0.7, 0.7, 0.7])
            color = np.ones_like(verts) * self.colors_dict[color_type][None, :]
        else:
            color0 = self.colors_dict[color_type] * 1.2
            color1 = self.colors_dict[color_type] * 1.2
            color2 = self.colors_dict[color_type] * 1.2
            color = np.ones_like(verts) * self.colors_dict[color_type][None, :]

        # render_smpl = rn.r
        if R is not None:
            assert R.shape == (3, 3), "Shape of rotation matrix should be (3, 3)"
            verts = np.dot(verts, R)

        rn.set(v=verts, f=faces, vc=color, bgcolor=np.zeros(3))

        if addlight:
            yrot = np.radians(120)  # angle of lights
            # # 1. 1. 0.7
            rn.vc = LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([-200, -100, -100]), yrot),
                vc=albedo,
                light_color=color0)

            # Construct Left Light
            rn.vc += LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([800, 10, 300]), yrot),
                vc=albedo,
                light_color=color1)

            # Construct Right Light
            rn.vc += LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
                vc=albedo,
                light_color=color2)

        rendered_image = rn.r
        visibility_image = rn.visibility_image

        image_list = [img] if type(img) is not list else img

        return_img = []
        for item in image_list:
            if self.ratio != 1:
                img_resized = resize(item, (item.shape[0] * self.ratio, item.shape[1] * self.ratio), anti_aliasing=True)
            else:
                img_resized = item / 255.

            try:
                img_resized[visibility_image != (2 ** 32 - 1)] = rendered_image[visibility_image != (2 ** 32 - 1)]
            except:
                logger.warning('Can not render mesh.')

            img_resized = (img_resized * 255).astype(np.uint8)
            res = img_resized

            if rgba:
                img_resized_rgba = np.zeros((img_resized.shape[0], img_resized.shape[1], 4))
                img_resized_rgba[:, :, :3] = img_resized
                img_resized_rgba[:, :, 3][visibility_image != (2 ** 32 - 1)] = 255
                res = img_resized_rgba.astype(np.uint8)
            return_img.append(res)

        if type(img) is not list:
            return_img = return_img[0]

        return return_img

