import numpy as np
import scipy.misc
import cv2
import skimage.transform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from models.smpl import SMPL
import numpy as np
# skimage.transform.resize()



def draw_joints2D(joints2D, ax=None, kintree_table=None, with_text=True, color='g'):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints2D[j1, 0], joints2D[j2, 0]],
                [joints2D[j1, 1], joints2D[j2, 1]],
                color=color, linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_text:
            ax.text(joints2D[j2, 0],
                    joints2D[j2, 1],
                    s='{}'.format(j2),
                    color=color,
                    fontsize=8)


def draw_joints3D(joints3D, ax=None, kintree_table=None, with_text=True, figure_name='figure_1',color='g'):
    if not ax:
        fig = plt.figure(figure_name)
        ax = fig.add_subplot(111)

    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                [joints3D[j1, 1], joints3D[j2, 1]],
                [joints3D[j1, 2], joints3D[j2, 2]],
                color=color, linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_text:
            ax.text(joints3D[j2, 0],
                    joints3D[j2, 1],
                    joints3D[j2, 2],
                    s='{}'.format(j2),
                    color=color,
                    fontsize=8)



def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines,figure_name='figure_1', filename=None):
    fig = plt.figure(figure_name)
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

    x_r = np.array([0, 256], dtype=np.float32)
    y_r = np.array([0, 256], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    # ax.set_xlim([0,cfg.input_shape[1]])
    # ax.set_ylim([0,1])
    # ax.set_zlim([-cfg.input_shape[0],0])
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(kp_mask, str(i1), p1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(kp_mask, str(i2), p2, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def preprocess_image(img):

    img_size=224
    if np.max(img.shape[:2]) != img_size:
        # print('Resizing so the max image size is %d..' % img_size)
        scale = (float(img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]

    crop, proc_param = scale_and_crop(img, scale, center, img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop

def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor



def scale_and_crop(image, scale, center, img_size):
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min( br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = scipy.misc.imresize(new_img, res)
    return new_img


def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def augm_params():
    """Get augmentation parameters."""
    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling
    if True:
        # We flip with probability 1/2
        if np.random.uniform() <= 0.5:
            flip = 1

        # Each channel is multiplied with a number
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(1 - 0.4, 1 + 0.4,
                               3)  # np.random.uniform()

        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
        rot = min(2 * 30,
                  max(-2 * 30,
                      np.random.randn() * 30))  # np.random.randn()

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(1 + 0.25,
                 max(1 - 0.2, np.random.randn() * 0.25 + 1))
        # but it is zero with probability 3/5
        if np.random.uniform() <= 0.6:
            rot = 0

    return flip, pn, rot, sc


def rgb_processing(rgb_img, center, scale, rot, flip, pn):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale,
                   [224, 224], rot=rot)
    # flip the image
    if flip:
        rgb_img = flip_img(rgb_img)
    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    # (3,224,224),float,[0,1]
    # rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
    return rgb_img

def flip_kp(kp):
    """Flip keypoints."""
    flipped_parts = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
    kp = kp[flipped_parts]
    kp[:,0] = - kp[:,0]
    return kp

def j2d_processing( kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [224, 224], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / 224 - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

def show_mesh( V=[],gt_V=[],kp3d=[],gt3d=[],show_gtV=False,show_gt3d=False,show_kp=False,name='NULL'):
    fig = plt.figure(name)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color='black', marker='o')
    if show_gtV:
        ax.scatter(gt_V[:, 0], gt_V[:, 1], gt_V[:, 2], s=1, color='green', marker='o')
    if show_gt3d:
        ax.scatter(gt3d[:, 0], gt3d[:, 1], gt3d[:, 2], s=20, color='blue', marker='o')
        for x in range(len(gt3d)):
            ax.text(gt3d[x, 0], gt3d[x, 1], gt3d[x, 2], str(x), color='blue')
    if show_kp:
        ax.scatter(kp3d[:, 0], kp3d[:, 1], kp3d[:, 2], s=20, color='red', marker='o')
        for x in range(len(kp3d)):
            ax.text(kp3d[x, 0], kp3d[x, 1], kp3d[x, 2], str(x), color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    plt.legend()
    plt.show()
    # plt.close()







import os
import config as cfg

if __name__ == '__main__':
    # data = np.load('./h36m_train_new.npz')
    # for i in range(len(data['imgname'])):
    #     img_path = data['imgname'][i]
    #     img_path = os.path.join('/home/sunqingping/PycharmProjects/data/c2f_vol.zip/c2f_vol', img_path)
    #     smpl_kp = data['S_img'][i][:, :2]
    #     h36m_kp = data['part'][i][:, :2][cfg.J24_TO_J17]
    #     img = cv2.imread(img_path)
    #     img_smpl = vis_keypoints(img, np.hstack([smpl_kp, np.ones([24, 1])]).transpose(1, 0), cfg.smpl_skeleton)
    #     cv2.imshow('img_smpl', img_smpl)
    # 
    #     img_h36m = vis_keypoints(img, np.hstack([h36m_kp, np.ones([17, 1])]).transpose(1, 0), cfg.human36_skeleton)
    #     cv2.imshow('img_h36m', img_h36m)
    # 
    # 
    # 
    #     cv2.waitKey()
    # data = np.load('./datasets/extras/h36m_pose2mesh_train.npz')
    # for i in range(len(data['imgname'])):
    #     img_path = data['imgname'][i]
    #     img_path = os.path.join('/home/sunqingping/mnt/code/Pose2Mesh_RELEASE/data/Human36M/images', img_path)
    #     smpl_kp = data['S_smpl'][i][:, :2]
    #     h36m_kp = data['part'][i]
    #     img = cv2.imread(img_path)
    #     img_smpl = vis_keypoints(img, np.hstack([smpl_kp, np.ones([24, 1])]).transpose(1, 0), cfg.smpl_skeleton)
    #     cv2.imshow('img_smpl', img_smpl)
    #
    #     img_h36m = vis_keypoints(img, np.hstack([h36m_kp, np.ones([17, 1])]).transpose(1, 0), cfg.human36_skeleton)
    #     cv2.imshow('img_h36m', img_h36m)
    #
    #
    #
    #     cv2.waitKey()
    # data_coco = np.load('./datasets/extras/coco_pose2mesh_train.npz')
    # for i in range(len(data_coco['imgname'])):
    #     img_path = data_coco['imgname'][i]
    #     img_path = os.path.join('/home/sunqingping/mnt/code/Pose2Mesh_RELEASE/data/COCO/images/train2017', img_path)
    #     smpl_kp = data_coco['S_smpl'][i][:, :2]
    #     coco_kp = data_coco['part'][i][[cfg.J24_TO_J19],:]
    #     img = cv2.imread(img_path)
    #     img_gt = vis_keypoints(img, coco_kp[0].transpose(1, 0), cfg.coco_skeleton)
    #     img_pj_smpl = vis_keypoints(img,  np.hstack([smpl_kp, np.ones([24, 1])]).transpose(1, 0), cfg.smpl_skeleton)
    #     cv2.imshow('ground_turth', img_gt)
    #     cv2.imshow('smpl', img_pj_smpl)
    #     cv2.waitKey()

    # scaleFactor = 1.2
    # I = scipy.misc.imread('./1.jpg')
    # ys, xs = np.where(np.min(I, axis=2) < 255)
    # bbox = np.array([np.min(xs), np.min(ys), np.max(xs) + 1, np.max(ys) + 1])
    # bbox = [10,0,40,55]
    # center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    # scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.
    # #
    # # # Get augmentation parameters
    # flip, pn, rot, sc = augm_params()
    # # print(rot)
    # cv2.imshow('before', I)
    # cv2.waitKey()
    # img = rgb_processing(I, center, sc*scale, rot, flip, pn)
    # #
    # # cv2.imshow('after', img)
    # # cv2.waitKey()
    #
    # # use edge to pad image and crop
    # # img = preprocess_image(I)
    #
    # cv2.imshow('after', img)
    # cv2.waitKey()


# # --------------------------------------------------------------------------------------------------------------------
#     # SMPL(24) + human3.6(17,except 3 points)
#     # SMPL(24)
#     JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]
#     # SMPL(24) to LSP(14)
#     J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
#     # human3.6(17) to LSP(14)
#     H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    smpl = SMPL()
    V = smpl.v_template
    # smpl_kp3d = smpl.get_smpl_joints(V[None,:,:])[0]
    # train_kp3d = smpl.get_full_joints(V[None,:,:])[0]
    #
    # fig = plt.figure('SMPL_38_J_Regression')
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color='black', marker='o')
    # ax.scatter(train_kp3d[:, 0], train_kp3d[:, 1], train_kp3d[:, 2], s=20, color='red', marker='o')
    # # ax.scatter(smpl_kp3d[:, 0], smpl_kp3d[:, 1], smpl_kp3d[:, 2], s=20, color='blue', marker='o')
    # for x in range(len(train_kp3d)):
    #     ax.text(train_kp3d[x, 0], train_kp3d[x, 1], train_kp3d[x, 2], str(x), color='red')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.legend()
    # plt.show()

    J_regressor_eval = np.load('data/J_regressor_extra.npy')
#     kp3d = np.matmul(J, V)  # *1000
#     kp3d_14 = kp3d[J24_TO_J14, :]
#     kp3d_E = np.matmul(J_E, V)
    kp3d_eval = np.matmul(J_regressor_eval, V)
    print(kp3d_eval.shape)
#
#     # kp3d_14.cpu().numpy()
#     # kp3d_E = kp3d_E.cpu().numpy()
#
#     # ----------- human3.6 keypoint-----------------
    fig = plt.figure('SMPL_24_J_Regression')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color='black', marker='o')
    ax.scatter(kp3d_eval[:, 0], kp3d_eval[:, 1], kp3d_eval[:, 2], s=20, color='blue', marker='o')

    for x in range(len(kp3d_eval)):
        ax.text(kp3d_eval[x, 0], kp3d_eval[x, 1], kp3d_eval[x, 2], str(x), color='red')
#
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()
#
#     # -------------- lsp keypoint ----------
#     fig_ = plt.figure('SMPL_LSP_14_Select_from_SMPL_24_J_Regression')
#     ax_ = fig_.add_subplot(111, projection='3d')
#     ax_.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color='black', marker='o')
#     ax_.scatter(kp3d_14[:, 0], kp3d_14[:, 1], kp3d_14[:, 2], s=20, color='blue', marker='o')
#
#     for x in range(len(kp3d_14)):
#         ax_.text(kp3d_14[x, 0], kp3d_14[x, 1], kp3d_14[x, 2], str(J24_TO_J14[x]), color='red')
#
#     # ---------train keypoint-------------
#     train_keypoint = smpl.get_joints(V.unsqueeze(dim=0)).squeeze()
#     fig_1 = plt.figure('Human3.6_Train_Concatenate_with_J_regressor_extra')
#     ax_1 = fig_1.add_subplot(111, projection='3d')
#
#     ax_1.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color='black', marker='o')
#     ax_1.scatter(train_keypoint[:, 0], train_keypoint[:, 1], train_keypoint[:, 2], s=20, color='blue', marker='o')
#
#     for x in range(24):
#         ax_1.text(train_keypoint[x, 0], train_keypoint[x, 1], train_keypoint[x, 2], str(x), color='red')
#
#     # ---------eval keypoint-------------
#
#     fig_2 = plt.figure('eval_keypoint')
#     ax_2 = fig_2.add_subplot(111, projection='3d')
#
#     ax_2.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color='black', marker='o')
#     ax_2.scatter(kp3d_eval[:, 0], kp3d_eval[:, 1], kp3d_eval[:, 2], s=20, color='blue', marker='o')
#
#     for x in range(17):
#         ax_2.text(kp3d_eval[x, 0], kp3d_eval[x, 1], kp3d_eval[x, 2], str(x), color='red')
#
#     # ---------eval keypoint-------------
#
#     fig_3 = plt.figure('Extra_Regression')
#     ax_3 = fig_3.add_subplot(111, projection='3d')
#
#     ax_3.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color='black', marker='o')
#     ax_3.scatter(kp3d_E[:, 0], kp3d_E[:, 1], kp3d_E[:, 2], s=20, color='blue', marker='o')
#
#     for x in range(14):
#         ax_3.text(kp3d_E[x, 0], kp3d_E[x, 1], kp3d_E[x, 2], str(x + 24), color='red')
#
#     plt.legend()
#     plt.show()
