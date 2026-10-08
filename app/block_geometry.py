"""Horizontal block top geometry in robot-base metres; no hardware dependencies."""
import numpy as np
from scipy.optimize import least_squares


def fit_known_cube(points, reference_closing, camera_origin, size=.025, diagnostics=None):
    """Fit visible finite faces of an upright cube. No hidden center is assumed.

    Metres in robot base coordinates. A visible side anchors its normal axis;
    otherwise the top must span that full axis. Side-only views are rejected.
    """
    if not np.isfinite(size) or not .008 <= size <= .15:
        raise ValueError('积木尺寸配置无效')
    camera = np.asarray(camera_origin, dtype=float)
    if camera.shape != (3,) or not np.isfinite(camera).all():
        raise ValueError('拍摄时相机位置无效')
    cloud = np.asarray(points, dtype=float)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if len(cloud) < 80:
        raise ValueError('立方体有效点云不足')
    # Spatial sampling balances repeated side/depth pixels and bounds runtime.
    _, indices = np.unique(np.floor(cloud / .0008).astype(int), axis=0, return_index=True)
    cloud = cloud[np.sort(indices)]
    cloud = cloud[::max(1, int(np.ceil(len(cloud) / 2000)))]
    if len(cloud) < 80:
        raise ValueError('立方体有效空间覆盖不足')
    lo, hi = np.percentile(cloud, [2, 98], axis=0)
    if np.any(hi - lo > size * 1.65 + .004):
        raise ValueError('点云尺寸超过已知立方体，检查分割和深度')
    half = size / 2

    def geometry(params):
        cx, cy, top, yaw = params
        c, s = np.cos(yaw), np.sin(yaw)
        basis = np.array([[c, -s], [s, c]])
        local = (cloud[:, :2] - [cx, cy]) @ basis
        view = (camera[:2] - [cx, cy]) @ basis
        outside = np.maximum(np.abs(local) - half, 0)
        zoutside = np.maximum.reduce((cloud[:, 2] - top, top - size - cloud[:, 2], np.zeros(len(cloud))))
        faces = [np.sqrt((cloud[:, 2] - top)**2 + np.sum(outside**2, axis=1))]
        for axis in range(2):
            if abs(view[axis]) > half + .002:
                faces.append(np.sqrt((local[:, axis] - np.sign(view[axis]) * half)**2
                                     + outside[:, 1-axis]**2 + zoutside**2))
            else:
                faces.append(np.full(len(cloud), size * 4))
        return np.asarray(faces).T, local, view, basis

    def residual(params):
        return geometry(params)[0].min(axis=1)

    solutions = []
    for degrees in range(0, 90, 10):
        yaw = np.deg2rad(degrees)
        initial = [*(lo[:2] + hi[:2]) / 2, hi[2], yaw]
        result = least_squares(residual, initial, loss='soft_l1', f_scale=.0006,
                               diff_step=1e-4, max_nfev=65,
                               bounds=([lo[0]-half, lo[1]-half, hi[2]-.004, yaw-.3],
                                       [hi[0]+half, hi[1]+half, hi[2]+.004, yaw+.3]))
        errors = residual(result.x)
        score = np.mean(np.minimum(errors, .004)**2)
        solutions.append((score, result.x))
    solutions.sort(key=lambda item: item[0])
    score, params = solutions[0]
    faces, local, view, basis = geometry(params)
    errors = faces.min(axis=1)
    # Mask edges and RGB-D discontinuities produce non-surface returns. Assess
    # the supported surface independently, but require a majority consensus.
    inliers = errors < .003
    inlier_fraction = float(np.mean(inliers))
    inlier_rms = float(np.sqrt(np.mean(errors[inliers]**2))) if inliers.any() else float('inf')
    if diagnostics is not None:
        diagnostics.update(sample_count=len(cloud), inlier_fraction=inlier_fraction,
                           inlier_rms_m=inlier_rms, fit_rms_m=float(np.sqrt(score)))
    if inlier_fraction < .75 or inlier_rms > .0015:
        raise ValueError('点云不能可靠匹配已知立方体（有效表面 %.0f%%，残差 %.2f mm），停止抓取'
                         % (inlier_fraction * 100, inlier_rms * 1000))
    for other_score, other in solutions[1:]:
        angle = abs((other[3] - params[3] + np.pi/4) % (np.pi/2) - np.pi/4)
        if (other_score <= score + .00015**2
                and (np.linalg.norm(other[:2] - params[:2]) > .002 or angle > np.deg2rad(5))):
            raise ValueError('立方体位姿存在多个可能解，需要更完整视角')
    top_mask = (faces.argmin(axis=1) == 0) & (faces[:, 0] < .0015)
    top_points = local[top_mask]
    if len(top_points) < 40 or np.min(np.ptp(top_points, axis=0)) < .008:
        raise ValueError('顶面证据不足，不能仅凭侧面生成抓取位姿')
    # A sloped/noisy slice must not masquerade as a horizontal top.
    plane = np.linalg.lstsq(np.c_[top_points, np.ones(len(top_points))], cloud[top_mask, 2], rcond=None)[0]
    if np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(8)):
        raise ValueError('顶面不水平，检查深度和标定')
    anchored = []
    for axis in range(2):
        side = ((faces.argmin(axis=1) == axis + 1) & (faces[:, axis+1] < .0015)
                & (cloud[:, 2] < params[2] - .003))
        supported = (side.sum() >= 30 and np.ptp(cloud[side, 2]) >= .006
                     and np.ptp(local[side, 1-axis]) >= .010)
        lower, upper = np.percentile(top_points[:, axis], [2, 98])
        complete = lower <= -half + .003 and upper >= half - .003
        if not supported and not complete:
            raise ValueError('遮挡方向缺少完整顶面或侧面约束，停止抓取')
        anchored.append(bool(supported))
    reference = np.asarray(reference_closing, dtype=float)[:2]
    if np.linalg.norm(reference) < 1e-6:
        reference = np.array([0., 1.])
    closing = max((sign * basis[:, axis] for axis in range(2) for sign in (-1, 1)),
                  key=lambda direction: direction @ reference)
    x_axis = np.array([0., 0., -1.])
    y_axis = np.r_[closing, 0.]
    rotation = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))
    if diagnostics is not None:
        diagnostics.update(method='known_cube_visible_faces', size_m=float(size),
                           fit_rms_m=float(np.sqrt(score)), side_constraints=anchored,
                           sample_count=len(cloud))
    return params[:2], float(params[2]), rotation, float(size)


def _fit_rectangle(xy):
    origin = np.median(xy, axis=0)
    centered = xy - origin
    best = None
    for degrees in np.arange(0., 90., .5):
        angle = np.deg2rad(degrees)
        basis = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        projected = centered @ basis
        low, high = np.percentile(projected, [1, 99], axis=0)
        area = np.prod(high - low)
        if best is None or area < best[0]:
            best = area, basis, low, high
    _, basis, low, high = best
    return origin, basis, low, high


def fit_block_top(points, reference_closing):
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 40:
        raise ValueError('积木有效点云不足，不能确定抓取姿态')
    height = np.percentile(points[:, 2], 85)
    top = points[np.abs(points[:, 2] - height) <= .003]
    if len(top) < 40:
        raise ValueError('积木顶面点不足，请在顶面完整可见时重新检测')
    # Bound fitting cost without changing the spatial sampling order.
    xy = top[::max(1, len(top) // 4000), :2]
    origin, basis, low, high = _fit_rectangle(xy)
    xy = xy - origin
    size = high - low
    if size.min() < .008 or size.max() > .15:
        raise ValueError('顶面尺寸异常，拒绝生成抓取姿态')
    # Reject sparse/line-like surfaces; this cannot certify an occluded full top.
    grid = np.floor(((xy @ basis) - low) / size * 8).astype(int)
    grid = grid[((grid >= 0) & (grid < 8)).all(axis=1)]
    if len(np.unique(grid, axis=0)) < 40:
        raise ValueError('顶面覆盖不足，无法可靠拟合矩形')
    upper = points[(points[:, 2] >= height - .012) & (points[:, 2] <= height + .003)]
    full_projected = (upper[:, :2] - origin) @ basis
    full_low, full_high = np.percentile(full_projected, [1, 99], axis=0)
    overflow = np.maximum(low - full_low, full_high - high)
    if overflow.max() > .004:
        raise ValueError('顶面拟合未覆盖目标，停止抓取')
    reference = np.asarray(reference_closing, dtype=float)[:2]
    if np.linalg.norm(reference) < 1e-6:
        reference = np.array([0., 1.])
    reference = reference / np.linalg.norm(reference)
    candidates = []
    for index in range(2):
        if size[index] <= size.min() * 1.08:
            for sign in (-1, 1):
                direction = sign * basis[:, index]
                candidates.append((direction @ reference, direction, size[index]))
    _, closing, width = max(candidates, key=lambda item: item[0])
    x_axis = np.array([0., 0., -1.])
    y_axis = np.r_[closing, 0.]
    rotation = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))
    center = origin + basis @ ((low + high) / 2)
    return center, float(np.median(top[:, 2])), rotation, float(width)
