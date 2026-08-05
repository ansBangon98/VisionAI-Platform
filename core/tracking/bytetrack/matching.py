import numpy as np
from . import kalman_filter

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency.
    scipy_linear_sum_assignment = None

try:
    from scipy.spatial.distance import cdist
except ImportError:  # pragma: no cover - optional dependency.
    cdist = None

def merge_matches(m1, m2, shape):
    O, _P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    if m1.size == 0 or m2.size == 0:
        match = []
    else:
        m2_by_left = {left: right for left, right in m2}
        match = [
            (left, m2_by_left[middle])
            for left, middle in m1
            if middle in m2_by_left
        ]
    unmatched_O = tuple(set(range(O)) - {i for i, _j in match})
    unmatched_Q = tuple(set(range(Q)) - {j for _i, j in match})

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if scipy_linear_sum_assignment is None:
        return _greedy_assignment(cost_matrix, thresh)

    finite_cost_matrix = np.nan_to_num(
        cost_matrix,
        nan=1e6,
        posinf=1e6,
        neginf=-1e6,
    )
    rows, cols = scipy_linear_sum_assignment(finite_cost_matrix)
    matches = [
        [row, col]
        for row, col in zip(rows, cols)
        if cost_matrix[row, col] <= thresh
    ]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - {row for row, _ in matches})
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - {col for _, col in matches})
    matches_array = np.asarray(matches, dtype=int).reshape((-1, 2))
    return matches_array, np.asarray(unmatched_a), np.asarray(unmatched_b)


def _greedy_assignment(cost_matrix, thresh):
    candidate_rows, candidate_cols = np.where(cost_matrix <= thresh)
    candidates = sorted(
        (
            (float(cost_matrix[row, col]), int(row), int(col))
            for row, col in zip(candidate_rows, candidate_cols)
        ),
        key=lambda item: item[0],
    )

    matched_rows = set()
    matched_cols = set()
    matches = []
    for _cost, row, col in candidates:
        if row in matched_rows or col in matched_cols:
            continue
        matched_rows.add(row)
        matched_cols.add(col)
        matches.append([row, col])

    unmatched_a = np.asarray(
        sorted(set(range(cost_matrix.shape[0])) - matched_rows),
        dtype=int,
    )
    unmatched_b = np.asarray(
        sorted(set(range(cost_matrix.shape[1])) - matched_cols),
        dtype=int,
    )
    matches_array = np.asarray(matches, dtype=int).reshape((-1, 2))
    return matches_array, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    return _bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float),
    )


def _bbox_ious(atlbrs, btlbrs):
    top_left = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    bottom_right = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])
    wh = np.maximum(0.0, bottom_right - top_left)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.maximum(0.0, atlbrs[:, 2] - atlbrs[:, 0]) * np.maximum(
        0.0,
        atlbrs[:, 3] - atlbrs[:, 1],
    )
    area_b = np.maximum(0.0, btlbrs[:, 2] - btlbrs[:, 0]) * np.maximum(
        0.0,
        btlbrs[:, 3] - btlbrs[:, 1],
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-6)


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    if cdist is None:
        raise RuntimeError("scipy is required for embedding distance matching.")
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
