"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
from my_track.PAkalman_filter import PAkalmanFilter
import matching
from my_sort.sort_basetrack import BaseTrack, SortTrackState


class STrack(BaseTrack):
    shared_kalman = PAkalmanFilter()

    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance, self.k = None, None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != SortTrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance,self.k.copy())  #可以参照对应修改

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks]) #加一个multi_k
            multi_k = np.asarray([st.k for st in stracks],dtype= object)
            for i, st in enumerate(stracks):
                if st.state != SortTrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance,multi_k)  # mutlt_k
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):  #这里也要加上
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):  #激活一个新的轨迹
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance,self.k = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = SortTrackState.Tracked
        if frame_id == 1:  #这里是第一帧的关键
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, cost_IOU, new_id=False):  #重新激活，对于lost的轨迹而言的  这里的新轨迹还是缺乏一个cost_IOU属性
        self.score = new_track.score
        self.cost_IOU = cost_IOU
        self.mean, self.covariance, self.k = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh,),self.score, self.cost_IOU   #注意这里的cost还没有指定
        )
        self.tracklet_len = 0
        self.state = SortTrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()


    def update(self, detection, frame_id, cost_IOU): #这里说是轨迹，其实是检测，不能在这里用IOU
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = detection.tlwh
        self.score = detection.score  #加入的
        self.cost_IOU = cost_IOU  #加入的
        self.mean, self.covariance, self.k = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh),self.score, self.cost_IOU)
        self.state = SortTrackState.Tracked
        self.is_activated = True

        # self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
          [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
          the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
          [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if (score == None):
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Mysort(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]  #成功匹配的轨迹
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh #小于这个阈值的，都不能生成新轨迹，这里还加了个0.1，可以对比下效果
        self.max_time_lost = 30
        self.kalman_filter = PAkalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []  # 激活的
        refind_stracks = []  # 重新找回的
        lost_stracks = []  # 失去的
        removed_stracks = []  # 删除的
        # outputbox【batchsize, x1, y1,x2, y2, score, class】
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        # outputbox【batchsize, x1, y1,x2, y2, score1,score2, class】
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2  #对于YOLOx输出数据的装化
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        # 根据框的得分进行划分
        # 高分框
        remain_inds = scores > self.args.track_thresh
        dets = bboxes[remain_inds] #高分检测盒
        scores_keep = scores[remain_inds] #高分检测的分数
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track) #非激活的轨迹放到不确定集合里面
            else:
                tracked_stracks.append(track) #其余放到确定集合里去

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks,self.lost_stracks)
        # 高分目标的首次关联，结合损失的轨迹与现有的轨迹 #轨迹还是要带上k这个属性，但是LOST不好带吧，没有的默认为1吧
        # strack_pool = joint_stracks(strack_pool, unconfirmed)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)  # IOU距离在这里计算了 ，这里的dists是矩阵
        matches, u_track, u_detection = matching.linear_assignment(dists,thresh=self.args.match_thresh)  # 加外观的时候可以关注这个 ，这里匹配上的要在带上cost——IOU
        #匹配上的轨迹，老的就继续，lost的就恢复，unconf的就看看是否满足条件，如果大于3帧的就.is activated
        for itracked, idet in matches:
            track = strack_pool[itracked]  # 轨迹列表
            det = detections[idet]  # 对应的检测
            cost_IOU = dists[itracked, idet]
            if track.state == SortTrackState.Tracked:
                track.update(det, self.frame_id, cost_IOU)  # 这里用的居然是检测
                activated_starcks.append(track)
            elif track.state == SortTrackState.Lost:
                track.re_activate(det, self.frame_id, cost_IOU, new_id=False)
                refind_stracks.append(track)

        #对于没有匹配上的轨迹，如果是lost的看寿命大于30删除，如果是正常的转为lost，如果是不确定的直接删除
        for i in u_track:
            track = strack_pool[i]
            if track.state == SortTrackState.Lost:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)
            elif track.state == SortTrackState.Tracked:
                track.state = SortTrackState.Lost
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
        for itracked, idet in matches:
            cost_IOU = dists[itracked, idet]
            unconfirmed[itracked].update(detections[idet], self.frame_id,cost_IOU)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == SortTrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
