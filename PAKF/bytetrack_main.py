import cv2 as cv
import argparse
from yolox.my_yolo import YoloX
from yolox.utils.val_dataloader import LoadImages
from tracker.byte_tracker import BYTETracker
from tracker.timer import Timer
from yolox.utils.visualize import plot_tracking,increment_path
import os
import pathlib as Path
from loguru import logger
import time




# YOLOX相关参数
def parse_opt():
    parser = argparse.ArgumentParser()
    # 存放需要检测数据的文件夹
    # 存放跟权重相应模型的配置文件
    parser.add_argument('--exp_file', default='./yolox/exp/yolox_x_mix_det', help='exp path')
    # 存放模型权重的路径
    parser.add_argument('--weight_path', default='./yolox/weights/bytetrack_x_mot17.pth.tar', help='weight path')
    parser.add_argument("--save",default=True,help='save results')
    parser.add_argument('--save_txt', default=False, help='save txt results')
    parser.add_argument('--show', default=False, help='show frame img')
    parser.add_argument('--save_eval_name',default='MOT17-11',help='save eval txt name')


    opt = parser.parse_args()
    return opt


# 跟踪器相关参数
def parse_args():
    parser = argparse.ArgumentParser()
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")  #origin=0.8
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,     #1.6
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')   #10
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    args = parser.parse_args()
    return args

def write_results(filename, results):#写结果的函数
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                          w=round(w, 1), h=round(h, 1), s=round(score, 2))#舍去小数点后1位数字
                f.write(line)
    logger.info('save results to {}'.format(filename))


def main():
    opt = parse_opt()
    args = parse_args()
    data_path = './yolox/dataset/MOT17-'
    save_dir = 'outdata/runs/MOT17-'
    eval_path = 'outdata/runs/MOT17-'
    # 载入YOLOX
    yolo = YoloX(opt.exp_file, opt.weight_path)
    # 载入跟踪器
    tracker = BYTETracker(args)

    # 载入测试数据
    dataset = LoadImages(data_path, img_size=yolo.test_size)
    timer = Timer()
    results = []
    frame_id = 0
    vid_path, vid_writer = [None] * 1, [None] * 1
    desktop_path = eval_path
    filename = desktop_path + '.txt'

    # 创建保存目录
    if opt.save:
        # Directories
        save_dir = increment_path(save_dir)  # increment run
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 做一个目录
    for path, im, im0s, vid_cap, s in dataset:
        frame_id += 1
        # 输入output [x1, y1, x2, y2, score1, score2, class]
        outputs = yolo.detect_bounding_box(im, im0s)
        if outputs[0] is not None:
            # 跟新跟踪器
            online_targets = tracker.update(outputs[0], [im0s.shape[0], im0s.shape[1]], (800, 1440))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

        timer.toc()
        online_im = plot_tracking(
            im0s, online_tlwhs, online_ids, online_scores, frame_id=frame_id, fps=1. / timer.average_time
        )#在线的图片
        if opt.show:
            cv.imshow('img', online_im)
            cv.waitKey(50)

        if opt.save:
            image_name = path[path.rfind('\\') + 1:]
            save_path = os.path.join(save_dir, image_name)
            if dataset.mode == 'image':
                cv.imwrite(save_path, online_im)
            else:
                if vid_path[0] != save_path:  # new video
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv.VideoWriter):
                        vid_writer[0].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    vid_writer[0] = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0s)

    write_results(filename=filename, results=results)
    print("the {} epoch is over !".format(i))




if __name__ == '__main__':
    main()