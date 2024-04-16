"""
utils.py
"""

import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt


def cal_curvature(traj):
    peds_traj = np.concatenate(np.array(traj)).astype(None)
    peds_total = np.unique(peds_traj[:, 0]).tolist()

    for idx_ped in peds_total:
        a = peds_traj[peds_traj[:, 0] == idx_ped, 5:7]
        if len(a) == 1:
            curvature = 0
            continue
        else:
            plt.figure()
            plt.xlim((-2, 15))
            plt.ylim((-2, 15))
            plt.plot(a[:, 0], a[:, 1], '*')
            plt.show()

            dx_dt = np.gradient(a[:, 0])
            dy_dt = np.gradient(a[:, 1])
            velocity = np.array([[dx_dt[i], dy_dt[i]]
                                for i in range(dx_dt.size)])
            ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
            tangent = np.array([1 / ds_dt] * 2).transpose() * velocity
            tangent_x = tangent[:, 0]
            tangent_y = tangent[:, 1]

            deriv_tangent_x = np.gradient(tangent_x)
            deriv_tangent_y = np.gradient(tangent_y)

            dT_dt = np.array([[deriv_tangent_x[i], deriv_tangent_y[i]]
                             for i in range(deriv_tangent_x.size)])

            length_dT_dt = np.sqrt(
                deriv_tangent_x *
                deriv_tangent_x +
                deriv_tangent_y *
                deriv_tangent_y)

            normal = np.array([1 / length_dT_dt] * 2).transpose() * dT_dt

            d2s_dt2 = np.gradient(ds_dt)
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)

            curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / \
                (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
            t_component = np.array([d2s_dt2] * 2).transpose()
            n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()

            acceleration = t_component * tangent + n_component * normal

    return curvature


def derivative(traj):
    peds_traj = np.concatenate(np.array(traj)).astype(None)
    peds_total = np.unique(peds_traj[:, 0]).tolist()
    max_d = 0

    for idx_ped in peds_total:
        a = peds_traj[peds_traj[:, 0] == idx_ped, 5:7]

        p2 = np.array([a[0, 0], a[0, 1]])
        p1 = np.array([a[-1, 0], a[-1, 1]])

        if len(a) - 2 > 0:
            for idx_point in range(len(a) - 2):
                p3 = np.array([a[idx_point + 1, 0], a[idx_point + 1, 1]])
                d1 = abs((p2[1] - p1[1]) * p3[0] - (p2[0] - p1[0])
                         * p3[1] + p2[0] * p1[1] - p2[1] * p1[0])
                d2 = np.sqrt(
                    np.square(
                        p2[1] -
                        p1[1]) +
                    np.square(
                        p2[0] -
                        p1[0]))
                if d2 == 0:
                    d = 1
                else:
                    d = d1 / d2
                if d > max_d:
                    max_d = d

    return max_d


class DataLoader:
    def __init__(
            self,
            datasets,
            seq_length,
            pred_length,
            batch_size,
            data_dir,
            infer=True):
        self.num_batches = None
        self.double_data = None
        self.strdata = None
        self.frame_index = None
        self.numpedslist = None
        self.dataset_index = None
        self.framelist = None
        self.data = None
        self.datasets = datasets
        self.datadir = data_dir
        self.data_file = os.path.join('../cpkl_basic', 'trajectories.cpkl')
        self.datasets = datasets
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.infer = infer
        self.idx_frame = 0
        self.idx_d = 0
        self.idx_sample_frame = 0
        self.pred_length = pred_length
        self.idx_batch = 0
        self.dd_batch = 0
        self.frame_preprocess()

    # read datafile / save cpkl
    def frame_preprocess(self):
        all_frame_data = []
        framelist_data = []
        numpeds_data = []
        dataset_index = 0

        frame_index = []
        all_str_streams = []

        for directory in self.datasets:
            file_path = os.path.join(self.datadir, directory)
            data = np.genfromtxt(file_path, delimiter=',')

            framelist = np.unique(data[0, :]).tolist()
            framelist_data.append(framelist)

            numpeds_data.append([])
            all_frame_data.append([])
            frame_index.append([])
            all_str_streams.append([])

            for ind, frame in enumerate(framelist):
                pedsinframe = data[:, data[0, :] == frame]
                pedslist = pedsinframe[1, :].tolist()
                numpeds_data[dataset_index].append(len(pedslist))
                pedswithpos = []
                strstreams = []

                if len(pedslist) > 0:
                    frame_index[dataset_index].append(frame)
                for ped in pedslist:
                    offsetx = pedsinframe[2, pedsinframe[1, :] == ped][0]
                    offsety = pedsinframe[3, pedsinframe[1, :] == ped][0]
                    velocityx = pedsinframe[4, pedsinframe[1, :] == ped][0]
                    velocityy = pedsinframe[5, pedsinframe[1, :] == ped][0]
                    currentx = pedsinframe[6, pedsinframe[1, :] == ped][0]
                    currenty = pedsinframe[7, pedsinframe[1, :] == ped][0]

                    currxy = np.zeros((1, 2))
                    currxy[:, 0] = currentx
                    currxy[:, 1] = currenty

                    lastxy = np.zeros((1, 2))
                    lastxy[:, 0] = currentx - offsetx
                    lastxy[:, 1] = currenty - offsety

                    str_data = directory.split('_')

                    pedswithpos.append(
                        [ped, offsetx, offsety, velocityx, velocityy, currentx, currenty])
                    strstreams.append(str_data)

                all_frame_data[dataset_index].append(np.array(pedswithpos))
                all_str_streams[dataset_index].append(strstreams)

            dataset_index += 1

        self.data = all_frame_data
        self.framelist = framelist_data
        self.numpedslist = numpeds_data
        self.dataset_index = dataset_index - 1

        self.frame_index = frame_index
        self.strdata = all_str_streams

        # save trajectory.cpkl
        f = open(self.data_file, 'wb')
        pickle.dump(
            (all_frame_data,
             framelist_data,
             numpeds_data),
            f,
            protocol=2)
        f.close()

        counter = 0
        for dataset in range(len(self.data)):
            all_frame_data = self.data[dataset]
            print(
                'training data from dataset {} with {} trajectories'.format(
                    self.datasets[dataset],
                    len(all_frame_data)))

            if self.infer:
                counter += int(len(all_frame_data) -
                               self.pred_length - self.pred_length + 1)
            else:
                counter += int(len(all_frame_data) / self.seq_length)

        if self.infer:
            self.num_batches = int(counter / self.batch_size)
        else:
            self.num_batches = int(counter / self.batch_size) * 2

        #        if self.infer:
        #            self.cal_double_data()
        #            self.double_batch()
        #            self.num_batches = self.num_batches + self.dd_batch

        print('total number of training batches:', self.num_batches)

    def cal_double_data(self):
        self.double_data = []
        for idx_d in range(self.dataset_index + 1):
            self.double_data.append([])
        dataset_index = self.dataset_index
        frame_data = self.data[dataset_index]
        idx = 0
        while True:
            if idx + self.seq_length + self.pred_length - 1 < len(frame_data):
                seq_source_frame_data = frame_data[idx:idx +
                                                   self.seq_length + self.pred_length]
                curr = derivative(seq_source_frame_data)
                if curr > 0.70:
                    self.double_data[dataset_index].append(
                        seq_source_frame_data)

                idx += random.randint(1, self.seq_length)
            else:
                dataset_index = dataset_index - 1
                if dataset_index < 0:
                    break
                frame_data = self.data[dataset_index]
                idx = 0

    def double_batch(self):
        self.dd_batch = 0
        dataset_index = self.dataset_index
        while dataset_index >= 0:
            self.dd_batch += len(self.double_data[dataset_index])
            dataset_index = dataset_index - 1
        self.dd_batch = int(self.dd_batch / self.batch_size)
        self.dd_batch = self.dd_batch * 2

    def next_batch(self, randomUpdate=True):
        # source data
        s_data = []
        # target data
        t_data = []
        # dataset data
        d = []
        m_data = []

        i = 0

        while i < self.batch_size:
            if self.idx_batch < self.num_batches - self.dd_batch + 1:
                idx = self.idx_frame
                frame_data = self.data[self.dataset_index]
                mask_data = self.strdata[self.dataset_index]

                if idx + self.seq_length + \
                        self.pred_length - 1 < len(frame_data):
                    seq_source_frame_data = frame_data[idx:idx +
                                                       self.seq_length + self.pred_length]
                    seq_source_mask_data = mask_data[idx:idx +
                                                     self.seq_length + self.pred_length]

                    s_data.append(seq_source_frame_data)
                    m_data.append(seq_source_mask_data)

                    if self.infer:
                        self.idx_frame += 1
                    else:
                        self.idx_frame += random.randint(1, self.seq_length)

                    i += 1
                else:
                    self.idx_frame = 0
                    self.reset_dataset_pointer()

            else:
                idx = self.idx_d
                frame_data = self.double_data[self.dataset_index]
                if idx < len(frame_data):
                    s_data.append(frame_data[idx])
                    self.idx_d = self.idx_d + 1
                    i += 1
                else:
                    self.idx_d = 0
                    self.reset_double_dataset_pointer()

        self.idx_batch = self.idx_batch + 1
        return s_data, m_data, t_data, d

    def next_sample_batch(self):
        # source data
        s_data = []
        # target data
        t_data = []
        # dataset data
        d = []

        m_data = []

        # if the end
        if_end = False

        i = 0

        while i < self.batch_size:

            frame_data = self.data[self.dataset_index]
            mask_data = self.strdata[self.dataset_index]
            idx = self.idx_sample_frame
            frame_index = self.frame_index[self.dataset_index]
            if idx + self.seq_length + self.pred_length - 1 < len(frame_data):

                seq_source_frame_data = frame_data[idx:idx + self.seq_length]
                seq_target_frame_data = frame_data[idx:idx +
                                                   self.seq_length + self.pred_length]
                seq_source_mask_data = mask_data[idx:idx +
                                                 self.seq_length + self.pred_length]

                s_data.append(seq_source_frame_data)
                m_data.append(seq_source_mask_data)
                t_data.append(seq_target_frame_data)

                self.start_index = frame_index[idx + self.seq_length - 1]
                self.idx_sample_frame += 1

                d.append(self.dataset_index)

                i += 1
            elif self.dataset_index > 0:

                self.dataset_index = self.dataset_index - 1

            else:
                if_end = True
                break

        return s_data, m_data, t_data, d, if_end

    def reset_batch_pointer(self):
        self.idx_batch = 0

    def reset_dataset_pointer(self):
        self.dataset_index = self.dataset_index - 1
        if self.dataset_index < 0:
            self.dataset_index = len(self.data) - 1

    def reset_double_dataset_pointer(self):
        self.dataset_index = self.dataset_index + 1
        if self.dataset_index >= len(self.double_data):
            self.dataset_index = 0
