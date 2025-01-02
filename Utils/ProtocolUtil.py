import numpy as np
import math



def apa(predict_labels: np.ndarray = np.array([]),
                           anomaly_segments: list = [], alarm_coefficient: float = 0.5,beita :float = 4.0,adjust_end = True) -> np.ndarray:

    modified_labels = np.copy(predict_labels).astype(float)

    for segment in anomaly_segments:
        # print(segment)
        if len(segment) != 2:
            continue

        anomaly_range = segment[-1] - segment[0] + 1

        num_anomaly_detected = 0
        t_first = -1

        anomaly_detected_list = []

        for index in range(segment[0], segment[-1] + 1):
            if predict_labels[index] > 0:
                num_anomaly_detected += 1
                anomaly_detected_list.append(index)
                if t_first < 0:
                    t_first = index

        if t_first < segment[0]:
            t_first = -1

        if t_first < 0:
            continue

        # p_s
        p_out = math.ceil(alarm_coefficient * math.ceil(beita * math.log(anomaly_range)))

        startIndex = max(segment[0] - p_out,0)
        for index in range(startIndex, segment[0]):
            if modified_labels[index] > 0 :
                r_s = (1 - alarm_coefficient) * (segment[0] - index) / p_out
                modified_labels[index] = r_s

        if adjust_end:
            for index in range(segment[-1] , min(segment[-1] + p_out,len(predict_labels))):
                if modified_labels[index] > 0 :
                    r_e = (1 - alarm_coefficient) * (index - segment[-1]) / p_out
                    modified_labels[index] = r_e

        adjust_pe_list = []
        # p_i

        for index, t_i in enumerate(anomaly_detected_list):
            if index >= len(anomaly_detected_list) - 1:
                continue

            p_i = math.ceil((num_anomaly_detected / anomaly_range) * (anomaly_detected_list[index + 1] - t_i) + math.ceil( beita * math.log(anomaly_range)))
            adjust_pe_list.append(p_i)

        p_i = math.ceil((num_anomaly_detected / anomaly_range) * (segment[-1] - anomaly_detected_list[-1]) + math.ceil(beita * math.log(anomaly_range)))
        adjust_pe_list.append(p_i)

        for ind, p_i in enumerate(adjust_pe_list):

            anomaly_detected_closest = anomaly_detected_list[ind]

            end_index = min(segment[-1], anomaly_detected_closest + p_i + 1)

            for index in range(anomaly_detected_closest, end_index+1):
                if modified_labels[index] < 1:

                    r_e = - ( (index - anomaly_detected_closest) / anomaly_range) + 1
                    modified_labels[index] = r_e



    return modified_labels

def pa(predict_labels: np.ndarray = np.array([]),
                              anomaly_segments: list = []):

    modified_labels = predict_labels.copy().astype(float)
    for segment in anomaly_segments:
        if len(segment) != 2:
            continue
        flag = False
        for index in range(segment[0], segment[-1] + 1):
            if predict_labels[index] > 0:
                flag = True
                break
        if flag:
            modified_labels[segment[0]:segment[-1]+1] = 1

    return modified_labels

def spa(predict_labels: np.ndarray = np.array([]),ground_labels: np.ndarray = np.array([]),
                              anomaly_segments: list = []):

    modified_labels = predict_labels.copy().astype(float)
    for segment in anomaly_segments:
        if len(segment) != 2:
            continue

        anomaly_range = segment[-1] - segment[0] + 1

        num_anomaly_detected = 0
        t_first = -1

        anomaly_detected_list = []

        for index in range(segment[0], segment[-1] + 1):
            if predict_labels[index] > 0:
                num_anomaly_detected += 1
                anomaly_detected_list.append(index)
                if t_first < 0:
                    t_first = index

        if t_first < segment[0]:
            t_first = -1

        if t_first < 0:
            continue

        p_m = math.floor(num_anomaly_detected/anomaly_range * (segment[-1] - t_first))

        for index in range(t_first , t_first + p_m ):

            if ground_labels[index] > 0 and predict_labels[index] < 1:
                r_s = 1 - (index - segment[0]) / anomaly_range
                modified_labels[index] = r_s



        for index in range(segment[0] , segment[-1]):

            if ground_labels[index] > 0 and predict_labels[index] > 0:
                r_s = 1 - (t_first - segment[0]) / anomaly_range
                modified_labels[index] = r_s

    return modified_labels

def getAlpha(FP,TN):
    print("getAlpha FP:",FP)
    print("getAlpha TN:",TN)
    return 1 - (FP/(FP + TN))

def getBeta(segments ,anomalyScore ,miu ,sigma):
    print("getBeta miu:",miu)
    print("getBeta sigma:", sigma)
    segments_length = 0
    for seg in segments:
        segments_length += len(seg)
    betaMax = math.ceil(segments_length/math.log(segments_length))
    print("getBeta betaMax:", betaMax)
    for i in range(1,betaMax+1):
        print("getBeta beta:",i)
        mean_s = 0
        mean_e = 0
        for seg in segments:
            if seg[0] - math.ceil(i * math.log(segments_length) + 1  ) >= 0 and math.ceil(i * math.log(segments_length) + 1) > 0 :
                mean_s += anomalyScore[max(seg[0] - math.ceil(i * math.log(segments_length) + 1),0):seg[0]].mean()

            if seg[-1] + math.ceil(i * math.log(segments_length) + 1 ) <= anomalyScore.shape[0]-1 and math.ceil(i * math.log(segments_length) + 1) > 0 :
                mean_e += anomalyScore[seg[-1]:min(seg[-1] + math.ceil(i * math.log(segments_length) + 1),anomalyScore.shape[0]-1)].mean()
        print("getBeta mean_s:",mean_s)
        print("getBeta mean_e:",  mean_e)
        print("getBeta miu + 0 * sigma:",miu + 0 * sigma)
        if max( mean_s,mean_e ) < miu :
            return i

    return betaMax
