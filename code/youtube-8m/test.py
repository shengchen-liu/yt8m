import os
from collections import defaultdict, Counter
import pickle



SUBMIT_PATH = '/media/shengchen/Shengchen/yt8m/predictions'
SIGFIGS = 6

def read_models(model_weights, blend=None):
    if not blend:
        blend = defaultdict(Counter)
    for m, w in model_weights.items():
        print(m, w)
        with open(os.path.join(SUBMIT_PATH, m + '.csv'), 'r') as f:
            f.readline()
            for l in f:
                id, r = l.split(',')
                # id, r = int(id), r.split(' ')
                id, r = id, r.split(' ')
                n = len(r) // 2
                for i in range(0, n, 2):
                    k = int(r[i])
                    v = int(10**(SIGFIGS - 1) * float(r[i+1]))
                    blend[id][k] += w * v
    return blend


def write_models(blend, file_name, total_weight):
    with open(os.path.join(SUBMIT_PATH, file_name + '.csv'), 'w') as f:
        f.write('VideoID,LabelConfidencePairs\n')
        for id, v in blend.items():
            l = ' '.join(['{} {:{}f}'.format(t[0]
                                            , float(t[1]) / 10 ** (SIGFIGS - 1) / total_weight
                                            , SIGFIGS) for t in v.most_common(20)])
            f.write(','.join([str(id), l + '\n']))
    return None


def main():
    model_pred = {
        'prediction_multiscale_cnn_lstm_model_batch128_decay08':0.84134,
        'predictions_gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe': 0.82856,
        'predictions_video_level_MoeModel': 0.82936

    }
    print("hello")
    avg = read_models(model_pred)
    write_models(avg, 'average_submission_version1_weighted', sum(model_pred.values()))

if __name__ == '__main__':
    main()