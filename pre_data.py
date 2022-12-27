import warnings
import pandas as pd
import os
import numpy as np
import scipy.io as sio
from option import Options

warnings.filterwarnings("ignore")

def get_key(file_name):
    file_name = file_name.split('_')
    key = ''
    for i in range(len(file_name)):
        if file_name[i] == 'rois':
            key = key[:-1]
            break
        else:
            key += file_name[i]
            key += '_'
    return key

def load_data(path1, path2 ,name,class_path,c_kind):
    profix = path1
    dirs = os.listdir(profix)
    dirss = np.sort(dirs)
    all = {}
    labels = {}
    all_data = []
    label = []
    data = pd.read_csv(path2)
    count = 0
    for filename in dirss:
        if get_key(filename).split('_')[0] == c_kind:
            print('get_filename: ', get_key(filename).split('_')[0], ' c_kind: ',c_kind)
            a = np.loadtxt(path1 + filename)
            a = a.transpose()
            all[filename] = a
            all_data.append(a)
            for i in range(len(data)):
                if get_key(filename) == data['FILE_ID'][i]:
                    print('get_key(filename): ', get_key(filename), ' data[FILE_ID][i]: ', data['FILE_ID'][i],' data[DX_GROUP][i]',data['DX_GROUP'][i],'central:',data['SITE_ID'][i])
                    if int(data['DX_GROUP'][i]) == 2:
                        labels[filename] = 0
                        label.append(0)
                    else:
                        labels[filename] = 1
                        label.append(1)
                    break
            count += 1
    label = np.array(label)
    np.savetxt(class_path+'/'+str(count)+'_label_' + name+'_'+ c_kind + '.txt', label, fmt='%s')
    return all_data, label

def cal_pcc_np(data, data_cor):
    '''
    :param data:  graph   871 * 200 * ?
    :return:  adj
    '''
    corr_matrix = []
    for key in range(len(data)):  # every sample
        corr_mat = np.corrcoef(data[key])
        corr_mat = np.arctanh(corr_mat - np.eye(corr_mat.shape[0]))
        corr_matrix.append(corr_mat)
    data_array = np.array(corr_matrix)  # 871 200 200
    where_are_nan = np.isnan(data_array)  # find out nan data
    where_are_inf = np.isinf(data_array)  # find out inf data
    for bb in range(0, len(data)):
        for i in range(0, dim):
            for j in range(0, dim):
                if where_are_nan[bb][i][j]:
                    data_array[bb][i][j] = 0
                if where_are_inf[bb][i][j]:
                    data_array[bb][i][j] = 1
                if data_array[bb][i][j] >= opt.adj_thresh:
                    data_array[bb][i][j] = 1
                elif data_array[bb][i][j] < opt.adj_thresh*(-1):
                    data_array[bb][i][j] = -1
                else:
                    data_array[bb][i][j] = 0


    corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    data_array = [corr_p, corr_n]
    data_array = np.array(data_array)  # 2 871 200 200
    data_array = np.transpose(data_array, (1, 0, 2, 3))
    subject_file = data_cor + '/shift_ASD_opt_BrainNet' + '_' + str(len(data_array)) + '_' + name + '_' + c_kind + '_.mat'
    sio.savemat(subject_file, {'shift_ASD_opt_BrainNet'+'_'+c_kind: data_array})

def get_options():
    opt = Options().initialize()
    return opt
if __name__ == '__main__':
    data_len = 871
    dim = 200
    name = 'cc200'
    opt = get_options()
    data_path = 'data/ABIDE-871/' + name + '/ABIDE_pcp/cpac/filt_global/'
    class_path = opt.LABLE_DIR
    label_path = opt.CSV_FILE
    data_path_cor =  opt.DATA_DIR
    central_kind = ['Caltech','CMU','KKI','Leuven','MaxMun','NYU','OHSU','Olin','Pitt','SBL','SDSU',
                    'Stanford','Trinity','UCLA','UM','USM','Yale']

    for c_kind in central_kind:
        print('c_kind: ', c_kind)
        raw_data, labels = load_data(data_path, label_path, name, class_path, c_kind)
        cal_pcc_np(raw_data, data_path_cor)