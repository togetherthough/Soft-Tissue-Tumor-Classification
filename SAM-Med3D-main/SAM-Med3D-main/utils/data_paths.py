''' Example-1: manually list all dataset paths '''
# img_datas = [
# 'sam3d_train/medical_data_all/COVID_lesion/COVID1920_ct',
# 'sam3d_train/medical_data_all/COVID_lesion/Chest_CT_Scans_with_COVID-19_ct',
# 'sam3d_train/medical_data_all/adrenal/WORD_ct',
# ]
''' Example-2: use glob to automatically list all dataset paths '''
import os.path as osp
from glob import glob

PROJ_DIR = osp.dirname(osp.dirname(__file__))
# Point to the prepared GIST dataset: expects imagesTr/ and labelsTr under this dir
img_datas = [osp.join(PROJ_DIR, "data", "train", "gist", "ct_GIST")]
