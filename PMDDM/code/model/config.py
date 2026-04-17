import os


#############DIR#############
MY_CURRENT_PATH = 'path to project dir '
MY_DATA_PATH = os.getenv('MY_DATA_PATH')



#############DATASET#############
### 536DATASET(MIDD) ###
MSDATASET_ALIAS = '536dataset'
MSDATASET_DATAINFO = os.path.join(MY_DATA_PATH, '536dataset/audio_featurs/eGeMAPSv02_88dim/merged_info.csv')
MSDATASET_CSV = os.path.join(MY_DATA_PATH,  '536dataset/520_phq9_del.csv')

# 加载音频相关特征
MSDATASET_OPENSMILE_SR_2000 = os.path.join(MY_DATA_PATH, '536dataset/audio_featurs/eGeMAPSv02_88dim_samplerate_2000/merged_fea.pt')
MSDATASET_OPENSMILE_LOUDNESS_NORM = os.path.join(MY_DATA_PATH, '536dataset/audio_featurs/eGeMAPSv02_88dim_loundness_norm/merged_fea.pt')
MSDATASET_OPENSMILE = os.path.join(MY_DATA_PATH, '536dataset/audio_featurs/eGeMAPSv02_88dim/merged_fea.pt')
MSDATASET_XLSR = os.path.join(MY_DATA_PATH, '536dataset/audio_featurs/xlsr_features/xlsr_mean_1024dim/merged_fea.pt')
MSDATASET_XLSR_LOUDNESS_NORM = os.path.join(MY_DATA_PATH, '536dataset/audio_featurs/xlsr_features/xlsr_mean_1024dim_loudness_norm/merged_fea.pt')
MSDATASET_ComParE = os.path.join(MY_DATA_PATH, '536dataset/audio_featurs/ComParE_2016/merged_fea.pt')

# 加载文本相关特征
MSDATASET_DATAINFO_TEXT = os.path.join(MY_DATA_PATH, '536dataset/bertemb/merged_text.csv')
MSDATASET_BERT = os.path.join(MY_DATA_PATH, '536dataset/bertemb/merged_bertemb.pt')

# 加载spk_emb
MSDATASET_SPK_EMB = os.path.join(MY_DATA_PATH, '536dataset/speaker_emb/speaker_emb_1024dim/merged_fea.pt')


### MODMA ###
MODMA_DATAINFO = os.path.join(MY_DATA_PATH, 'MODMA/audio_features/eGeMAPSv02_88dim/modma_18ques.csv')
MODMA_CSV =  os.path.join(MY_DATA_PATH, 'MODMA/MODMA_information.csv')

# 加载音频相关特征
MODMA_OPENSMILE = os.path.join(MY_DATA_PATH, 'MODMA/audio_features/eGeMAPSv02_88dim/modma_fea.pt')
MODMA_ComParE = os.path.join(MY_DATA_PATH, 'MODMA/audio_features/ComParE_2016/modma_fea.pt')
MODMA_XLSR = os.path.join(MY_DATA_PATH, 'MODMA/audio_features/xlsr_features/xlsr_mean_1024dim/modma_fea.pt')
MODMA_MFCC = os.path.join(MY_DATA_PATH, 'MODMA/audio_features/mfcc/modma_fea.pt')

# 加载文本相关特征
MODMA_DATAINFO_TEXT = os.path.join(MY_DATA_PATH, 'MODMA/bert_embedding/modma_text.csv')
MODMA_BERT = os.path.join(MY_DATA_PATH, 'MODMA/bert_embedding/modma_bertemb.pt')

# 加载spk_emb
MODMA_SPK_EMB = os.path.join(MY_DATA_PATH, 'MODMA/speaker_emb/speaker_emb_1024dim/modma_spkemb.pt')


### CMDC ###
CMDC_DATAINFO = os.path.join(MY_DATA_PATH, 'CMDC/audio_features/eGeMAPSv02_88dim/cmdc_audio_path.csv')

# 加载音频相关特征
CMDC_OPENSMILE = os.path.join(MY_DATA_PATH, 'CMDC/audio_features/eGeMAPSv02_88dim/cmdc_fea.pt')

# 加载文本相关特征
CMDC_DATAINFO_TEXT = os.path.join(MY_DATA_PATH, 'CMDC/text_emb/bert_emb/cmdc_text_path.csv')
CMDC_BERT = os.path.join(MY_DATA_PATH, 'CMDC/text_emb/bert_emb/cmdc_bert.pt')

# 加载spk_emb
CMDC_SPK_EMB = os.path.join(MY_DATA_PATH, 'CMDC/speaker_emb/speaker_emb_1024dim/cmdc_spkemb.pt')



#############PARA#############
# SPEAKER_EMB_FUSION_METHOD = 'Gate'   # Gate，Atten, No
CROSSDATA_CSV = os.path.join(MY_DATA_PATH,  'crossdataset/modam_multistimu.csv')
if __name__ == '__main__':
    print(MSDATASET_OPENSMILE)


