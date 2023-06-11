class CFG:
    RANDOM_STATE = 42
    BERT_BATCH_SIZE = 32
    BERT_EPOCHS = 5
    BATCH_SIZE = 32
    EPOCHS = 50
    
    MAX_LENGTH = 512
    N_CLASSES = 15
    PATIENCE = 10
    
    PRETRAINED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    DATA_PATH = './data'

    MODEL_CSV_FILE = 'labeled.csv'
    UPDATE_CSV_FILE = 'labeled.csv'
    
    SBERT_MODEL_PATH = './sbert_model'
    SBERT_MODEL_FOLDER = '23-02-10_1059'

    CLASSIFIER_MODEL_PATH = './model'
    CLASSIFIER_MODEL_DATE = '23-02-10_1253'
    CLASSIFIER_MODEL_FILE = 'BASELINE'
    CLASSIFIER_MODEL_NEW_FILE = 'UPDATE'

    EVALUATE_CSV_FILE = 'labeld_sample_07.csv'
    PREDICT_CSV_FILE = 'total_reviews_clean.csv'
    PREDICTED_CSV = 'review_total_predicted.csv'

    TAG_CSV = 'stores_final_tag.csv'
    
    CLASS_NAMES = [
        '가성비','귀여운','넓은','단체','만족',
        '모던','분위기','비주얼','아늑','위생',
        '응대','이색음식','이색테마','클래식','혼자'
    ]
