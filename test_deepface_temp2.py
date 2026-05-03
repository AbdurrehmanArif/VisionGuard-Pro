from deepface import DeepFace
import os

db_path = 'dataset_faces'
test_img = os.path.join(db_path, 'mohammadukkasha_268466', os.listdir(os.path.join(db_path, 'mohammadukkasha_268466'))[0])

try:
    dfs = DeepFace.find(img_path=test_img, db_path=db_path, enforce_detection=False, silent=True)
    if len(dfs) > 0 and len(dfs[0]) > 0:
        print(dfs[0][['identity', 'distance']])
except Exception as e:
    print('Error:', e)
