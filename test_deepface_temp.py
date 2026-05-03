from deepface import DeepFace
import os

db_path = 'dataset_faces'
test_img = os.path.join(db_path, 'mohammadukkasha_268466', os.listdir(os.path.join(db_path, 'mohammadukkasha_268466'))[0])

print(f'Testing image: {test_img}')
try:
    dfs = DeepFace.find(img_path=test_img, db_path=db_path, enforce_detection=False, silent=True)
    if len(dfs) > 0 and len(dfs[0]) > 0:
        print(dfs[0].head())
        print('Matched:', dfs[0].iloc[0]['identity'])
    else:
        print('No match found for own image?!')
except Exception as e:
    print('Error:', e)
