import pickle

with open('emo_to_diseases.pkl', 'wb') as f:
    w1 = [[0] * 16] * 7
    w2 = [[1] * 11] * 16
    b1 = [0] * 16
    b2 = [1] * 11
    
    pickle.dump((w1, w2, b1, b2, '0.01'), f)