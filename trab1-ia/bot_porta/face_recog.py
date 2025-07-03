import numpy as np
import os
from deepface import DeepFace

#Processa todas as imagens e salva os dados do reconhecimento facial
def proc_img(dir_path):
    embedding_objs = []
    all_img = []

    for file_name in os.listdir(dir_path):
        if file_name == "desconhecidos":
            continue
        this_img = os.path.join(dir_path, file_name)

        try:
            embedding = DeepFace.represent(img_path=this_img, model_name='Facenet512', enforce_detection=True)
            embedding_objs.append(embedding[0]['embedding'])

            if dir_path == "teste/desconhecidos/":
                file_name = file_name.split(".")[0]
                all_img.append(file_name)
            else:
                file_name = file_name.split("_")[0]
                all_img.append(file_name)
        except Exception:
            print(f"Rosto não foi reconhecido em {this_img}")

    return np.array(embedding_objs), np.array(all_img)

#-------------------------------------------------------------------------------------------------------------------------------------------

emb_train, img_train = proc_img("treino/")
emb_valid, img_valid = proc_img("validacao/")
emb_test, img_test = proc_img("teste/")
emb_test_desc, img_test_desc = proc_img("teste/desconhecidos/")

np.savetxt("treino_emb.txt", emb_train, "%.8f")
np.savetxt("treino_img.txt", img_train, "%s")
np.savetxt("validacao_emb.txt", emb_valid, "%.8f")
np.savetxt("validacao_img.txt", img_valid, "%s")
np.savetxt("teste_emb.txt", emb_test, "%.8f")
np.savetxt("teste_img.txt", img_test, "%s")
np.savetxt("desc_emb.txt", emb_test_desc, "%.8f")
np.savetxt("desc_img.txt", img_test_desc, "%s")
