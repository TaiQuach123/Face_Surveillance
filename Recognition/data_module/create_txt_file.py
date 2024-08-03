import os

def create_casia_text_file(save_path, casia_folder):
    with open(save_path, 'w') as f:
        for i, path in enumerate(os.listdir(casia_folder)):
            label = i
            for img_path in os.listdir(os.path.join(casia_folder,path)):
                img_path = os.path.join(path, img_path)
                f.write(img_path + ' ' + str(label) + '\n')

if __name__ == "__main__":
    save_path = "casia_text_file.txt"
    casia_folder = "data/CASIA-maxpy-clean"
    create_casia_text_file(save_path=save_path, casia_folder=casia_folder)