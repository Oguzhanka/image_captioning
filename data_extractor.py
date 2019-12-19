import os
import h5py
import pickle
import urllib.request


def get_data(file_name):

    recreate = not os.path.exists("./dataset/data.pkl")

    if recreate:
        data = {}
        with h5py.File(file_name, "r") as f:
            for key in f.keys():
                if key in ["train_imid", "train_url"]:
                    data[key] = list(f[key])

                else:
                    continue

        pickle.dump(data, open("./dataset/data.pkl", "wb"))

    data = pickle.load(open("./dataset/data.pkl", "rb"))
    corrupted_ids = []

    corrupted_image_ids = []

    print("Image extraction started!")
    print("Download: [", end="", flush=True)
    for i in range(len(data["train_url"])):
        image_id = data["train_imid"][i]
        image_url = str(data["train_url"][i])[2:-2]
        try:
            urllib.request.urlretrieve(image_url, "./dataset/images/" + str(image_id) + ".jpg")

        except:
            corrupted_ids.append(i)
            corrupted_image_ids.append(i)
            if i % (len(data["train_url"]) // 10) == 0:
                print("=", end="", flush=True)
                print("]")
                print("Done with " + str(len(corrupted_ids)) + " corrupted images.")

    print("Corrupted_ids:")
    print(len(corrupted_ids))


if __name__ == '__main__':
    get_data('dataset/eee443_project_dataset_train.h5')
