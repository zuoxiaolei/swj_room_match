from keras.layers import Input, Dense
from keras.models import Model, load_model
import numpy as np
from glob import glob
import pickle as pickle
from keras.preprocessing.image import img_to_array, load_img
from lshash import LSHash
from keras import backend as K
from gevent.pool import Pool


def generate_arrays_from_file():
    '''
    读取图片
    :return:
    '''
    files = glob("./data/image/*.png")
    files = files
    x = []
    pool = Pool(size=100)
    res = pool.imap(lambda filename: img_to_array(load_img(filename)) / 255, files)
    for img_area in res:
        x.append(img_area)
    x = np.array(x)
    return x


def get_train_img_data(save_path="data/train.npy"):
    '''
    图片数据保存成np.array
    :param save_path:
    :return:
    '''
    train_data = generate_arrays_from_file()
    np.save(save_path, train_data)


def build_auto_encode_model(shape=[48 * 64 * 3], encoding_dim=100):
    '''
    建立自编码神经网络模型
    :param shape:
    :param encoding_dim:
    :return:
    '''
    input_img = Input(shape=shape)
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    out_shape = np.product(shape)
    decoded = Dense(out_shape, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, autoencoder


def train_auto_encode_model(encoder_model_path="./data/encoder.h5"):
    '''
    图片数据训练自编码神经网络
    :param encoder_model_path:
    :return:
    '''
    X = np.load("data/train.npy")
    X = X.reshape(X.shape[0], -1)
    X_train = X[int(round(X.shape[0] * 0.2)):, :]
    X_test = X[0:int(round(X.shape[0] * 0.2)), :]
    encoder, autoencoder = build_auto_encode_model()
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True, validation_data=(X_test, X_test))
    encoder.save(encoder_model_path)


def index_room():
    '''
    lsh算法索引图片特征
    :return:
    '''
    files = glob("./data/features/*.csv")
    files_ids = [filename.split("\\")[-1].replace(".csv", "") for filename in files]
    X = np.load("data/train.npy")
    X = X.reshape(X.shape[0], -1)
    encoder = load_model("data/encoder.h5")
    dimension = 100
    lsh_hash = LSHash(hash_size=32, input_dim=dimension)
    compress_feature = encoder.predict(X)
    for num, ele in enumerate(compress_feature.tolist()):
        lsh_hash.index(ele, extra_data=files_ids[num])
    with open("data/lsh.pkl", "wb") as fh:
        pickle.dump(lsh_hash, fh)


def get_picture_feature(picture_path):
    '''
    根据自编码神经网络算出图片的100维特征
    :param picture_path:
    :return:
    '''
    img_area = img_to_array(load_img(picture_path)) / 255.0
    img_area = np.array([img_area])
    img_area = img_area.reshape(img_area.shape[0], -1)
    encoder = load_model("data/encoder.h5")
    compress_feature = encoder.predict(img_area)
    return compress_feature.tolist()[0]


def get_similar_room_example(pic_path = r"data\image\00908484.png"):
    with open("data/lsh.pkl", "rb") as fh:
        lsh = pickle.load(fh)
        res = get_picture_feature(pic_path)
        res = lsh.query(res, num_results=10) # 查询前10个最相似的图片
        print("query room id is {}".format(pic_path))
        for ele in res:
            data_id = ele[0][1]
            distance = ele[1]
            print("similar room id is {0} and the distance is distance {1}\n".format(data_id, distance))


if __name__ == '__main__':
    # get_train_img_data() #读取图片数据
    # train_auto_encode_model() #训练模型
    # index_room() # 给降维的图片数据建立索引
    get_similar_room_example() #检索相似房间
    K.clear_session()
