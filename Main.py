import Lane_Lines as laneline
import tensorflow as tf
import numpy as np
import argparse



def get_lines():
    lines = laneline.video(path="/Users/dylanmashini/PycharmProjects/Self Driving/video.mp4", save_video=False, output_lines=True)

    return  lines

lines = get_lines()
new_lines = []
lines = np.asarray(lines)


print(type(lines))
print(lines[0].shape)
print(lines[0])
fake_y = []
for i in range(0, 218):
    fake_y.append(0.05)

fake_y = np.asarray(fake_y)
for j in range(0, len(lines)):
    print(j)
    line = lines[j].astype("float64")
    print("Before")
    print(line)
    #Gave me height #1
    line[0][0][0] = line[0][0][0]/960
    line[1][0][0] = line[1][0][0]/960
    new_lines.append(line)
    print("After")
    print(line)
new_lines = np.asarray(new_lines)
new_lines = new_lines.astype("float64")




print(len(fake_y))
def build_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="elu"))
    model.add(tf.keras.layers.Dense(128, activation="elu"))
    model.add(tf.keras.layers.Dense(128, activation="elu"))
    model.add(tf.keras.layers.Dense(128, activation="elu"))
    model.add(tf.keras.layers.Dense(1, activation="elu"))


    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    model.fit(new_lines, fake_y)
    model.summary()




build_model()