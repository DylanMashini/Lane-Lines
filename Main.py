import Lane_Lines as laneline
import tensorflow as tf



def get_lines():
    lines = laneline.video("/Users/dylanmashini/PycharmProjects/Self Driving/video.mp4", True, True)
    return  lines

get_lines()




