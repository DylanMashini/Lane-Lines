import Lane_Lines as laneline



def get_lines():
    lines = laneline.video("./video.mp4", True, True)
    return  lines

get_lines()




