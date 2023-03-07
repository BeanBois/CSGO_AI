# _*_ coding : utf-8 _*_
# @Time : 2022/6/30 0:03
# @Author : Lucid1ty
# @File : mouse_control
# @Project : Yolov5ForCSGO

import pynput
import random
from time import sleep
def lock(aims, mouse, x, y):
    # Get the coordinates of the current mouse
    mouse_pos_x, mouse_pos_y = mouse.position
    dist_list = []
    # Iterate through each target
    for det in aims:
        # Takes only the coordinates of the target
        _, x_c, y_c, _, _ = det
        dist = (x * float(x_c) - mouse_pos_x) ** 2 + (y * float(y_c) - mouse_pos_y) ** 2
        dist_list.append(dist)

    det = aims[dist_list.index(min(dist_list))]
    tag, x_center, y_center, width, height = det
    tag = int(tag)
    x_center, width = x * float(x_center), x * float(width)
    y_center, height = y * float(y_center), x * float(height)
    # Type selection
    if tag == 0:    # 0 head, 1 body
        mouse.position = (x_center, y_center)


if __name__ == '__main__':
    
    def on_move(x, y):
        pass


    def on_click(x, y, button, pressed):
        lock_mode = False 
        if pressed and button == button.x2:  # mouse button 5
            lock_mode = not lock_mode
            print('lock mode', 'no' if lock_mode else 'off')


    def on_scroll(x, y, dx, dy):
        pass    
    
    mouse = pynput.mouse.Controller()
    listener = pynput.mouse.Listener(
    on_move=on_move,
    on_click=on_click,
    on_scroll=on_scroll
    )
    listener.start()
    x, y = 1920, 1080
    for i in range(10):
        ix,iy = random.randint(0, x), random.randint(0, y)
        mouse.position = (ix, iy)
        print(f'done moving to ({ix}, {iy})')
        sleep(1)
