from GameInterface.key_output import set_pos, left_click, hold_left_click, release_left_click, ReleaseKey, HoldKey, hold_right_click, release_right_click
from GameInterface.key_input import mouse_check, mouse_l_click_check, mouse_r_click_check, key_check
import time


class MovementControl:
    A_KEY_CODE = 0X41
    S_KEY_CODE = 0X53
    W_KEY_CODE = 0X57
    D_KEY_CODE = 0X44
    SPACEBAR_KEY_CODE = 0X20
    
    
    def is_moving():
        return key_check(MovementControl.A_KEY_CODE) or key_check(MovementControl.S_KEY_CODE) or key_check(MovementControl.W_KEY_CODE) or key_check(MovementControl.D_KEY_CODE)

    def stop_moving():
        if key_check(MovementControl.A_KEY_CODE):
            ReleaseKey(MovementControl.A_KEY_CODE)
        if key_check(MovementControl.S_KEY_CODE):
            ReleaseKey(MovementControl.S_KEY_CODE)
        if key_check(MovementControl.W_KEY_CODE):
            ReleaseKey(MovementControl.W_KEY_CODE)
        if key_check(MovementControl.D_KEY_CODE):
            ReleaseKey(MovementControl.D_KEY_CODE)
    

class MouseControl:
    
    #adapted code from Yolov5ForCSGO github repo
    def lock(aims, x, y):
        mouse_pos_x, mouse_pos_y = mouse_check()
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
            # mouse.position = (x_center, y_center)
            set_pos(x_center, y_center)

    #recoil occurs when agent is firing, or if agent is moving alot 
    def check_for_recoil():
        return mouse_l_click_check()

    def reset_recoil():
        release_left_click()
    
    def aim_down_sights():
        if not mouse_r_click_check():
            hold_right_click()
    
    def release_aim_down():
        if mouse_r_click_check():
            release_right_click()
        
    def aim_and_fire(aims,x,y):
        #check that there is a target and either
        if len(aims) > 0:
            recoil = False
            #check for recoil and if true reset recoil
            if MouseControl.check_for_recoil():
                MouseControl.reset_recoil()
                recoil = True
            
            if MovementControl.is_moving():
                MovementControl.stop_moving()
                recoil = True
            
            if recoil:  
                time.sleep(0.1)
                
            MouseControl.lock(aims,x,y)
            left_click()
        else:
            if mouse_l_click_check():
                release_left_click()
            
            
            
if __name__ == '__main__':
    mouse_control = MouseControl() 
    
    #try to aim at an image
    from .input_data_utils import CSGOImageProcessor
    import os 
    import cv2
    csgo_image_processor = CSGOImageProcessor(None)
    test_images = os.listdir('test_images')
    for test_image in test_images:
        image = cv2.imread('test_images/' + test_image)
        csgo_image_processor.update_image(image)
        radar_image = csgo_image_processor.get_radar_image()
        center_image = csgo_image_processor.get_center_image()
        aims,x,y = csgo_image_processor.scan_center_image_for_enemy(center_image)
        mouse_control.lock(aims, x, y) if len(aims) > 0 else None
        csgo_image_processor.visualize_scan_center_image_for_enemy(center_image)
    
    
    #