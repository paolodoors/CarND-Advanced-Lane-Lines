import cv2
import numpy as np

class DisplayWindow():
    '''
           0                            1280    1600    1920
    0      +-------------------------------+-------+-------+ 0
           |                               |       |       |
           |                               |  TI   |  TII  |
           |                               +-------+-------+ 240
           |            MAIN               |       |       |
           |                               |  TIV  |  TIII |
           |                               +-------+-------+ 480
           |                               |               |
    720    +-------------------------------+               |
           |             TEXT              |               |
    840    +-------+-------+-------+-------+    SECOND     |
           |       |       |       |       |               |
           |  BI   |  BII  | BIII  |  BIV  |               |
    1080   +-------+-------+-------+-------+---------------+ 1080
           0     320     640     960    1280            1920
    '''

    sections = {'main':     [(0, 0),        (1280, 720)],
                'second':   [(480, 1280),   (640, 600)],
                'text':     [(720, 0),      (1280, 120)],
                't1':       [(0, 1280),     (320, 240)],
                't2':       [(0, 1600),     (320, 240)],
                't3':       [(240, 1280),   (320, 240)],
                't4':       [(240, 1600),   (320, 240)],
                'b1':       [(840, 0),      (320, 240)],
                'b2':       [(840, 320),    (320, 240)],
                'b3':       [(840, 640),    (320, 240)],
                'b4':       [(840, 960),    (320, 240)]}

    def __init__(self, debug=False):
        self.screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.debug = debug

    def __set_image(self, image, size):
        if len(image.shape) < 3:
            image = np.stack((image,) * 3, axis=2) * 255
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def set_region(self, region, params):
        if region not in self.sections:
            print ("Error, section {} doesn't exists".format(region))
            return None

        (y, x) = self.sections[region][0]
        (width, height) = self.sections[region][1]

        if region is not 'text':
            if region in ['main', 'text'] or self.debug:
                self.screen[y:y+height, x:x+width] = self.__set_image(params, (width, height))
        else:
            font = cv2.FONT_HERSHEY_COMPLEX
            panel = np.zeros((height, width, 3), dtype=np.uint8)
            line = 60
            for text in params:
                cv2.putText(panel, text, (30, line), font, 1, (255, 0, 0), 2)
                line += 30
            if self.debug:
                self.screen[y:y+height, x:x+width] = panel
            else:
                offset = 50
                patch = self.screen[offset:offset+height, offset:offset+width]
                blended = cv2.addWeighted(patch, 1, panel, 0.8, 0)
                self.screen[offset:offset+height, offset:offset+width] = blended
                

    def get_output(self):
        if self.debug:
            display = self.screen
        else:
            (y, x) = self.sections['main'][0]
            (width, height) = self.sections['main'][1]
            display = self.screen[y:y+height, x:x+width]

        return display
