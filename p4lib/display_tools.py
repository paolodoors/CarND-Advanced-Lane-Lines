import cv2
import numpy as np

class DiagWindow():
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

    def __init__(self):
        self.screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

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
            self.screen[y:y+height, x:x+width] = self.__set_image(params, (width, height))
        else:
            font = cv2.FONT_HERSHEY_COMPLEX
            panel = np.zeros((height, width, 3), dtype=np.uint8)
            line = 60
            for text in params:
                cv2.putText(panel, text, (30, line), font, 1, (255, 0, 0), 2)
                line += 30
            self.screen[y:y+height, x:x+width] = panel

    def get_output(self):
        return self.screen
