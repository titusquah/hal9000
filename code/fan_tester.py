
import pyfirmata
import time


board = pyfirmata.Arduino(
    "com3")

it = pyfirmata.util.Iterator(board)
it.start()


pntxt2 = "d:{}:o".format(3)
dpin1 = board.get_pin(pntxt2)

dpin1.mode = 3

dpin1.write(0)
dpin1.write(0.17)
time.sleep(3)
dpin1.write(0)
board.exit()
