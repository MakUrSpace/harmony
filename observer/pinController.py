import argparse
from time import sleep
import os
import json

from periphery import GPIO


gpio_p16 = None
gpio_p18 = None
gpio_p22 = None


if __name__ == '__main__':
    gpio_p16 = GPIO("/dev/gpiochip2", 9, "out")
    gpio_p18 = GPIO("/dev/gpiochip4", 10, "out")
    gpio_p22 = GPIO("/dev/gpiochip4", 12, "out")

    happy = gpio_p18
    sad = gpio_p16
    working = gpio_p22

    working.write(True)

    try:
        while True:
            with open("interest.json", "r") as f:
                robotState = json.loads(f.read())
            if robotState['happy'] > 50:
                happy.write(True)
                sad.write(False)
            elif robotState['happy'] > 25:
                happy.write(True)
                sad.write(True)
            else:
                happy.write(False)
                sad.write(True)

            sleep(1)
    finally:
        gpio_p16.close()
        gpio_p18.close()
        gpio_p22.close()

