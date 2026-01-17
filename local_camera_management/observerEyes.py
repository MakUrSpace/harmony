import requests as req
import argparse
from time import sleep
import subprocess


""" Control a strip of ws2812 LEDs """
from rpi_ws281x import PixelStrip, Color


colors = {
    "white": Color(180, 180, 180),
    "blue": Color(0, 0, 255),
    "green":  Color(0, 255, 0),
    "red": Color(255, 0, 0),
    "off": Color(0, 0, 0)
}


def identify_color(color):
    if color in colors:
        return color
    for tcn, test_color in colors.items():
        if test_color == color:
            return tcn
    raise Exception(f"Color ({color}) not recognized")



DCOLOR = 'red'
PIN = 18
LEDS = 36


def set_strip_to_color(color=None):
    color = DCOLOR if color is None else color
    assert type(color) == str
    color = colors[color.lower()]
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
    strip.show()


def validateEyes():
    reset = False
    for eye in [0]: # , 2, 4, 6]:
        for i in range(3):
            resp = req.get(f"http://localhost:808{eye}/snapshot")
            if resp.status_code == 500:
                reset = True
                print("Restarting...")
                subprocess.run(f"sudo systemctl restart cam{eye}-camera-streamer", shell=True)
                sleep(1)
                print("Trying again")
            else:
                print(f"Success! {resp.status_code}")
                break
    if resp.status_code != 200:
        raise Exception(f"Failed to establish connection to eyes")
    if reset:
        subprocess.run(f"sudo systemctl restart observerServer.service", shell=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ObserverEyes',
        description='Validates Observer camera services are running and responsive')
    strip = PixelStrip(num=LEDS, pin=PIN, freq_hz=800000, dma=10, invert=False, brightness=255, channel=0)
    strip.begin()
    color = colors[DCOLOR]
    print("Setting lights to red")
    set_strip_to_color("red")
    while True:
        print("Validating Eyes")
        validateEyes()
        set_strip_to_color("white")
        sleep(30)
