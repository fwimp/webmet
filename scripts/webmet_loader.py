#!/usr/bin/env python3

# A simple loader for web photos.
# Allows you to generate a quick csv containing hub locations, pixel-cm scalings, and links to the relevant image file

from matplotlib import pyplot as plt
import numpy as np
import skimage as img
from skimage import io
import csv
import sys
import logging

# Set up logging
logger = logging.getLogger("webmet_loader")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
chandler = logging.StreamHandler()
chandler.setLevel(logging.DEBUG)
chandler.setFormatter(formatter)
logger.addHandler(chandler)
logger.setLevel(logging.INFO)

# Theme setup
hubcolour = "red"
calibcolour = "blue"


def tellme(s):
    # logger.info(s)
    plt.title(s, fontsize=12)
    plt.draw()


class WebImage:
    def __init__(self, path):
        self.path = path
        self.image = io.imread(path)
        self.hub = None
        self.calib_scale = None
        self.calib_start = None
        self.calib_end = None
        self.single_calib_point = True

        # Safely init fig stuff just in case __gethub__ etc. is called natively for some reason
        self.fig = None
        self.ax = None
        self.hubpoint = None
        self.calib_startpoint = None
        self.calib_endpoint = None
        self.calib_line = None
        self.cid_click = 0
        self.cit_keys = 0

    def analyse(self):
        self.fig = plt.figure(figsize=(16, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.image)
        tellme("Select the hub with a click")
        # Preinit calibration line
        self.calib_line = self.ax.plot([0,0], [0,0], c=calibcolour, marker="o", visible=False)[0]
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.__gethub__)
        self.cit_keys = self.fig.canvas.mpl_connect("key_press_event", self.__onpress__)
        plt.show()

        # Calc calib scale
        if self.calib_start is not None and self.calib_end is not None:
            self.calib_scale = np.sqrt(abs(self.calib_start[0] - self.calib_end[0])**2 + abs(self.calib_start[1] - self.calib_end[1])**2)
        return self

    def __gethub__(self, click):
        # Find hub
        self.hub = (int(click.xdata), int(click.ydata))
        logger.debug("hub @ {}".format(self.hub))

        # Plot the hub and save the reference, note that as this returns a list, we need to get the only entry.
        self.hubpoint = self.ax.plot(*self.hub, c=hubcolour, marker="o")[0]
        plt.draw()

        # Kill click connection for hub select
        self.fig.canvas.mpl_disconnect(self.cid_click)
        # Set click connection for hub reset
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.__resethub__)

        tellme("Now select the calibration size (hover over the start and end and press b/e respectively.)\n"
               "Hub = {}, b = {}, e = {}".format(self.hub, self.calib_start, self.calib_end))
        return self.hub

    def __resethub__(self, click):
        if click.button == 3:
            logger.debug("Resetting hub".format(self.hub))
            self.hub = None

            # Remove hubpoint from plot and class and redraw
            self.ax.lines.remove(self.hubpoint)
            self.hubpoint = None
            plt.draw()

            # Kill click connection for hub reset
            self.fig.canvas.mpl_disconnect(self.cid_click)
            # Set click connection for hub select
            self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.__gethub__)

            tellme("Select the hub with a click")
        return None

    def __recalc_calib_line__(self):
        # Recalculate calibration line on call (this seems slow but also seems to work
        # Redraws canvas
        if self.calib_start is None or self.calib_end is None:
            return
        self.calib_line.set_xdata([self.calib_start[0], self.calib_end[0]])
        self.calib_line.set_ydata([self.calib_start[1], self.calib_end[1]])
        self.calib_line.set_visible(True)

    def __update_calib_point__(self, startend):
        # Update calibration point
        # Redraws canvas
        if startend == "start":
            if self.calib_startpoint is None:
                self.calib_startpoint = self.ax.plot(*self.calib_start, c=calibcolour, marker="o")[0]
            else:
                self.calib_startpoint.set_xdata(self.calib_start[0])
                self.calib_startpoint.set_ydata(self.calib_start[1])
        elif startend == "end":
            if self.calib_endpoint is None:
                self.calib_endpoint = self.ax.plot(*self.calib_end, c=calibcolour, marker="o")[0]
            else:
                self.calib_endpoint.set_xdata(self.calib_end[0])
                self.calib_endpoint.set_ydata(self.calib_end[1])
        self.__recalc_calib_line__()
        plt.draw()

    def __onpress__(self, press):
        if press.key in ["b", "e", "enter", "escape"]:
            if press.key == "b":
                # Get location
                self.calib_start = (int(press.xdata), int(press.ydata))
                logger.debug("start calib @ {}".format(self.calib_start))
                self.__update_calib_point__("start")
                # Refresh prompt
                if self.hub is not None:
                    tellme(
                        "Now select the calibration size (hover over the start and end and press b/e respectively.)\nHub = {}, b = {}, e = {}".format(
                            self.hub, self.calib_start, self.calib_end))

            elif press.key == "e":
                self.calib_end = (int(press.xdata), int(press.ydata))
                logger.debug("end calib @ {}".format(self.calib_end))
                self.__update_calib_point__("end")
                if self.hub is not None:
                    tellme(
                        "Now select the calibration size (hover over the start and end and press b/e respectively.)\nHub = {}, b = {}, e = {}".format(
                            self.hub, self.calib_start, self.calib_end))

            else:
                plt.close()
            return None


def write_to_csv(webs, file):
    if isinstance(webs, WebImage):
        webs = [webs]
    with open(file, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["path", "hubx", "huby", "scalefactor"])
        for w in webs:
            csvwriter.writerow([w.path, *w.hub, w.calib_scale])


def main(argv):
    # Expect argv[1] to be one file path for now
    print(argv)
    argv.append("../data/test/new_method_2_resize.JPG")
    webs = argv[1:]
    csvfile = "../results/loader_test.csv"
    logger.info(f"Processing {len(webs)} webs")
    web_outputs = [WebImage(x) for x in webs]
    web_outputs = [x.analyse() for x in web_outputs]
    logger.info(f"Writing web data to {csvfile}")
    write_to_csv(web_outputs, csvfile)
    logger.info("Processing complete")
    return 0


if __name__ == "__main__":
    status = main(sys.argv)
    sys.exit(status)
