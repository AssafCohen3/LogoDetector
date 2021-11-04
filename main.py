import json
import sqlite3

from cv2 import cv2
import matplotlib.pyplot as plt
import requests
import os
import urllib.request


NBA_COURT_LENGTH_METERS = 28.7
PATH_TO_IMAGES = "images/"
PATH_TO_SAVED_IMAGES = "saved/"
PATH_TO_ERRORS = "errors/"
PATH_TO_JSON = 'jsons/'
DB_PATH = "courts.sqlite"
PRINT_DEBUG = 0


def print_debug(img, title="", debug_mode=1, cmapp=None):
    if debug_mode == 0 or PRINT_DEBUG == 1:
        plt.imshow(img, cmap=cmapp)
        plt.title(title)
        plt.show()


def get_images():
    json_files = [pos_json for pos_json in os.listdir(PATH_TO_JSON) if pos_json.endswith('.json')]
    count = 1
    for filename in json_files:
        with open(PATH_TO_JSON + filename, "r") as f:
            images_json = json.load(f)
        images_json = images_json["photoset"]["photo"]
        for image_json in images_json:
            if "url_l" not in image_json:
                continue
            image_url = image_json["url_l"]
            image_name = image_json["title"] + "_" + image_json["id"]
            image_name = image_name.replace(" ", "_").replace("/", "$") + ".jpg"
            if os.path.isfile(PATH_TO_IMAGES + image_name):
                print(f"{count}. image {image_name} already downloaded")
            else:
                urllib.request.urlretrieve(image_url, PATH_TO_IMAGES + image_name)
                print(f"{count}. image {image_name} downloaded")
            count = count + 1


def get_court_contour(contours):
    return min((cnt for cnt in contours if cv2.boundingRect(cnt)[2] >= 700), key=lambda cnt: cv2.boundingRect(cnt)[2], default=None)


def get_logo_contour(contours):
    return max((cnt for cnt in contours if cv2.boundingRect(cnt)[2] <= 500), key=lambda cnt: cv2.boundingRect(cnt)[2], default=None)


def print_logo(filename, save=False, show=True):
    image = cv2.imread(PATH_TO_IMAGES + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = image.copy()
    image[image > 254] = 0
    print_debug(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print_debug(gray)
    size = gray.shape
    center = (size[1] / 2, size[0] / 2)
    # create a binary thresholded image
    gray = cv2.dilate(gray, None)
    gray = cv2.erode(gray, None)
    print_debug(gray)
    gray[gray > 100] -= 100
    print_debug(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print_debug(binary, cmapp="gray")
    thresh_gray = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    print_debug(thresh_gray, cmapp="gray")

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, rect_kernel)
    print_debug(thresh_gray, cmapp="gray")

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    print_debug(thresh_gray, cmapp="gray")


    # show it
    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw all contours
    contours_filtered = list(filter(lambda cnt: cv2.pointPolygonTest(cnt, center, True) >= -20,contours))
    colour = (0, 255, 0)

    court_cnt = get_court_contour(contours_filtered)
    logo_cnt = get_logo_contour(contours_filtered)
    logo_size_meters = None

    if court_cnt is None or logo_cnt is None:
        print(f"Error!!! {filename}. {len(contours_filtered)=}, {len(contours)=}")
        for cnt in contours_filtered:
            print(f"{cv2.contourArea(cnt)}, {cv2.boundingRect(cnt)[2]}")
        print("#############################################")
        colour = (255, 0, 0)
    else:
        w_court = cv2.boundingRect(court_cnt)[2]
        print(f"court length: {w_court}")
        length_ratio = NBA_COURT_LENGTH_METERS / w_court
        w_logo = cv2.boundingRect(logo_cnt)[2]
        print(f"logo length: {w_logo}")
        logo_size_meters = w_logo * length_ratio
        print(f"logo length meters: {logo_size_meters}")
        print("************")
        contours = [court_cnt, logo_cnt]
    image_copy = cv2.drawContours(image_copy, contours, -1, colour, 10)

    if logo_cnt is not None and court_cnt is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        logo_x, logo_y, logo_w, logo_h = cv2.boundingRect(logo_cnt)
        image_copy = cv2.rectangle(image_copy, (logo_x, logo_y), (logo_x+logo_w,logo_y+logo_h), (255,255,0),2)
        image_copy = cv2.putText(image_copy, str(logo_size_meters)[:4], (logo_x, logo_y-5), font, .5, (0, 0, 0), 2, cv2.CV_16U)

    # image = cv2.drawContours(image, contours_filtered, -1, (0, 255, 255), 10)
    # show the image with the drawn contours
    if show:
        a = 1
        cv2.imshow(filename, image_copy)
        #print_debug(image_copy, title=filename, debug_mode=0)
    if save:
        plt.imshow(image_copy)
        plt.savefig(PATH_TO_SAVED_IMAGES + filename)
    return logo_size_meters, image_copy


def add_court_to_db(conn, filename, court_length):
    sql = ''' INSERT INTO Courts(filename, LOGO_LENGTH)
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (filename, court_length))
    conn.commit()

def is_court_exist(conn, filename):
    cur = conn.cursor()
    if cur.execute("SELECT 1 FROM Courts WHERE filename=?", (filename, )).fetchone():
        return True
    return False


def print_logos(from_index, count, conn, show=True, save=False):
    image_files = [pos_img for pos_img in os.listdir(PATH_TO_IMAGES) if pos_img.endswith('.jpg')]
    image_files = sorted(image_files)
    print(len(image_files))
    i = 1
    for img_file in image_files[from_index:from_index + count]:
        print(f"{i}:")
        if is_court_exist(conn, img_file):
            print("Court already saved.")
            continue
        logo_length, image = print_logo(img_file, save=save, show=show)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if logo_length is None:
            print("ERRORRRRRRRRRRR#################")
            cv2.imwrite(PATH_TO_ERRORS + img_file, image)
        else:
            cv2.imshow(img_file, image)
            k = cv2.waitKey(0) % 256
            cv2.destroyAllWindows()
            if k == ord('f'):
                print(f"image {img_file} not accepted. saving image to errors.")
                cv2.imwrite(PATH_TO_ERRORS + img_file, image)
            elif k == ord('q'):
                print("Exiting")
                break
            else:
                print(f"image {img_file} accepted. saving")
                #add_court_to_db(conn, img_file, logo_length)
        i = i + 1

conn = sqlite3.connect(DB_PATH)
print_logos(0, 200, conn)

names = ["Capital_Centre_(1988$1992)_8379298249.jpg",
         "Cole_Field_House_(1972$1973)_8358737300.jpg",
         "Madison_Square_Garden_IV_(1986$1991)_8333224380.jpg",
         "Philips_Arena_(2013$2014)_8550736585.jpg",
         "Philips_Arena_(2014$2015)_14908152423.jpg",
         "St._Louis_Arena_(1961$1964)_8433843944.jpg",
         "State_Farm_Arena_(2018$2019)_30902304467.jpg"]
#print_logo(names[6])
