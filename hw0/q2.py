from PIL import Image
import argparse
def main():
    parser = argparse.ArgumentParser(description='q1.py')
    parser.add_argument('IMAGE_FILE_PATH', type=str,
                    help='IMAGE_FILE_PATH')
    args = parser.parse_args()
    img = Image.open(args.IMAGE_FILE_PATH)
    divide_two(img, args.IMAGE_FILE_PATH)

def divide_two(img, IMAGE_FILE_PATH):
    rgb_im = img.convert('RGB')
    im_new = Image.new("RGB", rgb_im.size)
    pixel_map = im_new.load()
    width, height = rgb_im.size
    for i in range(width):
        for j in range(height):
            r, g, b = rgb_im.getpixel((i, j))
            pixel_map[i, j] = (r // 2, g // 2, b // 2)
    im_new.save("./Q2.png", "PNG")

if __name__ == '__main__':
    main()