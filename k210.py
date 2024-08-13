from Maix import GPIO
from fpioa_manager import fm
from machine import UART
import sensor, image, lcd, time, gc, sys
import KPU as kpu

input_size = (224, 224)
labels = ['梨', '苹果', '西瓜', '香蕉']
apples = ['pear', 'apple', 'watermelon', 'banana']
anchors = [5.78, 5.03, 3.8, 3.88, 2.0, 2.06, 1.25, 1.34, 2.78, 3.0]

def init_io():
    fm.register(GPIO.GPIOHS17, fm.fpioa.UART2_TX)
    fm.register(GPIO.GPIOHS16, fm.fpioa.UART2_RX)
    uart_B = UART(UART.UART2, 9600, 8, 0, 0, timeout=1000, read_buf_len=4096)

    fm.register(22, fm.fpioa.GPIO0)
    pear = GPIO(GPIO.GPIO0,GPIO.OUT) # 梨

    fm.register(23, fm.fpioa.GPIO1)
    apple = GPIO(GPIO.GPIO1, GPIO.OUT) # 苹果

    fm.register(24, fm.fpioa.GPIO2)
    watermelon = GPIO(GPIO.GPIO2, GPIO.OUT) # 西瓜

    fm.register(25, fm.fpioa.GPIO3)
    banana = GPIO(GPIO.GPIO3, GPIO.OUT) # 香蕉

    return uart_B, pear, apple, watermelon, banana

def load_labels():
    try:
        with open('labels.txt','r') as f:
            return eval(f.read())
    except Exception:
        return None

def load_model(model_addr):
    try:
        task = kpu.load(model_addr)
        kpu.init_yolo2(task, 0.6, 0.3, 5, anchors)
        return task
    except Exception:
        return None

def detect_fruit(task, labels, uart, gpio_list, img):
    objects = kpu.run_yolo2(task, img)
    if objects:
        for obj in objects:
            pos = obj.rect()
            img.draw_rectangle(pos)
            fruit_name = apples[obj.classid()]
            for i, gpio in enumerate(gpio_list):
                if i == obj.classid():
                    gpio.value(1)
                    uart.write("#K210={}*".format(i+1))
                else:
                    gpio.value(0)
            img.draw_string(pos[0], pos[1], "{}:{:.2f}".format(fruit_name, obj.value()),
                                    scale=2, color=(0, 0, 255))
    else:
        for gpio in gpio_list:
            gpio.value(0)
        uart.write("#K210=0*")
        img.draw_string(10, 10, "no fruit",scale=2, color=(0, 255, 0))


def main(anchors, labels=labels, model_addr="/sd/fruit.kmodel", sensor_window=input_size, lcd_rotation=0, sensor_hmirror=False, sensor_vflip=True):
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing(sensor_window)
    sensor.set_hmirror(sensor_hmirror)
    sensor.set_vflip(sensor_vflip)
    sensor.run(1)

    lcd.init(type=1)
    lcd.rotation(lcd_rotation)
    lcd.clear(lcd.WHITE)

    gpio_list = init_io()
    uart_B, pear, apple, watermelon, banana = gpio_list

    if not labels:
        labels = load_labels()
    if not labels:
        print("no labels.txt")
        img = image.Image(size=(320, 240))
        img.draw_string(90, 110, "no labels.txt", color=(255, 0, 0), scale=2)
        lcd.display(img)
        return 1

    try:
        img = image.Image("startup.jpg")
        lcd.display(img)
    except Exception:
        img = image.Image(size=(320, 240))
        img.draw_string(90, 110, "loading model...", color=(255, 255, 255), scale=2)
        lcd.display(img)

    task = load_model(model_addr)
    if task is None:
        print("failed to load model")
        img = image.Image(size=(320, 240))
        img.draw_string(90, 110, "failed to load model", color=(255, 0, 0), scale=2)
        lcd.display(img)
        return 1

    try:
        while True:
            img = sensor.snapshot()
            detect_fruit(task, labels, uart_B, [pear, apple, watermelon, banana], img)
            lcd.display(img)

    except Exception as e:
        print(e)
        lcd_show_except(e)
    finally:
        if task is not None:
            kpu.deinit(task)
        gc.collect()

if __name__ == "__main__":
    try:
        main(anchors, labels=labels, model_addr=0x300000, lcd_rotation=0)
    except Exception as e:
        sys.print_exception(e)
    finally:
        gc.collect()
