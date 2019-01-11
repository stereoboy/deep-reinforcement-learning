import os
import sys
import time
import subprocess

BEGIN_CNT=0
TEST_SIZE=5

def main():
    try:
        print(range(TEST_SIZE))
        for cnt in range(BEGIN_CNT, BEGIN_CNT + TEST_SIZE):
            print(cnt)
            args = 'python train.py 1>/dev/null'
            subp2 = subprocess.Popen([args], shell=True)
            subp2.communicate()

            time.sleep(3)
    except Exception as exc:
        print("[ERROR] {}".format(exc))
        sys.exit()

if __name__== '__main__':
    main()
